import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _make_vq_model_vq16(codebook_size=16384, codebook_embed_dim=8):
    return VQModel(
        ModelArgs(
            codebook_size=codebook_size,
            codebook_embed_dim=codebook_embed_dim,
            codebook_l2_norm=True,
            codebook_show_usage=True,
            commit_loss_beta=0.25,
            entropy_loss_ratio=0.0,
            encoder_ch_mult=[1, 1, 2, 2, 4],
            decoder_ch_mult=[1, 1, 2, 2, 4],
            z_channels=256,
            dropout_p=0.0,
        )
    )


def _make_vq_model_vq8(codebook_size=16384, codebook_embed_dim=8):
    return VQModel(
        ModelArgs(
            codebook_size=codebook_size,
            codebook_embed_dim=codebook_embed_dim,
            codebook_l2_norm=True,
            codebook_show_usage=True,
            commit_loss_beta=0.25,
            entropy_loss_ratio=0.0,
            encoder_ch_mult=[1, 2, 2, 4],
            decoder_ch_mult=[1, 2, 2, 4],
            z_channels=256,
            dropout_p=0.0,
        )
    )


# =============================================================================
# LlamaGen VQ-VAE model (bundled from tokenizer/tokenizer_image/vq_model.py)
# =============================================================================

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "batch":
        return nn.SyncBatchNorm(in_channels)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type="group"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type="group", dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        for mid_block in self.mid:
            h = mid_block(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        h = self.conv_in(z)
        for mid_block in self.mid:
            h = mid_block(h)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        z = torch.einsum("b c h w -> b h w c", z).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, torch.einsum("nd -> d n", embedding))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * _compute_entropy_loss(-d)

        z_q = z + (z_q - z).detach()
        z_q = torch.einsum("b h w c -> b c h w", z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


def _compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError(f"Entropy loss {loss_type} not supported")
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


class ModelArgs:
    def __init__(
        self,
        codebook_size=16384,
        codebook_embed_dim=8,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        commit_loss_beta=0.25,
        entropy_loss_ratio=0.0,
        encoder_ch_mult=(1, 1, 2, 2, 4),
        decoder_ch_mult=(1, 1, 2, 2, 4),
        z_channels=256,
        dropout_p=0.0,
    ):
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.codebook_l2_norm = codebook_l2_norm
        self.codebook_show_usage = codebook_show_usage
        self.commit_loss_beta = commit_loss_beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.encoder_ch_mult = encoder_ch_mult
        self.decoder_ch_mult = decoder_ch_mult
        self.z_channels = z_channels
        self.dropout_p = dropout_p


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            ch=128,
            ch_mult=config.encoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )
        self.decoder = Decoder(
            z_channels=config.z_channels,
            ch=128,
            ch_mult=config.decoder_ch_mult,
            out_channels=3,
        )
        self.quantize = VectorQuantizer(
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


# =============================================================================
# Backend-specific tokenizer adapters
# =============================================================================

class LlamaGenVQTokenizer:
    """Tokenizer adapter for LlamaGen VQ-VAE checkpoints (vq_ds16_c2i.pt, vq_ds8_c2i.pt)."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device,
        downsample_factor: int,
        codebook_size: int,
        codebook_embed_dim: int,
    ):
        self._model = model
        self._model_name = str(model_name)
        self._device = torch.device(device)
        self._downsample_factor = int(downsample_factor)
        self._codebook_size = int(codebook_size)
        self._codebook_embed_dim = int(codebook_embed_dim)

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: Optional[torch.device] = None, **kwargs):
        if not name_or_path:
            raise ValueError("image_maskgit_vq_model_name is required for discrete MaskGIT")

        dev = torch.device("cpu") if device is None else torch.device(device)
        p = str(name_or_path).lower()

        if "ds8" in p:
            model = _make_vq_model_vq8(codebook_size=16384, codebook_embed_dim=8)
            downsample_factor = 8
        else:
            model = _make_vq_model_vq16(codebook_size=16384, codebook_embed_dim=8)
            downsample_factor = 16

        state = torch.load(name_or_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            missing, unexpected = model.load_state_dict(state, strict=False)
            raise RuntimeError(
                f"LlamaGen VQ checkpoint '{name_or_path}' is incompatible with the vendored tokenizer. "
                f"missing={missing} unexpected={unexpected}"
            ) from exc

        model.requires_grad_(False)
        model.eval()
        model.to(dev)

        return cls(
            model=model,
            model_name=name_or_path,
            device=dev,
            downsample_factor=downsample_factor,
            codebook_size=16384,
            codebook_embed_dim=8,
        )

    def to(self, device: torch.device):
        self._device = torch.device(device)
        self._model.to(self._device)
        return self

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def codebook_dim(self) -> int:
        return self._codebook_embed_dim

    @property
    def mask_token_id(self) -> int:
        return int(self._codebook_size)

    @property
    def vocab_size(self) -> int:
        return int(self._codebook_size) + 1

    def infer_grid_shape(self, image_size: int) -> Tuple[int, int]:
        h = int(image_size) // self._downsample_factor
        w = int(image_size) // self._downsample_factor
        return (h, w)

    def encode(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be [B,C,H,W], got {tuple(pixel_values.shape)}")
        if int(pixel_values.size(1)) != 3:
            raise ValueError(f"pixel_values must have 3 channels, got {int(pixel_values.size(1))}")

        x = pixel_values.to(device=self._device, dtype=torch.float32)
        x = x.clamp(0.0, 1.0) * 2.0 - 1.0

        with torch.no_grad():
            _, _, info = self._model.encode(x)

        indices = info[2]
        bsz = int(pixel_values.size(0))
        gh = int(pixel_values.size(2)) // self._downsample_factor
        gw = int(pixel_values.size(3)) // self._downsample_factor
        token_ids = indices.reshape(bsz, gh * gw).to(dtype=torch.long)
        return token_ids.contiguous(), (gh, gw)

    def decode(self, token_ids: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        if token_ids.dim() == 3:
            token_ids = token_ids.reshape(token_ids.size(0), -1)
        elif token_ids.dim() == 2:
            pass
        else:
            raise ValueError(f"token_ids must be [B,T] or [B,H,W], got {tuple(token_ids.shape)}")

        gh, gw = int(grid_shape[0]), int(grid_shape[1])
        bsz, tok = int(token_ids.size(0)), int(token_ids.size(1))
        if gh * gw != tok:
            raise ValueError(f"grid_shape {grid_shape} does not match token count {tok}")

        indices = token_ids.to(device=self._device, dtype=torch.long).reshape(bsz, gh, gw)
        with torch.no_grad():
            dec = self._model.decode_code(indices, shape=[bsz, self._codebook_embed_dim, gh, gw], channel_first=True)

        return ((dec / 2.0) + 0.5).clamp(0.0, 1.0)


class MaskGitVQGANTokenizer:
    """Tokenizer adapter for MaskGIT VQGAN checkpoints (HuggingFace .bin format)."""

    def __init__(self, vq_model, model_name: str, device: torch.device):
        self._model = vq_model
        self._model_name = str(model_name)
        self._device = torch.device(device)

    @classmethod
    def from_pretrained(cls, model_name: str, device: Optional[torch.device] = None, **kwargs):
        try:
            from diffusers import VQModel
        except Exception as exc:
            raise ImportError("diffusers is required for MaskGitVQGANTokenizer") from exc

        if not model_name:
            raise ValueError("image_maskgit_vq_model_name is required for discrete MaskGIT")

        dev = torch.device("cpu") if device is None else torch.device(device)
        vq_model = VQModel.from_pretrained(model_name, **kwargs)
        vq_model.requires_grad_(False)
        vq_model.eval()
        vq_model.to(dev)
        return cls(vq_model=vq_model, model_name=model_name, device=dev)

    def to(self, device: torch.device):
        self._device = torch.device(device)
        self._model.to(self._device)
        return self

    @property
    def codebook_size(self) -> int:
        quantize = getattr(self._model, "quantize", None)
        if quantize is not None and hasattr(quantize, "n_e"):
            return int(quantize.n_e)
        return int(getattr(self._model.config, "num_vq_embeddings", 0))

    @property
    def codebook_dim(self) -> int:
        quantize = getattr(self._model, "quantize", None)
        if quantize is not None and hasattr(quantize, "vq_embed_dim"):
            return int(quantize.vq_embed_dim)
        return int(getattr(self._model.config, "vq_embed_dim", 0))

    @property
    def mask_token_id(self) -> int:
        return int(self.codebook_size)

    @property
    def vocab_size(self) -> int:
        return int(self.codebook_size) + 1

    def infer_grid_shape(self, image_size: int) -> Tuple[int, int]:
        dummy = torch.zeros((1, 3, int(image_size), int(image_size)), device=self._device, dtype=torch.float32)
        token_ids, grid_shape = self.encode(dummy)
        _ = token_ids
        return int(grid_shape[0]), int(grid_shape[1])

    def encode(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be [B,C,H,W], got {tuple(pixel_values.shape)}")
        if int(pixel_values.size(1)) != 3:
            raise ValueError(f"pixel_values must have 3 channels, got {int(pixel_values.size(1))}")

        x = pixel_values.to(device=self._device, dtype=torch.float32)
        x = x.clamp(0.0, 1.0) * 2.0 - 1.0

        with torch.no_grad():
            latents = self._model.encode(x).latents
            _, _, info = self._model.quantize(latents)

        indices = info[2]
        bsz = int(latents.size(0))
        gh = int(latents.size(2))
        gw = int(latents.size(3))
        token_ids = indices.reshape(bsz, gh * gw).to(dtype=torch.long)
        return token_ids.contiguous(), (gh, gw)

    def decode(self, token_ids: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        if token_ids.dim() == 3:
            token_ids = token_ids.reshape(token_ids.size(0), -1)
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B,T] or [B,H,W], got {tuple(token_ids.shape)}")

        gh, gw = int(grid_shape[0]), int(grid_shape[1])
        bsz, tok = int(token_ids.size(0)), int(token_ids.size(1))
        if gh * gw != tok:
            raise ValueError(f"grid_shape {grid_shape} does not match token count {tok}")

        indices = token_ids.to(device=self._device, dtype=torch.long).reshape(bsz, -1)
        with torch.no_grad():
            quant = self._model.quantize.get_codebook_entry(
                indices, shape=(bsz, gh, gw, int(self.codebook_dim))
            )
            quant = self._model.post_quant_conv(quant)
            dec = self._model.decoder(quant)

        return ((dec / 2.0) + 0.5).clamp(0.0, 1.0)


# =============================================================================
# Unified dispatcher (backwards-compatible entry point)
# =============================================================================


class ImageMaskGITVQTokenizer:
    """
    Unified dispatcher for discrete image VQ tokenizers.

    Dispatches to the appropriate backend based on the model name/path:
      - LlamaGen checkpoints (vq_ds16_c2i.pt, vq_ds8_c2i.pt, or paths containing "llamagen")
      - MaskGIT VQGAN (HuggingFace .bin format or paths containing "maskgit")
      - Diffusers VQModel (default fallback)
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        if not model_name:
            raise ValueError("image_maskgit_vq_model_name is required for image_maskgit_variant=discrete")

        p = str(model_name).lower()
        if "llamagen" in p or "vq_ds16" in p or "vq_ds8" in p:
            return LlamaGenVQTokenizer.from_pretrained(model_name, device=device, **kwargs)
        elif "maskgit" in p or p.endswith(".bin"):
            return MaskGitVQGANTokenizer.from_pretrained(model_name, device=device, **kwargs)
        else:
            return MaskGitVQGANTokenizer.from_pretrained(model_name, device=device, **kwargs)
