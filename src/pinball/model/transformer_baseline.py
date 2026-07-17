# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers.normalization import make_norm


logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    use_rope: bool = False
    use_abs_pos_emb: bool = True
    attn_backend: str = "auto"
    gradient_checkpointing: bool = False
    tie_weights: bool = True
    ffn_type: str = "swiglu"
    causal: bool = True
    class_cond_enable: bool = False
    num_classes: int = 0
    class_cond_drop_prob: float = 0.1


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 131072, base: float = 10000.0):
        super().__init__()
        rotary_dim = head_dim - (head_dim % 2)
        self.rotary_dim = int(rotary_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / max(1, rotary_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = int(max_seq_len)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        seq_len = min(int(seq_len), int(self.max_seq_len))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self._cos_cached = emb.cos().to(dtype=dtype)
        self._sin_cached = emb.sin().to(dtype=dtype)
        self._seq_len_cached = int(seq_len)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        if self.rotary_dim <= 0:
            return q, k

        bsz, _, seq_len, _ = q.shape
        needed_len = int(seq_len)
        if position_ids is not None and position_ids.numel() > 0:
            needed_len = max(needed_len, int(position_ids.max().item()) + 1)

        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < needed_len
            or self._cos_cached.device != q.device
            or self._cos_cached.dtype != q.dtype
        ):
            self._build_cache(needed_len, q.device, q.dtype)

        if position_ids is None:
            cos = self._cos_cached[:seq_len].view(1, 1, seq_len, self.rotary_dim)
            sin = self._sin_cached[:seq_len].view(1, 1, seq_len, self.rotary_dim)
        else:
            pos = position_ids.to(device=q.device, dtype=torch.long).clamp_min(0)
            pos = pos.clamp_max(max(0, self._seq_len_cached - 1))
            cos = self._cos_cached[pos].view(bsz, 1, seq_len, self.rotary_dim)
            sin = self._sin_cached[pos].view(bsz, 1, seq_len, self.rotary_dim)

        q_rot = q[..., : self.rotary_dim]
        k_rot = k[..., : self.rotary_dim]
        q_pass = q[..., self.rotary_dim :]
        k_pass = k[..., self.rotary_dim :]
        q = torch.cat([q_rot * cos + _rotate_half(q_rot) * sin, q_pass], dim=-1)
        k = torch.cat([k_rot * cos + _rotate_half(k_rot) * sin, k_pass], dim=-1)
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})")
        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.head_dim = int(config.n_embd // config.n_head)
        self.causal = bool(config.causal)
        self.dropout = float(config.dropout)
        self.attn_backend = str(config.attn_backend).lower()
        if self.attn_backend not in {"auto", "flash", "sdpa", "eager"}:
            self.attn_backend = "auto"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.block_size) if config.use_rope else None
        self.backend_used = "eager"

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, _, seq_len, _ = q.shape
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.causal:
            causal = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril().view(1, 1, seq_len, seq_len)
            att = att.masked_fill(~causal, torch.finfo(att.dtype).min)
        if attention_mask is not None:
            key_mask = attention_mask.to(device=q.device, dtype=torch.bool).view(bsz, 1, 1, seq_len)
            att = att.masked_fill(~key_mask, torch.finfo(att.dtype).min)
        att = F.softmax(att.float(), dim=-1).to(dtype=q.dtype)
        att = self.attn_dropout(att)
        return att @ v

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids=position_ids)

        use_sdpa = self.attn_backend in {"auto", "flash", "sdpa"} and hasattr(F, "scaled_dot_product_attention")
        y = None
        if use_sdpa:
            dropout_p = self.dropout if self.training else 0.0
            try:
                if attention_mask is None or bool(attention_mask.to(dtype=torch.bool).all().item()):
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=self.causal)
                else:
                    key_mask = attention_mask.to(device=x.device, dtype=torch.bool).view(bsz, 1, 1, seq_len)
                    if self.causal:
                        causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril().view(1, 1, seq_len, seq_len)
                        key_mask = key_mask & causal
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=dropout_p, is_causal=False)
                self.backend_used = "sdpa"
            except Exception as exc:
                if self.attn_backend in {"flash", "sdpa"}:
                    logger.warning("Transformer SDPA backend failed; falling back to eager attention: %r", exc)
                y = None

        if y is None:
            y = self._manual_attention(q, k, v, attention_mask)
            self.backend_used = "eager"

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


class GELUMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return self.dropout(x)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Match PackedSwiGLUFFN used by the active Pinball path.
        inner = int((8.0 / 3.0) * int(config.n_embd))
        inner = max(256, ((inner + 255) // 256) * 256)
        self.gate_proj = nn.Linear(config.n_embd, inner, bias=False)
        self.up_proj = nn.Linear(config.n_embd, inner, bias=False)
        self.down_proj = nn.Linear(inner, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = make_norm(config.n_embd, norm_type=config.norm_type, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config.n_embd, norm_type=config.norm_type, eps=config.norm_eps)
        ffn_type = str(config.ffn_type).lower().strip()
        if ffn_type == "gelu":
            self.mlp = GELUMLP(config)
        elif ffn_type == "swiglu":
            self.mlp = SwiGLUMLP(config)
        else:
            raise ValueError(f"Unknown Transformer FFN type: {config.ffn_type!r}")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, position_ids=position_ids)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model_type = "transformer"
        self.vocab_size = int(config.vocab_size)
        self.hidden_dim = int(config.n_embd)
        self.max_seq_len = int(config.block_size)
        self.block_size = int(config.block_size)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None) if tokenizer is not None else None
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.attn_backend = str(config.attn_backend).lower()
        self.use_rope = bool(config.use_rope)
        self.use_abs_pos_emb = bool(config.use_abs_pos_emb)

        # Class conditioning (image MaskGIT): FiLM on the token embeddings, mirroring the
        # pinball model's _apply_film_conditioning (same null-index/CFG-drop semantics).
        self.num_classes = int(config.num_classes)
        self.class_cond_drop_prob = max(0.0, min(1.0, float(config.class_cond_drop_prob)))
        self.class_null_index = int(self.num_classes)
        self.class_embedding: Optional[nn.Embedding] = None
        self.cond_film: Optional[nn.Linear] = None
        if bool(config.class_cond_enable) and self.num_classes > 0:
            self.class_embedding = nn.Embedding(self.num_classes + 1, config.n_embd)
            self.cond_film = nn.Linear(config.n_embd, 2 * config.n_embd)

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.pad_token_id)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd) if config.use_abs_pos_emb else None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config.n_embd, norm_type=config.norm_type, eps=config.norm_eps)
        self.output_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.output_projection.weight = self.token_embedding.weight

        self._last_ce_loss = None
        self._last_copy_dst_ce_loss = None
        self._last_copy_dst_token_count = 0
        self._last_objective_loss = None
        self._last_transformer_attn_backend = "unknown"
        self.apply(self._init_weights)


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        reveal_target_ids: Optional[torch.Tensor] = None,
        reveal_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del reveal_target_ids, reveal_mask, kwargs
        if input_ids.dim() != 2:
            raise ValueError(f"TransformerLM expects token ids [B,T], got {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        if seq_len > self.block_size:
            input_ids = input_ids[:, -self.block_size :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.block_size :]
            if position_ids is not None:
                position_ids = position_ids[:, -self.block_size :]
            seq_len = int(self.block_size)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, seq_len)

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            pos = position_ids.clamp_min(0).clamp_max(self.block_size - 1)
            x = x + self.position_embedding(pos)
        if self.class_embedding is not None:
            if class_labels is None:
                cls = torch.full((bsz,), int(self.class_null_index), dtype=torch.long, device=x.device)
            else:
                cls = class_labels.to(device=x.device, dtype=torch.long).view(-1)
                if int(cls.numel()) != int(bsz):
                    cls = cls.expand(int(bsz)) if int(cls.numel()) == 1 else cls[: int(bsz)]
                cls = cls.clamp(min=0, max=max(0, int(self.num_classes)))
                if self.training and self.class_cond_drop_prob > 0.0:
                    drop_mask = torch.rand(int(bsz), device=x.device) < float(self.class_cond_drop_prob)
                    if bool(drop_mask.any()):
                        cls = cls.clone()
                        cls[drop_mask] = int(self.class_null_index)
            gamma, beta = self.cond_film(self.class_embedding(cls)).chunk(2, dim=-1)
            x = x * (1.0 + torch.sigmoid(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        x = self.drop(x)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, attention_mask, position_ids, use_reentrant=False)
            else:
                x = block(x, attention_mask=attention_mask, position_ids=position_ids)

        x = self.ln_f(x)
        logits = self.output_projection(x)
        if self.blocks:
            self._last_transformer_attn_backend = self.blocks[-1].attn.backend_used
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        self.eval()
        current_ids = input_ids.clone()
        max_length = max(1, int(max_length))

        for _ in range(max_length):
            idx_cond = current_ids[:, -self.block_size :]
            logits = self(idx_cond)
            next_logits = logits[:, -1, :]
            if repetition_penalty and float(repetition_penalty) != 1.0:
                for b in range(current_ids.size(0)):
                    seen = torch.unique(current_ids[b])
                    scores = next_logits[b, seen]
                    next_logits[b, seen] = torch.where(scores < 0, scores * float(repetition_penalty), scores / float(repetition_penalty))
            next_logits = next_logits / max(float(temperature), 1e-6)
            if top_k and int(top_k) > 0:
                k = min(int(top_k), next_logits.size(-1))
                vals, _ = torch.topk(next_logits, k)
                next_logits = next_logits.masked_fill(next_logits < vals[:, [-1]], -float("inf"))
            if top_p and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > float(top_p)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(indices_to_remove, -float("inf"))
            probs = F.softmax(next_logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            current_ids = torch.cat((current_ids, next_token), dim=1)
            if self.eos_token_id is not None and bool((next_token == int(self.eos_token_id)).all().item()):
                break
        return current_ids
