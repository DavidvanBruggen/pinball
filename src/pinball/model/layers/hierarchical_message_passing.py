# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import inspect
import time
from functools import lru_cache
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
# scatter_add is the only torch_scatter op this module uses. Prefer PyG's native
# scatter (pure PyTorch, no compiled extension) so installs don't need torch-scatter
# wheels; fall back to torch_scatter if present.
try:
    from torch_geometric.utils import scatter as _pyg_scatter

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        res = _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
        if out is not None:
            out = out + res
            return out
        return res
except ImportError:  # pragma: no cover - very old PyG
    from torch_scatter import scatter_add
import math
from .positional_encoding import RotaryPositionalEncoding
from .normalization import make_norm
from ...utils.amp import bf16_supported
from typing import Optional, Dict, Any, Tuple, List, Callable

logger = logging.getLogger(__name__)


def _gate_logit(value: float, eps: float = 1.0e-4) -> torch.Tensor:
    value = min(1.0 - eps, max(eps, float(value)))
    return torch.tensor(math.log(value / (1.0 - value)), dtype=torch.float32)

try:
    import xformers.ops as _xops
except Exception:
    _xops = None


def _unwrap_flash_result(result: Any) -> Any:
    """FlashAttention variants may return (out, lse, ...); keep the tensor output."""
    if isinstance(result, tuple):
        return result[0]
    return result


@lru_cache(maxsize=8)
def _flash_attn_signature_keys(flash_attn_func: Callable[..., Any]) -> Tuple[str, ...]:
    try:
        return tuple(inspect.signature(flash_attn_func).parameters.keys())
    except Exception:
        return ()


def _flash_attn_supports_dropout(flash_attn_func: Callable[..., Any]) -> bool:
    keys = _flash_attn_signature_keys(flash_attn_func)
    return "dropout_p" in keys or "dropout" in keys


def _flash_attn_supports_window_size(flash_attn_func: Callable[..., Any]) -> bool:
    return "window_size" in _flash_attn_signature_keys(flash_attn_func)


def _flash_attn_kwargs(
    flash_attn_func: Callable[..., Any],
    causal: bool,
    dropout_p: float,
    window_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    keys = set(_flash_attn_signature_keys(flash_attn_func))
    kwargs: Dict[str, Any] = {"causal": causal}
    if window_size is not None:
        if "window_size" not in keys:
            raise RuntimeError("FlashAttention backend does not support window_size in this build")
        kwargs["window_size"] = window_size
    if "dropout_p" in keys:
        kwargs["dropout_p"] = dropout_p
    elif "dropout" in keys:
        kwargs["dropout"] = dropout_p
    elif dropout_p > 0.0:
        raise RuntimeError("FlashAttention backend does not support dropout in this build")
    return kwargs


def _smoke_test_flash_attn(
    flash_attn_func: Callable[..., Any],
    device_index: int,
) -> bool:
    """Verify FlashAttention import and a real forward/backward kernel launch."""
    if not torch.cuda.is_available():
        return False

    dtype = torch.bfloat16 if bf16_supported() else torch.float16
    device = torch.device("cuda", int(device_index))
    bsz, seqlen, num_heads, head_dim = 1, 64, 4, 64

    try:
        with torch.enable_grad():
            q = torch.randn(bsz, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(bsz, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(bsz, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
            out = flash_attn_func(q, k, v, **_flash_attn_kwargs(flash_attn_func, causal=True, dropout_p=0.0, window_size=(4, 4)))
            out = _unwrap_flash_result(out)
            loss = out.float().square().mean()
            loss.backward()
        torch.cuda.synchronize(device)
        return bool(torch.isfinite(out).all().item())
    except Exception as exc:
        logger.debug("FlashAttention smoke test failed on cuda:%d: %r", int(device_index), exc)
        return False


@lru_cache(maxsize=8)
def _pick_attention_backend_cached(device_index: int, cap_major: int, cap_minor: int) -> Tuple[str, Optional[Callable[..., Any]]]:
    """Pick the attention backend for one CUDA device capability."""
    cap = (int(cap_major), int(cap_minor))
    device_name = torch.cuda.get_device_name(int(device_index))

    if cap >= (8, 0):
        fa3_exc = None
        try:
            from flash_attn_interface import flash_attn_func
        except Exception as exc:
            fa3_exc = exc
        else:
            if _smoke_test_flash_attn(flash_attn_func, int(device_index)):
                logger.info("Using FlashAttention-3 backend on GPU %s capability %s.", device_name, cap)
                return "fa3", flash_attn_func
            logger.warning(
                "FlashAttention-3 smoke test failed on GPU %s capability %s; trying FlashAttention-2.",
                device_name,
                cap,
            )

        fa2_exc = None
        try:
            from flash_attn import flash_attn_func
        except Exception as exc:
            fa2_exc = exc
        else:
            if _smoke_test_flash_attn(flash_attn_func, int(device_index)):
                logger.info("Using FlashAttention-2 backend on GPU %s capability %s.", device_name, cap)
                return "fa2", flash_attn_func
            logger.warning(
                "FlashAttention-2 smoke test failed on GPU %s capability %s.",
                device_name,
                cap,
            )

        logger.warning(
            "FlashAttention unavailable for requested flash backend on GPU %s capability %s (fa3_import=%r, fa2_import=%r).",
            device_name,
            cap,
            fa3_exc,
            fa2_exc,
        )
        return "sdpa", None

    logger.info(
        "GPU %s capability %s does not support FlashAttention; using PyTorch SDPA as emergency fallback.",
        device_name,
        cap,
    )
    return "sdpa", None


def pick_attention_backend(device: Optional[torch.device] = None) -> Tuple[str, Optional[Callable[..., Any]]]:
    """Select the best available attention backend for the requested CUDA device."""
    if not torch.cuda.is_available():
        return "sdpa", None

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    if device.type != "cuda":
        return "sdpa", None

    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    cap_major, cap_minor = torch.cuda.get_device_capability(device_index)
    return _pick_attention_backend_cached(device_index, int(cap_major), int(cap_minor))


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    backend: str = "sdpa",
    flash_func: Optional[Callable[..., Any]] = None,
    window_size: Optional[Tuple[int, int]] = None,
    flash_dtype_cast: bool = False,
) -> torch.Tensor:
    """Unified attention wrapper for [B, S, H, D] tensors."""
    if backend in {"fa2", "fa3"} and flash_func is not None:
        orig_dtype = q.dtype
        if flash_dtype_cast and orig_dtype not in (torch.float16, torch.bfloat16):
            work_dtype = torch.bfloat16 if bf16_supported() else torch.float16
            q_fa = q.to(dtype=work_dtype)
            k_fa = k.to(dtype=work_dtype)
            v_fa = v.to(dtype=work_dtype)
        elif orig_dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"FlashAttention backend '{backend}' requires fp16 or bf16 inputs, got {orig_dtype}. "
                "Enable mixed precision or set local_attn_flash_dtype_cast=True to allow explicit casting."
            )
        else:
            q_fa, k_fa, v_fa = q, k, v
        out = flash_func(q_fa.contiguous(), k_fa.contiguous(), v_fa.contiguous(),
                         **_flash_attn_kwargs(flash_func, causal=causal, dropout_p=dropout_p, window_size=window_size))
        out = _unwrap_flash_result(out)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        return out

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=causal,
    )
    return out.transpose(1, 2).contiguous()



_FLEX_COMPILED = None


def _flex_compiled_singleton():
    """torch.compile(flex_attention) once per process (shared across layers/models)."""
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_COMPILED


class EdgeConditioner(nn.Module):
    """Lightweight edge context -> attention bias and value gate."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        head_dim: int,
        num_edge_types: int = 16,
        edge_type_embedding_dim: int = 32,
        hidden_dim: int = 64,
        logit_bias_per_head: bool = True,
        value_gate_per_head: bool = False,
        value_gate_per_channel: bool = True,
        logit_bias_init_zero: bool = True,
        gate_init_identity: bool = True,
        dropout: float = 0.0,
        edge_gate_scale: float = 0.1,
        node_condition_enable: bool = False,
        node_condition_detach: bool = True,
        node_condition_dim: int = 32,
        node_condition_mode: str = "src_dst_prod",
        node_condition_zero_init: bool = True,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_edge_types = max(1, int(num_edge_types))
        self.logit_bias_per_head = bool(logit_bias_per_head)
        self.value_gate_per_head = bool(value_gate_per_head)
        self.value_gate_per_channel = bool(value_gate_per_channel)
        self.edge_gate_scale = float(edge_gate_scale)
        self.node_condition_enable = bool(node_condition_enable)
        self.node_condition_detach = bool(node_condition_detach)
        self.node_condition_mode = str(node_condition_mode).lower()
        if self.node_condition_mode not in {"src_dst", "src_dst_diff", "src_dst_prod", "qk_like"}:
            self.node_condition_mode = "src_dst_prod"

        self.edge_type_embedding = nn.Embedding(self.num_edge_types, int(edge_type_embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(int(edge_type_embedding_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        node_dim = int(node_condition_dim)
        self.src_proj = None
        self.dst_proj = None
        self.node_pair_mlp = None
        if self.node_condition_enable:
            self.src_proj = nn.Linear(self.model_dim, node_dim, bias=False)
            self.dst_proj = nn.Linear(self.model_dim, node_dim, bias=False)
            if self.node_condition_mode == "src_dst":
                node_input_dim = 2 * node_dim
            elif self.node_condition_mode == "src_dst_diff":
                node_input_dim = 3 * node_dim
            elif self.node_condition_mode == "qk_like":
                node_input_dim = node_dim
            else:
                node_input_dim = 4 * node_dim
            self.node_pair_mlp = nn.Sequential(
                nn.LayerNorm(node_input_dim),
                nn.Linear(node_input_dim, int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            if bool(node_condition_zero_init):
                final = self.node_pair_mlp[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        logit_out = self.num_heads if self.logit_bias_per_head else 1
        if self.value_gate_per_channel:
            gate_out = self.head_dim * (self.num_heads if self.value_gate_per_head else 1)
        else:
            gate_out = self.num_heads if self.value_gate_per_head else 1
        self.logit_head = nn.Linear(int(hidden_dim), logit_out)
        self.gate_head = nn.Linear(int(hidden_dim), gate_out)

        if bool(logit_bias_init_zero):
            nn.init.zeros_(self.logit_head.weight)
            nn.init.zeros_(self.logit_head.bias)
        if bool(gate_init_identity):
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

    def _node_pair_features(self, src_hidden: torch.Tensor, dst_hidden: torch.Tensor) -> torch.Tensor:
        if bool(self.node_condition_detach):
            src_hidden = src_hidden.detach()
            dst_hidden = dst_hidden.detach()
        src_z = self.src_proj(src_hidden)
        dst_z = self.dst_proj(dst_hidden)
        if self.node_condition_mode == "src_dst":
            return torch.cat([src_z, dst_z], dim=-1)
        if self.node_condition_mode == "src_dst_diff":
            return torch.cat([src_z, dst_z, dst_z - src_z], dim=-1)
        if self.node_condition_mode == "qk_like":
            return src_z * dst_z
        return torch.cat([src_z, dst_z, src_z * dst_z, dst_z - src_z], dim=-1)

    def forward(
        self,
        edge_type: Optional[torch.Tensor],
        ref: torch.Tensor,
        src_hidden: Optional[torch.Tensor] = None,
        dst_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if edge_type is None:
            edge_count = int(src_hidden.size(1)) if src_hidden is not None and src_hidden.dim() == 3 else int(ref.size(0))
            edge_type = torch.zeros((edge_count,), device=ref.device, dtype=torch.long)
        edge_type = edge_type.to(device=ref.device, dtype=torch.long).clamp(min=0, max=self.num_edge_types - 1)
        ctx = self.net(self.edge_type_embedding(edge_type).to(dtype=ref.dtype))
        node_ctx = None
        if self.node_condition_enable and src_hidden is not None and dst_hidden is not None:
            if ctx.dim() == 2 and src_hidden.dim() == 3:
                ctx = ctx.unsqueeze(0).expand(src_hidden.size(0), -1, -1)
            node_ctx = self.node_pair_mlp(self._node_pair_features(src_hidden, dst_hidden)).to(dtype=ctx.dtype)
            ctx = ctx + node_ctx
        logit_bias = self.logit_head(ctx).to(dtype=ref.dtype)
        raw_gate = self.gate_head(ctx).to(dtype=ref.dtype)
        value_gate = 1.0 + self.edge_gate_scale * torch.tanh(raw_gate)
        if self.value_gate_per_channel:
            if self.value_gate_per_head:
                value_gate = value_gate.view(*value_gate.shape[:-1], self.num_heads, self.head_dim)
            else:
                value_gate = value_gate.view(*value_gate.shape[:-1], 1, self.head_dim)
        else:
            if self.value_gate_per_head:
                value_gate = value_gate.view(*value_gate.shape[:-1], self.num_heads, 1)
            else:
                value_gate = value_gate.view(*value_gate.shape[:-1], 1, 1)
        return logit_bias, value_gate, node_ctx



class HierarchicalMessagePassing(MessagePassing):
    """
    Custom message passing layer for hierarchical graphs with level awareness.
    
    This layer integrates information across different hierarchical levels
    using a modified attention mechanism that considers the hierarchical
    structure of the graph.
    """
    def __init__(
        self, 
        hidden_dim, 
        num_heads=8, 
        edge_dim=None, 
        level_dim=8, 
        dropout=0.1,
        use_edge_attr=True,
        max_seq_len=131072,
        learn_edge_from_attn: bool = True,
        l0_local_backend: str = "pyg",
        l0_local_window: int = 0,
        l0_local_causal_default: bool = False,
        local_attn_config: Optional[Dict] = None,
        local_attn_level_role_bias_enable: bool = True,
        local_attn_level_role_bias_scale: float = 1.0,
        local_attn_flash_dtype_cast: bool = False,
        cross_level_packed: bool = False,
        cross_level_qkv: str = "shared",
        local_attn_sampled_mode: str = "safe_sdpa",
        sparse_attn_mode: str = "off",
        sparse_attn_chunk_size: int = 0,
        per_level_local_qkv: bool = False,   # per-level Q/K/V for intra-level (local-window) attention;
                                             # the cross-level graph/edge path keeps the shared backbone QKV
        num_local_levels: int = 4,
        per_level_attn_mult: Optional[List[float]] = None,  # per-level local-attn dim mult (scales num_heads; head_dim fixed)
        local_attn_head_dim: int = 0,  # 0 = hidden//num_heads; >0 = up/down-project local attn to this head_dim
        local_pack_level_bias: bool = False,  # per-level per-head K/V tags for the packed mixed-level local call
        # Unified-softmax combination of the mixed window and the coarse lane for coarse
        # queries (LSE merge of the two calls) instead of adding the two attention outputs;
        # see _apply_local_pack_out. False = additive combine (previous behavior).
        local_pack_lane_merge: bool = False,
        # ONE flex_attention call with the block-sparse union mask (see model spec builder).
        local_pack_flex_union: bool = False,
        # Bidirectional pack (MaskGIT/diffusion): when the runtime causal flags are OFF,
        # run the packed mixed window + coarse lane as symmetric two-sided windows instead
        # of skipping pack entirely. Bidi has no AR order to leak, so the "closed window"
        # staggering is unnecessary — a node reads past AND future neighbors/coarse
        # summaries (incl. its own open parent). Reach becomes +-window slots per side.
        # flex_union/lane_merge stay causal-only (their masks/LSE recompute assume the
        # causal window) -> the additive combine is used.
        local_pack_bidirectional: bool = False,
        # xq/HQD packed-read fixes: score+read the fetched far tokens with the PRE-RoPE
        # shared q/k (content-only matching — the RoPE'd L0<->L0 geometry is only trained
        # inside the local window, so far relative angles are out-of-distribution noise),
        # and give the read softmax a learned zero-value sink key so a query with no
        # useful candidates can no-op instead of emitting a forced weighted average.
        hqd_read_prerope: bool = False,
        hqd_read_sink: bool = False,
        rope_level_axis_enable: bool = False,
        rope_level_axis_scale: float = 32.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        rope_mode: str = "auto",
        attention_source_gating_enable: bool = False,
        attention_source_gate_init_graph: float = 1.0,
        attention_source_gate_init_local: float = 0.5,
        attention_source_gate_init_hqd: float = 0.1,
        attention_source_gate_debug: bool = False,
        lateral_edge_trace_enable: bool = False,
        lateral_edge_trace_mode: str = "windowed_approx",
        lateral_edge_trace_decay: float = 0.95,
        lateral_edge_trace_eta: float = 0.02,
        lateral_edge_trace_alpha: float = 0.25,
        lateral_edge_trace_max: float = 2.0,
        lateral_edge_trace_per_head: bool = True,
        lateral_edge_trace_credit: str = "attn",
        lateral_edge_trace_center_per_dst: bool = True,
        lateral_edge_trace_update_during_eval: bool = False,
        lateral_edge_trace_detach: bool = True,
        lateral_edge_trace_debug: bool = False,
        edge_conditioning_enable: bool = False,
        edge_type_generator_enable: bool = False,
        edge_type_embedding_dim: int = 32,
        edge_condition_hidden_dim: int = 64,
        edge_condition_num_types: int = 16,
        edge_logit_bias_enable: bool = True,
        edge_value_gate_enable: bool = True,
        edge_logit_bias_per_head: bool = True,
        edge_value_gate_per_head: bool = False,
        edge_value_gate_per_channel: bool = True,
        edge_gate_init_identity: bool = True,
        edge_logit_bias_init_zero: bool = True,
        edge_condition_dropout: float = 0.0,
        edge_condition_debug: bool = False,
        edge_node_condition_enable: bool = False,
        edge_node_condition_detach: bool = True,
        edge_node_condition_dim: int = 32,
        edge_node_condition_mode: str = "src_dst_prod",
        edge_node_condition_zero_init: bool = True,
        edge_gate_scale: float = 0.1,
        attn_sparse_qk_mult: float = 1.0,
        attn_sparse_v_mult: float = 1.0,
    ):
        super().__init__(aggr='mean', node_dim=0)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_dim // num_heads
        self.edge_dim = edge_dim
        self.level_dim = level_dim
        self.use_edge_attr = use_edge_attr
        # --- Instantiate RoPE using the argument ---
        # RoPE typically operates on pairs within the head dimension.
        self.rope_mode = str(rope_mode).lower()
        if self.rope_mode not in {"auto", "1d", "2d", "2d_axial", "axial", "grid2d"}:
            self.rope_mode = "auto"
        self.rotary_pos_enc = RotaryPositionalEncoding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_mode=self.rope_mode,
        )
        # Local-attention head_dim override: decouples the per-level LOCAL attention head_dim
        # from hidden//num_heads. Lets a NARROW residual (small hidden_dim) still run full-
        # resolution attention by UP-projecting q/k/v to num_heads*local_attn_head_dim (e.g.
        # hidden 192 -> attn head_dim 64), attend, project back to hidden. Needs its own RoPE at
        # the wide head_dim (RoPE rotates pairs within head_dim). 0 => use head_dim (no change).
        self.local_attn_head_dim = int(local_attn_head_dim) if int(local_attn_head_dim) > 0 else self.head_dim
        if self.local_attn_head_dim != self.head_dim:
            self.rotary_pos_enc_attn = RotaryPositionalEncoding(
                dim=self.local_attn_head_dim,
                max_seq_len=max_seq_len,
                rope_mode=self.rope_mode,
            )
        else:
            self.rotary_pos_enc_attn = self.rotary_pos_enc
        # Packed mixed-level local call (local_pack_cross_level): per-level per-head K/V tags.
        # The scatter path carries a learned level-PAIR logit bias (_edge_level_attention_bias)
        # that flash cannot take; adding a zero-init embedding to the RoPE'd K instead makes
        # q·e_level a query-dependent level bias inside the same softmax (and the V tag lets
        # the output carry its source level). Zero init = bit-identical to untagged at start.
        self.local_pack_level_bias = bool(local_pack_level_bias)
        self.local_pack_lane_merge = bool(local_pack_lane_merge)
        self.local_pack_flex_union = bool(local_pack_flex_union)
        self.local_pack_bidirectional = bool(local_pack_bidirectional)
        if self.local_pack_level_bias:
            self.local_pack_level_k_emb = nn.Parameter(torch.zeros(4, self.num_heads, self.head_dim))
            self.local_pack_level_v_emb = nn.Parameter(torch.zeros(4, self.num_heads, self.head_dim))
        self.hqd_read_prerope = bool(hqd_read_prerope)
        # Zero-value sink key for the packed fetch read: zero init = sink logit 0 for every
        # query (a neutral extra softmax slot), so the model learns per-head/per-query how
        # much fetched content to discard. Value side is implicitly zero (no v param).
        self.hqd_read_sink_k = (
            nn.Parameter(torch.zeros(self.num_heads, self.head_dim)) if bool(hqd_read_sink) else None
        )
        self.l0_local_backend = str(l0_local_backend).lower()
        if self.l0_local_backend not in {"pyg", "flash", "xformers", "sdpa"}:
            self.l0_local_backend = "pyg"
        self.l0_local_window = max(0, int(l0_local_window))
        self.l0_local_causal_default = bool(l0_local_causal_default)
        self.l0_local_runtime_enable = False
        self.l0_local_runtime_causal = self.l0_local_causal_default
        self._l0_local_backend_warned = False

        # Multi-level local window attention config: {level: {window, causal, backend}}
        self.local_attn_config: Dict[int, Dict] = dict(local_attn_config) if local_attn_config else {}
        self.local_attn_level_role_bias_enable = bool(local_attn_level_role_bias_enable)
        self.local_attn_level_role_bias_scale = float(local_attn_level_role_bias_scale)
        self.local_attn_flash_dtype_cast = bool(local_attn_flash_dtype_cast)
        self.cross_level_packed = bool(cross_level_packed)
        self.local_attn_sampled_mode = str(local_attn_sampled_mode).lower()
        if self.local_attn_sampled_mode not in {"safe_sdpa", "flash_sorted", "off"}:
            self.local_attn_sampled_mode = "safe_sdpa"
        self.sparse_attn_mode = str(sparse_attn_mode).lower().replace("-", "_")
        if self.sparse_attn_mode not in {"off", "dst_block_checkpoint"}:
            self.sparse_attn_mode = "off"
        self.sparse_attn_chunk_size = max(0, int(sparse_attn_chunk_size))
        self.rope_level_axis_enable = bool(rope_level_axis_enable)
        self.rope_level_axis_scale = float(rope_level_axis_scale)
        self.norm_type = "rmsnorm" if norm_type is None else str(norm_type).lower()
        self.norm_eps = float(norm_eps)
        if self.rope_level_axis_enable and self.local_attn_level_role_bias_enable:
            logger.info(
                "Level-axis RoPE enabled; auto-disabling local-attn level role bias to preserve flash eligibility where possible."
            )
            self.local_attn_level_role_bias_enable = False
        # Runtime enable flag (set externally, like l0_local_runtime_enable)
        self.local_attn_runtime_enable: bool = False
        # Global runtime gate for local-attention causality (set externally by model AR mode)
        self.local_attn_runtime_causal_gate: bool = True
        # Optional runtime grouping vector [N] to isolate local attention neighborhoods.
        # When set, nodes only attend locally within their own group id at each level.
        self.local_attn_runtime_group: Optional[torch.Tensor] = None
        # Optional runtime per-level grid metadata for spatial local attention.
        # Expected shape: {level_idx: (grid_h, grid_w)}
        self.local_attn_runtime_level_grid_shapes: Dict[int, Tuple[int, int]] = {}
        self.local_attn_runtime_spatial_metric: str = "chebyshev"
        self._local_attn_warned: bool = False
        self._local_attn_runtime_logged_keys = set()
        self.local_attn_dense_mask_max_tokens = 8192
        self.hqd_sparse_project_active_only = False
        self.hqd_attn_impl = "scatter"        # scatter (edge softmax+scatter_add) | dense (fused gathered SDPA/flash)
        self.hqd_dense_backend = "sdpa"        # sdpa | flash (flash-varlen, falls back to sdpa)
        self.hqd_dense_max_pad_ratio = 1.5     # dense/sdpa: fall back to scatter if padded slots (G*Kmax) exceed this x #edges
        self.hqd_profile_enable = False
        self.hqd_graph_witness_enable = False
        self.hqd_graph_witness_topk = 4
        self.hqd_packed_witness_chunk_size = 2048
        self._last_graph_witness_ids: Optional[Dict[int, torch.Tensor]] = None
        self._last_graph_witness_scores: Optional[Dict[int, torch.Tensor]] = None
        # Backbone witness capture: the packed cross-level lane stashes each parent's
        # top-k direct children by its OWN (already computed) attention logits — the
        # near-free witness source (vs the scatter-path capture, which needs ragged
        # per-edge sorting, or the recompute build, which re-scores all descendants).
        self.hqd_backbone_witness_capture = False
        self.hqd_backbone_witness_topk = 4
        self._last_backbone_witness_ids: Optional[Dict[int, torch.Tensor]] = None
        self._last_backbone_witness_scores: Optional[Dict[int, torch.Tensor]] = None
        self._last_hqd_apply_ms: Optional[float] = None
        self.hqd_runtime_selector: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None
        self._last_hqd_runtime_added_total: Optional[int] = None
        self._last_hqd_runtime_stage_stats: Optional[Dict[str, int]] = None
        self._last_hqd_runtime_profile_stats: Optional[Dict[str, float]] = None
        self.attention_source_gating_enable = bool(attention_source_gating_enable)
        self.attention_source_gate_debug = bool(attention_source_gate_debug)
        self._last_attention_source_gates: Optional[Dict[str, float]] = None
        if self.attention_source_gating_enable:
            self.graph_source_gate_logit = nn.Parameter(_gate_logit(attention_source_gate_init_graph))
            self.local_source_gate_logit = nn.Parameter(_gate_logit(attention_source_gate_init_local))
            self.hqd_source_gate_logit = nn.Parameter(_gate_logit(attention_source_gate_init_hqd))
        self.lateral_edge_trace_enable = bool(lateral_edge_trace_enable)
        self.lateral_edge_trace_mode = str(lateral_edge_trace_mode).lower().replace("-", "_")
        if self.lateral_edge_trace_mode not in {"windowed_approx", "true_scatter", "true_lateral"}:
            self.lateral_edge_trace_mode = "windowed_approx"
        self.lateral_edge_trace_decay = float(lateral_edge_trace_decay)
        self.lateral_edge_trace_eta = float(lateral_edge_trace_eta)
        self.lateral_edge_trace_alpha = float(lateral_edge_trace_alpha)
        self.lateral_edge_trace_max = float(lateral_edge_trace_max)
        self.lateral_edge_trace_per_head = bool(lateral_edge_trace_per_head)
        self.lateral_edge_trace_credit = str(lateral_edge_trace_credit).lower()
        self.lateral_edge_trace_center_per_dst = bool(lateral_edge_trace_center_per_dst)
        self.lateral_edge_trace_update_during_eval = bool(lateral_edge_trace_update_during_eval)
        self.lateral_edge_trace_detach = bool(lateral_edge_trace_detach)
        self.lateral_edge_trace_debug = bool(lateral_edge_trace_debug)
        self._last_lateral_edge_trace_stats: Optional[Dict[str, float]] = None
        self.register_buffer("graph_edge_trace", torch.empty(0), persistent=False)
        self.register_buffer("local_rel_trace", torch.empty(0), persistent=False)
        self.edge_conditioning_enable = bool(edge_conditioning_enable)
        self.edge_type_generator_enable = bool(edge_type_generator_enable)
        self.edge_logit_bias_enable = bool(edge_logit_bias_enable)
        self.edge_value_gate_enable = bool(edge_value_gate_enable)
        self.edge_condition_debug = bool(edge_condition_debug)
        self.edge_node_condition_enable = bool(edge_node_condition_enable)
        self.edge_node_condition_detach = bool(edge_node_condition_detach)
        self.edge_node_condition_dim = int(edge_node_condition_dim)
        self.edge_node_condition_mode = str(edge_node_condition_mode).lower()
        self.edge_node_condition_zero_init = bool(edge_node_condition_zero_init)
        self.edge_gate_scale = float(edge_gate_scale)
        self._last_edge_condition_stats: Optional[Dict[str, float]] = None
        self.edge_conditioner = None
        if self.edge_conditioning_enable:
            self.edge_conditioner = EdgeConditioner(
                model_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_edge_types=int(edge_condition_num_types),
                edge_type_embedding_dim=int(edge_type_embedding_dim),
                hidden_dim=int(edge_condition_hidden_dim),
                logit_bias_per_head=bool(edge_logit_bias_per_head),
                value_gate_per_head=bool(edge_value_gate_per_head),
                value_gate_per_channel=bool(edge_value_gate_per_channel),
                logit_bias_init_zero=bool(edge_logit_bias_init_zero),
                gate_init_identity=bool(edge_gate_init_identity),
                dropout=float(edge_condition_dropout),
                edge_gate_scale=self.edge_gate_scale,
                node_condition_enable=self.edge_node_condition_enable,
                node_condition_detach=self.edge_node_condition_detach,
                node_condition_dim=self.edge_node_condition_dim,
                node_condition_mode=self.edge_node_condition_mode,
                node_condition_zero_init=self.edge_node_condition_zero_init,
            )

        self.learn_edge_from_attn = bool(learn_edge_from_attn and use_edge_attr)
        if self.learn_edge_from_attn:
            # maps [E, num_heads] -> [E, 1]
            self.edge_combine = nn.Linear(num_heads, 1, bias=False)

        # stack-based cache to smuggle edge weights out of message()
        # each entry is set by message() during its corresponding propagate() call;
        # forward() pushes before and pops after, making it safe for recursion
        self._edge_attr_stack: List[Optional[torch.Tensor]] = []
        
        
        # Project queries, keys, values (shared backbone — used by the cross-level
        # graph/edge message passing, i.e. moving information up/down the hierarchy).
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Optional per-level Q/K/V for the intra-level (local-window) attention path.
        # The cross-level/edge path above stays shared (the "vertical backbone"). Each
        # level gets its own intra-level projections, copy-initialized from the shared
        # ones so an enabled model starts bit-identical to the shared-QKV baseline and
        # then specializes. See _sync_per_level_qkv_from_shared() for the resume path.
        self.per_level_local_qkv = bool(per_level_local_qkv)
        self.num_local_levels = int(num_local_levels)
        if self.per_level_local_qkv:
            self.q_proj_level = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_local_levels)])
            self.k_proj_level = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_local_levels)])
            self.v_proj_level = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_local_levels)])
            self._sync_per_level_qkv_from_shared()

        # Per-level LOCAL-attention dimension: each level's intra-level window attention runs at
        # num_heads_L = round(mult_L * num_heads) heads with head_dim FIXED (full per-head
        # resolution). L0 (most nodes) -> fewer heads = cheap; coarse levels -> more heads =
        # capacity (U-Net schedule inside the hierarchy). hidden->d_L q/k/v + d_L->hidden out
        # proj per level; cross-level backbone stays at hidden. Reuses the Fat-QKV decoupled-dim
        # pattern, per level, on the local path.
        self.per_level_attn_mult = [float(m) for m in (per_level_attn_mult or [])]
        self.per_level_attn_active = len(self.per_level_attn_mult) > 0
        if self.per_level_attn_active:
            self.per_level_attn_heads = []
            self.q_proj_attn_level = nn.ModuleList()
            self.k_proj_attn_level = nn.ModuleList()
            self.v_proj_attn_level = nn.ModuleList()
            self.out_proj_attn_level = nn.ModuleList()
            for m in self.per_level_attn_mult:
                h = max(1, int(round(m * self.num_heads)))
                d = h * self.local_attn_head_dim
                self.per_level_attn_heads.append(h)
                self.q_proj_attn_level.append(nn.Linear(hidden_dim, d))
                self.k_proj_attn_level.append(nn.Linear(hidden_dim, d))
                self.v_proj_attn_level.append(nn.Linear(hidden_dim, d))
                self.out_proj_attn_level.append(nn.Linear(d, hidden_dim))

        # Level embedding
        self.level_embedding = nn.Embedding(4, level_dim)  # 4 levels: L0, L1, L2, L3

        # Per-level cross-level (backbone + HQD) Q/K/V. Default "shared" reuses the single
        # q/k/v_proj (current behaviour). The other modes route each node through its LEVEL's
        # projection so an L0 query attends to an L3 key in a level-specific subspace (the
        # vertical-routing analogue of per-level local QKV). Copy-initialized from the shared
        # projections so an enabled model starts bit-identical, then specializes.
        #   per_level_qk  : per-level Q and K, shared V (routing lives in q.k; V is the payload)
        #   per_level_qkv : per-level Q, K and V
        #   reuse_local   : reuse the existing per-level LOCAL projections (no new params)
        self.cross_level_qkv = str(cross_level_qkv or "shared").lower()
        _nlv = int(self.level_embedding.num_embeddings)
        if self.cross_level_qkv in ("per_level_qk", "per_level_qkv"):
            self.q_proj_xlevel = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(_nlv)])
            self.k_proj_xlevel = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(_nlv)])
            if self.cross_level_qkv == "per_level_qkv":
                self.v_proj_xlevel = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(_nlv)])
            self._sync_cross_level_qkv_from_shared()
            # Resuming a checkpoint that predates this feature: the xlevel projections are
            # "missing" -> fan them out from the just-loaded shared q/k/v so the warm start
            # is bit-identical to shared and then specializes.
            self.register_load_state_dict_post_hook(self._xlevel_load_post_hook)


        # Edge transformation (if edge attributes available)
        if use_edge_attr and edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
            
        # Level attention
        self.level_attn = nn.Linear(level_dim * 2, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # --- Fat-QKV: decoupled wide Q/K and V head dims for the SPARSE-SCATTER paths
        # (cross-level graph message passing + HQD scatter read). Flash/SDPA local-window
        # attention requires qk_head_dim == v_head_dim, so it always keeps the shared
        # head_dim projections above; only the scatter paths consult the wide ones.
        #   attn_sparse_qk_mult: widen qk_head_dim (lifts the per-head routing-form rank,
        #       which is currently capped at head_dim < model_dim). Useful up to qk==model_dim.
        #   attn_sparse_v_mult : widen v_head_dim (richer payload carried THROUGH the sparse
        #       aggregation — the capacity a dense model can't afford). Down-projected by out.
        # Both default to 1.0 (off): the scatter paths reuse q/k/v_proj + out_proj and are
        # bit-identical to the baseline. The wide modules are created only when enabled.
        self.attn_sparse_qk_mult = float(attn_sparse_qk_mult)
        self.attn_sparse_v_mult = float(attn_sparse_v_mult)
        self.qk_head_dim_w = max(1, int(round(self.head_dim * self.attn_sparse_qk_mult)))
        self.v_head_dim_w = max(1, int(round(self.head_dim * self.attn_sparse_v_mult)))
        self.sparse_qk_wide = self.qk_head_dim_w != self.head_dim
        self.sparse_v_wide = self.v_head_dim_w != self.head_dim
        self.sparse_wide_enable = bool(self.sparse_qk_wide or self.sparse_v_wide)
        if self.sparse_wide_enable:
            qk_dim_w = self.num_heads * self.qk_head_dim_w
            v_dim_w = self.num_heads * self.v_head_dim_w
            self.q_proj_w = nn.Linear(hidden_dim, qk_dim_w)
            self.k_proj_w = nn.Linear(hidden_dim, qk_dim_w)
            self.v_proj_w = nn.Linear(hidden_dim, v_dim_w)
            self.out_proj_w = nn.Linear(v_dim_w, hidden_dim)
            # Wide edge-feature projection so the additive q_i·edge_features term matches the
            # wide qk head dim (only needed when qk is widened and edge features are used).
            if self.sparse_qk_wide and use_edge_attr and edge_dim is not None:
                self.edge_proj_w = nn.Linear(edge_dim, qk_dim_w)
            # Separate RoPE sized to the wide qk head dim (RoPE rotates pairs within head_dim).
            if self.sparse_qk_wide:
                self.rotary_pos_enc_w = RotaryPositionalEncoding(
                    dim=self.qk_head_dim_w,
                    max_seq_len=max_seq_len,
                    rope_mode=self.rope_mode,
                )
            else:
                self.rotary_pos_enc_w = self.rotary_pos_enc
            # Wide qk is incompatible with the additive edge_proj feature term (sized to
            # head_dim) and per-channel value gating (sized to head_dim); the scatter paths
            # fall back to the dim-agnostic scalar variants when wide. See _forward_batched.
            logger.info(
                "Fat-QKV enabled on sparse paths: qk_head_dim %d->%d, v_head_dim %d->%d",
                self.head_dim, self.qk_head_dim_w, self.head_dim, self.v_head_dim_w,
            )
        self.dropout = nn.Dropout(dropout)

        

    def _source_gate(self, source: str, ref: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self, "attention_source_gating_enable", False)):
            return ref.new_tensor(1.0)
        param = getattr(self, f"{source}_source_gate_logit")
        value = torch.sigmoid(param).to(device=ref.device, dtype=ref.dtype)
        if bool(getattr(self, "attention_source_gate_debug", False)):
            stats = self._last_attention_source_gates or {}
            stats[str(source)] = float(value.detach().float().item())
            self._last_attention_source_gates = stats
        return value

    def _compute_source_gates(self, ref: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if not bool(getattr(self, "attention_source_gating_enable", False)):
            return None

        gates = {
            "graph": torch.sigmoid(self.graph_source_gate_logit).to(device=ref.device, dtype=ref.dtype),
            "local": torch.sigmoid(self.local_source_gate_logit).to(device=ref.device, dtype=ref.dtype),
            "hqd": torch.sigmoid(self.hqd_source_gate_logit).to(device=ref.device, dtype=ref.dtype),
        }
        if bool(getattr(self, "attention_source_gate_debug", False)):
            self._last_attention_source_gates = {
                name: float(value.detach().float().item())
                for name, value in gates.items()
            }
        return gates

    def reset_lateral_edge_traces(self) -> None:
        self.graph_edge_trace = torch.empty(0, device=self.graph_edge_trace.device)
        self.local_rel_trace = torch.empty(0, device=self.local_rel_trace.device)
        self._last_lateral_edge_trace_stats = None

    def _trace_updates_allowed(self) -> bool:
        return bool(self.training or getattr(self, "lateral_edge_trace_update_during_eval", False))

    def _trace_mode_allows_graph(self) -> bool:
        return bool(
            getattr(self, "lateral_edge_trace_enable", False)
            and str(getattr(self, "lateral_edge_trace_mode", "windowed_approx")) in {"true_scatter", "true_lateral"}
        )

    def _trace_mode_allows_window(self) -> bool:
        return bool(
            getattr(self, "lateral_edge_trace_enable", False)
            and str(getattr(self, "lateral_edge_trace_mode", "windowed_approx")) in {"windowed_approx", "true_lateral"}
        )

    def _trace_heads(self) -> int:
        return int(self.num_heads if getattr(self, "lateral_edge_trace_per_head", True) else 1)

    def _ensure_graph_edge_trace(self, num_edges: int, ref: torch.Tensor) -> torch.Tensor:
        heads = self._trace_heads()
        shape = (int(num_edges), heads)
        if tuple(self.graph_edge_trace.shape) != shape or self.graph_edge_trace.device != ref.device or self.graph_edge_trace.dtype != ref.dtype:
            self.graph_edge_trace = torch.zeros(shape, device=ref.device, dtype=ref.dtype)
        return self.graph_edge_trace

    def _ensure_local_rel_trace(self, num_offsets: int, ref: torch.Tensor) -> torch.Tensor:
        heads = self._trace_heads()
        shape = (heads, int(num_offsets))
        if tuple(self.local_rel_trace.shape) != shape or self.local_rel_trace.device != ref.device or self.local_rel_trace.dtype != ref.dtype:
            self.local_rel_trace = torch.zeros(shape, device=ref.device, dtype=ref.dtype)
        return self.local_rel_trace

    def _record_lateral_trace_stats(self, kind: str, trace: torch.Tensor, credit: Optional[torch.Tensor], update_norm: Optional[torch.Tensor]) -> None:
        if not bool(getattr(self, "lateral_edge_trace_debug", False)):
            return
        with torch.no_grad():
            stats = {
                f"{kind}_mean_abs_trace": float(trace.detach().abs().mean().item()) if trace.numel() else 0.0,
                f"{kind}_max_abs_trace": float(trace.detach().abs().max().item()) if trace.numel() else 0.0,
            }
            if credit is not None and credit.numel():
                credit_f = credit.detach().float()
                stats[f"{kind}_credit_mean"] = float(credit_f.mean().item())
                stats[f"{kind}_credit_std"] = float(credit_f.std(unbiased=False).item())
            if update_norm is not None:
                stats[f"{kind}_trace_update_norm"] = float(update_norm.detach().float().item())
            current = self._last_lateral_edge_trace_stats or {}
            current.update(stats)
            self._last_lateral_edge_trace_stats = current

    def _apply_graph_trace_bias(self, logits: torch.Tensor, num_edges: int) -> torch.Tensor:
        if not self._trace_mode_allows_graph() or float(getattr(self, "lateral_edge_trace_alpha", 0.0)) == 0.0:
            return logits
        trace = self._ensure_graph_edge_trace(num_edges, logits)
        bias = trace if trace.size(-1) == self.num_heads else trace.expand(-1, self.num_heads)
        if logits.dim() == 3:
            return logits + float(self.lateral_edge_trace_alpha) * bias.unsqueeze(0)
        return logits + float(self.lateral_edge_trace_alpha) * bias

    def _edge_conditioning_active(self) -> bool:
        return bool(getattr(self, "edge_conditioning_enable", False) and self.edge_conditioner is not None)

    def _edge_condition(
        self,
        edge_type: Optional[torch.Tensor],
        ref: torch.Tensor,
        src_hidden: Optional[torch.Tensor] = None,
        dst_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._edge_conditioning_active():
            return None, None
        logit_bias, value_gate, node_ctx = self.edge_conditioner(edge_type, ref, src_hidden=src_hidden, dst_hidden=dst_hidden)
        if not bool(getattr(self, "edge_logit_bias_enable", True)):
            logit_bias = None
        if not bool(getattr(self, "edge_value_gate_enable", True)):
            value_gate = None
        if bool(getattr(self, "edge_condition_debug", False)):
            stats: Dict[str, float] = {}
            if logit_bias is not None and logit_bias.numel() > 0:
                lb = logit_bias.detach().float()
                stats["edge_logit_bias_mean"] = float(lb.mean().item())
                stats["edge_logit_bias_std"] = float(lb.std(unbiased=False).item())
                stats["edge_logit_bias_absmax"] = float(lb.abs().max().item())
            if value_gate is not None and value_gate.numel() > 0:
                vg = value_gate.detach().float()
                stats["edge_value_gate_mean"] = float(vg.mean().item())
                stats["edge_value_gate_std"] = float(vg.std(unbiased=False).item())
                stats["edge_value_gate_absmax"] = float(vg.abs().max().item())
                stats["edge_value_gate_min"] = float(vg.min().item())
                stats["edge_value_gate_max"] = float(vg.max().item())
            if node_ctx is not None and node_ctx.numel() > 0:
                nc = node_ctx.detach().float()
                stats["edge_node_ctx_norm"] = float(nc.norm(dim=-1).mean().item())
                stats["edge_node_ctx_std"] = float(nc.std(unbiased=False).item())
                stats["edge_node_ctx_absmax"] = float(nc.abs().max().item())
            self._last_edge_condition_stats = stats
        return logit_bias, value_gate

    def _apply_edge_logit_bias(self, logits: torch.Tensor, logit_bias: Optional[torch.Tensor]) -> torch.Tensor:
        if logit_bias is None:
            return logits
        if logit_bias.size(-1) == 1 and logits.size(-1) != 1:
            logit_bias = logit_bias.expand(*logit_bias.shape[:-1], logits.size(-1))
        if logits.dim() == 3 and logit_bias.dim() == 2:
            return logits + logit_bias.unsqueeze(0)
        return logits + logit_bias

    def _apply_edge_value_gate(self, values: torch.Tensor, value_gate: Optional[torch.Tensor]) -> torch.Tensor:
        if value_gate is None:
            return values
        if values.dim() == 4 and value_gate.dim() == 3:
            return values * value_gate.unsqueeze(0)
        return values * value_gate

    # --- Fat-QKV accessors for the sparse-scatter paths (cross-level + HQD). When the
    # wide flags are off these return the shared baseline projections / head_dim, so the
    # callers are bit-identical to the original code.
    @property
    def sparse_qk_head_dim(self) -> int:
        return self.qk_head_dim_w if getattr(self, "sparse_wide_enable", False) else self.head_dim

    @property
    def sparse_v_head_dim(self) -> int:
        return self.v_head_dim_w if getattr(self, "sparse_wide_enable", False) else self.head_dim

    def sparse_proj_q(self, x: torch.Tensor) -> torch.Tensor:
        return (self.q_proj_w if getattr(self, "sparse_wide_enable", False) else self.q_proj)(x)

    def sparse_proj_k(self, x: torch.Tensor) -> torch.Tensor:
        return (self.k_proj_w if getattr(self, "sparse_wide_enable", False) else self.k_proj)(x)

    def sparse_proj_v(self, x: torch.Tensor) -> torch.Tensor:
        return (self.v_proj_w if getattr(self, "sparse_wide_enable", False) else self.v_proj)(x)

    def sparse_out_proj(self, x: torch.Tensor) -> torch.Tensor:
        return (self.out_proj_w if getattr(self, "sparse_wide_enable", False) else self.out_proj)(x)

    def sparse_apply_rope(self, qk_flat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        rope = self.rotary_pos_enc_w if getattr(self, "sparse_qk_wide", False) else self.rotary_pos_enc
        return rope.apply_rotary_pos_emb(qk_flat, pos)

    def _edge_level_attention_bias(
        self,
        node_level: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute static level-pair attention bias without per-edge MLP calls."""
        nl = node_level.to(device=device, dtype=torch.long)
        src_level = nl.index_select(0, src)
        dst_level = nl.index_select(0, dst)
        num_levels = int(self.level_embedding.num_embeddings)
        if src_level.numel() == 0:
            return self.level_attn.weight.new_zeros((0, self.num_heads)).to(device=device)
        emb = self.level_embedding.weight
        pair_dst = torch.arange(num_levels, device=device, dtype=torch.long).repeat_interleave(num_levels)
        pair_src = torch.arange(num_levels, device=device, dtype=torch.long).repeat(num_levels)
        pair_concat = torch.cat([emb.index_select(0, pair_dst), emb.index_select(0, pair_src)], dim=-1)
        pair_weights = self.level_attn(pair_concat)
        pair_diff = torch.abs(emb.index_select(0, pair_dst)[:, 0:1] - emb.index_select(0, pair_src)[:, 0:1])
        pair_bias = pair_weights * (1.0 / (1.0 + pair_diff))
        pair_id = dst_level * num_levels + src_level
        return pair_bias.index_select(0, pair_id)

    @staticmethod
    def _level_slice_from_offsets(
        level_offsets: Optional[torch.Tensor],
        level: int,
        num_nodes: int,
    ) -> Optional[Tuple[int, int]]:
        if level_offsets is None:
            return None
        if not isinstance(level_offsets, torch.Tensor):
            return None
        if level_offsets.numel() <= int(level) + 1:
            return None
        start = int(level_offsets[int(level)].detach().item())
        end = int(level_offsets[int(level) + 1].detach().item())
        if start < 0 or end < start or end > int(num_nodes):
            return None
        return start, end

    def _batched_dst_index_flat(self, dst: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        cache = getattr(self, "_batched_dst_index_cache", {})
        key = (int(dst.data_ptr()), int(batch_size), int(num_nodes), str(dst.device))
        cached = cache.get(key)
        if cached is not None:
            return cached
        batch_offsets = (torch.arange(int(batch_size), device=dst.device, dtype=torch.long) * int(num_nodes)).view(int(batch_size), 1)
        index_flat = (dst.view(1, -1) + batch_offsets).reshape(-1)
        if len(cache) > 16:
            cache.clear()
        cache[key] = index_flat
        self._batched_dst_index_cache = cache
        return index_flat

    # ------------------------------------------------------------------
    # Tier-3 packed cross-level backbone attention (additive, opt-in).
    # A dense/batched alternative to the per-edge scatter aggregation: group each
    # destination's parents into a padded fixed-K block (PER DESTINATION LEVEL, so
    # each level gets its own K) and run [B,Q,K,H,D] attention. Tensor-core friendly,
    # no segment-softmax/scatter. Pays off at high, fairly-uniform cross-level degree.
    #
    # CONSERVE GRAPH OPS: opt-in via `cross_level_packed`; the scatter path stays the
    # default and the fallback for every graph feature this v1 doesn't pack (edge
    # features, edge conditioning, learned edges, graph trace, witness capture, fat-QKV).
    # Mathematically identical to scatter for the supported (basic backbone) case.
    # ------------------------------------------------------------------
    def _cross_level_packed_eligible(self, edge_attr) -> bool:
        if not bool(getattr(self, "cross_level_packed", False)):
            return False
        if getattr(self, "sparse_wide_enable", False):
            return False  # fat-QKV uses decoupled wide q/k/v; v1 packs the narrow shared q/k/v
        if edge_attr is not None and self.use_edge_attr:
            return False
        if self._edge_conditioning_active():
            return False
        if bool(getattr(self, "learn_edge_from_attn", False)):
            return False
        if self._trace_mode_allows_graph():
            return False
        if bool(getattr(self, "hqd_graph_witness_enable", False)):
            return False
        return True

    def _build_cross_level_packed_candidates(self, src, dst, node_level, num_nodes):
        """Per-destination-level padded parent lists. Static for an edge set -> cached.
        Returns list of (level, dst_nodes[QL], cand[QL,Kmax] (-1 pad), src_levels[QL,Kmax])."""
        cache = getattr(self, "_cross_level_packed_cache", {})
        key = (int(src.data_ptr()), int(dst.data_ptr()), int(num_nodes), str(dst.device))
        cached = cache.get(key)
        if cached is not None:
            return cached
        device = dst.device
        nl = node_level.to(device=device, dtype=torch.long)
        dst_level_e = nl.index_select(0, dst)
        blocks = []
        for L in torch.unique(dst_level_e).tolist():
            emask = dst_level_e == int(L)
            d = dst[emask]
            s = src[emask]
            if d.numel() == 0:
                continue
            dst_nodes, inv = torch.unique(d, return_inverse=True)  # [QL], inv[E_L]
            QL = int(dst_nodes.numel())
            order = torch.argsort(inv, stable=True)
            inv_s = inv.index_select(0, order)
            s_s = s.index_select(0, order)
            counts = torch.bincount(inv_s, minlength=QL)
            Kmax = int(counts.max().item()) if QL > 0 else 0
            if Kmax <= 0:
                continue
            group_start = torch.zeros(QL, dtype=torch.long, device=device)
            if QL > 1:
                group_start[1:] = counts.cumsum(0)[:-1]
            slot = torch.arange(inv_s.numel(), device=device, dtype=torch.long) - group_start.index_select(0, inv_s)
            cand = torch.full((QL, Kmax), -1, dtype=torch.long, device=device)
            cand[inv_s, slot] = s_s
            cand_levels = nl.index_select(0, cand.clamp(min=0).reshape(-1)).view(QL, Kmax)
            src_levels = torch.where(cand >= 0, cand_levels, torch.full_like(cand, -1))
            blocks.append((int(L), dst_nodes, cand, src_levels))
        if len(cache) > 8:
            cache.clear()
        cache[key] = blocks
        self._cross_level_packed_cache = cache
        return blocks

    def _cross_level_packed_attention(self, q, k, v, src, dst, node_level, num_nodes, B, qk_hd, v_hd):
        """Packed dense cross-level attention; returns [B, num_nodes, H, v_hd] (pre out_proj),
        mathematically equal to the scatter aggregation for the basic backbone case."""
        blocks = self._build_cross_level_packed_candidates(src, dst, node_level, int(num_nodes))
        if not blocks:
            return None
        device = q.device
        H = self.num_heads
        nl = node_level.to(device=device, dtype=torch.long)
        num_levels = int(self.level_embedding.num_embeddings)
        # Same level-pair bias table the scatter path uses (per (dst_level, src_level)).
        emb = self.level_embedding.weight
        pair_dst = torch.arange(num_levels, device=device, dtype=torch.long).repeat_interleave(num_levels)
        pair_src = torch.arange(num_levels, device=device, dtype=torch.long).repeat(num_levels)
        pair_concat = torch.cat([emb.index_select(0, pair_dst), emb.index_select(0, pair_src)], dim=-1)
        pair_weights = self.level_attn(pair_concat)
        pair_diff = torch.abs(emb.index_select(0, pair_dst)[:, 0:1] - emb.index_select(0, pair_src)[:, 0:1])
        pair_bias = pair_weights * (1.0 / (1.0 + pair_diff))  # [num_levels^2, H]

        out = q.new_zeros((B, int(num_nodes), H, v_hd))
        flat_k = k.reshape(B * int(num_nodes), H, qk_hd)
        flat_v = v.reshape(B * int(num_nodes), H, v_hd)
        batch_off = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1) * int(num_nodes)
        neg = torch.finfo(q.dtype).min
        scale = 1.0 / math.sqrt(float(qk_hd))
        capture_witness = bool(getattr(self, "hqd_backbone_witness_capture", False))
        cap_k = max(1, int(getattr(self, "hqd_backbone_witness_topk", 4)))
        cap_ids: Dict[int, torch.Tensor] = {}
        cap_scores: Dict[int, torch.Tensor] = {}
        for (L, dst_nodes, cand, src_levels) in blocks:
            QL = int(dst_nodes.numel())
            Kmax = int(cand.size(1))
            valid = cand >= 0
            safe = cand.clamp(min=0)
            gidx = (safe.unsqueeze(0) + batch_off).reshape(-1)  # [B*QL*Kmax]
            k_c = flat_k.index_select(0, gidx).view(B, QL, Kmax, H, qk_hd)
            v_c = flat_v.index_select(0, gidx).view(B, QL, Kmax, H, v_hd)
            q_c = q.index_select(1, dst_nodes)  # [B, QL, H, D]
            scores = (q_c.unsqueeze(2) * k_c).sum(-1) * scale  # [B, QL, Kmax, H]
            pair_id = int(L) * num_levels + src_levels.clamp(min=0)  # [QL, Kmax]
            lvl_bias = pair_bias.index_select(0, pair_id.reshape(-1)).view(QL, Kmax, H)
            scores = scores + lvl_bias.unsqueeze(0)
            scores = scores.masked_fill(~valid.view(1, QL, Kmax, 1), neg)
            if capture_witness and 1 <= int(L) <= 3:
                # Free witness harvest: each parent's top-k DIRECT children by the
                # attention logits just computed (detached metadata, no extra GEMM).
                with torch.no_grad():
                    child_mask = src_levels == (int(L) - 1)          # [QL, Kmax]
                    s = scores.detach().float().mean(dim=-1)         # [B, QL, Kmax]
                    fneg = torch.finfo(s.dtype).min
                    s = s.masked_fill(~(child_mask & valid).view(1, QL, Kmax), fneg)
                    kk = min(cap_k, Kmax)
                    vals, idx = torch.topk(s, k=kk, dim=-1)          # [B, QL, kk]
                    ids = cand.clamp(min=0).view(1, QL, Kmax).expand(B, QL, Kmax).gather(-1, idx)
                    ids = torch.where(vals > (fneg * 0.5), ids, torch.full_like(ids, -1))
                    full_ids = torch.full((B, int(num_nodes), cap_k), -1, dtype=torch.long, device=device)
                    full_sc = torch.full((B, int(num_nodes), cap_k), fneg, dtype=s.dtype, device=device)
                    full_ids[:, :, :kk].index_copy_(1, dst_nodes, ids)
                    full_sc[:, :, :kk].index_copy_(1, dst_nodes, vals)
                    cap_ids[int(L)] = full_ids
                    cap_scores[int(L)] = full_sc
            weights = torch.softmax(scores, dim=2)
            weights = torch.where(valid.view(1, QL, Kmax, 1), weights, torch.zeros_like(weights))
            weights = self.dropout(weights)  # parity with the scatter path's post-softmax dropout
            msg = (weights.unsqueeze(-1) * v_c).sum(2)  # [B, QL, H, v_hd]
            out.index_copy_(1, dst_nodes, msg.to(out.dtype))
        # Parity with the scatter path's detached witness/trace resets (both no-ops here).
        self._last_graph_witness_ids = None
        self._last_graph_witness_scores = None
        self._last_backbone_witness_ids = cap_ids or None
        self._last_backbone_witness_scores = cap_scores or None
        return out

    def _chunk_index_flat(self, dst_chunk: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        batch_offsets = (torch.arange(int(batch_size), device=dst_chunk.device, dtype=torch.long) * int(num_nodes)).view(int(batch_size), 1)
        return (dst_chunk.view(1, -1) + batch_offsets).reshape(-1)

    def _edge_attr_select_to_batched(self, edge_attr: torch.Tensor, edge_ids: torch.Tensor, batch_size: int, num_edges: int) -> torch.Tensor:
        selected_edges = int(edge_ids.numel())
        if edge_attr.dim() == 1:
            return edge_attr.index_select(0, edge_ids).view(1, selected_edges, 1).expand(batch_size, -1, -1)
        if edge_attr.dim() == 2:
            if edge_attr.size(0) == batch_size and edge_attr.size(1) == num_edges:
                return edge_attr.index_select(1, edge_ids).unsqueeze(-1)
            if edge_attr.size(0) == num_edges:
                return edge_attr.index_select(0, edge_ids).unsqueeze(0).expand(batch_size, -1, -1)
        if edge_attr.dim() == 3 and edge_attr.size(0) == batch_size and edge_attr.size(1) == num_edges:
            return edge_attr.index_select(1, edge_ids)
        raise ValueError(f"Unsupported batched edge_attr shape for checkpointed sparse attention: {tuple(edge_attr.shape)}")

    def _add_edge_attr_to_chunk_logits(
        self,
        attn_chunk: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        edge_ids: torch.Tensor,
        batch_size: int,
        num_edges: int,
        q_i_chunk: torch.Tensor,
    ) -> torch.Tensor:
        if edge_attr is None or not self.use_edge_attr:
            return attn_chunk
        edge_attr_b = self._edge_attr_select_to_batched(edge_attr, edge_ids, batch_size, num_edges)
        feat_dim = int(edge_attr_b.size(-1))
        if feat_dim == 1:
            return attn_chunk + edge_attr_b.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads)
        if self.edge_dim is not None and feat_dim == int(self.edge_dim):
            selected_edges = int(edge_ids.numel())
            edge_features = self.edge_proj(edge_attr_b.reshape(batch_size * selected_edges, feat_dim))
            edge_features = edge_features.view(batch_size, selected_edges, self.num_heads, self.head_dim)
            return attn_chunk + (q_i_chunk * edge_features).sum(dim=-1) / math.sqrt(self.head_dim)
        raise ValueError(f"Unsupported edge_attr feature dim {feat_dim}; expected 1 or edge_dim={self.edge_dim}")

    def _dst_block_ranges_from_sorted_dst(self, dst_sorted: torch.Tensor, target_edges: int) -> List[Tuple[int, int]]:
        if dst_sorted.numel() == 0:
            return []
        unique_dst, counts = torch.unique_consecutive(dst_sorted, return_counts=True)
        del unique_dst
        counts_cpu = counts.detach().to("cpu", dtype=torch.long).tolist()
        ranges: List[Tuple[int, int]] = []
        start = 0
        cursor = 0
        target_edges = max(1, int(target_edges))
        for count in counts_cpu:
            count = int(count)
            if cursor > start and (cursor - start + count) > target_edges:
                ranges.append((start, cursor))
                start = cursor
            cursor += count
        if cursor > start:
            ranges.append((start, cursor))
        return ranges

    def _sparse_graph_attention_chunked_batched(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_level: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
        num_edges: int,
    ) -> Optional[torch.Tensor]:
        mode = str(getattr(self, "sparse_attn_mode", "off")).lower().replace("-", "_")
        block_edges = int(getattr(self, "sparse_attn_chunk_size", 0) or 0)
        if mode != "dst_block_checkpoint" or block_edges <= 0 or num_edges <= 0:
            return None
        if self._edge_conditioning_active() or self.learn_edge_from_attn or self._trace_mode_allows_graph():
            return None

        B = int(q.size(0))
        device = q.device
        order = torch.argsort(dst, stable=True)
        dst_sorted = dst.index_select(0, order)
        ranges = self._dst_block_ranges_from_sorted_dst(dst_sorted, block_edges)
        if not ranges:
            return None

        use_checkpoint = bool(self.training and torch.is_grad_enabled())
        out = q.new_zeros((B, int(num_nodes), self.hidden_dim))
        has_edge_attr = bool(edge_attr is not None and self.use_edge_attr)
        edge_attr_arg = edge_attr if has_edge_attr else q.new_empty((0,))
        dropout_p = float(self.dropout.p) if self.training else 0.0

        for start, end in ranges:
            edge_ids = order[start:end]
            src_b = src.index_select(0, edge_ids)
            dst_b = dst_sorted[start:end]
            unique_dst, dst_inverse = torch.unique_consecutive(dst_b, return_inverse=True)
            unique_count = int(unique_dst.numel())
            if unique_count <= 0:
                continue

            def _make_block_forward(
                src_block: torch.Tensor,
                dst_block: torch.Tensor,
                edge_ids_block: torch.Tensor,
                dst_inverse_block: torch.Tensor,
                unique_count_block: int,
            ):
                edge_count_block = int(edge_ids_block.numel())

                def _block_forward(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, edge_attr_in: torch.Tensor) -> torch.Tensor:
                    q_i = q_in[:, dst_block]
                    k_j = k_in[:, src_block]
                    v_j = v_in[:, src_block]
                    logits = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
                    logits = logits + self._edge_level_attention_bias(node_level, src_block, dst_block, device).unsqueeze(0)
                    if has_edge_attr:
                        logits = self._add_edge_attr_to_chunk_logits(logits, edge_attr_in, edge_ids_block, B, num_edges, q_i)
                    idx_flat = (dst_inverse_block.view(1, -1) + (torch.arange(B, device=device, dtype=torch.long) * unique_count_block).view(B, 1)).reshape(-1)
                    weights = softmax(logits.reshape(B * edge_count_block, self.num_heads), idx_flat)
                    if dropout_p > 0.0:
                        weights = F.dropout(weights, p=dropout_p, training=True)
                    messages = v_j * weights.view(B, edge_count_block, self.num_heads).unsqueeze(-1)
                    out_flat = scatter_add(
                        messages.reshape(B * edge_count_block, self.num_heads, self.head_dim),
                        idx_flat,
                        dim=0,
                        dim_size=B * unique_count_block,
                    )
                    out_block = out_flat.view(B, unique_count_block, self.num_heads, self.head_dim).reshape(B, unique_count_block, self.hidden_dim)
                    return self.out_proj(out_block)

                return _block_forward

            block_forward = _make_block_forward(src_b, dst_b, edge_ids, dst_inverse, unique_count)

            if use_checkpoint:
                out_block = torch.utils.checkpoint.checkpoint(block_forward, q, k, v, edge_attr_arg, use_reentrant=False)
            else:
                out_block = block_forward(q, k, v, edge_attr_arg)
            out.index_copy_(1, unique_dst, out_block)
        return out

    def _update_graph_edge_trace(self, weights: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> None:
        if not self._trace_mode_allows_graph() or not self._trace_updates_allowed():
            return
        with torch.no_grad():
            credit = weights.detach() if bool(getattr(self, "lateral_edge_trace_detach", True)) else weights
            if credit.dim() == 3:
                credit = credit.mean(dim=0)
            if not bool(getattr(self, "lateral_edge_trace_per_head", True)):
                credit = credit.mean(dim=-1, keepdim=True)
            if bool(getattr(self, "lateral_edge_trace_center_per_dst", True)) and credit.numel() > 0:
                dst = dst.to(device=credit.device, dtype=torch.long)
                denom = scatter_add(torch.ones((dst.numel(), 1), device=credit.device, dtype=credit.dtype), dst, dim=0, dim_size=int(num_nodes)).clamp_min(1.0)
                mean = scatter_add(credit, dst, dim=0, dim_size=int(num_nodes)) / denom
                credit = credit - mean.index_select(0, dst)
            trace = self._ensure_graph_edge_trace(credit.size(0), credit)
            old = trace.clone() if bool(getattr(self, "lateral_edge_trace_debug", False)) else None
            trace.mul_(float(self.lateral_edge_trace_decay)).add_(credit, alpha=float(self.lateral_edge_trace_eta))
            trace.clamp_(min=-float(self.lateral_edge_trace_max), max=float(self.lateral_edge_trace_max))
            update_norm = (trace - old).norm() if old is not None else None
            self._record_lateral_trace_stats("graph", trace, credit, update_norm)

    def _compute_traced_local_attn_for_indices(
        self,
        q_lvl: torch.Tensor,
        k_lvl: torch.Tensor,
        v_lvl: torch.Tensor,
        window: int,
        causal: bool,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self._trace_mode_allows_window() or int(window) <= 0:
            return None
        B, T, Hh, Dh = q_lvl.shape
        if T <= 1:
            return None

        qh = q_lvl.transpose(1, 2)  # [B,H,T,D]
        kh = k_lvl.transpose(1, 2)
        vh = v_lvl.transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(float(max(1, Dh)))

        pos = torch.arange(T, device=q_lvl.device, dtype=torch.long)
        rel = pos.view(1, T) - pos.view(T, 1)  # key - query
        if bool(causal):
            allow = (rel <= 0) & (rel >= -int(window))
            rel_index = (-rel).clamp(min=0, max=int(window))
            num_offsets = int(window) + 1
        else:
            allow = rel.abs() <= int(window)
            rel_index = (rel + int(window)).clamp(min=0, max=2 * int(window))
            num_offsets = 2 * int(window) + 1

        if attn_bias is None:
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~allow.view(1, 1, T, T), neg)
        else:
            if attn_bias.dim() == 2:
                scores = scores + attn_bias.view(1, 1, T, T)
            elif attn_bias.dim() == 3:
                scores = scores + attn_bias.view(1, attn_bias.size(0), T, T)
            elif attn_bias.dim() == 4:
                scores = scores + attn_bias
            else:
                raise ValueError(f"Unsupported local attn bias shape: {tuple(attn_bias.shape)}")
            allow = allow & torch.isfinite(attn_bias.reshape(-1, T, T).amax(dim=0))

        trace = self._ensure_local_rel_trace(num_offsets, scores)
        trace_bias = trace if trace.size(0) == Hh else trace.expand(Hh, -1)
        if float(getattr(self, "lateral_edge_trace_alpha", 0.0)) != 0.0:
            scores = scores + float(self.lateral_edge_trace_alpha) * trace_bias[:, rel_index].view(1, Hh, T, T)

        weights = torch.softmax(scores, dim=-1)
        if self._trace_updates_allowed():
            with torch.no_grad():
                credit = weights.detach() if bool(getattr(self, "lateral_edge_trace_detach", True)) else weights
                credit = credit.mean(dim=0)
                valid = allow.reshape(-1)
                rel_flat = rel_index.reshape(-1)[valid]
                credit_flat = credit.reshape(Hh, -1)[:, valid].transpose(0, 1)
                delta = scatter_add(credit_flat, rel_flat, dim=0, dim_size=num_offsets)
                counts = scatter_add(torch.ones((rel_flat.numel(), 1), device=q_lvl.device, dtype=credit.dtype), rel_flat, dim=0, dim_size=num_offsets).clamp_min(1.0)
                delta = (delta / counts).transpose(0, 1)
                if not bool(getattr(self, "lateral_edge_trace_per_head", True)):
                    delta = delta.mean(dim=0, keepdim=True)
                if bool(getattr(self, "lateral_edge_trace_center_per_dst", True)):
                    delta = delta - delta.mean(dim=-1, keepdim=True)
                trace = self._ensure_local_rel_trace(num_offsets, delta)
                old = trace.clone() if bool(getattr(self, "lateral_edge_trace_debug", False)) else None
                trace.mul_(float(self.lateral_edge_trace_decay)).add_(delta, alpha=float(self.lateral_edge_trace_eta))
                trace.clamp_(min=-float(self.lateral_edge_trace_max), max=float(self.lateral_edge_trace_max))
                update_norm = (trace - old).norm() if old is not None else None
                self._record_lateral_trace_stats("local", trace, delta, update_norm)

        dropout_p = float(self.dropout.p) if self.training else 0.0
        if dropout_p > 0.0:
            weights = F.dropout(weights, p=dropout_p, training=True)
        out_h = torch.matmul(weights, vh)
        return out_h.transpose(1, 2).contiguous()

    
    
    @torch.no_grad()
    def _sync_per_level_qkv_from_shared(self):
        """Copy the shared backbone Q/K/V into every per-level intra-level projection.

        Called at init (so an enabled model starts identical to the shared-QKV model) and
        again after loading a checkpoint that predates per-level QKV (the shared weights
        are loaded first, then fanned out here). No-op when per-level QKV is disabled.
        """
        if not getattr(self, "per_level_local_qkv", False):
            return
        for shared, level_list in (
            (self.q_proj, self.q_proj_level),
            (self.k_proj, self.k_proj_level),
            (self.v_proj, self.v_proj_level),
        ):
            for proj in level_list:
                proj.weight.copy_(shared.weight)
                if proj.bias is not None and shared.bias is not None:
                    proj.bias.copy_(shared.bias)

    @torch.no_grad()
    def _sync_cross_level_qkv_from_shared(self):
        """Copy the shared backbone Q/K/V into every per-level CROSS-level projection, so an
        enabled model starts identical to shared-QKV (and a pre-cross-level-QKV checkpoint
        fans out after its shared weights load). No-op for the 'shared'/'reuse_local' modes."""
        if getattr(self, "cross_level_qkv", "shared") not in ("per_level_qk", "per_level_qkv"):
            return
        pairs = [(self.q_proj, self.q_proj_xlevel), (self.k_proj, self.k_proj_xlevel)]
        if self.cross_level_qkv == "per_level_qkv":
            pairs.append((self.v_proj, self.v_proj_xlevel))
        for shared, level_list in pairs:
            for proj in level_list:
                proj.weight.copy_(shared.weight)
                if proj.bias is not None and shared.bias is not None:
                    proj.bias.copy_(shared.bias)

    @staticmethod
    def _xlevel_load_post_hook(module, incompatible_keys):
        """If a loaded checkpoint omitted the per-level cross-level projections (pre-feature),
        re-fan them out from the shared q/k/v that DID load, so resume starts == shared."""
        if getattr(module, "cross_level_qkv", "shared") not in ("per_level_qk", "per_level_qkv"):
            return
        missing = list(getattr(incompatible_keys, "missing_keys", []) or [])
        if any(("q_proj_xlevel" in k or "k_proj_xlevel" in k or "v_proj_xlevel" in k) for k in missing):
            module._sync_cross_level_qkv_from_shared()
            for k in [m for m in missing if "_proj_xlevel" in m]:
                try:
                    incompatible_keys.missing_keys.remove(k)
                except (ValueError, AttributeError):
                    pass

    def _resolve_xlevel_lists(self):
        """Return (q_list, k_list, v_list) of per-level projection ModuleLists for the active
        cross_level_qkv mode, or Nones for the shared path. A None v_list => shared V."""
        mode = getattr(self, "cross_level_qkv", "shared")
        if mode == "per_level_qk":
            return self.q_proj_xlevel, self.k_proj_xlevel, None
        if mode == "per_level_qkv":
            return self.q_proj_xlevel, self.k_proj_xlevel, self.v_proj_xlevel
        if mode == "reuse_local" and getattr(self, "per_level_local_qkv", False):
            return self.q_proj_level, self.k_proj_level, self.v_proj_level
        return None, None, None

    def _xlevel_active(self):
        ql, kl, _ = self._resolve_xlevel_lists()
        return ql is not None and kl is not None

    def _apply_rope_bnhd(self, t_bnhd, rope_pos, B, num_nodes):
        """Apply the same RoPE the shared q/k path uses to a [B,N,H,head_dim] tensor."""
        if not (hasattr(self, "rotary_pos_enc") and rope_pos is not None):
            return t_bnhd
        flat = t_bnhd.reshape(B * num_nodes, self.num_heads, self.head_dim)
        if isinstance(rope_pos, torch.Tensor) and rope_pos.dim() == 2:
            pos_rep = rope_pos.view(1, num_nodes, rope_pos.size(-1)).expand(B, num_nodes, rope_pos.size(-1)).reshape(B * num_nodes, rope_pos.size(-1))
        else:
            pos_rep = rope_pos.view(1, num_nodes).expand(B, num_nodes).reshape(-1)
        flat = self.rotary_pos_enc.apply_rotary_pos_emb(flat, pos_rep)
        return flat.view(B, num_nodes, self.num_heads, self.head_dim)

    def _cross_level_qkv_route(self, x, B, num_nodes, node_level, level_offsets, rope_pos, q_shared, k_shared, v_shared):
        """Route each node through its LEVEL's cross-level projection, returning per-level
        qx/kx/vx [B,N,H,head_dim] (RoPE applied to qx/kx). Each node is projected once.

        Levels are CONTIGUOUS blocks in the node ordering, so we slice by level_offsets (one
        host sync for the offsets, then sync-free contiguous GEMMs). This avoids the per-level
        nonzero()+advanced-index gather/scatter, whose host syncs + non-contiguous copies
        otherwise dominate (turning a ~1-3% FLOP cost into a ~40% wall-clock hit)."""
        H, D = self.num_heads, self.head_dim
        qlist, klist, vlist = self._resolve_xlevel_lists()
        # Contiguous per-level [start, end) bounds (one sync); None => fall back to masked path.
        bounds = None
        if isinstance(level_offsets, torch.Tensor) and level_offsets.numel() >= 2:
            lo = level_offsets.detach().to("cpu").tolist()
            bounds = [(int(lo[i]), int(lo[i + 1])) for i in range(len(lo) - 1)]

        def route(level_list, shared_final, shared_proj, is_qk):
            if level_list is None:                 # shared tensor (RoPE'd for q/k, raw for v)
                return shared_final
            # Lazily allocate `out` with the projection-output dtype (autocast bf16) on first
            # write -- the shared tensor may have been skipped (None) so we can't read its dtype.
            out = None
            def _put(sl_lo, sl_hi, t):
                nonlocal out
                if out is None:
                    out = torch.empty(B, num_nodes, H, D, dtype=t.dtype, device=x.device)
                out[:, sl_lo:sl_hi] = t
            if bounds is not None:
                end = 0
                for L in range(len(level_list)):
                    if L >= len(bounds):
                        break
                    s, e = bounds[L]
                    if e > s:
                        _put(s, e, level_list[L](x[:, s:e]).view(B, e - s, H, D))
                    end = max(end, e)
                if end < num_nodes:                # levels beyond the per-level list => shared
                    _put(end, num_nodes, shared_proj(x[:, end:]).view(B, num_nodes - end, H, D))
            else:
                nl = node_level.to(device=x.device, dtype=torch.long)
                covered = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
                for L in range(len(level_list)):
                    idx = (nl == L).nonzero(as_tuple=False).view(-1)
                    if idx.numel() == 0:
                        continue
                    pj = level_list[L](x[:, idx]).view(B, idx.numel(), H, D)
                    if out is None:
                        out = torch.empty(B, num_nodes, H, D, dtype=pj.dtype, device=x.device)
                    out[:, idx] = pj
                    covered[idx] = True
                if not bool(covered.all()):
                    rest = (~covered).nonzero(as_tuple=False).view(-1)
                    pj = shared_proj(x[:, rest]).view(B, rest.numel(), H, D)
                    if out is None:
                        out = torch.empty(B, num_nodes, H, D, dtype=pj.dtype, device=x.device)
                    out[:, rest] = pj
            if out is None:
                out = torch.empty(B, num_nodes, H, D,
                                  dtype=(shared_final.dtype if shared_final is not None else x.dtype),
                                  device=x.device)
            return self._apply_rope_bnhd(out, rope_pos, B, num_nodes) if is_qk else out

        qx = route(qlist, q_shared, self.q_proj, True)
        kx = route(klist, k_shared, self.k_proj, True)
        vx = route(vlist, v_shared, self.v_proj, False)  # value carries no RoPE
        return qx, kx, vx

    def forward(self, x, edge_index, node_level, level_offsets=None, positions=None, edge_attr=None, hqd_edges=None, active_levels=None, input_norm=None, edge_type=None):
        self._last_attention_source_gates = None
        self._last_edge_condition_stats = None
        if x.dim() == 3:
            return self._forward_batched(x, edge_index, node_level, level_offsets=level_offsets, positions=positions, edge_attr=edge_attr, hqd_edges=hqd_edges, active_levels=active_levels, input_norm=input_norm, edge_type=edge_type)
        if input_norm is not None:
            x = input_norm(x)

        num_nodes = x.size(0)
        device = x.device
        source_gates = self._compute_source_gates(x)
        if edge_type is not None:
            edge_type = edge_type.to(device=device, dtype=torch.long)
            if edge_type.numel() != int(edge_index.size(1)):
                edge_type = None

        level_emb = self.level_embedding(node_level)

        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        

        pos_to_use = None
        # --- Determine positions for RoPE ---
        if positions is not None:
             # Use directly provided positions (e.g., from _process_level)
             pos_to_use = positions
        elif level_offsets is not None:
             # Calculate intra-level positions from offsets (e.g., from _apply_refinement_cycles)
             max_level = node_level.max().item()
             # Basic check for valid offsets
             if isinstance(level_offsets, (list, tuple)) and len(level_offsets) > max_level + 1:
                 try:
                     # Ensure level_offsets is a tensor for indexing
                     if not isinstance(level_offsets, torch.Tensor):
                         level_offsets_tensor = torch.tensor(level_offsets, device=device)
                     else:
                         level_offsets_tensor = level_offsets

                     start_offsets = level_offsets_tensor[node_level]
                     intra_level_pos = torch.arange(num_nodes, device=device) - start_offsets
                     pos_to_use = intra_level_pos
                 except Exception as e:
                     print(f"Warning: Failed to compute intra-level positions from offsets: {e}.")
             else:
                 print(f"Warning: level_offsets not suitable for RoPE calculation (len: {len(level_offsets) if level_offsets is not None else 'None'}, max_level: {max_level}).")
        # --- End position determination ---

        rope_pos = self._build_rope_positions(pos_to_use, node_level=node_level)
        # --- Apply RoPE if positions were determined ---
        if hasattr(self, 'rotary_pos_enc') and rope_pos is not None:
            try:
                self.rotary_pos_enc.local_attn_runtime_level_grid_shapes = dict(getattr(self, "local_attn_runtime_level_grid_shapes", {}))
                q = self.rotary_pos_enc.apply_rotary_pos_emb(q, rope_pos)
                k = self.rotary_pos_enc.apply_rotary_pos_emb(k, rope_pos)
            except Exception as e:
                print(f"Warning: Failed to apply RoPE: {e}. Skipping RoPE application.")
        # --- End RoPE ---
        # --- choose attention implementation ---
        #     out = self.kernel_linear_attention(
        #         q=q, k=k, v=v,
        #         phi_q =phi_q,
        #         phi_k =phi_k,
        #         edge_index=edge_index,
        #         level_emb=level_emb,
        #         edge_attr=edge_attr if self.use_edge_attr else None,
        #         num_nodes=num_nodes,
        #     )
        # Sparse softmax attention via MessagePassing
        self._edge_attr_stack.append(None)
        try:
            out = self.propagate(
                edge_index,
                q=q, k=k, v=v,
                level_emb=level_emb,
                edge_attr=edge_attr if self.use_edge_attr else None,
                edge_type=edge_type,
                node_hidden=x,
                size=None,
            )
        finally:
            new_edge_attr = self._edge_attr_stack.pop()
        
        # ----------------------------------------------------------
        # out = self.propagate(
        #     edge_index,
        #     q=q, k=k, v=v,
        #     #phi_q=phi_q,
        #     #phi_k=phi_k,
        #     level_emb=level_emb,
        #     edge_attr=edge_attr if self.use_edge_attr else None,
        #     size=None
        # )

        out = out.view(-1, self.hidden_dim)
        out = self.out_proj(out)
        if source_gates is not None:
            out = source_gates["graph"] * out

        # The NeighborLoader sampled path calls this 2D forward with partial
        # subgraphs.  Keep its local-attention behavior aligned with the native
        # true-batch path by reusing the same local attention helpers with a
        # singleton batch dimension.  `local_attn_runtime_group` prevents local
        # attention from mixing nodes that came from different original samples.
        _use_multi = bool(getattr(self, "local_attn_runtime_enable", False)) and bool(self.local_attn_config)
        _use_legacy_l0 = bool(getattr(self, "l0_local_runtime_enable", False)) and self.l0_local_backend != "pyg"
        if _use_multi:
            global_causal_gate = bool(getattr(self, "local_attn_runtime_causal_gate", True))
            runtime_group = getattr(self, "local_attn_runtime_group", None)
            q_b = q.unsqueeze(0)
            k_b = k.unsqueeze(0)
            v_b = v.unsqueeze(0)
            for lvl, cfg in self.local_attn_config.items():
                lvl_int = int(lvl)
                cfg_causal = bool(cfg.get("causal", False))
                effective_causal = cfg_causal and global_causal_gate
                if lvl_int == 0:
                    effective_causal = effective_causal and bool(
                        getattr(self, "l0_local_runtime_causal", self.l0_local_causal_default)
                    )
                lvl_result = self._compute_level_local_out_batched(
                    q=q_b,
                    k=k_b,
                    v=v_b,
                    node_level=node_level,
                    level=lvl_int,
                    window=int(cfg.get("window", 0)),
                    causal=effective_causal,
                    backend=str(cfg.get("backend", "sdpa")),
                    node_group=runtime_group,
                    positions=pos_to_use,
                )
                if lvl_result is not None:
                    lvl_idx, lvl_proj = lvl_result
                    lvl_proj_2d = lvl_proj.squeeze(0)
                    if source_gates is not None:
                        lvl_proj_2d = source_gates["local"] * lvl_proj_2d
                    out[lvl_idx, :] = out[lvl_idx, :] + lvl_proj_2d
        elif _use_legacy_l0:
            q_b = q.unsqueeze(0)
            k_b = k.unsqueeze(0)
            v_b = v.unsqueeze(0)
            l0_local = self._compute_l0_local_out_batched(q=q_b, k=k_b, v=v_b, node_level=node_level)
            if l0_local is not None:
                l0_idx, l0_proj = l0_local
                l0_proj_2d = l0_proj.squeeze(0)
                if source_gates is not None:
                    l0_proj_2d = source_gates["local"] * l0_proj_2d
                out[l0_idx, :] = out[l0_idx, :] + l0_proj_2d

        return out, new_edge_attr   

    def _forward_batched(self, x, edge_index, node_level, level_offsets=None, positions=None, edge_attr=None, hqd_edges=None, active_levels=None, input_norm=None, edge_type=None):
        """
        Batched sparse message passing with shared topology.

        Args:
            x: Node features [B, N, hidden_dim]
            edge_index: Shared edge indices [2, E]
            node_level: Shared node levels [N]
            positions: Optional shared positions [N] (or [B, N], first row used)
            edge_attr: Optional edge attrs [E,*] or [B,E,*]
            hqd_edges: Optional tuple (b_idx, src_idx, dst_idx) for HQD sparse attention.

        Returns:
            out: [B, N, hidden_dim]
            new_edge_attr: [B, E] or None
        """
        B, num_nodes, _ = x.shape
        device = x.device
        source_gates = self._compute_source_gates(x)
        self._last_graph_witness_ids = None
        self._last_graph_witness_scores = None


        active_level_set = None if active_levels is None else {int(level) for level in active_levels}
        src, dst = edge_index
        src = src.to(device=device, dtype=torch.long)
        dst = dst.to(device=device, dtype=torch.long)
        if edge_type is not None:
            edge_type = edge_type.to(device=device, dtype=torch.long)
        if active_level_set is not None:
            keep = torch.zeros_like(dst, dtype=torch.bool)
            node_level_device = node_level.to(device=device, dtype=torch.long)
            dst_levels = node_level_device.index_select(0, dst)
            for level in active_level_set:
                keep = keep | (dst_levels == int(level))
            if not bool(keep.any()):
                return x.new_zeros(x.shape), None
            src = src[keep]
            dst = dst[keep]
            if edge_attr is not None:
                if edge_attr.dim() == 1 and edge_attr.size(0) == keep.numel():
                    edge_attr = edge_attr[keep]
                elif edge_attr.dim() == 2:
                    if edge_attr.size(0) == keep.numel():
                        edge_attr = edge_attr[keep]
                    elif edge_attr.size(1) == keep.numel():
                        edge_attr = edge_attr[:, keep]
                elif edge_attr.dim() == 3 and edge_attr.size(1) == keep.numel():
                    edge_attr = edge_attr[:, keep, :]
            if edge_type is not None and edge_type.numel() == keep.numel():
                edge_type = edge_type.to(device=device, dtype=torch.long)[keep]
        num_edges = src.numel()
        if edge_type is not None and edge_type.numel() != num_edges:
            edge_type = None

        pos_to_use = None
        if positions is not None:
            if isinstance(positions, torch.Tensor):
                if positions.dim() == 2 and positions.size(0) == num_nodes:
                    pos_to_use = positions.to(device)
                elif positions.dim() >= 3:
                    pos_to_use = positions.to(device)
                elif positions.dim() == 2 and positions.size(0) == B:
                    pos_to_use = positions[0].to(device)
                else:
                    pos_to_use = positions.to(device)
            else:
                pos_to_use = torch.as_tensor(positions, device=device)
        elif level_offsets is not None:
            max_level = node_level.max().item()
            if isinstance(level_offsets, (list, tuple, torch.Tensor)) and len(level_offsets) > max_level + 1:
                if not isinstance(level_offsets, torch.Tensor):
                    level_offsets_tensor = torch.tensor(level_offsets, device=device)
                else:
                    level_offsets_tensor = level_offsets.to(device)
                start_offsets = level_offsets_tensor[node_level]
                pos_to_use = torch.arange(num_nodes, device=device) - start_offsets

        rope_pos = self._build_rope_positions(pos_to_use, node_level=node_level)

        if active_level_set is not None:
            def _select_normed(nodes: torch.Tensor) -> torch.Tensor:
                selected = x.index_select(1, nodes)
                return input_norm(selected) if input_norm is not None else selected

            dst_nodes, dst_inverse = torch.unique(dst, sorted=True, return_inverse=True)
            src_nodes, src_inverse = torch.unique(src, sorted=True, return_inverse=True)
            if dst_nodes.numel() == 0 or src_nodes.numel() == 0:
                return x.new_zeros(x.shape), None

            dst_normed = _select_normed(dst_nodes)
            q = self.q_proj(dst_normed).view(B, dst_nodes.numel(), self.num_heads, self.head_dim)
            src_normed = _select_normed(src_nodes)
            k = self.k_proj(src_normed).view(B, src_nodes.numel(), self.num_heads, self.head_dim)
            v = self.v_proj(src_normed).view(B, src_nodes.numel(), self.num_heads, self.head_dim)

            if hasattr(self, 'rotary_pos_enc') and rope_pos is not None:
                self.rotary_pos_enc.local_attn_runtime_level_grid_shapes = dict(getattr(self, "local_attn_runtime_level_grid_shapes", {}))
                if isinstance(rope_pos, torch.Tensor) and rope_pos.dim() == 2:
                    q_pos = rope_pos.index_select(0, dst_nodes).view(1, -1, rope_pos.size(-1)).expand(B, -1, -1).reshape(B * dst_nodes.numel(), rope_pos.size(-1))
                    k_pos = rope_pos.index_select(0, src_nodes).view(1, -1, rope_pos.size(-1)).expand(B, -1, -1).reshape(B * src_nodes.numel(), rope_pos.size(-1))
                else:
                    q_pos = rope_pos.index_select(0, dst_nodes).view(1, -1).expand(B, -1).reshape(-1)
                    k_pos = rope_pos.index_select(0, src_nodes).view(1, -1).expand(B, -1).reshape(-1)
                q_flat = q.reshape(B * dst_nodes.numel(), self.num_heads, self.head_dim)
                k_flat = k.reshape(B * src_nodes.numel(), self.num_heads, self.head_dim)
                q = self.rotary_pos_enc.apply_rotary_pos_emb(q_flat, q_pos).view(B, dst_nodes.numel(), self.num_heads, self.head_dim)
                k = self.rotary_pos_enc.apply_rotary_pos_emb(k_flat, k_pos).view(B, src_nodes.numel(), self.num_heads, self.head_dim)

            q_i = q[:, dst_inverse]
            k_j = k[:, src_inverse]
            v_j = v[:, src_inverse]

            attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
            level_bias = self._edge_level_attention_bias(node_level, src, dst, device)
            attn = attn + level_bias.unsqueeze(0)

            if edge_attr is not None and self.use_edge_attr:
                if edge_attr.dim() == 1:
                    edge_attr_b = edge_attr.view(1, num_edges, 1).expand(B, -1, -1)
                elif edge_attr.dim() == 2:
                    if edge_attr.size(0) == B and edge_attr.size(1) == num_edges:
                        edge_attr_b = edge_attr.unsqueeze(-1)
                    elif edge_attr.size(0) == num_edges:
                        edge_attr_b = edge_attr.unsqueeze(0).expand(B, -1, -1)
                    else:
                        raise ValueError(f"Unsupported batched edge_attr shape: {tuple(edge_attr.shape)}")
                elif edge_attr.dim() == 3 and edge_attr.size(0) == B and edge_attr.size(1) == num_edges:
                    edge_attr_b = edge_attr
                else:
                    raise ValueError(f"Unsupported batched edge_attr shape: {tuple(edge_attr.shape)}")

                feat_dim = edge_attr_b.size(-1)
                if feat_dim == 1:
                    attn = attn + edge_attr_b.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads)
                elif self.edge_dim is not None and feat_dim == self.edge_dim:
                    edge_attr_flat = edge_attr_b.reshape(B * num_edges, feat_dim)
                    edge_features = self.edge_proj(edge_attr_flat).view(B, num_edges, self.num_heads, self.head_dim)
                    attn = attn + (q_i * edge_features).sum(dim=-1) / math.sqrt(self.head_dim)
                else:
                    raise ValueError(f"Unsupported edge_attr feature dim {feat_dim}; expected 1 or edge_dim={self.edge_dim}")

            if self._edge_conditioning_active():
                src_hidden_e = src_normed[:, src_inverse] if self.edge_node_condition_enable else None
                dst_hidden_e = dst_normed[:, dst_inverse] if self.edge_node_condition_enable else None
                edge_logit_bias, edge_value_gate = self._edge_condition(
                    edge_type,
                    attn.reshape(B * num_edges, self.num_heads),
                    src_hidden=src_hidden_e,
                    dst_hidden=dst_hidden_e,
                )
            else:
                edge_logit_bias, edge_value_gate = None, None
            attn = self._apply_edge_logit_bias(attn, edge_logit_bias)
            batch_offsets = (torch.arange(B, device=device, dtype=torch.long) * dst_nodes.numel()).view(B, 1)
            index_flat = (dst_inverse.view(1, -1) + batch_offsets).reshape(-1)
            if active_level_set is None:
                attn = self._apply_graph_trace_bias(attn, num_edges)
            attn_flat = softmax(attn.reshape(B * num_edges, self.num_heads), index_flat)
            self._capture_graph_witnesses(attn_flat.view(B, num_edges, self.num_heads), src, dst, node_level, int(num_nodes))
            if active_level_set is None:
                self._update_graph_edge_trace(attn_flat.view(B, num_edges, self.num_heads), dst, int(num_nodes))
            attn_flat = self.dropout(attn_flat)
            v_j = self._apply_edge_value_gate(v_j, edge_value_gate)
            messages = v_j * attn_flat.view(B, num_edges, self.num_heads).unsqueeze(-1)
            messages_flat = messages.reshape(B * num_edges, self.num_heads, self.head_dim)
            out_flat = scatter_add(messages_flat, index_flat, dim=0, dim_size=B * dst_nodes.numel())
            out_compact = out_flat.view(B, dst_nodes.numel(), self.num_heads, self.head_dim).reshape(B, dst_nodes.numel(), self.hidden_dim)
            out_compact = self.out_proj(out_compact)
            if source_gates is not None:
                out_compact = source_gates["graph"] * out_compact
            out = out_compact.new_zeros(B, num_nodes, self.hidden_dim)
            out.index_copy_(1, dst_nodes, out_compact)

            _use_multi = bool(getattr(self, "local_attn_runtime_enable", False)) and bool(self.local_attn_config)
            _use_legacy_l0 = bool(getattr(self, "l0_local_runtime_enable", False)) and self.l0_local_backend != "pyg"
            if _use_multi or _use_legacy_l0:
                global_causal_gate = bool(getattr(self, "local_attn_runtime_causal_gate", True))
                runtime_group = getattr(self, "local_attn_runtime_group", None)
                local_cfg_items = self.local_attn_config.items() if _use_multi else [(0, {"window": self.l0_local_window, "causal": getattr(self, "l0_local_runtime_causal", self.l0_local_causal_default), "backend": self.l0_local_backend})]
                for lvl, cfg in local_cfg_items:
                    lvl_int = int(lvl)
                    if lvl_int not in active_level_set:
                        continue
                    lvl_idx = torch.nonzero(node_level.to(device=device, dtype=torch.long) == lvl_int, as_tuple=False).view(-1)
                    if lvl_idx.numel() <= 1:
                        continue
                    cfg_causal = bool(cfg.get("causal", False))
                    effective_causal = cfg_causal and global_causal_gate
                    if lvl_int == 0:
                        effective_causal = effective_causal and bool(getattr(self, "l0_local_runtime_causal", self.l0_local_causal_default))
                    lvl_normed = _select_normed(lvl_idx)
                    # Per-level intra-level Q/K/V (falls back to the shared backbone QKV
                    # when per-level is disabled or the level is out of range).
                    if self.per_level_local_qkv and 0 <= lvl_int < self.num_local_levels:
                        q_proj_l, k_proj_l, v_proj_l = (
                            self.q_proj_level[lvl_int], self.k_proj_level[lvl_int], self.v_proj_level[lvl_int])
                    else:
                        q_proj_l, k_proj_l, v_proj_l = self.q_proj, self.k_proj, self.v_proj
                    q_lvl = q_proj_l(lvl_normed).view(B, lvl_idx.numel(), self.num_heads, self.head_dim)
                    k_lvl = k_proj_l(lvl_normed).view(B, lvl_idx.numel(), self.num_heads, self.head_dim)
                    v_lvl = v_proj_l(lvl_normed).view(B, lvl_idx.numel(), self.num_heads, self.head_dim)
                    if hasattr(self, 'rotary_pos_enc') and rope_pos is not None:
                        if isinstance(rope_pos, torch.Tensor) and rope_pos.dim() == 2:
                            lvl_pos = rope_pos.index_select(0, lvl_idx).view(1, -1, rope_pos.size(-1)).expand(B, -1, -1).reshape(B * lvl_idx.numel(), rope_pos.size(-1))
                        else:
                            lvl_pos = rope_pos.index_select(0, lvl_idx).view(1, -1).expand(B, -1).reshape(-1)
                        q_lvl = self.rotary_pos_enc.apply_rotary_pos_emb(q_lvl.reshape(B * lvl_idx.numel(), self.num_heads, self.head_dim), lvl_pos).view(B, lvl_idx.numel(), self.num_heads, self.head_dim)
                        k_lvl = self.rotary_pos_enc.apply_rotary_pos_emb(k_lvl.reshape(B * lvl_idx.numel(), self.num_heads, self.head_dim), lvl_pos).view(B, lvl_idx.numel(), self.num_heads, self.head_dim)
                    level_grid_shape = self._resolve_level_grid_shape(lvl_int)
                    spatial_metric = str(getattr(self, "local_attn_runtime_spatial_metric", "chebyshev")).lower()
                    role_bias_enabled = bool(getattr(self, "local_attn_level_role_bias_enable", True))
                    use_dense_local_bias = int(cfg.get("window", 0)) > 0 and (level_grid_shape is not None or role_bias_enabled)
                    if str(cfg.get("backend", "sdpa")) == "flash" and use_dense_local_bias:
                        raise RuntimeError(
                            f"Flash local attention for level {lvl_int} cannot use spatial/role-bias masks because they require dense T x T bias tensors. "
                            "Disable local attention level role bias/spatial local masks, or use --l0_local_backend pyg."
                        )
                    local_idx = torch.arange(lvl_idx.numel(), device=device, dtype=torch.long)
                    if runtime_group is None:
                        attn_bias = None
                        force_sdpa = False
                        if use_dense_local_bias:
                            if pos_to_use is not None and isinstance(pos_to_use, torch.Tensor) and pos_to_use.size(0) == num_nodes:
                                local_pos = pos_to_use.to(device=device).index_select(0, lvl_idx)
                            else:
                                local_pos = torch.arange(lvl_idx.numel(), device=device, dtype=torch.long)
                            bias_grid_shape = level_grid_shape if (level_grid_shape is not None and int(level_grid_shape[0]) * int(level_grid_shape[1]) == int(lvl_idx.numel())) else None
                            attn_bias = self._build_level_aware_local_attn_bias(
                                local_pos=local_pos,
                                grid_shape=bias_grid_shape,
                                window=int(cfg.get("window", 0)),
                                causal=effective_causal,
                                metric=spatial_metric,
                                level=lvl_int,
                                device=q_lvl.device,
                                dtype=q_lvl.dtype,
                            )
                            force_sdpa = True
                        local_out = self._compute_local_attn_for_indices(
                            q=q_lvl,
                            k=k_lvl,
                            v=v_lvl,
                            idx=local_idx,
                            window=int(cfg.get("window", 0)),
                            causal=effective_causal,
                            backend=str(cfg.get("backend", "sdpa")),
                            level=lvl_int,
                            attn_bias=attn_bias,
                            force_sdpa=force_sdpa,
                        )
                        if local_out is not None:
                            if source_gates is not None:
                                local_out = source_gates["local"] * local_out
                            out[:, lvl_idx, :] = out[:, lvl_idx, :] + local_out
                    else:
                        # Grouped local attention is uncommon for text; preserve correctness by
                        # using compact projections per active level but group splits locally.
                        group_vec = runtime_group.to(device=device, dtype=torch.long)
                        lvl_groups = group_vec.index_select(0, lvl_idx)
                        for gid in torch.unique(lvl_groups):
                            mask = lvl_groups == gid
                            grp_pos = torch.nonzero(mask, as_tuple=False).view(-1)
                            if grp_pos.numel() <= 1:
                                continue
                            attn_bias = None
                            force_sdpa = False
                            if use_dense_local_bias:
                                grp_idx = lvl_idx.index_select(0, grp_pos)
                                if pos_to_use is not None and isinstance(pos_to_use, torch.Tensor) and pos_to_use.size(0) == num_nodes:
                                    local_pos = pos_to_use.to(device=device).index_select(0, grp_idx)
                                else:
                                    local_pos = torch.arange(grp_pos.numel(), device=device, dtype=torch.long)
                                bias_grid_shape = level_grid_shape if (level_grid_shape is not None and int(level_grid_shape[0]) * int(level_grid_shape[1]) == int(grp_pos.numel())) else None
                                attn_bias = self._build_level_aware_local_attn_bias(
                                    local_pos=local_pos,
                                    grid_shape=bias_grid_shape,
                                    window=int(cfg.get("window", 0)),
                                    causal=effective_causal,
                                    metric=spatial_metric,
                                    level=lvl_int,
                                    device=q_lvl.device,
                                    dtype=q_lvl.dtype,
                                )
                                force_sdpa = True
                            local_out = self._compute_local_attn_for_indices(
                                q=q_lvl,
                                k=k_lvl,
                                v=v_lvl,
                                idx=grp_pos,
                                window=int(cfg.get("window", 0)),
                                causal=effective_causal,
                                backend=str(cfg.get("backend", "sdpa")),
                                level=lvl_int,
                                attn_bias=attn_bias,
                                force_sdpa=force_sdpa,
                            )
                            if local_out is not None:
                                if source_gates is not None:
                                    local_out = source_gates["local"] * local_out
                                out[:, lvl_idx.index_select(0, grp_pos), :] = out[:, lvl_idx.index_select(0, grp_pos), :] + local_out

            return out, None

        if input_norm is not None:
            x = input_norm(x)
        # Shared q/k/v feed the local-window path, HQD (sparse + packed-witness), fat-V, and the
        # shared-mode cross path / per-level route fallback. HQD edge decisions happen later in
        # this forward, so we cannot safely predict at this point whether they're needed -> always
        # compute them. (Per-level cross_level_qkv adds qx/kx/vx on top; that extra is inherent.)
        q = self.q_proj(x).view(B, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, num_nodes, self.num_heads, self.head_dim)

        # Packed cross-level local attention (model-injected spec; consumed in the local
        # block below by _apply_local_pack_out). Keep PRE-RoPE q/k references: the packed
        # mixed-level sequence is RoPE'd at token-scale close positions, while the standard
        # rotation below places coarse nodes at level-local indices (wrong scale for a mixed
        # window). Shape-guarded: subset calls (multirate upper cycling etc.) see a stale-
        # sized spec and fall back to the per-level path untouched.
        _pack_spec = getattr(self, "_local_pack_spec", None)
        if _pack_spec is not None and int(_pack_spec.get("num_nodes", -1)) != int(num_nodes):
            _pack_spec = None
        q_prerope = k_prerope = None
        if _pack_spec is not None or bool(getattr(self, "hqd_read_prerope", False)):
            q_prerope, k_prerope = q, k

        pos_rep = None
        if hasattr(self, 'rotary_pos_enc') and rope_pos is not None:
            self.rotary_pos_enc.local_attn_runtime_level_grid_shapes = dict(getattr(self, "local_attn_runtime_level_grid_shapes", {}))
            if isinstance(rope_pos, torch.Tensor) and rope_pos.dim() == 2:
                pos_rep = rope_pos.view(1, num_nodes, rope_pos.size(-1)).expand(B, num_nodes, rope_pos.size(-1)).reshape(B * num_nodes, rope_pos.size(-1))
            else:
                pos_rep = rope_pos.view(1, num_nodes).expand(B, num_nodes).reshape(-1)
            q = self.rotary_pos_enc.apply_rotary_pos_emb(q.reshape(B * num_nodes, self.num_heads, self.head_dim), pos_rep).view(B, num_nodes, self.num_heads, self.head_dim)
            k = self.rotary_pos_enc.apply_rotary_pos_emb(k.reshape(B * num_nodes, self.num_heads, self.head_dim), pos_rep).view(B, num_nodes, self.num_heads, self.head_dim)

        # The above narrow (head_dim) q/k/v feed BOTH the cross-level scatter aggregation
        # and the flash/sdpa local-window path below, which require qk==v==head_dim. Under
        # fat-QKV the cross-level scatter instead consumes separate WIDE q/k/v (computed in
        # the explicit block below); the chunked fast path can't carry decoupled dims, so we
        # bypass it and fall through to the explicit aggregation.
        #
        # Per-level cross-level QKV (cross_level_qkv != "shared"): route each node through its
        # LEVEL's projection so the CROSS-level paths (chunked / packed / scatter) and the HQD
        # selector see qx/kx/vx, while the local-window path keeps the shared q/k/v above.
        # Disabled under fat-QKV (the wide path owns the cross-level q/k/v). qx/kx/vx == q/k/v
        # when the mode is "shared", so this is bit-identical by default.
        if (getattr(self, "cross_level_qkv", "shared") != "shared"
                and not getattr(self, "sparse_wide_enable", False) and self._xlevel_active()):
            qx, kx, vx = self._cross_level_qkv_route(x, B, num_nodes, node_level, level_offsets, rope_pos, q, k, v)
        else:
            qx, kx, vx = q, k, v
        out = None if getattr(self, "sparse_wide_enable", False) else self._sparse_graph_attention_chunked_batched(
            q=qx,
            k=kx,
            v=vx,
            src=src,
            dst=dst,
            node_level=node_level,
            edge_attr=edge_attr,
            num_nodes=int(num_nodes),
            num_edges=int(num_edges),
        )
        new_edge_attr = None
        # Tier-3 packed cross-level fast lane (additive, opt-in, scatter stays the default
        # fallback). Eligible only for the basic backbone case (no edge features/conditioning/
        # learned edges/trace/witness/fat-QKV); consumes the cross-level qx/kx/vx (== q/k/v
        # for shared QKV, per-level otherwise) and is mathematically identical to scatter.
        if out is None and not getattr(self, "sparse_wide_enable", False) and self._cross_level_packed_eligible(edge_attr):
            _packed = self._cross_level_packed_attention(
                qx, kx, vx, src, dst, node_level, int(num_nodes), B, self.head_dim, self.head_dim)
            if _packed is not None:
                out = self.sparse_out_proj(_packed.reshape(B, int(num_nodes), self.num_heads * self.head_dim))
        if out is None:
            # Fat-QKV: the cross-level scatter aggregation uses its own wide q/k/v so the
            # narrow tensors above remain intact for the flash/sdpa local path. qk_hd/v_hd
            # equal head_dim when the wide flags are off (qg/kg/vg are then just q/k/v).
            qk_hd = self.sparse_qk_head_dim
            v_hd = self.sparse_v_head_dim
            if getattr(self, "sparse_wide_enable", False):
                qg = self.sparse_proj_q(x).view(B, num_nodes, self.num_heads, qk_hd)
                kg = self.sparse_proj_k(x).view(B, num_nodes, self.num_heads, qk_hd)
                vg = self.sparse_proj_v(x).view(B, num_nodes, self.num_heads, v_hd)
                if hasattr(self, 'rotary_pos_enc') and rope_pos is not None:
                    if getattr(self, "sparse_qk_wide", False):
                        self.rotary_pos_enc_w.local_attn_runtime_level_grid_shapes = dict(
                            getattr(self, "local_attn_runtime_level_grid_shapes", {}))
                    qg_flat = self.sparse_apply_rope(qg.reshape(B * num_nodes, self.num_heads, qk_hd), pos_rep)
                    kg_flat = self.sparse_apply_rope(kg.reshape(B * num_nodes, self.num_heads, qk_hd), pos_rep)
                    qg = qg_flat.view(B, num_nodes, self.num_heads, qk_hd)
                    kg = kg_flat.view(B, num_nodes, self.num_heads, qk_hd)
            else:
                qg, kg, vg = qx, kx, vx
            q_i = qg[:, dst]  # [B, E, H, Dqk]
            k_j = kg[:, src]  # [B, E, H, Dqk]
            v_j = vg[:, src]  # [B, E, H, Dv]

            attn = (q_i * k_j).sum(dim=-1) / math.sqrt(qk_hd)  # [B, E, H]

            level_bias = self._edge_level_attention_bias(node_level, src, dst, device)
            attn = attn + level_bias.unsqueeze(0)

            if edge_attr is not None and self.use_edge_attr:
                if edge_attr.dim() == 1:
                    edge_attr_b = edge_attr.view(1, num_edges, 1).expand(B, -1, -1)
                elif edge_attr.dim() == 2:
                    if edge_attr.size(0) == B and edge_attr.size(1) == num_edges:
                        edge_attr_b = edge_attr.unsqueeze(-1)
                    elif edge_attr.size(0) == num_edges:
                        edge_attr_b = edge_attr.unsqueeze(0).expand(B, -1, -1)
                    else:
                        raise ValueError(f"Unsupported batched edge_attr shape: {tuple(edge_attr.shape)}")
                elif edge_attr.dim() == 3:
                    if edge_attr.size(0) == B and edge_attr.size(1) == num_edges:
                        edge_attr_b = edge_attr
                    else:
                        raise ValueError(f"Unsupported batched edge_attr shape: {tuple(edge_attr.shape)}")
                else:
                    raise ValueError(f"Unsupported batched edge_attr ndim: {edge_attr.dim()}")

                feat_dim = edge_attr_b.size(-1)
                if feat_dim == 1:
                    edge_attn = edge_attr_b.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads)
                    attn = attn + edge_attn
                elif self.edge_dim is not None and feat_dim == self.edge_dim:
                    edge_attr_flat = edge_attr_b.reshape(B * num_edges, feat_dim)
                    # Under wide qk, project the edge feature to the wide qk head dim so it
                    # matches q_i (edge_proj_w is created alongside the wide projections).
                    e_proj = self.edge_proj_w if getattr(self, "sparse_qk_wide", False) else self.edge_proj
                    edge_features = e_proj(edge_attr_flat).view(B, num_edges, self.num_heads, qk_hd)
                    edge_attn = (q_i * edge_features).sum(dim=-1) / math.sqrt(qk_hd)
                    attn = attn + edge_attn
                else:
                    raise ValueError(
                        f"Unsupported edge_attr feature dim {feat_dim}; expected 1 or edge_dim={self.edge_dim}"
                    )

            if self._edge_conditioning_active():
                src_hidden_e = x[:, src] if self.edge_node_condition_enable else None
                dst_hidden_e = x[:, dst] if self.edge_node_condition_enable else None
                edge_logit_bias, edge_value_gate = self._edge_condition(
                    edge_type,
                    attn.reshape(B * num_edges, self.num_heads),
                    src_hidden=src_hidden_e,
                    dst_hidden=dst_hidden_e,
                )
            else:
                edge_logit_bias, edge_value_gate = None, None
            # A per-channel value gate is sized to head_dim and can't broadcast over a wide
            # v_head_dim; drop it under fat-V (the per-head logit bias above is dim-agnostic).
            if getattr(self, "sparse_v_wide", False):
                edge_value_gate = None
            attn = self._apply_edge_logit_bias(attn, edge_logit_bias)
            index_flat = self._batched_dst_index_flat(dst, B, num_nodes)  # [B*E]

            attn = self._apply_graph_trace_bias(attn, num_edges)
            attn_flat = attn.reshape(B * num_edges, self.num_heads)
            attn_flat = softmax(attn_flat, index_flat)
            self._capture_graph_witnesses(attn_flat.view(B, num_edges, self.num_heads), src, dst, node_level, int(num_nodes))
            self._update_graph_edge_trace(attn_flat.view(B, num_edges, self.num_heads), dst, int(num_nodes))

            if self.learn_edge_from_attn:
                conf = attn_flat.max(dim=-1, keepdim=True).values
                effective = attn_flat * conf
                new_edge_attr = self.edge_combine(effective).squeeze(-1)
                new_edge_attr = torch.tanh(new_edge_attr).view(B, num_edges)
                if active_level_set is not None:
                    # Active-level compute uses a filtered edge set; keep the caller's
                    # full edge_attr tensor intact for later steps with different levels.
                    new_edge_attr = None

            attn_flat = self.dropout(attn_flat)
            attn = attn_flat.view(B, num_edges, self.num_heads)

            v_j = self._apply_edge_value_gate(v_j, edge_value_gate)
            messages = v_j * attn.unsqueeze(-1)  # [B,E,H,Dv]
            messages_flat = messages.reshape(B * num_edges, self.num_heads, v_hd)
            out_flat = scatter_add(messages_flat, index_flat, dim=0, dim_size=B * num_nodes)

            out = out_flat.view(B, num_nodes, self.num_heads, v_hd)
            out = out.reshape(B, num_nodes, self.num_heads * v_hd)
            out = self.sparse_out_proj(out)
        if source_gates is not None:
            out = source_gates["graph"] * out

        # Multi-level local window attention (backward-compat: l0_local_runtime_enable still works)
        _use_legacy_l0 = bool(getattr(self, "l0_local_runtime_enable", False)) and self.l0_local_backend != "pyg"
        _use_multi = bool(getattr(self, "local_attn_runtime_enable", False)) and bool(self.local_attn_config)
        if _use_multi:
            global_causal_gate = bool(getattr(self, "local_attn_runtime_causal_gate", True))
            runtime_group = getattr(self, "local_attn_runtime_group", None)
            # Packed cross-level local attention: one causal mixed-level windowed call
            # REPLACES the per-level local call for the spec's query levels (others keep
            # their lateral windows). Empty set = ineligible here, per-level path runs.
            _pack_covered: set = set()
            if _pack_spec is not None:
                _pack_covered = self._apply_local_pack_out(
                    out=out, q_pre=q_prerope, k_pre=k_prerope, v=v,
                    spec=_pack_spec, source_gates=source_gates,
                    active_level_set=active_level_set, B=B, num_nodes=num_nodes,
                )
            for lvl, cfg in self.local_attn_config.items():
                lvl_int = int(lvl)
                if lvl_int in _pack_covered:
                    continue
                if active_level_set is not None and lvl_int not in active_level_set:
                    continue
                cfg_causal = bool(cfg.get("causal", False))
                # L0 causality is additionally gated by runtime AR flags.
                # This preserves explicit per-level config for upper levels.
                effective_causal = cfg_causal and global_causal_gate
                if lvl_int == 0:
                    effective_causal = effective_causal and bool(
                        getattr(self, "l0_local_runtime_causal", self.l0_local_causal_default)
                    )
                if self.per_level_attn_active and lvl_int < len(self.per_level_attn_heads):
                    # per-level LOCAL attention dim: project x -> d_L, flash-windowed, out d_L->hidden
                    lvl_result = self._compute_level_local_out_perdim(
                        x_normed=x, node_level=node_level, level=lvl_int,
                        window=int(cfg.get("window", 0)), causal=effective_causal,
                        backend=str(cfg.get("backend", "sdpa")),
                        level_offsets=level_offsets, rope_pos=rope_pos,
                    )
                else:
                    lvl_result = self._compute_level_local_out_batched(
                        q=q, k=k, v=v, node_level=node_level,
                        level=lvl_int,
                        window=int(cfg.get("window", 0)),
                        causal=effective_causal,
                        backend=str(cfg.get("backend", "sdpa")),
                        node_group=runtime_group,
                        positions=pos_to_use,
                        level_offsets=level_offsets,
                    )
                if lvl_result is not None:
                    lvl_idx, lvl_proj = lvl_result
                    if source_gates is not None:
                        lvl_proj = source_gates["local"] * lvl_proj
                    if isinstance(lvl_idx, slice):
                        out[:, lvl_idx, :] = out[:, lvl_idx, :] + lvl_proj
                    else:
                        out[:, lvl_idx, :] = out[:, lvl_idx, :] + lvl_proj
        elif _use_legacy_l0 and (active_level_set is None or 0 in active_level_set):
            l0_local = self._compute_l0_local_out_batched(q=q, k=k, v=v, node_level=node_level)
            if l0_local is not None:
                l0_idx, l0_proj = l0_local
                if source_gates is not None:
                    l0_proj = source_gates["local"] * l0_proj
                out[:, l0_idx, :] = out[:, l0_idx, :] + l0_proj

        # HQD sparse attention (fused inside transformer, reusable q/k/v)
        self._last_hqd_runtime_added_total = None
        self._last_hqd_runtime_stage_stats = None
        self._last_hqd_runtime_profile_stats = None
        hqd_packed_l0 = None
        if hqd_edges is None and self.hqd_runtime_selector is not None:
            with torch.no_grad():
                # HQD selection (L0 query -> coarse keys) reads the cross-level qx/kx so it
                # inherits the per-level cross-level projections (== q/k for shared QKV).
                selected = self.hqd_runtime_selector(qx.detach(), kx.detach())
            if selected is not None:
                if isinstance(selected, dict):
                    edges = selected.get("edges", None)
                    if edges is not None:
                        hqd_b_idx, hqd_src_idx, hqd_dst_idx = edges
                    else:
                        hqd_b_idx = hqd_src_idx = hqd_dst_idx = None
                    hqd_packed_l0 = selected.get("packed_l0", None)
                    added_total = int(selected.get("added_total", 0))
                    stage_stats = selected.get("stage_stats", None)
                    profile_stats = selected.get("profile_stats", None)
                else:
                    hqd_b_idx, hqd_src_idx, hqd_dst_idx, added_total, stage_stats, profile_stats = selected
                self._last_hqd_runtime_added_total = int(added_total)
                self._last_hqd_runtime_stage_stats = dict(stage_stats) if stage_stats is not None else None
                self._last_hqd_runtime_profile_stats = dict(profile_stats) if profile_stats is not None else None
                if hqd_b_idx is not None and hqd_b_idx.numel() > 0:
                    hqd_edges = (hqd_b_idx, hqd_src_idx, hqd_dst_idx)

        # Fat-V: the HQD final read carries a wide value payload through the sparse
        # aggregation (selection still uses the narrow q/k above). v_hqd == v when off.
        if getattr(self, "sparse_v_wide", False) and (hqd_edges is not None or hqd_packed_l0 is not None):
            # x is already input-normed above (matching how the narrow v was projected).
            v_hqd = self.sparse_proj_v(x).view(B, num_nodes, self.num_heads, self.sparse_v_head_dim)
        else:
            v_hqd = v

        # Fetch reads span 100s-1000s of tokens; the RoPE'd L0<->L0 geometry is only trained
        # inside the local window, so with hqd_read_prerope the read matches on content
        # (pre-RoPE q/k) instead of out-of-distribution far rotations.
        q_hqd, k_hqd = q, k
        if bool(getattr(self, "hqd_read_prerope", False)) and q_prerope is not None:
            q_hqd, k_hqd = q_prerope, k_prerope

        if hqd_edges is not None:
            hqd_b_idx, hqd_src_idx, hqd_dst_idx = hqd_edges
            if hqd_src_idx is not None and hqd_src_idx.dim() == 3:
                # Packed per-query candidate table [B, Q, K] (-1 padded) from xq nomination:
                # fixed-K dense read — same math as the per-edge scatter, but GEMM-shaped and
                # without materializing per-edge q/k/v gathers in autograd.
                hqd_out = self._compute_hqd_packed_l0_attn(
                    q=q_hqd, k=k_hqd, v=v_hqd,
                    dst_nodes=hqd_dst_idx, candidate_nodes=hqd_src_idx,
                    num_nodes=num_nodes, B=B,
                )
            elif str(getattr(self, "hqd_attn_impl", "scatter")).lower() == "dense":
                hqd_out = self._compute_hqd_dense_attn(
                    q=q_hqd, k=k_hqd, v=v_hqd,
                    b_idx=hqd_b_idx, src_idx=hqd_src_idx, dst_idx=hqd_dst_idx,
                    num_nodes=num_nodes, B=B,
                    backend=str(getattr(self, "hqd_dense_backend", "sdpa")).lower(),
                )
            else:
                hqd_out = self._compute_hqd_sparse_attn(
                    q=q_hqd, k=k_hqd, v=v_hqd,
                    b_idx=hqd_b_idx, src_idx=hqd_src_idx, dst_idx=hqd_dst_idx,
                    num_nodes=num_nodes, B=B,
                )
            if hqd_out is not None:
                # xq per-query relevance gate (differentiable; set/cleared around the layer
                # call by the model when nominations feed this layer).
                _q_gate = getattr(self, "_hqd_out_query_gate", None)
                if _q_gate is not None:
                    hqd_out = hqd_out * _q_gate.to(dtype=hqd_out.dtype)
                if source_gates is not None:
                    hqd_out = source_gates["hqd"] * hqd_out
                out = out + hqd_out

        if hqd_packed_l0 is not None:
            packed_dst, packed_candidates = hqd_packed_l0
            hqd_out = self._compute_hqd_packed_l0_attn(
                q=q_hqd, k=k_hqd, v=v_hqd,
                dst_nodes=packed_dst, candidate_nodes=packed_candidates,
                num_nodes=num_nodes, B=B,
            )
            if hqd_out is not None:
                if source_gates is not None:
                    hqd_out = source_gates["hqd"] * hqd_out
                out = out + hqd_out

        return out, new_edge_attr

    def _build_local_attn_bias(self, t: int, window: int, causal: bool, device, dtype):
        self._guard_dense_local_attn_mask(t=t, window=window, backend="sdpa", reason="1D local attention mask")
        idx = torch.arange(t, device=device)
        dist = (idx.view(t, 1) - idx.view(1, t)).abs()
        allow = dist <= int(window)
        if causal:
            allow = allow & (idx.view(t, 1) >= idx.view(1, t))
        neg = torch.finfo(dtype).min
        bias = torch.full((t, t), neg, device=device, dtype=dtype)
        bias = bias.masked_fill(allow, 0.0)
        return bias

    def _guard_dense_local_attn_mask(self, t: int, window: int, backend: str, reason: str) -> None:
        max_tokens = int(getattr(self, "local_attn_dense_mask_max_tokens", 8192))
        if int(window) > 0 and int(t) > max_tokens:
            raise RuntimeError(
                f"Local attention backend '{backend}' would build a dense {int(t)}x{int(t)} mask for {reason}. "
                "Use --l0_local_backend flash for true sliding-window FlashAttention, or --l0_local_backend pyg "
                "to use only graph-based sparse attention."
            )

    def _resolve_level_grid_shape(self, level: int) -> Optional[Tuple[int, int]]:
        mapping = getattr(self, "local_attn_runtime_level_grid_shapes", None)
        if not isinstance(mapping, dict):
            return None
        shape = mapping.get(int(level), None)
        if shape is None:
            return None
        try:
            gh = int(shape[0])
            gw = int(shape[1])
        except Exception:
            return None
        if gh <= 0 or gw <= 0:
            return None
        return (gh, gw)

    def _should_try_2d_rope(self) -> bool:
        mode = str(getattr(self, "rope_mode", "auto")).lower()
        if mode == "1d":
            return False
        if mode in {"2d", "2d_axial", "axial", "grid2d"}:
            return True
        # auto
        mapping = getattr(self, "local_attn_runtime_level_grid_shapes", None)
        return isinstance(mapping, dict) and len(mapping) > 0

    def _append_level_axis_to_rope_positions(self, pos: torch.Tensor, node_level: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self, "rope_level_axis_enable", False)):
            return pos
        if pos is None:
            return pos
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos, device=node_level.device)
        level_scale = float(getattr(self, "rope_level_axis_scale", 32.0))
        level_axis = torch.round(node_level.to(device=pos.device, dtype=torch.float32) * level_scale).to(dtype=torch.long)
        if pos.dim() == 1:
            return torch.stack([pos.to(dtype=torch.long), level_axis], dim=-1)
        if pos.dim() == 2:
            if int(pos.size(-1)) >= 3:
                return pos.to(dtype=torch.long)
            return torch.cat([pos.to(dtype=torch.long), level_axis.unsqueeze(-1)], dim=-1)
        return pos

    def _build_rope_positions(
        self,
        positions: Optional[torch.Tensor],
        node_level: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if positions is None:
            return None
        if not torch.is_tensor(positions):
            pos = torch.as_tensor(positions, device=node_level.device)
        else:
            pos = positions.to(device=node_level.device)

        rope_mode = str(getattr(self, "rope_mode", "auto")).lower()
        if rope_mode == "1d" and pos.dim() >= 2 and int(pos.size(-1)) > 1:
            raise ValueError(
                f"rope_mode='1d' does not accept multi-axis positions with shape {tuple(pos.shape)}"
            )

        if pos.dim() == 3 and pos.size(-1) >= 2:
            pos = pos[0]

        # Already 2D coordinates.
        if pos.dim() == 2 and pos.size(-1) >= 2:
            return self._append_level_axis_to_rope_positions(pos.to(dtype=torch.long), node_level)

        if pos.dim() > 1:
            pos = pos.reshape(-1)
        pos = pos.to(dtype=torch.long)
        if pos.numel() != node_level.numel():
            return self._append_level_axis_to_rope_positions(pos, node_level)

        if not self._should_try_2d_rope():
            return self._append_level_axis_to_rope_positions(pos, node_level)

        mapping = getattr(self, "local_attn_runtime_level_grid_shapes", None)
        if not isinstance(mapping, dict) or len(mapping) == 0:
            return self._append_level_axis_to_rope_positions(pos, node_level)

        yx = torch.zeros((pos.numel(), 2), dtype=torch.long, device=pos.device)
        assigned = torch.zeros((pos.numel(),), dtype=torch.bool, device=pos.device)

        for lvl, shape in mapping.items():
            try:
                gh = int(shape[0])
                gw = int(shape[1])
            except Exception:
                continue
            if gh <= 0 or gw <= 0:
                continue
            mask = node_level.to(dtype=torch.long) == int(lvl)
            if not bool(mask.any()):
                continue
            total = int(gh * gw)
            local = pos[mask].clamp(min=0, max=max(0, total - 1))
            yx[mask, 0] = local // max(1, gw)
            yx[mask, 1] = local % max(1, gw)
            assigned[mask] = True

        if not bool(assigned.any()):
            return self._append_level_axis_to_rope_positions(pos, node_level)

        if not bool(assigned.all()):
            if rope_mode in {"2d", "2d_axial", "axial", "grid2d"}:
                raise ValueError(
                    "2D RoPE requested, but not all nodes could be mapped to grid coordinates."
                )
            logger.warning(
                "Partial 2D RoPE mapping for mode=%s; falling back to 1D positions for this call.",
                rope_mode,
            )
            return self._append_level_axis_to_rope_positions(pos, node_level)
        return self._append_level_axis_to_rope_positions(yx, node_level)

    def _build_spatial_local_attn_bias(
        self,
        local_pos: torch.Tensor,
        grid_shape: Tuple[int, int],
        window: int,
        causal: bool,
        metric: str,
        device,
        dtype,
    ) -> torch.Tensor:
        local_pos_tensor = local_pos if torch.is_tensor(local_pos) else torch.as_tensor(local_pos)
        t = int(local_pos_tensor.size(0)) if local_pos_tensor.dim() >= 1 else int(local_pos_tensor.numel())
        if t <= 0:
            return torch.empty((0, 0), device=device, dtype=dtype)
        self._guard_dense_local_attn_mask(t=t, window=window, backend="sdpa", reason="spatial local attention bias")

        gh = int(grid_shape[0])
        gw = int(grid_shape[1])
        total = int(gh * gw)
        lp = local_pos_tensor.to(device=device, dtype=torch.long)
        if lp.dim() == 0:
            lp = lp.view(1)
        elif lp.dim() == 1 and lp.numel() != t:
            lp = torch.arange(t, device=device, dtype=torch.long)

        if lp.dim() >= 2 and int(lp.size(-1)) >= 2:
            coords = lp.reshape(t, -1)[..., :2]
            yy = coords[:, 0]
            xx = coords[:, 1]
            if grid_shape is not None:
                gh = int(grid_shape[0])
                gw = int(grid_shape[1])
                if (
                    (yy < 0).any()
                    or (yy >= gh).any()
                    or (xx < 0).any()
                    or (xx >= gw).any()
                ):
                    raise ValueError(
                        f"Spatial local-attn coords out of bounds for grid_shape={tuple(grid_shape)}"
                    )
                width = gw
            else:
                width = max(1, int(xx.max().item()) + 1)

            dy = (yy.view(t, 1) - yy.view(1, t)).abs()
            dx = (xx.view(t, 1) - xx.view(1, t)).abs()

            metric_norm = str(metric).lower()
            if metric_norm == "manhattan":
                allow = (dy + dx) <= int(window)
            else:
                allow = torch.maximum(dy, dx) <= int(window)

            if causal:
                linear = yy * int(width) + xx
                allow = allow & (linear.view(t, 1) >= linear.view(1, t))

            neg = torch.finfo(dtype).min
            bias = torch.full((t, t), neg, device=device, dtype=dtype)
            bias = bias.masked_fill(allow, 0.0)
            return bias

        valid = (lp >= 0) & (lp < total)
        if not bool(valid.all()):
            lp = torch.arange(t, device=device, dtype=torch.long)
            valid = (lp >= 0) & (lp < total)
            if not bool(valid.all()):
                lp = lp.clamp(min=0, max=max(0, total - 1))

        yy = lp // max(1, gw)
        xx = lp % max(1, gw)
        dy = (yy.view(t, 1) - yy.view(1, t)).abs()
        dx = (xx.view(t, 1) - xx.view(1, t)).abs()

        metric_norm = str(metric).lower()
        if metric_norm == "manhattan":
            allow = (dy + dx) <= int(window)
        else:
            allow = torch.maximum(dy, dx) <= int(window)

        if causal:
            allow = allow & (lp.view(t, 1) >= lp.view(1, t))

        neg = torch.finfo(dtype).min
        bias = torch.full((t, t), neg, device=device, dtype=dtype)
        bias = bias.masked_fill(allow, 0.0)
        return bias

    def _compute_level_role_gate(
        self,
        level: int,
        device,
        dtype,
    ) -> Optional[torch.Tensor]:
        if not bool(getattr(self, "local_attn_level_role_bias_enable", True)):
            return None
        try:
            lvl = int(level)
            lvl = max(0, min(lvl, int(self.level_embedding.num_embeddings) - 1))
            lvl_tensor = torch.tensor([lvl], device=device, dtype=torch.long)
            lvl_emb = self.level_embedding(lvl_tensor)
            lvl_pair = torch.cat([lvl_emb, lvl_emb], dim=-1)
            gate = self.level_attn(lvl_pair).squeeze(0)
            gate = F.softplus(gate).to(device=device, dtype=dtype)
            scale = float(getattr(self, "local_attn_level_role_bias_scale", 1.0))
            return gate * scale
        except Exception as exc:
            if not getattr(self, "_local_attn_warned", False):
                logger.warning("Level role gate computation failed: %s", str(exc))
                self._local_attn_warned = True
            return None

    def _build_level_aware_local_attn_bias(
        self,
        local_pos: torch.Tensor,
        grid_shape: Optional[Tuple[int, int]],
        window: int,
        causal: bool,
        metric: str,
        level: int,
        device,
        dtype,
    ) -> torch.Tensor:
        local_pos_tensor = local_pos if torch.is_tensor(local_pos) else torch.as_tensor(local_pos)
        t = int(local_pos_tensor.size(0)) if local_pos_tensor.dim() >= 1 else int(local_pos_tensor.numel())
        if t <= 0:
            return torch.empty((0, 0), device=device, dtype=dtype)
        self._guard_dense_local_attn_mask(t=t, window=window, backend="sdpa", reason="level-aware local attention bias")

        lp = local_pos_tensor.to(device=device, dtype=torch.long)
        if lp.dim() == 0:
            lp = lp.view(1)
        elif lp.dim() == 1 and lp.numel() != t:
            lp = torch.arange(t, device=device, dtype=torch.long)

        if lp.dim() >= 2 and int(lp.size(-1)) >= 2:
            coords = lp.reshape(t, -1)[..., :2]
            yy = coords[:, 0]
            xx = coords[:, 1]
            if grid_shape is not None:
                gh = int(grid_shape[0])
                gw = int(grid_shape[1])
                if (
                    (yy < 0).any()
                    or (yy >= gh).any()
                    or (xx < 0).any()
                    or (xx >= gw).any()
                ):
                    raise ValueError(
                        f"Level-aware local-attn coords out of bounds for grid_shape={tuple(grid_shape)}"
                    )
                width = gw
            else:
                width = max(1, int(xx.max().item()) + 1)

            dy = (yy.view(t, 1) - yy.view(1, t)).abs()
            dx = (xx.view(t, 1) - xx.view(1, t)).abs()
            metric_norm = str(metric).lower()
            if metric_norm == "manhattan":
                dist = dy + dx
            else:
                dist = torch.maximum(dy, dx)

            window_int = int(window)
            allow = dist <= window_int
            if causal:
                linear = yy * int(width) + xx
                allow = allow & (linear.view(t, 1) >= linear.view(1, t))

            neg = torch.finfo(dtype).min
            base_bias = torch.full((t, t), neg, device=device, dtype=dtype)
            base_bias = base_bias.masked_fill(allow, 0.0)

            if window_int <= 0:
                return base_bias

            gate = self._compute_level_role_gate(level=level, device=device, dtype=dtype)
            if gate is None:
                return base_bias

            dist_norm = dist.to(dtype=dtype) / float(max(1, window_int))
            role_bias = -gate.view(-1, 1, 1) * dist_norm.view(1, t, t)
            role_bias = role_bias.mean(dim=0)
            role_bias = role_bias.masked_fill(~allow, 0.0)
            return base_bias + role_bias

        if grid_shape is not None:
            gh = int(grid_shape[0])
            gw = int(grid_shape[1])
            total = int(gh * gw)
            valid = (lp >= 0) & (lp < total)
            if not bool(valid.all()):
                lp = torch.arange(t, device=device, dtype=torch.long)
                valid = (lp >= 0) & (lp < total)
                if not bool(valid.all()):
                    lp = lp.clamp(min=0, max=max(0, total - 1))
            yy = lp // max(1, gw)
            xx = lp % max(1, gw)
            dy = (yy.view(t, 1) - yy.view(1, t)).abs()
            dx = (xx.view(t, 1) - xx.view(1, t)).abs()
            metric_norm = str(metric).lower()
            if metric_norm == "manhattan":
                dist = dy + dx
            else:
                dist = torch.maximum(dy, dx)
        else:
            dist = (lp.view(t, 1) - lp.view(1, t)).abs()

        window_int = int(window)
        allow = dist <= window_int
        if causal:
            allow = allow & (lp.view(t, 1) >= lp.view(1, t))

        neg = torch.finfo(dtype).min
        base_bias = torch.full((t, t), neg, device=device, dtype=dtype)
        base_bias = base_bias.masked_fill(allow, 0.0)

        if window_int <= 0:
            return base_bias

        gate = self._compute_level_role_gate(level=level, device=device, dtype=dtype)
        if gate is None:
            return base_bias

        dist_norm = dist.to(dtype=dtype) / float(max(1, window_int))
        role_bias = -gate.view(-1, 1, 1) * dist_norm.view(1, t, t)
        return base_bias.unsqueeze(0) + role_bias

    def _compute_hqd_sparse_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b_idx: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int,
        B: int,
    ) -> Optional[torch.Tensor]:
        if b_idx.numel() == 0:
            return None
        profile_enabled = bool(getattr(self, "hqd_profile_enable", False))
        _t0 = time.monotonic() if profile_enabled else 0.0

        b_idx = b_idx.to(device=q.device, dtype=torch.long)
        src_idx = src_idx.to(device=q.device, dtype=torch.long)
        dst_idx = dst_idx.to(device=q.device, dtype=torch.long)

        q_dst = q[b_idx, dst_idx]
        k_src = k[b_idx, src_idx]
        head_dim = max(1, int(q_dst.size(-1)))
        num_heads = max(1, int(q_dst.size(-2)))
        # Fat-V: the value head dim may differ from the qk head dim. Read it from v (==
        # head_dim when fat-V is off) and down-project with the matching out projection.
        v_dim = max(1, int(v.size(-1)))

        scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(float(head_dim))

        group_idx = b_idx * num_nodes + dst_idx
        weights = softmax(scores, group_idx)

        v_src = v[b_idx, src_idx]
        msg = v_src * weights.unsqueeze(-1)

        if bool(getattr(self, "hqd_sparse_project_active_only", False)):
            unique_groups, inverse = torch.unique(group_idx, sorted=False, return_inverse=True)
            active_heads = scatter_add(msg, inverse, dim=0, dim_size=int(unique_groups.numel()))
            active_hidden = active_heads.reshape(active_heads.size(0), num_heads * v_dim)
            active_proj = self.sparse_out_proj(active_hidden)
            out_flat = torch.zeros(
                (B * num_nodes, self.hidden_dim),
                device=q.device,
                dtype=active_proj.dtype,
            )
            out_flat.index_copy_(0, unique_groups, active_proj)
            self._last_hqd_apply_ms = (time.monotonic() - _t0) * 1000.0 if profile_enabled else None
            return out_flat.view(B, num_nodes, self.hidden_dim)

        out_flat = scatter_add(msg, group_idx, dim=0, dim_size=B * num_nodes)
        out = out_flat.view(B, num_nodes, num_heads, v_dim)
        out = out.reshape(B, num_nodes, num_heads * v_dim)
        out = self.sparse_out_proj(out)
        self._last_hqd_apply_ms = (time.monotonic() - _t0) * 1000.0 if profile_enabled else None
        return out

    def _capture_graph_witnesses(
        self,
        weights: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_level: torch.Tensor,
        num_nodes: int,
    ) -> None:
        """Store top contributing direct children from the existing graph attention.

        For each parent level L, rows in the resulting table contain top-k direct
        level-(L-1) source nodes that contributed to that parent during the current
        graph attention pass. This is intentionally detached metadata: later HQD can
        look it up without re-scoring the hierarchy or materializing an L0 edge list.
        """
        if not bool(getattr(self, "hqd_graph_witness_enable", False)):
            self._last_graph_witness_ids = None
            self._last_graph_witness_scores = None
            return
        k_top = max(1, int(getattr(self, "hqd_graph_witness_topk", 4)))
        if weights.numel() == 0 or src.numel() == 0 or dst.numel() == 0:
            self._last_graph_witness_ids = None
            self._last_graph_witness_scores = None
            return

        with torch.no_grad():
            device = weights.device
            B = int(weights.size(0))
            nl = node_level.to(device=device, dtype=torch.long)
            src_l = nl.index_select(0, src.to(device=device, dtype=torch.long))
            dst_l = nl.index_select(0, dst.to(device=device, dtype=torch.long))
            score_e = weights.detach().mean(dim=-1)  # [B, E]
            ids_by_level: Dict[int, torch.Tensor] = {}
            scores_by_level: Dict[int, torch.Tensor] = {}
            for level in (1, 2, 3):
                mask = (dst_l == int(level)) & (src_l == int(level - 1))
                if not bool(mask.any()):
                    continue
                edge_pos = torch.nonzero(mask, as_tuple=False).view(-1)
                src_m = src.index_select(0, edge_pos).to(device=device, dtype=torch.long)
                dst_m = dst.index_select(0, edge_pos).to(device=device, dtype=torch.long)
                ids = torch.full((B, int(num_nodes), k_top), -1, device=device, dtype=torch.long)
                vals_out = torch.full((B, int(num_nodes), k_top), float("-inf"), device=device, dtype=score_e.dtype)
                for b in range(B):
                    scores = score_e[b].index_select(0, edge_pos)
                    if scores.numel() == 0:
                        continue
                    # Stable sort by score first, then by destination; within each destination
                    # the stable order remains descending score, so rank is top-k per parent.
                    by_score = torch.argsort(scores, descending=True, stable=True)
                    dst_s = dst_m.index_select(0, by_score)
                    src_s = src_m.index_select(0, by_score)
                    score_s = scores.index_select(0, by_score)
                    by_dst = torch.argsort(dst_s, descending=False, stable=True)
                    dst_s = dst_s.index_select(0, by_dst)
                    src_s = src_s.index_select(0, by_dst)
                    score_s = score_s.index_select(0, by_dst)
                    _, counts = torch.unique_consecutive(dst_s, return_counts=True)
                    starts = torch.cumsum(counts, dim=0) - counts
                    rank = torch.arange(dst_s.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(starts, counts)
                    keep = rank < k_top
                    if bool(keep.any()):
                        ids[b, dst_s[keep], rank[keep]] = src_s[keep]
                        vals_out[b, dst_s[keep], rank[keep]] = score_s[keep]
                ids_by_level[int(level)] = ids
                scores_by_level[int(level)] = vals_out
            self._last_graph_witness_ids = ids_by_level or None
            self._last_graph_witness_scores = scores_by_level or None

    def _compute_hqd_packed_l0_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dst_nodes: torch.Tensor,
        candidate_nodes: torch.Tensor,
        num_nodes: int,
        B: int,
    ) -> Optional[torch.Tensor]:
        """Packed fixed-K L0 witness read, avoiding flattened sparse edge materialization."""
        if dst_nodes.numel() == 0 or candidate_nodes.numel() == 0:
            return None
        profile_enabled = bool(getattr(self, "hqd_profile_enable", False))
        _t0 = time.monotonic() if profile_enabled else 0.0
        device = q.device
        dst_nodes = dst_nodes.to(device=device, dtype=torch.long)
        candidate_nodes = candidate_nodes.to(device=device, dtype=torch.long)
        Bc, Q, K = int(candidate_nodes.size(0)), int(candidate_nodes.size(1)), int(candidate_nodes.size(2))
        if Bc != int(B) or Q != int(dst_nodes.numel()) or K <= 0:
            return None

        num_heads = max(1, int(q.size(-2)))
        head_dim = max(1, int(q.size(-1)))
        v_dim = max(1, int(v.size(-1)))  # Fat-V: value head dim (== head_dim when off)
        chunk = max(1, int(getattr(self, "hqd_packed_witness_chunk_size", 2048)))
        neg = torch.finfo(q.dtype).min
        batch_offsets = torch.arange(int(B), device=device, dtype=torch.long).view(int(B), 1, 1) * int(num_nodes)
        flat_k = k.reshape(int(B) * int(num_nodes), num_heads, head_dim)
        flat_v = v.reshape(int(B) * int(num_nodes), num_heads, v_dim)
        # xq nomination always targets the full contiguous L0 range: read queries by slice
        # and write messages by slice (one CopySlices backward) instead of index_select +
        # index_add_ over [B*N] rows, whose scatter-style backward dominated the read cost.
        # Non-contiguous dst (HQD witness sets) keeps the index path.
        d0 = int(dst_nodes[0].item()) if Q > 0 else 0
        contig = Q > 1 and bool(
            (dst_nodes == torch.arange(d0, d0 + Q, device=device, dtype=torch.long)).all().item()
        )
        msgs: list = []
        out_flat = None
        if not contig:
            out_flat = torch.zeros((int(B) * int(num_nodes), num_heads, v_dim), device=device, dtype=q.dtype)

        for start in range(0, Q, chunk):
            end = min(Q, start + chunk)
            cand = candidate_nodes[:, start:end, :]
            valid = cand >= 0
            if not contig and not bool(valid.any()):
                continue
            safe = cand.clamp(min=0)
            gather_idx = (safe + batch_offsets).reshape(-1)
            k_c = flat_k.index_select(0, gather_idx).view(int(B), end - start, K, num_heads, head_dim)
            v_c = flat_v.index_select(0, gather_idx).view(int(B), end - start, K, num_heads, v_dim)
            if contig:
                q_c = q[:, d0 + start : d0 + end]  # [B,Qc,H,D]
            else:
                q_c = q.index_select(1, dst_nodes[start:end])  # [B,Qc,H,D]
            scores = (q_c.unsqueeze(2) * k_c).sum(dim=-1) / math.sqrt(float(head_dim))  # [B,Qc,K,H]
            scores = scores.masked_fill(~valid.unsqueeze(-1), neg)
            sink_k = getattr(self, "hqd_read_sink_k", None)
            if sink_k is not None:
                # Zero-value sink slot: lets a query dump softmax mass when no candidate is
                # relevant (a forced read over junk fetches is pure noise otherwise). The
                # sink logit competes per query/head; its value contribution is zero, so
                # mass on the sink shrinks the message instead of averaging irrelevant v's.
                sink_logit = (q_c * sink_k.view(1, 1, num_heads, head_dim).to(dtype=q_c.dtype)).sum(-1)
                scores = torch.cat([scores, (sink_logit / math.sqrt(float(head_dim))).unsqueeze(2)], dim=2)
            weights = torch.softmax(scores, dim=2)
            if sink_k is not None:
                weights = weights[:, :, :K, :]
            weights = torch.where(valid.unsqueeze(-1), weights, torch.zeros_like(weights))
            msg = (weights.unsqueeze(-1) * v_c).sum(dim=2)  # [B,Qc,H,Dv]
            msg = msg.to(dtype=q.dtype)
            if contig:
                msgs.append(msg)
            else:
                groups = (torch.arange(int(B), device=device, dtype=torch.long).view(int(B), 1) * int(num_nodes)) + dst_nodes[start:end].view(1, -1)
                out_flat.index_add_(0, groups.reshape(-1), msg.reshape(int(B) * (end - start), num_heads, v_dim))

        if contig:
            m_all = torch.cat(msgs, dim=1) if len(msgs) > 1 else msgs[0]  # [B, Q, H, Dv]
            out_buf = torch.zeros((int(B), int(num_nodes), num_heads, v_dim), device=device, dtype=q.dtype)
            out_buf[:, d0 : d0 + Q] = m_all
            out = out_buf.reshape(int(B), int(num_nodes), num_heads * v_dim)
        else:
            out = out_flat.view(int(B), int(num_nodes), num_heads, v_dim).reshape(int(B), int(num_nodes), num_heads * v_dim)
        out = self.sparse_out_proj(out)
        self._last_hqd_apply_ms = (time.monotonic() - _t0) * 1000.0 if profile_enabled else None
        return out

    def _compute_hqd_dense_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b_idx: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int,
        B: int,
        backend: str = "sdpa",
    ) -> Optional[torch.Tensor]:
        """Dense per-destination attention over the HQD edge set.

        Mathematically the same operation as ``_compute_hqd_sparse_attn`` (each
        destination attends, with a softmax, over the candidate sources on its
        incident edges), but instead of an edge-wise gather + segment-softmax +
        ``scatter_add`` it groups each destination's edges into a padded
        ``[G, Kmax, H, D]`` block and runs a fused attention (SDPA, or
        flash-varlen when available). The edge *set* is identical, so causality
        is identical: a destination only ever sees causal sources, and a padded
        slot is masked out of the softmax. This trades the bandwidth/launch-bound
        scatter for a fused kernel, the dominant HQD-read cost at long context.
        """
        if b_idx.numel() == 0:
            return None
        profile_enabled = bool(getattr(self, "hqd_profile_enable", False))
        _t0 = time.monotonic() if profile_enabled else 0.0

        device = q.device
        b_idx = b_idx.to(device=device, dtype=torch.long)
        src_idx = src_idx.to(device=device, dtype=torch.long)
        dst_idx = dst_idx.to(device=device, dtype=torch.long)
        head_dim = max(1, int(q.size(-1)))
        num_heads = max(1, int(q.size(-2)))
        v_dim = max(1, int(v.size(-1)))  # Fat-V: value head dim (== head_dim when off)

        # Group edges by (batch, dst). uniq is sorted; inv maps each edge -> group.
        group_idx = b_idx * num_nodes + dst_idx
        uniq, inv = torch.unique(group_idx, sorted=True, return_inverse=True)
        G = int(uniq.numel())
        group_b = torch.div(uniq, num_nodes, rounding_mode="floor")
        group_dst = uniq - group_b * num_nodes

        counts = torch.bincount(inv, minlength=G)
        # Per-edge rank within its group (stable sort -> deterministic ordering).
        order = torch.argsort(inv, stable=True)
        first_offset = torch.zeros(G, device=device, dtype=torch.long)
        if G > 1:
            first_offset[1:] = torch.cumsum(counts, dim=0)[:-1]
        E = int(inv.numel())
        rank_sorted = torch.arange(E, device=device, dtype=torch.long) - first_offset.index_select(0, inv.index_select(0, order))
        rank = torch.empty(E, device=device, dtype=torch.long)
        rank[order] = rank_sorted

        q_g = q[group_b, group_dst]                      # [G, H, D]

        flash_varlen = None
        # flash-varlen requires the value head dim to match qk; under fat-V (v_dim != head_dim)
        # use the SDPA path, which supports a differing value dim.
        if backend == "flash" and v_dim == head_dim:
            try:  # best-effort; falls back to padded SDPA below
                from flash_attn import flash_attn_varlen_func as flash_varlen  # type: ignore
            except Exception:
                flash_varlen = None

        if flash_varlen is not None and q_g.is_cuda:
            # Ragged path: no padding. flash_attn_varlen_func wants packed 3D tensors
            # q [total_q, H, D] and k/v [total_k, H, D] with int32 cu_seqlens. One query
            # per group (cu_q = arange), keys/values concatenated in group order.
            edge_b = group_b.index_select(0, inv.index_select(0, order))
            edge_src = src_idx.index_select(0, order)
            k_var = k[edge_b, edge_src].contiguous()    # [E, H, D]
            v_var = v[edge_b, edge_src].contiguous()
            cu_q = torch.arange(0, G + 1, device=device, dtype=torch.int32)
            cu_k = torch.zeros(G + 1, device=device, dtype=torch.int32)
            cu_k[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
            max_k = int(counts.max().item()) if G > 0 else 0
            out_g = flash_varlen(
                q_g.contiguous(),                       # [G, H, D] -> total_q=G, seqlen 1
                k_var, v_var,
                cu_q, cu_k, 1, max_k,
                causal=False,
            )                                           # [G, H, D]
        else:
            Kmax = int(counts.max().item()) if G > 0 else 0
            # Padded SDPA pads every destination to the *max* degree, so it only wins
            # when degrees are roughly uniform. With a ragged HQD edge set (e.g. L0
            # queries ~28 edges but coarse bag nodes ~128), padding wastes most of the
            # block (G*Kmax >> #edges) and is several x slower than scatter. Fall back
            # to scatter when the padding-waste ratio is too high; flash-varlen (ragged)
            # has no padding and is the right dense backend for uneven degrees.
            pad_ratio = float(getattr(self, "hqd_dense_max_pad_ratio", 1.5))
            if G > 0 and int(G) * Kmax > pad_ratio * int(b_idx.numel()):
                return self._compute_hqd_sparse_attn(
                    q=q, k=k, v=v, b_idx=b_idx, src_idx=src_idx, dst_idx=dst_idx,
                    num_nodes=num_nodes, B=B,
                )
            cand_src = torch.full((G, Kmax), -1, device=device, dtype=torch.long)
            cand_src[inv, rank] = src_idx
            valid = cand_src >= 0                       # [G, Kmax]
            safe = cand_src.clamp(min=0)
            gb = group_b.view(G, 1).expand(G, Kmax)
            k_g = k[gb, safe]                           # [G, Kmax, H, D]
            v_g = v[gb, safe]                           # [G, Kmax, H, Dv]
            # Canonical SDPA layout [G(batch), H, L, D]; contiguous so the fused/
            # mem-efficient backward kernel reads correct strides (non-contiguous
            # permuted views cause an illegal memory access in backward on CUDA).
            # SDPA permits Dv != D: scores come from q,k (D); the output follows v (Dv).
            q_s = q_g.unsqueeze(2).contiguous()         # [G, H, 1, D]
            k_s = k_g.permute(0, 2, 1, 3).contiguous()  # [G, H, Kmax, D]
            v_s = v_g.permute(0, 2, 1, 3).contiguous()  # [G, H, Kmax, Dv]
            attn_mask = valid.view(G, 1, 1, Kmax)       # broadcast over heads
            out = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=attn_mask)  # [G, H, 1, Dv]
            out_g = out.squeeze(2)                      # [G, H, Dv]

        if bool(getattr(self, "hqd_sparse_project_active_only", False)):
            active_hidden = out_g.reshape(G, num_heads * v_dim)
            active_proj = self.sparse_out_proj(active_hidden)
            out_flat = torch.zeros((B * num_nodes, self.hidden_dim), device=device, dtype=active_proj.dtype)
            out_flat.index_copy_(0, uniq, active_proj)
            self._last_hqd_apply_ms = (time.monotonic() - _t0) * 1000.0 if profile_enabled else None
            return out_flat.view(B, num_nodes, self.hidden_dim)

        out_flat = torch.zeros((B * num_nodes, num_heads, v_dim), device=device, dtype=out_g.dtype)
        out_flat.index_copy_(0, uniq, out_g)
        out = self.sparse_out_proj(out_flat.reshape(B, num_nodes, num_heads * v_dim))
        self._last_hqd_apply_ms = (time.monotonic() - _t0) * 1000.0 if profile_enabled else None
        return out

    def _compute_l0_local_out_batched(self, q, k, v, node_level):
        l0_idx = torch.nonzero(node_level == 0, as_tuple=False).view(-1)
        if l0_idx.numel() <= 1:
            return None

        backend = str(getattr(self, "l0_local_backend", "pyg")).lower()
        window = int(getattr(self, "l0_local_window", 0))
        if backend == "pyg" or window <= 0:
            return None

        causal = bool(getattr(self, "l0_local_runtime_causal", self.l0_local_causal_default))
        log_key = ("l0", str(backend), int(window), bool(causal))
        if log_key not in self._local_attn_runtime_logged_keys:
            logger.info(
                "[HMP:LOCAL] level=0 path=l0-legacy backend=%s window=%d causal=%s tokens=%d",
                str(backend),
                int(window),
                bool(causal),
                int(l0_idx.numel()),
            )
            self._local_attn_runtime_logged_keys.add(log_key)

        q_l0 = q[:, l0_idx, :, :]
        k_l0 = k[:, l0_idx, :, :]
        v_l0 = v[:, l0_idx, :, :]
        B, T, Hh, Dh = q_l0.shape
        dropout_p = float(self.dropout.p) if self.training else 0.0
        role_bias_enabled = bool(getattr(self, "local_attn_level_role_bias_enable", True))
        attn_bias = None
        if backend == "flash" and role_bias_enabled and window > 0:
            raise RuntimeError(
                "Flash local attention cannot use local_attn_level_role_bias_enable because it requires a dense T x T bias. "
                "Disable local attention level role bias for long-context flash runs."
            )
        if role_bias_enabled and window > 0:
            local_pos = torch.arange(T, device=q_l0.device, dtype=torch.long)
            attn_bias = self._build_level_aware_local_attn_bias(
                local_pos=local_pos,
                grid_shape=None,
                window=window,
                causal=causal,
                metric="chebyshev",
                level=0,
                device=q_l0.device,
                dtype=q_l0.dtype,
            )

        if backend == "sdpa" and self._trace_mode_allows_window():
            traced_out = self._compute_traced_local_attn_for_indices(
                q_lvl=q_l0,
                k_lvl=k_l0,
                v_lvl=v_l0,
                window=window,
                causal=causal,
                attn_bias=attn_bias,
            )
            if traced_out is not None:
                out_l0 = traced_out.reshape(B, T, Hh * Dh)
                out_l0 = self.out_proj(out_l0)
                return l0_idx, out_l0

        resolved_backend = None
        flash_attn_func = None
        if backend == "flash" and attn_bias is None:
            resolved_backend, flash_attn_func = pick_attention_backend(q_l0.device)
        flash_supports_dropout = bool(
            backend == "flash"
            and resolved_backend in {"fa2", "fa3"}
            and flash_attn_func is not None
            and _flash_attn_supports_dropout(flash_attn_func)
        )

        out_l0 = None
        if (
            attn_bias is None
            and backend == "flash"
            and resolved_backend in {"fa2", "fa3"}
            and flash_attn_func is not None
            and (dropout_p <= 0.0 or flash_supports_dropout)
        ):
            if causal:
                win = (int(window), 0)
            else:
                win = (int(window), int(window))
            out_l0 = attention_forward(
                q_l0,
                k_l0,
                v_l0,
                causal=causal,
                dropout_p=dropout_p,
                backend=resolved_backend,
                flash_func=flash_attn_func,
                window_size=win,
                flash_dtype_cast=bool(getattr(self, "local_attn_flash_dtype_cast", False)),
            )

        if out_l0 is None and attn_bias is None and backend == "xformers" and _xops is not None:
            if causal and window <= 0:
                attn_bias = _xops.LowerTriangularMask()
                out_l0 = _xops.memory_efficient_attention(q_l0, k_l0, v_l0, attn_bias=attn_bias, p=dropout_p)

        if out_l0 is None:
            if backend in ("flash", "xformers"):
                raise RuntimeError(
                    f"L0 local attention backend '{backend}' could not run. "
                    "Dense SDPA fallback is disabled (quadratic, not suitable for local windows). "
                    "Use --l0_local_backend pyg for graph-based sparse attention, "
                    "or --l0_local_backend sdpa for dense windowed SDPA."
                )
            # backend == "sdpa": explicit user choice, run dense windowed SDPA
            qh = q_l0.transpose(1, 2)  # [B,H,T,D]
            kh = k_l0.transpose(1, 2)
            vh = v_l0.transpose(1, 2)
            if attn_bias is not None:
                if attn_bias.dim() == 2:
                    attn_mask = attn_bias.view(1, 1, T, T)
                elif attn_bias.dim() == 3:
                    attn_mask = attn_bias.view(1, attn_bias.size(0), T, T)
                else:
                    attn_mask = attn_bias
                out_h = F.scaled_dot_product_attention(
                    qh, kh, vh,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
            elif causal and window <= 0:
                out_h = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, dropout_p=dropout_p, is_causal=True)
            else:
                bias = self._build_local_attn_bias(
                    t=T, window=max(1, window), causal=causal,
                    device=qh.device, dtype=qh.dtype,
                )
                out_h = F.scaled_dot_product_attention(
                    qh, kh, vh,
                    attn_mask=bias.view(1, 1, T, T),
                    dropout_p=dropout_p,
                    is_causal=False,
                )
            out_l0 = out_h.transpose(1, 2)

        out_l0 = out_l0.reshape(B, T, Hh * Dh)
        out_l0 = self.out_proj(out_l0)
        return l0_idx, out_l0

    def _compute_local_attn_for_indices(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx: torch.Tensor,
        window: int,
        causal: bool,
        backend: str,
        level: int,
        attn_bias: Optional[torch.Tensor] = None,
        force_sdpa: bool = False,
    ) -> Optional[torch.Tensor]:
        if idx.numel() <= 1:
            return None

        q_lvl = q[:, idx, :, :]
        k_lvl = k[:, idx, :, :]
        v_lvl = v[:, idx, :, :]
        return self._compute_local_attn_from_qkv(
            q_lvl=q_lvl,
            k_lvl=k_lvl,
            v_lvl=v_lvl,
            window=window,
            causal=causal,
            backend=backend,
            level=level,
            attn_bias=attn_bias,
            force_sdpa=force_sdpa,
        )

    def _compute_local_attn_from_qkv(
        self,
        q_lvl: torch.Tensor,
        k_lvl: torch.Tensor,
        v_lvl: torch.Tensor,
        window: int,
        causal: bool,
        backend: str,
        level: int,
        attn_bias: Optional[torch.Tensor] = None,
        force_sdpa: bool = False,
        skip_out_proj: bool = False,  # return per-head [B,T,H,D] BEFORE out_proj (for LSE merge)
    ) -> Optional[torch.Tensor]:
        B, T, Hh, Dh = q_lvl.shape
        if T <= 1:
            return None
        dropout_p = float(self.dropout.p) if self.training else 0.0

        if backend == "sdpa" and self._trace_mode_allows_window():
            traced_out = self._compute_traced_local_attn_for_indices(
                q_lvl=q_lvl,
                k_lvl=k_lvl,
                v_lvl=v_lvl,
                window=window,
                causal=causal,
                attn_bias=attn_bias,
            )
            if traced_out is not None:
                if skip_out_proj:
                    return traced_out.reshape(B, T, Hh, Dh)
                out_lvl = traced_out.reshape(B, T, Hh * Dh)
                out_lvl = self.out_proj(out_lvl)
                return out_lvl

        out_lvl = None

        if attn_bias is not None:
            force_sdpa = True

        resolved_backend = None
        flash_attn_func = None
        if backend == "flash" and not force_sdpa:
            resolved_backend, flash_attn_func = pick_attention_backend(q_lvl.device)
        flash_supports_dropout = bool(
            backend == "flash"
            and resolved_backend in {"fa2", "fa3"}
            and flash_attn_func is not None
            and _flash_attn_supports_dropout(flash_attn_func)
        )

        # Try flash attention first
        if (
            (not force_sdpa)
            and backend == "flash"
            and resolved_backend in {"fa2", "fa3"}
            and flash_attn_func is not None
            and (dropout_p <= 0.0 or flash_supports_dropout)
        ):
            win = (int(window), 0) if causal else (int(window), int(window))
            out_lvl = attention_forward(
                q_lvl,
                k_lvl,
                v_lvl,
                causal=causal,
                dropout_p=dropout_p,
                backend=resolved_backend,
                flash_func=flash_attn_func,
                window_size=win,
                flash_dtype_cast=bool(getattr(self, "local_attn_flash_dtype_cast", False)),
            )

        # Try xformers
        if out_lvl is None and (not force_sdpa) and backend == "xformers" and _xops is not None:
            if causal and window <= 0:
                x_attn_bias = _xops.LowerTriangularMask()
                out_lvl = _xops.memory_efficient_attention(q_lvl, k_lvl, v_lvl, attn_bias=x_attn_bias, p=dropout_p)

        # SDPA fallback — enabled only for explicit 'sdpa' backend choice
        if out_lvl is None:
            if backend in ("flash", "xformers"):
                raise RuntimeError(
                    f"Local attention backend '{backend}' unavailable for level {level}. "
                    "Dense SDPA fallback is disabled (quadratic, not suitable for local windows). "
                    "Use --l0_local_backend pyg for graph-based sparse attention, "
                    "or --l0_local_backend sdpa for dense windowed SDPA."
                )
            # backend == "sdpa": explicit user choice, run dense windowed SDPA
            qh = q_lvl.transpose(1, 2)  # [B,H,T,D]
            kh = k_lvl.transpose(1, 2)
            vh = v_lvl.transpose(1, 2)
            if attn_bias is not None:
                if attn_bias.dim() == 2:
                    attn_mask = attn_bias.view(1, 1, T, T)
                elif attn_bias.dim() == 3:
                    attn_mask = attn_bias.view(1, attn_bias.size(0), T, T)
                elif attn_bias.dim() == 4:
                    attn_mask = attn_bias
                else:
                    raise ValueError(f"Unsupported local attn bias shape: {tuple(attn_bias.shape)}")
                out_h = F.scaled_dot_product_attention(
                    qh, kh, vh,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
            elif causal and window <= 0:
                out_h = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, dropout_p=dropout_p, is_causal=True)
            else:
                bias = self._build_local_attn_bias(
                    t=T, window=max(1, window), causal=causal,
                    device=qh.device, dtype=qh.dtype,
                )
                out_h = F.scaled_dot_product_attention(
                    qh, kh, vh,
                    attn_mask=bias.view(1, 1, T, T),
                    dropout_p=dropout_p,
                    is_causal=False,
                )
            out_lvl = out_h.transpose(1, 2)

        if skip_out_proj:
            return out_lvl.reshape(B, T, Hh, Dh)
        out_lvl = out_lvl.reshape(B, T, Hh * Dh)
        out_lvl = self.out_proj(out_lvl)
        return out_lvl

    def _compute_level_local_out_perdim(self, x_normed, node_level, level, window, causal, backend, level_offsets, rope_pos):
        """Per-level LOCAL window attention at the level's own dim (num_heads_L, head_dim fixed).
        Projects x -> d_L q/k/v, flash-windowed attention, out_proj d_L -> hidden. Returns
        (slice_or_idx, out_L[B,n_L,hidden]) to match _compute_level_local_out_batched's contract.
        Flash-only (raises otherwise, like the shared local path)."""
        L = int(level)
        if not self.per_level_attn_active or L >= len(self.per_level_attn_heads):
            return None
        if int(window) <= 0:
            return None
        level_slice = self._level_slice_from_offsets(level_offsets, L, int(node_level.numel()))
        if level_slice is not None:
            s, e = level_slice
            idx = None
            n_L = int(e - s)
        else:
            idx = (node_level.to(device=x_normed.device, dtype=torch.long) == L).nonzero(as_tuple=False).view(-1)
            n_L = int(idx.numel())
            s = e = None
        if n_L <= 1:
            return None
        B = int(x_normed.size(0))
        Dh = int(self.local_attn_head_dim)   # attention head_dim (may be up-projected from hidden//num_heads)
        h_L = int(self.per_level_attn_heads[L])
        x_L = x_normed[:, s:e, :] if idx is None else x_normed.index_select(1, idx)  # [B, n_L, hidden]
        q_L = self.q_proj_attn_level[L](x_L).view(B, n_L, h_L, Dh)
        k_L = self.k_proj_attn_level[L](x_L).view(B, n_L, h_L, Dh)
        v_L = self.v_proj_attn_level[L](x_L).view(B, n_L, h_L, Dh)
        if hasattr(self, "rotary_pos_enc_attn") and rope_pos is not None:
            if isinstance(rope_pos, torch.Tensor) and rope_pos.dim() == 2:
                base = rope_pos[s:e] if idx is None else rope_pos.index_select(0, idx)
                pos_L = base.view(1, n_L, base.size(-1)).expand(B, n_L, base.size(-1)).reshape(B * n_L, base.size(-1))
            else:
                base = rope_pos[s:e] if idx is None else rope_pos.index_select(0, idx)
                pos_L = base.view(1, n_L).expand(B, n_L).reshape(-1)
            q_L = self.rotary_pos_enc_attn.apply_rotary_pos_emb(q_L.reshape(B * n_L, h_L, Dh), pos_L).view(B, n_L, h_L, Dh)
            k_L = self.rotary_pos_enc_attn.apply_rotary_pos_emb(k_L.reshape(B * n_L, h_L, Dh), pos_L).view(B, n_L, h_L, Dh)
        resolved_backend, flash_func = pick_attention_backend(x_normed.device)
        if str(backend) != "flash" or resolved_backend not in {"fa2", "fa3"} or flash_func is None:
            raise RuntimeError(
                f"per-level attention dim requires the flash backend for level {L} "
                f"(got backend={backend}, resolved={resolved_backend}). Use attn_backend: flash."
            )
        dropout_p = float(self.dropout.p) if self.training else 0.0
        win = (int(window), 0) if bool(causal) else (int(window), int(window))
        o_L = attention_forward(
            q_L, k_L, v_L, causal=bool(causal), dropout_p=dropout_p, backend=resolved_backend,
            flash_func=flash_func, window_size=win,
            flash_dtype_cast=bool(getattr(self, "local_attn_flash_dtype_cast", False)),
        )
        o_L = self.out_proj_attn_level[L](o_L.reshape(B, n_L, h_L * Dh))  # d_L -> hidden
        return (slice(s, e) if idx is None else idx), o_L

    def _local_pack_log_once(self, msg: str) -> None:
        seen = getattr(self, "_local_pack_logged", None)
        if seen is None:
            seen = set()
            self._local_pack_logged = seen
        if msg not in seen:
            logger.warning("[HMP:PACK] %s", msg)
            seen.add(msg)

    @staticmethod
    def _lse_chunk_eager(
        qc: torch.Tensor, ks: torch.Tensor, pos_v: torch.Tensor, col: torch.Tensor,
        window: int, scale: float,
    ) -> torch.Tensor:
        logits = torch.einsum("bqhd,bshd->bqhs", qc, ks).float() * scale
        valid = (col <= pos_v) & (col >= pos_v - window)  # [cq, S]
        logits = logits.masked_fill(~valid.view(1, qc.size(1), 1, ks.size(1)), float("-inf"))
        return torch.logsumexp(logits, dim=-1)  # [B, cq, H]

    def _lse_chunk_fn(self, device: torch.device):
        """CUDA: torch.compile the chunk kernel (fuses mask+logsumexp, rematerializes the
        fp32 logits in backward instead of saving them). Shapes are skeleton-static, so
        only a handful of graphs compile. Any compile/runtime failure falls back to eager
        permanently (same pattern as hier_refresh_compile)."""
        if device.type != "cuda" or getattr(self, "_lse_compile_failed", False):
            return self._lse_chunk_eager
        fn = getattr(self, "_lse_chunk_compiled", None)
        if fn is None:
            try:
                fn = torch.compile(self._lse_chunk_eager, dynamic=False)
            except Exception as exc:  # pragma: no cover - env-dependent
                self._local_pack_log_once(f"lse compile unavailable ({exc}); eager fallback")
                self._lse_compile_failed = True
                return self._lse_chunk_eager
            self._lse_chunk_compiled = fn
        return fn

    def _packed_window_lse(
        self,
        q_sel: torch.Tensor,   # [B, nq, H, D] queries (post-RoPE), ascending positions
        k_all: torch.Tensor,   # [B, Nk, H, D] keys (post-RoPE, post level-tag)
        q_pos: torch.Tensor,   # [nq] long, query positions in k_all row coordinates (ascending)
        window: int,           # causal sliding window: keys j with q_pos - window <= j <= q_pos
        chunk: int = 256,
        pos_host: Optional[List[int]] = None,  # host copy of q_pos (skeleton-static; avoids syncs)
    ) -> torch.Tensor:
        """Differentiable per-query softmax log-normalizer of a causal sliding-window
        attention call (same key range as flash window_size=(window, 0): W+1 keys incl.
        self). Used by the lane LSE merge; flash's returned LSE is non-differentiable, so
        the merge weights are recomputed here — chunks of consecutive queries share one
        contiguous key SLICE (no gathers), logits in fp32 for a stable logsumexp. Returns
        [B, nq, H]. NOTE: ignores attention-prob dropout (merge weights are computed
        dropout-free; the merged outputs themselves still carry dropout)."""
        B, nq, H, D = q_sel.shape
        scale = 1.0 / math.sqrt(D)
        w = int(window)
        if pos_host is None:
            pos_host = q_pos.tolist()
        chunk_fn = self._lse_chunk_fn(q_sel.device)
        pieces: List[torch.Tensor] = []
        for c0 in range(0, nq, int(chunk)):
            c1 = min(nq, c0 + int(chunk))
            k0 = max(0, int(pos_host[c0]) - w)
            k1 = int(pos_host[c1 - 1]) + 1
            ks = k_all[:, k0:k1]                       # [B, S, H, D] contiguous slice
            col = torch.arange(k0, k1, device=k_all.device).view(1, -1)
            pos_v = q_pos[c0:c1].view(-1, 1)
            try:
                pieces.append(chunk_fn(q_sel[:, c0:c1], ks, pos_v, col, w, scale))
            except Exception as exc:  # pragma: no cover - env-dependent (triton etc.)
                if chunk_fn is self._lse_chunk_eager:
                    raise
                self._local_pack_log_once(f"lse compiled kernel failed ({exc}); eager fallback")
                self._lse_compile_failed = True
                chunk_fn = self._lse_chunk_eager
                pieces.append(chunk_fn(q_sel[:, c0:c1], ks, pos_v, col, w, scale))
        return torch.cat(pieces, dim=1)

    def _flex_union_attn(
        self, qp: torch.Tensor, kp: torch.Tensor, vp: torch.Tensor, spec: Dict
    ) -> torch.Tensor:
        """One flex_attention call over the [coarse | tokens] split layout with the
        block-sparse union mask. Inputs are the mixed-order RoPE'd/tagged q/k/v
        [B, N, H, D]; returns split-order per-head outputs [B, N, H, D] (rows map to
        nodes via spec["flex_query_nodes"]). The BlockMask is content-independent and
        cached in the spec (once per skeleton); the compiled kernel is shared."""
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        perm = spec["flex_perm"]
        bm = spec.get("flex_block_mask")
        if bm is None:
            r = spec["flex_r_mixed"]
            lane = spec["flex_lane_rank"]
            isc = spec["flex_is_coarse"]
            w_mix = int(spec["window"])
            w_lane = int(spec["lane_window"])

            def mask_mod(b, h, qi, ki):
                dr = r[qi] - r[ki]
                band = (dr >= 0) & (dr <= w_mix)
                dl = lane[qi] - lane[ki]
                coarse = isc[qi] & isc[ki] & (dl >= 0) & (dl <= w_lane) & (dr >= 0)
                return band | coarse

            n = int(perm.numel())
            bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n,
                                   device=str(perm.device))
            spec["flex_block_mask"] = bm

        q_s = qp.index_select(1, perm).transpose(1, 2)  # [B, H, N, D]
        k_s = kp.index_select(1, perm).transpose(1, 2)
        v_s = vp.index_select(1, perm).transpose(1, 2)
        # Compiled kernel only for real graphs: the inductor lowering asserts on tiny
        # sequences (< one mask block, e.g. generation-prefix graphs), and eager flex
        # (math composite, O(n^2) but n is tiny there) is fine for those.
        if q_s.is_cuda and int(perm.numel()) >= 512:
            fn = _flex_compiled_singleton()
            # Default tiles exceed the workstation Blackwell's 101KB shared memory.
            out = fn(q_s, k_s, v_s, block_mask=bm,
                     kernel_options={"BLOCK_M": 64, "BLOCK_N": 64})
        else:
            out = flex_attention(q_s, k_s, v_s, block_mask=bm)
        return out.transpose(1, 2)  # [B, N, H, D] split order

    def _apply_local_pack_out(
        self,
        out: torch.Tensor,
        q_pre: Optional[torch.Tensor],
        k_pre: Optional[torch.Tensor],
        v: torch.Tensor,
        spec: Dict,
        source_gates: Optional[Dict[str, torch.Tensor]],
        active_level_set: Optional[set],
        B: int,
        num_nodes: int,
    ) -> set:
        """Packed cross-level local attention: ONE causal sliding-window attention over ALL
        nodes interleaved by close time (model-built spec), so fine queries see recent CLOSED
        coarse summaries and coarse queries see their strictly-past children inside the same
        softmax as their lateral neighbors. Adds the result into `out` for the spec's query
        levels and returns that level set; an empty set means ineligible-here and the caller
        runs the normal per-level path. Leak-free by construction: packed order is close-time
        order, so a causal window only exposes nodes with ar_time <= the query's (equal-time
        ties resolved at spec build per hier_ar_allow_same_time)."""
        query_levels = set(int(l) for l in spec.get("query_levels", (0,)))
        # Anything that changes the meaning of "one causal pass over close-time order"
        # falls back to the per-level path (never silently wrong).
        global_causal_gate = bool(getattr(self, "local_attn_runtime_causal_gate", True))
        l0_causal = bool(getattr(self, "l0_local_runtime_causal", self.l0_local_causal_default))
        pack_causal = True
        if not (global_causal_gate and l0_causal):
            if bool(getattr(self, "local_pack_bidirectional", False)):
                # Bidirectional (MaskGIT/diffusion): nothing to leak, so the packed windows
                # simply become symmetric two-sided (+-window slots) — a node reads past AND
                # future neighbors/coarse summaries, including its own open parent.
                pack_causal = False
                if bool(getattr(self, "local_pack_flex_union", False)) or bool(
                    getattr(self, "local_pack_lane_merge", False)
                ):
                    self._local_pack_log_once(
                        "bidirectional pack: flex_union/lane_merge are causal-only; additive combine"
                    )
            else:
                self._local_pack_log_once("non-causal runtime flags; packed cross-level local skipped")
                return set()
        if getattr(self, "local_attn_runtime_group", None) is not None or bool(
            getattr(self, "local_attn_runtime_sampled", False)
        ):
            self._local_pack_log_once("sampled/grouped local runtime; packed cross-level local skipped")
            return set()
        if self.per_level_attn_active:
            self._local_pack_log_once("per-level local attn dims active; packed cross-level local skipped")
            return set()
        if active_level_set is not None and not query_levels.issubset(active_level_set):
            return set()
        window = int(spec.get("window", 0))
        if window <= 0 or q_pre is None or k_pre is None:
            return set()

        perm = spec["perm"]
        pos = spec["pos"]
        sel = spec["query_sel"]
        tgt = spec["query_nodes"]
        qp = q_pre.index_select(1, perm)
        kp = k_pre.index_select(1, perm)
        vp = v.index_select(1, perm)
        if hasattr(self, "rotary_pos_enc") and pos is not None:
            pos_rep = pos.view(1, num_nodes).expand(B, num_nodes).reshape(-1)
            qp = self.rotary_pos_enc.apply_rotary_pos_emb(
                qp.reshape(B * num_nodes, self.num_heads, self.head_dim), pos_rep
            ).view(B, num_nodes, self.num_heads, self.head_dim)
            kp = self.rotary_pos_enc.apply_rotary_pos_emb(
                kp.reshape(B * num_nodes, self.num_heads, self.head_dim), pos_rep
            ).view(B, num_nodes, self.num_heads, self.head_dim)
        # Level tags (local_pack_level_bias): post-RoPE additive K/V embeddings per source
        # level — the flash-eligible stand-in for the scatter path's level-pair logit bias.
        lvl_packed = spec.get("levels", None)
        if getattr(self, "local_pack_level_bias", False) and lvl_packed is not None:
            kp = kp + self.local_pack_level_k_emb.index_select(0, lvl_packed).unsqueeze(0).to(kp.dtype)
            vp = vp + self.local_pack_level_v_emb.index_select(0, lvl_packed).unsqueeze(0).to(vp.dtype)
        # FLEX UNION (local_pack_flex_union): ONE flex_attention call over the block-
        # coherent split layout replaces the mixed call + lane + merge — the block-sparse
        # mask ("within W mixed slots OR both-coarse within lane_window coarse slots") IS
        # the union softmax, with exact dedup (OR semantics, no +ln2 double count) and no
        # LSE recompute. Same flash-style tiled online softmax, Triton-codegen'd. No
        # attention-prob dropout on this path (flex has no dropout_p). Falls back to the
        # merge/additive flow below on any failure (permanently, log-once).
        if (
            bool(getattr(self, "local_pack_flex_union", False))
            and "flex_perm" in spec
            and pack_causal  # flex mask encodes the causal window; bidi -> additive path
            and not getattr(self, "_flex_union_failed", False)
        ):
            if set(int(l) for l in spec.get("query_levels", ())) >= {0, 1, 2, 3}:
                try:
                    out_flex = self._flex_union_attn(qp, kp, vp, spec)
                    contrib = self.out_proj(out_flex.reshape(B, out_flex.size(1), -1))
                    if source_gates is not None:
                        contrib = source_gates["local"] * contrib
                    out.index_add_(1, spec["flex_query_nodes"], contrib.to(dtype=out.dtype))
                    return query_levels
                except Exception as exc:  # pragma: no cover - env-dependent (triton etc.)
                    self._local_pack_log_once(f"flex union failed ({exc}); merge/additive fallback")
                    # Permanent opt-out only for real-graph failures; a tiny-graph hiccup
                    # (generation prefixes) must not poison the training path.
                    if int(spec["flex_perm"].numel()) >= 512:
                        self._flex_union_failed = True
            else:
                self._local_pack_log_once("flex union needs all levels queried; merge/additive fallback")

        # LSE merge (local_pack_lane_merge): combine the mixed window and the coarse lane
        # for coarse queries as ONE softmax over the union of both key sets, instead of
        # adding two independently-normalized outputs. With Z = exp(lse) per query/head:
        #   merged = (Z_mixed*attn_mixed + Z_lane*attn_lane) / (Z_mixed + Z_lane)
        # which is exactly the softmax over the concatenated key lists (keys in both
        # windows are counted twice — a fixed +ln2 logit bias on the most recent coarse
        # rows). Weights apply per head BEFORE out_proj, so both calls return raw per-head
        # outputs here and the mixed projection is applied explicitly (same ops as inside
        # the helper — bit-identical math).
        lane_merge = (
            bool(getattr(self, "local_pack_lane_merge", False)) and "lane_perm" in spec and pack_causal
        )
        out_pack_raw = None
        if lane_merge:
            out_pack_raw = self._compute_local_attn_from_qkv(
                q_lvl=qp, k_lvl=kp, v_lvl=vp,
                window=window, causal=pack_causal,
                backend=str(spec.get("backend", "sdpa")), level=-1,
                skip_out_proj=True,
            )
            if out_pack_raw is None:
                return set()
            out_pack = self.out_proj(out_pack_raw.reshape(B, num_nodes, -1))
        else:
            out_pack = self._compute_local_attn_from_qkv(
                q_lvl=qp, k_lvl=kp, v_lvl=vp,
                window=window, causal=pack_causal,
                backend=str(spec.get("backend", "sdpa")), level=-1,
            )
            if out_pack is None:
                return set()
        contrib = out_pack.index_select(1, sel)
        if source_gates is not None:
            contrib = source_gates["local"] * contrib
        out.index_add_(1, tgt, contrib.to(dtype=out.dtype))

        # Coarse lane: second packed window (causal in AR mode, two-sided in bidi) over ONLY
        # the coarse rows (close-time order preserved by construction), restoring wide coarse
        # LATERAL reach at coarse density prices — additive to the main-window read above.
        lane_perm = spec.get("lane_perm", None)
        if lane_perm is not None:
            n_lane = int(lane_perm.numel())
            ql = q_pre.index_select(1, lane_perm)
            kl = k_pre.index_select(1, lane_perm)
            vl = v.index_select(1, lane_perm)
            lane_pos = spec.get("lane_pos", None)
            if hasattr(self, "rotary_pos_enc") and lane_pos is not None:
                lane_pos_rep = lane_pos.view(1, n_lane).expand(B, n_lane).reshape(-1)
                ql = self.rotary_pos_enc.apply_rotary_pos_emb(
                    ql.reshape(B * n_lane, self.num_heads, self.head_dim), lane_pos_rep
                ).view(B, n_lane, self.num_heads, self.head_dim)
                kl = self.rotary_pos_enc.apply_rotary_pos_emb(
                    kl.reshape(B * n_lane, self.num_heads, self.head_dim), lane_pos_rep
                ).view(B, n_lane, self.num_heads, self.head_dim)
            lane_levels = spec.get("lane_levels", None)
            if getattr(self, "local_pack_level_bias", False) and lane_levels is not None:
                kl = kl + self.local_pack_level_k_emb.index_select(0, lane_levels).unsqueeze(0).to(kl.dtype)
                vl = vl + self.local_pack_level_v_emb.index_select(0, lane_levels).unsqueeze(0).to(vl.dtype)
            lane_window = int(spec.get("lane_window", 0))
            out_lane = self._compute_local_attn_from_qkv(
                q_lvl=ql, k_lvl=kl, v_lvl=vl,
                window=lane_window, causal=pack_causal,
                backend=str(spec.get("backend", "sdpa")), level=-2,
                skip_out_proj=lane_merge,
            )
            if out_lane is not None and lane_merge:
                # out already holds the mixed contribution for the lane queries, so write
                # the DELTA to the merged result: merged - mixed = (1-w)*(attn_lane -
                # attn_mixed) with w = Z_mixed/(Z_mixed+Z_lane) = sigmoid(lse_m - lse_l).
                # out_proj's bias cancels in the difference -> bias-free linear.
                lane_q_packed = spec["lane_query_packed_rows"]  # lane queries' rows in the MIXED packing
                lane_q_sel = spec["lane_query_sel"]             # lane queries' rows in the LANE packing
                raw_m = out_pack_raw.index_select(1, lane_q_packed)   # [B, nq, H, D]
                raw_l = out_lane.index_select(1, lane_q_sel)          # [B, nq, H, D]
                # Chunk size trades slice overcompute (S ~ stride*chunk + W grows with
                # chunk) against kernel launches (chunks = nq/chunk). Measured on the
                # Blackwell at 8196 (tok/s): 64=79.9 256=101.2 512=108.1 1024=108.8
                # 4096=104.3 — launch-bound until ~1k slices, then overcompute bites.
                lse_m = self._packed_window_lse(
                    qp.index_select(1, lane_q_packed), kp, lane_q_packed, window,
                    chunk=1024, pos_host=spec.get("lane_query_packed_rows_host"),
                )
                lse_l = self._packed_window_lse(
                    ql.index_select(1, lane_q_sel), kl, lane_q_sel, lane_window,
                    chunk=1024, pos_host=spec.get("lane_query_sel_host"),
                )
                w_m = torch.sigmoid(lse_m - lse_l).unsqueeze(-1).to(raw_m.dtype)  # [B,nq,H,1]
                delta = ((1.0 - w_m) * (raw_l - raw_m)).reshape(B, int(lane_q_sel.numel()), -1)
                delta = F.linear(delta, self.out_proj.weight)
                if source_gates is not None:
                    delta = source_gates["local"] * delta
                out.index_add_(1, spec["lane_query_nodes"], delta.to(dtype=out.dtype))
            elif out_lane is not None:
                lane_contrib = out_lane.index_select(1, spec["lane_query_sel"])
                if source_gates is not None:
                    lane_contrib = source_gates["local"] * lane_contrib
                out.index_add_(1, spec["lane_query_nodes"], lane_contrib.to(dtype=out.dtype))
        return query_levels

    def _compute_level_local_out_batched(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        node_level: torch.Tensor,
        level: int,
        window: int,
        causal: bool,
        backend: str,
        node_group: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        level_offsets: Optional[torch.Tensor] = None,
    ):
        """Compute local window attention for nodes at a specific level.

        Generalizes _compute_l0_local_out_batched to any level.
        Returns (level_idx_tensor, projected_output) or None.
        """
        level_slice = self._level_slice_from_offsets(level_offsets, int(level), int(node_level.numel()))
        lvl_idx = None
        if level_slice is None:
            lvl_idx = torch.nonzero(node_level == level, as_tuple=False).view(-1)
            token_count = int(lvl_idx.numel())
        else:
            token_count = int(level_slice[1] - level_slice[0])
        if token_count <= 1:
            return None
        if window <= 0 and not causal:
            return None

        level_grid_shape = self._resolve_level_grid_shape(level)
        spatial_metric = str(getattr(self, "local_attn_runtime_spatial_metric", "chebyshev")).lower()
        role_bias_enabled = bool(getattr(self, "local_attn_level_role_bias_enable", True))

        group_vec = None
        if node_group is not None and isinstance(node_group, torch.Tensor):
            if node_group.numel() == node_level.numel():
                group_vec = node_group.to(device=node_level.device, dtype=torch.long)

        pos_vec = None
        if positions is not None and isinstance(positions, torch.Tensor):
            if positions.dim() >= 1 and positions.size(0) == node_level.numel():
                pos_vec = positions.to(device=node_level.device, dtype=torch.long)

        sampled_mode = str(getattr(self, "local_attn_sampled_mode", "safe_sdpa")).lower()
        sampled_runtime = bool(getattr(self, "local_attn_runtime_sampled", False))
        if sampled_mode not in {"safe_sdpa", "flash_sorted", "off"}:
            sampled_mode = "safe_sdpa"
        if sampled_runtime and sampled_mode == "flash_sorted" and pos_vec is not None and pos_vec.dim() == 1:
            # In sampled sequence mode, FlashAttention is made correct by sorting
            # per-level/per-group indices by real position. Do not treat cached
            # full-graph grid metadata as a dense spatial mask for partial batches.
            level_grid_shape = None

        use_2d_spatial = bool(
            int(window) > 0
            and (
                level_grid_shape is not None
                or (pos_vec is not None and pos_vec.dim() >= 2 and int(pos_vec.size(-1)) >= 2)
            )
        )
        if sampled_runtime and sampled_mode == "off":
            return None
        path_tag = "2d-spatial" if use_2d_spatial else "1d"
        if role_bias_enabled:
            path_tag = f"{path_tag}-levelrole"
        if group_vec is not None:
            path_tag = f"{path_tag}-grouped"
        if sampled_runtime:
            path_tag = f"{path_tag}-sampled-{sampled_mode}"
        log_key = (
            int(level),
            int(window),
            bool(causal),
            str(backend),
            path_tag,
            tuple(level_grid_shape) if level_grid_shape is not None else None,
            bool(role_bias_enabled),
        )
        if log_key not in self._local_attn_runtime_logged_keys:
            if level_grid_shape is not None and int(window) > 0:
                logger.info(
                    "[HMP:LOCAL] level=%d path=%s backend=%s window=%d causal=%s grid=%dx%d metric=%s role_bias=%s tokens=%d",
                    int(level),
                    path_tag,
                    str(backend),
                    int(window),
                    bool(causal),
                    int(level_grid_shape[0]),
                    int(level_grid_shape[1]),
                    spatial_metric,
                    bool(role_bias_enabled),
                    int(token_count),
                )
            else:
                logger.info(
                    "[HMP:LOCAL] level=%d path=%s backend=%s window=%d causal=%s role_bias=%s tokens=%d",
                    int(level),
                    path_tag,
                    str(backend),
                    int(window),
                    bool(causal),
                    bool(role_bias_enabled),
                    int(token_count),
                )
            self._local_attn_runtime_logged_keys.add(log_key)

        if group_vec is None:
            can_use_slice = level_slice is not None and not (sampled_runtime and sampled_mode == "flash_sorted")
            if can_use_slice:
                start, end = level_slice
                q_lvl = q[:, start:end, :, :]
                k_lvl = k[:, start:end, :, :]
                v_lvl = v[:, start:end, :, :]
                attn_bias = None
                force_sdpa = False
                backend_eff = backend
                if sampled_runtime and sampled_mode == "safe_sdpa":
                    backend_eff = "sdpa"
                    force_sdpa = True
                if backend_eff == "flash" and int(window) > 0 and (level_grid_shape is not None or role_bias_enabled):
                    raise RuntimeError(
                        f"Flash local attention for level {int(level)} cannot use spatial/role-bias masks because they require dense T x T bias tensors. "
                        "Disable local attention level role bias/spatial local masks, or use --l0_local_backend pyg."
                    )
                if int(window) > 0 and (level_grid_shape is not None or role_bias_enabled or force_sdpa):
                    local_pos = pos_vec[start:end] if pos_vec is not None else torch.arange(token_count, device=q.device, dtype=torch.long)
                    bias_grid_shape = level_grid_shape if (level_grid_shape is not None and int(level_grid_shape[0]) * int(level_grid_shape[1]) == token_count) else None
                    attn_bias = self._build_level_aware_local_attn_bias(
                        local_pos=local_pos,
                        grid_shape=bias_grid_shape,
                        window=int(window),
                        causal=bool(causal),
                        metric=spatial_metric,
                        level=int(level),
                        device=q.device,
                        dtype=q.dtype,
                    )
                    force_sdpa = True
                out_lvl = self._compute_local_attn_from_qkv(
                    q_lvl=q_lvl,
                    k_lvl=k_lvl,
                    v_lvl=v_lvl,
                    window=window,
                    causal=causal,
                    backend=backend_eff,
                    level=level,
                    attn_bias=attn_bias,
                    force_sdpa=force_sdpa,
                )
                if out_lvl is None:
                    return None
                return slice(start, end), out_lvl

            if lvl_idx is None:
                lvl_idx = torch.arange(level_slice[0], level_slice[1], device=node_level.device, dtype=torch.long)
            if sampled_runtime and sampled_mode == "flash_sorted" and pos_vec is not None and pos_vec.dim() == 1:
                order = torch.argsort(pos_vec[lvl_idx], stable=True)
                lvl_idx = lvl_idx[order]
            attn_bias = None
            force_sdpa = False
            backend_eff = backend
            if sampled_runtime and sampled_mode == "safe_sdpa":
                backend_eff = "sdpa"
                force_sdpa = True
            if backend_eff == "flash" and int(window) > 0 and (level_grid_shape is not None or role_bias_enabled):
                raise RuntimeError(
                    f"Flash local attention for level {int(level)} cannot use spatial/role-bias masks because they require dense T x T bias tensors. "
                    "Disable local attention level role bias/spatial local masks, or use --l0_local_backend pyg."
                )
            if int(window) > 0 and (level_grid_shape is not None or role_bias_enabled or force_sdpa):
                local_pos = pos_vec[lvl_idx] if pos_vec is not None else torch.arange(lvl_idx.numel(), device=lvl_idx.device, dtype=torch.long)
                bias_grid_shape = level_grid_shape if (level_grid_shape is not None and int(level_grid_shape[0]) * int(level_grid_shape[1]) == int(lvl_idx.numel())) else None
                attn_bias = self._build_level_aware_local_attn_bias(
                    local_pos=local_pos,
                    grid_shape=bias_grid_shape,
                    window=int(window),
                    causal=bool(causal),
                    metric=spatial_metric,
                    level=int(level),
                    device=q.device,
                    dtype=q.dtype,
                )
                force_sdpa = True
            out_lvl = self._compute_local_attn_for_indices(
                q=q,
                k=k,
                v=v,
                idx=lvl_idx,
                window=window,
                causal=causal,
                backend=backend_eff,
                level=level,
                attn_bias=attn_bias,
                force_sdpa=force_sdpa,
            )
            if out_lvl is None:
                return None
            return lvl_idx, out_lvl

        if lvl_idx is None:
            lvl_idx = torch.arange(level_slice[0], level_slice[1], device=node_level.device, dtype=torch.long)
        lvl_groups = group_vec[lvl_idx]
        unique_groups = torch.unique(lvl_groups)

        if (
            sampled_runtime
            and sampled_mode == "flash_sorted"
            and backend == "flash"
            and pos_vec is not None
            and pos_vec.dim() == 1
            and level_grid_shape is None
            and not role_bias_enabled
            and q.dim() == 4
            and int(q.size(0)) == 1
            and unique_groups.numel() > 1
            and int(window) > 0
        ):
            group_rows_varlen = []
            group_lengths_varlen = []
            for gid in unique_groups:
                grp_idx = lvl_idx[lvl_groups == gid]
                if grp_idx.numel() <= 1:
                    continue
                order = torch.argsort(pos_vec[grp_idx], stable=True)
                grp_idx = grp_idx[order]
                group_rows_varlen.append(grp_idx)
                group_lengths_varlen.append(int(grp_idx.numel()))
            if group_rows_varlen:
                try:
                    from flash_attn import flash_attn_varlen_func

                    flat_idx_varlen = torch.cat(group_rows_varlen, dim=0)
                    lengths = torch.as_tensor(group_lengths_varlen, device=q.device, dtype=torch.int32)
                    cu_seqlens = torch.empty(lengths.numel() + 1, device=q.device, dtype=torch.int32)
                    cu_seqlens[0] = 0
                    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
                    max_seqlen = int(lengths.max().detach().item())
                    dropout_p = float(self.dropout.p) if self.training else 0.0
                    q_var = q[0].index_select(0, flat_idx_varlen)
                    k_var = k[0].index_select(0, flat_idx_varlen)
                    v_var = v[0].index_select(0, flat_idx_varlen)
                    orig_dtype = q_var.dtype
                    if orig_dtype not in (torch.float16, torch.bfloat16):
                        if bool(getattr(self, "local_attn_flash_dtype_cast", False)):
                            work_dtype = torch.bfloat16 if bf16_supported() else torch.float16
                            q_var = q_var.to(dtype=work_dtype)
                            k_var = k_var.to(dtype=work_dtype)
                            v_var = v_var.to(dtype=work_dtype)
                        else:
                            raise RuntimeError("FlashAttention varlen requires fp16/bf16 inputs")
                    win = (int(window), 0) if causal else (int(window), int(window))
                    out_var = flash_attn_varlen_func(
                        q_var.contiguous(),
                        k_var.contiguous(),
                        v_var.contiguous(),
                        cu_seqlens,
                        cu_seqlens,
                        max_seqlen,
                        max_seqlen,
                        dropout_p=dropout_p,
                        causal=bool(causal),
                        window_size=win,
                    )
                    out_var = _unwrap_flash_result(out_var)
                    if out_var.dtype != orig_dtype:
                        out_var = out_var.to(dtype=orig_dtype)
                    out_var = out_var.reshape(1, flat_idx_varlen.numel(), -1)
                    out_var = self.out_proj(out_var)
                    self._last_local_attn_sampled_fast_path = "varlen"
                    return flat_idx_varlen, out_var
                except Exception:
                    pass

            group_rows = []
            group_lengths = []
            for gid in unique_groups:
                grp_idx = lvl_idx[lvl_groups == gid]
                if grp_idx.numel() <= 1:
                    group_rows = []
                    break
                order = torch.argsort(pos_vec[grp_idx], stable=True)
                grp_idx = grp_idx[order]
                group_rows.append(grp_idx)
                group_lengths.append(int(grp_idx.numel()))
            if group_rows and len(set(group_lengths)) == 1:
                dropout_p = float(self.dropout.p) if self.training else 0.0
                resolved_backend, flash_attn_func = pick_attention_backend(q.device)
                flash_supports_dropout = bool(
                    resolved_backend in {"fa2", "fa3"}
                    and flash_attn_func is not None
                    and _flash_attn_supports_dropout(flash_attn_func)
                )
                if (
                    resolved_backend in {"fa2", "fa3"}
                    and flash_attn_func is not None
                    and (dropout_p <= 0.0 or flash_supports_dropout)
                ):
                    packed_idx = torch.stack(group_rows, dim=0)
                    flat_idx = packed_idx.reshape(-1)
                    q_pack = q[0].index_select(0, flat_idx).view(
                        packed_idx.size(0),
                        packed_idx.size(1),
                        q.size(2),
                        q.size(3),
                    )
                    k_pack = k[0].index_select(0, flat_idx).view_as(q_pack)
                    v_pack = v[0].index_select(0, flat_idx).view_as(q_pack)
                    win = (int(window), 0) if causal else (int(window), int(window))
                    out_pack = attention_forward(
                        q_pack,
                        k_pack,
                        v_pack,
                        causal=causal,
                        dropout_p=dropout_p,
                        backend=resolved_backend,
                        flash_func=flash_attn_func,
                        window_size=win,
                        flash_dtype_cast=bool(getattr(self, "local_attn_flash_dtype_cast", False)),
                    )
                    out_pack = out_pack.reshape(packed_idx.size(0), packed_idx.size(1), -1)
                    out_pack = self.out_proj(out_pack).reshape(1, flat_idx.numel(), self.hidden_dim)
                    self._last_local_attn_sampled_fast_path = "equal"
                    return flat_idx, out_pack

        if sampled_runtime and sampled_mode == "flash_sorted" and group_vec is not None:
            self._last_local_attn_sampled_fast_path = "loop"
        out_parts = []
        idx_parts = []
        for gid in unique_groups:
            mask = (lvl_groups == gid)
            if not bool(mask.any()):
                continue
            grp_pos = torch.nonzero(mask, as_tuple=False).view(-1)
            if grp_pos.numel() <= 1:
                continue
            grp_idx = lvl_idx[grp_pos]
            if sampled_runtime and sampled_mode == "flash_sorted" and pos_vec is not None and pos_vec.dim() == 1:
                order = torch.argsort(pos_vec[grp_idx], stable=True)
                grp_idx = grp_idx[order]
            attn_bias = None
            force_sdpa = False
            backend_eff = backend
            if sampled_runtime and sampled_mode == "safe_sdpa":
                backend_eff = "sdpa"
                force_sdpa = True
            if backend_eff == "flash" and int(window) > 0 and (level_grid_shape is not None or role_bias_enabled):
                raise RuntimeError(
                    f"Flash local attention for level {int(level)} cannot use grouped spatial/role-bias masks because they require dense T x T bias tensors. "
                    "Disable local attention level role bias/spatial local masks, or use --l0_local_backend pyg."
                )
            if int(window) > 0 and (level_grid_shape is not None or role_bias_enabled or force_sdpa):
                local_pos = pos_vec[grp_idx] if pos_vec is not None else torch.arange(grp_idx.numel(), device=grp_idx.device, dtype=torch.long)
                bias_grid_shape = level_grid_shape if (level_grid_shape is not None and int(level_grid_shape[0]) * int(level_grid_shape[1]) == int(grp_idx.numel())) else None
                attn_bias = self._build_level_aware_local_attn_bias(
                    local_pos=local_pos,
                    grid_shape=bias_grid_shape,
                    window=int(window),
                    causal=bool(causal),
                    metric=spatial_metric,
                    level=int(level),
                    device=q.device,
                    dtype=q.dtype,
                )
                force_sdpa = True
            grp_out = self._compute_local_attn_for_indices(
                q=q,
                k=k,
                v=v,
                idx=grp_idx,
                window=window,
                causal=causal,
                backend=backend_eff,
                level=level,
                attn_bias=attn_bias,
                force_sdpa=force_sdpa,
            )
            if grp_out is None:
                continue
            idx_parts.append(grp_idx)
            out_parts.append(grp_out)

        if not idx_parts:
            return None

        if len(idx_parts) == 1:
            return idx_parts[0], out_parts[0]

        return torch.cat(idx_parts, dim=0), torch.cat(out_parts, dim=1)


    

    
    

    def message(self, q_i, k_j, v_j, level_emb_i, level_emb_j, edge_attr=None, edge_type=None, node_hidden_i=None, node_hidden_j=None, index=None):
        """
        Compute messages with hierarchical attention.
        
        Args:
            q_i: Query vectors for target nodes
            k_j: Key vectors for source nodes
            v_j: Value vectors for source nodes
            level_emb_i: Level embeddings for target nodes
            level_emb_j: Level embeddings for source nodes
            edge_attr: Optional edge attributes
            index: Target node indices
            
        Returns:
            messages: Attention-weighted messages
        """
        # Compute attention scores
        # [num_edges, num_heads, head_dim] * [num_edges, num_heads, head_dim] -> [num_edges, num_heads]
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
        #attn = torch.clamp(attn, min=-1e3, max=1e3)
        # Apply level-based attention adjustment
        # Concatenate level embeddings and compute level attention
        level_concat = torch.cat([level_emb_i, level_emb_j], dim=-1)  # [num_edges, level_dim*2]
        level_weights = self.level_attn(level_concat)  # [num_edges, num_heads]
        #level_weights = torch.clamp(level_weights, min=-1e3, max=1e3)
        # Scale level weights based on level difference
        level_diff = torch.abs(level_emb_i[:, 0:1] - level_emb_j[:, 0:1])
        level_scale = 1.0 / (1.0 + level_diff)  # Higher weight for same level
        
        # Apply level weights to attention
        attn = attn + level_weights * level_scale
        
        # Apply edge attributes if available
        if edge_attr is not None and self.use_edge_attr:
            if edge_attr.dim() == 1 or (edge_attr.dim() == 2 and edge_attr.size(-1) == 1):
                edge_scalar = edge_attr.view(-1, 1)
                attn = attn + edge_scalar
            elif self.edge_dim is not None and edge_attr.dim() == 2 and edge_attr.size(-1) == self.edge_dim:
                edge_features = self.edge_proj(edge_attr).view(-1, self.num_heads, self.head_dim)
                edge_attn = (q_i * edge_features).sum(dim=-1) / math.sqrt(self.head_dim)
                attn = attn + edge_attn

        edge_logit_bias, edge_value_gate = self._edge_condition(
            edge_type,
            attn,
            src_hidden=node_hidden_j,
            dst_hidden=node_hidden_i,
        )
        attn = self._apply_edge_logit_bias(attn, edge_logit_bias)
        
        attn = self._apply_graph_trace_bias(attn, attn.size(0))
        # Normalize with PyG's efficient softmax (per target node)
        attn = softmax(attn, index)
        if index is not None and index.numel() > 0:
            trace_num_nodes = int(index.max().detach().item()) + 1
            self._update_graph_edge_trace(attn, index, trace_num_nodes)
        # use torch softmax for better performance
        #attn = torch.softmax(attn, index)

        # ---------- derive scalar edge weights from multi-head attn (pre-dropout) ----------
        if self.learn_edge_from_attn:
            # "importance": per-head attention itself
            # "confidence" proxy: head peakedness per edge (max over heads)
            conf = attn.max(dim=-1, keepdim=True).values  # [E, 1]
            effective = attn * conf                       # [E, H]

            # strongly recommended to avoid a shortcut channel; remove if you *want* coupling
            #effective = effective.detach()

            new_edge_attr = self.edge_combine(effective).squeeze(-1)  # [E]
            # stabilize scale for routing into next layer
            new_edge_attr = torch.tanh(new_edge_attr)
            self._edge_attr_stack[-1] = new_edge_attr
        else:
            self._edge_attr_stack[-1] = None

        attn = self.dropout(attn)


        # Apply attention to values
        v_j = self._apply_edge_value_gate(v_j, edge_value_gate)
        return v_j * attn.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
    
    def message_edge_graphtransformer(self, q_i, k_j, v_j, level_emb_i, level_emb_j, edge_attr=None, index=None): #HT_message
        """
        Compute messages with hierarchical attention.
        
        Args:
            q_i: Query vectors for target nodes
            k_j: Key vectors for source nodes
            v_j: Value vectors for source nodes
            level_emb_i: Level embeddings for target nodes
            level_emb_j: Level embeddings for source nodes
            edge_attr: Optional edge attributes
            index: Target node indices
            
        Returns:
            messages: Attention-weighted messages
        """
        #print("Using HT_message function with edge attributes added to keys and values.")
        # Add edge attributes to keys and values
        if edge_attr is not None and self.edge_dim is not None and self.use_edge_attr:
            edge_features = self.edge_proj(edge_attr).view(-1, self.num_heads, self.head_dim)
            k_j = k_j + edge_features # Add to keys
            v_j = v_j + edge_features # Add to values (optional, but TransformerConv does it)
        # Compute attention scores
        # [num_edges, num_heads, head_dim] * [num_edges, num_heads, head_dim] -> [num_edges, num_heads]
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Apply level-based attention adjustment
        # Concatenate level embeddings and compute level attention
        level_concat = torch.cat([level_emb_i, level_emb_j], dim=-1)  # [num_edges, level_dim*2]
        level_weights = self.level_attn(level_concat)  # [num_edges, num_heads]
        
        # Scale level weights based on level difference
        level_diff = torch.abs(level_emb_i[:, 0:1] - level_emb_j[:, 0:1])
        level_scale = 1.0 / (1.0 + level_diff)  # Higher weight for same level
        
        # Apply level weights to attention
        attn = attn + level_weights * level_scale
        
        # # Apply edge attributes if available
        # if edge_attr is not None and self.edge_dim is not None and self.use_edge_attr:
        #     edge_features = self.edge_proj(edge_attr).view(-1, self.num_heads, self.head_dim)
        #     edge_attn = (q_i * edge_features).sum(dim=-1) / math.sqrt(self.head_dim)
        #     attn = attn + edge_attn
        
        # Normalize with PyG's efficient softmax (per target node)
        attn = softmax(attn, index)
        # use torch softmax for better performance
        #attn = torch.softmax(attn, index)
        attn = self.dropout(attn)
        
        # Apply attention to values
        return v_j * attn.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
    
    
    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregate messages using PyG's efficient scatter operations.
        
        Args:
            inputs: Messages to aggregate
            index: Target node indices
            dim_size: Size of output dimension
            
        Returns:
            aggregated: Aggregated messages
        """
        # Sum messages for each target node
        return scatter_add(inputs, index, dim=self.node_dim, dim_size=dim_size)



class HierarchicalTransformerLayer(nn.Module):
    """
    Hierarchical Transformer Layer that combines message passing with residual connections.
    
    This layer applies hierarchical message passing followed by feed-forward network,
    similar to a standard Transformer layer but with hierarchical awareness.
    """
    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        dropout=0.1,
        edge_dim=None,
        use_edge_attr=True,
        learn_edge_from_attn: bool = True,
        max_seq_len=131072,
        l0_local_backend: str = "pyg",
        l0_local_window: int = 0,
        l0_local_causal_default: bool = False,
        use_beta_gating: bool = False,
        make_beta_gating: bool = False,
        local_attn_config: Optional[Dict] = None,
        local_attn_level_role_bias_enable: bool = True,
        local_attn_level_role_bias_scale: float = 1.0,
        local_attn_flash_dtype_cast: bool = False,
        cross_level_packed: bool = False,
        cross_level_qkv: str = "shared",
        local_attn_sampled_mode: str = "safe_sdpa",
        sparse_attn_mode: str = "off",
        sparse_attn_chunk_size: int = 0,
        per_level_local_qkv: bool = False,
        num_local_levels: int = 4,
        per_level_attn_mult: Optional[List[float]] = None,  # per-level local-attn dim mult (scales num_heads; head_dim fixed)
        local_attn_head_dim: int = 0,  # 0 = hidden//num_heads; >0 = up/down-project local attn to this head_dim
        local_pack_level_bias: bool = False,  # per-level per-head K/V tags for the packed mixed-level local call
        local_pack_lane_merge: bool = False,  # LSE-merge the mixed window + coarse lane (unified softmax)
        local_pack_flex_union: bool = False,  # ONE flex_attention call w/ block-sparse union mask
        local_pack_bidirectional: bool = False,  # bidi (MaskGIT/diffusion): two-sided packed windows
        hqd_read_prerope: bool = False,   # xq/HQD fetch read + stage-3 score on PRE-RoPE q/k (content-only)
        hqd_read_sink: bool = False,      # learned zero-value sink slot in the packed fetch read softmax
        per_level_ffn_dims: Optional[List[int]] = None,  # per-level FFN "processing dim" (inner ~ mult*dim); [] => uniform
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        rope_level_axis_enable: bool = False,
        rope_level_axis_scale: float = 32.0,
        rope_mode: str = "auto",
        attention_source_gating_enable: bool = False,
        attention_source_gate_init_graph: float = 1.0,
        attention_source_gate_init_local: float = 0.5,
        attention_source_gate_init_hqd: float = 0.1,
        attention_source_gate_debug: bool = False,
        lateral_edge_trace_enable: bool = False,
        lateral_edge_trace_mode: str = "windowed_approx",
        lateral_edge_trace_decay: float = 0.95,
        lateral_edge_trace_eta: float = 0.02,
        lateral_edge_trace_alpha: float = 0.25,
        lateral_edge_trace_max: float = 2.0,
        lateral_edge_trace_per_head: bool = True,
        lateral_edge_trace_credit: str = "attn",
        lateral_edge_trace_center_per_dst: bool = True,
        lateral_edge_trace_update_during_eval: bool = False,
        lateral_edge_trace_detach: bool = True,
        lateral_edge_trace_debug: bool = False,
        edge_conditioning_enable: bool = False,
        edge_type_generator_enable: bool = False,
        edge_type_embedding_dim: int = 32,
        edge_condition_hidden_dim: int = 64,
        edge_condition_num_types: int = 16,
        edge_logit_bias_enable: bool = True,
        edge_value_gate_enable: bool = True,
        edge_logit_bias_per_head: bool = True,
        edge_value_gate_per_head: bool = False,
        edge_value_gate_per_channel: bool = True,
        edge_gate_init_identity: bool = True,
        edge_logit_bias_init_zero: bool = True,
        edge_condition_dropout: float = 0.0,
        edge_condition_debug: bool = False,
        edge_node_condition_enable: bool = False,
        edge_node_condition_detach: bool = True,
        edge_node_condition_dim: int = 32,
        edge_node_condition_mode: str = "src_dst_prod",
        edge_node_condition_zero_init: bool = True,
        edge_gate_scale: float = 0.1,
        attn_sparse_qk_mult: float = 1.0,
        attn_sparse_v_mult: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_beta_gating = use_beta_gating
        self.make_beta_gating = make_beta_gating
        self.norm_type = "rmsnorm" if norm_type is None else str(norm_type).lower()
        self.norm_eps = float(norm_eps)
        # Hierarchical message passing for attention
        self.message_passing = HierarchicalMessagePassing(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            use_edge_attr=use_edge_attr,
            learn_edge_from_attn=learn_edge_from_attn,
            max_seq_len=self.max_seq_len,
            l0_local_backend=l0_local_backend,
            l0_local_window=l0_local_window,
            l0_local_causal_default=l0_local_causal_default,
            local_attn_config=local_attn_config,
            local_attn_level_role_bias_enable=local_attn_level_role_bias_enable,
            local_attn_level_role_bias_scale=local_attn_level_role_bias_scale,
            local_attn_flash_dtype_cast=local_attn_flash_dtype_cast,
            cross_level_packed=cross_level_packed,
            cross_level_qkv=cross_level_qkv,
            local_attn_sampled_mode=local_attn_sampled_mode,
            sparse_attn_mode=sparse_attn_mode,
            sparse_attn_chunk_size=sparse_attn_chunk_size,
            per_level_local_qkv=per_level_local_qkv,
            num_local_levels=num_local_levels,
            per_level_attn_mult=per_level_attn_mult,
            local_attn_head_dim=local_attn_head_dim,
            local_pack_level_bias=local_pack_level_bias,
            local_pack_lane_merge=local_pack_lane_merge,
            local_pack_flex_union=local_pack_flex_union,
            local_pack_bidirectional=local_pack_bidirectional,
            hqd_read_prerope=hqd_read_prerope,
            hqd_read_sink=hqd_read_sink,
            norm_type=norm_type,
            norm_eps=norm_eps,
            rope_level_axis_enable=rope_level_axis_enable,
            rope_level_axis_scale=rope_level_axis_scale,
            rope_mode=rope_mode,
            attention_source_gating_enable=attention_source_gating_enable,
            attention_source_gate_init_graph=attention_source_gate_init_graph,
            attention_source_gate_init_local=attention_source_gate_init_local,
            attention_source_gate_init_hqd=attention_source_gate_init_hqd,
            attention_source_gate_debug=attention_source_gate_debug,
            lateral_edge_trace_enable=lateral_edge_trace_enable,
            lateral_edge_trace_mode=lateral_edge_trace_mode,
            lateral_edge_trace_decay=lateral_edge_trace_decay,
            lateral_edge_trace_eta=lateral_edge_trace_eta,
            lateral_edge_trace_alpha=lateral_edge_trace_alpha,
            lateral_edge_trace_max=lateral_edge_trace_max,
            lateral_edge_trace_per_head=lateral_edge_trace_per_head,
            lateral_edge_trace_credit=lateral_edge_trace_credit,
            lateral_edge_trace_center_per_dst=lateral_edge_trace_center_per_dst,
            lateral_edge_trace_update_during_eval=lateral_edge_trace_update_during_eval,
            lateral_edge_trace_detach=lateral_edge_trace_detach,
            lateral_edge_trace_debug=lateral_edge_trace_debug,
            edge_conditioning_enable=edge_conditioning_enable,
            edge_type_generator_enable=edge_type_generator_enable,
            edge_type_embedding_dim=edge_type_embedding_dim,
            edge_condition_hidden_dim=edge_condition_hidden_dim,
            edge_condition_num_types=edge_condition_num_types,
            edge_logit_bias_enable=edge_logit_bias_enable,
            edge_value_gate_enable=edge_value_gate_enable,
            edge_logit_bias_per_head=edge_logit_bias_per_head,
            edge_value_gate_per_head=edge_value_gate_per_head,
            edge_value_gate_per_channel=edge_value_gate_per_channel,
            edge_gate_init_identity=edge_gate_init_identity,
            edge_logit_bias_init_zero=edge_logit_bias_init_zero,
            edge_condition_dropout=edge_condition_dropout,
            edge_condition_debug=edge_condition_debug,
            edge_node_condition_enable=edge_node_condition_enable,
            edge_node_condition_detach=edge_node_condition_detach,
            edge_node_condition_dim=edge_node_condition_dim,
            edge_node_condition_mode=edge_node_condition_mode,
            edge_node_condition_zero_init=edge_node_condition_zero_init,
            edge_gate_scale=edge_gate_scale,
            attn_sparse_qk_mult=attn_sparse_qk_mult,
            attn_sparse_v_mult=attn_sparse_v_mult,
        )

        class SwiGLUFFN(nn.Module):
            """
            LLaMA/GPT-OSS style SwiGLU feed-forward:
            y = Dropout( down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) ) )

            Args:
                hidden_dim: model dimension
                dropout: dropout prob applied after the down projection
                ffn_mult: base expansion factor (default 4.0 like standard FFN)
                swiglu_factor: multiplicative factor to reduce inner dim for SwiGLU
                            (use 2/3 to match LLaMA param budget; use 1.0 for 4x drop-in)
                multiple_of: round inner dim up to this multiple (LLaMA rounds to e.g. 256)
                bias: set False to match LLaMA; True if you prefer biases
            """
            def __init__(self,
                        hidden_dim: int,
                        dropout: float = 0.0,
                        ffn_mult: float = 4.0,#4.0,2.0,   # base expansion factor before applying swiglu_factor
                        swiglu_factor: float = 2/3,   # LLaMA-style
                        multiple_of: int = 256,
                        bias: bool = False,
                        inner_dim: Optional[int] = None):  # per-level override of the SwiGLU inner dim
                super().__init__()
                # inner_dim override lets each hierarchy level pick its own FFN capacity (small
                # for L0 = cheap over many nodes; large for coarse levels = capacity over few).
                # The input/output stay hidden_dim so the residual stream + cross-level attention
                # are unaffected; only the expansion width is per-level.
                if inner_dim is not None and int(inner_dim) > 0:
                    inner = int(inner_dim)
                else:
                    inner = int(ffn_mult * swiglu_factor * hidden_dim)
                if multiple_of is not None and multiple_of > 0:
                    inner = (inner + multiple_of - 1) // multiple_of * multiple_of  # round up
                
                self.gate_proj = nn.Linear(hidden_dim, inner, bias=bias)  # produces gate
                self.up_proj   = nn.Linear(hidden_dim, inner, bias=bias)  # produces value
                self.down_proj = nn.Linear(inner, hidden_dim, bias=bias)
                self.dropout   = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # gate with SiLU (a.k.a. swish), element-wise product with up path
                gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
                out = self.down_proj(gated)
                return self.dropout(out)
            
        # Feed-forward network
        # self.ffn = nn.Sequential(
        #     nn.Linear(hidden_dim, 4 * hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(4 * hidden_dim, hidden_dim),
        #     nn.Dropout(dropout)
        # )

        self.ffn = SwiGLUFFN(hidden_dim, dropout=0.0, ffn_mult=4.0, swiglu_factor=2/3, multiple_of=256, bias=False)

        # Per-level FFN: one SwiGLU per hierarchy level with a per-level expansion width
        # (inner ~ 4*2/3*dim_L). L0 (most nodes) gets a small dim -> cheap; coarse levels (few
        # nodes) get a large dim -> capacity, mimicking a U-Net's channel growth at the
        # bottleneck. Input/output stay hidden_dim so the residual + cross-level path are intact.
        self.per_level_ffn_dims = [int(d) for d in (per_level_ffn_dims or [])]
        self.ffn_level = nn.ModuleList()
        if self.per_level_ffn_dims:
            for d in self.per_level_ffn_dims:
                inner = int(4.0 * (2.0 / 3.0) * int(d))
                self.ffn_level.append(SwiGLUFFN(hidden_dim, dropout=0.0, multiple_of=256, bias=False, inner_dim=inner))

        # Layer normalization
        self.norm1 = make_norm(hidden_dim, norm_type=self.norm_type, eps=self.norm_eps)
        self.norm2 = make_norm(hidden_dim, norm_type=self.norm_type, eps=self.norm_eps)
        
        # --- Layers for Beta Gating (if enabled) ---
        if self.make_beta_gating:
            # This lin_skip transforms the input 'x' before gating with attention output
            # Its output dimension must match the output dimension of message_passing.out_proj
            #self.lin_skip_attn = nn.Linear(hidden_dim, hidden_dim, bias=True) # Or nn.Linear
            #self.lin_beta_attn = nn.Linear(3 * hidden_dim, 1, bias=True)    # Or nn.Linear
            self.lin_beta_attn = nn.Linear(hidden_dim, hidden_dim, bias=True)    # Or nn.Linear
            # For FFN residual
            #self.lin_skip_ffn = nn.Linear(hidden_dim, hidden_dim, bias=True)  # Or nn.Linear
            #self.lin_beta_ffn = nn.Linear(3 * hidden_dim, 1, bias=True)     # Or nn.Linear

        #if self.make_beta_gating:
            # ATTENTION skip ~ identity
            #nn.init.eye_(self.lin_skip_attn.weight)
            #nn.init.zeros_(self.lin_skip_attn.bias)

            # FFN skip ~ identity
            #nn.init.eye_(self.lin_skip_ffn.weight)
            #nn.init.zeros_(self.lin_skip_ffn.bias)

            # Optional tiny noise so it's not *exact* identity:
            #self.lin_skip_attn.weight.data += 1e-3 * torch.randn_like(self.lin_skip_attn.weight)
            #self.lin_skip_ffn.weight.data  += 1e-3 * torch.randn_like(self.lin_skip_ffn.weight)
        
        if self.make_beta_gating:
            # ATTENTION gate
            nn.init.zeros_(self.lin_beta_attn.weight)
            nn.init.constant_(self.lin_beta_attn.bias, 2.0)   # sigmoid(2) ≈ 0.88

            # FFN gate
            #nn.init.zeros_(self.lin_beta_ffn.weight)
            #nn.init.constant_(self.lin_beta_ffn.bias, 2.0)   # sigmoid(2) ≈ 0.88

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def reset_lateral_edge_traces(self) -> None:
        if hasattr(self.message_passing, "reset_lateral_edge_traces"):
            self.message_passing.reset_lateral_edge_traces()
    
    def _apply_ffn(self, x_normed, node_level, level_offsets=None):
        """Per-level FFN dispatch: each hierarchy level's nodes go through their own SwiGLU
        (per-level expansion width). Falls back to the shared FFN when per_level_ffn_dims is
        unset. Uses contiguous level_offsets slices when available (one host sync), else a
        masked path (for the active-level subset, which isn't level-contiguous)."""
        if not self.per_level_ffn_dims:
            return self.ffn(x_normed)
        nlv = len(self.per_level_ffn_dims)
        is3d = x_normed.dim() == 3
        N = int(x_normed.size(-2))
        out = torch.empty_like(x_normed)

        def _put(lo, hi, sl):
            res = sl
            if is3d:
                out[:, lo:hi, :] = res
            else:
                out[lo:hi, :] = res

        bounds = None
        if isinstance(level_offsets, torch.Tensor) and level_offsets.numel() >= 2:
            loff = level_offsets.detach().to("cpu").tolist()
            bounds = [(int(loff[i]), int(loff[i + 1])) for i in range(len(loff) - 1)]
        if bounds is not None:
            end = 0
            for L in range(min(nlv, len(bounds))):
                s, e = bounds[L]
                if e > s:
                    sl = x_normed[:, s:e, :] if is3d else x_normed[s:e, :]
                    _put(s, e, self.ffn_level[L](sl))
                end = max(end, e)
            if end < N:  # coarse levels beyond the per-level list -> shared FFN
                sl = x_normed[:, end:, :] if is3d else x_normed[end:, :]
                _put(end, N, self.ffn(sl))
            return out
        # masked fallback (non-contiguous node ordering, e.g. active-level subset)
        nl = node_level.to(device=x_normed.device, dtype=torch.long)
        covered = torch.zeros(N, dtype=torch.bool, device=x_normed.device)
        for L in range(nlv):
            idx = (nl == L).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            sl = x_normed.index_select(1 if is3d else 0, idx)
            res = self.ffn_level[L](sl)
            out.index_copy_(1 if is3d else 0, idx, res)
            covered[idx] = True
        if not bool(covered.all()):
            rest = (~covered).nonzero(as_tuple=False).view(-1)
            res = self.ffn(x_normed.index_select(1 if is3d else 0, rest))
            out.index_copy_(1 if is3d else 0, rest, res)
        return out

    def forward(self, x, edge_index, node_level, level_offsets=None, positions=None, edge_attr=None, hqd_edges=None, active_levels=None, edge_type=None):
        """
        Forward pass through the hierarchical transformer layer.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            node_level: Level of each node [num_nodes]
            level_offsets: List/Tensor of start indices for each level (for RoPE)
            edge_attr: Optional edge attributes [num_edges, edge_dim]
            hqd_edges: Optional tuple (b_idx, src_idx, dst_idx) for HQD sparse attention.

        Returns:
            updated_x: Updated node features [num_nodes, hidden_dim]
        """
        
        # Apply hierarchical message passing with residual connection
        # Pass level_offsets down
        normed_x = self.norm1(x) if active_levels is None else x
        active_input_norm = None if active_levels is None else self.norm1
        attn_out, new_edge_attr = self.message_passing(
            normed_x,
            edge_index,
            node_level,
            level_offsets=level_offsets,
            positions=positions, # Pass positions down
            edge_attr=edge_attr,
            hqd_edges=hqd_edges,
            active_levels=active_levels,
            input_norm=active_input_norm,
            edge_type=edge_type,
        )

        if new_edge_attr is not None:
            edge_attr = new_edge_attr
            

        active_idx = None
        if active_levels is not None:
            active_mask = torch.zeros_like(node_level, dtype=torch.bool, device=x.device)
            node_level_device = node_level.to(device=x.device, dtype=torch.long)
            for level in active_levels:
                active_mask = active_mask | (node_level_device == int(level))
            active_idx = torch.nonzero(active_mask, as_tuple=False).view(-1)
            if active_idx.numel() == 0:
                return x, new_edge_attr

        if active_idx is not None:
            if x.dim() == 3:
                x_active = x.index_select(1, active_idx)
                attn_active = attn_out.index_select(1, active_idx)
            else:
                x_active = x.index_select(0, active_idx)
                attn_active = attn_out.index_select(0, active_idx)

            if self.use_beta_gating:
                beta_val_attn = self.lin_beta_attn(attn_active).sigmoid()
                x_active = beta_val_attn * x_active + (1 - beta_val_attn) * self.dropout(attn_active)
            else:
                x_active = x_active + self.dropout(attn_active)

            ffn_out_active = self._apply_ffn(self.norm2(x_active), node_level.index_select(0, active_idx), None)
            x_active = x_active + self.dropout(ffn_out_active)

            x_next = x.clone()
            if x.dim() == 3:
                x_next.index_copy_(1, active_idx, x_active.to(dtype=x_next.dtype))
            else:
                x_next.index_copy_(0, active_idx, x_active.to(dtype=x_next.dtype))
            return x_next, new_edge_attr

        if self.use_beta_gating:
            # Old computationally expensive way with two gatings
            # x_identity_attn = x # Store input for first residual
            # # x_r is the transformed skip path
            # x_r_attn = self.lin_skip_attn(x_identity_attn) # W_1 x_i equivalent
            # # attn_out_aggregated is m_i equivalent
            # beta_val_attn = self.lin_beta_attn(
            #     torch.cat([attn_out, x_r_attn, attn_out - x_r_attn], dim=-1)
            # ).sigmoid()
            # x = beta_val_attn * x_r_attn + (1 - beta_val_attn) * self.dropout(attn_out)

            x_identity_attn = x # Store input for first residual
            # x_r is the transformed skip path
            x_r_attn = x_identity_attn
            # attn_out_aggregated is m_i equivalent
            beta_val_attn = self.lin_beta_attn(attn_out).sigmoid()
            x_beta = beta_val_attn * x_r_attn + (1 - beta_val_attn) * self.dropout(attn_out)
            if active_idx is None:
                x = x_beta
            else:
                x_next = x.clone()
                if x.dim() == 3:
                    x_next[:, active_idx, :] = x_beta[:, active_idx, :]
                else:
                    x_next[active_idx, :] = x_beta[active_idx, :]
                x = x_next
        else: # Standard additive residual
            if active_idx is None:
                x = x + self.dropout(attn_out)
            else:
                x_next = x.clone()
                if x.dim() == 3:
                    x_next[:, active_idx, :] = x[:, active_idx, :] + self.dropout(attn_out[:, active_idx, :])
                else:
                    x_next[active_idx, :] = x[active_idx, :] + self.dropout(attn_out[active_idx, :])
                x = x_next

        #x_identity_ffn = x # Store input for second residual
        if active_idx is None:
            ffn_out = self._apply_ffn(self.norm2(x), node_level, level_offsets)
        elif x.dim() == 3:
            ffn_out_active = self._apply_ffn(self.norm2(x[:, active_idx, :]), node_level.index_select(0, active_idx), None)
            ffn_out = None
        else:
            ffn_out_active = self._apply_ffn(self.norm2(x[active_idx, :]), node_level.index_select(0, active_idx), None)
            ffn_out = None
        if self.use_beta_gating:
            if active_idx is None:
                x = x + self.dropout(ffn_out)
            else:
                x_next = x.clone()
                if x.dim() == 3:
                    x_next[:, active_idx, :] = x[:, active_idx, :] + self.dropout(ffn_out_active)
                else:
                    x_next[active_idx, :] = x[active_idx, :] + self.dropout(ffn_out_active)
                x = x_next
            # decided two gatings are too much, swiglu and ffn gating
            # x_r_ffn = self.lin_skip_ffn(x_identity_ffn) # W_1 x_i equivalent
            # # ffn_out is m_i equivalent
            # beta_val_ffn = self.lin_beta_ffn(
            #     torch.cat([ffn_out, x_r_ffn, ffn_out - x_r_ffn], dim=-1)
            # ).sigmoid()
            # x = beta_val_ffn * x_r_ffn + (1 - beta_val_ffn) * self.dropout(ffn_out)
        else: # Standard additive residual
            if active_idx is None:
                x = x + self.dropout(ffn_out)
            else:
                x_next = x.clone()
                if x.dim() == 3:
                    x_next[:, active_idx, :] = x[:, active_idx, :] + self.dropout(ffn_out_active)
                else:
                    x_next[active_idx, :] = x[active_idx, :] + self.dropout(ffn_out_active)
                x = x_next

        return x, new_edge_attr
