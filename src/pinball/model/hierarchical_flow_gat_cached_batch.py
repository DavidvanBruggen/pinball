# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import atexit
import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import scatter # Import scatter for aggregation

from torch_geometric.loader import NeighborLoader
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedTokenizerBase
import math
import time
import logging
from typing import Optional, Dict, List, Tuple, Union, Any, Sequence

from .layers.positional_encoding import RotaryPositionalEncoding, LagrangianPositionalEncoding
from .layers.normalization import RMSNorm, make_norm
from .layers.hierarchical_message_passing import HierarchicalTransformerLayer, attention_forward, pick_attention_backend
from .hierarchy.unified_hierarchy_builder import UnifiedHierarchyBuilder, EdgeFeatureGenerator

logger = logging.getLogger(__name__)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _normalize_level_grid_shape_map(
    level_grid_shapes: Optional[Union[Dict[int, Tuple[int, int]], List[Any], Tuple[Any, ...]]],
) -> Dict[int, Tuple[int, int]]:
    if level_grid_shapes is None:
        return {}

    if isinstance(level_grid_shapes, dict):
        items = level_grid_shapes.items()
    elif isinstance(level_grid_shapes, (list, tuple)):
        items = enumerate(level_grid_shapes)
    else:
        return {}

    normalized: Dict[int, Tuple[int, int]] = {}
    for lvl_idx, shape in items:
        if shape is None:
            continue
        try:
            if len(shape) != 2:
                continue
        except Exception:
            continue
        try:
            gh = int(shape[0])
            gw = int(shape[1])
        except Exception:
            continue
        if gh > 0 and gw > 0:
            normalized[int(lvl_idx)] = (gh, gw)
    return normalized




def _build_node_pos_local_from_offsets(
    total_nodes: int,
    level_offsets: Optional[Union[List[int], torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    node_pos_local = torch.arange(int(total_nodes), device=device, dtype=torch.long)
    if level_offsets is None:
        return node_pos_local

    if not isinstance(level_offsets, torch.Tensor):
        level_offsets = torch.as_tensor(level_offsets, device=device, dtype=torch.long)
    else:
        level_offsets = level_offsets.to(device=device, dtype=torch.long)

    level_offsets = level_offsets.view(-1)
    if level_offsets.numel() < 2:
        return node_pos_local

    for lvl_idx in range(level_offsets.numel() - 1):
        start = int(level_offsets[lvl_idx].item())
        end = int(level_offsets[lvl_idx + 1].item())
        if end > start:
            node_pos_local[start:end] = torch.arange(end - start, device=device, dtype=torch.long)
    return node_pos_local


def _expand_int_list(value: Optional[Union[List[int], Tuple[int, ...]]], length: int, default: int) -> List[int]:
    if value is None:
        items: List[int] = []
    else:
        items = [int(x) for x in value]
    if not items:
        items = [int(default)]
    if len(items) < int(length):
        items = items + [items[-1]] * (int(length) - len(items))
    return [int(x) for x in items[: int(length)]]


def _parse_pinball_cross_query_pairs(value: Optional[Union[List[Any], Tuple[Any, ...], str]]) -> List[Tuple[int, int]]:
    if value is None:
        return []
    if isinstance(value, str):
        items: List[Any] = [value]
    else:
        items = list(value)
    pairs: List[Tuple[int, int]] = []
    for item in items:
        if isinstance(item, str):
            cleaned = item.replace("->", ":").replace("<-", ":").replace(",", ":")
            parts = [p for p in cleaned.split(":") if p != ""]
            if len(parts) != 2:
                raise ValueError(f"Invalid pinball_cross_query_pairs entry {item!r}; expected 'upper:lower'")
            upper, lower = int(parts[0]), int(parts[1])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            upper, lower = int(item[0]), int(item[1])
        else:
            raise ValueError(f"Invalid pinball_cross_query_pairs entry {item!r}; expected pair")
        if upper < 0 or lower < 0:
            raise ValueError("pinball_cross_query_pairs levels must be non-negative")
        pairs.append((upper, lower))
    return pairs


class LowRankAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, bias: bool = False):
        super().__init__()
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        rank = max(1, min(int(rank), in_dim, out_dim))
        self.down = nn.Linear(in_dim, rank, bias=bias)
        self.up = nn.Linear(rank, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class PackedSwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0, ffn_dim: int = 0):
        super().__init__()
        # LLaMA-style SwiGLU uses 2/3 of a 4x FFN, i.e. 8/3 * hidden. ffn_dim > 0 overrides
        # the "processing dim" (same semantics as per_level_ffn_dims: inner ~ 8/3 * ffn_dim),
        # so coarse-query refiners can carry a fat FFN; 0 = width from hidden_dim (unchanged).
        inner = int((8.0 / 3.0) * (int(ffn_dim) if int(ffn_dim) > 0 else int(hidden_dim)))
        inner = max(256, ((inner + 255) // 256) * 256)
        self.gate_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PackedTransformerBlock(nn.Module):
    """Dense packed transformer block for tiny upper-level Pinball refinement."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        norm_type: str,
        norm_eps: float,
        attn_backend: str = "auto",
        window: int = 0,
        causal: bool = True,
        flash_dtype_cast: bool = False,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError("PackedTransformerBlock hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_backend = str(attn_backend).lower()
        if self.attn_backend not in {"auto", "flash", "sdpa"}:
            raise ValueError("pinball multirate attention backend must be 'auto', 'flash', or 'sdpa'")
        self.window = max(0, int(window))
        self.causal = bool(causal)
        self.flash_dtype_cast = bool(flash_dtype_cast)
        self.norm1 = make_norm(hidden_dim, norm_type=norm_type, eps=norm_eps)
        self.norm2 = make_norm(hidden_dim, norm_type=norm_type, eps=norm_eps)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn = PackedSwiGLUFFN(hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _sdpa_window_bias(self, t: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.window <= 0 and not self.causal:
            return None
        idx = torch.arange(t, device=device)
        allow = torch.ones((t, t), device=device, dtype=torch.bool)
        if self.window > 0:
            allow = (idx.view(t, 1) - idx.view(1, t)).abs() <= int(self.window)
        if self.causal:
            allow = allow & (idx.view(t, 1) >= idx.view(1, t))
        neg = torch.finfo(dtype).min
        bias = torch.full((t, t), neg, device=device, dtype=dtype)
        return bias.masked_fill(allow, 0.0)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)
        dropout_p = float(self.dropout.p) if self.training else 0.0
        backend = self.attn_backend
        if backend == "auto":
            backend = "flash" if q.device.type == "cuda" else "sdpa"
        if backend == "flash":
            resolved_backend, flash_func = pick_attention_backend(q.device)
            if resolved_backend not in {"fa2", "fa3"} or flash_func is None:
                if self.attn_backend == "flash":
                    raise RuntimeError("pinball multirate flash backend requested but FlashAttention is unavailable")
                backend = "sdpa"
            else:
                win = None
                if self.window > 0:
                    win = (int(self.window), 0) if self.causal else (int(self.window), int(self.window))
                y = attention_forward(
                    q,
                    k,
                    v,
                    causal=bool(self.causal),
                    dropout_p=dropout_p,
                    backend=resolved_backend,
                    flash_func=flash_func,
                    window_size=win,
                    flash_dtype_cast=bool(self.flash_dtype_cast),
                )
                return self.out_proj(y.reshape(bsz, seq_len, self.hidden_dim))
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        attn_mask = None
        is_causal = bool(self.causal and self.window <= 0)
        if self.window > 0:
            attn_mask = self._sdpa_window_bias(seq_len, q.device, q.dtype).view(1, 1, seq_len, seq_len)
            is_causal = False
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.hidden_dim)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self._attention(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PinballPackedLevelRefiner(nn.Module):
    """Read-once/write-once packed dense refiner over selected hierarchy levels."""

    def __init__(
        self,
        work_dim: int,
        num_heads: int,
        levels: List[int],
        max_steps: int,
        dropout: float,
        norm_type: str,
        norm_eps: float,
        attn_backend: str,
        window: int,
        causal: bool,
        shared_weights: bool,
        workspace_tokens: int = 0,
        flash_dtype_cast: bool = False,
    ):
        super().__init__()
        self.work_dim = int(work_dim)
        self.levels = [int(level) for level in levels]
        self.max_steps = max(0, int(max_steps))
        self.workspace_tokens = max(0, int(workspace_tokens))
        self.shared_weights = bool(shared_weights)
        block_count = 1 if self.shared_weights else max(1, self.max_steps)
        self.blocks = nn.ModuleList(
            [
                PackedTransformerBlock(
                    hidden_dim=work_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    attn_backend=attn_backend,
                    window=window,
                    causal=causal,
                    flash_dtype_cast=flash_dtype_cast,
                )
                for _ in range(block_count)
            ]
        )
        self.level_scales = nn.ParameterDict({str(level): nn.Parameter(torch.tensor(1.0e-2)) for level in self.levels})
        if self.workspace_tokens > 0:
            self.workspace = nn.Parameter(torch.zeros(1, self.workspace_tokens, self.work_dim))
            nn.init.normal_(self.workspace, mean=0.0, std=0.02)
        else:
            self.workspace = None

    def forward(
        self,
        x: torch.Tensor,
        node_level: torch.Tensor,
        steps: Optional[int] = None,
        extra_prefix: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # extra_prefix: {level: [B, P, H]} frozen context rows (KV-cached decode) prepended
        # in time order before this graph's rows of that level — they serve as causal keys
        # for the windowed attention exactly like the pre-tail nodes they cache, and their
        # (recomputed) outputs are sliced off before write-back.
        steps = self.max_steps if steps is None else max(0, min(int(steps), self.max_steps))
        if steps <= 0 or x.dim() != 3:
            return x
        out = x
        bsz = x.size(0)
        for level in self.levels:
            idx = torch.nonzero(node_level == int(level), as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            z0 = out.index_select(1, idx)
            z = z0
            n_prefix = 0
            if extra_prefix is not None and int(level) in extra_prefix:
                pre = extra_prefix[int(level)]
                if pre is not None and pre.dim() == 3 and pre.size(1) > 0:
                    n_prefix = int(pre.size(1))
                    z = torch.cat([pre.to(device=z.device, dtype=z.dtype), z], dim=1)
            use_workspace = self.workspace is not None and int(level) == 3
            if use_workspace:
                # Prefix learned workspace tokens; causal packed attention keeps L3 from reading future L3 tokens through workspace.
                z = torch.cat([self.workspace.to(device=z.device, dtype=z.dtype).expand(bsz, -1, -1), z], dim=1)
            for step in range(steps):
                block = self.blocks[0] if self.shared_weights else self.blocks[min(step, len(self.blocks) - 1)]
                z = block(z)
            skip = (self.workspace_tokens if use_workspace else 0) + n_prefix
            if skip > 0:
                z_level = z[:, skip:, :]
            else:
                z_level = z
            scale = self.level_scales[str(level)].to(device=z_level.device, dtype=z_level.dtype)
            updated = z0 + scale * (z_level - z0)
            out = out.clone()
            out.index_copy_(1, idx, updated.to(dtype=out.dtype))
        return out


class PinballPackedCrossAttentionRefiner(nn.Module):
    """Packed upper<-lower cross-attention for cheap detail retrieval after graph sync."""

    def __init__(
        self,
        work_dim: int,
        num_heads: int,
        steps: int,
        dropout: float,
        norm_type: str,
        norm_eps: float,
        attn_backend: str,
        topk_l2: int,
        causal: bool,
        shared_weights: bool,
        update_l2_enable: bool = False,
        update_l2_scale_init: float = 1.0e-2,
        query_level: int = 3,
        memory_level: int = 2,
        memory_window: int = 0,
        memory_recency_window: int = 0,
        selection_mode: str = "global_mean",
        write_scale_init: float = 1.0e-2,
        flash_dtype_cast: bool = False,
        ffn_dim: int = 0,  # FFN "processing dim" for the query-row FFN (0 = work_dim width)
    ):
        super().__init__()
        work_dim = int(work_dim)
        num_heads = int(num_heads)
        if work_dim % num_heads != 0:
            raise ValueError("PinballPackedCrossAttentionRefiner work_dim must be divisible by num_heads")
        self.work_dim = work_dim
        self.num_heads = num_heads
        self.head_dim = work_dim // num_heads
        self.steps = max(0, int(steps))
        self.topk_l2 = max(0, int(topk_l2))
        self.topk_memory = self.topk_l2
        self.query_level = int(query_level)
        self.memory_level = int(memory_level)
        self.memory_window = max(0, int(memory_window))
        # Recency cap on the ATTENTION READ (ar-time units): a query only reads memory
        # closing within the last `memory_recency_window` time units. Applied inside the
        # causal mask, so it is per-query and leak-free (only restricts the PAST); with
        # window >= context length it is bit-identical to off. 0 = unlimited. This bounds
        # the per-query read set for linear scaling at long context (content retrieval
        # beyond the window is xq_nominate's job; deeper hierarchy levels extend reach).
        self.memory_recency_window = max(0, int(memory_recency_window))
        self.selection_mode = str(selection_mode).lower().replace("-", "_")
        if self.selection_mode not in {"global_mean", "per_query"}:
            raise ValueError("pinball cross-query selection_mode must be 'global_mean' or 'per_query'")
        self.causal = bool(causal)
        self.attn_backend = str(attn_backend).lower()
        if self.attn_backend not in {"auto", "flash", "sdpa"}:
            raise ValueError("pinball upper cross-attn backend must be 'auto', 'flash', or 'sdpa'")
        self.shared_weights = bool(shared_weights)
        self.update_l2_enable = bool(update_l2_enable)
        self.flash_dtype_cast = bool(flash_dtype_cast)
        block_count = 1 if self.shared_weights else max(1, self.steps)
        self.q_norms = nn.ModuleList([make_norm(work_dim, norm_type=norm_type, eps=norm_eps) for _ in range(block_count)])
        self.kv_norms = nn.ModuleList([make_norm(work_dim, norm_type=norm_type, eps=norm_eps) for _ in range(block_count)])
        self.q_projs = nn.ModuleList([nn.Linear(work_dim, work_dim, bias=False) for _ in range(block_count)])
        self.kv_projs = nn.ModuleList([nn.Linear(work_dim, 2 * work_dim, bias=False) for _ in range(block_count)])
        self.out_projs = nn.ModuleList([nn.Linear(work_dim, work_dim, bias=False) for _ in range(block_count)])
        self.ffn_norms = nn.ModuleList([make_norm(work_dim, norm_type=norm_type, eps=norm_eps) for _ in range(block_count)])
        # The FFN processes the QUERY rows (the level this refiner writes), so a per-query-
        # level fat width puts episode capacity where nodes are geometrically few — and
        # cycling re-runs these same weights (tied across cycles), scaling compute not params.
        self.ffns = nn.ModuleList([PackedSwiGLUFFN(work_dim, dropout=dropout, ffn_dim=int(ffn_dim)) for _ in range(block_count)])
        self.dropout = nn.Dropout(dropout)
        self.write_scale = nn.Parameter(torch.tensor(float(write_scale_init)))
        self.l2_update_scale = nn.Parameter(torch.tensor(float(update_l2_scale_init)))
        self.l2_update_norm = make_norm(work_dim, norm_type=norm_type, eps=norm_eps)
        self.l2_update_proj = nn.Linear(work_dim, work_dim, bias=False)
        self.l2_update_gate = nn.Linear(2 * work_dim, work_dim, bias=True)
        nn.init.constant_(self.l2_update_gate.bias, -2.0)

    def _select_memory(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_time: Optional[torch.Tensor] = None,
        query_time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.topk_memory <= 0 and self.memory_window <= 0:
            selected = torch.arange(memory.size(1), device=memory.device, dtype=torch.long).view(1, -1).expand(memory.size(0), -1)
            return memory, selected, memory_time
        if self.selection_mode == "per_query":
            q = F.normalize(query, dim=-1)
            k = F.normalize(memory, dim=-1)
            scores = torch.einsum("bqd,bnd->bqn", q, k)
            mem_t = None
            qt = None
            if memory_time is not None:
                mem_t = memory_time.to(device=memory.device, dtype=torch.long)
                if mem_t.dim() == 1:
                    mem_t = mem_t.view(1, -1).expand(memory.size(0), -1)
            if query_time is not None:
                qt = query_time.to(device=memory.device, dtype=torch.long)
                if qt.dim() == 1:
                    qt = qt.view(1, -1).expand(memory.size(0), -1)
            if self.causal and mem_t is not None and qt is not None:
                scores = scores.masked_fill(mem_t.unsqueeze(1) > qt.unsqueeze(-1), torch.finfo(scores.dtype).min)
            if self.memory_window > 0 and mem_t is not None and qt is not None:
                scores = scores.masked_fill((qt.unsqueeze(-1) - mem_t.unsqueeze(1)).abs() > int(self.memory_window), torch.finfo(scores.dtype).min)
            topk = int(self.topk_memory) if self.topk_memory > 0 else int(memory.size(1))
            topk = min(max(1, topk), int(memory.size(1)))
            idx = torch.topk(scores, k=topk, dim=-1).indices
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, memory.size(-1))
            selected_memory = torch.gather(memory.unsqueeze(1).expand(-1, query.size(1), -1, -1), dim=2, index=gather_idx)
            selected_time = None
            if mem_t is not None:
                selected_time = torch.gather(mem_t.unsqueeze(1).expand(-1, query.size(1), -1), dim=2, index=idx)
            return selected_memory, idx, selected_time
        q = F.normalize(query.mean(dim=1), dim=-1)
        k = F.normalize(memory, dim=-1)
        scores = torch.einsum("bd,bnd->bn", q, k)
        mem_t = None
        max_query_time = None
        if memory_time is not None:
            mem_t = memory_time.to(device=memory.device, dtype=torch.long)
            if mem_t.dim() == 1:
                mem_t = mem_t.view(1, -1).expand(memory.size(0), -1)
        if query_time is not None:
            qt = query_time.to(device=memory.device, dtype=torch.long)
            max_query_time = qt.max().view(1, 1) if qt.dim() == 1 else qt.max(dim=1, keepdim=True).values
        if self.causal and mem_t is not None and max_query_time is not None:
            scores = scores.masked_fill(mem_t > max_query_time, torch.finfo(scores.dtype).min)
        if self.memory_window > 0 and mem_t is not None and max_query_time is not None:
            scores = scores.masked_fill((max_query_time - mem_t).abs() > int(self.memory_window), torch.finfo(scores.dtype).min)
        topk = int(self.topk_memory) if self.topk_memory > 0 else int(memory.size(1))
        topk = min(max(1, topk), int(memory.size(1)))
        idx = torch.topk(scores, k=topk, dim=1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, memory.size(-1))
        selected_time = None
        if mem_t is not None:
            selected_time = torch.gather(mem_t, dim=1, index=idx)
        return torch.gather(memory, dim=1, index=gather_idx), idx, selected_time

    def _causal_mask(
        self,
        query_time: Optional[torch.Tensor],
        memory_time: Optional[torch.Tensor],
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.causal:
            return None
        if query_time is None or memory_time is None:
            return None
        qt = query_time.to(device=device, dtype=torch.long)
        mt = memory_time.to(device=device, dtype=torch.long)
        if qt.dim() == 1 and mt.dim() == 1:
            if qt.numel() != int(q_len) or mt.numel() != int(k_len):
                return None
            allow = mt.view(1, k_len) <= qt.view(q_len, 1)
            if self.memory_recency_window > 0:
                allow = allow & ((qt.view(q_len, 1) - mt.view(1, k_len)) <= int(self.memory_recency_window))
            neg = torch.finfo(dtype).min
            bias = torch.full((q_len, k_len), neg, device=device, dtype=dtype)
            return bias.masked_fill(allow, 0.0)
        if qt.dim() == 1:
            qt = qt.view(1, q_len).expand(mt.size(0), -1)
        if mt.dim() == 1:
            mt = mt.view(1, k_len).expand(qt.size(0), -1)
        if qt.dim() != 2 or mt.dim() != 2 or qt.size(1) != int(q_len) or mt.size(1) != int(k_len) or qt.size(0) != mt.size(0):
            return None
        allow = mt.view(mt.size(0), 1, k_len) <= qt.view(qt.size(0), q_len, 1)
        if self.memory_recency_window > 0:
            allow = allow & (
                (qt.view(qt.size(0), q_len, 1) - mt.view(mt.size(0), 1, k_len)) <= int(self.memory_recency_window)
            )
        neg = torch.finfo(dtype).min
        bias = torch.full((qt.size(0), q_len, k_len), neg, device=device, dtype=dtype)
        return bias.masked_fill(allow, 0.0)

    def _cross_attention(
        self,
        h3: torch.Tensor,
        h2: torch.Tensor,
        query_time: Optional[torch.Tensor],
        memory_time: Optional[torch.Tensor],
        block_idx: int,
    ) -> torch.Tensor:
        bsz, q_len, _ = h3.shape
        k_len = int(h2.size(1))
        if h2.dim() == 4:
            k_len = int(h2.size(2))
            q = self.q_projs[block_idx](self.q_norms[block_idx](h3)).view(bsz, q_len, self.num_heads, self.head_dim)
            k_in, v_in = self.kv_projs[block_idx](self.kv_norms[block_idx](h2)).chunk(2, dim=-1)
            k = k_in.view(bsz, q_len, k_len, self.num_heads, self.head_dim)
            v = v_in.view(bsz, q_len, k_len, self.num_heads, self.head_dim)
            dropout_p = float(self.dropout.p) if self.training else 0.0
            qh = q.reshape(bsz * q_len, 1, self.num_heads, self.head_dim).transpose(1, 2)
            kh = k.reshape(bsz * q_len, k_len, self.num_heads, self.head_dim).transpose(1, 2)
            vh = v.reshape(bsz * q_len, k_len, self.num_heads, self.head_dim).transpose(1, 2)
            mask = None
            valid_q = None  # [bsz, q_len] False where a query has NO causal memory (all-masked row)
            if self.causal and query_time is not None and memory_time is not None:
                qt = query_time.to(device=h3.device, dtype=torch.long)
                mt = memory_time.to(device=h3.device, dtype=torch.long)
                if qt.dim() == 1:
                    qt = qt.view(1, q_len).expand(bsz, -1)
                if mt.dim() == 2:
                    mt = mt.unsqueeze(0).expand(bsz, -1, -1)
                elif mt.dim() == 1:
                    mt = mt.view(1, 1, k_len).expand(bsz, q_len, -1)
                if mt.dim() != 3 or mt.size(0) != bsz or mt.size(1) != q_len or mt.size(2) != k_len:
                    mt = None
                if mt is not None:
                    allow = mt <= qt.unsqueeze(-1)
                    if self.memory_recency_window > 0:
                        allow = allow & ((qt.unsqueeze(-1) - mt) <= int(self.memory_recency_window))
                    neg = torch.finfo(qh.dtype).min
                    mask = torch.full((bsz * q_len, 1, 1, k_len), neg, device=h3.device, dtype=qh.dtype)
                    mask = mask.masked_fill(allow.reshape(bsz * q_len, 1, 1, k_len), 0.0)
                    valid_q = allow.any(dim=-1)  # a finite-min row would else softmax to uniform over FUTURE memory
            y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, dropout_p=dropout_p, is_causal=False)
            y = y.transpose(1, 2).contiguous().reshape(bsz, q_len, self.work_dim)
            if valid_q is not None:
                y = y * valid_q.unsqueeze(-1).to(dtype=y.dtype)
            return self.out_projs[block_idx](y)
        q = self.q_projs[block_idx](self.q_norms[block_idx](h3)).view(bsz, q_len, self.num_heads, self.head_dim)
        k_in, v_in = self.kv_projs[block_idx](self.kv_norms[block_idx](h2)).chunk(2, dim=-1)
        k = k_in.view(bsz, k_len, self.num_heads, self.head_dim)
        v = v_in.view(bsz, k_len, self.num_heads, self.head_dim)
        dropout_p = float(self.dropout.p) if self.training else 0.0
        attn_bias = self._causal_mask(query_time, memory_time, q_len, k_len, q.device, q.dtype)
        backend = self.attn_backend
        if backend == "auto":
            backend = "flash" if q.device.type == "cuda" and attn_bias is None else "sdpa"
        if backend == "flash":
            if attn_bias is not None:
                raise RuntimeError("pinball upper cross-attn causal mask requires backend='auto' or 'sdpa', not explicit flash")
            resolved_backend, flash_func = pick_attention_backend(q.device)
            if resolved_backend not in {"fa2", "fa3"} or flash_func is None:
                if self.attn_backend == "flash":
                    raise RuntimeError("pinball upper cross-attn flash backend requested but FlashAttention is unavailable")
                backend = "sdpa"
            else:
                y = attention_forward(
                    q,
                    k,
                    v,
                    causal=False,
                    dropout_p=dropout_p,
                    backend=resolved_backend,
                    flash_func=flash_func,
                    window_size=None,
                    flash_dtype_cast=bool(self.flash_dtype_cast),
                )
                return self.out_projs[block_idx](y.reshape(bsz, q_len, self.work_dim))
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        if attn_bias is None:
            mask = None
        elif attn_bias.dim() == 2:
            mask = attn_bias.view(1, 1, q_len, k_len)
        else:
            mask = attn_bias.view(attn_bias.size(0), 1, q_len, k_len)
        y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, dropout_p=dropout_p, is_causal=False)
        y = y.transpose(1, 2).contiguous().reshape(bsz, q_len, self.work_dim)
        if mask is not None:
            # Zero any query row with NO causal memory: its bias row is all finite-min, so softmax
            # would spread uniformly over FUTURE memory (leak). all-masked -> zero contribution.
            valid_q = (mask > (torch.finfo(mask.dtype).min * 0.5)).any(dim=-1).view(mask.size(0), q_len)
            y = y * valid_q.reshape(-1, q_len, 1).to(dtype=y.dtype)
        return self.out_projs[block_idx](y)

    def forward(
        self,
        x: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor] = None,
        query_keep: Optional[Tuple[torch.Tensor, int]] = None,
        extra_memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        # extra_memory: (x_mem [B, M, H], mem_time [M]) frozen memory rows (KV-cached decode)
        # PREPENDED to this graph's memory-level rows before selection: cached pre-tail coarse
        # nodes the tail graph doesn't contain. mem_time must be on the same (shifted) clock
        # as node_ar_time. Read-only: the update_l2 write-back is skipped when present.
        if self.steps <= 0 or x.dim() != 3:
            return x, {"steps": 0, "memory_selected": 0, "memory_updated": 0, "query_nodes": 0}
        memory_idx = torch.nonzero(node_level == int(self.memory_level), as_tuple=False).view(-1)
        query_idx = torch.nonzero(node_level == int(self.query_level), as_tuple=False).view(-1)
        if memory_idx.numel() == 0 or query_idx.numel() == 0:
            return x, {"steps": 0, "memory_selected": 0, "memory_updated": 0, "query_nodes": int(query_idx.numel())}
        memory_all = x.index_select(1, memory_idx)
        query_start = x.index_select(1, query_idx)
        memory_time = None
        query_time = None
        if node_ar_time is not None:
            query_time = node_ar_time.index_select(0, query_idx)
            memory_time = node_ar_time.index_select(0, memory_idx)
        has_extra_memory = False
        if extra_memory is not None:
            mem_x, mem_t = extra_memory
            if mem_x is not None and mem_x.dim() == 3 and mem_x.size(1) > 0:
                memory_all = torch.cat([mem_x.to(device=memory_all.device, dtype=memory_all.dtype), memory_all], dim=1)
                if memory_time is not None and mem_t is not None:
                    memory_time = torch.cat(
                        [mem_t.to(device=memory_time.device, dtype=memory_time.dtype), memory_time], dim=0
                    )
                has_extra_memory = True
        memory, selected_memory_local, selected_memory_time = self._select_memory(query_start, memory_all, memory_time=memory_time, query_time=query_time)
        if selected_memory_time is not None:
            memory_time_for_attn = selected_memory_time
        else:
            memory_time_for_attn = memory_time if memory.dim() == 3 and memory.size(1) == memory_all.size(1) else None
        # Active-rows execution (adaptive multirate): queries are independent here
        # (cross-attn into memory + pointwise FFN), so computing only the kept rows is
        # exact. Disabled when the memory write-back is on (its ctx is a mean over ALL
        # queries). Memory selection above already ran on the full query set, so the
        # selected memory is identical to the full path.
        keep = None
        n_act = -1
        if query_keep is not None and not self.update_l2_enable:
            keep, n_act = query_keep  # n_act precomputed by the caller — NO host sync here
            if n_act == 0:
                # every query row of this level is frozen; all writes would be discarded
                return x, {"steps": 0, "memory_selected": 0, "memory_updated": 0, "query_nodes": 0,
                           "query_level": int(self.query_level), "memory_level": int(self.memory_level)}
            if n_act < 0 or n_act >= keep.size(1):
                keep = None
        order = None
        pad_valid = None
        if keep is not None:
            # Stable argsort puts kept rows first (order preserved); the tail pads each
            # batch row to a rectangle with DISTINCT inactive rows, whose redundant
            # updates are discarded on write-back.
            order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :n_act]
            pad_valid = keep.gather(1, order)
            query_start = query_start.gather(1, order.unsqueeze(-1).expand(-1, -1, query_start.size(-1)))
            if query_time is not None:
                qt_full = query_time.view(1, -1).expand(x.size(0), -1) if query_time.dim() == 1 else query_time
                query_time = qt_full.gather(1, order)
            if memory.dim() == 4:
                memory = memory.gather(1, order.view(order.size(0), n_act, 1, 1).expand(-1, -1, memory.size(2), memory.size(3)))
                if memory_time_for_attn is not None and memory_time_for_attn.dim() == 3:
                    memory_time_for_attn = memory_time_for_attn.gather(1, order.unsqueeze(-1).expand(-1, -1, memory_time_for_attn.size(2)))
        query = query_start
        for step in range(self.steps):
            block_idx = 0 if self.shared_weights else min(step, len(self.q_projs) - 1)
            query = query + self.dropout(self._cross_attention(query, memory, query_time, memory_time_for_attn, block_idx))
            query = query + self.dropout(self.ffns[block_idx](self.ffn_norms[block_idx](query)))
        updated = query_start + self.write_scale.to(device=query.device, dtype=query.dtype) * (query - query_start)
        out = x.clone()
        if order is not None:
            updated = torch.where(pad_valid.unsqueeze(-1), updated, query_start)
            flat_pos = query_idx.view(1, -1).expand(x.size(0), -1).gather(1, order)
            out.scatter_(1, flat_pos.unsqueeze(-1).expand(-1, -1, out.size(-1)), updated.to(dtype=out.dtype))
        else:
            out.index_copy_(1, query_idx, updated.to(dtype=out.dtype))
        memory_updated = 0
        if self.update_l2_enable and (not has_extra_memory) and memory.dim() == 3 and selected_memory_local is not None and selected_memory_local.numel() > 0 and int(self.memory_level) != 0:
            ctx = query.mean(dim=1, keepdim=True)
            delta = self.l2_update_proj(self.l2_update_norm(ctx)).expand(-1, memory.size(1), -1)
            gate = torch.sigmoid(self.l2_update_gate(torch.cat([memory, delta], dim=-1)))
            memory_updated_tensor = memory + self.l2_update_scale.to(device=memory.device, dtype=memory.dtype) * gate * delta
            for batch_idx in range(out.size(0)):
                global_idx = memory_idx.index_select(0, selected_memory_local[batch_idx].to(device=memory_idx.device, dtype=torch.long))
                out[batch_idx].index_copy_(0, global_idx, memory_updated_tensor[batch_idx].to(dtype=out.dtype))
            memory_updated = int(selected_memory_local.size(1))
        return out, {
            "steps": int(self.steps),
            "l2_selected": int(memory.size(1)) if int(self.memory_level) == 2 else 0,
            "l2_updated": int(memory_updated) if int(self.memory_level) == 2 else 0,
            "l3_nodes": int(query_idx.numel()) if int(self.query_level) == 3 else 0,
            "memory_selected": int(memory.size(1) * memory.size(2)) if memory.dim() == 4 else int(memory.size(1)),
            "memory_updated": int(memory_updated),
            "query_nodes": int(query_idx.numel()),
            "query_level": int(self.query_level),
            "memory_level": int(self.memory_level),
            "selection_mode": 1 if self.selection_mode == "per_query" else 0,
        }


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True):
        super().__init__()
        k = max(1, int(kernel_size))
        self.left_pad = max(0, k - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=max(1, int(stride)),
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over channels, independently per time step."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        c = int(channels)
        self.norm1 = ChannelLayerNorm1d(c)
        self.norm2 = ChannelLayerNorm1d(c)
        self.conv1 = CausalConv1d(c, c, kernel_size=kernel_size)
        self.conv2 = CausalConv1d(c, c, kernel_size=kernel_size)
        self.dropout = nn.Dropout(max(0.0, float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x), approximate="tanh"))
        h = self.dropout(h)
        h = self.conv2(F.gelu(self.norm2(h), approximate="tanh"))
        return x + h


class NonCausalResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        c = int(channels)
        k = max(1, int(kernel_size))
        pad = k // 2
        self.norm1 = ChannelLayerNorm1d(c)
        self.norm2 = ChannelLayerNorm1d(c)
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=pad, bias=True)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=pad, bias=True)
        self.dropout = nn.Dropout(max(0.0, float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x), approximate="tanh"))
        h = self.dropout(h)
        h = self.conv2(F.gelu(self.norm2(h), approximate="tanh"))
        return x + h


class HyenaOperator(nn.Module):
    """Causal depthwise long convolution with an implicit, length-agnostic filter,
    evaluated by FFT in O(T log T). Roughly a single-order Hyena operator.

    Shape: [B, C, T] -> [B, C, T] (channels-first, matching the token U-Net).

    The filter h[C, T] is produced by a small MLP over positional features (so it is
    recomputed for any length T and carries no length-specific parameters) and shaped by
    a learnable per-channel exponential decay window (some channels stay near-global, others
    local). Causality: both h and the signal are right-padded to >= 2T and only the first T
    outputs are kept, so y[t] depends solely on inputs <= t -- no future leakage. The FFT
    runs in fp32 for numerical stability even under bf16 autocast.

    `kernel_size` is accepted for signature compatibility with CausalResBlock1d but is
    unused: the long conv is global, not a fixed local window.
    """

    def __init__(
        self,
        channels: int,
        filter_mlp_dim: int = 64,
        pos_emb_dim: int = 33,
        min_decay: float = 0.5,
        max_decay: float = 6.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        length_invariant: bool = False,
    ):
        super().__init__()
        c = int(channels)
        self.channels = c
        # length_invariant: absolute sinusoidal positions + a per-channel decay in ABSOLUTE
        # tokens (learnable inverse time-constant), so h[c,t] depends only on absolute t. This
        # makes the filter (and thus the past outputs) stable as the sequence GROWS -> safe for
        # incremental / KV-cache generation. Default False keeps the length-normalized behaviour.
        self.length_invariant = bool(length_invariant)
        if self.length_invariant:
            # per-channel decay rate spread geometrically from ~1/8 (local) to ~1/2048 (global).
            self.log_inv_tau = nn.Parameter(
                torch.linspace(math.log(1.0 / 8.0), math.log(1.0 / 2048.0), c)
            )
        # The FFT (fp32, padded to next_pow2(2T)) materializes large complex tensors at the
        # full sequence length; storing them for backward is the dominant memory cost of a
        # full-length stem. Checkpointing recomputes the FFT in backward instead.
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.pos_emb_dim = int(pos_emb_dim)
        # Data-controlled gates + value branch (order-2 Hyena style).
        self.in_proj = nn.Linear(c, 3 * c)
        self.out_proj = nn.Linear(c, c)
        self.dropout = nn.Dropout(max(0.0, float(dropout)))
        # Implicit filter: positional features -> per-channel filter values.
        self.filter_mlp = nn.Sequential(
            nn.Linear(self.pos_emb_dim, int(filter_mlp_dim)),
            nn.GELU(),
            nn.Linear(int(filter_mlp_dim), int(filter_mlp_dim)),
            nn.GELU(),
            nn.Linear(int(filter_mlp_dim), c),
        )
        # Per-channel decay (kept positive via softplus); spread from near-global to local.
        self.decay_raw = nn.Parameter(torch.linspace(float(min_decay), float(max_decay), c))
        self._pos_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def _pos_features(self, length: int, device: torch.device) -> torch.Tensor:
        key = (int(length), str(device))
        cached = self._pos_cache.get(key, None)
        if cached is not None:
            return cached
        L = max(1, int(length))
        if self.length_invariant:
            # ABSOLUTE sinusoidal features (bounded, length-invariant): position t always maps
            # to the same feature vector regardless of L.
            t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(-1)  # [L,1] absolute
            feats = []
            n_freqs = max(1, self.pos_emb_dim // 2)
            for i in range(n_freqs):
                freq = 1.0 / (10000.0 ** (2.0 * i / self.pos_emb_dim))
                feats.append(torch.sin(freq * t))
                feats.append(torch.cos(freq * t))
            f = torch.cat(feats, dim=-1)  # [L, 2*n_freqs]
        else:
            t = torch.linspace(0.0, 1.0, steps=L, device=device, dtype=torch.float32).unsqueeze(-1)  # [L,1]
            feats = [t]
            n_freqs = (self.pos_emb_dim - 1) // 2
            for i in range(n_freqs):
                freq = (2.0 ** i) * math.pi
                feats.append(torch.sin(freq * t))
                feats.append(torch.cos(freq * t))
            f = torch.cat(feats, dim=-1)  # [L, 1 + 2*n_freqs]
        if f.size(-1) < self.pos_emb_dim:
            f = F.pad(f, (0, self.pos_emb_dim - f.size(-1)))
        else:
            f = f[:, : self.pos_emb_dim]
        # Only cache during training, where the length is fixed (one entry per resolution).
        # Generation hits a new length every step; caching those would accumulate resident GPU
        # tensors that never free. The features are cheap O(L) sin/cos -- recompute in eval.
        if self.training:
            self._pos_cache[key] = f
        return f

    def _filter(self, length: int, device: torch.device) -> torch.Tensor:
        L = max(1, int(length))
        pos = self._pos_features(L, device)            # [L, P]
        h = self.filter_mlp(pos)                       # [L, C]
        if self.length_invariant:
            # Decay in ABSOLUTE tokens: window[t,c] = exp(-inv_tau_c * t), depends only on t.
            t_abs = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(-1)  # [L,1]
            inv_tau = torch.exp(self.log_inv_tau).clamp(min=1e-6, max=1.0).unsqueeze(0)  # [1,C]
            window = torch.exp(-inv_tau * t_abs)       # [L, C]
        else:
            t_norm = torch.linspace(0.0, 1.0, steps=L, device=device, dtype=torch.float32).unsqueeze(-1)  # [L,1]
            alpha = F.softplus(self.decay_raw).clamp(min=1e-4).unsqueeze(0)  # [1, C]
            window = torch.exp(-alpha * t_norm)        # [L, C], length-relative shape
        h = h * window
        return h.transpose(0, 1).contiguous()          # [C, L]

    @staticmethod
    def _fft_conv(v_bct: torch.Tensor, h_ct: torch.Tensor) -> torch.Tensor:
        # Causal depthwise conv via FFT; fp32 transform for stability.
        T = int(v_bct.size(-1))
        n = 1
        while n < 2 * T:
            n <<= 1
        vf = torch.fft.rfft(v_bct.float(), n=n, dim=-1)
        hf = torch.fft.rfft(h_ct.float(), n=n, dim=-1).unsqueeze(0)  # [1, C, nf]
        y = torch.fft.irfft(vf * hf, n=n, dim=-1)[..., :T]
        return y.to(v_bct.dtype)

    def _forward_impl(self, x_bct: torch.Tensor) -> torch.Tensor:
        B, C, T = x_bct.shape
        x = x_bct.transpose(1, 2)                       # [B,T,C]
        v, g_pre, g_post = self.in_proj(x).chunk(3, dim=-1)
        v = (v * torch.sigmoid(g_pre)).transpose(1, 2)  # [B,C,T] data-controlled pre-gate
        h = self._filter(T, x_bct.device)               # [C,T]
        y = self._fft_conv(v, h).transpose(1, 2)        # [B,T,C]
        y = y * torch.sigmoid(g_post)                   # post-gate
        y = self.dropout(self.out_proj(y))
        return y.transpose(1, 2).contiguous()           # [B,C,T]

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x_bct, use_reentrant=False)
        return self._forward_impl(x_bct)


class HyenaResBlock1d(nn.Module):
    """Drop-in replacement for CausalResBlock1d using a global (FFT) Hyena mixer plus a
    pointwise FFN. Same signature and [B,C,T] -> [B,C,T] contract, so it slots straight
    into CausalTokenUNet's enc/bottleneck/up block lists."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
        filter_mlp_dim: int = 64,
        pos_emb_dim: int = 33,
        gradient_checkpointing: bool = False,
        length_invariant: bool = False,
    ):
        super().__init__()
        c = int(channels)
        self.norm1 = ChannelLayerNorm1d(c)
        self.norm2 = ChannelLayerNorm1d(c)
        self.hyena = HyenaOperator(
            c, filter_mlp_dim=filter_mlp_dim, pos_emb_dim=pos_emb_dim, dropout=dropout,
            gradient_checkpointing=gradient_checkpointing, length_invariant=length_invariant,
        )
        # Pointwise position-wise FFN (kernel_size=1 stays causal).
        self.ff1 = CausalConv1d(c, 2 * c, kernel_size=1)
        self.ff2 = CausalConv1d(2 * c, c, kernel_size=1)
        self.dropout = nn.Dropout(max(0.0, float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.hyena(F.gelu(self.norm1(x), approximate="tanh"))
        h = self.ff2(F.gelu(self.ff1(F.gelu(self.norm2(x), approximate="tanh")), approximate="tanh"))
        return x + self.dropout(h)


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm-like normalization over channels for 2D feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class CausalMaskedConv2d(nn.Module):
    """
    Raster-causal masked 2D convolution.
    For each output (y, x), only consumes positions <= (y, x) in raster order.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        include_center: bool = True,
    ):
        super().__init__()
        k = max(1, int(kernel_size))
        pad = k // 2
        self.conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=k,
            stride=max(1, int(stride)),
            padding=pad,
            bias=bool(bias),
        )
        mask = torch.ones_like(self.conv.weight)
        cy = k // 2
        cx = k // 2
        if cy + 1 < k:
            mask[:, :, cy + 1 :, :] = 0
        if cx + 1 < k:
            mask[:, :, cy, cx + 1 :] = 0
        if not bool(include_center):
            mask[:, :, cy, cx] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight * self.mask
        return F.conv2d(
            x,
            w,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class CausalResBlock2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        c = int(channels)
        self.norm1 = ChannelLayerNorm2d(c)
        self.norm2 = ChannelLayerNorm2d(c)
        self.conv1 = CausalMaskedConv2d(c, c, kernel_size=kernel_size, include_center=True)
        self.conv2 = CausalMaskedConv2d(c, c, kernel_size=kernel_size, include_center=True)
        self.dropout = nn.Dropout(max(0.0, float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x), approximate="tanh"))
        h = self.dropout(h)
        h = self.conv2(F.gelu(self.norm2(h), approximate="tanh"))
        return x + h


class NonCausalResBlock2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        c = int(channels)
        k = max(1, int(kernel_size))
        pad = k // 2
        self.norm1 = ChannelLayerNorm2d(c)
        self.norm2 = ChannelLayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, kernel_size=k, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=k, padding=pad, bias=True)
        self.dropout = nn.Dropout(max(0.0, float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x), approximate="tanh"))
        h = self.dropout(h)
        h = self.conv2(F.gelu(self.norm2(h), approximate="tanh"))
        return x + h


class SpatialTokenUNet2D(nn.Module):
    """
    Spatial 2D token U-Net.
    Input/Output shape: [B, T, H] with explicit or inferred grid shape (gh, gw), T=gh*gw.
    """

    def __init__(
        self,
        hidden_dim: int,
        scale: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.0,
        lookahead_enable: bool = False,
        lookahead_kernel_size: int = 5,
        lookahead_blocks: int = 2,
        causal: bool = False,
    ):
        super().__init__()
        if not _is_power_of_two(int(scale)):
            raise ValueError(f"SpatialTokenUNet2D requires scale as power-of-two, got {scale}")
        self.hidden_dim = int(hidden_dim)
        self.scale = int(scale)
        self.n_down = int(math.log2(self.scale))
        self.lookahead_enable = bool(lookahead_enable)
        self.lookahead_kernel_size = int(lookahead_kernel_size)
        self.lookahead_blocks = int(lookahead_blocks)
        self.causal = bool(causal)

        block_cls = CausalResBlock2d if self.causal else NonCausalResBlock2d
        self.enc_blocks = nn.ModuleList(
            [block_cls(self.hidden_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(self.n_down)]
        )
        if self.causal:
            self.downsamplers = nn.ModuleList(
                [CausalMaskedConv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, include_center=True) for _ in range(self.n_down)]
            )
        else:
            self.downsamplers = nn.ModuleList(
                [nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=True) for _ in range(self.n_down)]
            )

        self.bottleneck = block_cls(self.hidden_dim, kernel_size=kernel_size, dropout=dropout)

        self.up_blocks = nn.ModuleList(
            [block_cls(self.hidden_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(self.n_down)]
        )
        self.out_norm = ChannelLayerNorm2d(self.hidden_dim)
        if self.causal:
            self.out_proj = CausalMaskedConv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, include_center=True)
        else:
            self.out_proj = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, padding=0, bias=True)

        self.lookahead_refine = nn.ModuleList()
        if self.lookahead_enable and self.lookahead_blocks > 0:
            self.lookahead_refine = nn.ModuleList(
                [
                    NonCausalResBlock2d(
                        self.hidden_dim,
                        kernel_size=self.lookahead_kernel_size,
                        dropout=dropout,
                    )
                    for _ in range(self.lookahead_blocks)
                ]
            )

    def _infer_grid_shape(self, token_len: int, grid_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        if grid_shape is not None:
            gh = int(grid_shape[0])
            gw = int(grid_shape[1])
            if gh > 0 and gw > 0 and gh * gw == int(token_len):
                return gh, gw
        side = int(round(math.sqrt(max(1, int(token_len)))))
        if side * side == int(token_len):
            return side, side
        return int(token_len), 1

    def _bt_to_bchw(
        self,
        x_bth: torch.Tensor,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x_bth.dim() != 3:
            raise ValueError(f"SpatialTokenUNet2D expects [B,T,H], got {tuple(x_bth.shape)}")
        bsz, tok, hid = x_bth.shape
        gh, gw = self._infer_grid_shape(int(tok), grid_shape=grid_shape)
        if gh * gw != int(tok):
            raise ValueError(f"Invalid grid_shape ({gh},{gw}) for token length {tok}")
        x = x_bth.transpose(1, 2).reshape(bsz, hid, gh, gw).contiguous()
        return x, (gh, gw)

    def _bchw_to_bt(self, x_bchw: torch.Tensor) -> torch.Tensor:
        bsz, hid, gh, gw = x_bchw.shape
        return x_bchw.reshape(bsz, hid, gh * gw).transpose(1, 2).contiguous()

    def _resize_to_hw(self, x_bchw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        th, tw = int(target_hw[0]), int(target_hw[1])
        if int(x_bchw.size(-2)) == th and int(x_bchw.size(-1)) == tw:
            return x_bchw
        return F.interpolate(x_bchw, size=(th, tw), mode="nearest")

    def encode(
        self,
        x_bth: torch.Tensor,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, in_hw = self._bt_to_bchw(x_bth, grid_shape=grid_shape)
        skips: List[torch.Tensor] = []
        for enc, down in zip(self.enc_blocks, self.downsamplers):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        tokens = self._bchw_to_bt(x)
        context = {
            "skips": skips,
            "target_grid_shape": (int(in_hw[0]), int(in_hw[1])),
            "coarse_grid_shape": (int(x.size(-2)), int(x.size(-1))),
            "graph_grid_shape": (int(x.size(-2)), int(x.size(-1))),
        }
        return tokens, context

    def _decode_core(self, h_bth: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        coarse_hw = context.get("coarse_grid_shape", None)
        x, _ = self._bt_to_bchw(h_bth, grid_shape=coarse_hw)
        skips = list(context.get("skips", []))
        for i in range(self.n_down - 1, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i < len(skips):
                x = self._resize_to_hw(x, (int(skips[i].size(-2)), int(skips[i].size(-1))))
                x = x + skips[i]
            x = self.up_blocks[i](x)
        target_hw = context.get("target_grid_shape", (int(x.size(-2)), int(x.size(-1))))
        x = self._resize_to_hw(x, (int(target_hw[0]), int(target_hw[1])))
        x = self.out_proj(F.gelu(self.out_norm(x), approximate="tanh"))
        return x

    def decode(self, h_bth: torch.Tensor, context: Dict[str, Any], mode: str = "strict") -> torch.Tensor:
        mode_norm = str(mode).lower()
        strict_bchw = self._decode_core(h_bth, context)
        if mode_norm in {"strict", "causal"}:
            return self._bchw_to_bt(strict_bchw)
        if mode_norm in {"lookahead", "future"}:
            if not self.lookahead_enable or len(self.lookahead_refine) == 0:
                return self._bchw_to_bt(strict_bchw)
            x = strict_bchw
            for block in self.lookahead_refine:
                x = block(x)
            return self._bchw_to_bt(x)
        raise ValueError(f"Unknown SpatialTokenUNet2D decode mode: {mode}")

    def decode_dual(self, h_bth: torch.Tensor, context: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        strict = self.decode(h_bth, context, mode="strict")
        if not self.lookahead_enable or len(self.lookahead_refine) == 0:
            return strict, None
        lookahead = self.decode(h_bth, context, mode="lookahead")
        return strict, lookahead

    def forward(self, x_bth: torch.Tensor, grid_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        tokens, context = self.encode(x_bth, grid_shape=grid_shape)
        return self.decode(tokens, context, mode="strict")


class RGBUNetConvBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        k = max(1, int(kernel_size))
        pad = k // 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm1 = nn.GroupNorm(1, int(in_channels))
        self.act1 = nn.GELU(approximate="tanh")
        self.conv1 = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=k, padding=pad, bias=True)
        self.drop = nn.Dropout2d(max(0.0, float(dropout)))
        self.norm2 = nn.GroupNorm(1, int(out_channels))
        self.act2 = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(int(out_channels), int(out_channels), kernel_size=k, padding=pad, bias=True)
        cd = None if cond_dim is None or int(cond_dim) <= 0 else int(cond_dim)
        self.cond1 = nn.Linear(cd, 2 * self.in_channels) if cd is not None else None
        self.cond2 = nn.Linear(cd, 2 * self.out_channels) if cd is not None else None
        if self.cond1 is not None:
            nn.init.zeros_(self.cond1.weight)
            nn.init.zeros_(self.cond1.bias)
        if self.cond2 is not None:
            nn.init.zeros_(self.cond2.weight)
            nn.init.zeros_(self.cond2.bias)

    def _apply_cond(self, x: torch.Tensor, cond: Optional[torch.Tensor], proj: Optional[nn.Linear]) -> torch.Tensor:
        if cond is None or proj is None:
            return x
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if int(cond.size(0)) != int(x.size(0)):
            if int(cond.size(0)) == 1:
                cond = cond.expand(int(x.size(0)), -1)
            else:
                cond = cond[: int(x.size(0))]
        gb = proj(cond.to(device=x.device, dtype=proj.weight.dtype)).to(dtype=x.dtype)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x)
        x = self._apply_cond(x, cond, self.cond1)
        x = self.conv1(self.act1(x))
        x = self.drop(x)
        x = self.norm2(x)
        x = self._apply_cond(x, cond, self.cond2)
        x = self.conv2(self.act2(x))
        return x


class RGBUNetSeparableConvBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        c = int(channels)
        self.channels = c
        k = max(1, int(kernel_size))
        pad = k // 2
        self.norm = nn.GroupNorm(1, c)
        self.act = nn.GELU(approximate="tanh")
        self.depthwise = nn.Conv2d(c, c, kernel_size=k, padding=pad, groups=c, bias=False)
        self.pointwise = nn.Conv2d(c, c, kernel_size=1, padding=0, bias=True)
        self.drop = nn.Dropout2d(max(0.0, float(dropout)))
        cd = None if cond_dim is None or int(cond_dim) <= 0 else int(cond_dim)
        self.cond = nn.Linear(cd, 2 * c) if cd is not None else None
        if self.cond is not None:
            nn.init.zeros_(self.cond.weight)
            nn.init.zeros_(self.cond.bias)

    def _apply_cond(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None or self.cond is None:
            return x
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if int(cond.size(0)) != int(x.size(0)):
            if int(cond.size(0)) == 1:
                cond = cond.expand(int(x.size(0)), -1)
            else:
                cond = cond[: int(x.size(0))]
        gb = self.cond(cond.to(device=x.device, dtype=self.cond.weight.dtype)).to(dtype=x.dtype)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        x = self._apply_cond(x, cond)
        x = self.act(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.drop(x)
        return x


class RGBUNetDownsampleBlock2D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0, cond_dim: Optional[int] = None):
        super().__init__()
        c = int(channels)
        self.channels = c
        k = max(1, int(kernel_size))
        pad = k // 2
        self.norm = nn.GroupNorm(1, c)
        self.act = nn.GELU(approximate="tanh")
        self.conv = nn.Conv2d(c, c, kernel_size=k, stride=2, padding=pad, bias=True)
        self.drop = nn.Dropout2d(max(0.0, float(dropout)))
        cd = None if cond_dim is None or int(cond_dim) <= 0 else int(cond_dim)
        self.cond = nn.Linear(cd, 2 * c) if cd is not None else None
        if self.cond is not None:
            nn.init.zeros_(self.cond.weight)
            nn.init.zeros_(self.cond.bias)

    def _apply_cond(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None or self.cond is None:
            return x
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if int(cond.size(0)) != int(x.size(0)):
            if int(cond.size(0)) == 1:
                cond = cond.expand(int(x.size(0)), -1)
            else:
                cond = cond[: int(x.size(0))]
        gb = self.cond(cond.to(device=x.device, dtype=self.cond.weight.dtype)).to(dtype=x.dtype)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        x = self._apply_cond(x, cond)
        x = self.act(x)
        x = self.conv(x)
        x = self.drop(x)
        return x


class RGBTokenUNet2D(nn.Module):
    """Dynamic RGB<->token U-Net bridge.

    Mirrors the dynamic U-Net structure used in the reference implementation:
    - dynamic depth from downsample_factor (power-of-two)
    - explicit per-stage modules for easy hook-based explainability.
    - narrower high-resolution decoder stages with learned stride-2 downsampling
    """

    def __init__(
        self,
        token_dim: int,
        downsample_factor: int = 16,
        base_channels: int = 64,
        kernel_size: int = 5,
        decode_kernel_size: int = 3,
        decode_separable: bool = True,
        max_channels: int = 512,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        if not _is_power_of_two(int(downsample_factor)):
            raise ValueError(f"RGBTokenUNet2D requires power-of-two downsample_factor, got {downsample_factor}")
        self.token_dim = int(token_dim)
        self.downsample_factor = int(downsample_factor)
        self.n_down = int(math.log2(self.downsample_factor))
        self.base_channels = max(16, int(base_channels))
        self.max_channels = max(self.base_channels, int(max_channels))
        self.kernel_size = max(1, int(kernel_size))
        self.decode_kernel_size = max(1, int(decode_kernel_size))
        self.decode_separable = bool(decode_separable)
        self.dropout = float(dropout)
        self.cond_dim = None if cond_dim is None or int(cond_dim) <= 0 else int(cond_dim)

        self.encoder_channels = [self.base_channels]
        for _ in range(self.n_down):
            self.encoder_channels.append(min(self.encoder_channels[-1] * 2, self.max_channels))
        self.decoder_channels = list(reversed(self.encoder_channels[1:]))

        self.stem = nn.Conv2d(3, self.base_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, bias=True)

        self.down_blocks = nn.ModuleDict()
        self.skip_projs = nn.ModuleDict()
        self.downsamples = nn.ModuleDict()

        for i in range(self.n_down):
            in_ch = int(self.encoder_channels[i])
            out_ch = int(self.encoder_channels[i + 1])
            self.down_blocks[f"down_block_{i}"] = RGBUNetConvBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                cond_dim=self.cond_dim,
            )
            self.skip_projs[f"skip_proj_{i}"] = nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=True)
            self.downsamples[f"downsample_{i}"] = RGBUNetDownsampleBlock2D(
                channels=out_ch,
                kernel_size=3,
                dropout=self.dropout,
                cond_dim=self.cond_dim,
            )

        self.to_tokens = RGBUNetConvBlock2D(
            in_channels=int(self.encoder_channels[-1]),
            out_channels=self.token_dim,
            kernel_size=1,
            dropout=self.dropout,
            cond_dim=self.cond_dim,
        )

        self.up_pre = nn.ModuleDict()
        self.up_post = nn.ModuleDict()
        self.skip_gates = nn.ParameterList()
        current_ch = self.token_dim
        for i in range(self.n_down):
            self.up_pre[f"up_pre_{i}"] = RGBUNetConvBlock2D(
                in_channels=current_ch,
                out_channels=int(self.decoder_channels[i]),
                kernel_size=1,
                dropout=self.dropout,
                cond_dim=self.cond_dim,
            )
            if self.decode_separable:
                self.up_post[f"up_post_{i}"] = RGBUNetSeparableConvBlock2D(
                    channels=int(self.decoder_channels[i]),
                    kernel_size=self.decode_kernel_size,
                    dropout=self.dropout,
                    cond_dim=self.cond_dim,
                )
            else:
                self.up_post[f"up_post_{i}"] = RGBUNetConvBlock2D(
                    in_channels=int(self.decoder_channels[i]),
                    out_channels=int(self.decoder_channels[i]),
                    kernel_size=self.decode_kernel_size,
                    dropout=self.dropout,
                    cond_dim=self.cond_dim,
                )
            self.skip_gates.append(nn.Parameter(torch.tensor(-4.0)))
            current_ch = int(self.decoder_channels[i])

        final_rgb_in_channels = int(self.decoder_channels[-1]) if self.n_down > 0 else self.token_dim
        self.to_rgb = nn.Conv2d(final_rgb_in_channels, 3, kernel_size=1, padding=0, bias=True)

    def _bt_to_bchw(self, x_bth: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        bsz, tok, hid = x_bth.shape
        gh = int(grid_shape[0])
        gw = int(grid_shape[1])
        if gh * gw != int(tok):
            raise ValueError(f"Invalid grid_shape {grid_shape} for token length {tok}")
        return x_bth.transpose(1, 2).reshape(bsz, hid, gh, gw).contiguous()

    def _bchw_to_bt(self, x_bchw: torch.Tensor) -> torch.Tensor:
        bsz, hid, gh, gw = x_bchw.shape
        return x_bchw.reshape(bsz, hid, gh * gw).transpose(1, 2).contiguous()

    def _resize_to_hw(self, x_bchw: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        th = int(target_hw[0])
        tw = int(target_hw[1])
        if int(x_bchw.size(-2)) == th and int(x_bchw.size(-1)) == tw:
            return x_bchw
        return F.interpolate(x_bchw, size=(th, tw), mode="nearest")

    def encode(self, pixel_values: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if pixel_values.dim() != 4 or int(pixel_values.size(1)) != 3:
            raise ValueError(f"RGBTokenUNet2D.encode expects [B,3,H,W], got {tuple(pixel_values.shape)}")
        x = self.stem(pixel_values)
        skips: List[torch.Tensor] = []
        for i in range(self.n_down):
            x = self.down_blocks[f"down_block_{i}"](x, cond=cond)
            skips.append(self.skip_projs[f"skip_proj_{i}"](x))
            x = self.downsamples[f"downsample_{i}"](x, cond=cond)
        x = self.to_tokens(x, cond=cond)
        tokens = self._bchw_to_bt(x)
        context = {
            "skips": skips,
            "target_hw": (int(pixel_values.size(-2)), int(pixel_values.size(-1))),
            "coarse_grid_shape": (int(x.size(-2)), int(x.size(-1))),
            "graph_grid_shape": (int(x.size(-2)), int(x.size(-1))),
        }
        return tokens, context

    def decode(
        self,
        token_features: torch.Tensor,
        context: Dict[str, Any],
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coarse_hw = context.get("coarse_grid_shape", None)
        if coarse_hw is None:
            raise ValueError("RGBTokenUNet2D.decode requires coarse_grid_shape in context")
        x = self._bt_to_bchw(token_features, grid_shape=(int(coarse_hw[0]), int(coarse_hw[1])))
        skips = list(context.get("skips", []))
        for stage_order in range(self.n_down):
            skip_idx = self.n_down - 1 - stage_order
            x = self.up_pre[f"up_pre_{stage_order}"](x, cond=cond)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if 0 <= skip_idx < len(skips):
                skip = self._resize_to_hw(skips[skip_idx], (int(x.size(-2)), int(x.size(-1))))
                skip_gate = torch.sigmoid(self.skip_gates[stage_order])
                x = x + skip_gate.view(1, 1, 1, 1) * skip
            x = self.up_post[f"up_post_{stage_order}"](x, cond=cond)
        target_hw = context.get("target_hw", (int(x.size(-2)), int(x.size(-1))))
        x = self._resize_to_hw(x, (int(target_hw[0]), int(target_hw[1])))
        return self.to_rgb(x)


class CausalTokenUNet(nn.Module):
    """
    Causal 1D U-Net for token features.

    Input/Output shape: [B, T, H]
    - Uses causal convolutions only (no future leakage).
    - Downsamples by powers of two, then upsamples back to original length.
    """

    def __init__(
        self,
        hidden_dim: int,
        scale: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.0,
        lookahead_enable: bool = False,
        lookahead_kernel_size: int = 5,
        lookahead_blocks: int = 2,
        block: str = "cnn",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if not _is_power_of_two(int(scale)):
            raise ValueError(f"CausalTokenUNet requires scale as power-of-two, got {scale}")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.hidden_dim = int(hidden_dim)
        self.scale = int(scale)
        self.n_down = int(math.log2(self.scale))
        self.lookahead_enable = bool(lookahead_enable)
        self.lookahead_kernel_size = int(lookahead_kernel_size)
        self.lookahead_blocks = int(lookahead_blocks)
        block_norm = str(block).lower()
        if block_norm not in {"cnn", "hyena"}:
            logger.warning("Unknown CausalTokenUNet block='%s'; falling back to 'cnn'", block)
            block_norm = "cnn"
        self.block = block_norm

        def _resblock() -> nn.Module:
            # enc/bottleneck/up blocks: causal global (Hyena) or causal local (CNN).
            # Downsamplers stay strided CausalConv1d, so Hyena mixes globally *before*
            # each pool (mode 1: global-aware pooling); at scale=1 only the bottleneck
            # runs, giving a single token-level global op (mode 2).
            if self.block == "hyena":
                return HyenaResBlock1d(
                    self.hidden_dim, kernel_size=kernel_size, dropout=dropout,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
            return CausalResBlock1d(self.hidden_dim, kernel_size=kernel_size, dropout=dropout)

        self.enc_blocks = nn.ModuleList([_resblock() for _ in range(self.n_down)])
        self.downsamplers = nn.ModuleList(
            [CausalConv1d(self.hidden_dim, self.hidden_dim, kernel_size=2, stride=2) for _ in range(self.n_down)]
        )

        self.bottleneck = _resblock()

        self.up_blocks = nn.ModuleList([_resblock() for _ in range(self.n_down)])
        self.out_norm = ChannelLayerNorm1d(self.hidden_dim)
        self.out_proj = CausalConv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)

        self.lookahead_refine = nn.ModuleList()
        if self.lookahead_enable and self.lookahead_blocks > 0:
            self.lookahead_refine = nn.ModuleList(
                [
                    NonCausalResBlock1d(
                        self.hidden_dim,
                        kernel_size=self.lookahead_kernel_size,
                        dropout=dropout,
                    )
                    for _ in range(self.lookahead_blocks)
                ]
            )

    def _resize_to_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        cur_len = int(x.size(-1))
        if cur_len == int(target_len):
            return x
        if cur_len > int(target_len):
            return x[..., : int(target_len)]
        pad_amt = int(target_len) - cur_len
        return F.pad(x, (0, pad_amt))

    def encode(self, x_bth: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if x_bth.dim() != 3:
            raise ValueError(f"CausalTokenUNet.encode expects [B,T,H], got shape {tuple(x_bth.shape)}")

        x = x_bth.transpose(1, 2).contiguous()  # [B,H,T]
        skips = []

        for enc, down in zip(self.enc_blocks, self.downsamplers):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)
        tokens = x.transpose(1, 2).contiguous()  # [B,Tc,H]
        context = {
            "skips": skips,
            "target_len": int(x_bth.size(1)),
        }
        return tokens, context

    def _decode_causal(self, h_bth: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        if h_bth.dim() != 3:
            raise ValueError(f"CausalTokenUNet.decode expects [B,T,H], got shape {tuple(h_bth.shape)}")

        x = h_bth.transpose(1, 2).contiguous()  # [B,H,Tc]
        skips = list(context.get("skips", []))

        for i in range(self.n_down - 1, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i < len(skips):
                x = self._resize_to_length(x, int(skips[i].size(-1)))
                x = x + skips[i]
            x = self.up_blocks[i](x)

        target_len = int(context.get("target_len", x.size(-1)))
        x = self._resize_to_length(x, target_len)
        x = self.out_proj(F.gelu(self.out_norm(x), approximate="tanh"))
        return x.transpose(1, 2).contiguous()

    def decode(self, h_bth: torch.Tensor, context: Dict[str, Any], mode: str = "strict") -> torch.Tensor:
        mode_norm = str(mode).lower()
        strict = self._decode_causal(h_bth, context)
        if mode_norm in {"strict", "causal"}:
            return strict
        if mode_norm in {"lookahead", "future"}:
            if not self.lookahead_enable or len(self.lookahead_refine) == 0:
                return strict
            x = strict.transpose(1, 2).contiguous()  # [B,H,T]
            for block in self.lookahead_refine:
                x = block(x)
            return x.transpose(1, 2).contiguous()
        raise ValueError(f"Unknown CausalTokenUNet decode mode: {mode}")

    def decode_dual(self, h_bth: torch.Tensor, context: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        strict = self.decode(h_bth, context, mode="strict")
        if not self.lookahead_enable or len(self.lookahead_refine) == 0:
            return strict, None
        lookahead = self.decode(h_bth, context, mode="lookahead")
        return strict, lookahead

    def forward(self, x_bth: torch.Tensor) -> torch.Tensor:
        tokens, context = self.encode(x_bth)
        return self.decode(tokens, context, mode="strict")


class CausalDynamicTokenUNet(nn.Module):
    """True dynamic causal 1D U-Net for token features (drop-in for CausalTokenUNet).

    Unlike the constant-width CausalTokenUNet, this is a PROPER U-Net:
      - channels GROW at coarser scales (widths[i] = hidden_dim + growth*i, capped) so
        capacity is invested where multi-token structure lives;
      - skips are CONCATENATED and fused by a conv (classic U-Net), not added;
      - a channel-change 1x1 projection precedes each res-block so growth is learned.
    The graph sits in the bottleneck: encode() downsamples L0 tokens to hidden_dim coarse
    tokens (fed to the graph), decode() upsamples the refined coarse tokens back to full
    resolution, fusing the encoder skips around the graph. Everything is CAUSAL (AR-safe:
    strided causal conv down, nearest up, causal convs). Supports cnn (local causal conv) or
    hyena (global FFT) blocks. Same encode/decode/decode_dual contract as CausalTokenUNet.
    """

    def __init__(
        self,
        hidden_dim: int,
        scale: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.0,
        block: str = "cnn",
        channel_growth: int = 256,
        max_channels: int = 0,
        concat_skips: bool = True,
        lookahead_enable: bool = False,
        lookahead_kernel_size: int = 5,
        lookahead_blocks: int = 2,
        gradient_checkpointing: bool = False,
        hyena_length_invariant: bool = False,
    ):
        super().__init__()
        if not _is_power_of_two(int(scale)):
            raise ValueError(f"CausalDynamicTokenUNet requires scale power-of-two, got {scale}")
        self.hyena_length_invariant = bool(hyena_length_invariant)
        self.hidden_dim = int(hidden_dim)
        self.scale = int(scale)
        self.n_down = int(math.log2(self.scale))
        self.concat_skips = bool(concat_skips)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        block_norm = str(block).lower()
        if block_norm not in {"cnn", "hyena"}:
            logger.warning("Unknown CausalDynamicTokenUNet block='%s'; falling back to 'cnn'", block)
            block_norm = "cnn"
        self.block = block_norm
        self.lookahead_enable = bool(lookahead_enable)

        g = max(0, int(channel_growth))
        mx = int(max_channels) if int(max_channels) > 0 else 2 * self.hidden_dim
        # widths[i] = channels at encoder level i (0=finest==input, n_down=bottleneck), growing.
        widths = [min(self.hidden_dim + g * i, mx) for i in range(self.n_down + 1)]
        widths[0] = self.hidden_dim
        self.widths = widths

        def _block(ch: int) -> nn.Module:
            if self.block == "hyena":
                return HyenaResBlock1d(ch, kernel_size=kernel_size, dropout=dropout,
                                       gradient_checkpointing=self.gradient_checkpointing,
                                       length_invariant=self.hyena_length_invariant)
            return CausalResBlock1d(ch, kernel_size=kernel_size, dropout=dropout)

        def _stage(in_ch: int, out_ch: int) -> nn.ModuleList:
            # channel-change 1x1 causal proj (learned), then a constant-width res-block.
            proj = CausalConv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
            return nn.ModuleList([proj, _block(out_ch)])

        self.enc_stages = nn.ModuleList([_stage(widths[i], widths[i + 1]) for i in range(self.n_down)])
        self.downsamplers = nn.ModuleList(
            [CausalConv1d(widths[i + 1], widths[i + 1], kernel_size=2, stride=2) for i in range(self.n_down)]
        )
        self.bottleneck = _stage(widths[self.n_down], widths[self.n_down])
        # graph runs at hidden_dim: project bottleneck -> hidden_dim (tokens), and back on decode.
        self.to_tokens = CausalConv1d(widths[self.n_down], self.hidden_dim, kernel_size=1)
        self.from_tokens = CausalConv1d(self.hidden_dim, widths[self.n_down], kernel_size=1)
        # up stage i fuses (upsampled widths[i+1]) [+ skip widths[i+1] if concat] -> widths[i].
        self.up_stages = nn.ModuleList(
            [_stage(widths[i + 1] * (2 if self.concat_skips else 1), widths[i]) for i in range(self.n_down)]
        )
        self.out_norm = ChannelLayerNorm1d(self.hidden_dim)
        self.out_proj = CausalConv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)

        self.lookahead_refine = nn.ModuleList()
        if self.lookahead_enable and int(lookahead_blocks) > 0:
            self.lookahead_refine = nn.ModuleList(
                [NonCausalResBlock1d(self.hidden_dim, kernel_size=int(lookahead_kernel_size), dropout=dropout)
                 for _ in range(int(lookahead_blocks))]
            )

    @staticmethod
    def _run_stage(stage: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        proj, block = stage
        return block(proj(x))

    def _resize_to_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        cur_len = int(x.size(-1))
        if cur_len == int(target_len):
            return x
        if cur_len > int(target_len):
            return x[..., : int(target_len)]
        return F.pad(x, (0, int(target_len) - cur_len))

    def encode(self, x_bth: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if x_bth.dim() != 3:
            raise ValueError(f"CausalDynamicTokenUNet.encode expects [B,T,H], got {tuple(x_bth.shape)}")
        x = x_bth.transpose(1, 2).contiguous()  # [B,H,T]
        skips = []
        for i in range(self.n_down):
            x = self._run_stage(self.enc_stages[i], x)
            skips.append(x)
            x = self.downsamplers[i](x)
        x = self._run_stage(self.bottleneck, x)
        tokens = self.to_tokens(x).transpose(1, 2).contiguous()  # [B,Tc,H]
        return tokens, {"skips": skips, "target_len": int(x_bth.size(1))}

    def _decode_causal(self, h_bth: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        if h_bth.dim() != 3:
            raise ValueError(f"CausalDynamicTokenUNet.decode expects [B,T,H], got {tuple(h_bth.shape)}")
        x = self.from_tokens(h_bth.transpose(1, 2).contiguous())  # [B, widths[n], Tc]
        skips = list(context.get("skips", []))
        for i in range(self.n_down - 1, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i < len(skips):
                x = self._resize_to_length(x, int(skips[i].size(-1)))
                x = torch.cat([x, skips[i]], dim=1) if self.concat_skips else (x + skips[i])
            x = self._run_stage(self.up_stages[i], x)
        target_len = int(context.get("target_len", x.size(-1)))
        x = self._resize_to_length(x, target_len)
        x = self.out_proj(F.gelu(self.out_norm(x), approximate="tanh"))
        return x.transpose(1, 2).contiguous()

    def decode(self, h_bth: torch.Tensor, context: Dict[str, Any], mode: str = "strict") -> torch.Tensor:
        mode_norm = str(mode).lower()
        strict = self._decode_causal(h_bth, context)
        if mode_norm in {"strict", "causal"}:
            return strict
        if mode_norm in {"lookahead", "future"}:
            if not self.lookahead_enable or len(self.lookahead_refine) == 0:
                return strict
            x = strict.transpose(1, 2).contiguous()
            for block in self.lookahead_refine:
                x = block(x)
            return x.transpose(1, 2).contiguous()
        raise ValueError(f"Unknown CausalDynamicTokenUNet decode mode: {mode}")

    def decode_dual(self, h_bth: torch.Tensor, context: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        strict = self.decode(h_bth, context, mode="strict")
        if not self.lookahead_enable or len(self.lookahead_refine) == 0:
            return strict, None
        return strict, self.decode(h_bth, context, mode="lookahead")

    def forward(self, x_bth: torch.Tensor) -> torch.Tensor:
        tokens, context = self.encode(x_bth)
        return self.decode(tokens, context, mode="strict")


class HierarchyReconHead(nn.Module):
    """
    Reconstruct L0 features from L2/L3 ancestors using only H-sized MLPs.

    - No params depend on N0, N2, N3, or sequence length.
    - Uses simple mean over ancestor features per L0 node + a linear projection.
    """
    def __init__(self, hidden_dim: int, use_L2: bool = True, use_L3: bool = True):
        super().__init__()
        self.use_L2 = use_L2
        self.use_L3 = use_L3

        n_sources = int(use_L2) + int(use_L3)
        if n_sources == 0:
            raise ValueError("HierarchyReconHead: at least one of use_L2/use_L3 must be True.")

        self.proj = nn.Linear(n_sources * hidden_dim, hidden_dim)

    def forward(
        self,
        h0: torch.Tensor,                      # [N0, H]
        l0_to_l2: Optional[List[List[int]]] = None,  # len N0, each element: List[int]
        h2: Optional[torch.Tensor] = None,     # [N2, H]
        l0_to_l3: Optional[List[List[int]]] = None,  # len N0, each element: List[int]
        h3: Optional[torch.Tensor] = None,     # [N3, H]
    ) -> Optional[torch.Tensor]:
        """
        Returns h_hat0 [N0, H] or None if no valid sources are available.
        """
        device = h0.device
        dtype  = h0.dtype
        N0, H  = h0.shape

        parts: List[torch.Tensor] = []

        # ---- L0 <- L2 ----
        if self.use_L2 and (h2 is not None) and (l0_to_l2 is not None):
            N2 = h2.size(0)
            if len(l0_to_l2) != N0:
                raise ValueError(f"l0_to_l2 length {len(l0_to_l2)} != N0 {N0}")

            agg2 = torch.zeros(N0, H, device=device, dtype=dtype)

            for i in range(N0):
                idxs = l0_to_l2[i]
                if not idxs:
                    continue
                idx_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
                # robust against stray indices
                idx_t = idx_t[(idx_t >= 0) & (idx_t < N2)]
                if idx_t.numel() == 0:
                    continue
                agg2[i] = h2[idx_t].mean(dim=0)

            parts.append(agg2)

        # ---- L0 <- L3 ----
        if self.use_L3 and (h3 is not None) and (l0_to_l3 is not None):
            N3 = h3.size(0)
            if len(l0_to_l3) != N0:
                raise ValueError(f"l0_to_l3 length {len(l0_to_l3)} != N0 {N0}")

            agg3 = torch.zeros(N0, H, device=device, dtype=dtype)

            for i in range(N0):
                idxs = l0_to_l3[i]
                if not idxs:
                    continue
                idx_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
                idx_t = idx_t[(idx_t >= 0) & (idx_t < N3)]
                if idx_t.numel() == 0:
                    continue
                agg3[i] = h3[idx_t].mean(dim=0)

            parts.append(agg3)

        if not parts:
            return None

        # concat along feature dim and project back to H
        h_cat  = torch.cat(parts, dim=-1)  # [N0, n_sources*H]
        h_hat0 = self.proj(h_cat)          # [N0, H]
        return h_hat0


def _build_l0_to_lx_from_edges(
    node_level: torch.Tensor,   # [N], 0/1/2/3
    edge_index: torch.Tensor,   # [2, E]
    level_x: int,               # 2 or 3
) -> List[List[int]]:
    """
    Build a list-of-lists mapping L0 nodes -> L{level_x} ancestors using edges.

    For each L0 node i, l0_to_lx[i] is a list of *local indices* into h_x
    (where h_x = x[node_level == level_x]).
    """
    assert node_level.dim() == 1, "node_level must be [N]"
    N = node_level.size(0)

    lvl0_mask = (node_level == 0)
    lvlx_mask = (node_level == level_x)

    idx0 = torch.nonzero(lvl0_mask, as_tuple=False).view(-1)  # global indices
    idxx = torch.nonzero(lvlx_mask, as_tuple=False).view(-1)

    N0 = idx0.numel()
    if N0 == 0 or idxx.numel() == 0:
        return [[] for _ in range(N0)]

    # Global -> local index maps
    global_to_l0 = -torch.ones(N, dtype=torch.long, device=node_level.device)
    global_to_lx = -torch.ones(N, dtype=torch.long, device=node_level.device)

    global_to_l0[idx0] = torch.arange(N0, device=node_level.device)
    global_to_lx[idxx] = torch.arange(idxx.numel(), device=node_level.device)

    l0_to_lx: List[List[int]] = [[] for _ in range(N0)]

    src, dst = edge_index  # [E], [E]
    # Walk edges and record L0 <-> Lx connections
    # NB: we use .tolist() to make Python ints for the list-of-lists
    for s, d in zip(src.tolist(), dst.tolist()):
        ls = int(node_level[s])
        ld = int(node_level[d])

        # L0 (s) -> Lx (d)
        if ls == 0 and ld == level_x:
            i0 = int(global_to_l0[s])
            ix = int(global_to_lx[d])
            if i0 >= 0 and ix >= 0:
                l0_to_lx[i0].append(ix)

        # Lx (s) -> L0 (d)
        if ls == level_x and ld == 0:
            i0 = int(global_to_l0[d])
            ix = int(global_to_lx[s])
            if i0 >= 0 and ix >= 0:
                l0_to_lx[i0].append(ix)

    return l0_to_lx

from typing import Optional
import torch
import torch.nn.functional as F

def _compute_pair_aux_loss(
    g,
    low_level: int,
    high_level: int,
    detach_target: bool = True,
    loss_mode: str = "mse",
) -> torch.Tensor:
    """
    Fast, vectorized aux loss: predict level `low_level` features from their
    level-`high_level` neighbors via mean aggregation.

    - g.x           : [N, H]
    - g.node_level  : [N] with 0/1/2/3
    - g.edge_index  : [2, E]

    Returns scalar tensor.
    """
    x = g.x
    node_level = g.node_level
    edge_index = g.edge_index

    device = x.device
    N, H = x.shape

    if node_level is None or edge_index is None:
        return torch.tensor(0.0, device=device)

    src, dst = edge_index  # [E], [E]

    # Edges low -> high
    mask_lh = (node_level[src] == low_level) & (node_level[dst] == high_level)
    # Edges high -> low (reverse direction)
    mask_hl = (node_level[src] == high_level) & (node_level[dst] == low_level)

    child_idx = torch.cat([src[mask_lh], dst[mask_hl]], dim=0)   # "low" nodes
    parent_idx = torch.cat([dst[mask_lh], src[mask_hl]], dim=0)  # "high" nodes

    if child_idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Aggregate parent features per child (global indexing, vectorized)
    agg = torch.zeros(N, H, device=device, dtype=x.dtype)
    cnt = torch.zeros(N, 1, device=device, dtype=x.dtype)

    agg.index_add_(0, child_idx, x[parent_idx])
    ones = torch.ones(child_idx.size(0), 1, device=device, dtype=x.dtype)
    cnt.index_add_(0, child_idx, ones)

    low_mask = (node_level == low_level)
    if not low_mask.any():
        return torch.tensor(0.0, device=device)

    pred = agg[low_mask] / cnt[low_mask].clamp_min(1.0)

    target = x[low_mask]
    if detach_target:
        target = target.detach()

    # Same mode math as _compute_hier_aux_pair_loss (the causal/jepa paths' helper):
    # raw MSE on unnormalized features is unbounded — it tracks feature-norm growth
    # quadratically and can come to dominate the objective.
    if loss_mode in ("mse_norm", "nmse"):
        sq_err = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        denom = target.pow(2).mean(dim=-1).clamp_min(1e-8)
        return (sq_err / denom).mean()
    if loss_mode == "cosine":
        cos = F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        return ((1.0 - cos) * 0.5).mean()
    return F.mse_loss(pred, target)

def compute_hierarchy_aux_loss(
    g,
    detach_target: bool = True,
    link_low0: bool = False,      # un-detach ONLY the L0 target (link aux grad -> Hyena pool)
    w_l2_from_l3: float = 1.0,
    w_l1_from_l2: float = 1.0,
    w_l0_from_l1: float = 1.0,
    w_l0_from_l3: float = 0.25,   # smaller weight by default
    loss_mode: str = "mse",       # "mse" | "mse_norm" | "cosine" (hier_aux_loss_mode)
) -> torch.Tensor:
    """
    Combined hierarchy aux loss:

      - L2 predicted from L3
      - L1 predicted from L2
      - L0 predicted from L1
      - (optionally) L0 predicted directly from L3

    All terms are fast and vectorized. If a particular level-pair has
    no connecting edges, its term is just 0.0.
    """
    device = g.x.device

    loss_total = torch.tensor(0.0, device=device)

    # L2 <- L3
    if w_l2_from_l3 != 0.0:
        l = _compute_pair_aux_loss(g, low_level=2, high_level=3, detach_target=detach_target, loss_mode=loss_mode)
        loss_total = loss_total + w_l2_from_l3 * l

    # L1 <- L2
    if w_l1_from_l2 != 0.0:
        l = _compute_pair_aux_loss(g, low_level=1, high_level=2, detach_target=detach_target, loss_mode=loss_mode)
        loss_total = loss_total + w_l1_from_l2 * l

    # L0 <- L1  (short-range reconstruction)
    _detach_l0 = detach_target and not link_low0
    if w_l0_from_l1 != 0.0:
        l = _compute_pair_aux_loss(g, low_level=0, high_level=1, detach_target=_detach_l0, loss_mode=loss_mode)
        loss_total = loss_total + w_l0_from_l1 * l

    # L0 <- L3  (long-range “closing the loop”; only has effect if 0–3 edges exist)
    if w_l0_from_l3 != 0.0:
        l = _compute_pair_aux_loss(g, low_level=0, high_level=3, detach_target=_detach_l0, loss_mode=loss_mode)
        loss_total = loss_total + w_l0_from_l3 * l

    return loss_total

def compute_hierarchy_aux_loss_onlyl2l3(
    g,
    detach_target: bool = True,
) -> torch.Tensor:
    """
    Combined hierarchy aux loss:
      - L2 predicted from L3
      - L1 predicted from L2

    You can change weights or which pairs are used.
    """
    device = g.x.device

    loss_l2 = _compute_pair_aux_loss(g, low_level=2, high_level=3, detach_target=detach_target)
    loss_l1 = _compute_pair_aux_loss(g, low_level=1, high_level=2, detach_target=detach_target)

    return (loss_l1 + loss_l2) if (loss_l1 is not None and loss_l2 is not None) else \
           (loss_l1 if loss_l2 is None else loss_l2) if (loss_l1 is not None or loss_l2 is not None) else \
           torch.tensor(0.0, device=device)


def compute_hierarchy_aux_loss_old(
    g: Data,
    recon_head: HierarchyReconHead,
    detach_target: bool = True,
) -> torch.Tensor:
    """
    Compute aux loss encouraging L2/L3 to reconstruct L0 features.

    Expects:
      - g.x           : [N, H]
      - g.node_level  : [N] with 0/1/2/3
      - g.edge_index  : [2, E]

    Returns:
      scalar tensor (loss), or 0.0 if not applicable.
    """
    x = g.x
    node_level = g.node_level
    edge_index = g.edge_index
    device = x.device

    if x is None or node_level is None or edge_index is None:
        return torch.tensor(0.0, device=device)

    # ---- Slice per level ----
    lvl0_mask = (node_level == 0)
    lvl2_mask = (node_level == 2)
    lvl3_mask = (node_level == 3)

    idx0 = torch.nonzero(lvl0_mask, as_tuple=False).view(-1)
    if idx0.numel() == 0:
        return torch.tensor(0.0, device=device)

    h0 = x[idx0]  # [N0, H]

    h2 = x[torch.nonzero(lvl2_mask, as_tuple=False).view(-1)] if lvl2_mask.any() else None
    h3 = x[torch.nonzero(lvl3_mask, as_tuple=False).view(-1)] if lvl3_mask.any() else None

    # ---- Build mappings L0 -> L2 / L3 via edges ----
    l0_to_l2 = _build_l0_to_lx_from_edges(node_level, edge_index, level_x=2) if h2 is not None else None
    l0_to_l3 = _build_l0_to_lx_from_edges(node_level, edge_index, level_x=3) if h3 is not None else None

    # ---- Get reconstruction ----
    h_hat0 = recon_head(
        h0,
        l0_to_l2=l0_to_l2, h2=h2,
        l0_to_l3=l0_to_l3, h3=h3,
    )
    if h_hat0 is None:
        return torch.tensor(0.0, device=device)

    target = h0.detach() if detach_target else h0
    aux = F.mse_loss(h_hat0, target)
    return aux


class HierarchicalFlowGAT(nn.Module):
    """
    Modified Hierarchical Flow GAT with shared transformer layers for initialization and refinement.
    
    This version uses the same transformer layers for both the initial level-specific processing
    and the subsequent refinement cycles, reducing model complexity while maintaining flexibility.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: list = [2, 2, 2, 1],
        dropout: float = 0.1,
        compression_ratios: list = [64, 16, 8],
        overlap_ratios: list = [0.5, 0.5, 0.5],
        max_seq_len: int = 131072,
        input_mode : str = "tokens",
        tie_weights: bool = True,  # Whether to tie weights between token embedding and output projection
        gen_frontier_consistent: bool = True,  # AR gen: pad prefix so coarse windows match training
        use_final_layer_for_prediction: bool = True,
        norm_type: str = "layer_norm",  # "layer_norm" or "batch_norm"
        norm_eps: float = 1e-6,
        add_self_loops: bool = True,
        add_long_range_edges: bool = True,
        long_range_distance: int = False,
        iterative_refinement_cycles: int = 3, # <<< Cycles for Stage 1 (0 disables)
        unified_refinement_cycles: int = 0,   # <<< Cycles for Stage 2 (0 disables)
        refinement_cycles: int = 2,
        use_edge_attr: bool = True,  # Option to use edge attributes
        learn_edge_from_attn: bool = True,  # Learn edge attributes from attention
        share_transformers: bool = True,  # Option to share transformer layers
        num_refinement_layers: int = 2, # For unified style when share=False
        per_level_local_qkv: bool = False,  # per-level intra-level Q/K/V in refinement layers (backbone QKV stays shared)
        per_level_ffn_dims: Optional[List[int]] = None,  # per-level FFN processing dim (U-Net-style: L0 small, coarse big)
        per_level_attn_mult: Optional[List[float]] = None,  # per-level local-attn dim mult (scales num_heads; head_dim fixed)
        local_attn_head_dim: int = 0,  # 0 = hidden//num_heads; >0 = up-project local attn to this head_dim (narrow residual)
        upper_init: str = "mask",  # "mask" | "zeros" | "pooled": how coarse (L1+) nodes are seeded at input
        upper_seed_gate_init: float = 1.0,  # pooled seed blend gate init (0 = exact mask init, warm-start safe)
        witness_reuse: str = "layer",  # "layer" = rebuild witness table per HQD layer; "step" = build once per forward
        lap_pe_k: int = 10, # Number of Laplacian eigenvectors for positional encoding
        refinement_style: str = "unified", # Default to new style, "unified" , "iterative_level"
        use_gradient_checkpointing: bool = False,
        local_connectivity_window_size: int = 0,#0#4  # Size of local connectivity window for dense connections
        l0_windowgraph: int = 0, # 1 , add dense edges in l0 graph it's k either side, making a window k*2+1 total connectivity
        rope_mode: str = "auto",  # "auto" | "1d" | "2d_axial"
        lambda_ce_anchor: float = 0,#0.0,
        use_aux_loss: bool = False, # Whether to compute the hierarchy reconstruction auxiliary loss
        lambda_hier_aux: float = 0.1,
        hier_aux_mode: str = "jepa_mlp",  # "mean_mse" | "jepa_mlp"
        hier_aux_predictor_type: str = "mlp",  # "linear" | "mlp"
        hier_aux_loss_mode: str = "mse",  # "mse" | "mse_norm" | "cosine"
        hier_aux_detach_target: bool = True,
        hier_aux_link_l0_target: bool = False,  # un-detach ONLY the L0 (Hyena-pooled) aux target
        hier_aux_unit_norm: bool = False,
        hier_aux_w_l2_from_l3: float = 1.0,
        hier_aux_w_l1_from_l2: float = 1.0,
        hier_aux_w_l0_from_l1: float = 1.0,
        hier_aux_w_l0_from_l3: float = 0.25,
        hier_aux_ar_strict: bool = False,
        hier_aux_ar_disable_l0_from_l3: bool = False,
        hier_ar_enable: bool = False,  # Enable hierarchy-level autoregressive edge filtering
        hier_ar_allow_same_time: bool = True,  # If true, allow src_time == dst_time
        hier_ar_filter_zip: bool = False,  # Apply AR filter to dynamic zipper edges
        l0_ar_enable: bool = False,  # Make L0 intra-level edges causal (time-forward only)
        enable_l0_parent_edges: bool = False,
        l0_parent_edges_bidirectional: bool = True,
        l0_parent_edge_min_level: int = 1,  # Minimum ancestor level to connect to L0 (e.g., 2 means L0 connects to L2 and L3, but not L1)
        l0_parent_edge_max_level: Optional[int] = None,  # Maximum ancestor level to connect to L0 (None = top level)
        ensure_l0_past_l1_edges: bool = False,  # Ensure that L0 nodes have edges from past L1 nodes (if l0_ar_enable)   
        ensure_past_hier_edges_all_levels: bool = False,  # Extend past bridge edges to L1toL0, L2→L1 and L3→L2 (and L2→L0, L3→L0 if enable_l0_parent_edges)
        ensure_l0_past_parent_edges: bool = False,  # STAGGERED uncut L0->past-parent context: L0->past-L1, then past-L2-of-that-L1, then
                                                    #   past-L3-of-that-L2 (each step uses the prior parent's time, so every edge is
                                                    #   strictly past -> survives the AR filter, no leak). Legacy off = bit-identical.
        l0_past_parent_min_level: int = 1,          # lowest parent level to emit a staggered L0 edge for (1 = include the past-L1 edge)
        l0_past_parent_max_level: Optional[int] = None,  # highest parent level (None = top level)
        l0_past_l1_edge_type_id: Optional[int] = None,#5,
        # Coarse co-prediction: L1/L2/L3 project (causal, strictly-past) into L0's pre-head
        # features so the coarse compute gets a DIRECT next-token gradient + read-back path at
        # the head (fixes the late-ascent underuse). Per L0 position t, each level contributes
        # its most-recent CLOSED node (window_end <= t) -> leak-free. gate init 0 = warm-start
        # bit-identical (contribution is exactly zero regardless of proj init).
        hier_copredict_l0: bool = False,
        hier_copredict_levels: Optional[List[int]] = None,  # default [1,2,3]; clipped to available coarse levels
        hier_copredict_gate_init: float = 0.0,
        # Debug monitor: periodically log the essential learnable gates/scales (copredict, cross-query
        # write scales, upper/top refiner level scales) so you can watch which pathways open/collapse.
        pinball_monitor_gates: bool = False,
        pinball_monitor_gates_every: int = 100,   # log cadence in forward calls
        # Per-layer upward refresh: after every k-th refinement layer, re-pool each level's
        # child-window mean into its parent via a gated residual, bottom-up (L1 from the
        # current L0, then L2 from the UPDATED L1, ...). Abstraction then ascends every layer
        # instead of accumulating until the last layers (the late-ascent fix). Causal for the
        # same reason as the pooled seed: a parent's ar_time is its child window's END, so the
        # pooled window reveals nothing a reader could not already reach through causal upward
        # edges. gate init 0 = exact no-op (warm-start bit-identical).
        hier_upward_refresh: bool = False,
        hier_upward_refresh_every: int = 1,       # apply after every k-th layer call
        hier_upward_refresh_gate_init: float = 0.0,
        # Per-layer downward refresh — the linear-cost DENSE replacement for the bridges/
        # staggered scatter edges: each fine node gathers the MOST-RECENT CLOSED node of each
        # strictly-coarser level (the same node a bridge edge points at), projected per pair
        # and added with a learnable gate. Project-then-gather: the Linear runs on the few
        # coarse rows, the gather is bandwidth-only -> O(N) total, no attention, no scatter.
        # Applied top-down chained (L2 reads L3, then L1 reads the UPDATED L2, ...), so L3
        # reaches L0 within one layer. Causal: gathered window_end <= reader time (allow_same
        # follows hier_ar_allow_same_time). gate 0 = exact no-op. Content-ADDRESSED far reads
        # stay with the cross-query refiners; dynamic edges (HQD/zipper) are unaffected.
        hier_downward_refresh: bool = False,
        hier_downward_refresh_pairs: Optional[List[str]] = None,  # "fine:coarse"; None = all downward pairs
        hier_downward_refresh_every: int = 1,
        hier_downward_refresh_gate_init: float = 0.0,
        # Drop the STATIC cross-level edges from the refinement edge set (the scatter vertical
        # backbone), leaving vertical flow to the refreshes + cross-query. Graph construction,
        # node metadata and the dynamic-edge paths (HQD/zipper inject their own ephemeral
        # edges at attention time) are untouched — sparse/dynamic approaches stay possible.
        drop_static_cross_level_edges: bool = False,
        # torch.compile the up/down refresh functions (they are kernel-LAUNCH-bound: many tiny
        # elementwise ops x 12 layers; inductor fuses them, measured ~14% whole-step win).
        # Applied in TRAINING mode only — the training seq len is fixed (one compile), while
        # generation varies lengths every step and would trigger recompile storms; eval/gen
        # stay eager (numerics differ ~1e-6 from fusion, training-side only).
        hier_refresh_compile: bool = False,
        # Compile each refinement LAYER's forward (the whole per-layer step: qkv, pack spec
        # consumption, local flash, FFN, gates) — the widest compileable region below the
        # model-level forward (whose cache bookkeeping/timing side effects break dynamo).
        # Targets the launch-bound regime (short sequences, e.g. image grids: profiling at
        # T=64 showed ~15k tiny kernels/step vs ~10ms of GEMM). Training mode only, lazy,
        # probation fallback to eager on first runtime failure (same policy as
        # hier_refresh_compile; both knobs compose — the refreshes live outside the layers).
        hier_layer_compile: bool = False,
        # HQD v2 — cross-query-guided nomination (NSA-style selective sparse attention). At the
        # multirate midpoint, each L0 query scores the CLOSED L3 windows using the 0:3
        # cross-query refiner's TRAINED q/k (per-query, causal — never global_mean), descends
        # into the top-k winners' L1 children (cosine scores), and nominates the winning L1
        # nodes' L0 children as ephemeral L0->L0 edges for all layers after the midpoint —
        # the same injection path v1 HQD uses, so it works with the static cross-level edges
        # dropped. Selection is no-grad (gradient flows through the attention over the added
        # edges, as in v1); children tables come from window arithmetic, not edge lists.
        xq_nominate_enable: bool = False,
        xq_nominate_topk_l3: int = 2,      # closed L3 windows shortlisted per query
        xq_nominate_topk_l1: int = 2,      # L1 winners kept among the shortlist's children
        xq_nominate_exclude_local: bool = True,  # skip sources already inside the L0 local window
        xq_nominate_max_layers: int = 0,   # layers fed after nomination (0 = all remaining);
                                           # the sparse-edge path costs per layer — this is the budget dial
        # When to nominate: "auto" = after the middle layer (queries/summaries are mature by
        # then and the remaining layers integrate what was fetched); an int = after that many
        # layers (1-based), for earlier/later retrieval experiments. Independent of the
        # multirate schedule — nomination runs even with multirate disabled.
        xq_nominate_after_layer: Union[str, int] = "auto",
        # Re-nominate every k layers after the first (0 = once per forward). Each round
        # RE-SELECTS with the current (evolved) queries — retrieved content changes what the
        # next round looks for (iterative / multi-hop retrieval), instead of freezing one edge
        # set for the rest of a deep stack. Scoring is cheap (~1-2ms); the per-interval fed
        # layers are still capped by xq_nominate_max_layers (budget resets each round).
        xq_nominate_every: int = 0,
        # Stage-3 scored prune (0 = off = legacy "all children"). Without it every L1 winner
        # dumps ALL its L0 children (~48 sources/query, mostly padding around the one useful
        # token — measured on the 100ep run). >0: score the candidates with the SAME shared
        # RoPE'd q/k the consuming read uses (NSA's select-what-you'd-attend-to) and keep the
        # top-k. Precision up, read cost down ~3x at 16.
        xq_nominate_topk_l0: int = 0,
        # Consume nominations as a packed fixed-K per-query candidate table (dense GEMM-shaped
        # attention, _compute_hqd_packed_l0_attn) instead of a flat per-edge scatter list. The
        # scatter path materializes per-edge q/k/v gathers in autograd (~GBs at seq 8k, the
        # measured 165->110 tok/s cost); packed is uniform [B,Q,K] — same math, fraction of the
        # memory. Requires topk_l0 > 0 (fixed K).
        xq_nominate_packed_read: bool = False,
        # Per-query differentiable relevance gate: hqd read output scaled by
        # sigmoid(w * stage1_top_score + b). Gives queries a no-op option (the ungated read
        # HURT never-recurring tokens by ~1.6% on the 100ep run) AND routes a task gradient
        # into the 0:3 selector q/k through the score (NSA-style selector training — the
        # selection top-k itself stays no-grad). w init 0 / b init 2.2 => starts ~0.90 open.
        xq_nominate_gate_enable: bool = False,
        # Stage-2 (L1 descent) scoring: "cosine" = raw feature cosine (legacy placeholder) |
        # "local_qk" = the layer's trained per-level local q/k (L0-query proj x L1-key proj;
        # falls back to the shared q/k, then cosine). No new params.
        xq_nominate_stage2: str = "cosine",
        # Packed cross-level LOCAL attention: interleave ALL levels' nodes by close time
        # (ar_time) and run ONE causal sliding-window attention over the mixed sequence, in
        # place of the per-level local call for `local_pack_query_levels`. Fine queries then
        # see the recent CLOSED coarse summaries (staggered ancestors) and coarse queries see
        # their strictly-past children INSIDE the same softmax as their lateral neighbors —
        # query-adaptive vertical mixing (softmax competition) instead of scalar-gated
        # residual refresh writes. Leak-free by construction: causal over close-time order.
        # RoPE for the packed call is applied at token-scale close positions (the per-level
        # local path RoPEs coarse nodes level-locally, the wrong scale for mixed windows).
        # Levels NOT in query_levels keep their per-level lateral windows untouched, and the
        # packed rows are ephemeral to the local-attention call (refiners/HQD/xq unaffected).
        local_pack_cross_level: bool = False,
        # Packed window in mixed SLOTS (0 = reuse L0's local window). With strides 8/16/32
        # a 128-slot window spans ~105 tokens plus ~23 coarse rows (~22% dilution).
        local_pack_window: int = 0,
        # Levels whose local attention is taken from the packed call (None = [0]). Add 1-3
        # to also route coarse laterals through the mixed window (shrinks their lateral
        # token-reach to the packed span — raise local_pack_window accordingly).
        local_pack_query_levels: Optional[List[int]] = None,
        # Per-level per-head K/V tags for the packed call: flash can't take the scatter
        # path's level-pair logit bias, so a zero-init embedding added to the RoPE'd K makes
        # q·e_level a query-dependent level bias inside the same softmax (V tag marks the
        # output's source level). Zero init = bit-identical to untagged at start.
        local_pack_level_bias: bool = False,
        # COARSE LANE: a second packed causal window over ONLY the coarse nodes (L1-L3,
        # close-time order). The main mixed window denominates reach in mixed slots, which
        # are ~78% tokens — coarse laterals shrink to ~16 same-level neighbors. The lane's
        # slots are all coarse (7 per 32 tokens), so a cheap window restores the wide coarse
        # lateral reach the hierarchy's economics lean on (896 slots = 4096 tokens = L3's
        # full pre-pack span, for ~the cost of the main call again — coarse rows are few).
        # Applies to coarse levels IN local_pack_query_levels (additive to their main-window
        # read); keys are always all coarse rows. Same causality argument as the main pack.
        local_pack_coarse_lane: bool = False,
        # Lane window in COARSE slots (L1+L2+L3 rows). 896 ~= 4096 tokens of span at
        # strides 8/16/32 (128 L3 + 256 L2 + 512 L1 per span).
        local_pack_coarse_window: int = 512,
        # Combine the mixed window + coarse lane for coarse queries as ONE unified softmax
        # over the union of both key sets (log-sum-exp merge of the two flash calls),
        # instead of adding two independently-normalized outputs. Parameter-free and
        # context-selective per key: each branch's weight is its share of total attention
        # mass, so a single relevant lane key can outvote a full mixed window. Keys in both
        # windows (the ~most recent coarse rows) are counted twice (+ln2 logit bias).
        # False = additive combine (previous behavior, bit-identical).
        local_pack_lane_merge: bool = False,
        local_pack_coarse_global: bool = False,
        local_pack_l0_coarse_bands: bool = False,
        local_pack_l0_window: int = 0,
        local_pack_l0_coarse_windows: Optional[List[int]] = None,
        local_pack_l0_bands_checkpoint: bool = False,
        local_pack_l0_coarse_rank_window: int = 0,
        local_pack_coarse_global_gate_init: float = 0.0,
        # ONE flex_attention call with a block-sparse union mask replaces the mixed call +
        # lane + merge (exact-dedup unified softmax, no LSE recompute). Needs torch>=2.5,
        # all levels in query_levels, and the coarse lane on. Falls back to merge/additive
        # on any failure. NOTE: flex has no attention-prob dropout (residual/FFN dropout
        # unaffected).
        local_pack_flex_union: bool = False,
        # DropNode on the hierarchy: zero a coarse row's VALUE in the packed K/V set per
        # (batch, row, step). L0 never dropped; residual stream and refresh paths untouched.
        hier_node_dropout: float = 0.0,
        hier_node_dropout_per_level: Optional[Sequence[float]] = None,
        # Bidi pack (MaskGIT/diffusion): when the runtime causal flags are off, run the packed
        # windows two-sided instead of skipping pack. flex_union/lane_merge -> additive in bidi.
        local_pack_bidirectional: bool = False,
        # xq/HQD fetch-read fixes (see mp layer): score+read fetched far tokens with the
        # PRE-RoPE q/k (the RoPE'd L0<->L0 geometry is untrained past the local window ->
        # far fetches read as noise, the selector learns to stay near), and a learned
        # zero-value sink slot so junk fetches can no-op inside the read softmax.
        xq_nominate_read_prerope: bool = False,
        xq_nominate_read_sink: bool = False,
        # Predictive coarse aux ("next-concept" prediction): each coarse node predicts the
        # DETACHED pooled child summary of a strictly-future window at its own level — the LM
        # objective one timescale up. Unlike the reconstruction aux (which rewards summarizing
        # the PAST, i.e. copying scores), a summary only scores here by being PREDICTIVE.
        # Training-only loss; the forward pass is untouched (no leak surface).
        hier_predaux_enable: bool = False,
        lambda_hier_predaux: float = 0.05,
        hier_predaux_levels: Optional[List[int]] = None,  # default [1,2,3]; clipped to coarse levels
        hier_predaux_horizon: int = 0,            # windows ahead; 0 = auto (first NON-overlapping window)
        hier_predaux_loss_mode: str = "cosine",   # cosine | mse_norm | mse
        hier_predaux_predictor: str = "mlp",      # mlp | linear
        # "next" is the AR form (forecast only). In MASKED/bidirectional training the future is
        # partially visible, so "next" degrades toward copying — use "both" there: a second
        # per-level head also predicts the PREVIOUS non-overlapping window, turning the loss
        # into masked-window summary prediction (the meaningful bidirectional analogue).
        hier_predaux_directions: str = "next",    # next | both
        l0_alpha_enable: bool = False, # Whether to apply a learnable alpha to L0 features in the recon head
        l0_local_backend: str = "flash",  # pyg | flash | xformers | sdpa
        l0_local_window: int = 128,
        ablate_levels: Optional[List[int]] = None,  # hierarchy levels to disable (cut all their edges) for ablation studies; L0 cannot be ablated
        report_level_connectivity: bool = False,  # log a per-level connectivity / inert-level report on the first forward
        local_attn_levels: Optional[List[int]] = [0, 1, 2, 3],#None,  # levels for local window attn (default: [0] = L0 only)
        local_attn_windows: Optional[List[int]] = [128, 16, 64, 128],#None,  # per-level windows (if None, use l0_local_window for all)
        local_attn_causal_levels: Optional[List[int]] = None,  # levels where local attn is causal (default: [0])
        local_attn_level_role_bias_enable: bool = False,
        local_attn_level_role_bias_scale: float = 1.0,
        local_attn_flash_dtype_cast: bool = False,
        cross_level_packed: bool = False,
        cross_level_qkv: str = "shared",
        local_attn_sampled_mode: str = "safe_sdpa",
        sparse_attn_mode: str = "off",
        sparse_attn_chunk_size: int = 0,
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
        rope_level_axis_enable: bool = False,
        rope_level_axis_scale: float = 32.0,
        TRM: bool = False, # Whether to use the last cycle gradient
        use_neighbor_sampling: bool = True,
        num_neighbors: list = [8],#[4096],  # e.g., [32] for 32 neighbors per layer
        sampling_batch_size: int = 131072*64,#16384*4,#8192,#4096,#131072*4,
        sampling_seed_budget: int = 512,#16384*64,#512//8, # seqlength divided by 8
        neighbor_sampling_backend: str = "auto",  # "auto" (default) or "pyg"
        zip_enable: bool = False,
        zip_attn_agg: str = "ema", # "max" or "ema"
        zip_attn_ema_beta: float = 0.9,
        zip_depth: str = "l0",#"l0","l1",
        zip_granularity: str = "per_layer", # "per_cycle" or "per_layer"
        zip_persist: str = "decay", #"decay" (default), "cycle_only", or "layer_only".
        zip_edge_decay: float = 0.0,#0.9,
        zip_edge_drop_threshold: float = 1e-4,#0.05,
        zip_max_l3_pairs: int = 4,
        zip_max_l2_pairs: int = 4,
        zip_max_children_per_parent: int = 4,
        zip_max_candidate_edges: int = 16384,#1024*50,#4096,#1024,
        zip_max_dyn_edges_total: int = 16384,#16384,#4096,
        zip_max_dyn_edges_per_sample: int = 4096,
        zip_log_enable: bool = False,
        zip_edge_select_mode: str = "percentile", # "percentile" (default), "absolute", "topk", or "relative_max".
        zip_edge_percentile: float = 0.9,
        zip_edge_relative_ratio: float = 0.8,
        zip_l3_all_pairs_every: int = 1,
        zip_l3_pairs_cap: int = 512,
        zip_l3_all_pairs_sample: bool = True,
        zip_l3_pair_sampling_mode: str = "attn_entropy",
        zip_score_mode: str = "fast_qk", # "mp_full" (legacy) or "fast_qk" (faster)
        zip_execution_mode: str = "ephemeral_msg",  # "edge_mutation" or "ephemeral_msg"
        zip_select_scope: str = "global",  # "global" or "per_dst"
        zip_per_dst_topk: int = 1,
        zip_l3_pair_attn_weight: float = 1.0,#0.7,
        zip_l3_pair_entropy_weight: float = 0.0,#0.3,
        zip_gate_mode: str = "paramfree", # "ste" or "soft" or "paramfree"
        zip_gate_tau: float = 0.5,
        zip_gate_center: float = 0.0,
        zip_gate_scope: str = "all",
        zip_paramfree_gate: bool = True,
        zip_paramfree_gate_tau: float = 0.10,
        zip_paramfree_gate_center: float = 0.20,
        zip_mass_budget_per_dst: float = 0.20,
        zip_norm_clip_ratio: float = 0.20,
        zip_warmup_steps: int = 0,
        zip_select_from_softmax: bool = True,
        zip_use_beta_gate: bool = True,
        zip_msg_eta: float = 1.0,
        zip_beta_init_bias: float = 2.0,
        refinement_batch_mode: str = "true_batch_nozip",  # "blockdiag" or "true_batch_nozip"
        true_batch_strict: bool = True,
        true_batch_aux_mode: str = "exact_blockdiag",  # "per_sample_mean" or "exact_blockdiag"
        autoenc_graph_mode: str = "off",  # "off" or "twin_shared_l3"
        autoenc_coupled_feedback: bool = True,
        token_unet_enable: bool = False,
        token_unet_mode: str = "stem",  # "stem" | "coarse_tokenize"
        token_unet_dim: str = "auto",  # "auto" | "1d" | "2d"
        token_unet_2d_causal: bool = False,
        token_unet_scale: int = 1,
        token_unet_kernel_size: int = 5,
        token_unet_dropout: float = 0.0,
        token_unet_right_edge_targets: bool = True,
        token_unet_lookahead_decode_enable: bool = False,
        token_unet_lookahead_kernel_size: int = 5,
        token_unet_lookahead_blocks: int = 2,
        token_unet_block: str = "cnn",  # "cnn" (local causal conv) | "hyena" (global FFT long conv)
        token_unet_dynamic: bool = False,   # true = channel-growing concat-skip causal U-Net
        token_unet_channel_growth: int = 256,  # +channels per downsample level (dynamic U-Net)
        token_unet_max_channels: int = 0,   # cap on U-Net channels (0 => 2*hidden_dim)
        token_unet_concat_skips: bool = True,  # concat (true) vs additive (false) skips
        token_unet_hyena_length_invariant: bool = False,  # absolute-position Hyena (gen-stable)
        graph_geometry_mode: str = "sequence",  # "sequence" | "grid2d"
        graph_grid_height: int = 0,
        graph_grid_width: int = 0,
        # Curve mode: reorder L0 tokens ONCE by a space-filling curve after the stem (inverse
        # at output). Sequence-mode 1D windows then pool contiguous curve ranges = compact
        # spatial blocks (exact quadtree at comp 4 / overlap 0 / pow2 grids). Geometry stays
        # "sequence" so pack/skeleton/refresh all run. See layers/spatial_curve.py.
        spatial_curve: str = "none",            # "none" | "hilbert" | "morton" | "raster"
        spatial_dims: Optional[List[int]] = None,  # e.g. [32, 32]; fallback graph_grid_h/w, then sqrt
        # Axial ND RoPE for the packed cross-level local path: spec carries true ND coords
        # (L0 = curve coords, coarse = child-window centroid) so pack attention scores use
        # exact spatial offsets instead of 1D curve-index rotations. Needs spatial_curve on.
        local_pack_rope_axial: bool = False,
        # Bidi downward refresh: gather each fine node's CONTAINING parent (nearest window
        # center) instead of "most recent closed" — the spatially-correct variant for
        # MaskGIT/diffusion. LEAKS under AR (own parent's window contains the token).
        hier_downward_bidi_parent: bool = False,
        graph_spatial_metric: str = "chebyshev",  # "chebyshev" | "manhattan"
        graph_downsample_factor: int = 2,
        class_cond_enable: bool = False,
        num_classes: int = 0,
        class_cond_drop_prob: float = 0.1,
        diffusion_timestep_embed_dim: int = 256,
        refine_cond_mode: str = "none",  # "none" | "film" | "film_concat"
        refine_cond_strength: float = 1.0,
        refine_cond_concat_gate_init: float = -2.0,
        pinball_level_dims: Optional[List[int]] = None,
        pinball_work_dim: Optional[int] = None,
        pinball_work_num_heads: Optional[int] = None,
        pinball_adapter_type: str = "low_rank",
        pinball_adapter_rank: int = 128,
        pinball_level_cycle_enable: bool = False,
        pinball_level_cycles: Optional[List[int]] = None,
        pinball_cycle_schedule: str = "staggered_flush",
        pinball_message_fn: str = "original",
        pinball_graph_active_compute: str = "all",
        pinball_level_cycle_mode: str = "extra_cycles",
        pinball_multirate_enable: bool = False,
        pinball_multirate_schedule: str = "after_full_stack",
        pinball_multirate_midpoint_layer: Union[str, int] = "auto",
        pinball_multirate_midpoint_repeats: int = 1,
        pinball_multirate_skip_after_last_cycle: bool = True,
        pinball_multirate_debug: bool = False,
        pinball_multirate_adaptive: bool = False,
        pinball_multirate_energy_tau: float = 0.05,
        pinball_multirate_max_cycles: int = 4,
        pinball_multirate_active_rows: bool = True,
        pinball_upper_refine_steps: int = 0,
        pinball_top_refine_steps: int = 0,
        pinball_l3_workspace_tokens: int = 0,
        pinball_upper_refine_shared_weights: bool = True,
        pinball_top_refine_shared_weights: bool = True,
        pinball_consolidate_after_upper: bool = False,
        pinball_upper_refine_every: int = 1,
        pinball_top_refine_every: int = 1,
        pinball_multirate_attn_backend: str = "auto",
        pinball_upper_refine_window: int = 0,
        pinball_top_refine_window: int = 0,
        pinball_upper_refine_causal: bool = True,
        pinball_top_refine_causal: bool = True,
        pinball_upper_cross_attn_steps: int = 0,
        pinball_upper_query_topk_l2: int = 0,
        pinball_upper_cross_attn_backend: str = "auto",
        pinball_upper_cross_attn_causal: bool = True,
        pinball_upper_cross_attn_shared_weights: bool = True,
        pinball_upper_update_l2_enable: bool = False,
        pinball_upper_update_l2_scale_init: float = 1.0e-2,
        pinball_upper_cross_write_scale_init: float = 1.0e-2,
        pinball_cross_query_pairs: Optional[List[str]] = None,
        pinball_cross_query_steps: int = 0,
        pinball_cross_query_topk: int = 0,
        pinball_cross_query_l0_window: int = 0,
        # Recency cap (ar-time units) on every cross-query pair's attention read: a query
        # only reads memory closing within the last W time units. Per-query, inside the
        # causal mask (leak-free); bit-identical to off while W >= context length. Bounds
        # the per-query read set so the episode stays linear-scaling at long context.
        pinball_cross_query_memory_window: int = 0,
        pinball_cross_query_backend: str = "auto",
        pinball_cross_query_causal: bool = True,
        pinball_cross_query_shared_weights: bool = True,
        # Per-QUERY-LEVEL FFN "processing dim" for the episode refiners (index = the
        # refiner's query_level; 0 or missing = default work_dim width). Same semantics as
        # per_level_ffn_dims — e.g. [0, 1152, 1152, 1536] fattens the 1:3 refiner's FFN
        # while the 0:3 refiner (whose FFN runs on the many L0 rows) stays cheap. Cycling
        # (multirate repeats / adaptive settling) re-runs these weights tied, so extra
        # cycles scale coarse compute without scaling params.
        pinball_cross_query_ffn_dims: Optional[List[int]] = None,
        pinball_cross_query_update_memory_enable: bool = False,
        pinball_cross_query_selection: str = "global_mean",
        pinball_cross_query_write_scale_init: float = 1.0e-2,
        rgb_token_unet_enable: bool = False,
        rgb_token_unet_downsample: int = 16,
        rgb_token_unet_base_channels: int = 64,
        rgb_token_unet_kernel_size: int = 5,
        rgb_token_unet_decode_kernel_size: int = 3,
        rgb_token_unet_decode_separable: bool = True,
        rgb_token_unet_max_channels: int = 512,
        hierarchical_query_descent_enable: bool = False,
        hqd_topk_l3: int = 4,
        hqd_topk_l2: int = 4,
        hqd_topk_l1: int = 4,
        hqd_topk_l0: int = 64,
        hqd_l0_topk_enable: bool = True,
        hqd_include_local_window: bool = False,
        hqd_local_window_size: int = 0,
        hqd_causal: Union[str, bool, None] = "auto",
        hqd_use_existing_zipper_projections: bool = True,
        hqd_debug: bool = False,
        hqd_granularity: str = "per_layer",
        hqd_every_n: int = -1,
        hqd_every_n_offset: int = 0,
        hqd_reuse_previous: bool = False,
        hqd_reuse_max_age: int = 0,
        hqd_query_chunk_size: int = 524288,
        hqd_query_level: int = 0,
        hqd_stop_level: int = 0,
        hqd_keep_stage_survivors: bool = False,
        hqd_handoff_to_l0: bool = False,
        hqd_global_topk: int = 0,
        hqd_assume_disjoint_children: bool = False,
        hqd_validate_disjoint_children: bool = False,
        hqd_sparse_project_active_only: bool = False,
        hqd_tiled_apply: bool = False,
        hqd_static_descent: bool = False,
        hqd_static_compile: bool = False,
        hqd_tile_size: int = 64,
        hqd_tile_topk: int = 64,
        hqd_attn_impl: str = "scatter",
        hqd_dense_backend: str = "sdpa",
        hqd_descent_stop_at: int = 0,
        hqd_shallow_read_level: int = 0,
        hqd_coarse_route_levels: Optional[List[int]] = None,
        hqd_coarse_route_topk: int = 8,
        hqd_select_inside_message_passing: bool = False,
        hqd_graph_witness_enable: bool = False,
        hqd_graph_witness_topk: int = 4,
        hqd_packed_witness_read: bool = False,
        hqd_packed_witness_source: str = "graph",
        hqd_packed_witness_chunk_size: int = 2048,
        # --- Witness packets: selected coarse nodes expose bounded pointers to exact
        # lower-level children ("what it cannot forget"), force-included into the HQD
        # final-stage candidate set so rare/important children survive top-k pruning. ---
        use_witness_packets: bool = False,
        use_summary_witnesses: bool = True,
        use_rare_witnesses: bool = True,
        summary_witness_per_head: bool = False,
        witness_k_summary: int = 4,
        witness_k_rare: int = 4,
        witness_max_per_parent: int = 0,
        witness_levels: Optional[List[int]] = None,
        witness_lambda_rare: float = 0.3,
        witness_score_bias: float = 5.0,
        # Which selected levels to mix into the HQD final read. [0] = L0 only (default,
        # original behavior); [0,1,2,3] also attends to the selected L1/L2/L3 summary nodes.
        hqd_read_levels: Optional[List[int]] = None,
        # Per-level "window bag": scatter the L0 routing's per-level selections back up so
        # coarse nodes also attend to a bounded bag at their level (lateral reach). [] = off.
        # routing: "global" (all level-L nodes share the union bag) | "per_position" (each
        # node gets the bag from its own descendants). Causal-safe via src_max <= dst_min.
        hqd_window_bag_levels: Optional[List[int]] = None,
        hqd_window_bag_routing: str = "global",
        hqd_window_bag_topk: int = 0,  # per_position: 0=natural bound, >0=hard per-node cap
        verbose: bool = False,
    ):
        """
        Initialize the Hierarchical Flow GAT with shared transformer option.
        
        Args:
            vocab_size: Vocabulary size for token prediction
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads per layer
            num_layers: Number of transformer layers per hierarchy level
            dropout: Dropout probability
            compression_ratios: Compression ratios between hierarchy levels
            overlap_ratios: Overlap ratios between adjacent summaries
            max_seq_len: Maximum sequence length
            use_final_layer_for_prediction: Whether to use highest level for prediction
            add_self_loops: Whether to add self-loops to nodes
            add_long_range_edges: Whether to add edges between non-adjacent nodes
            long_range_distance: Maximum distance for long-range connections
            refinement_cycles: Number of bidirectional refinement cycles
            use_edge_attr: Whether to use edge attributes
            share_transformers: Whether to share transformers between level processing and refinement
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_hier_levels = len(num_layers)
        if len(compression_ratios) != self.num_hier_levels - 1:
             raise ValueError("Length of compression_ratios must be num_layers - 1")
        if len(overlap_ratios) != self.num_hier_levels - 1:
             raise ValueError("Length of overlap_ratios must be num_layers - 1")
        self.num_layers = num_layers
        self.dropout_rate = dropout # Store dropout rate
        # Level ablation: hierarchy levels whose edges are all cut (nodes kept but isolated),
        # so they contribute nothing to the L0 prediction. L0 (level 0) is never ablated.
        self.ablate_levels = sorted({int(l) for l in (ablate_levels or []) if int(l) != 0})
        if self.ablate_levels:
            logger.info("Level ablation active: disabling hierarchy levels %s (edges cut)", self.ablate_levels)
        self.report_level_connectivity_enabled = bool(report_level_connectivity)
        self._last_topology = None       # (edge_index, node_level) cached each forward for the reporter
        self._level_report_logged = False
        self.compression_ratios = compression_ratios
        self.input_mode = input_mode # Store input mode
        self.overlap_ratios = overlap_ratios
        self.max_seq_len = max_seq_len
        self.gen_frontier_consistent = bool(gen_frontier_consistent)
        self.use_final_layer_for_prediction = use_final_layer_for_prediction
        self.add_self_loops = add_self_loops
        self.add_long_range_edges = add_long_range_edges
        self.long_range_distance = long_range_distance
        self.iterative_refinement_cycles = iterative_refinement_cycles # Stage 1 cycles
        self.unified_refinement_cycles = unified_refinement_cycles # Stage 2 cycles
        self.refinement_cycles = refinement_cycles
        self.use_edge_attr = use_edge_attr
        self.learn_edge_from_attn = bool(learn_edge_from_attn)
        self.share_transformers = share_transformers # Store even if only used by unified
        self.num_refinement_layers = num_refinement_layers # Store even if only used by unified
        self.per_level_local_qkv = bool(per_level_local_qkv)
        self.per_level_ffn_dims = [int(d) for d in (per_level_ffn_dims or [])]
        self.per_level_attn_mult = [float(m) for m in (per_level_attn_mult or [])]
        self.local_attn_head_dim = int(local_attn_head_dim)
        # Packed cross-level local attention: stored BEFORE transformer construction (the
        # level-bias flag is threaded into every layer's mp constructor as parameters).
        self.local_pack_cross_level = bool(local_pack_cross_level)
        self.local_pack_window = max(0, int(local_pack_window or 0))
        self.local_pack_query_levels = [
            int(l) for l in (local_pack_query_levels if local_pack_query_levels is not None else [0])
        ]
        self.local_pack_level_bias = bool(local_pack_level_bias)
        self.local_pack_coarse_lane = bool(local_pack_coarse_lane)
        self.local_pack_coarse_window = max(0, int(local_pack_coarse_window or 0))
        self.local_pack_lane_merge = bool(local_pack_lane_merge)
        self.local_pack_coarse_global = bool(local_pack_coarse_global)
        self.local_pack_l0_coarse_bands = bool(local_pack_l0_coarse_bands)
        self.local_pack_l0_window = int(local_pack_l0_window or 0)
        self.local_pack_l0_bands_checkpoint = bool(local_pack_l0_bands_checkpoint)
        self.local_pack_l0_coarse_rank_window = int(local_pack_l0_coarse_rank_window or 0)
        self.local_pack_l0_coarse_windows = list(local_pack_l0_coarse_windows or [])
        self.local_pack_coarse_global_gate_init = float(local_pack_coarse_global_gate_init)
        self.local_pack_flex_union = bool(local_pack_flex_union)
        self.hier_node_dropout = float(hier_node_dropout)
        self.hier_node_dropout_per_level = (
            list(hier_node_dropout_per_level) if hier_node_dropout_per_level is not None else None)
        self.local_pack_bidirectional = bool(local_pack_bidirectional)
        self.xq_nominate_read_prerope = bool(xq_nominate_read_prerope)
        self.xq_nominate_read_sink = bool(xq_nominate_read_sink)
        if self.local_pack_cross_level and int(local_attn_head_dim or 0) > 0:
            raise ValueError(
                "local_pack_cross_level mixes levels in one attention call and requires the "
                "shared local q/k/v (set local_attn_head_dim: 0)"
            )
        if self.local_pack_coarse_lane and not any(
            int(l) > 0 for l in self.local_pack_query_levels
        ):
            logger.warning(
                "local_pack_coarse_lane is inert: no coarse level in local_pack_query_levels "
                "(coarse levels then keep their per-level lateral windows)."
            )
        self.lap_pe_k = lap_pe_k
        self.refinement_style = refinement_style
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.local_connectivity_window_size = local_connectivity_window_size
        self.l0_windowgraph = l0_windowgraph
        geom_mode = str(graph_geometry_mode).lower()
        if geom_mode not in {"sequence", "grid2d"}:
            logger.warning("Unknown graph_geometry_mode='%s'; falling back to 'sequence'", geom_mode)
            geom_mode = "sequence"
        self.graph_geometry_mode = geom_mode
        self.graph_grid_height = max(0, int(graph_grid_height))
        self.graph_grid_width = max(0, int(graph_grid_width))
        self.spatial_curve = str(spatial_curve or "none").lower()
        self.spatial_dims = tuple(int(d) for d in spatial_dims) if spatial_dims else None
        self.local_pack_rope_axial = bool(local_pack_rope_axial)
        self.hier_downward_bidi_parent = bool(hier_downward_bidi_parent)
        self._spatial_curve_dev_cache: Dict = {}   # (dims, curve, device) -> (perm, inv, coords)
        self._active_spatial_curve = None           # set per forward: (perm, inv, coords) or None
        self._spatial_curve_coords_l0 = None        # [T, ndim] coords in CURVE order (this forward)
        self._spatial_curve_warned = False
        if self.hier_downward_bidi_parent and bool(hier_ar_enable):
            logger.warning(
                "hier_downward_bidi_parent with hier_ar_enable: the containing parent's window "
                "includes the token itself — this LEAKS under AR training. Bidi/MaskGIT only."
            )
        metric = str(graph_spatial_metric).lower()
        if metric not in {"chebyshev", "manhattan"}:
            logger.warning("Unknown graph_spatial_metric='%s'; falling back to 'chebyshev'", metric)
            metric = "chebyshev"
        self.graph_spatial_metric = metric
        self.graph_downsample_factor = max(1, int(graph_downsample_factor))
        self._cached_level_grid_shapes: Dict[int, Tuple[int, int]] = {}
        self.class_cond_enable = bool(class_cond_enable)
        self.num_classes = max(0, int(num_classes))
        self.class_cond_drop_prob = max(0.0, min(1.0, float(class_cond_drop_prob)))
        self.diffusion_timestep_embed_dim = max(16, int(diffusion_timestep_embed_dim))
        self.class_null_index = int(self.num_classes)
        self.class_embedding: Optional[nn.Embedding] = None
        if self.class_cond_enable and self.num_classes > 0:
            self.class_embedding = nn.Embedding(self.num_classes + 1, self.hidden_dim)
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(self.diffusion_timestep_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_film = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        refine_mode = str(refine_cond_mode).lower()
        if refine_mode not in {"none", "film", "film_concat"}:
            logger.warning("Unknown refine_cond_mode='%s'; falling back to 'none'", refine_mode)
            refine_mode = "none"
        self.refine_cond_mode = refine_mode
        self.refine_cond_strength = max(0.0, float(refine_cond_strength))
        self.refine_cond_concat_gate_init = float(refine_cond_concat_gate_init)
        self.refine_cond_film: Optional[nn.Linear] = None
        self.refine_cond_concat_proj: Optional[nn.Linear] = None
        self.refine_cond_concat_gate: Optional[nn.Parameter] = None
        self._refine_cond_runtime_logged: bool = False
        if self.refine_cond_mode in {"film", "film_concat"}:
            self.refine_cond_film = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        if self.refine_cond_mode == "film_concat":
            self.refine_cond_concat_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.refine_cond_concat_gate = nn.Parameter(torch.tensor(float(self.refine_cond_concat_gate_init)))
        self.pinball_work_dim = int(pinball_work_dim) if pinball_work_dim is not None else int(self.hidden_dim)
        if self.pinball_work_dim <= 0:
            raise ValueError("pinball_work_dim must be positive")
        self.pinball_work_num_heads = int(pinball_work_num_heads) if pinball_work_num_heads is not None else int(self.num_heads)
        if self.pinball_work_num_heads <= 0:
            raise ValueError("pinball_work_num_heads must be positive")
        if self.pinball_work_dim % self.pinball_work_num_heads != 0:
            raise ValueError("pinball_work_dim must be divisible by pinball_work_num_heads")
        self.pinball_level_dims = _expand_int_list(pinball_level_dims, self.num_hier_levels, int(self.hidden_dim))
        if any(int(dim) <= 0 for dim in self.pinball_level_dims):
            raise ValueError("pinball_level_dims must contain positive integers")
        self.pinball_adapter_type = str(pinball_adapter_type).lower().replace("-", "_")
        if self.pinball_adapter_type not in {"identity", "dense", "low_rank"}:
            raise ValueError("pinball_adapter_type must be 'identity', 'dense', or 'low_rank'")
        self.pinball_adapter_rank = max(1, int(pinball_adapter_rank))
        self.pinball_level_cycle_enable = bool(pinball_level_cycle_enable)
        self.pinball_level_cycles = _expand_int_list(pinball_level_cycles, self.num_hier_levels, int(self.refinement_cycles))
        self.pinball_cycle_schedule = str(pinball_cycle_schedule).lower().replace("-", "_")
        if self.pinball_cycle_schedule not in {"frontloaded", "spread", "staggered_flush"}:
            raise ValueError("pinball_cycle_schedule must be 'frontloaded', 'spread', or 'staggered_flush'")
        self.pinball_message_fn = str(pinball_message_fn).lower().replace("-", "_")
        if self.pinball_message_fn != "original":
            raise ValueError("Only pinball_message_fn='original' is implemented for original Pinball hetero adapters")
        self.pinball_graph_active_compute = str(pinball_graph_active_compute).lower().replace("-", "_")
        if self.pinball_graph_active_compute not in {"all", "destination_only"}:
            raise ValueError("pinball_graph_active_compute must be 'all' or 'destination_only'")
        self.pinball_level_cycle_mode = str(pinball_level_cycle_mode).lower().replace("-", "_")
        if self.pinball_level_cycle_mode not in {"extra_cycles", "destination_only"}:
            raise ValueError("pinball_level_cycle_mode must be 'extra_cycles' or 'destination_only'")
        self.pinball_multirate_enable = bool(pinball_multirate_enable)
        self.pinball_multirate_schedule = str(pinball_multirate_schedule).lower().replace("-", "_")
        if self.pinball_multirate_schedule not in {"after_full_stack", "midpoint", "after_full_cycle", "midpoint_each_cycle"}:
            raise ValueError("pinball_multirate_schedule must be 'after_full_stack', 'midpoint', 'after_full_cycle', or 'midpoint_each_cycle'")
        self.pinball_multirate_midpoint_layer = pinball_multirate_midpoint_layer
        self.pinball_multirate_midpoint_repeats = max(1, int(pinball_multirate_midpoint_repeats))
        self.pinball_multirate_skip_after_last_cycle = bool(pinball_multirate_skip_after_last_cycle)
        self.pinball_multirate_debug = bool(pinball_multirate_debug)
        self.pinball_multirate_adaptive = bool(pinball_multirate_adaptive)
        self.pinball_multirate_energy_tau = float(pinball_multirate_energy_tau)
        self.pinball_multirate_max_cycles = max(1, int(pinball_multirate_max_cycles))
        self.pinball_multirate_active_rows = bool(pinball_multirate_active_rows)
        self.pinball_upper_refine_steps = max(0, int(pinball_upper_refine_steps))
        self.pinball_top_refine_steps = max(0, int(pinball_top_refine_steps))
        self.pinball_l3_workspace_tokens = max(0, int(pinball_l3_workspace_tokens))
        self.pinball_upper_refine_shared_weights = bool(pinball_upper_refine_shared_weights)
        self.pinball_top_refine_shared_weights = bool(pinball_top_refine_shared_weights)
        self.pinball_consolidate_after_upper = bool(pinball_consolidate_after_upper)
        if self.pinball_consolidate_after_upper:
            logger.warning("pinball_consolidate_after_upper is not implemented yet; ignoring it for now")
        self.pinball_upper_refine_every = max(1, int(pinball_upper_refine_every))
        self.pinball_top_refine_every = max(1, int(pinball_top_refine_every))
        self.pinball_multirate_attn_backend = str(pinball_multirate_attn_backend).lower()
        if self.pinball_multirate_attn_backend not in {"auto", "flash", "sdpa"}:
            raise ValueError("pinball_multirate_attn_backend must be 'auto', 'flash', or 'sdpa'")
        self.pinball_upper_refine_window = max(0, int(pinball_upper_refine_window))
        self.pinball_top_refine_window = max(0, int(pinball_top_refine_window))
        self.pinball_upper_refine_causal = bool(pinball_upper_refine_causal)
        self.pinball_top_refine_causal = bool(pinball_top_refine_causal)
        self.pinball_upper_cross_attn_steps = max(0, int(pinball_upper_cross_attn_steps))
        self.pinball_upper_query_topk_l2 = max(0, int(pinball_upper_query_topk_l2))
        self.pinball_upper_cross_attn_backend = str(pinball_upper_cross_attn_backend).lower()
        if self.pinball_upper_cross_attn_backend not in {"auto", "flash", "sdpa"}:
            raise ValueError("pinball_upper_cross_attn_backend must be 'auto', 'flash', or 'sdpa'")
        self.pinball_upper_cross_attn_causal = bool(pinball_upper_cross_attn_causal)
        self.pinball_upper_cross_attn_shared_weights = bool(pinball_upper_cross_attn_shared_weights)
        self.pinball_upper_update_l2_enable = bool(pinball_upper_update_l2_enable)
        self.pinball_upper_update_l2_scale_init = float(pinball_upper_update_l2_scale_init)
        self.pinball_upper_cross_write_scale_init = float(pinball_upper_cross_write_scale_init)
        self.pinball_cross_query_pairs = _parse_pinball_cross_query_pairs(pinball_cross_query_pairs)
        self.pinball_cross_query_steps = max(0, int(pinball_cross_query_steps))
        self.pinball_cross_query_topk = max(0, int(pinball_cross_query_topk))
        self.pinball_cross_query_l0_window = max(0, int(pinball_cross_query_l0_window))
        self.pinball_cross_query_memory_window = max(0, int(pinball_cross_query_memory_window))
        self.pinball_cross_query_backend = str(pinball_cross_query_backend).lower()
        if self.pinball_cross_query_backend not in {"auto", "flash", "sdpa"}:
            raise ValueError("pinball_cross_query_backend must be 'auto', 'flash', or 'sdpa'")
        self.pinball_cross_query_causal = bool(pinball_cross_query_causal)
        self.pinball_cross_query_shared_weights = bool(pinball_cross_query_shared_weights)
        self.pinball_cross_query_ffn_dims = [int(d) for d in (pinball_cross_query_ffn_dims or [])]
        self.pinball_cross_query_update_memory_enable = bool(pinball_cross_query_update_memory_enable)
        self.pinball_cross_query_selection = str(pinball_cross_query_selection).lower().replace("-", "_")
        if self.pinball_cross_query_selection not in {"global_mean", "per_query"}:
            raise ValueError("pinball_cross_query_selection must be 'global_mean' or 'per_query'")
        self.pinball_cross_query_write_scale_init = float(pinball_cross_query_write_scale_init)
        self.pinball_work_projection_effective = int(self.pinball_work_dim) != int(self.hidden_dim)
        if self.pinball_work_projection_effective and bool(self.share_transformers):
            raise ValueError("pinball_work_dim != hidden_dim requires share_transformers=False")
        self.pinball_work_in = nn.Identity() if not self.pinball_work_projection_effective else nn.Linear(self.hidden_dim, self.pinball_work_dim, bias=False)
        self.pinball_work_out = nn.Identity() if not self.pinball_work_projection_effective else nn.Linear(self.pinball_work_dim, self.hidden_dim, bias=False)
        self.use_aux_loss = use_aux_loss
        hier_aux_mode_norm = str(hier_aux_mode).lower()
        if hier_aux_mode_norm not in {"mean_mse", "jepa_mlp"}:
            logger.warning("Unknown hier_aux_mode='%s'; falling back to 'mean_mse'", hier_aux_mode_norm)
            hier_aux_mode_norm = "mean_mse"
        self.hier_aux_mode = hier_aux_mode_norm
        predictor_type_norm = str(hier_aux_predictor_type).lower()
        if predictor_type_norm not in {"linear", "mlp"}:
            logger.warning("Unknown hier_aux_predictor_type='%s'; falling back to 'mlp'", predictor_type_norm)
            predictor_type_norm = "mlp"
        self.hier_aux_predictor_type = predictor_type_norm
        hier_aux_loss_mode_norm = str(hier_aux_loss_mode).lower()
        if hier_aux_loss_mode_norm not in {"mse", "mse_norm", "nmse", "cosine"}:
            logger.warning("Unknown hier_aux_loss_mode='%s'; falling back to 'mse'", hier_aux_loss_mode_norm)
            hier_aux_loss_mode_norm = "mse"
        self.hier_aux_loss_mode = hier_aux_loss_mode_norm
        self.hier_aux_detach_target = bool(hier_aux_detach_target)
        self.hier_aux_link_l0_target = bool(hier_aux_link_l0_target)
        self.hier_aux_unit_norm = bool(hier_aux_unit_norm)
        self.hier_aux_w_l2_from_l3 = float(hier_aux_w_l2_from_l3)
        self.hier_aux_w_l1_from_l2 = float(hier_aux_w_l1_from_l2)
        self.hier_aux_w_l0_from_l1 = float(hier_aux_w_l0_from_l1)
        self.hier_aux_w_l0_from_l3 = float(hier_aux_w_l0_from_l3)
        self.hier_aux_ar_strict = bool(hier_aux_ar_strict)
        self.hier_aux_ar_disable_l0_from_l3 = bool(hier_aux_ar_disable_l0_from_l3)
        self.hier_aux_pair_predictors: Optional[nn.ModuleDict] = None
        if self.hier_aux_mode == "jepa_mlp":
            if self.hier_aux_predictor_type == "mlp":
                self.hier_aux_pair_predictors = nn.ModuleDict(
                    {
                        "l2_from_l3": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.SiLU(),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                        "l1_from_l2": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.SiLU(),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                        "l0_from_l1": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.SiLU(),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                        "l0_from_l3": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.SiLU(),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                    }
                )
            else:
                self.hier_aux_pair_predictors = nn.ModuleDict(
                    {
                        "l2_from_l3": nn.Linear(hidden_dim, hidden_dim),
                        "l1_from_l2": nn.Linear(hidden_dim, hidden_dim),
                        "l0_from_l1": nn.Linear(hidden_dim, hidden_dim),
                        "l0_from_l3": nn.Linear(hidden_dim, hidden_dim),
                    }
                )
        self.hier_ar_enable = bool(hier_ar_enable)
        self.hier_ar_allow_same_time = bool(hier_ar_allow_same_time)
        self.hier_ar_filter_zip = bool(hier_ar_filter_zip)
        self.l0_ar_enable = bool(l0_ar_enable)
        # Flags to track if LapPE warning was printed for a level
        self._lap_pe_warning_printed = [False] * self.num_hier_levels
         # Caches for reuse between forward passes
        self._cached_seq_len: Optional[int] = None
        self._cached_level_graphs: Optional[List[Data]] = None
        self._cached_unified_graph: Optional[Data] = None
        self._cached_unified_graph_key: Optional[Tuple[Any, ...]] = None
        self._cached_l0_ar_enable: Optional[bool] = None
        self._cached_geometry_tag: Optional[Tuple[Any, ...]] = None
        self.enable_l0_parent_edges = bool(enable_l0_parent_edges)
        self.l0_parent_edge_type_id = 7
        self.l0_parent_edges_bidirectional = bool(l0_parent_edges_bidirectional)
        self.l0_parent_edge_min_level = max(1, int(l0_parent_edge_min_level))
        if l0_parent_edge_max_level is None:
            self.l0_parent_edge_max_level = None
        else:
            self.l0_parent_edge_max_level = max(self.l0_parent_edge_min_level, int(l0_parent_edge_max_level))
        self.ensure_l0_past_l1_edges = bool(ensure_l0_past_l1_edges)
        self.ensure_past_hier_edges_all_levels = bool(ensure_past_hier_edges_all_levels)
        self.ensure_l0_past_parent_edges = bool(ensure_l0_past_parent_edges)
        self.l0_past_parent_min_level = max(1, int(l0_past_parent_min_level))
        if l0_past_parent_max_level is None:
            self.l0_past_parent_max_level = None
        else:
            self.l0_past_parent_max_level = max(self.l0_past_parent_min_level, int(l0_past_parent_max_level))
        self.l0_past_l1_edge_type_id = (
            None if l0_past_l1_edge_type_id is None else int(l0_past_l1_edge_type_id)
        )
        self.hier_recon_head = HierarchyReconHead(
            hidden_dim=self.hidden_dim,
            use_L2=True,
            use_L3=True,
        )
        
        self.lambda_ce_anchor = lambda_ce_anchor


        self.lambda_hier_aux = float(lambda_hier_aux)
        self._last_hier_aux_loss: Optional[torch.Tensor] = None
        self._last_hier_aux_pair_losses: Dict[str, float] = {}
        self._hier_aux_l0_from_l3_disabled_logged = False

        self.l0_alpha_enable = l0_alpha_enable
        self.l0_local_backend = str(l0_local_backend).lower()
        if self.l0_local_backend not in {"pyg", "flash", "xformers", "sdpa"}:
            self.l0_local_backend = "pyg"
        self.l0_local_window = max(0, int(l0_local_window))
        self.rope_level_axis_enable = bool(rope_level_axis_enable)
        self.rope_level_axis_scale = float(rope_level_axis_scale)
        self.local_attn_level_role_bias_enable = bool(local_attn_level_role_bias_enable)
        self.local_attn_level_role_bias_scale = float(local_attn_level_role_bias_scale)
        self.local_attn_flash_dtype_cast = bool(local_attn_flash_dtype_cast)
        self.cross_level_packed = bool(cross_level_packed)
        self.cross_level_qkv = str(cross_level_qkv or "shared").lower()
        self.local_attn_sampled_mode = str(local_attn_sampled_mode).lower()
        if self.local_attn_sampled_mode not in {"safe_sdpa", "flash_sorted", "off"}:
            self.local_attn_sampled_mode = "safe_sdpa"
        self.sparse_attn_mode = str(sparse_attn_mode).lower().replace("-", "_")
        if self.sparse_attn_mode not in {"off", "dst_block_checkpoint"}:
            self.sparse_attn_mode = "off"
        self.sparse_attn_chunk_size = max(0, int(sparse_attn_chunk_size))
        self.attention_source_gating_enable = bool(attention_source_gating_enable)
        self.attention_source_gate_init_graph = float(attention_source_gate_init_graph)
        self.attention_source_gate_init_local = float(attention_source_gate_init_local)
        self.attention_source_gate_init_hqd = float(attention_source_gate_init_hqd)
        self.attention_source_gate_debug = bool(attention_source_gate_debug)
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
        self.edge_conditioning_enable = bool(edge_conditioning_enable)
        self.edge_type_generator_enable = bool(edge_type_generator_enable)
        self.edge_type_embedding_dim = int(edge_type_embedding_dim)
        self.edge_condition_hidden_dim = int(edge_condition_hidden_dim)
        self.edge_condition_num_types = int(edge_condition_num_types)
        self.edge_logit_bias_enable = bool(edge_logit_bias_enable)
        self.edge_value_gate_enable = bool(edge_value_gate_enable)
        self.edge_logit_bias_per_head = bool(edge_logit_bias_per_head)
        self.edge_value_gate_per_head = bool(edge_value_gate_per_head)
        self.edge_value_gate_per_channel = bool(edge_value_gate_per_channel)
        self.edge_gate_init_identity = bool(edge_gate_init_identity)
        self.edge_logit_bias_init_zero = bool(edge_logit_bias_init_zero)
        self.edge_condition_dropout = float(edge_condition_dropout)
        self.edge_condition_debug = bool(edge_condition_debug)
        self.edge_node_condition_enable = bool(edge_node_condition_enable)
        self.edge_node_condition_detach = bool(edge_node_condition_detach)
        self.edge_node_condition_dim = int(edge_node_condition_dim)
        self.edge_node_condition_mode = str(edge_node_condition_mode).lower()
        self.edge_node_condition_zero_init = bool(edge_node_condition_zero_init)
        self.edge_gate_scale = float(edge_gate_scale)
        self.attn_sparse_qk_mult = float(attn_sparse_qk_mult)
        self.attn_sparse_v_mult = float(attn_sparse_v_mult)
        if self.rope_level_axis_enable and self.local_attn_level_role_bias_enable:
            logger.info(
                "Level-axis RoPE enabled; auto-disabling local-attn level role bias to preserve flash eligibility where possible."
            )
            self.local_attn_level_role_bias_enable = False
        self.rope_mode = str(rope_mode).lower()
        if self.rope_mode in {"2d", "axial", "grid2d"}:
            self.rope_mode = "2d_axial"
        if self.rope_mode not in {"auto", "1d", "2d_axial"}:
            self.rope_mode = "auto"

        # Multi-level local window attention config
        _la_levels = list(local_attn_levels) if local_attn_levels is not None else [0]
        _la_levels = sorted(set(int(l) for l in _la_levels))
        if local_attn_windows is not None and len(local_attn_windows) == len(_la_levels):
            _la_windows = [max(0, int(w)) for w in local_attn_windows]
        else:
            # Use l0_local_window for all requested levels
            _la_windows = [self.l0_local_window] * len(_la_levels)
        _la_causal = set(int(l) for l in (local_attn_causal_levels if local_attn_causal_levels is not None else [0]))
        # Build a dict: level -> {window, causal, backend}
        self.local_attn_config: Dict[int, Dict[str, Any]] = {}
        if self.graph_geometry_mode != "sequence":
            logger.info(
                "Graph geometry mode '%s': enabling spatial local-attention windows (radius-based) when backend != 'pyg'.",
                self.graph_geometry_mode,
            )
        logger.info(
            "Local-attention level role bias: enable=%s scale=%.3f",
            bool(self.local_attn_level_role_bias_enable),
            float(self.local_attn_level_role_bias_scale),
        )
        logger.info(
            "RoPE level axis: enable=%s scale=%.3f",
            bool(self.rope_level_axis_enable),
            float(self.rope_level_axis_scale),
        )
        for lvl, win in zip(_la_levels, _la_windows):
            if win > 0 and self.l0_local_backend != "pyg":
                self.local_attn_config[lvl] = {
                    "window": win,
                    "causal": (lvl in _la_causal),
                    "backend": self.l0_local_backend,
                }
        logger.info(
            "L0 local backend effective: %s window=%d active_levels=%s",
            str(self.l0_local_backend),
            int(self.l0_local_window),
            list(self.local_attn_config.keys()),
        )
        # Keep backward compat: if only level 0 and its window matches l0_local_window, nothing changes
        self._local_attn_active_levels: List[int] = sorted(self.local_attn_config.keys())

        self.TRM = TRM
        self.use_neighbor_sampling = use_neighbor_sampling
        self.num_neighbors = num_neighbors if num_neighbors is not None else [-1]
        self.sampling_batch_size = sampling_batch_size
        self.neighbor_sampling_backend = str(neighbor_sampling_backend).lower()
        self._neighbor_sampling_backend_cache_key: Optional[Tuple[str, str, int, bool, bool]] = None
        self._neighbor_sampling_backend_resolved: Optional[str] = None
        self.sampling_seed_budget = sampling_seed_budget
        self.zip_enable = zip_enable
        self.zip_attn_agg = zip_attn_agg
        self.zip_attn_ema_beta = zip_attn_ema_beta
        self.zip_depth = zip_depth
        self.zip_granularity = zip_granularity
        self.zip_persist = zip_persist
        self.zip_edge_decay = zip_edge_decay
        self.zip_edge_drop_threshold = zip_edge_drop_threshold
        self.zip_max_l3_pairs = zip_max_l3_pairs
        self.zip_max_l2_pairs = zip_max_l2_pairs
        self.zip_max_children_per_parent = zip_max_children_per_parent
        self.zip_max_candidate_edges = zip_max_candidate_edges
        self.zip_max_dyn_edges_total = zip_max_dyn_edges_total
        self.zip_max_dyn_edges_per_sample = max(0, int(zip_max_dyn_edges_per_sample))
        self.zip_log_enable = zip_log_enable
        self.zip_edge_select_mode = zip_edge_select_mode
        self.zip_edge_percentile = zip_edge_percentile
        self.zip_edge_relative_ratio = zip_edge_relative_ratio
        self.zip_l3_all_pairs_every = zip_l3_all_pairs_every
        self.zip_l3_pairs_cap = zip_l3_pairs_cap
        self.zip_l3_all_pairs_sample = zip_l3_all_pairs_sample
        self.zip_l3_pair_sampling_mode = zip_l3_pair_sampling_mode
        zip_score_mode = str(zip_score_mode).lower()
        if zip_score_mode not in {"mp_full", "fast_qk"}:
            logger.warning("Unknown zip_score_mode='%s'; falling back to 'mp_full'", zip_score_mode)
            zip_score_mode = "mp_full"
        self.zip_score_mode = zip_score_mode
        zip_execution_mode = str(zip_execution_mode).lower()
        if zip_execution_mode not in {"edge_mutation", "ephemeral_msg"}:
            logger.warning("Unknown zip_execution_mode='%s'; falling back to 'edge_mutation'", zip_execution_mode)
            zip_execution_mode = "edge_mutation"
        self.zip_execution_mode = zip_execution_mode
        zip_select_scope = str(zip_select_scope).lower()
        if zip_select_scope not in {"global", "per_dst"}:
            logger.warning("Unknown zip_select_scope='%s'; falling back to 'global'", zip_select_scope)
            zip_select_scope = "global"
        self.zip_select_scope = zip_select_scope
        self.zip_per_dst_topk = max(1, int(zip_per_dst_topk))
        self.zip_l3_pair_attn_weight = zip_l3_pair_attn_weight
        self.zip_l3_pair_entropy_weight = zip_l3_pair_entropy_weight
        self.zip_gate_mode = zip_gate_mode
        self.zip_gate_tau = zip_gate_tau
        self.zip_gate_center = zip_gate_center
        self.zip_gate_scope = zip_gate_scope
        self.zip_paramfree_gate = bool(zip_paramfree_gate)
        self.zip_paramfree_gate_tau = float(zip_paramfree_gate_tau)
        self.zip_paramfree_gate_center = float(zip_paramfree_gate_center)
        self.zip_mass_budget_per_dst = float(zip_mass_budget_per_dst)
        self.zip_norm_clip_ratio = float(zip_norm_clip_ratio)
        self.zip_warmup_steps = max(0, int(zip_warmup_steps))
        self.zip_select_from_softmax = bool(zip_select_from_softmax)
        self.zip_use_beta_gate = bool(zip_use_beta_gate)
        self.zip_msg_eta = float(zip_msg_eta)
        self.zip_beta_init_bias = float(zip_beta_init_bias)
        self.hierarchical_query_descent_enable = bool(hierarchical_query_descent_enable)
        self.hqd_topk_l3 = max(1, int(hqd_topk_l3))
        self.hqd_topk_l2 = max(1, int(hqd_topk_l2))
        self.hqd_topk_l1 = max(1, int(hqd_topk_l1))
        self.hqd_topk_l0 = max(1, int(hqd_topk_l0))
        self.hqd_l0_topk_enable = bool(hqd_l0_topk_enable)
        self.hqd_include_local_window = bool(hqd_include_local_window)
        self.hqd_local_window_size = max(0, int(hqd_local_window_size))
        if isinstance(hqd_causal, str):
            hqd_causal_norm = hqd_causal.strip().lower()
            if hqd_causal_norm in {"auto", "none", ""}:
                self.hqd_causal = None
            elif hqd_causal_norm in {"true", "1", "yes", "y"}:
                self.hqd_causal = True
            elif hqd_causal_norm in {"false", "0", "no", "n"}:
                self.hqd_causal = False
            else:
                logger.warning("Unknown hqd_causal='%s'; falling back to auto", hqd_causal)
                self.hqd_causal = None
        elif hqd_causal is None:
            self.hqd_causal = None
        else:
            self.hqd_causal = bool(hqd_causal)
        self.hqd_use_existing_zipper_projections = bool(hqd_use_existing_zipper_projections)
        self.hqd_debug = bool(hqd_debug)
        self.verbose = bool(verbose)
        self.hqd_granularity = str(hqd_granularity).strip().lower()
        if self.hqd_granularity not in {"per_layer", "per_cycle"}:
            logger.warning("Unknown hqd_granularity='%s'; falling back to per_layer", hqd_granularity)
            self.hqd_granularity = "per_layer"
        self.hqd_every_n = int(hqd_every_n)
        if self.hqd_every_n == 0:
            logger.warning("hqd_every_n=0 is invalid; disabling every-N scheduling (use hqd_granularity instead).")
            self.hqd_every_n = -1
        # Shift the every-N schedule: HQD runs at layers offset, offset+n, ... With the
        # backbone witness source, offset >= 1 means every HQD layer already has a packed-
        # lane capture from an earlier layer this forward — no recompute fallback at layer 0.
        self.hqd_every_n_offset = max(0, int(hqd_every_n_offset))
        self.hqd_reuse_previous = bool(hqd_reuse_previous)
        self.hqd_reuse_max_age = max(0, int(hqd_reuse_max_age))
        self.hqd_query_chunk_size = max(1, int(hqd_query_chunk_size))
        self.hqd_query_level = max(0, min(2, int(hqd_query_level)))
        self.hqd_stop_level = max(0, min(2, int(hqd_stop_level)))
        self.hqd_keep_stage_survivors = bool(hqd_keep_stage_survivors)
        # Read the parameter, not self: self.hqd_static_descent is assigned further below.
        if self.hqd_keep_stage_survivors and not bool(hqd_static_descent):
            # Only the static core accumulates survivors; the stage-by-stage reference
            # descent rebuilds candidate sets per level and would silently ignore this.
            logger.warning(
                "hqd_keep_stage_survivors requires hqd_static_descent=True; "
                "the reference descent ignores it and will read L0 only.")
        if self.hqd_stop_level > self.hqd_query_level:
            logger.warning("hqd_stop_level=%d > hqd_query_level=%d; clamping stop_level to query_level", self.hqd_stop_level, self.hqd_query_level)
            self.hqd_stop_level = self.hqd_query_level
        self.hqd_handoff_to_l0 = bool(hqd_handoff_to_l0)
        self.hqd_global_topk = max(0, int(hqd_global_topk))
        self.hqd_assume_disjoint_children = bool(hqd_assume_disjoint_children)
        self.hqd_validate_disjoint_children = bool(hqd_validate_disjoint_children)
        self.hqd_sparse_project_active_only = bool(hqd_sparse_project_active_only)
        self.hqd_tiled_apply = bool(hqd_tiled_apply)
        self.hqd_static_descent = bool(hqd_static_descent)
        self.hqd_static_compile = bool(hqd_static_compile)
        self._hqd_static_compiled = None
        self._hqd_static_fallback_logged = False
        self.hqd_tile_size = max(1, int(hqd_tile_size))
        self.hqd_tile_topk = max(1, int(hqd_tile_topk))
        self.hqd_attn_impl = str(hqd_attn_impl).lower() if str(hqd_attn_impl).lower() in ("scatter", "dense") else "scatter"
        self.hqd_dense_backend = str(hqd_dense_backend).lower() if str(hqd_dense_backend).lower() in ("sdpa", "flash") else "sdpa"
        self.hqd_descent_stop_at = int(hqd_descent_stop_at) if int(hqd_descent_stop_at) in (0, 2) else 0
        self.hqd_shallow_read_level = int(hqd_shallow_read_level) if int(hqd_shallow_read_level) in (0, 2, 3) else 0
        # Bidirectional coarse routing (option 2): coarse nodes attend among themselves and
        # surface bag<->bag L0 edges. NOT AR-causal (coarse-as-query). Bidi/masked-diffusion only.
        self.hqd_coarse_route_levels = sorted({int(l) for l in (hqd_coarse_route_levels or []) if int(l) in (1, 2, 3)})
        self.hqd_coarse_route_topk = max(1, int(hqd_coarse_route_topk))
        self.hqd_select_inside_message_passing = bool(hqd_select_inside_message_passing)
        self.hqd_graph_witness_enable = bool(hqd_graph_witness_enable)
        self.hqd_graph_witness_topk = max(1, int(hqd_graph_witness_topk))
        self.hqd_packed_witness_read = bool(hqd_packed_witness_read)
        _packed_src = str(hqd_packed_witness_source).strip().lower()
        self.hqd_packed_witness_source = _packed_src if _packed_src in {"graph", "recompute", "backbone"} else "graph"
        self.hqd_packed_witness_chunk_size = max(1, int(hqd_packed_witness_chunk_size))
        if (
            self.hqd_packed_witness_source == "backbone"
            and getattr(self, "hierarchical_query_descent_enable", False)
            and not bool(cross_level_packed)
        ):
            logger.warning(
                "hqd_packed_witness_source=backbone harvests witnesses from the packed "
                "cross-level lane, which needs cross_level_packed=true; witness builds "
                "will fall back to the recompute path until it is enabled."
            )

        # Witness packets (see _build_witness_l0_ids). Active only when HQD is enabled,
        # query_level == 0, and at least one of summary/rare is on.
        self.use_witness_packets = bool(use_witness_packets)
        self.use_summary_witnesses = bool(use_summary_witnesses)
        self.use_rare_witnesses = bool(use_rare_witnesses)
        self.summary_witness_per_head = bool(summary_witness_per_head)
        self.witness_k_summary = max(0, int(witness_k_summary))
        self.witness_k_rare = max(0, int(witness_k_rare))
        self.witness_max_per_parent = max(0, int(witness_max_per_parent))
        # Which parent levels expose L0 bags. [1] = today (L1->L0 direct children).
        # [1,2,3] also builds cross-level L2->L0 / L3->L0 bags so a coarse node can be
        # read and reach its important L0 descendants without descending.
        self.witness_levels = sorted({int(l) for l in (witness_levels or [1]) if int(l) in (1, 2, 3)}) or [1]
        self.witness_lambda_rare = float(witness_lambda_rare)
        self.witness_score_bias = float(witness_score_bias)
        _read_levels = hqd_read_levels if hqd_read_levels is not None else [0]
        self.hqd_read_levels = sorted({int(l) for l in _read_levels if 0 <= int(l) <= 3}) or [0]
        self.hqd_window_bag_levels = sorted({int(l) for l in (hqd_window_bag_levels or []) if int(l) in (1, 2, 3)})
        _bag_routing = str(hqd_window_bag_routing).strip().lower()
        self.hqd_window_bag_routing = _bag_routing if _bag_routing in {"global", "per_position"} else "global"
        self.hqd_window_bag_topk = max(0, int(hqd_window_bag_topk))

        batch_mode = str(refinement_batch_mode).lower()
        if batch_mode not in {"blockdiag", "true_batch_nozip"}:
            logger.warning("Unknown refinement_batch_mode='%s'; falling back to 'blockdiag'", batch_mode)
            batch_mode = "blockdiag"
        self.refinement_batch_mode = batch_mode
        self.true_batch_strict = bool(true_batch_strict)
        aux_mode = str(true_batch_aux_mode).lower()
        if aux_mode not in {"per_sample_mean", "exact_blockdiag"}:
            logger.warning("Unknown true_batch_aux_mode='%s'; falling back to 'per_sample_mean'", aux_mode)
            aux_mode = "per_sample_mean"
        self.true_batch_aux_mode = aux_mode
        ae_mode = str(autoenc_graph_mode).lower()
        if ae_mode not in {"off", "twin_shared_l3"}:
            logger.warning("Unknown autoenc_graph_mode='%s'; falling back to 'off'", ae_mode)
            ae_mode = "off"
        self.autoenc_graph_mode = ae_mode
        self.autoenc_coupled_feedback = bool(autoenc_coupled_feedback)
        self.token_unet_enable = bool(token_unet_enable)
        unet_mode = str(token_unet_mode).lower()
        if unet_mode not in {"stem", "coarse_tokenize"}:
            logger.warning("Unknown token_unet_mode='%s'; falling back to 'stem'", unet_mode)
            unet_mode = "stem"
        self.token_unet_mode = unet_mode
        unet_dim = str(token_unet_dim).lower()
        if unet_dim not in {"auto", "1d", "2d"}:
            logger.warning("Unknown token_unet_dim='%s'; falling back to 'auto'", unet_dim)
            unet_dim = "auto"
        self.token_unet_dim = unet_dim
        self.token_unet_2d_causal = bool(token_unet_2d_causal)
        self.token_unet_is_2d = False
        self.token_unet_scale = int(token_unet_scale)
        self.token_unet_kernel_size = int(token_unet_kernel_size)
        self.token_unet_dropout = float(token_unet_dropout)
        self.token_unet_right_edge_targets = bool(token_unet_right_edge_targets)
        self.token_unet_lookahead_decode_enable = bool(token_unet_lookahead_decode_enable)
        self.token_unet_lookahead_kernel_size = int(token_unet_lookahead_kernel_size)
        self.token_unet_lookahead_blocks = int(token_unet_lookahead_blocks)
        self.token_unet_block = str(token_unet_block).lower()
        self.token_unet_dynamic = bool(token_unet_dynamic)
        self.token_unet_channel_growth = int(token_unet_channel_growth)
        self.token_unet_max_channels = int(token_unet_max_channels)
        self.token_unet_concat_skips = bool(token_unet_concat_skips)
        self.token_unet_hyena_length_invariant = bool(token_unet_hyena_length_invariant)
        self.rgb_token_unet_enable = bool(rgb_token_unet_enable)
        self.rgb_token_unet_downsample = max(1, int(rgb_token_unet_downsample))
        self.rgb_token_unet_base_channels = max(16, int(rgb_token_unet_base_channels))
        self.rgb_token_unet_kernel_size = max(1, int(rgb_token_unet_kernel_size))
        self.rgb_token_unet_decode_kernel_size = max(1, int(rgb_token_unet_decode_kernel_size))
        self.rgb_token_unet_decode_separable = bool(rgb_token_unet_decode_separable)
        self.rgb_token_unet_max_channels = max(self.rgb_token_unet_base_channels, int(rgb_token_unet_max_channels))
        self._token_unet_emit_lookahead_logits = False
        self._last_token_unet_lookahead_logits = None
        self._runtime_l0_grid_shape: Optional[Tuple[int, int]] = None
        self._last_refinement_batch_mode: Optional[str] = None
        self._last_refinement_batch_fallback: Optional[str] = None
        self._lap_pe_missing_warned: bool = False
        self._l0_local_runtime_logged: bool = False
        self._rope_runtime_logged_keys = set()
        self._local_attn_runtime_logged_keys = set()
        self._last_zip_added_total: Optional[int] = None
        self._last_zip_stage_stats: Optional[Dict[str, int]] = None
        self._last_zip_profile_stats: Optional[Dict[str, float]] = None
        self._last_zip_msg_norm_ratio_mean: Optional[float] = None
        self._last_zip_msg_norm_ratio_max: Optional[float] = None
        self._last_hqd_added_total: Optional[int] = None
        self._last_hqd_selected_total: Optional[int] = None
        self._last_hqd_stage_stats: Optional[Dict[str, int]] = None
        self._last_hqd_profile_stats: Optional[Dict[str, float]] = None
        self._last_hqd_avg_l0: Optional[float] = None
        self._last_hqd_mean_dist = None
        self._last_hqd_local_frac = None
        self._hqd_reuse_cache: Optional[Dict[str, Any]] = None
        self._hqd_runtime_logged: bool = False
        self._hqd_skeleton_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._hqd_skeleton_cache_max_entries: int = 8
        self._zipper_children_table_cache: Dict[Tuple[Any, ...], Dict[str, torch.Tensor]] = {}
        self._zipper_children_table_cache_max_entries: int = 16
        self._zipper_children_cache_hits: int = 0
        self._zipper_children_cache_misses: int = 0
        self._zipper_inject_workspace: Dict[Tuple[Any, ...], torch.Tensor] = {}
        self._zipper_inject_workspace_max_entries: int = 4
        self._zip_msg_step: int = 0
        
        if tokenizer.mask_token_id is None:
            # Attempt to add a mask token if missing - might be better done outside
            # For now, raise error or use a default like padding id
            logger.warning("Tokenizer does not have a mask_token_id. Using pad_token_id as fallback for summary init.")
            self.mask_token_id = tokenizer.pad_token_id
            if self.mask_token_id is None:
                 # If still None, maybe use 0 or raise error
                 self.mask_token_id = 0
                 logger.error("No mask or pad token found. Using token 0 for summary init. This might be suboptimal.")
            # raise ValueError("Tokenizer must have a mask_token_id for imputation pooling.")
        else:
            self.mask_token_id = tokenizer.mask_token_id

        # Create token embedding
        #self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        if input_mode == "tokens":
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        elif input_mode == "features":
            self.token_embedding = nn.Linear(self.vocab_size, self.hidden_dim, bias=False)
            print("Using feature input mode with linear embedding layer.")
            #self.token_embedding = nn.Identity()
        else:
            raise ValueError("input_mode must be 'tokens' or 'features'")
        
        if self.input_mode == "features":
            self.mask_vector = nn.Parameter(torch.zeros(1, hidden_dim))

        self.token_unet: Optional[nn.Module] = None
        if self.token_unet_enable:
            use_2d = False
            if self.token_unet_dim == "2d":
                use_2d = True
            elif self.token_unet_dim == "auto":
                use_2d = str(getattr(self, "graph_geometry_mode", "sequence")).lower() == "grid2d"
            self.token_unet_is_2d = bool(use_2d)
            if self.token_unet_is_2d and self.token_unet_block == "hyena":
                logger.warning("token_unet_block='hyena' is 1D-only; using CNN blocks for the 2D token U-Net.")
            if self.token_unet_is_2d:
                self.token_unet = SpatialTokenUNet2D(
                    hidden_dim=self.hidden_dim,
                    scale=self.token_unet_scale,
                    kernel_size=self.token_unet_kernel_size,
                    dropout=self.token_unet_dropout,
                    lookahead_enable=self.token_unet_lookahead_decode_enable,
                    lookahead_kernel_size=self.token_unet_lookahead_kernel_size,
                    lookahead_blocks=self.token_unet_lookahead_blocks,
                    causal=self.token_unet_2d_causal,
                )
            elif bool(getattr(self, "token_unet_dynamic", False)):
                self.token_unet = CausalDynamicTokenUNet(
                    hidden_dim=self.hidden_dim,
                    scale=self.token_unet_scale,
                    kernel_size=self.token_unet_kernel_size,
                    dropout=self.token_unet_dropout,
                    block=self.token_unet_block,
                    channel_growth=int(getattr(self, "token_unet_channel_growth", 256)),
                    max_channels=int(getattr(self, "token_unet_max_channels", 0)),
                    concat_skips=bool(getattr(self, "token_unet_concat_skips", True)),
                    lookahead_enable=self.token_unet_lookahead_decode_enable,
                    lookahead_kernel_size=self.token_unet_lookahead_kernel_size,
                    lookahead_blocks=self.token_unet_lookahead_blocks,
                    gradient_checkpointing=bool(getattr(self, "use_gradient_checkpointing", False)),
                    hyena_length_invariant=bool(getattr(self, "token_unet_hyena_length_invariant", False)),
                )
            else:
                self.token_unet = CausalTokenUNet(
                    hidden_dim=self.hidden_dim,
                    scale=self.token_unet_scale,
                    kernel_size=self.token_unet_kernel_size,
                    dropout=self.token_unet_dropout,
                    lookahead_enable=self.token_unet_lookahead_decode_enable,
                    lookahead_kernel_size=self.token_unet_lookahead_kernel_size,
                    lookahead_blocks=self.token_unet_lookahead_blocks,
                    block=self.token_unet_block,
                    gradient_checkpointing=bool(getattr(self, "use_gradient_checkpointing", False)),
                )
            logger.info(
                "Enabled token U-Net: mode=%s dim=%s block=%s causal2d=%s scale=%d kernel=%d dropout=%.3f lookahead=%s lookahead_kernel=%d lookahead_blocks=%d",
                str(self.token_unet_mode),
                "2d" if self.token_unet_is_2d else "1d",
                ("cnn" if self.token_unet_is_2d else str(self.token_unet_block)),
                bool(self.token_unet_2d_causal),
                int(self.token_unet_scale),
                int(self.token_unet_kernel_size),
                float(self.token_unet_dropout),
                bool(self.token_unet_lookahead_decode_enable),
                int(self.token_unet_lookahead_kernel_size),
                int(self.token_unet_lookahead_blocks),
            )

        self.rgb_token_unet: Optional[RGBTokenUNet2D] = None
        if self.rgb_token_unet_enable:
            self.rgb_token_unet = RGBTokenUNet2D(
                token_dim=int(self.vocab_size),
                downsample_factor=int(self.rgb_token_unet_downsample),
                base_channels=int(self.rgb_token_unet_base_channels),
                kernel_size=int(self.rgb_token_unet_kernel_size),
                decode_kernel_size=int(self.rgb_token_unet_decode_kernel_size),
                decode_separable=bool(self.rgb_token_unet_decode_separable),
                max_channels=int(self.rgb_token_unet_max_channels),
                dropout=float(self.token_unet_dropout),
                cond_dim=int(self.hidden_dim),
            )
            logger.info(
                "Enabled RGB token U-Net bridge: token_dim=%d downsample=%d base_channels=%d encoder_channels=%s decoder_channels=%s kernel=%d decode_kernel=%d decode_sep=%s max_channels=%d cond_dim=%d",
                int(self.vocab_size),
                int(self.rgb_token_unet_downsample),
                int(self.rgb_token_unet_base_channels),
                list(self.rgb_token_unet.encoder_channels),
                list(self.rgb_token_unet.decoder_channels),
                int(self.rgb_token_unet_kernel_size),
                int(self.rgb_token_unet_decode_kernel_size),
                bool(self.rgb_token_unet_decode_separable),
                int(self.rgb_token_unet_max_channels),
                int(self.hidden_dim),
            )
        
        # Rotary positional encoding
        self.rotary_pos_enc = RotaryPositionalEncoding(hidden_dim, max_seq_len)
        
        # Old Lagrangian positional encoding do not use
        #self.lagrangian_pos_enc = LagrangianPositionalEncoding(hidden_dim)
        
        if self.lap_pe_k > 0:
            self.lap_pe_transform = AddLaplacianEigenvectorPE(
                k=self.lap_pe_k,
                attr_name='lap_pe', # Features will be stored in data.lap_pe
                is_undirected=True # Assume graph is undirected for efficiency
            )
            # Projection layer maps k features -> hidden_dim
            self.lap_pe_proj = nn.Linear(self.lap_pe_k, hidden_dim)
        else:
            self.lap_pe_transform = None
            self.lap_pe_proj = None
        self._lap_pe_device_cache_key: Optional[Tuple[Any, ...]] = None
        self._lap_pe_device_cache: Optional[torch.Tensor] = None

        # Shared normalization config (needed before transformer construction)
        self.norm_type = "rmsnorm" if norm_type is None else str(norm_type).lower()
        self.norm_eps = float(norm_eps)

        # Create transformer layers for each level
        self.level_transformers = nn.ModuleList()
        for level_idx, level_layers in enumerate(num_layers):
            level_modules = nn.ModuleList()
            logger.info(f"Creating {level_layers} dedicated level layers for hierarchy build.")
            for _ in range(level_layers):
                level_modules.append(
                    HierarchicalTransformerLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        edge_dim=hidden_dim if use_edge_attr else None,
                        use_edge_attr=use_edge_attr,
                        learn_edge_from_attn=self.learn_edge_from_attn,
                        max_seq_len=max_seq_len,
                        l0_local_backend=self.l0_local_backend,
                        l0_local_window=self.l0_local_window,
                        l0_local_causal_default=self.hier_ar_enable,
                        local_attn_config=self.local_attn_config,
                        local_attn_level_role_bias_enable=self.local_attn_level_role_bias_enable,
                        local_attn_level_role_bias_scale=self.local_attn_level_role_bias_scale,
                        local_attn_flash_dtype_cast=self.local_attn_flash_dtype_cast,
                        local_pack_level_bias=bool(getattr(self, "local_pack_level_bias", False)),
                        local_pack_lane_merge=bool(getattr(self, "local_pack_lane_merge", False)),
                        local_pack_coarse_global=bool(getattr(self, "local_pack_coarse_global", False)),
                        local_pack_l0_coarse_bands=bool(getattr(self, "local_pack_l0_coarse_bands", False)),
                        local_pack_l0_window=int(getattr(self, "local_pack_l0_window", 0)),
                        local_pack_l0_coarse_windows=list(getattr(self, "local_pack_l0_coarse_windows", []) or []),
                        local_pack_l0_bands_checkpoint=bool(getattr(self, "local_pack_l0_bands_checkpoint", False)),
                        local_pack_l0_coarse_rank_window=int(getattr(self, "local_pack_l0_coarse_rank_window", 0)),
                        hqd_keep_stage_survivors=bool(getattr(self, "hqd_keep_stage_survivors", False)),
                        local_pack_coarse_global_gate_init=float(getattr(self, "local_pack_coarse_global_gate_init", 0.0)),
                        local_pack_flex_union=bool(getattr(self, "local_pack_flex_union", False)),
                        hier_node_dropout=float(getattr(self, "hier_node_dropout", 0.0)),
                        hier_node_dropout_per_level=getattr(self, "hier_node_dropout_per_level", None),
                        local_pack_bidirectional=bool(getattr(self, "local_pack_bidirectional", False)),
                        local_pack_rope_axial=bool(getattr(self, "local_pack_rope_axial", False)),
                        hqd_read_prerope=bool(getattr(self, "xq_nominate_read_prerope", False)),
                        hqd_read_sink=bool(getattr(self, "xq_nominate_read_sink", False)),
                        cross_level_packed=self.cross_level_packed,
                        cross_level_qkv=self.cross_level_qkv,
                        local_attn_sampled_mode=self.local_attn_sampled_mode,
                        sparse_attn_mode=self.sparse_attn_mode,
                        sparse_attn_chunk_size=self.sparse_attn_chunk_size,
                        norm_type=self.norm_type,
                        norm_eps=self.norm_eps,
                        rope_level_axis_enable=self.rope_level_axis_enable,
                        rope_level_axis_scale=self.rope_level_axis_scale,
                        rope_mode=self.rope_mode,
                        attention_source_gating_enable=self.attention_source_gating_enable,
                        attention_source_gate_init_graph=self.attention_source_gate_init_graph,
                        attention_source_gate_init_local=self.attention_source_gate_init_local,
                        attention_source_gate_init_hqd=self.attention_source_gate_init_hqd,
                        attention_source_gate_debug=self.attention_source_gate_debug,
                        lateral_edge_trace_enable=self.lateral_edge_trace_enable,
                        lateral_edge_trace_mode=self.lateral_edge_trace_mode,
                        lateral_edge_trace_decay=self.lateral_edge_trace_decay,
                        lateral_edge_trace_eta=self.lateral_edge_trace_eta,
                        lateral_edge_trace_alpha=self.lateral_edge_trace_alpha,
                        lateral_edge_trace_max=self.lateral_edge_trace_max,
                        lateral_edge_trace_per_head=self.lateral_edge_trace_per_head,
                        lateral_edge_trace_credit=self.lateral_edge_trace_credit,
                        lateral_edge_trace_center_per_dst=self.lateral_edge_trace_center_per_dst,
                        lateral_edge_trace_update_during_eval=self.lateral_edge_trace_update_during_eval,
                        lateral_edge_trace_detach=self.lateral_edge_trace_detach,
                        lateral_edge_trace_debug=self.lateral_edge_trace_debug,
                        edge_conditioning_enable=self.edge_conditioning_enable,
                        edge_type_generator_enable=self.edge_type_generator_enable,
                        edge_type_embedding_dim=self.edge_type_embedding_dim,
                        edge_condition_hidden_dim=self.edge_condition_hidden_dim,
                        edge_condition_num_types=self.edge_condition_num_types,
                        edge_logit_bias_enable=self.edge_logit_bias_enable,
                        edge_value_gate_enable=self.edge_value_gate_enable,
                        edge_logit_bias_per_head=self.edge_logit_bias_per_head,
                        edge_value_gate_per_head=self.edge_value_gate_per_head,
                        edge_value_gate_per_channel=self.edge_value_gate_per_channel,
                        edge_gate_init_identity=self.edge_gate_init_identity,
                        edge_logit_bias_init_zero=self.edge_logit_bias_init_zero,
                        edge_condition_dropout=self.edge_condition_dropout,
                        edge_condition_debug=self.edge_condition_debug,
                        edge_node_condition_enable=self.edge_node_condition_enable,
                        edge_node_condition_detach=self.edge_node_condition_detach,
                        edge_node_condition_dim=self.edge_node_condition_dim,
                        edge_node_condition_mode=self.edge_node_condition_mode,
                        edge_node_condition_zero_init=self.edge_node_condition_zero_init,
                        edge_gate_scale=self.edge_gate_scale,
                        attn_sparse_qk_mult=self.attn_sparse_qk_mult,
                        attn_sparse_v_mult=self.attn_sparse_v_mult,
                    )
                )
            self.level_transformers.append(level_modules)
        
        # Create separate refinement transformer layers only if not sharing
        self.refinement_transformers = None # Initialize as None
        #if self.refinement_style == "unified" and not self.share_transformers:
        if not self.share_transformers:
            if self.num_refinement_layers > 0:
                self.refinement_transformers = nn.ModuleList()
                logger.info(f"Creating {self.num_refinement_layers} dedicated refinement layers for unified style.")
                for _ in range(self.num_refinement_layers):
                    self.refinement_transformers.append(HierarchicalTransformerLayer(
                        hidden_dim=self.pinball_work_dim, num_heads=self.pinball_work_num_heads, dropout=dropout,
                        edge_dim=self.pinball_work_dim if self.use_edge_attr else None,
                        use_edge_attr=self.use_edge_attr,
                        per_level_local_qkv=self.per_level_local_qkv,
                        num_local_levels=int(getattr(self, "num_hier_levels", 4)),
                        per_level_ffn_dims=self.per_level_ffn_dims,
                        per_level_attn_mult=self.per_level_attn_mult,
                        local_attn_head_dim=int(getattr(self, "local_attn_head_dim", 0)),
                        learn_edge_from_attn=self.learn_edge_from_attn,
                        max_seq_len=self.max_seq_len,
                        l0_local_backend=self.l0_local_backend,
                        l0_local_window=self.l0_local_window,
                        l0_local_causal_default=self.hier_ar_enable,
                        local_attn_config=self.local_attn_config,
                        local_attn_level_role_bias_enable=self.local_attn_level_role_bias_enable,
                        local_attn_level_role_bias_scale=self.local_attn_level_role_bias_scale,
                        local_attn_flash_dtype_cast=self.local_attn_flash_dtype_cast,
                        local_pack_level_bias=bool(getattr(self, "local_pack_level_bias", False)),
                        local_pack_lane_merge=bool(getattr(self, "local_pack_lane_merge", False)),
                        local_pack_coarse_global=bool(getattr(self, "local_pack_coarse_global", False)),
                        local_pack_l0_coarse_bands=bool(getattr(self, "local_pack_l0_coarse_bands", False)),
                        local_pack_l0_window=int(getattr(self, "local_pack_l0_window", 0)),
                        local_pack_l0_coarse_windows=list(getattr(self, "local_pack_l0_coarse_windows", []) or []),
                        local_pack_l0_bands_checkpoint=bool(getattr(self, "local_pack_l0_bands_checkpoint", False)),
                        local_pack_l0_coarse_rank_window=int(getattr(self, "local_pack_l0_coarse_rank_window", 0)),
                        hqd_keep_stage_survivors=bool(getattr(self, "hqd_keep_stage_survivors", False)),
                        local_pack_coarse_global_gate_init=float(getattr(self, "local_pack_coarse_global_gate_init", 0.0)),
                        local_pack_flex_union=bool(getattr(self, "local_pack_flex_union", False)),
                        hier_node_dropout=float(getattr(self, "hier_node_dropout", 0.0)),
                        hier_node_dropout_per_level=getattr(self, "hier_node_dropout_per_level", None),
                        local_pack_bidirectional=bool(getattr(self, "local_pack_bidirectional", False)),
                        local_pack_rope_axial=bool(getattr(self, "local_pack_rope_axial", False)),
                        hqd_read_prerope=bool(getattr(self, "xq_nominate_read_prerope", False)),
                        hqd_read_sink=bool(getattr(self, "xq_nominate_read_sink", False)),
                        cross_level_packed=self.cross_level_packed,
                        cross_level_qkv=self.cross_level_qkv,
                        local_attn_sampled_mode=self.local_attn_sampled_mode,
                        sparse_attn_mode=self.sparse_attn_mode,
                        sparse_attn_chunk_size=self.sparse_attn_chunk_size,
                        norm_type=self.norm_type,
                        norm_eps=self.norm_eps,
                        rope_level_axis_enable=self.rope_level_axis_enable,
                        rope_level_axis_scale=self.rope_level_axis_scale,
                        rope_mode=self.rope_mode,
                        attention_source_gating_enable=self.attention_source_gating_enable,
                        attention_source_gate_init_graph=self.attention_source_gate_init_graph,
                        attention_source_gate_init_local=self.attention_source_gate_init_local,
                        attention_source_gate_init_hqd=self.attention_source_gate_init_hqd,
                        attention_source_gate_debug=self.attention_source_gate_debug,
                        lateral_edge_trace_enable=self.lateral_edge_trace_enable,
                        lateral_edge_trace_mode=self.lateral_edge_trace_mode,
                        lateral_edge_trace_decay=self.lateral_edge_trace_decay,
                        lateral_edge_trace_eta=self.lateral_edge_trace_eta,
                        lateral_edge_trace_alpha=self.lateral_edge_trace_alpha,
                        lateral_edge_trace_max=self.lateral_edge_trace_max,
                        lateral_edge_trace_per_head=self.lateral_edge_trace_per_head,
                        lateral_edge_trace_credit=self.lateral_edge_trace_credit,
                        lateral_edge_trace_center_per_dst=self.lateral_edge_trace_center_per_dst,
                        lateral_edge_trace_update_during_eval=self.lateral_edge_trace_update_during_eval,
                        lateral_edge_trace_detach=self.lateral_edge_trace_detach,
                        lateral_edge_trace_debug=self.lateral_edge_trace_debug,
                        edge_conditioning_enable=self.edge_conditioning_enable,
                        edge_type_generator_enable=self.edge_type_generator_enable,
                        edge_type_embedding_dim=self.edge_type_embedding_dim,
                        edge_condition_hidden_dim=self.edge_condition_hidden_dim,
                        edge_condition_num_types=self.edge_condition_num_types,
                        edge_logit_bias_enable=self.edge_logit_bias_enable,
                        edge_value_gate_enable=self.edge_value_gate_enable,
                        edge_logit_bias_per_head=self.edge_logit_bias_per_head,
                        edge_value_gate_per_head=self.edge_value_gate_per_head,
                        edge_value_gate_per_channel=self.edge_value_gate_per_channel,
                        edge_gate_init_identity=self.edge_gate_init_identity,
                        edge_logit_bias_init_zero=self.edge_logit_bias_init_zero,
                        edge_condition_dropout=self.edge_condition_dropout,
                        edge_condition_debug=self.edge_condition_debug,
                        edge_node_condition_enable=self.edge_node_condition_enable,
                        edge_node_condition_detach=self.edge_node_condition_detach,
                        edge_node_condition_dim=self.edge_node_condition_dim,
                        edge_node_condition_mode=self.edge_node_condition_mode,
                        edge_node_condition_zero_init=self.edge_node_condition_zero_init,
                        edge_gate_scale=self.edge_gate_scale,
                        attn_sparse_qk_mult=self.attn_sparse_qk_mult,
                        attn_sparse_v_mult=self.attn_sparse_v_mult,
                    ))
            else:
                logger.warning("Unified refinement selected with share_transformers=False but num_refinement_layers=0. No dedicated refinement layers created.")


        # Layer / RMS normalization
        self.layer_norm = make_norm(hidden_dim, norm_type=self.norm_type, eps=self.norm_eps)
        self.pinball_refinement_norm = make_norm(self.pinball_work_dim, norm_type=self.norm_type, eps=self.norm_eps)
        self.pinball_upper_refiner = None
        self.pinball_top_refiner = None
        self.pinball_upper_cross_refiner = None
        self.pinball_cross_query_refiners = nn.ModuleList()
        if self.pinball_multirate_enable:
            if self.pinball_upper_cross_attn_steps > 0:
                self.pinball_upper_cross_refiner = PinballPackedCrossAttentionRefiner(
                    work_dim=self.pinball_work_dim,
                    num_heads=self.pinball_work_num_heads,
                    steps=self.pinball_upper_cross_attn_steps,
                    dropout=dropout,
                    norm_type=self.norm_type,
                    norm_eps=self.norm_eps,
                    attn_backend=self.pinball_upper_cross_attn_backend,
                    topk_l2=self.pinball_upper_query_topk_l2,
                    causal=self.pinball_upper_cross_attn_causal,
                    shared_weights=self.pinball_upper_cross_attn_shared_weights,
                    update_l2_enable=self.pinball_upper_update_l2_enable,
                    update_l2_scale_init=self.pinball_upper_update_l2_scale_init,
                    query_level=3,
                    memory_level=2,
                    memory_window=0,
                    selection_mode="global_mean",
                    write_scale_init=self.pinball_upper_cross_write_scale_init,
                    flash_dtype_cast=self.local_attn_flash_dtype_cast,
                    ffn_dim=(self.pinball_cross_query_ffn_dims[3]
                             if len(self.pinball_cross_query_ffn_dims) > 3 else 0),
                )
            if self.pinball_cross_query_steps > 0 and self.pinball_cross_query_pairs:
                for query_level, memory_level in self.pinball_cross_query_pairs:
                    self.pinball_cross_query_refiners.append(
                        PinballPackedCrossAttentionRefiner(
                            work_dim=self.pinball_work_dim,
                            num_heads=self.pinball_work_num_heads,
                            steps=self.pinball_cross_query_steps,
                            dropout=dropout,
                            norm_type=self.norm_type,
                            norm_eps=self.norm_eps,
                            attn_backend=self.pinball_cross_query_backend,
                            topk_l2=self.pinball_cross_query_topk,
                            causal=self.pinball_cross_query_causal,
                            shared_weights=self.pinball_cross_query_shared_weights,
                            update_l2_enable=self.pinball_cross_query_update_memory_enable,
                            update_l2_scale_init=self.pinball_upper_update_l2_scale_init,
                            query_level=int(query_level),
                            memory_level=int(memory_level),
                            memory_window=self.pinball_cross_query_l0_window if int(memory_level) == 0 else 0,
                            memory_recency_window=self.pinball_cross_query_memory_window,
                            selection_mode=self.pinball_cross_query_selection,
                            write_scale_init=self.pinball_cross_query_write_scale_init,
                            flash_dtype_cast=self.local_attn_flash_dtype_cast,
                            ffn_dim=(self.pinball_cross_query_ffn_dims[int(query_level)]
                                     if int(query_level) < len(self.pinball_cross_query_ffn_dims) else 0),
                        )
                    )
            if self.pinball_upper_refine_steps > 0:
                self.pinball_upper_refiner = PinballPackedLevelRefiner(
                    work_dim=self.pinball_work_dim,
                    num_heads=self.pinball_work_num_heads,
                    levels=[2, 3],
                    max_steps=self.pinball_upper_refine_steps,
                    dropout=dropout,
                    norm_type=self.norm_type,
                    norm_eps=self.norm_eps,
                    attn_backend=self.pinball_multirate_attn_backend,
                    window=self.pinball_upper_refine_window,
                    causal=self.pinball_upper_refine_causal,
                    shared_weights=self.pinball_upper_refine_shared_weights,
                    workspace_tokens=0,
                    flash_dtype_cast=self.local_attn_flash_dtype_cast,
                )
            if self.pinball_top_refine_steps > 0:
                self.pinball_top_refiner = PinballPackedLevelRefiner(
                    work_dim=self.pinball_work_dim,
                    num_heads=self.pinball_work_num_heads,
                    levels=[3],
                    max_steps=self.pinball_top_refine_steps,
                    dropout=dropout,
                    norm_type=self.norm_type,
                    norm_eps=self.norm_eps,
                    attn_backend=self.pinball_multirate_attn_backend,
                    window=self.pinball_top_refine_window,
                    causal=self.pinball_top_refine_causal,
                    shared_weights=self.pinball_top_refine_shared_weights,
                    workspace_tokens=self.pinball_l3_workspace_tokens,
                    flash_dtype_cast=self.local_attn_flash_dtype_cast,
                )
            logger.info(
                "Pinball multirate: enable=%s upper_steps=%d top_steps=%d workspace=%d backend=%s upper_window=%d top_window=%d",
                bool(self.pinball_multirate_enable),
                int(self.pinball_upper_refine_steps),
                int(self.pinball_top_refine_steps),
                int(self.pinball_l3_workspace_tokens),
                str(self.pinball_multirate_attn_backend),
                int(self.pinball_upper_refine_window),
                int(self.pinball_top_refine_window),
            )
            if self.pinball_upper_cross_refiner is not None:
                logger.info(
                    "Pinball multirate cross-attn: steps=%d topk_l2=%d backend=%s causal=%s shared=%s update_l2=%s write_scale_init=%.3g",
                    int(self.pinball_upper_cross_attn_steps),
                    int(self.pinball_upper_query_topk_l2),
                    str(self.pinball_upper_cross_attn_backend),
                    bool(self.pinball_upper_cross_attn_causal),
                    bool(self.pinball_upper_cross_attn_shared_weights),
                    bool(self.pinball_upper_update_l2_enable),
                    float(self.pinball_upper_cross_write_scale_init),
                )
            if len(self.pinball_cross_query_refiners) > 0:
                logger.info(
                    "Pinball generic cross-query: pairs=%s steps=%d topk=%d l0_window=%d backend=%s causal=%s update_memory=%s selection=%s write_scale_init=%.3g",
                    list(self.pinball_cross_query_pairs),
                    int(self.pinball_cross_query_steps),
                    int(self.pinball_cross_query_topk),
                    int(self.pinball_cross_query_l0_window),
                    str(self.pinball_cross_query_backend),
                    bool(self.pinball_cross_query_causal),
                    bool(self.pinball_cross_query_update_memory_enable),
                    str(self.pinball_cross_query_selection),
                    float(self.pinball_cross_query_write_scale_init),
                )
        logger.info("Normalization: type=%s eps=%.1e", self.norm_type, self.norm_eps)

        # Initialize a single scalar parameter (alpha) at 0.1 or 0.0, for blending residual connections
        self.alpha = nn.Parameter(torch.tensor(0.999))
        
        # Level projections (for transforming features when creating higher levels)
        self.level_projections = nn.ModuleList()
        for _ in range(len(compression_ratios)):
            self.level_projections.append(nn.Linear(hidden_dim, hidden_dim))

        # Pooled coarse seeding: instead of mask-initializing L1+ nodes, seed each coarse node
        # with the (level-projected) mean of its child window, blended by a learnable per-level
        # gate: seed = base + gate * (pooled - base). gate=0 reproduces mask init exactly.
        # Causal: a coarse node's ar_time is its child window's END, so pooling the full window
        # reveals nothing a reader could not already see through the causal upward edges.
        self.witness_reuse = str(witness_reuse).lower()
        if self.witness_reuse not in ("layer", "step"):
            raise ValueError("witness_reuse must be 'layer' or 'step'")
        self.upper_init = str(upper_init).lower()
        if self.upper_init not in ("mask", "zeros", "pooled"):
            raise ValueError("upper_init must be 'mask', 'zeros' or 'pooled'")
        if self.upper_init == "pooled":
            self.upper_seed_gates = nn.Parameter(
                torch.full((len(compression_ratios),), float(upper_seed_gate_init))
            )
        self._pooled_seed_idx_cache: Dict[tuple, list] = {}
        
        # Output projection for token prediction
        #self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # For use_level_prediction
        self.highest_to_token_projection = nn.Linear(self.hidden_dim, self.hidden_dim) 

        # --- Modify Output Projection ---
        # Create WITHOUT bias initially if tying
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.return_token_features: bool = False

        # optional weight tying
        if tie_weights:# and input_mode == "tokens":
            self.output_projection.weight = self.token_embedding.weight  # standard tying
            logger.info("Applied weight tying between token embedding and output projection.")
        elif tie_weights:
            logger.warning("Weight tying ignored because input comes from features.")

        # --- Coarse co-prediction into L0 pre-head features (see _copredict_inject) ---
        self.hier_copredict_l0 = bool(hier_copredict_l0)
        num_levels = len(compression_ratios) + 1
        _copred_levels = [1, 2, 3] if hier_copredict_levels is None else list(hier_copredict_levels)
        self.hier_copredict_levels = [int(l) for l in _copred_levels if 1 <= int(l) <= num_levels - 1]
        if self.hier_copredict_l0 and self.hier_copredict_levels:
            self.copredict_proj = nn.ModuleDict({
                str(l): nn.Linear(self.hidden_dim, self.hidden_dim) for l in self.hier_copredict_levels
            })
            self.copredict_gate = nn.ParameterDict({
                str(l): nn.Parameter(torch.tensor(float(hier_copredict_gate_init)))
                for l in self.hier_copredict_levels
            })
            logger.info("Coarse co-prediction enabled for levels %s (gate init %.3g).",
                        self.hier_copredict_levels, float(hier_copredict_gate_init))

        # --- Per-layer upward refresh (see _apply_upward_refresh) ---
        self.hier_upward_refresh = bool(hier_upward_refresh)
        self.hier_upward_refresh_every = max(1, int(hier_upward_refresh_every))
        if self.hier_upward_refresh:
            self.upward_refresh_proj = nn.ModuleList(
                nn.Linear(self.hidden_dim, self.hidden_dim) for _ in compression_ratios
            )
            self.upward_refresh_gates = nn.Parameter(
                torch.full((len(compression_ratios),), float(hier_upward_refresh_gate_init))
            )
            logger.info("Per-layer upward refresh enabled (every %d layer(s), gate init %.3g).",
                        self.hier_upward_refresh_every, float(hier_upward_refresh_gate_init))

        # --- Per-layer downward refresh (see _apply_downward_refresh) ---
        self.hier_downward_refresh = bool(hier_downward_refresh)
        self.hier_downward_refresh_every = max(1, int(hier_downward_refresh_every))
        self.drop_static_cross_level_edges = bool(drop_static_cross_level_edges)
        if self.hier_downward_refresh:
            if hier_downward_refresh_pairs is None:
                pair_keys = [f"{q}:{m}" for q in range(num_levels - 1) for m in range(q + 1, num_levels)]
            else:
                pair_keys = []
                for p in hier_downward_refresh_pairs:
                    q, m = (int(v) for v in str(p).split(":"))
                    if 0 <= q < m <= num_levels - 1:
                        pair_keys.append(f"{q}:{m}")
            self.downward_refresh_proj = nn.ModuleDict(
                {k: nn.Linear(self.hidden_dim, self.hidden_dim) for k in pair_keys}
            )
            self.downward_refresh_gates = nn.ParameterDict(
                {k: nn.Parameter(torch.tensor(float(hier_downward_refresh_gate_init))) for k in pair_keys}
            )
            logger.info("Per-layer downward refresh enabled for pairs %s (every %d layer(s), gate init %.3g).",
                        pair_keys, self.hier_downward_refresh_every, float(hier_downward_refresh_gate_init))
        self.hier_refresh_compile = bool(hier_refresh_compile)
        self.hier_layer_compile = bool(hier_layer_compile)

        # --- HQD v2: cross-query-guided nomination (see _cross_query_nominate) ---
        self.xq_nominate_enable = bool(xq_nominate_enable)
        self.xq_nominate_topk_l3 = max(1, int(xq_nominate_topk_l3))
        self.xq_nominate_topk_l1 = max(1, int(xq_nominate_topk_l1))
        self.xq_nominate_exclude_local = bool(xq_nominate_exclude_local)
        self.xq_nominate_max_layers = max(0, int(xq_nominate_max_layers))
        self.xq_nominate_after_layer = xq_nominate_after_layer
        self.xq_nominate_every = max(0, int(xq_nominate_every))
        self.xq_nominate_topk_l0 = max(0, int(xq_nominate_topk_l0))
        self.xq_nominate_packed_read = bool(xq_nominate_packed_read)
        self.xq_nominate_gate_enable = bool(xq_nominate_gate_enable)
        self.xq_nominate_stage2 = str(xq_nominate_stage2).lower()
        if self.xq_nominate_packed_read and self.xq_nominate_topk_l0 <= 0:
            raise ValueError("xq_nominate_packed_read requires xq_nominate_topk_l0 > 0 (fixed K)")
        if self.xq_nominate_gate_enable:
            # gate = sigmoid(w * s/sqrt(d) + b), b=2.2 => near-open start (~0.9 at s=0) that
            # learns per-query selectivity. w MUST start nonzero: the selector projections'
            # gradient is dL/dgate * w, so w=0 would leave the NSA-style selector-training
            # path dead until w drifts off zero.
            self.xq_gate_w = nn.Parameter(torch.full((), 0.5))
            self.xq_gate_b = nn.Parameter(torch.full((), 2.2))
        self._xq_nom_edges: Optional[tuple] = None
        self._xq_query_gate_full: Optional[torch.Tensor] = None  # [B,N,1], per nomination round
        self._xq_children_cache: Optional[tuple] = None

        # --- Packed cross-level local attention (attrs stored pre-construction above; see
        # _local_pack_build_spec) ---
        self._local_pack_spec_cache: Optional[tuple] = None
        self._last_local_pack_spec_stats: Optional[Dict[str, int]] = None
        if self.local_pack_cross_level:
            logger.info(
                "Packed cross-level local attention enabled (window=%s slots, query_levels=%s, "
                "level_bias=%s).",
                self.local_pack_window if self.local_pack_window > 0 else "l0",
                self.local_pack_query_levels,
                self.local_pack_level_bias,
            )
        if self.xq_nominate_enable:
            logger.info(
                "Cross-query nomination enabled (topk L3=%d, L1=%d, L0=%d, exclude_local=%s, "
                "packed=%s, gate=%s, stage2=%s).",
                self.xq_nominate_topk_l3, self.xq_nominate_topk_l1, self.xq_nominate_topk_l0,
                self.xq_nominate_exclude_local, self.xq_nominate_packed_read,
                self.xq_nominate_gate_enable, self.xq_nominate_stage2)

        # --- Predictive coarse aux (see _compute_predictive_aux_loss) ---
        self.hier_predaux_enable = bool(hier_predaux_enable)
        self.lambda_hier_predaux = float(lambda_hier_predaux)
        _pa_levels = [1, 2, 3] if hier_predaux_levels is None else list(hier_predaux_levels)
        self.hier_predaux_levels = [int(l) for l in _pa_levels if 1 <= int(l) <= num_levels - 1]
        self.hier_predaux_horizon = int(hier_predaux_horizon)
        self.hier_predaux_loss_mode = str(hier_predaux_loss_mode).lower()
        self._last_hier_predaux_loss: Optional[torch.Tensor] = None
        if self.hier_predaux_enable and self.hier_predaux_levels:
            def _predaux_head() -> nn.Module:
                if str(hier_predaux_predictor).lower() == "linear":
                    return nn.Linear(self.hidden_dim, self.hidden_dim)
                return nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            self.predaux_predictor = nn.ModuleDict(
                {str(l): _predaux_head() for l in self.hier_predaux_levels}
            )
            self.hier_predaux_directions = str(hier_predaux_directions).lower()
            if self.hier_predaux_directions not in ("next", "both"):
                raise ValueError("hier_predaux_directions must be 'next' or 'both'")
            if self.hier_predaux_directions == "both":
                self.predaux_predictor_prev = nn.ModuleDict(
                    {str(l): _predaux_head() for l in self.hier_predaux_levels}
                )
            logger.info("Predictive coarse aux enabled for levels %s (lambda %.3g, loss %s, dir %s).",
                        self.hier_predaux_levels, self.lambda_hier_predaux,
                        self.hier_predaux_loss_mode, self.hier_predaux_directions)

        # Debug gate/scale monitor (see gate_monitor()).
        self.pinball_monitor_gates = bool(pinball_monitor_gates)
        self.pinball_monitor_gates_every = max(1, int(pinball_monitor_gates_every))
        self._gate_monitor_calls = 0


        # --- Edge Feature Generator (Instantiation with correct size) ---
        self.edge_feature_generator = None
        if self.use_edge_attr: # Check the flag passed during init
            try:
                # Make sure the import path is correct for your project structure
                from .hierarchy.unified_hierarchy_builder import EdgeFeatureGenerator

                # Calculate needed size for edge types
                max_level_idx = self.num_hier_levels - 1
                # Within-level types go up to max_level_idx
                # Cross-level types go up to 5 + 2 * (max_level_idx - 1)
                max_within_level_type = max_level_idx
                max_cross_level_type = 0
                if self.num_hier_levels > 1:
                     max_cross_level_type = 5 + 2 * (max_level_idx - 1)

                # Embedding size needs to be max_type + 1 (due to 0-indexing)
                num_edge_types_needed = max(max_within_level_type, max_cross_level_type) + 1

                logger.info(f"Instantiating EdgeFeatureGenerator with num_edge_types = {num_edge_types_needed}")
                self.edge_feature_generator = EdgeFeatureGenerator(
                    hidden_dim=self.hidden_dim,
                    num_edge_types=num_edge_types_needed
                )# .to(self.token_embedding.weight.device) # Move later or rely on model.to(device)

            except ImportError:
                logger.warning("EdgeFeatureGenerator class not found. Cannot generate edge attributes.")
                self.use_edge_attr = False # Disable flag if generator not found
            except Exception as e:
                 logger.error(f"Error instantiating EdgeFeatureGenerator: {e}")
                 self.edge_feature_generator = None
                 self.use_edge_attr = False # Disable flag on error
        # --- End Edge Feature Generator ---

        # OLD
        # # --- Edge Feature Generator (Optional - Instantiate if needed) ---
        # self.edge_feature_generator = None
        # if self.use_edge_attr: # If you plan to generate edge features
        #     try:
        #         from .hierarchy.unified_hierarchy_builder import EdgeFeatureGenerator
        #         self.edge_feature_generator = EdgeFeatureGenerator(self.hidden_dim).to(self.token_embedding.weight.device) # Ensure on correct device
        #     except ImportError:
        #         logger.warning("EdgeFeatureGenerator not found. Cannot generate edge attributes.")
        # # --- End Edge Feature Generator ---

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.cond_film.weight, std=1e-3)
        nn.init.zeros_(self.cond_film.bias)
        if self.refine_cond_film is not None:
            nn.init.zeros_(self.refine_cond_film.weight)
            nn.init.zeros_(self.refine_cond_film.bias)
        if self.refine_cond_concat_proj is not None:
            nn.init.zeros_(self.refine_cond_concat_proj.weight)
            nn.init.zeros_(self.refine_cond_concat_proj.bias)
        if self.refine_cond_concat_gate is not None:
            with torch.no_grad():
                self.refine_cond_concat_gate.fill_(float(self.refine_cond_concat_gate_init))
        self._init_zipper_beta_gates()
        self.level_mappings = None
        logger.info(
            "Initialized Hierarchical Flow GAT. Refinement Style: '%s'. LapPE k=%d geometry=%s grid=(%d,%d) spatial_metric=%s rope_mode=%s class_cond=%s num_classes=%d",
            str(self.refinement_style),
            int(self.lap_pe_k if self.lap_pe_transform else 0),
            str(self.graph_geometry_mode),
            int(self.graph_grid_height),
            int(self.graph_grid_width),
            str(self.graph_spatial_metric),
            str(self.rope_mode),
            bool(self.class_embedding is not None),
            int(self.num_classes),
        )
        logger.info(
            "Refinement conditioning: mode=%s strength=%.3f concat_gate_init=%.3f",
            str(self.refine_cond_mode),
            float(self.refine_cond_strength),
            float(self.refine_cond_concat_gate_init),
        )
        
        # Store level mappings for each example
        self.level_mappings = None
        
        logger.info(f"Initialized Hierarchical Flow GAT with {sum(p.numel() for p in self.parameters())} parameters")

    def reset_lateral_edge_traces(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            reset_fn = getattr(module, "reset_lateral_edge_traces", None)
            if callable(reset_fn):
                reset_fn()

    def _zipper_refinement_transformers(self) -> List[nn.Module]:
        if self.share_transformers:
            return [m for level_mods in self.level_transformers for m in level_mods]
        if self.refinement_transformers is not None:
            return list(self.refinement_transformers)
        return []

    def _init_zipper_beta_gates(self) -> None:
        if (not bool(getattr(self, "zip_use_beta_gate", True))) or bool(getattr(self, "zip_paramfree_gate", False)):
            return
        transformers = self._zipper_refinement_transformers()
        if not transformers:
            return
        for tr in transformers:
            gate = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, float(self.zip_beta_init_bias))
            tr.zip_lin_beta_attn = gate

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
            if getattr(module, "weight", None) is not None:
                module.weight.data.fill_(1.0)

    def _resolve_l0_grid_shape_for_tokens(self, token_len: int) -> Optional[Tuple[int, int]]:
        if str(getattr(self, "graph_geometry_mode", "sequence")).lower() != "grid2d":
            return None
        token_len = int(token_len)
        if token_len <= 0:
            return None

        runtime = getattr(self, "_runtime_l0_grid_shape", None)
        if runtime is not None:
            gh = int(runtime[0])
            gw = int(runtime[1])
            if gh > 0 and gw > 0 and gh * gw == token_len:
                return (gh, gw)

        gh = int(getattr(self, "graph_grid_height", 0))
        gw = int(getattr(self, "graph_grid_width", 0))
        if gh > 0 and gw > 0 and gh * gw == token_len:
            return (gh, gw)

        side = int(round(math.sqrt(max(1, token_len))))
        if side * side == token_len:
            return (side, side)
        return None

    def _resolve_spatial_curve(self, seq_len_graph: int, device) -> Optional[tuple]:
        """(perm, inv_perm, coords) on `device` for curve mode, or None (raster passthrough).

        Deliberately separate from _resolve_l0_grid_shape_for_tokens: that helper is the
        grid2d-machinery gate; curve mode runs with graph_geometry_mode == "sequence".
        Dims priority: spatial_dims (prod must equal seq_len_graph) -> (graph_grid_height,
        graph_grid_width) -> exact square sqrt -> warn-once + None.
        """
        curve = str(getattr(self, "spatial_curve", "none")).lower()
        if curve in ("", "none"):
            return None
        T = int(seq_len_graph)
        if T <= 1:
            return None
        dims = None
        sd = getattr(self, "spatial_dims", None)
        if sd:
            prod = 1
            for d in sd:
                prod *= int(d)
            if prod == T:
                dims = tuple(int(d) for d in sd)
        if dims is None:
            gh = int(getattr(self, "graph_grid_height", 0))
            gw = int(getattr(self, "graph_grid_width", 0))
            if gh > 0 and gw > 0 and gh * gw == T:
                dims = (gh, gw)
        if dims is None:
            side = int(round(math.sqrt(T)))
            if side * side == T:
                dims = (side, side)
        if dims is None:
            if not getattr(self, "_spatial_curve_warned", False):
                logger.warning(
                    "spatial_curve='%s': cannot resolve grid dims for seq_len %d "
                    "(spatial_dims=%s, grid %dx%d); raster passthrough.",
                    curve, T, sd,
                    int(getattr(self, "graph_grid_height", 0)),
                    int(getattr(self, "graph_grid_width", 0)),
                )
                self._spatial_curve_warned = True
            return None
        key = (dims, curve, str(device))
        hit = self._spatial_curve_dev_cache.get(key)
        if hit is None:
            from .layers.spatial_curve import build_curve
            perm, inv_perm, coords = build_curve(dims, curve)
            hit = (perm.to(device), inv_perm.to(device), coords.to(device))
            if len(self._spatial_curve_dev_cache) > 16:
                self._spatial_curve_dev_cache.clear()
            self._spatial_curve_dev_cache[key] = hit
        return hit

    def _get_embeddings(self, input_ids, position_ids=None, max_seq_len=None):
        """
        Get token embeddings with rotary positional encoding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Optional position IDs [batch_size, seq_len]
            
        Returns:
            embeddings: Token embeddings with positional information
        """
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)

        if (
            self.token_unet is not None
            and token_embeds.dim() == 3
            and str(getattr(self, "token_unet_mode", "stem")).lower() == "stem"
        ):
            if bool(getattr(self, "token_unet_is_2d", False)):
                grid_shape = self._resolve_l0_grid_shape_for_tokens(int(token_embeds.size(1)))
                token_embeds = self.token_unet(token_embeds, grid_shape=grid_shape)
            else:
                token_embeds = self.token_unet(token_embeds)
        
        # Apply rotary positional encoding
        #embeddings = self.rotary_pos_enc(token_embeds)
        
        return token_embeds

    def _build_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.to(dtype=torch.float32)
        half = dim // 2
        if half <= 0:
            return timesteps.unsqueeze(-1)
        freqs = torch.exp(
            -math.log(float(max_period))
            * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _compute_conditioning_vector(
        self,
        batch_size: int,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        cond_vec: Optional[torch.Tensor] = None

        if self.class_embedding is not None:
            if class_labels is None:
                cls = torch.full(
                    (int(batch_size),),
                    int(self.class_null_index),
                    dtype=torch.long,
                    device=device,
                )
            else:
                cls = class_labels.to(device=device, dtype=torch.long).view(-1)
                if int(cls.numel()) != int(batch_size):
                    if int(cls.numel()) == 1:
                        cls = cls.expand(int(batch_size))
                    else:
                        cls = cls[:int(batch_size)]
                cls = cls.clamp(min=0, max=max(0, int(self.num_classes)))
                if self.training and self.class_cond_drop_prob > 0.0:
                    drop_mask = torch.rand(int(batch_size), device=device) < float(self.class_cond_drop_prob)
                    if bool(drop_mask.any()):
                        cls = cls.clone()
                        cls[drop_mask] = int(self.class_null_index)
            cond_vec = self.class_embedding(cls)

        if timesteps is not None:
            t = timesteps.to(device=device)
            if t.dim() == 0:
                t = t.expand(int(batch_size))
            if t.dim() > 1:
                t = t.view(-1)
            if int(t.numel()) != int(batch_size):
                if int(t.numel()) == 1:
                    t = t.expand(int(batch_size))
                else:
                    t = t[:int(batch_size)]
            t_emb = self._build_timestep_embedding(t, int(self.diffusion_timestep_embed_dim))
            t_emb = self.time_embed_mlp(t_emb.to(dtype=self.cond_film.weight.dtype))
            cond_vec = t_emb if cond_vec is None else (cond_vec + t_emb)

        return cond_vec

    def _apply_film_conditioning(
        self,
        token_embeddings_bt: torch.Tensor,
        cond_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cond_vec is None:
            return token_embeddings_bt
        if token_embeddings_bt.dim() != 3:
            return token_embeddings_bt
        gb = self.cond_film(cond_vec)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.sigmoid(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return token_embeddings_bt * (1.0 + gamma) + beta

    def _prepare_refine_cond_vector(
        self,
        cond_vec: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if cond_vec is None:
            return None
        if not torch.is_tensor(cond_vec):
            cond = torch.as_tensor(cond_vec, device=device)
        else:
            cond = cond_vec.to(device=device)

        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        elif cond.dim() > 2:
            cond = cond.view(cond.size(0), -1)

        if cond.size(-1) != int(self.hidden_dim):
            if cond.size(-1) > int(self.hidden_dim):
                cond = cond[..., : int(self.hidden_dim)]
            else:
                pad = torch.zeros(
                    (int(cond.size(0)), int(self.hidden_dim - cond.size(-1))),
                    device=device,
                    dtype=cond.dtype,
                )
                cond = torch.cat([cond, pad], dim=-1)

        if int(cond.size(0)) != int(batch_size):
            if int(cond.size(0)) == 1:
                cond = cond.expand(int(batch_size), -1)
            elif int(cond.size(0)) > int(batch_size):
                cond = cond[: int(batch_size)]
            elif int(cond.size(0)) > 0:
                pad_rows = int(batch_size - cond.size(0))
                cond = torch.cat([cond, cond[-1:].expand(pad_rows, -1)], dim=0)
            else:
                return None

        return cond

    def _apply_refinement_conditioning_bnh(
        self,
        x_bnh: torch.Tensor,
        cond_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mode = str(getattr(self, "refine_cond_mode", "none")).lower()
        if mode == "none" or cond_vec is None or x_bnh.dim() != 3:
            return x_bnh

        bsz, _, hdim = x_bnh.shape
        if int(hdim) != int(self.hidden_dim):
            return x_bnh

        cond = self._prepare_refine_cond_vector(
            cond_vec=cond_vec,
            batch_size=int(bsz),
            device=x_bnh.device,
        )
        if cond is None:
            return x_bnh

        x_out = x_bnh
        strength = float(getattr(self, "refine_cond_strength", 1.0))

        if self.refine_cond_film is not None:
            film_dtype = self.refine_cond_film.weight.dtype
            gb = self.refine_cond_film(cond.to(dtype=film_dtype))
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = torch.tanh(gamma).unsqueeze(1).to(dtype=x_out.dtype)
            beta = beta.unsqueeze(1).to(dtype=x_out.dtype)
            x_out = x_out * (1.0 + strength * gamma) + strength * beta

        if mode == "film_concat" and self.refine_cond_concat_proj is not None:
            cond_exp = cond.to(dtype=x_out.dtype).unsqueeze(1).expand(int(bsz), int(x_out.size(1)), int(self.hidden_dim))
            concat_in = torch.cat([x_out, cond_exp], dim=-1)
            proj_dtype = self.refine_cond_concat_proj.weight.dtype
            concat_delta = self.refine_cond_concat_proj(concat_in.to(dtype=proj_dtype)).to(dtype=x_out.dtype)
            if self.refine_cond_concat_gate is not None:
                gate = torch.sigmoid(self.refine_cond_concat_gate).to(dtype=x_out.dtype)
            else:
                gate = torch.tensor(1.0, device=x_out.device, dtype=x_out.dtype)
            x_out = x_out + strength * gate * concat_delta

        if not bool(getattr(self, "_refine_cond_runtime_logged", False)):
            concat_gate = None
            if self.refine_cond_concat_gate is not None:
                concat_gate = float(torch.sigmoid(self.refine_cond_concat_gate.detach()).item())
            logger.info(
                "Refinement conditioning active: mode=%s strength=%.3f concat_gate=%s",
                str(mode),
                float(strength),
                "n/a" if concat_gate is None else f"{concat_gate:.4f}",
            )
            self._refine_cond_runtime_logged = True

        return x_out

    def _apply_refinement_conditioning_flat(
        self,
        x_flat: torch.Tensor,
        batch_size: int,
        nodes_per_sample: int,
        cond_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mode = str(getattr(self, "refine_cond_mode", "none")).lower()
        if mode == "none" or cond_vec is None or x_flat.dim() != 2:
            return x_flat
        if int(batch_size) <= 0 or int(nodes_per_sample) <= 0:
            return x_flat
        expected = int(batch_size) * int(nodes_per_sample)
        if int(x_flat.size(0)) != expected:
            return x_flat
        x_bnh = x_flat.reshape(int(batch_size), int(nodes_per_sample), int(x_flat.size(-1)))
        x_bnh = self._apply_refinement_conditioning_bnh(x_bnh, cond_vec)
        return x_bnh.reshape(expected, int(x_flat.size(-1)))

    def _token_unet_encode_for_graph(
        self,
        token_embeds_bt: torch.Tensor,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if token_embeds_bt.dim() != 3:
            return token_embeds_bt, None
        if self.token_unet is None:
            return token_embeds_bt, None
        if str(getattr(self, "token_unet_mode", "stem")).lower() != "coarse_tokenize":
            return token_embeds_bt, None
        if bool(getattr(self, "token_unet_is_2d", False)):
            use_grid = grid_shape
            if use_grid is None:
                use_grid = self._resolve_l0_grid_shape_for_tokens(int(token_embeds_bt.size(1)))
            coarse_tokens, decode_context = self.token_unet.encode(token_embeds_bt, grid_shape=use_grid)
            if decode_context is not None and use_grid is not None:
                decode_context["dense_grid_shape"] = (int(use_grid[0]), int(use_grid[1]))
        else:
            coarse_tokens, decode_context = self.token_unet.encode(token_embeds_bt)
        return coarse_tokens, decode_context

    def _token_unet_decode_from_graph(
        self,
        graph_token_features_bt: torch.Tensor,
        decode_context: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if graph_token_features_bt.dim() != 3:
            return graph_token_features_bt, None
        if self.token_unet is None or decode_context is None:
            return graph_token_features_bt, None
        strict, lookahead = self.token_unet.decode_dual(graph_token_features_bt, decode_context)
        return strict, lookahead

    def _rgb_token_unet_conditioning_vector(
        self,
        batch_size: int,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if cond_vec is not None:
            return self._prepare_refine_cond_vector(
                cond_vec=cond_vec,
                batch_size=int(batch_size),
                device=device,
            )
        return self._compute_conditioning_vector(
            batch_size=int(batch_size),
            device=device,
            class_labels=class_labels,
            timesteps=timesteps,
        )

    def encode_rgb_to_tokens(
        self,
        pixel_values: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.rgb_token_unet is None:
            raise RuntimeError("RGB token U-Net bridge is not enabled")
        bridge_cond = self._rgb_token_unet_conditioning_vector(
            batch_size=int(pixel_values.size(0)),
            device=pixel_values.device,
            class_labels=class_labels,
            timesteps=timesteps,
            cond_vec=cond_vec,
        )
        tokens, context = self.rgb_token_unet.encode(pixel_values, cond=bridge_cond)
        if isinstance(context, dict):
            gg = context.get("graph_grid_shape", None)
            if gg is not None and len(gg) == 2:
                gh = int(gg[0])
                gw = int(gg[1])
                if gh > 0 and gw > 0:
                    self._runtime_l0_grid_shape = (gh, gw)
        return tokens, context

    def decode_tokens_to_rgb(
        self,
        token_features: torch.Tensor,
        context: Dict[str, Any],
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.rgb_token_unet is None:
            raise RuntimeError("RGB token U-Net bridge is not enabled")
        bridge_cond = self._rgb_token_unet_conditioning_vector(
            batch_size=int(token_features.size(0)),
            device=token_features.device,
            class_labels=class_labels,
            timesteps=timesteps,
            cond_vec=cond_vec,
        )
        return self.rgb_token_unet.decode(token_features, context, cond=bridge_cond)

    def _map_reveal_targets_to_coarse(
        self,
        reveal_target_ids: Optional[torch.Tensor],
        reveal_mask: Optional[torch.Tensor],
        coarse_len: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if reveal_target_ids is None or reveal_target_ids.dim() != 2:
            return reveal_target_ids, reveal_mask, None

        coarse_len = int(coarse_len)
        if coarse_len <= 0:
            return None, None, None

        full_len = int(reveal_target_ids.size(1))
        device = reveal_target_ids.device
        if bool(getattr(self, "token_unet_right_edge_targets", True)):
            idx = (torch.arange(coarse_len, device=device, dtype=torch.long) + 1) * int(max(1, self.token_unet_scale)) - 1
            idx = idx.clamp(min=0, max=max(0, full_len - 1))
        else:
            idx = torch.linspace(
                0,
                max(0, full_len - 1),
                steps=coarse_len,
                device=device,
            ).round().to(dtype=torch.long)

        reveal_ids_coarse = reveal_target_ids.index_select(1, idx)
        reveal_mask_coarse = None
        if reveal_mask is not None and reveal_mask.shape == reveal_target_ids.shape:
            reveal_mask_coarse = reveal_mask.to(dtype=torch.bool).index_select(1, idx)

        return reveal_ids_coarse, reveal_mask_coarse, idx

    def _infer_level_grid_shape(
        self,
        num_nodes: int,
        level_idx: int,
        explicit_grid_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        if str(getattr(self, "graph_geometry_mode", "sequence")).lower() != "grid2d":
            return None
        if explicit_grid_shape is not None:
            gh = int(explicit_grid_shape[0])
            gw = int(explicit_grid_shape[1])
            if gh > 0 and gw > 0 and gh * gw == int(num_nodes):
                return (gh, gw)
            return None

        if int(level_idx) == 0:
            runtime = getattr(self, "_runtime_l0_grid_shape", None)
            if runtime is not None:
                gh = int(runtime[0])
                gw = int(runtime[1])
                if gh > 0 and gw > 0 and gh * gw == int(num_nodes):
                    return (gh, gw)

        if int(level_idx) in self._cached_level_grid_shapes:
            gh, gw = self._cached_level_grid_shapes[int(level_idx)]
            if gh * gw == int(num_nodes):
                return (gh, gw)

        if int(level_idx) == 0:
            gh = int(getattr(self, "graph_grid_height", 0))
            gw = int(getattr(self, "graph_grid_width", 0))
            if gh > 0 and gw > 0 and gh * gw == int(num_nodes):
                return (gh, gw)
            side = int(round(math.sqrt(max(1, int(num_nodes)))))
            if side * side == int(num_nodes):
                return (side, side)
        return None

    def _build_grid2d_coords(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        yy = torch.arange(int(h), device=device, dtype=torch.long).unsqueeze(1).expand(int(h), int(w))
        xx = torch.arange(int(w), device=device, dtype=torch.long).unsqueeze(0).expand(int(h), int(w))
        return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)

    def _build_grid2d_window_edges(
        self,
        h: int,
        w: int,
        radius: int,
        causal_l0: bool,
        device: torch.device,
    ) -> List[List[int]]:
        h = int(h)
        w = int(w)
        r = max(0, int(radius))
        if h <= 0 or w <= 0 or r <= 0:
            return []

        coords = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        if str(getattr(self, "graph_spatial_metric", "chebyshev")).lower() == "manhattan":
            coords = [(dy, dx) for dy, dx in coords if abs(dy) + abs(dx) <= r]

        edges: List[List[int]] = []
        for y in range(h):
            for x in range(w):
                dst = y * w + x
                for dy, dx in coords:
                    if dy == 0 and dx == 0:
                        continue
                    sy = y + dy
                    sx = x + dx
                    if sy < 0 or sy >= h or sx < 0 or sx >= w:
                        continue
                    src = sy * w + sx
                    if causal_l0 and src > dst:
                        continue
                    edges.append([src if causal_l0 else dst, dst if causal_l0 else src])
        return edges
    

    
    def _build_level_graph(
        self,
        features: torch.Tensor,
        level_idx: int = 0,
        node_coord: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Build graph for a single level with careful device handling.
        Allows for denser local connectivity via local_connectivity_window_size.
        """
        try:
            device = features.device
            num_nodes = features.size(0)
            
            # Truncation logic for L0 (remains the same)
            if level_idx == 0 and num_nodes > self.max_seq_len:
                logger.warning(f"L0: Truncating sequence from {num_nodes} to {self.max_seq_len}") # Use logger
                num_nodes = self.max_seq_len
                features = features[:self.max_seq_len]
                if node_coord is not None and node_coord.numel() > 0:
                    node_coord = node_coord[:self.max_seq_len]
            # features_cpu = features.detach().cpu() # Keep graph construction on device if possible

            # Reset caches if sequence length changes
            runtime_grid = getattr(self, "_runtime_l0_grid_shape", None) if int(level_idx) == 0 else None
            if runtime_grid is not None:
                runtime_gh = int(runtime_grid[0])
                runtime_gw = int(runtime_grid[1])
            else:
                runtime_gh = int(getattr(self, "graph_grid_height", 0))
                runtime_gw = int(getattr(self, "graph_grid_width", 0))
            geom_tag = (
                str(getattr(self, "graph_geometry_mode", "sequence")),
                int(runtime_gh),
                int(runtime_gw),
                str(getattr(self, "graph_spatial_metric", "chebyshev")),
                int(getattr(self, "graph_downsample_factor", 2)),
            )
            if (
                level_idx == 0
                and (
                    self._cached_seq_len != num_nodes
                    or self._cached_l0_ar_enable is None
                    or self._cached_l0_ar_enable != bool(self.l0_ar_enable)
                    or self._cached_geometry_tag != geom_tag
                )
            ):
                self._cached_seq_len = num_nodes
                self._cached_geometry_tag = geom_tag
                self._cached_level_graphs = [None] * self.num_hier_levels
                self._cached_unified_graph = None
                self._cached_unified_graph_key = None
                self._cached_level_grid_shapes = {}

            # Reuse cached level graph if available
            if (
                self._cached_level_graphs is not None and
                level_idx < len(self._cached_level_graphs)
            ):
                cached_graph = self._cached_level_graphs[level_idx]
                if cached_graph is not None and cached_graph.num_nodes == num_nodes:
                    graph_cached = Data(
                        x=features,
                        edge_index=cached_graph.edge_index,
                        node_level=cached_graph.node_level,
                        num_nodes=cached_graph.num_nodes,
                    )
                    if hasattr(cached_graph, "grid_shape"):
                        graph_cached.grid_shape = tuple(cached_graph.grid_shape)
                    if hasattr(cached_graph, "node_coord") and getattr(cached_graph, "node_coord") is not None:
                        graph_cached.node_coord = cached_graph.node_coord
                    return graph_cached

            edges_list = [] # Use a list to collect edge pairs

            if num_nodes <= 0: # Handle empty graph case
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                node_level_tensor = torch.empty((0,), dtype=torch.long, device=device) # Renamed to avoid conflict
                # Add num_nodes to Data object for consistency
                return Data(x=features, edge_index=edge_index, node_level=node_level_tensor, num_nodes=num_nodes)

            grid_shape_eff = self._infer_level_grid_shape(num_nodes, level_idx, explicit_grid_shape=grid_shape)
            node_coord_eff = None
            k_neighbors = 0
            if grid_shape_eff is not None:
                if node_coord is not None and node_coord.numel() == (num_nodes * 2):
                    node_coord_eff = node_coord.to(device=device, dtype=torch.long).view(num_nodes, 2)
                else:
                    node_coord_eff = self._build_grid2d_coords(grid_shape_eff[0], grid_shape_eff[1], device=device)
                self._cached_level_grid_shapes[int(level_idx)] = (int(grid_shape_eff[0]), int(grid_shape_eff[1]))

            # Dense local connectivity (within-level); self-loops are added later.
            if num_nodes > 1:
                if int(self.local_connectivity_window_size) <= 0:
                    k_neighbors = 0
                    warned_levels = getattr(self, "_zero_local_connectivity_warned_levels", set())
                    if level_idx not in warned_levels:
                        logger.warning(
                            "Level %d: local_connectivity_window_size=%d -> disabling non-self within-level edges.",
                            int(level_idx),
                            int(self.local_connectivity_window_size),
                        )
                        warned_levels.add(level_idx)
                        self._zero_local_connectivity_warned_levels = warned_levels
                else:
                    if level_idx == 0:
                        # L0 uses its dedicated graph-window knob when dense graph edges are enabled.
                        k_neighbors = int(self.l0_windowgraph)
                    else:
                        # Upper levels scale neighborhood size with level depth.
                        k_neighbors = int(self.local_connectivity_window_size + (level_idx * 8))
                    k_neighbors = max(0, min(num_nodes - 1, k_neighbors))

                if k_neighbors > 0:
                    if grid_shape_eff is not None:
                        h, w = int(grid_shape_eff[0]), int(grid_shape_eff[1])
                        causal_l0 = bool(self.l0_ar_enable and int(level_idx) == 0)
                        logger.debug(
                            "Level %d: spatial grid graph %dx%d radius=%d metric=%s causal_l0=%s",
                            int(level_idx),
                            h,
                            w,
                            int(k_neighbors),
                            str(getattr(self, "graph_spatial_metric", "chebyshev")),
                            causal_l0,
                        )
                        edges_list.extend(
                            self._build_grid2d_window_edges(
                                h=h,
                                w=w,
                                radius=k_neighbors,
                                causal_l0=causal_l0,
                                device=device,
                            )
                        )
                    else:
                        logger.debug("Level %d: dense local graph k_neighbors=%d", int(level_idx), int(k_neighbors))
                        for i in range(num_nodes):
                            for j in range(max(0, i - k_neighbors), i):
                                if self.l0_ar_enable and level_idx == 0:
                                    # Causal L0: only past -> current
                                    edges_list.append([j, i])
                                else:
                                    edges_list.append([i, j])
                            if not (self.l0_ar_enable and level_idx == 0):
                                for j in range(i + 1, min(num_nodes, i + 1 + k_neighbors)):
                                    edges_list.append([i, j])


            # Add self-loops if specified (common for both branches)
            if self.add_self_loops and num_nodes > 0:
                for i in range(num_nodes):
                    edges_list.append([i, i])

            # Create edge_index tensor and remove duplicates
            if edges_list:
                edge_index_temp = torch.tensor(edges_list, dtype=torch.long, device=device).t()
                # Make undirected and remove duplicates efficiently
                # This ensures edges are (src,trg) and (trg,src) and unique pairs, and handles self-loops correctly.
                # It sorts column-wise (src,trg) pairs and removes duplicates.
                # Then adds reverse edges if not present (to_undirected part).
                if edge_index_temp.numel() > 0:
                    edge_index = torch.unique(edge_index_temp, dim=1) # Basic duplicate removal
                    # For a truly undirected graph with no redundant reverse edges if already added:
                    # Can be complex. Simplest robust way if PyG is available:
                    # from torch_geometric.utils import to_undirected
                    # edge_index = to_undirected(edge_index_temp, num_nodes=num_nodes)
                    # For now, let's assume the loops above correctly generate bidirectionality
                    # and self-loops, and torch.unique is sufficient if any direct duplicates arose.
                    # A more robust manual unique for undirected after adding both ways:
                    # unique_sorted_pairs = set()
                    # final_edges_list = []
                    # for k_idx in range(edge_index_temp.size(1)):
                    #     u, v = edge_index_temp[0, k_idx].item(), edge_index_temp[1, k_idx].item()
                    #     pair = tuple(sorted((u,v)))
                    #     if u == v: # self-loop
                    #          if pair not in unique_sorted_pairs:
                    #              final_edges_list.append([u,v])
                    #              unique_sorted_pairs.add(pair)
                    #     elif pair not in unique_sorted_pairs:
                    #         final_edges_list.append([u,v])
                    #         final_edges_list.append([v,u]) # ensure both directions
                    #         unique_sorted_pairs.add(pair)
                    # if final_edges_list:
                    #     edge_index = torch.tensor(final_edges_list, dtype=torch.long, device=device).t().contiguous()
                    # else:
                    #     edge_index = torch.empty((2,0), dtype=torch.long, device=device)

                else:
                    edge_index = torch.empty((2,0), dtype=torch.long, device=device)

            else: # No edges generated (e.g., single node graph without self-loops flag)
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            node_level_tensor = torch.full((num_nodes,), level_idx, dtype=torch.long, device=device) # Renamed variable

            graph = Data(
                x=features,
                edge_index=edge_index,
                node_level=node_level_tensor,
                num_nodes=num_nodes # Explicitly add num_nodes
            )
            if grid_shape_eff is not None:
                graph.grid_shape = (int(grid_shape_eff[0]), int(grid_shape_eff[1]))
                if node_coord_eff is not None:
                    graph.node_coord = node_coord_eff

            grid_desc = "none"
            coord_source = "none"
            if grid_shape_eff is not None:
                grid_desc = f"{int(grid_shape_eff[0])}x{int(grid_shape_eff[1])}"
                if node_coord is not None and node_coord_eff is not None and int(node_coord.numel()) == int(num_nodes * 2):
                    coord_source = "provided"
                elif node_coord_eff is not None:
                    coord_source = "inferred"
            if self.verbose:
                logger.info(
                    "[HFGAT:LEVEL] built level=%d nodes=%d edges=%d grid=%s coord_source=%s local_k=%d",
                    int(level_idx),
                    int(num_nodes),
                    int(edge_index.size(1)),
                    grid_desc,
                    coord_source,
                    int(k_neighbors),
                )

            if (
                self._cached_level_graphs is not None and
                level_idx < len(self._cached_level_graphs) and
                self._cached_level_graphs[level_idx] is None
            ):
                cached = Data(
                    edge_index=edge_index.clone(),
                    node_level=node_level_tensor.clone(),
                    num_nodes=num_nodes,
                )
                if grid_shape_eff is not None:
                    cached.grid_shape = (int(grid_shape_eff[0]), int(grid_shape_eff[1]))
                    if node_coord_eff is not None:
                        cached.node_coord = node_coord_eff.clone()
                self._cached_level_graphs[level_idx] = cached
            if level_idx == 0:
                self._cached_l0_ar_enable = bool(self.l0_ar_enable)
            return graph

        # Fallback Exception Handling (no change from your code)
        except Exception as e:
            logger.error(f"Error in building level graph for level {level_idx}: {e}", exc_info=True) # Log traceback
            # ... (your existing fallback graph creation logic) ...
            # ...
            # Create minimal fallback graph
            device = features.device
            # Ensure num_nodes_fallback is valid
            num_nodes_fallback = min(features.size(0), self.max_seq_len if level_idx == 0 else features.size(0))
            if num_nodes_fallback <= 0:
                return Data(x=torch.empty((0, features.size(1)), device=device), edge_index=torch.empty((2,0),dtype=torch.long,device=device), node_level=torch.empty((0,),dtype=torch.long,device=device), num_nodes=0)

            edges_cpu = []
            if num_nodes_fallback > 1:
                for i in range(num_nodes_fallback - 1): edges_cpu.append([i, i+1]) # Only forward for minimal
                # Add self-loops for minimal graph if flag is on
                # if self.add_self_loops: edges_cpu.extend([[i,i] for i in range(num_nodes_fallback)])

            edge_index_fallback = torch.tensor(edges_cpu, dtype=torch.long, device='cpu').t().to(device) if edges_cpu else torch.empty((2,0),dtype=torch.long,device=device)
            node_level_fallback = torch.full((num_nodes_fallback,), level_idx, dtype=torch.long, device='cpu').to(device)

            return Data(
                x=features[:num_nodes_fallback],
                edge_index=edge_index_fallback,
                node_level=node_level_fallback,
                num_nodes=num_nodes_fallback
            )

    ## Old build level graph, before adding dense local window.
    # def _build_level_graph(self, features, level_idx=0):
    #     """
    #     Build graph for a single level with careful device handling.
        
    #     Args:
    #         features: Node features [num_nodes, hidden_dim]
    #         level_idx: Level index (0=L0, 1=L1, 2=L2, 3=L3)
            
    #     Returns:
    #         graph: PyG Data object with the graph
    #     """
    #     try:
    #         # Get device info
    #         device = features.device
    #         cpu_device = torch.device('cpu')
            
    #         # Work on CPU for graph construction
    #         #features_cpu = features.detach().cpu()
    #         #num_nodes = features_cpu.size(0)
            
    #         ## Check if sequence length exceeds max_seq_len for level 0
    #         #if level_idx == 0 and num_nodes > self.max_seq_len:
    #         #    print(f"Truncating sequence from {num_nodes} to {self.max_seq_len}")
    #         #    num_nodes = self.max_seq_len
    #         #    features_cpu = features_cpu[:self.max_seq_len]
    #         #    features = features[:self.max_seq_len]  # Also truncate original
            
    #         num_nodes = features.size(0)
    #         # Check if sequence length exceeds max_seq_len for level 0
    #         if level_idx == 0 and num_nodes > self.max_seq_len:
    #             print(f"Truncating sequence from {num_nodes} to {self.max_seq_len}")
    #             num_nodes = self.max_seq_len
    #             features = features[:self.max_seq_len]
    #             features = features[:self.max_seq_len]  # Also truncate original
            
    #         # Create edges based on level
    #         edges = []
            
    #         # Different connectivity patterns based on level
    #         if num_nodes > 1:
    #             # Sequential connections (bidirectional)
    #             for i in range(num_nodes - 1):
    #                 edges.append([i, i+1])
    #                 edges.append([i+1, i])
                
    #             # Add self-loops if specified
    #             if self.add_self_loops:
    #                 for i in range(num_nodes):
    #                     edges.append([i, i])
                
    #             # Add long-range connections if specified
    #             if self.add_long_range_edges:
    #                 # Cap range distance for safety
    #                 range_distance = min(
    #                     self.long_range_distance * (level_idx + 1),
    #                     min(20, num_nodes // 4)  # Adaptive cap based on sequence length
    #                 )
    #                 for i in range(num_nodes):
    #                     for j in range(i + 2, min(i + range_distance, num_nodes)):
    #                         edges.append([i, j])
    #                         edges.append([j, i])
    #         elif num_nodes == 1:
    #             # Self-loop for single node
    #             edges = [[0, 0]]
            
    #         # Create tensors on CPU first
    #         if edges:
    #             edge_index_cpu = torch.tensor(edges, dtype=torch.long).t()
    #         else:
    #             edge_index_cpu = torch.zeros((2, 0), dtype=torch.long)
            
    #         node_level_cpu = torch.full((num_nodes,), level_idx, dtype=torch.long)
            
    #         # Then transfer to device
    #         edge_index = edge_index_cpu.to(device)
    #         node_level = node_level_cpu.to(device)
            
    #         # Create PyG Data object
    #         from torch_geometric.data import Data
    #         graph = Data(
    #             x=features,  # Already on device
    #             edge_index=edge_index,
    #             node_level=node_level
    #         )
            
    #         return graph
            
    #     except Exception as e:
    #         print(f"Error in building level graph: {e}")
    #         print("Creating minimal fallback graph")
            
    #         # Create minimal fallback graph
    #         device = features.device
    #         num_nodes = min(features.size(0), self.max_seq_len)
            
    #         # Create minimal edge structure (sequential only)
    #         edges_cpu = []
    #         for i in range(num_nodes - 1):
    #             edges_cpu.append([i, i+1])
            
    #         # Create tensors on CPU then move to device
    #         if edges_cpu:
    #             edge_index = torch.tensor(edges_cpu, dtype=torch.long, device='cpu').t().to(device)
    #         else:
    #             edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                
    #         # Create node levels on CPU then move to device
    #         node_level = torch.full((num_nodes,), level_idx, dtype=torch.long, device='cpu').to(device)
            
    #         # Create minimal graph
    #         from torch_geometric.data import Data
    #         return Data(
    #             x=features[:num_nodes],
    #             edge_index=edge_index,
    #             node_level=node_level
    #         )
            
    #     except Exception as e:
    #         print(f"Error in building level graph: {e}")
    #         print("Creating minimal fallback graph")
            
    #         # Create minimal fallback graph
    #         device = features.device
    #         num_nodes = min(features.size(0), self.max_seq_len)
            
    #         # Create minimal edge structure (sequential only)
    #         edges_cpu = []
    #         for i in range(num_nodes - 1):
    #             edges_cpu.append([i, i+1])
            
    #         # Create tensors on CPU then move to device
    #         if edges_cpu:
    #             edge_index = torch.tensor(edges_cpu, dtype=torch.long, device='cpu').t().to(device)
    #         else:
    #             edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                
    #         # Create node levels on CPU then move to device
    #         node_level = torch.full((num_nodes,), level_idx, dtype=torch.long, device='cpu').to(device)
            
    #         # Create minimal graph
    #         from torch_geometric.data import Data
    #         return Data(
    #             x=features[:num_nodes],
    #             edge_index=edge_index,
    #             node_level=node_level
    #         )
    
    def _process_level(self, graph, level_idx):
        """
        Process a single level with its dedicated transformer layers.
        
        Args:
            graph: PyG Data object with level graph
            level_idx: Level index
            
        Returns:
            processed_graph: Updated graph after processing
        """
        # First apply Lagrangian positional encoding if it's the token level
        #if level_idx == 0:
        #     # Skip Lagrangian if replacing with LapPE later, or apply conditionally
        #     # graph.x = self.lagrangian_pos_enc(graph.x, graph.edge_index)
        #     pass # Decide on Lagrangian later

        # Calculate simple 0..N-1 positions for this level's graph
        num_level_nodes = graph.x.size(0)
        level_positions = torch.arange(num_level_nodes, device=graph.x.device)
        if (
            str(getattr(self, "graph_geometry_mode", "sequence")).lower() == "grid2d"
            and hasattr(graph, "node_coord")
            and getattr(graph, "node_coord") is not None
            and int(getattr(graph, "node_coord").size(0)) == int(num_level_nodes)
            and int(getattr(graph, "node_coord").size(1)) == 2
        ):
            level_positions = getattr(graph, "node_coord").to(device=graph.x.device, dtype=torch.long)
        # Use level_idx directly for node_level tensor
        node_level_tensor = torch.full_like(level_positions, level_idx)
        if level_positions.dim() == 2:
            node_level_tensor = torch.full((num_level_nodes,), int(level_idx), device=graph.x.device, dtype=torch.long)
        processed_x = graph.x
        # Check if transformers exist for this level
        if level_idx < len(self.level_transformers):
            for transformer in self.level_transformers[level_idx]:
                # Prepare edge_attr if needed
                edge_attr_input = graph.edge_attr if hasattr(graph, 'edge_attr') and self.use_edge_attr else None
                if hasattr(transformer, "message_passing"):
                    mp = transformer.message_passing
                    if (
                        str(getattr(self, "graph_geometry_mode", "sequence")).lower() == "grid2d"
                        and hasattr(graph, "grid_shape")
                        and getattr(graph, "grid_shape") is not None
                        and len(getattr(graph, "grid_shape")) == 2
                    ):
                        gh = int(getattr(graph, "grid_shape")[0])
                        gw = int(getattr(graph, "grid_shape")[1])
                        if gh > 0 and gw > 0 and gh * gw == int(num_level_nodes):
                            mp.local_attn_runtime_level_grid_shapes = {int(level_idx): (gh, gw)}
                        else:
                            mp.local_attn_runtime_level_grid_shapes = {}
                    else:
                        mp.local_attn_runtime_level_grid_shapes = {}
                    mp.local_attn_runtime_spatial_metric = str(getattr(self, "graph_spatial_metric", "chebyshev"))
                if self.use_gradient_checkpointing:
                    result = checkpoint(
                        transformer,
                        processed_x, graph.edge_index, node_level_tensor,
                        positions=level_positions, level_offsets=None, edge_attr=edge_attr_input,
                        use_reentrant = False
                    )
                else:
                    result = transformer(
                        processed_x, graph.edge_index, node_level_tensor,
                        positions=level_positions, level_offsets=None, edge_attr=edge_attr_input
                    )
                processed_x = result[0] if isinstance(result, tuple) else result
            graph.x = self.layer_norm(processed_x) # Norm after all layers for the level
        else:
             logger.warning(f"No transformers defined for level {level_idx} in _process_level.")
        return graph
    
    def _pooled_seed_indices(self, level_sizes: List[int], device: torch.device) -> list:
        """Cached (child_idx, parent_idx, counts) per level for the pooled coarse seed.

        Mirrors the 1D window rule of _create_next_level: stride = comp*(1-overlap),
        parent i covers children [min(i*stride, n_lower-1), min(start+comp, n_lower)).
        """
        key = (tuple(int(s) for s in level_sizes), device.type, device.index)
        hit = self._pooled_seed_idx_cache.get(key)
        if hit is not None:
            return hit
        out = []
        for lvl in range(1, len(level_sizes)):
            n_lower = int(level_sizes[lvl - 1])
            n_higher = int(level_sizes[lvl])
            comp = int(self.compression_ratios[lvl - 1])
            stride = max(1, int(comp * (1 - self.overlap_ratios[lvl - 1])))
            child_list, parent_list = [], []
            for i in range(n_higher):
                start = min(i * stride, n_lower - 1)
                end = max(min(start + comp, n_lower), start + 1)
                child_list.append(torch.arange(start, end, dtype=torch.long))
                parent_list.append(torch.full((end - start,), i, dtype=torch.long))
            child_idx = torch.cat(child_list).to(device)
            parent_idx = torch.cat(parent_list).to(device)
            counts = torch.bincount(parent_idx, minlength=n_higher).clamp_min(1).to(device)
            out.append((child_idx, parent_idx, counts))
        if len(self._pooled_seed_idx_cache) > 16:
            self._pooled_seed_idx_cache.clear()
        self._pooled_seed_idx_cache[key] = out
        return out

    def _pooled_child_window_means(self, lower: torch.Tensor, lvl: int, n_parent: int) -> torch.Tensor:
        """Per-parent-window child means -> [B, n_parent, H] for level lvl pooling level lvl-1.

        Same window rule as _pooled_seed_indices / _create_next_level, but the regular bulk
        windows go through a ZERO-COPY unfold view (no 2x child materialization, no scatter);
        only the few clamped tail windows are sliced explicitly. All bounds are python ints —
        no GPU sync, no data-dependent shapes.
        """
        B, n_lower, H = lower.shape
        comp = int(self.compression_ratios[lvl - 1])
        stride = max(1, int(comp * (1 - self.overlap_ratios[lvl - 1])))
        n_bulk = min(n_parent, (n_lower - comp) // stride + 1) if n_lower >= comp else 0
        parts = []
        if n_bulk > 0:
            parts.append(lower.unfold(1, comp, stride)[:, :n_bulk].mean(dim=-1))
        for i in range(n_bulk, n_parent):
            start = min(i * stride, n_lower - 1)
            end = max(min(start + comp, n_lower), start + 1)
            parts.append(lower[:, start:end].mean(dim=1, keepdim=True))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def _level_offsets_list(self, level_offsets: torch.Tensor) -> List[int]:
        """level_offsets as python ints without a per-call GPU sync: the skeleton cache keeps
        the same offsets tensor alive across forwards, so key on its identity and .tolist()
        only when it actually changes (seq-len/graph change)."""
        key = (int(level_offsets.data_ptr()), int(level_offsets.numel()), str(level_offsets.device))
        cached = getattr(self, "_level_offsets_list_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        vals = [int(o) for o in level_offsets.tolist()]
        self._level_offsets_list_cache = (key, vals)
        return vals

    def _apply_pooled_upper_seed(self, x_cat: torch.Tensor, level_sizes: List[int]) -> torch.Tensor:
        """Seed coarse levels bottom-up with gated, level-projected child-window means.

        x_cat: [B, N_total, H] with the L0 slice already holding per-sample token embeddings
        and the coarse slices holding the base init (mask/zeros). Only the main hierarchy
        slices (first sum(level_sizes) nodes) are touched; appended AE decoder nodes keep
        their base init. seed = base + gate_l * (proj(pooled) - base); gate=0 == old behaviour.
        (proj(mean(children)) == mean(proj(children)) for a Linear, so pooling FIRST projects
        n_parent nodes instead of ~2x n_child — same output, ~1/3 the projection FLOPs.)
        """
        offsets = [0]
        for s in level_sizes:
            offsets.append(offsets[-1] + int(s))
        pieces = [x_cat[:, : offsets[1], :]]
        lower = x_cat[:, offsets[0] : offsets[1], :]
        for lvl in range(1, len(level_sizes)):
            pooled = self.level_projections[lvl - 1](
                self._pooled_child_window_means(lower, lvl, int(level_sizes[lvl]))
            )
            base = x_cat[:, offsets[lvl] : offsets[lvl + 1], :]
            gate = self.upper_seed_gates[lvl - 1].to(dtype=pooled.dtype)
            cur = base + gate * (pooled - base)
            pieces.append(cur)
            lower = cur
        pieces.append(x_cat[:, offsets[-1] :, :])
        return torch.cat(pieces, dim=1)

    def _apply_upward_refresh(self, x: torch.Tensor, level_offsets: torch.Tensor) -> torch.Tensor:
        """Gated per-layer upward re-pool: parent = parent + gate_l * proj_l(mean of child window).

        x: [B, N, H]. Runs bottom-up on the UPDATED lower slice, so one call carries L0's
        current state all the way to the top. Window rule and causality are those of
        _apply_pooled_upper_seed (parent ar_time = child window END); nodes past the main
        hierarchy slices (level_offsets[-1]:) are untouched. gate=0 is an exact no-op.
        One full-tensor rebuild per call (single cat), sync-free.

        Eager wrapper: resolves the offsets (cached .tolist() — data-dependent, must stay
        OUTSIDE any compiled graph) and dispatches to the pure-tensor core, which is the
        torch.compile target when hier_refresh_compile is on.
        """
        offsets = tuple(self._level_offsets_list(level_offsets))
        core = self._refresh_callable(self._upward_refresh_core, "_upward_refresh_compiled")
        return core(x, offsets)

    def _upward_refresh_core(self, x: torch.Tensor, offsets: tuple) -> torch.Tensor:
        level_sizes = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        pieces = [x[:, : offsets[1], :]]
        lower = x[:, offsets[0] : offsets[1], :]
        for lvl in range(1, len(level_sizes)):
            pooled = self._pooled_child_window_means(lower, lvl, int(level_sizes[lvl]))
            gate = self.upward_refresh_gates[lvl - 1].to(dtype=pooled.dtype)
            cur = x[:, offsets[lvl] : offsets[lvl + 1], :] + gate * self.upward_refresh_proj[lvl - 1](pooled)
            pieces.append(cur)
            lower = cur
        pieces.append(x[:, offsets[-1] :, :])
        return torch.cat(pieces, dim=1)

    def _refresh_callable(self, eager, cache_attr: str):
        """Return the torch.compile'd version of a refresh fn in training mode (fixed seq len
        -> one compile, fused elementwise chains), the eager one otherwise (eval/generation
        vary lengths and would recompile every step).

        Compilation is lazy, so failures surface on the FIRST EXECUTION (e.g. inductor/triton
        missing support for a new GPU arch), not at torch.compile() time — the probation
        wrapper catches that, logs once, and permanently falls back to eager instead of
        killing the run. After the first successful call the raw compiled fn is cached."""
        if not (getattr(self, "hier_refresh_compile", False) and self.training):
            return eager
        fn = getattr(self, cache_attr, None)
        if fn is None:
            try:
                compiled = torch.compile(eager, dynamic=False)
            except Exception as e:
                logger.warning("hier_refresh_compile: torch.compile unavailable (%s); staying eager.", e)
                setattr(self, cache_attr, eager)
                return eager

            def _probation(*args, _c=compiled, _e=eager, _attr=cache_attr, **kwargs):
                try:
                    out = _c(*args, **kwargs)
                except Exception as err:
                    logger.warning(
                        "hier_refresh_compile: compiled %s failed at runtime (%s: %s); "
                        "falling back to eager permanently.",
                        _attr, type(err).__name__, err,
                    )
                    setattr(self, _attr, _e)
                    return _e(*args, **kwargs)
                setattr(self, _attr, _c)  # first success -> cache the raw compiled fn
                return out

            fn = _probation
            setattr(self, cache_attr, fn)
        return fn

    def _layer_callable(self, transformer):
        """torch.compile'd refinement-layer forward in training mode when hier_layer_compile
        is on (fixed train shapes -> one compile per layer; fuses the eager orchestration
        around the flash/pack calls — the launch-bound cost at short sequences). Same lazy +
        probation policy as _refresh_callable: first runtime failure logs once and falls back
        to eager permanently. Eval/generation always eager (varying shapes would recompile).

        The 12 layers share one forward code object, so dynamo needs a cache slot per module
        instance — bump cache_size_limit once so late layers don't silently stay eager."""
        if not (getattr(self, "hier_layer_compile", False) and self.training):
            return transformer
        fn = getattr(transformer, "_pinball_compiled_forward", None)
        if fn is None:
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.cache_size_limit = max(int(_dynamo.config.cache_size_limit), 64)
                # Warm the backend-resolution dict EAGERLY so dynamo traces the constant
                # dict-hit path instead of the get_device_name miss branch (a break per call).
                try:
                    from .layers.hierarchical_message_passing import pick_attention_backend
                    pick_attention_backend(next(transformer.parameters()).device)
                except Exception:
                    pass
                compiled = torch.compile(transformer.forward, dynamic=False)
            except Exception as e:
                logger.warning("hier_layer_compile: torch.compile unavailable (%s); staying eager.", e)
                transformer._pinball_compiled_forward = transformer.forward
                return transformer.forward

            def _probation(*args, _c=compiled, _t=transformer, **kwargs):
                try:
                    out = _c(*args, **kwargs)
                except Exception as err:
                    logger.warning(
                        "hier_layer_compile: compiled layer failed at runtime (%s: %s); "
                        "falling back to eager permanently.",
                        type(err).__name__, err,
                    )
                    _t._pinball_compiled_forward = _t.forward
                    return _t.forward(*args, **kwargs)
                _t._pinball_compiled_forward = _c  # first success -> cache the raw compiled fn
                return out

            fn = _probation
            transformer._pinball_compiled_forward = fn
        return fn

    def _downward_gather_plan(self, level_offsets: torch.Tensor, node_ar_time: torch.Tensor) -> Dict[str, tuple]:
        """Per (fine,coarse) pair: (chosen [n_fine] most-recent CLOSED coarse node, valid mask).
        node_ar_time is content-independent and skeleton-cached, so this is computed once per
        graph shape and keyed on tensor identity (no per-layer sync, no recompute)."""
        key = (
            int(level_offsets.data_ptr()), int(node_ar_time.data_ptr()),
            int(node_ar_time.numel()), str(node_ar_time.device),
        )
        cached = getattr(self, "_downward_gather_plan_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        offsets = self._level_offsets_list(level_offsets)
        num_levels = len(offsets) - 1
        allow_same = bool(getattr(self, "hier_ar_allow_same_time", True))
        bidi_parent = bool(getattr(self, "hier_downward_bidi_parent", False))
        t = node_ar_time.to(torch.long)
        plan: Dict[str, tuple] = {}
        for q in range(num_levels - 1):
            q_time = t[offsets[q] : offsets[q + 1]]
            for m in range(q + 1, num_levels):
                pair = f"{q}:{m}"
                if pair not in self.downward_refresh_proj:
                    continue
                m_time = t[offsets[m] : offsets[m + 1]]
                if q_time.numel() == 0 or m_time.numel() == 0:
                    continue
                if bidi_parent:
                    # Bidi (MaskGIT/diffusion) variant: each fine node reads its CONTAINING
                    # parent (nearest window center under the same 1D rule as
                    # _pooled_seed_indices, chained q -> m so "my L3 = parent of my L2 =
                    # parent of my L1"), not the last-closed-along-the-sequence node. In
                    # curve mode this is the spatial ancestor. LEAKS under AR (the parent
                    # window contains the token) — ctor warns on that combination.
                    n_q = int(offsets[q + 1] - offsets[q])
                    idx = torch.arange(n_q, device=t.device, dtype=torch.float32)
                    for l in range(q + 1, m + 1):
                        comp = int(self.compression_ratios[l - 1])
                        stride = max(1, int(comp * (1 - self.overlap_ratios[l - 1])))
                        n_parent = int(offsets[l + 1] - offsets[l])
                        idx = torch.round((idx - (comp - 1) / 2.0) / stride).clamp_(
                            0, max(0, n_parent - 1)
                        )
                    chosen = idx.long()
                    valid = torch.ones((1, n_q, 1), dtype=torch.bool, device=t.device)
                    plan[pair] = (chosen, valid)
                    continue
                sorted_t, sidx = torch.sort(m_time)
                pos = torch.searchsorted(sorted_t, q_time, right=allow_same) - 1
                valid = (pos >= 0).view(1, -1, 1)
                chosen = sidx[pos.clamp(min=0)]
                plan[pair] = (chosen, valid)
        self._downward_gather_plan_cache = (key, plan)
        return plan

    def _local_pack_node_coords(
        self, level_offsets: torch.Tensor, n_total: int
    ) -> Optional[torch.Tensor]:
        """[n_total, ndim] float32 per-node ND coords for the packed-local axial RoPE
        (local_pack_rope_axial). L0 rows = the curve-mode coords (CURVE order, one per L0
        node); level l>=1 rows = centroid of the child window's coords under the SAME 1D
        window rule (reusing _pooled_seed_indices tables), chained bottom-up so an L2
        centroid averages L1 centroids. Returns None when no curve is active or when
        n_total != level_offsets[-1] (appended AE-decoder L0' nodes — don't guess tail
        coords; the consumer falls back to 1D close-time RoPE)."""
        coords0 = getattr(self, "_spatial_curve_coords_l0", None)
        if coords0 is None:
            return None
        offsets = self._level_offsets_list(level_offsets)
        if int(offsets[-1]) != int(n_total):
            if not getattr(self, "_pack_axial_tail_warned", False):
                logger.info(
                    "local_pack_rope_axial: %d extra non-hierarchy nodes present; "
                    "1D close-time RoPE fallback for this graph shape.",
                    int(n_total) - int(offsets[-1]),
                )
                self._pack_axial_tail_warned = True
            return None
        level_sizes = [int(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1)]
        if level_sizes[0] != int(coords0.size(0)):
            return None
        device = coords0.device
        cur = coords0.to(torch.float32)
        parts = [cur]
        tables = self._pooled_seed_indices(level_sizes, device)
        for lvl in range(1, len(level_sizes)):
            child_idx, parent_idx, counts = tables[lvl - 1]
            cent = torch.zeros(level_sizes[lvl], cur.size(1), device=device, dtype=torch.float32)
            cent.index_add_(0, parent_idx, cur.index_select(0, child_idx))
            cent = cent / counts.to(torch.float32).unsqueeze(-1)
            parts.append(cent)
            cur = cent
        return torch.cat(parts, dim=0)

    def _local_pack_build_spec(
        self,
        level_offsets: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: torch.Tensor,
    ) -> Optional[Dict[str, object]]:
        """Skeleton-cached spec for packed cross-level LOCAL attention (local_pack_cross_level):
        all nodes sorted by close time (ar_time), plus token-scale RoPE positions and the
        packed-row selection for the query levels. Ties at equal close time put coarse nodes
        BEFORE the L0 token when hier_ar_allow_same_time (a token may read a window closing
        AT its own position — same convention as _downward_gather_plan). Causality of the
        consuming attention follows from the sort: a causal window over close-time order only
        exposes nodes with ar_time <= the query's. Content-independent, cached on tensor
        identity like _downward_gather_plan."""
        key = (
            int(level_offsets.data_ptr()), int(node_ar_time.data_ptr()),
            int(node_ar_time.numel()), str(node_ar_time.device),
        )
        cached = getattr(self, "_local_pack_spec_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        n_total = int(node_ar_time.numel())
        t = node_ar_time.to(torch.long)
        lvl = node_level.to(device=t.device, dtype=torch.long)
        if int(lvl.numel()) != n_total or n_total <= 1:
            return None
        num_levels = max(1, int(lvl.max().item()) + 1)
        if bool(getattr(self, "hier_ar_allow_same_time", True)):
            # coarse first at equal close time: L1 < L2 < L3 < L0-token
            tie = torch.where(lvl > 0, lvl - 1, torch.full_like(lvl, num_levels - 1))
        else:
            tie = lvl  # token first; a same-time coarse row lands after it (invisible)
        stride = max(num_levels, 4)
        perm = torch.argsort(t * stride + tie, stable=True)
        pos = t.index_select(0, perm).contiguous()
        lvl_packed = lvl.index_select(0, perm)
        q_levels = tuple(int(l) for l in (getattr(self, "local_pack_query_levels", None) or [0]))
        sel_mask = torch.zeros_like(lvl_packed, dtype=torch.bool)
        for l in q_levels:
            sel_mask |= lvl_packed == int(l)
        sel = torch.nonzero(sel_mask, as_tuple=False).view(-1)
        cfg0 = dict(getattr(self, "local_attn_config", {}) or {}).get(0, {})
        window = int(getattr(self, "local_pack_window", 0) or 0)
        if window <= 0:
            window = int(cfg0.get("window", getattr(self, "l0_local_window", 0)) or 0)
        backend = str(cfg0.get("backend", getattr(self, "l0_local_backend", "sdpa")))
        spec: Dict[str, object] = {
            "num_nodes": n_total,
            "perm": perm,
            "pos": pos,
            "levels": lvl_packed.clamp(0, 3).contiguous(),  # per packed row, for the K/V level tags
            # Packed ROW indices per level, precomputed once per skeleton. Consumers must
            # use these rather than calling (levels == L).nonzero() in the layer loop:
            # nonzero() has a data-dependent output shape, so it forces a torch.compile
            # graph break, and it would re-derive a constant on every layer of every step.
            # Same reason lane_perm is built here instead of in _apply_local_pack_out.
            "level_rows": [
                (lvl_packed == L).nonzero(as_tuple=False).view(-1).contiguous()
                for L in range(4)
            ],
            "query_sel": sel,
            "query_nodes": perm.index_select(0, sel),
            "query_levels": q_levels,
            "window": window,
            "backend": backend,
        }
        # Axial ND RoPE (local_pack_rope_axial): true ND coords per packed row (curve mode).
        # The layer re-RoPEs q/k with exact spatial offsets instead of 1D close-time index;
        # missing pos_nd (text runs, AE-extended graphs) keeps the 1D behavior.
        if bool(getattr(self, "local_pack_rope_axial", False)):
            pos_nd_all = self._local_pack_node_coords(level_offsets, n_total)
            if pos_nd_all is not None:
                spec["pos_nd"] = pos_nd_all.index_select(0, perm).contiguous()
        # Coarse lane: same interleave restricted to L1-L3 rows (subselecting the sorted perm
        # preserves close-time order, so the causality argument carries over unchanged). The
        # lane window is denominated in COARSE slots — wide lateral reach at coarse density
        # prices. Queries: coarse levels that are packed-mode (in query_levels); keys: all.
        lane_q_levels = tuple(l for l in q_levels if int(l) > 0)
        if (
            bool(getattr(self, "local_pack_coarse_lane", False))
            and len(lane_q_levels) > 0
            and int(getattr(self, "local_pack_coarse_window", 0)) > 0
        ):
            lane_rows = torch.nonzero(lvl_packed > 0, as_tuple=False).view(-1)
            if lane_rows.numel() > 1:
                lane_levels = lvl_packed.index_select(0, lane_rows).clamp(0, 3).contiguous()
                lane_sel_mask = torch.zeros_like(lane_levels, dtype=torch.bool)
                for l in lane_q_levels:
                    lane_sel_mask |= lane_levels == int(l)
                lane_sel = torch.nonzero(lane_sel_mask, as_tuple=False).view(-1)
                lane_perm = perm.index_select(0, lane_rows).contiguous()
                spec["lane_perm"] = lane_perm
                # Lane row -> packed row, so hier_node_dropout drops the same node in the
                # lane as in the mixed window. Precomputed here for the same reason as
                # level_rows: nonzero() in the layer loop breaks the graph.
                spec["lane_rows"] = lane_rows.contiguous()
                spec["lane_pos"] = pos.index_select(0, lane_rows).contiguous()
                if "pos_nd" in spec:
                    spec["lane_pos_nd"] = spec["pos_nd"].index_select(0, lane_rows).contiguous()
                spec["lane_levels"] = lane_levels
                spec["lane_query_sel"] = lane_sel
                spec["lane_query_nodes"] = lane_perm.index_select(0, lane_sel)
                spec["lane_window"] = int(self.local_pack_coarse_window)
                # For the LSE merge: lane queries' row positions in the MIXED packing (to
                # fetch their mixed-window output/LSE), plus host copies of both position
                # arrays so the LSE recompute can slice keys without device syncs.
                lane_q_packed = lane_rows.index_select(0, lane_sel).contiguous()
                spec["lane_query_packed_rows"] = lane_q_packed
                spec["lane_query_packed_rows_host"] = lane_q_packed.tolist()
                spec["lane_query_sel_host"] = lane_sel.tolist()
                # FLEX UNION (local_pack_flex_union): ONE flex_attention call replaces the
                # mixed call + lane + LSE merge — a block-sparse mask expresses the union
                # key set ("within W mixed slots OR both-coarse within lane_window coarse
                # slots") as ONE softmax with exact dedup. Block sparsity needs a
                # block-COHERENT layout: [all coarse rows | all tokens] (each in close
                # order; the naive interleave scatters coarse at ~22% density and defeats
                # tile skipping — measured 11x slower). Same key sets via static rank
                # arrays: r = rank in the mixed packing, lane = rank in the coarse region.
                # Requires all levels queried (guarded at the consumer).
                if bool(getattr(self, "local_pack_flex_union", False)):
                    flex_perm_local = torch.cat([lane_rows, torch.nonzero(lvl_packed == 0, as_tuple=False).view(-1)])
                    spec["flex_perm"] = flex_perm_local.contiguous()          # split pos -> mixed row
                    spec["flex_r_mixed"] = flex_perm_local.contiguous()      # mixed rank per split pos
                    n_lane_rows = int(lane_rows.numel())
                    ar_split = torch.arange(flex_perm_local.numel(), device=perm.device)
                    spec["flex_lane_rank"] = torch.where(
                        ar_split < n_lane_rows, ar_split, torch.zeros_like(ar_split))
                    spec["flex_is_coarse"] = ar_split < n_lane_rows
                    spec["flex_query_nodes"] = perm.index_select(0, flex_perm_local)
                    spec["flex_n_lane"] = n_lane_rows
                    # COARSE RANK FOR *EVERY* ROW, not just coarse ones. flex_lane_rank
                    # above is 0 for token rows, which is fine while the union's coarse
                    # clause is gated on is_coarse[QUERY] -- but it makes the clause
                    # unusable for L0 queries. This is the count of coarse rows at or
                    # before each row in MIXED order, so an L0 query gets the coarse rank
                    # nearest its own position and a radius in coarse-rank space means
                    # the same thing for every query level. Coarse rows are sparse in
                    # position (~1 per 5.6 packed slots here), so R coarse ranks span
                    # ~5.6*R packed slots -- that is the whole point: reach per key is
                    # far cheaper on coarse rows than on tokens.
                    crank_mixed = (lvl_packed > 0).to(torch.long).cumsum(0) - 1
                    spec["flex_crank"] = crank_mixed.clamp_min(0).index_select(
                        0, flex_perm_local).contiguous()
                    spec["flex_levels"] = lvl_packed.clamp(0, 3).index_select(
                        0, flex_perm_local).contiguous()
        self._local_pack_spec_cache = (key, spec)
        return spec

    def _apply_downward_refresh(
        self, x: torch.Tensor, level_offsets: torch.Tensor, node_ar_time: torch.Tensor
    ) -> torch.Tensor:
        """Gated per-layer downward gather: fine node t reads, from each configured coarser
        level, that level's most-recent closed node (window_end <= t) — the same node a
        bridge/staggered edge points at, as a dense O(N) gather instead of scatter attention.
        Chained top-down (coarser levels update first), one full-tensor rebuild per call.
        gate 0 = exact no-op.

        Eager wrapper: resolves the cached offsets + gather plan (data-dependent bookkeeping,
        must stay OUTSIDE any compiled graph) and dispatches to the pure-tensor core — the
        torch.compile target when hier_refresh_compile is on."""
        offsets = tuple(self._level_offsets_list(level_offsets))
        plan = self._downward_gather_plan(level_offsets, node_ar_time)
        core = self._refresh_callable(self._downward_refresh_core, "_downward_refresh_compiled")
        return core(x, offsets, plan)

    def _downward_refresh_core(self, x: torch.Tensor, offsets: tuple, plan: Dict[str, tuple]) -> torch.Tensor:
        num_levels = len(offsets) - 1
        cur = {lvl: x[:, offsets[lvl] : offsets[lvl + 1], :] for lvl in range(num_levels)}
        for q in range(num_levels - 2, -1, -1):
            upd = None
            for m in range(q + 1, num_levels):
                pair = f"{q}:{m}"
                if pair not in plan:
                    continue
                chosen, valid = plan[pair]
                proj = self.downward_refresh_proj[pair](cur[m])          # few coarse rows
                g = proj.index_select(1, chosen) * valid.to(proj.dtype)  # bandwidth-only gather
                contrib = self.downward_refresh_gates[pair] * g
                upd = contrib if upd is None else upd + contrib
            if upd is not None:
                cur[q] = cur[q] + upd
        pieces = [x[:, : offsets[0], :]] + [cur[lvl] for lvl in range(num_levels)]
        pieces.append(x[:, offsets[-1] :, :])
        return torch.cat(pieces, dim=1)

    def _xq_children_tables(self, level_sizes: List[int], device: torch.device) -> tuple:
        """(l3_to_l1 [n3,K31], l1_to_l0 [n1,K10]) descendant index tables from the 1D window
        rule (same as _pooled_seed_indices), padded with -1. Built once per graph shape on the
        host and cached — no dependence on edge lists."""
        key = (tuple(int(s) for s in level_sizes), str(device))
        cached = self._xq_children_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        def windows(n_child: int, lvl: int) -> List[range]:
            comp = int(self.compression_ratios[lvl - 1])
            stride = max(1, int(comp * (1 - self.overlap_ratios[lvl - 1])))
            n_parent = int(level_sizes[lvl])
            out = []
            for i in range(n_parent):
                start = min(i * stride, n_child - 1)
                end = max(min(start + comp, n_child), start + 1)
                out.append(range(start, end))
            return out

        w10 = windows(int(level_sizes[0]), 1)          # L1 -> its L0 children
        w21 = windows(int(level_sizes[1]), 2)          # L2 -> L1
        w32 = windows(int(level_sizes[2]), 3)          # L3 -> L2
        l3_l1 = [sorted({c1 for c2 in w for c1 in w21[c2]}) for w in w32]

        def pad(rows: List[List[int]]) -> torch.Tensor:
            width = max(len(r) for r in rows)
            t = torch.full((len(rows), width), -1, dtype=torch.long)
            for i, r in enumerate(rows):
                t[i, : len(r)] = torch.tensor(list(r), dtype=torch.long)
            return t.to(device)

        tables = (pad(l3_l1), pad([list(r) for r in w10]))
        self._xq_children_cache = (key, tables)
        return tables

    def _cross_query_nominate(
        self,
        x: torch.Tensor,                      # [B, N, H] current node features (post-midpoint)
        level_offsets: torch.Tensor,
        node_ar_time: torch.Tensor,
        block: Optional[nn.Module] = None,    # the layer that just ran (trained q/k for stages 2/3)
    ) -> Optional[tuple]:
        """HQD v2: nominate far L0 sources per L0 query as ephemeral reads.

        Stage 1: score every CLOSED L3 window per query with the 0:3 cross-query refiner's
        TRAINED q/k (falls back to feature cosine if that refiner is absent) — per-query and
        causal, so selection sees only the past. Stage 2: score the shortlisted L3 windows'
        L1 children (trained local q/k when xq_nominate_stage2=local_qk, else cosine), keep
        the winners. Stage 3 (topk_l0 > 0): score the winners' L0 children with the SAME
        shared RoPE'd q/k the consuming read uses and keep the top-k. Selection is no-grad
        (the gradient flows through attention over the added reads, like v1 HQD); the
        optional per-query gate re-scores the stage-1 winner WITH grad (selector training).

        Returns (b_idx, src_idx, dst_idx) flat edges, or — packed mode — a marker tuple
        (zeros(1), candidates [B, n0, K] (-1 padded), dst_nodes [n0]) detected downstream
        by src.dim() == 3 and consumed by the dense fixed-K packed read.
        """
        offsets = self._level_offsets_list(level_offsets)
        if len(offsets) < 5:
            return None
        o = offsets
        n0, n1, n3 = o[1] - o[0], o[2] - o[1], o[4] - o[3]
        if min(n0, n1, n3) <= 0:
            return None
        B, _, H = x.shape
        dev = x.device
        t = node_ar_time.to(device=dev, dtype=torch.long)
        q_t, t1, t3 = t[o[0] : o[1]], t[o[1] : o[2]], t[o[3] : o[4]]
        allow_same = bool(getattr(self, "hier_ar_allow_same_time", True))
        x0, x1, x3 = x[:, o[0] : o[1], :], x[:, o[1] : o[2], :], x[:, o[3] : o[4], :]

        refiner = None
        for r in getattr(self, "pinball_cross_query_refiners", []) or []:
            if int(r.query_level) == 0 and int(r.memory_level) == 3:
                refiner = r
                break

        # CRITICAL: run the refiner projections in the AMBIENT grad mode, NOT inside the
        # no_grad selection block. Under autocast the FIRST call to a Linear caches its
        # bf16 weight cast; a no_grad-first call caches a DETACHED cast that every later
        # grad-mode call in the same forward silently reuses — severing the multirate
        # episode's (and the relevance gate's) gradient to these projections. Measured on
        # the 100ep xq run: q/kv_projs grads were None for the whole run (the projections
        # never trained). Selection below uses detached views; cost is unchanged.
        q_live = k3_live = None
        if refiner is not None:
            q_live = refiner.q_projs[0](refiner.q_norms[0](x0))
            k3_live = refiner.kv_projs[0](refiner.kv_norms[0](x3))[..., : refiner.work_dim]

        with torch.no_grad():
            # stage 1: L0 queries score L3 memory with the 0:3 refiner's trained q/k
            if refiner is not None:
                q, k3 = q_live.detach(), k3_live.detach()
            else:
                q, k3 = F.normalize(x0.float(), dim=-1), F.normalize(x3.float(), dim=-1)
            sc3 = torch.einsum("bqd,bnd->bqn", q.float(), k3.float())
            closed3 = (t3.view(1, -1) <= q_t.view(-1, 1)) if allow_same else (t3.view(1, -1) < q_t.view(-1, 1))
            sc3 = sc3.masked_fill(~closed3.unsqueeze(0), float("-inf"))
            k3top = min(self.xq_nominate_topk_l3, n3)
            idx3 = sc3.topk(k3top, dim=-1).indices                     # [B, n0, k3top]
            q_has_any = closed3.any(dim=-1)                            # [n0]

            # stage 2: score the shortlist's L1 children
            l3_to_l1, l1_to_l0 = self._xq_children_tables(
                [o[i + 1] - o[i] for i in range(len(o) - 1)], dev
            )
            cand1 = l3_to_l1[idx3.reshape(B, -1)].view(B, n0, -1)      # [B, n0, C] (-1 padded)
            pad1 = cand1 < 0
            cand1c = cand1.clamp(min=0)
            mp = getattr(block, "message_passing", None) if block is not None else None
            norm1 = getattr(block, "norm1", None) if block is not None else None
            use_local_qk = (
                self.xq_nominate_stage2 == "local_qk" and mp is not None and norm1 is not None
            )
            if use_local_qk:
                # trained metric: the layer's L0-query / L1-key local projections (per-level
                # when available, shared backbone q/k otherwise). Full-dim dot == sum of the
                # per-head dots the attention itself would compute.
                x0n, x1n = norm1(x0), norm1(x1)
                if getattr(mp, "per_level_local_qkv", False) and mp.num_local_levels >= 2:
                    q1s, k1s = mp.q_proj_level[0](x0n), mp.k_proj_level[1](x1n)
                else:
                    q1s, k1s = mp.q_proj(x0n), mp.k_proj(x1n)
                sc1_full = torch.einsum("bqd,bnd->bqn", q1s.float(), k1s.float())
            else:
                sc1_full = torch.einsum(
                    "bqd,bnd->bqn", F.normalize(x0.float(), dim=-1), F.normalize(x1.float(), dim=-1)
                )                                                      # [B, n0, n1]
            closed1 = (t1.view(1, -1) <= q_t.view(-1, 1)) if allow_same else (t1.view(1, -1) < q_t.view(-1, 1))
            sc1_full = sc1_full.masked_fill(~closed1.unsqueeze(0), float("-inf"))
            sc1 = sc1_full.gather(2, cand1c).masked_fill(pad1, float("-inf"))
            m = min(self.xq_nominate_topk_l1, sc1.size(-1))
            top1 = sc1.topk(m, dim=-1)
            chosen1 = cand1c.gather(2, top1.indices)                   # [B, n0, m]
            chosen_ok = torch.isfinite(top1.values) & q_has_any.view(1, -1, 1)

            # expand winners to their L0 children (per-query candidate table, -1 = invalid)
            src = l1_to_l0[chosen1.reshape(B, -1)].view(B, n0, -1)     # [B, n0, m*K10] L0-local ids
            ok = (src >= 0) & chosen_ok.repeat_interleave(l1_to_l0.size(1), dim=-1)
            q_pos = torch.arange(n0, device=dev).view(1, -1, 1)
            if self.xq_nominate_exclude_local:
                ok &= src <= (q_pos - int(getattr(self, "l0_local_window", 0)))
            else:
                ok &= (src <= q_pos) if allow_same else (src < q_pos)
            # per-row dedup (overlapping L1 windows share children; duplicates would double-
            # weight the packed softmax). Sort each row, void consecutive equals.
            src = torch.where(ok, src, torch.full_like(src, -1))
            src, _ = src.sort(dim=-1, descending=True)                 # -1s sink to the end
            dup = torch.zeros_like(src, dtype=torch.bool)
            dup[..., 1:] = src[..., 1:] == src[..., :-1]
            src = torch.where(dup | (src < 0), torch.full_like(src, -1), src)

            # stage 3: scored prune to topk_l0 with the read's OWN shared RoPE'd q/k
            k0top = int(self.xq_nominate_topk_l0)
            if k0top > 0 and src.size(-1) > k0top:
                if mp is not None and norm1 is not None:
                    x0n = norm1(x0)
                    q0r = mp.q_proj(x0n).view(B, n0, mp.num_heads, mp.head_dim)
                    k0r = mp.k_proj(x0n).view(B, n0, mp.num_heads, mp.head_dim)
                    # read_prerope: the read matches on content (pre-RoPE), so score the
                    # prune the same way — RoPE'd far dots decay with distance and would
                    # bias stage 3 toward near candidates the read then disagrees with.
                    if hasattr(mp, "rotary_pos_enc") and not bool(
                        getattr(self, "xq_nominate_read_prerope", False)
                    ):
                        pos0 = torch.arange(n0, device=dev).view(1, -1).expand(B, -1).reshape(-1)
                        q0r = mp.rotary_pos_enc.apply_rotary_pos_emb(
                            q0r.reshape(B * n0, mp.num_heads, mp.head_dim), pos0
                        ).view(B, n0, mp.num_heads, mp.head_dim)
                        k0r = mp.rotary_pos_enc.apply_rotary_pos_emb(
                            k0r.reshape(B * n0, mp.num_heads, mp.head_dim), pos0
                        ).view(B, n0, mp.num_heads, mp.head_dim)
                    q0f = q0r.reshape(B, n0, -1).float()
                    k0f = k0r.reshape(B, n0, -1).float()
                else:  # no block handle: cosine fallback keeps the prune functional
                    q0f = F.normalize(x0.float(), dim=-1)
                    k0f = q0f
                pruned = torch.full((B, n0, k0top), -1, device=dev, dtype=src.dtype)
                CH = 2048  # query chunk
                # Score via chunked GEMM + score-gather: one [B,c,D]x[B,D,n0] matmul per
                # chunk, then gather the C candidate columns. Same dots as the old
                # [B,c,C,D] key-gather + einsum, but ~25x less memory traffic (the gather
                # expansion dominated the nomination round). The GEMM is O(n0^2*D) though,
                # so past the FLOP crossover fall back to the linear-in-n0 gather path.
                use_gemm = n0 <= int(getattr(self, "xq_stage3_gemm_max_n0", 16384))
                k0T = k0f.transpose(1, 2).contiguous() if use_gemm else None
                for s0 in range(0, n0, CH):
                    e0 = min(n0, s0 + CH)
                    cnd = src[:, s0:e0]                                 # [B, c, C]
                    vld = cnd >= 0
                    safe = cnd.clamp(min=0)
                    if use_gemm:
                        s3 = torch.matmul(q0f[:, s0:e0], k0T).gather(2, safe)
                    else:
                        k_g = k0f.gather(
                            1, safe.reshape(B, -1, 1).expand(-1, -1, k0f.size(-1))
                        ).view(B, e0 - s0, cnd.size(-1), -1)
                        s3 = torch.einsum("bqd,bqcd->bqc", q0f[:, s0:e0], k_g)
                    s3 = s3.masked_fill(~vld, float("-inf"))
                    ti = s3.topk(k0top, dim=-1)
                    picked = cnd.gather(2, ti.indices)
                    pruned[:, s0:e0] = torch.where(
                        torch.isfinite(ti.values), picked, torch.full_like(picked, -1))
                src = pruned
            elif k0top > 0:
                pad_w = k0top - src.size(-1)
                if pad_w > 0:
                    src = F.pad(src, (0, pad_w), value=-1)

            has_any = src >= 0
            n_edges = int(has_any.sum())
            if n_edges == 0:
                self._xq_query_gate_full = None
                return None
            # visibility (gate monitor): edges, queries reached, mean fetch distance
            q_hit = has_any.any(dim=-1)                                # [B, n0]
            self._last_xq_nom_count = n_edges
            self._last_xq_nom_queries = int(q_hit.any(dim=0).sum())
            dist = (q_pos - src).float()
            self._last_xq_nom_mean_dist = float(dist[has_any].mean())

        # per-query relevance gate — WITH grad: re-scores the stage-1 winner so the task
        # gradient trains (w, b) AND the 0:3 selector projections. Applied to the read
        # output at every consuming layer via the message-passing hook attr.
        self._xq_query_gate_full = None
        if self.xq_nominate_gate_enable and refiner is not None and hasattr(self, "xq_gate_w"):
            # reuse the LIVE (grad-mode) stage-1 projections: gather each query's winning
            # L3 window key and re-dot — the gate is d(gate)/d(selector q/k)'s entry point
            k_g = k3_live.gather(1, idx3[:, :, 0].unsqueeze(-1).expand(-1, -1, k3_live.size(-1)))
            # attention-style scale keeps s O(1) across widths, so the w/b init is portable
            s_top = (q_live.float() * k_g.float()).sum(dim=-1) / float(max(1, int(refiner.work_dim))) ** 0.5
            gate = torch.sigmoid(self.xq_gate_w.float() * s_top + self.xq_gate_b.float())
            gate = gate * q_hit.to(gate.dtype)                         # no candidates -> hard 0
            gate_full = torch.ones(B, int(o[-1]), 1, device=dev, dtype=x.dtype)
            gate_full[:, o[0] : o[1], 0] = gate.to(x.dtype)
            self._xq_query_gate_full = gate_full
            self._last_xq_gate_mean = float(gate[q_hit].detach().mean()) if bool(q_hit.any()) else 0.0

        if self.xq_nominate_packed_read:
            dst_nodes = o[0] + torch.arange(n0, device=dev, dtype=torch.long)
            return torch.zeros(1, device=dev, dtype=torch.long), (o[0] + src.clamp(min=0)).masked_fill(src < 0, -1).long(), dst_nodes

        # legacy flat edge list for the per-edge scatter read
        b_idx, qi, si = has_any.nonzero(as_tuple=True)
        src_idx = o[0] + src[b_idx, qi, si]
        dst_idx = o[0] + qi
        return b_idx.long(), src_idx.long(), dst_idx.long()

    def _create_next_level(self, lower_graph, level_idx, compression_ratio, overlap_ratio):
        """
        Create the next level in the hierarchy based on processed lower level.
        
        Args:
            lower_graph: Processed graph from lower level
            level_idx: Index for the new level to create
            compression_ratio: Compression ratio to use
            overlap_ratio: Overlap ratio to use
            
        Returns:
            next_level_graph: Graph for the next level
            level_mapping: Dictionary mapping lower nodes to higher nodes and vice versa
        """
        device = lower_graph.x.device
        lower_features = lower_graph.x
        num_lower_nodes = lower_features.size(0)

        grid_shape = getattr(lower_graph, "grid_shape", None)
        if (
            str(getattr(self, "graph_geometry_mode", "sequence")).lower() == "grid2d"
            and grid_shape is not None
            and len(grid_shape) == 2
            and int(grid_shape[0]) * int(grid_shape[1]) == int(num_lower_nodes)
        ):
            lower_h = int(grid_shape[0])
            lower_w = int(grid_shape[1])
            # 2D overlap-aware pooling:
            # kernel ~= sqrt(compression_ratio), stride ~= kernel * (1 - overlap_ratio).
            comp_factor = max(1, int(round(math.sqrt(max(1, int(compression_ratio))))))
            kernel_2d = max(int(getattr(self, "graph_downsample_factor", 2)), comp_factor)
            kernel_2d = max(1, int(kernel_2d))
            ov = max(0.0, min(0.99, float(overlap_ratio)))
            stride_2d = max(1, int(round(float(kernel_2d) * (1.0 - ov))))

            num_higher_h = max(1, (lower_h + stride_2d - 1) // stride_2d)
            num_higher_w = max(1, (lower_w + stride_2d - 1) // stride_2d)
            num_higher_nodes = int(num_higher_h * num_higher_w)

            if self.input_mode == "tokens":
                mask_tensor = torch.tensor([self.mask_token_id], dtype=torch.long, device=device)
                mask_embedding = self.token_embedding(mask_tensor)
                higher_features = mask_embedding.repeat(num_higher_nodes, 1)
            elif self.input_mode == "features":
                higher_features = torch.zeros(num_higher_nodes, self.hidden_dim, device=device)
            else:
                raise ValueError("input_mode must be 'tokens' or 'features'")

            lower_to_higher = {}
            higher_to_lower = {}

            for hy in range(num_higher_h):
                for hx in range(num_higher_w):
                    hi = int(hy * num_higher_w + hx)
                    y0 = int(hy * stride_2d)
                    x0 = int(hx * stride_2d)
                    y1 = min(lower_h, y0 + kernel_2d)
                    x1 = min(lower_w, x0 + kernel_2d)

                    lower_ids: List[int] = []
                    for y in range(y0, y1):
                        base = y * lower_w
                        for x in range(x0, x1):
                            li = int(base + x)
                            lower_ids.append(li)

                    if not lower_ids:
                        anchor = min(num_lower_nodes - 1, max(0, y0 * lower_w + x0))
                        lower_ids = [int(anchor)]

                    higher_to_lower[hi] = lower_ids
                    li_tensor = torch.as_tensor(lower_ids, device=device, dtype=torch.long)
                    projected_features = self.level_projections[level_idx - 1](lower_features.index_select(0, li_tensor))
                    higher_features[hi] = projected_features.mean(dim=0)

                    for li in lower_ids:
                        if li not in lower_to_higher:
                            lower_to_higher[li] = [hi]
                        else:
                            lower_to_higher[li].append(hi)

            higher_coords = self._build_grid2d_coords(num_higher_h, num_higher_w, device=device)
            next_level_graph = self._build_level_graph(
                higher_features,
                level_idx,
                node_coord=higher_coords,
                grid_shape=(num_higher_h, num_higher_w),
            )
            level_mapping = (lower_to_higher, higher_to_lower)
            if self.verbose:
                logger.info(
                    "[HFGAT:COMPRESS] level=%d grid2d lower=%dx%d(%d) -> higher=%dx%d(%d) kernel=%d stride=%d overlap=%.2f mapped_lower=%d/%d assignments=%d avg_window=%.2f",
                    int(level_idx),
                    int(lower_h),
                    int(lower_w),
                    int(num_lower_nodes),
                    int(num_higher_h),
                    int(num_higher_w),
                    int(num_higher_nodes),
                    int(kernel_2d),
                    int(stride_2d),
                    float(ov),
                    int(len(lower_to_higher)),
                    int(num_lower_nodes),
                    int(sum(len(v) for v in lower_to_higher.values())),
                    float(sum(len(v) for v in higher_to_lower.values()) / max(1, len(higher_to_lower))),
                )
            return next_level_graph, level_mapping
        
        # Calculate stride based on compression and overlap
        stride = max(1, int(compression_ratio * (1 - overlap_ratio)))
        
        # Calculate number of nodes in the new level
        num_higher_nodes = max(1, (num_lower_nodes - 1) // stride + 1)

        
        if self.input_mode == "tokens":
            # Initialize Higher Features with Mask Embedding 
            mask_tensor = torch.tensor([self.mask_token_id], dtype=torch.long, device=device)
            mask_embedding = self.token_embedding(mask_tensor) # Get embedding [1, hidden_dim]
            # Repeat the mask embedding for all new nodes
            higher_features = mask_embedding.repeat(num_higher_nodes, 1)
        elif self.input_mode == "features":
            # either zeros **or** a learnable vector
            higher_features = torch.zeros(num_higher_nodes, self.hidden_dim, device=device)
            #higher_features = (
            #    self.mask_vector.repeat(num_higher_nodes, 1)
            #    if hasattr(self, "mask_vector") else
            #    torch.zeros(num_higher_nodes, self.hidden_dim, device=device)
            #)
        else:
            raise ValueError("input_mode must be 'tokens' or 'features'")
        
        # Initialize mappings
        lower_to_higher = {}  # Maps from lower level nodes to higher level nodes
        higher_to_lower = {}  # Maps from higher level nodes to lower level nodes
        
        # Create summary nodes with overlap
        for i in range(num_higher_nodes):
            # Define range for this summary node
            start_idx = min(i * stride, num_lower_nodes - 1)
            end_idx = min(start_idx + compression_ratio, num_lower_nodes)
            
            # Store mappings
            higher_to_lower[i] = list(range(start_idx, end_idx))
            
            # Create summary through weighted pooling of processed lower features
            if start_idx < end_idx:
                # Apply projection to lower features before pooling
                projected_features = self.level_projections[level_idx - 1](lower_features[start_idx:end_idx])
                higher_features[i] = torch.mean(projected_features, dim=0)
                
                # Update lower to higher mapping
                for j in range(start_idx, end_idx):
                    if j not in lower_to_higher:
                        lower_to_higher[j] = [i]
                    else:
                        lower_to_higher[j].append(i)
            else:
                # Handle edge case
                higher_features[i] = self.level_projections[level_idx - 1](lower_features[start_idx])
                
                # Update mapping
                if start_idx not in lower_to_higher:
                    lower_to_higher[start_idx] = [i]
                else:
                    lower_to_higher[start_idx].append(i)
        
        # Create graph for the next level
        next_level_graph = self._build_level_graph(higher_features, level_idx)
        
        # Store mappings
        level_mapping = (lower_to_higher, higher_to_lower)
        if self.verbose:
            logger.info(
                "[HFGAT:COMPRESS] level=%d sequence lower=%d -> higher=%d compression=%d stride=%d overlap=%.2f mapped_lower=%d/%d assignments=%d avg_window=%.2f",
                int(level_idx),
                int(num_lower_nodes),
                int(num_higher_nodes),
                int(compression_ratio),
                int(stride),
                float(overlap_ratio),
                int(len(lower_to_higher)),
                int(num_lower_nodes),
                int(sum(len(v) for v in lower_to_higher.values())),
                float(sum(len(v) for v in higher_to_lower.values()) / max(1, len(higher_to_lower))),
            )
        
        return next_level_graph, level_mapping
    
    
    
    def _add_l0_parent_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        level_mappings: list,
        level_offsets: list,
        num_levels: int,
        device: torch.device,
        l0_edge_type_id: Optional[int] = None,
    ):
        import torch

        n0 = level_offsets[1] - level_offsets[0]
        if n0 <= 0 or num_levels <= 1:
            return edge_index, edge_type

        min_parent_level = max(1, int(getattr(self, "l0_parent_edge_min_level", 2)))
        max_parent_level_raw = getattr(self, "l0_parent_edge_max_level", 3)
        if max_parent_level_raw is None:
            max_parent_level = num_levels - 1
        else:
            max_parent_level = max(min_parent_level, int(max_parent_level_raw))

        total_nodes = int(level_offsets[-1]) if level_offsets else 0
        key_width = int(max(1, total_nodes))

        existing_keys = set()
        if edge_index is not None and edge_index.numel() > 0:
            base_src = edge_index[0].to(dtype=torch.long)
            base_dst = edge_index[1].to(dtype=torch.long)
            existing_keys = set((base_src * key_width + base_dst).detach().cpu().tolist())

        extra_src = []
        extra_dst = []
        extra_keys = set()

        lower_to_higher_all = [lm[0] for lm in level_mappings]

        for l0_local in range(n0):
            current = [l0_local]
            for level_idx, lower_to_higher in enumerate(lower_to_higher_all):
                higher_level_num = level_idx + 1
                nxt = []
                for li in current:
                    if li not in lower_to_higher:
                        continue
                    for hi_local in lower_to_higher[li]:
                        if min_parent_level <= higher_level_num <= max_parent_level:
                            g_hi = int(level_offsets[level_idx + 1] + hi_local)
                            g_l0 = int(l0_local)
                            k_down = (g_hi * key_width) + g_l0
                            if k_down not in existing_keys and k_down not in extra_keys:
                                extra_keys.add(k_down)
                                extra_src.append(g_hi)
                                extra_dst.append(g_l0)
                            if getattr(self, "l0_parent_edges_bidirectional", False):
                                k_up = (g_l0 * key_width) + g_hi
                                if k_up not in existing_keys and k_up not in extra_keys:
                                    extra_keys.add(k_up)
                                    extra_src.append(g_l0)
                                    extra_dst.append(g_hi)
                        nxt.append(hi_local)
                current = nxt
                if not current:
                    break

        if len(extra_src) == 0:
            return edge_index, edge_type

        extra_src = torch.tensor(extra_src, device=device, dtype=edge_index.dtype)
        extra_dst = torch.tensor(extra_dst, device=device, dtype=edge_index.dtype)

        extra_e = torch.stack([extra_src, extra_dst], dim=0)

        new_edge_index = torch.cat([edge_index, extra_e], dim=1)

        if edge_type is not None and l0_edge_type_id is not None:
            extra_et = torch.full(
                (extra_e.size(1),),
                int(l0_edge_type_id),
                dtype=edge_type.dtype,
                device=edge_type.device,
            )
            new_edge_type = torch.cat([edge_type, extra_et], dim=0)
        else:
            new_edge_type = edge_type

        return new_edge_index, new_edge_type

    def _add_past_bridge_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        level_offsets: List[int],
        node_ar_time: Optional[torch.Tensor],
        child_level: int = 0,
        parent_level: int = 1,
        edge_type_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if edge_index is None:
            return edge_index, edge_type
        if node_ar_time is None or node_ar_time.numel() == 0:
            return edge_index, edge_type

        num_levels = len(level_offsets) - 1
        if num_levels <= max(child_level, parent_level):
            return edge_index, edge_type

        child_start = int(level_offsets[child_level])
        child_end = int(level_offsets[child_level + 1])
        parent_start = int(level_offsets[parent_level])
        parent_end = int(level_offsets[parent_level + 1])
        nc = max(0, child_end - child_start)
        np = max(0, parent_end - parent_start)
        if nc <= 0 or np <= 0:
            return edge_index, edge_type

        child_times = node_ar_time[child_start:child_end].to(dtype=torch.long)
        parent_times = node_ar_time[parent_start:parent_end].to(dtype=torch.long)
        if child_times.numel() == 0 or parent_times.numel() == 0:
            return edge_index, edge_type

        parent_sorted_times, parent_sort_idx = torch.sort(parent_times)
        allow_same = bool(getattr(self, "hier_ar_allow_same_time", True))
        pos = torch.searchsorted(parent_sorted_times, child_times, right=allow_same) - 1
        valid = pos >= 0
        if not bool(valid.any()):
            return edge_index, edge_type

        child_local = torch.arange(nc, device=edge_index.device, dtype=torch.long)[valid]
        parent_local = parent_sort_idx[pos[valid]]
        cand_src = parent_local + parent_start
        cand_dst = child_local + child_start

        total_nodes = int(level_offsets[-1]) if level_offsets else int(node_ar_time.numel())
        key_width = int(max(1, total_nodes))

        existing_keys = set()
        if edge_index.numel() > 0:
            base_src = edge_index[0].to(dtype=torch.long)
            base_dst = edge_index[1].to(dtype=torch.long)
            existing_keys = set((base_src * key_width + base_dst).detach().cpu().tolist())

        extra_src: List[int] = []
        extra_dst: List[int] = []
        extra_keys = set()
        for s, d in zip(cand_src.detach().cpu().tolist(), cand_dst.detach().cpu().tolist()):
            k = (int(s) * key_width) + int(d)
            if k in existing_keys or k in extra_keys:
                continue
            extra_keys.add(k)
            extra_src.append(int(s))
            extra_dst.append(int(d))

        if len(extra_src) == 0:
            return edge_index, edge_type

        extra_src_t = torch.tensor(extra_src, device=edge_index.device, dtype=edge_index.dtype)
        extra_dst_t = torch.tensor(extra_dst, device=edge_index.device, dtype=edge_index.dtype)
        extra_e = torch.stack([extra_src_t, extra_dst_t], dim=0)

        new_edge_index = torch.cat([edge_index, extra_e], dim=1)
        if edge_type is not None:
            if edge_type_id is None:
                # Default to the usual downward cross-level type for the parent
                # source level: L1->L0=5, L2->L1/L0=7, L3->L2/L0=9.
                edge_type_id = 5 + (2 * max(0, int(parent_level) - 1))
            extra_et = torch.full(
                (extra_e.size(1),),
                int(edge_type_id),
                dtype=edge_type.dtype,
                device=edge_type.device,
            )
            new_edge_type = torch.cat([edge_type, extra_et], dim=0)
        else:
            new_edge_type = edge_type

        return new_edge_index, new_edge_type

    def _add_l0_staggered_past_parent_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        level_offsets: List[int],
        node_ar_time: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Connect each L0 token to a STAGGERED chain of strictly-past ancestors.

        Unlike `_add_l0_parent_edges` (own-ancestor, downward half always AR-cut) and unlike
        a direct L0->most-recent-past-L_k bridge, this walks the chain the way
        `ensure_past_hier_edges_all_levels` composes adjacent bridges: the past-L1 of the
        token, then the past-L2 of *that* past-L1, then the past-L3 of *that* past-L2, etc.
        Each step searches with the previously-chosen parent's time as the anchor, so every
        emitted parent has time < the token's time by a widening margin -> no edge can be
        cut by `_filter_edges_by_ar_time` and no future token can leak down into L0.
        """
        if edge_index is None:
            return edge_index, edge_type
        if node_ar_time is None or node_ar_time.numel() == 0:
            return edge_index, edge_type

        num_levels = len(level_offsets) - 1
        if num_levels <= 1:
            return edge_index, edge_type

        off0 = int(level_offsets[0])
        n0 = int(level_offsets[1]) - off0
        if n0 <= 0:
            return edge_index, edge_type

        device = edge_index.device
        allow_same = bool(getattr(self, "hier_ar_allow_same_time", False))
        min_level = max(1, int(getattr(self, "l0_past_parent_min_level", 1)))
        max_level_raw = getattr(self, "l0_past_parent_max_level", None)
        if max_level_raw is None:
            max_level = num_levels - 1
        else:
            max_level = min(num_levels - 1, max(min_level, int(max_level_raw)))
        if max_level < 1:
            return edge_index, edge_type

        l0_idx = torch.arange(n0, device=device, dtype=torch.long)
        # anchor_time[i] = the time the next-higher past parent must precede. Starts at the
        # token's own time; steps back to each chosen parent's time. -1 marks a broken chain
        # (no past parent at some level) so all higher levels are skipped for that token.
        anchor_time = node_ar_time[off0:off0 + n0].to(dtype=torch.long)
        alive = torch.ones(n0, dtype=torch.bool, device=device)

        extra_src: List[torch.Tensor] = []
        extra_dst: List[torch.Tensor] = []
        extra_type_ids: List[int] = []

        for parent_level in range(1, max_level + 1):
            pstart = int(level_offsets[parent_level])
            pend = int(level_offsets[parent_level + 1])
            np_ = max(0, pend - pstart)
            if np_ <= 0:
                break
            ptimes = node_ar_time[pstart:pend].to(dtype=torch.long)
            sorted_pt, sort_idx = torch.sort(ptimes)
            # most recent parent strictly before the anchor (<= when allow_same)
            pos = torch.searchsorted(sorted_pt, anchor_time, right=allow_same) - 1
            valid = alive & (pos >= 0)
            if not bool(valid.any()):
                break
            chosen_local = torch.zeros(n0, dtype=torch.long, device=device)
            chosen_local[valid] = sort_idx[pos[valid]]
            chosen_time = torch.full((n0,), -1, dtype=torch.long, device=device)
            chosen_time[valid] = ptimes[chosen_local[valid]]

            if parent_level >= min_level:
                sel = valid
                src = chosen_local[sel] + pstart
                dst = l0_idx[sel] + off0
                extra_src.append(src)
                extra_dst.append(dst)
                # match _add_past_bridge_edges' default downward type scheme
                extra_type_ids.append(5 + (2 * (parent_level - 1)))

            # step the anchor up to the chosen parent's time; kill broken chains
            anchor_time = torch.where(valid, chosen_time, anchor_time)
            alive = valid
            if not bool(alive.any()):
                break

        if not extra_src:
            return edge_index, edge_type

        total_nodes = int(level_offsets[-1]) if level_offsets else int(node_ar_time.numel())
        key_width = int(max(1, total_nodes))
        existing_keys = set()
        if edge_index.numel() > 0:
            base_src = edge_index[0].to(dtype=torch.long)
            base_dst = edge_index[1].to(dtype=torch.long)
            existing_keys = set((base_src * key_width + base_dst).detach().cpu().tolist())

        keep_src: List[int] = []
        keep_dst: List[int] = []
        keep_type: List[int] = []
        seen_keys = set()
        for src_t, dst_t, type_id in zip(extra_src, extra_dst, extra_type_ids):
            s_list = src_t.detach().cpu().tolist()
            d_list = dst_t.detach().cpu().tolist()
            for s, d in zip(s_list, d_list):
                k = (int(s) * key_width) + int(d)
                if k in existing_keys or k in seen_keys:
                    continue
                seen_keys.add(k)
                keep_src.append(int(s))
                keep_dst.append(int(d))
                keep_type.append(int(type_id))

        if not keep_src:
            return edge_index, edge_type

        extra_e = torch.stack(
            [
                torch.tensor(keep_src, device=device, dtype=edge_index.dtype),
                torch.tensor(keep_dst, device=device, dtype=edge_index.dtype),
            ],
            dim=0,
        )
        new_edge_index = torch.cat([edge_index, extra_e], dim=1)
        if edge_type is not None:
            extra_et = torch.tensor(keep_type, device=edge_type.device, dtype=edge_type.dtype)
            new_edge_type = torch.cat([edge_type, extra_et], dim=0)
        else:
            new_edge_type = edge_type
        return new_edge_index, new_edge_type

    def _add_l0_past_l1_bridge_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        level_offsets: List[int],
        node_ar_time: Optional[torch.Tensor],
        l1_edge_type_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._add_past_bridge_edges(
            edge_index=edge_index,
            edge_type=edge_type,
            level_offsets=level_offsets,
            node_ar_time=node_ar_time,
            child_level=0,
            parent_level=1,
            edge_type_id=l1_edge_type_id,
        )

    def _dedup_edges_keep_first(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        num_nodes: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if edge_index is None or edge_index.numel() == 0:
            return edge_index, edge_type

        e_count = int(edge_index.size(1))
        if e_count <= 1:
            return edge_index, edge_type

        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)
        width = int(max(1, num_nodes))
        keys = (src * width + dst).detach().cpu().tolist()

        seen = set()
        keep_idx = []
        for i, k in enumerate(keys):
            if k in seen:
                continue
            seen.add(k)
            keep_idx.append(i)

        if len(keep_idx) == e_count:
            return edge_index, edge_type

        keep = torch.tensor(keep_idx, dtype=torch.long, device=edge_index.device)
        edge_index = edge_index.index_select(1, keep)
        edge_type = edge_type.index_select(0, keep) if edge_type is not None else None
        return edge_index, edge_type

    def _unified_graph_cache_key(self, seq_len: int) -> Tuple[Any, ...]:
        return (
            int(seq_len),
            str(getattr(self, "graph_geometry_mode", "sequence")),
            int(getattr(self, "graph_grid_height", 0)),
            int(getattr(self, "graph_grid_width", 0)),
            str(getattr(self, "graph_spatial_metric", "chebyshev")),
            int(getattr(self, "graph_downsample_factor", 2)),
            bool(self.hier_ar_enable),
            bool(self.hier_ar_allow_same_time),
            bool(self.l0_ar_enable),
            bool(self.enable_l0_parent_edges),
            bool(self.l0_parent_edges_bidirectional),
            int(self.l0_parent_edge_min_level),
            -1 if self.l0_parent_edge_max_level is None else int(self.l0_parent_edge_max_level),
            int(self.long_range_distance or 0),
            bool(self.add_self_loops),
            bool(self.add_long_range_edges),
            bool(self.ensure_l0_past_l1_edges),
            bool(getattr(self, "ensure_past_hier_edges_all_levels", False)),
            bool(getattr(self, "ensure_l0_past_parent_edges", False)),
            int(getattr(self, "l0_past_parent_min_level", 1)),
            -1 if getattr(self, "l0_past_parent_max_level", None) is None else int(self.l0_past_parent_max_level),
            str(getattr(self, "autoenc_graph_mode", "off")),
            bool(getattr(self, "autoenc_coupled_feedback", True)),
        )

    def _compute_node_ar_time(
        self,
        level_offsets: List[int],
        level_mappings: List[Tuple[Dict[int, List[int]], Dict[int, List[int]]]],
        device: torch.device,
    ) -> torch.Tensor:
        num_levels = max(0, len(level_offsets) - 1)
        total_nodes = int(level_offsets[-1]) if level_offsets else 0
        ar_time = torch.zeros(total_nodes, dtype=torch.long, device=device)
        if num_levels <= 0 or total_nodes <= 0:
            return ar_time

        n0 = int(level_offsets[1] - level_offsets[0]) if num_levels > 0 else 0
        if n0 > 0:
            ar_time[level_offsets[0] : level_offsets[1]] = torch.arange(n0, device=device, dtype=torch.long)

        level_desc: List[Optional[torch.Tensor]] = [None] * num_levels
        level_desc[0] = torch.arange(max(n0, 1), device=device, dtype=torch.long)[:n0]

        for lvl in range(1, num_levels):
            start = int(level_offsets[lvl])
            end = int(level_offsets[lvl + 1])
            n_cur = max(0, end - start)
            if n_cur <= 0:
                level_desc[lvl] = torch.empty((0,), dtype=torch.long, device=device)
                continue

            prev_desc = level_desc[lvl - 1]
            if prev_desc is None or prev_desc.numel() == 0:
                prev_desc = torch.arange(n_cur, device=device, dtype=torch.long)

            higher_to_lower: Dict[int, List[int]] = {}
            if lvl - 1 < len(level_mappings):
                _, higher_to_lower = level_mappings[lvl - 1]

            desc = torch.empty((n_cur,), dtype=torch.long, device=device)
            prev_n = int(prev_desc.numel())
            for hi_local in range(n_cur):
                lower_nodes = higher_to_lower.get(hi_local, [])
                if lower_nodes:
                    child = torch.as_tensor(lower_nodes, dtype=torch.long, device=device)
                    valid = (child >= 0) & (child < prev_n)
                    child = child[valid]
                    if child.numel() > 0:
                        desc[hi_local] = prev_desc[child].max()
                        continue
                if prev_n <= 0:
                    desc[hi_local] = torch.tensor(0, device=device, dtype=torch.long)
                elif n_cur == 1:
                    desc[hi_local] = prev_desc[-1]
                else:
                    approx = int(round((hi_local * (prev_n - 1)) / max(1, n_cur - 1)))
                    approx = max(0, min(prev_n - 1, approx))
                    desc[hi_local] = prev_desc[approx]

            ar_time[start:end] = desc
            level_desc[lvl] = desc

        return ar_time

    def _copredict_inject(
        self,
        token_features: torch.Tensor,      # [B, T_graph, H] L0 pre-head features
        x_final: torch.Tensor,             # [B, N, H] all-level refined features (per-sample node dim)
        level_offsets: Optional[List[int]],
        node_ar_time: Optional[torch.Tensor],  # [N] per-sample window-end times (shared across batch)
        seq_len_graph: int,
    ) -> torch.Tensor:
        """Add each configured coarse level's causal, strictly-past summary into L0's pre-head
        features. For L0 position t, level L contributes its most-recent CLOSED node
        (window_end <= t, i.e. the same node the staggered edges pick), projected per level and
        scaled by a learnable gate (init 0 => identity). Leak-free: window_end <= t never reveals
        a token after t. node_ar_time is content-independent, so the gather indices are computed
        once and shared across the batch."""
        if not getattr(self, "hier_copredict_l0", False) or not getattr(self, "hier_copredict_levels", None):
            return token_features
        if node_ar_time is None or level_offsets is None or node_ar_time.numel() < int(level_offsets[-1]):
            return token_features
        if x_final.dim() != 3:
            return token_features

        dev = token_features.device
        allow_same = bool(getattr(self, "hier_ar_allow_same_time", True))
        q_time = node_ar_time[:seq_len_graph].to(torch.long)   # L0 window-end == token time
        out = token_features
        for lvl in self.hier_copredict_levels:
            s, e = int(level_offsets[lvl]), int(level_offsets[lvl + 1])
            if e <= s:
                continue
            ptime = node_ar_time[s:e].to(torch.long)
            sorted_t, sidx = torch.sort(ptime)
            # most-recent coarse node with window_end <= t (allow_same) or < t (strict AR)
            pos = torch.searchsorted(sorted_t, q_time, right=allow_same) - 1
            valid = (pos >= 0)
            chosen = sidx[pos.clamp(min=0)]                    # [T_graph]
            coarse = x_final[:, s:e, :]                        # [B, n_lvl, H]
            gathered = coarse.index_select(1, chosen.to(coarse.device)).to(dev)
            gathered = gathered * valid.to(dev, gathered.dtype).view(1, -1, 1)
            proj = self.copredict_proj[str(lvl)](gathered)
            out = out + self.copredict_gate[str(lvl)] * proj
        return out

    def _compute_predictive_aux_loss(
        self,
        x_bnh: torch.Tensor,                      # [B, N, H] refined all-level features
        level_offsets: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Next-concept prediction at the coarse timescale: the level-k node at window i
        predicts the DETACHED pooled child summary of window i+h at its own level. h (auto)
        is the first horizon whose window starts at/after this window's END (ceil(comp/stride);
        h=2 at 50% overlap), so the target holds strictly-future content and copying the
        current window scores nothing. Targets are detached (JEPA-style): gradient flows only
        into the predicting node, training the hierarchy to FORECAST — the same pressure CE
        puts on L0, one timescale up. Returns None when disabled/not training."""
        if not (self.training and getattr(self, "hier_predaux_enable", False)):
            return None
        if level_offsets is None or not getattr(self, "hier_predaux_levels", None):
            return None
        offsets = [int(o) for o in level_offsets.tolist()]
        level_sizes = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        idx_per_level = self._pooled_seed_indices(level_sizes, x_bnh.device)
        B = x_bnh.size(0)
        losses = []
        for lvl in self.hier_predaux_levels:
            if lvl >= len(level_sizes):
                continue
            n = int(level_sizes[lvl])
            comp = int(self.compression_ratios[lvl - 1])
            stride = max(1, int(comp * (1 - self.overlap_ratios[lvl - 1])))
            h = self.hier_predaux_horizon if self.hier_predaux_horizon > 0 else -(-comp // stride)
            if n <= h:
                continue
            child_idx, parent_idx, counts = idx_per_level[lvl - 1]
            children = x_bnh[:, offsets[lvl - 1] : offsets[lvl], :].index_select(1, child_idx)
            pooled = children.new_zeros(B, n, children.size(-1))
            pooled.index_add_(1, parent_idx, children)
            target = (pooled / counts.view(1, -1, 1).to(dtype=pooled.dtype)).detach()
            feats = x_bnh[:, offsets[lvl] : offsets[lvl + 1], :]
            pairs = [(self.predaux_predictor[str(lvl)](feats)[:, : n - h], target[:, h:])]
            if getattr(self, "hier_predaux_directions", "next") == "both":
                pairs.append((self.predaux_predictor_prev[str(lvl)](feats)[:, h:], target[:, : n - h]))
            mode = self.hier_predaux_loss_mode
            for p, t in pairs:
                p, t = p.float(), t.float()
                if mode == "cosine":
                    losses.append((1.0 - F.cosine_similarity(p, t, dim=-1)).mean())
                elif mode in ("mse_norm", "nmse"):
                    losses.append(F.mse_loss(F.normalize(p, dim=-1), F.normalize(t, dim=-1)))
                else:
                    losses.append(F.mse_loss(p, t))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def gate_monitor(self) -> Dict[str, float]:
        """Flat dict of the essential learnable gates/scales, so you can watch which pathways
        open up or collapse during training. Call it anywhere (eval loop, debugger), or flip
        pinball_monitor_gates to have the forward log it every pinball_monitor_gates_every calls.
        All near-0 = pathway unused; growing magnitude = the model is leaning on it."""
        d: Dict[str, float] = {}
        # Coarse co-prediction into L0 (per-level head gate).
        if getattr(self, "hier_copredict_l0", False) and hasattr(self, "copredict_gate"):
            for k, g in self.copredict_gate.items():
                d[f"copred.L{k}->L0.gate"] = float(g.detach())
        # Per-layer upward refresh (child-window re-pool into parent) gates.
        if getattr(self, "hier_upward_refresh", False) and hasattr(self, "upward_refresh_gates"):
            for i in range(self.upward_refresh_gates.numel()):
                d[f"upref.L{i}->L{i + 1}.gate"] = float(self.upward_refresh_gates[i].detach())
        # Per-layer downward refresh (most-recent-closed coarse gather) gates.
        if getattr(self, "hier_downward_refresh", False) and hasattr(self, "downward_refresh_gates"):
            for k, g in self.downward_refresh_gates.items():
                q, m = k.split(":")
                d[f"downref.L{q}<-L{m}.gate"] = float(g.detach())
        # HQD reach: mean selection distance in node ids (curve order for L0) and the
        # fraction landing inside the local window, i.e. slots spent on nodes the packed
        # attention already sees. Watch mean_dist collapsing toward the window half --
        # that is training closing the long-range route.
        _md = getattr(self, "_last_hqd_mean_dist", None)
        if _md is not None:
            d["hqd.mean_dist"] = float(_md)
        _lf = getattr(self, "_last_hqd_local_frac", None)
        if _lf is not None:
            d["hqd.local_overlap"] = float(_lf)
        # Attention source gates (attention_source_gating_enable + _debug): the layers
        # stash 0-dim tensors, so this is the ONLY sync and it happens once per monitor
        # interval. sigmoid(hqd) is the most direct "is the model choosing to use HQD"
        # readout there is -- every pre-gate run had to infer it from ERF instead.
        _sg: Dict[str, List[float]] = {}
        for _blk in getattr(self, "refinement_transformers", []) or []:
            _st = getattr(getattr(_blk, "message_passing", None), "_last_attention_source_gates", None)
            if isinstance(_st, dict):
                for _n, _v in _st.items():
                    _sg.setdefault(_n, []).append(float(_v))
        for _n, _vals in _sg.items():
            if _vals:
                d[f"srcgate.{_n}.mean"] = sum(_vals) / len(_vals)
                d[f"srcgate.{_n}.max"] = max(_vals)
                # min too: for a gate starting uniform, max-min IS the signal. A mean
                # that falls while the spread widens is the mechanism SPECIALISING by
                # depth (earning its place in some layers, correctly rejected in others);
                # a mean that falls with the spread staying tight is plain rejection.
                # Those two look identical in the mean alone.
                d[f"srcgate.{_n}.min"] = min(_vals)

        # Cross-query refiners (downward L0/L1/L2<-L3 or any query<-memory) residual write scale.
        for r in getattr(self, "pinball_cross_query_refiners", []) or []:
            d[f"crossq.L{r.query_level}<-L{r.memory_level}.write"] = float(r.write_scale.detach())
        # Upper cross refiner (L3<-L2).
        uc = getattr(self, "pinball_upper_cross_refiner", None)
        if uc is not None:
            d["uppercross.L3<-L2.write"] = float(uc.write_scale.detach())
        # Self-refiners: per-level residual scales.
        for name, mod in (("upperrefine", getattr(self, "pinball_upper_refiner", None)),
                          ("toprefine", getattr(self, "pinball_top_refiner", None))):
            if mod is not None and hasattr(mod, "level_scales"):
                for k, s in mod.level_scales.items():
                    d[f"{name}.L{k}.scale"] = float(s.detach())
        # Energy-gated adaptive multirate: settings + observed settling behavior.
        # cycles.ema pinned at max_cycles = tau too low (never settles, cap does the work);
        # cycles.ema ~1 = tau too high (always settles immediately); in between = regulating.
        # energy.last_cycle is the mean energy at the final cycle — set tau just above the
        # settled plateau this converges to on a warm run.
        if bool(getattr(self, "pinball_multirate_adaptive", False)):
            d["mradapt.tau"] = float(getattr(self, "pinball_multirate_energy_tau", 0.0))
            d["mradapt.max_cycles"] = float(getattr(self, "pinball_multirate_max_cycles", 1))
            d["mradapt.cycles.last"] = float(getattr(self, "_last_mr_adaptive_cycles", 0))
            d["mradapt.cycles.ema"] = float(getattr(self, "_mr_adaptive_cycles_ema", 0.0))
            d["mradapt.cycles.max_seen"] = float(getattr(self, "_mr_adaptive_cycles_max", 0))
            st = getattr(self, "_last_pinball_multirate_stats", None) or {}
            if st.get("adaptive"):
                d["mradapt.frozen_frac"] = float(st.get("adaptive_frozen_frac", 0.0))
                energies = st.get("adaptive_energy") or []
                if energies:
                    d["mradapt.energy.first_cycle"] = float(energies[0])
                    d["mradapt.energy.last_cycle"] = float(energies[-1])
        # HQD v2 (xq_nominate) selection stats — LAST nomination round of the last forward:
        # edges = ephemeral L0->L0 edges injected; queries = L0 positions that received >=1
        # far edge (÷ batch for per-seq); dist = mean src->dst token distance of the fetch;
        # rounds = nomination rounds fired (xq_nominate_every). edges/queries ~= fetched
        # tokens per reading position.
        if bool(getattr(self, "xq_nominate_enable", False)):
            d["xq.edges"] = float(getattr(self, "_last_xq_nom_count", 0))
            d["xq.queries"] = float(getattr(self, "_last_xq_nom_queries", 0))
            d["xq.mean_dist"] = float(getattr(self, "_last_xq_nom_mean_dist", 0.0))
            d["xq.rounds"] = float(getattr(self, "_last_xq_nom_calls", 0))
            # relevance gate: w/b are the learned mapping score->gate; gate.mean is the mean
            # gate over queries that fetched (drifting DOWN = model learning to say "no-op").
            if bool(getattr(self, "xq_nominate_gate_enable", False)) and hasattr(self, "xq_gate_w"):
                d["xq.gate.w"] = float(self.xq_gate_w.detach())
                d["xq.gate.b"] = float(self.xq_gate_b.detach())
                d["xq.gate.mean"] = float(getattr(self, "_last_xq_gate_mean", 0.0))
        # Packed cross-level local attention: mixed-sequence length (all levels), query rows
        # taken from the packed call, and the slot window it ran with.
        _ps = getattr(self, "_last_local_pack_spec_stats", None)
        if bool(getattr(self, "local_pack_cross_level", False)) and _ps:
            d["pack.len"] = float(_ps.get("len", 0))
            d["pack.queries"] = float(_ps.get("queries", 0))
            d["pack.window"] = float(_ps.get("window", 0))
            if _ps.get("lane_len", 0):
                d["pack.lane_len"] = float(_ps.get("lane_len", 0))
                d["pack.lane_window"] = float(_ps.get("lane_window", 0))
        return d

    def _maybe_log_gate_monitor(self) -> None:
        if not bool(getattr(self, "pinball_monitor_gates", False)):
            return
        self._gate_monitor_calls = int(getattr(self, "_gate_monitor_calls", 0)) + 1
        every = int(getattr(self, "pinball_monitor_gates_every", 100))
        if self._gate_monitor_calls == 1 or (self._gate_monitor_calls % every) == 0:
            mon = self.gate_monitor()
            if mon:
                logger.info("[gate-monitor #%d] %s", self._gate_monitor_calls,
                            "  ".join(f"{k}={v:+.4f}" for k, v in mon.items()))

    def _filter_edges_by_ar_time(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not bool(self.hier_ar_enable):
            return edge_index, edge_type
        if edge_index is None or edge_index.numel() == 0:
            return edge_index, edge_type
        if node_ar_time is None or node_ar_time.numel() == 0:
            return edge_index, edge_type

        src = edge_index[0]
        dst = edge_index[1]
        src_time = node_ar_time[src]
        dst_time = node_ar_time[dst]
        if self.hier_ar_allow_same_time:
            keep = src_time <= dst_time
        else:
            keep = src_time < dst_time

        if not bool(self.l0_ar_enable):
            both_l0 = (node_level[src] == 0) & (node_level[dst] == 0)
            keep = keep | both_l0

        if bool(keep.all()):
            return edge_index, edge_type

        filtered_edge_index = edge_index[:, keep]
        filtered_edge_type = edge_type[keep] if edge_type is not None else None
        return filtered_edge_index, filtered_edge_type

    def _apply_level_ablation(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
        node_level: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Drop every edge incident to an ablated level, isolating those nodes so they
        contribute nothing to the L0 prediction (level-ablation study dial)."""
        if not self.ablate_levels or edge_index is None or edge_index.numel() == 0:
            return edge_index, edge_type
        abl = torch.tensor(self.ablate_levels, device=edge_index.device, dtype=node_level.dtype)
        src_abl = torch.isin(node_level[edge_index[0]], abl)
        dst_abl = torch.isin(node_level[edge_index[1]], abl)
        keep = ~(src_abl | dst_abl)
        if bool(keep.all()):
            return edge_index, edge_type
        return edge_index[:, keep], (edge_type[keep] if edge_type is not None else None)

    def report_level_connectivity(self, log: bool = True) -> Optional[Dict[str, Any]]:
        """Report per-level connectivity and inert levels for the most recent forward.

        Returns the report dict (see utils.level_report.compute_level_connectivity) or None
        if no graph has been built yet. Set ``log=True`` to also emit a formatted table.
        """
        from ..utils.level_report import compute_level_connectivity, format_level_connectivity
        topo = getattr(self, "_last_topology", None)
        if topo is None or topo[0] is None:
            if log:
                logger.warning("report_level_connectivity: no graph built yet (run a forward first).")
            return None
        edge_index, node_level = topo
        report = compute_level_connectivity(edge_index, node_level)
        if log:
            logger.info("%s", format_level_connectivity(report))
        return report

    def _augment_unified_graph_twin_shared_l3(
        self,
        unified_x: torch.Tensor,
        unified_node_level: torch.Tensor,
        unified_edge_index: torch.Tensor,
        unified_edge_type: torch.Tensor,
        level_offsets: List[int],
        unified_node_ar_time: Optional[torch.Tensor],
        level_graphs: List[Data],
        level_mappings: List[Tuple[Dict[int, List[int]], Dict[int, List[int]]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Tuple[int, int], torch.Tensor]:
        if len(level_graphs) < 4 or len(level_offsets) < 5:
            node_pos_local = torch.arange(unified_x.size(0), device=device, dtype=torch.long)
            node_branch = torch.zeros(unified_x.size(0), device=device, dtype=torch.long)
            return (
                unified_x,
                unified_node_level,
                unified_edge_index,
                unified_edge_type,
                unified_node_ar_time,
                node_pos_local,
                (0, 0),
                node_branch,
            )

        n0 = int(level_graphs[0].x.size(0))
        n1 = int(level_graphs[1].x.size(0))
        n2 = int(level_graphs[2].x.size(0))
        n3 = int(level_graphs[3].x.size(0))

        def _decoder_init_block(n: int) -> torch.Tensor:
            if n <= 0:
                return torch.empty((0, unified_x.size(1)), device=device, dtype=unified_x.dtype)
            use_mask = (
                getattr(self, "input_mode", "tokens") == "tokens"
                and getattr(self, "upper_init", "mask") in ("mask", "pooled")
                and getattr(self, "mask_token_id", None) is not None
            )
            if use_mask:
                mask_id = torch.tensor([int(self.mask_token_id)], device=device, dtype=torch.long)
                mask_vec = self.token_embedding(mask_id).to(dtype=unified_x.dtype).view(1, -1)
                return mask_vec.repeat(n, 1)
            return torch.zeros((n, unified_x.size(1)), device=device, dtype=unified_x.dtype)

        dec_l0_start = int(unified_x.size(0))
        dec_l1_start = dec_l0_start + n0
        dec_l2_start = dec_l1_start + n1
        dec_l0_end = dec_l0_start + n0

        dec_l0 = _decoder_init_block(n0)
        dec_l1 = _decoder_init_block(n1)
        dec_l2 = _decoder_init_block(n2)
        unified_x = torch.cat([unified_x, dec_l0, dec_l1, dec_l2], dim=0)

        node_branch = torch.cat(
            [
                torch.zeros((dec_l0_start,), dtype=torch.long, device=device),
                torch.ones((n0 + n1 + n2,), dtype=torch.long, device=device),
            ],
            dim=0,
        )

        dec_levels = torch.cat(
            [
                torch.zeros((n0,), dtype=torch.long, device=device),
                torch.ones((n1,), dtype=torch.long, device=device),
                torch.full((n2,), 2, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        unified_node_level = torch.cat([unified_node_level, dec_levels], dim=0)

        pos_enc = torch.cat(
            [
                torch.arange(n0, device=device, dtype=torch.long),
                torch.arange(n1, device=device, dtype=torch.long),
                torch.arange(n2, device=device, dtype=torch.long),
                torch.arange(n3, device=device, dtype=torch.long),
            ],
            dim=0,
        )
        pos_dec = torch.cat(
            [
                torch.arange(n0, device=device, dtype=torch.long),
                torch.arange(n1, device=device, dtype=torch.long),
                torch.arange(n2, device=device, dtype=torch.long),
            ],
            dim=0,
        )
        node_pos_local = torch.cat([pos_enc, pos_dec], dim=0)

        if unified_node_ar_time is not None and unified_node_ar_time.numel() >= (n0 + n1 + n2 + n3):
            enc_l0 = unified_node_ar_time[level_offsets[0]:level_offsets[1]]
            enc_l1 = unified_node_ar_time[level_offsets[1]:level_offsets[2]]
            enc_l2 = unified_node_ar_time[level_offsets[2]:level_offsets[3]]
            unified_node_ar_time = torch.cat([unified_node_ar_time, enc_l0, enc_l1, enc_l2], dim=0)

        extra_edges: List[torch.Tensor] = []
        extra_types: List[torch.Tensor] = []
        dec_offsets = {0: dec_l0_start, 1: dec_l1_start, 2: dec_l2_start}

        for lvl in (0, 1, 2):
            g_lvl = level_graphs[lvl]
            if hasattr(g_lvl, "edge_index") and g_lvl.edge_index is not None and g_lvl.edge_index.numel() > 0:
                ei = g_lvl.edge_index.to(device) + int(dec_offsets[lvl])
                extra_edges.append(ei)
                extra_types.append(torch.full((ei.size(1),), lvl, dtype=torch.long, device=device))

        for lvl in (0, 1):
            if lvl >= len(level_mappings):
                continue
            lower_to_higher, _ = level_mappings[lvl]
            cross_pairs = []
            lo = int(dec_offsets[lvl])
            hi = int(dec_offsets[lvl + 1])
            for low_idx, high_indices in lower_to_higher.items():
                for high_idx in high_indices:
                    cross_pairs.append([int(low_idx) + lo, int(high_idx) + hi])
            if cross_pairs:
                cross = torch.tensor(cross_pairs, dtype=torch.long, device=device).t()
                rev = torch.stack([cross[1], cross[0]], dim=0)
                extra_edges.extend([cross, rev])
                extra_types.extend([
                    torch.full((cross.size(1),), 4 + 2 * lvl, dtype=torch.long, device=device),
                    torch.full((rev.size(1),), 5 + 2 * lvl, dtype=torch.long, device=device),
                ])

        if len(level_mappings) >= 3:
            lower_to_higher, _ = level_mappings[2]
            lo = int(dec_offsets[2])
            hi = int(level_offsets[3])
            cross_pairs = []
            for low_idx, high_indices in lower_to_higher.items():
                for high_idx in high_indices:
                    cross_pairs.append([int(low_idx) + lo, int(high_idx) + hi])
            if cross_pairs:
                cross = torch.tensor(cross_pairs, dtype=torch.long, device=device).t()
                if bool(getattr(self, "autoenc_coupled_feedback", True)):
                    rev = torch.stack([cross[1], cross[0]], dim=0)
                    extra_edges.extend([cross, rev])
                    extra_types.extend([
                        torch.full((cross.size(1),), 8, dtype=torch.long, device=device),
                        torch.full((rev.size(1),), 9, dtype=torch.long, device=device),
                    ])
                else:
                    rev = torch.stack([cross[1], cross[0]], dim=0)
                    extra_edges.append(rev)
                    extra_types.append(torch.full((rev.size(1),), 9, dtype=torch.long, device=device))

        if extra_edges:
            add_ei = torch.cat(extra_edges, dim=1)
            add_et = torch.cat(extra_types, dim=0)
            unified_edge_index = torch.cat([unified_edge_index, add_ei], dim=1)
            unified_edge_type = torch.cat([unified_edge_type, add_et], dim=0)

        return (
            unified_x,
            unified_node_level,
            unified_edge_index,
            unified_edge_type,
            unified_node_ar_time,
            node_pos_local,
            (dec_l0_start, dec_l0_end),
            node_branch,
        )

    def _build_unified_graph(self, level_graphs, level_mappings):
        # --- Paste your original _build_unified_graph code here ---
        # (Using the version from message #64 that includes edge_type and optional edge_attr generation)
        device = level_graphs[0].x.device
        
        seq_len = level_graphs[0].num_nodes
        graph_cache_key = self._unified_graph_cache_key(seq_len)
        twin_mode = str(getattr(self, "autoenc_graph_mode", "off")).lower() == "twin_shared_l3"

        if (
            not twin_mode
            and
            self._cached_unified_graph is not None
            and self._cached_seq_len == seq_len
            and self._cached_unified_graph_key == graph_cache_key
        ):
            skeleton = self._cached_unified_graph
            unified_x = torch.cat([g.x for g in level_graphs], dim=0)
            unified_node_level = skeleton.node_level
            unified_edge_index = skeleton.edge_index
            unified_edge_type = skeleton.edge_type
            level_offsets = skeleton.level_offsets
            unified_node_ar_time = getattr(skeleton, "node_ar_time", None)
            level_grid_shapes = list(getattr(skeleton, "level_grid_shapes", []))
            if len(level_grid_shapes) != len(level_graphs):
                level_grid_shapes = []
                for graph in level_graphs:
                    gs = getattr(graph, "grid_shape", None)
                    if gs is not None and len(gs) == 2:
                        gh = int(gs[0])
                        gw = int(gs[1])
                        level_grid_shapes.append((gh, gw) if gh > 0 and gw > 0 else None)
                    else:
                        level_grid_shapes.append(None)
            node_pos_local = getattr(skeleton, "node_pos_local", None)
            if node_pos_local is not None:
                node_pos_local = node_pos_local.to(device=device, dtype=torch.long)
            if node_pos_local is None or int(node_pos_local.numel()) != int(unified_x.size(0)):
                node_pos_local = _build_node_pos_local_from_offsets(
                    total_nodes=int(unified_x.size(0)),
                    level_offsets=level_offsets,
                    device=device,
                )
            ae_decoder_l0_slice = (0, 0)
            node_branch = None
        else:
            all_features = []
            node_levels = []
            level_offsets = [0]
            level_grid_shapes = []
            for level_idx, graph in enumerate(level_graphs):
                all_features.append(graph.x)
                node_levels.append(
                    getattr(
                        graph,
                        'node_level',
                        torch.full((graph.x.size(0),), level_idx, dtype=torch.long, device=device),
                    )
                )
                gs = getattr(graph, "grid_shape", None)
                if gs is not None and len(gs) == 2:
                    gh = int(gs[0])
                    gw = int(gs[1])
                    if gh > 0 and gw > 0 and gh * gw == int(graph.x.size(0)):
                        level_grid_shapes.append((gh, gw))
                    else:
                        level_grid_shapes.append(None)
                else:
                    level_grid_shapes.append(None)
                level_offsets.append(level_offsets[-1] + graph.x.size(0))

            unified_x = torch.cat(all_features, dim=0)
            unified_node_level = torch.cat(node_levels, dim=0)

            unified_edges = []
            edge_types = []
            for level_idx, graph in enumerate(level_graphs):
                if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                    offset = level_offsets[level_idx]
                    level_edges = graph.edge_index + offset
                    unified_edges.append(level_edges)
                    edge_types.append(
                        torch.full((level_edges.size(1),), level_idx, dtype=torch.long, device=device)
                    )

            for level_idx in range(len(level_graphs) - 1):
                if level_idx >= len(level_mappings):
                    continue
                lower_to_higher, higher_to_lower = level_mappings[level_idx]
                cross_edges = []
                lower_offset = level_offsets[level_idx]
                higher_offset = level_offsets[level_idx + 1]
                num_lower_nodes_total = level_offsets[level_idx + 1]
                num_higher_nodes_total = level_offsets[level_idx + 2]
                for lower_idx, higher_indices in lower_to_higher.items():
                    for higher_idx in higher_indices:
                        abs_lower_idx = lower_idx + lower_offset
                        abs_higher_idx = higher_idx + higher_offset
                        if (
                            abs_lower_idx < num_lower_nodes_total
                            and abs_higher_idx < num_higher_nodes_total
                        ):
                            cross_edges.append([abs_lower_idx, abs_higher_idx])
                            cross_edges.append([abs_higher_idx, abs_lower_idx])

                if cross_edges:
                    cross_edge_index = torch.tensor(cross_edges, dtype=torch.long, device=device).t()
                    unified_edges.append(cross_edge_index)
                    num_cross_edges = cross_edge_index.size(1) // 2
                    edge_types.append(
                        torch.full((num_cross_edges,), 4 + 2 * level_idx, dtype=torch.long, device=device)
                    )
                    edge_types.append(
                        torch.full((num_cross_edges,), 5 + 2 * level_idx, dtype=torch.long, device=device)
                    )

            unified_edge_index = (
                torch.cat(unified_edges, dim=1)
                if unified_edges
                else torch.zeros((2, 0), dtype=torch.long, device=device)
            )
            unified_edge_type = (
                torch.cat(edge_types, dim=0)
                if edge_types
                else torch.zeros((0,), dtype=torch.long, device=device)
            )

            unified_node_ar_time = self._compute_node_ar_time(
                level_offsets=level_offsets,
                level_mappings=level_mappings,
                device=device,
            )

            node_pos_local = None
            ae_decoder_l0_slice = (0, 0)
            node_branch = None
            if twin_mode:
                (
                    unified_x,
                    unified_node_level,
                    unified_edge_index,
                    unified_edge_type,
                    unified_node_ar_time,
                    node_pos_local,
                    ae_decoder_l0_slice,
                    node_branch,
                ) = self._augment_unified_graph_twin_shared_l3(
                    unified_x=unified_x,
                    unified_node_level=unified_node_level,
                    unified_edge_index=unified_edge_index,
                    unified_edge_type=unified_edge_type,
                    level_offsets=level_offsets,
                    unified_node_ar_time=unified_node_ar_time,
                    level_graphs=level_graphs,
                    level_mappings=level_mappings,
                    device=device,
                )

            if self.hier_ar_enable:
                unified_edge_index, unified_edge_type = self._filter_edges_by_ar_time(
                    edge_index=unified_edge_index,
                    edge_type=unified_edge_type,
                    node_level=unified_node_level,
                    node_ar_time=unified_node_ar_time,
                )

            if not twin_mode:
                if node_pos_local is None:
                    node_pos_local = _build_node_pos_local_from_offsets(
                        total_nodes=int(unified_x.size(0)),
                        level_offsets=level_offsets,
                        device=device,
                    )
                self._cached_unified_graph = Data(
                    edge_index=unified_edge_index.clone(),
                    edge_type=unified_edge_type.clone(),
                    node_level=unified_node_level.clone(),
                    level_offsets=level_offsets,
                    node_ar_time=unified_node_ar_time.clone() if unified_node_ar_time is not None else None,
                )
                self._cached_unified_graph.node_pos_local = node_pos_local.clone()
                self._cached_unified_graph.level_grid_shapes = list(level_grid_shapes)
                self._cached_unified_graph_key = graph_cache_key

        if node_pos_local is None:
            node_pos_local = _build_node_pos_local_from_offsets(
                total_nodes=int(unified_x.size(0)),
                level_offsets=level_offsets,
                device=device,
            )

        # ---- NEW: optional L0↤all-levels ancestor edges ----
        if (not twin_mode) and getattr(self, "enable_l0_parent_edges", False):
            edge_index, edge_type = self._add_l0_parent_edges(
                edge_index=unified_edge_index,
                edge_type=unified_edge_type,
                level_mappings=level_mappings,
                level_offsets=level_offsets,
                num_levels=len(level_offsets),
                device=unified_edge_index.device,
                l0_edge_type_id=self.l0_parent_edge_type_id,
            )
            unified_edge_index = edge_index
            unified_edge_type = edge_type

        if (not twin_mode) and bool(getattr(self, "hier_ar_enable", False)):
            past_bridge_pairs: List[Tuple[int, int]] = []
            if bool(getattr(self, "ensure_past_hier_edges_all_levels", False)):
                num_past_levels = len(level_offsets) - 1
                for parent_lvl in range(1, num_past_levels):
                    past_bridge_pairs.append((parent_lvl - 1, parent_lvl))
                if getattr(self, "enable_l0_parent_edges", False):
                    min_parent_level = max(1, int(getattr(self, "l0_parent_edge_min_level", 2)))
                    max_parent_level_raw = getattr(self, "l0_parent_edge_max_level", 3)
                    if max_parent_level_raw is None:
                        max_parent_level = num_past_levels - 1
                    else:
                        max_parent_level = max(min_parent_level, int(max_parent_level_raw))
                    max_parent_level = min(max_parent_level, num_past_levels - 1)
                    for parent_lvl in range(max(2, min_parent_level), max_parent_level + 1):
                        past_bridge_pairs.append((0, parent_lvl))
            elif bool(getattr(self, "ensure_l0_past_l1_edges", False)):
                past_bridge_pairs.append((0, 1))
            seen_past_pairs = set()
            for child_level, parent_level in past_bridge_pairs:
                pair_key = (int(child_level), int(parent_level))
                if pair_key in seen_past_pairs:
                    continue
                seen_past_pairs.add(pair_key)
                bridge_edge_type_id = None
                if pair_key == (0, 1):
                    bridge_edge_type_id = getattr(self, "l0_past_l1_edge_type_id", None)
                elif pair_key[0] == 0 and pair_key[1] > 1:
                    bridge_edge_type_id = getattr(self, "l0_parent_edge_type_id", None)
                unified_edge_index, unified_edge_type = self._add_past_bridge_edges(
                    edge_index=unified_edge_index,
                    edge_type=unified_edge_type,
                    level_offsets=level_offsets,
                    node_ar_time=unified_node_ar_time,
                    child_level=pair_key[0],
                    parent_level=pair_key[1],
                    edge_type_id=bridge_edge_type_id,
                )

            if bool(getattr(self, "ensure_l0_past_parent_edges", False)):
                unified_edge_index, unified_edge_type = self._add_l0_staggered_past_parent_edges(
                    edge_index=unified_edge_index,
                    edge_type=unified_edge_type,
                    level_offsets=level_offsets,
                    node_ar_time=unified_node_ar_time,
                )

        if self.hier_ar_enable:
            unified_edge_index, unified_edge_type = self._filter_edges_by_ar_time(
                edge_index=unified_edge_index,
                edge_type=unified_edge_type,
                node_level=unified_node_level,
                node_ar_time=unified_node_ar_time,
            )

        unified_edge_index, unified_edge_type = self._dedup_edges_keep_first(
            edge_index=unified_edge_index,
            edge_type=unified_edge_type,
            num_nodes=int(unified_x.size(0)),
        )

        if (
            twin_mode
            and bool(getattr(self, "hier_ar_enable", False))
            and unified_edge_index is not None
            and unified_edge_index.numel() > 0
            and unified_node_ar_time is not None
            and unified_node_ar_time.numel() >= int(unified_x.size(0))
        ):
            src = unified_edge_index[0]
            dst = unified_edge_index[1]
            src_time = unified_node_ar_time[src]
            dst_time = unified_node_ar_time[dst]
            if bool(getattr(self, "hier_ar_allow_same_time", False)):
                causal_ok = src_time <= dst_time
            else:
                causal_ok = src_time < dst_time
            non_causal = int((~causal_ok).sum().item())
            if non_causal > 0:
                logger.warning(
                    "Twin shared-L3 graph has %d non-causal edges after AR filtering.",
                    non_causal,
                )
            else:
                logger.debug("Twin shared-L3 graph causal audit passed (0 violations).")

        if self.ablate_levels:
            unified_edge_index, unified_edge_type = self._apply_level_ablation(
                edge_index=unified_edge_index,
                edge_type=unified_edge_type,
                node_level=unified_node_level,
            )

        unified_graph = Data(
            x=unified_x,
            edge_index=unified_edge_index,
            edge_type=unified_edge_type,
            node_level=unified_node_level,
            level_offsets=level_offsets,
            node_ar_time=unified_node_ar_time,
        )
        unified_graph.node_pos_local = node_pos_local
        unified_graph.level_grid_shapes = list(level_grid_shapes)

        # Cache lightweight topology (int tensors, no grad) for the level connectivity reporter.
        self._last_topology = (
            unified_edge_index.detach() if unified_edge_index is not None else None,
            unified_node_level.detach(),
        )
        if getattr(self, "report_level_connectivity_enabled", False) and not self._level_report_logged:
            self._level_report_logged = True
            try:
                self.report_level_connectivity(log=True)
            except Exception as exc:  # never let reporting break a forward
                logger.warning("Level connectivity report failed: %r", exc)
        if twin_mode:
            unified_graph.ae_decoder_l0_slice = ae_decoder_l0_slice
            if node_branch is not None:
                unified_graph.node_branch = node_branch

        level_sizes = [int(g.num_nodes) for g in level_graphs]
        level_edge_counts = [int(g.edge_index.size(1)) if getattr(g, "edge_index", None) is not None else 0 for g in level_graphs]
        total_level_edges = int(sum(level_edge_counts))
        if self.verbose:
            logger.info(
                "[HFGAT:UNIFIED] built graph levels=%d sizes=%s intra_edges=%s total_nodes=%d total_edges=%d cross_edges=%d grid_shapes=%s twin_mode=%s ar=%s",
                int(len(level_graphs)),
                level_sizes,
                level_edge_counts,
                int(unified_graph.num_nodes),
                int(unified_graph.num_edges),
                int(max(0, int(unified_graph.num_edges) - total_level_edges)),
                level_grid_shapes,
                bool(twin_mode),
                bool(self.hier_ar_enable),
            )

        # --- Generate edge_attr AFTER building base graph ---
        if (
            self.use_edge_attr
            and hasattr(self, 'edge_feature_generator')
            and self.edge_feature_generator is not None
            and unified_graph.num_edges > 0
        ):
            logger.debug(
                f"Checking unified graph integrity: Nodes={unified_graph.num_nodes}, Edges={unified_graph.num_edges}"
            )
            valid_graph_for_edge_attr = True
            max_node_idx_in_edges = unified_graph.edge_index.max()
            if max_node_idx_in_edges >= unified_graph.num_nodes:
                logger.error(
                    f"!!! Unified Graph Integrity Error (Edge Index): Max index {max_node_idx_in_edges.item()} >= num_nodes {unified_graph.num_nodes}"
                )
                valid_graph_for_edge_attr = False

            if hasattr(unified_graph, 'edge_type'):
                max_edge_type = unified_graph.edge_type.max()
                num_edge_types_expected = getattr(
                    self.edge_feature_generator, 'num_edge_types', 0
                )
                if max_edge_type >= num_edge_types_expected:
                    logger.error(
                        f"!!! Unified Graph Integrity Error (Edge Type): Max type {max_edge_type.item()} >= num_edge_types {num_edge_types_expected}"
                    )
                    valid_graph_for_edge_attr = False
            else:
                logger.error(
                    "!!! Unified Graph Integrity Error: edge_type attribute missing but needed for edge_attr."
                )
                valid_graph_for_edge_attr = False

            if valid_graph_for_edge_attr:
                logger.debug("Generating edge attributes for unified graph...")
                try:
                    self.edge_feature_generator.to(unified_graph.x.device)
                    #self.edge_feature_generator.to("cpu")  # Edge generator on CPU
                    unified_graph.edge_attr = self.edge_feature_generator(
                        unified_graph.x, unified_graph.edge_index, unified_graph.edge_type
                    )
                    logger.debug("Edge attributes generated successfully.")
                except Exception as e:
                    logger.warning(
                        f"Failed to generate edge attributes for unified graph: {e}",
                        exc_info=True,
                    )
                    if hasattr(unified_graph, 'edge_attr'):
                        del unified_graph.edge_attr
            else:
                logger.warning(
                    "Skipping edge attribute generation due to graph integrity issues."
                )

        return unified_graph


    # OLD
    # def _build_unified_graph(self, level_graphs, level_mappings):
    #     """
    #     Build unified graph connecting all processed levels.
        
    #     Args:
    #         level_graphs: List of processed level graphs
    #         level_mappings: List of level mappings (lower_to_higher, higher_to_lower)
            
    #     Returns:
    #         unified_graph: PyG Data object with unified graph
    #     """
    #     device = level_graphs[0].x.device
        
    #     # Initialize unified graph with features from all levels
    #     all_features = []
    #     node_levels = []
    #     level_offsets = [0]
        
    #     # Add features from each level
    #     for level_idx, graph in enumerate(level_graphs):
    #         all_features.append(graph.x)
    #         node_levels.append(graph.node_level)
    #         level_offsets.append(level_offsets[-1] + graph.x.size(0))
        
    #     # Combine features and levels
    #     unified_x = torch.cat(all_features, dim=0)
    #     unified_node_level = torch.cat(node_levels, dim=0)
        
    #     # Initialize edges and edge types
    #     unified_edges = []
    #     edge_types = []
        
    #     # Add within-level edges from each level
    #     for level_idx, graph in enumerate(level_graphs):
    #         # Adjust indices for unified graph
    #         offset = level_offsets[level_idx]
    #         level_edges = graph.edge_index.clone()
    #         level_edges += offset
            
    #         unified_edges.append(level_edges)
    #         edge_types.append(torch.full((level_edges.size(1),), level_idx, dtype=torch.long, device=device))
        
    #     # Add cross-level edges between adjacent levels
    #     for level_idx in range(len(level_graphs) - 1):
    #         # Get mappings
    #         lower_to_higher, higher_to_lower = level_mappings[level_idx]
            
    #         # Create edges from lower to higher and higher to lower
    #         cross_edges = []
            
    #         # Lower to higher connections
    #         lower_offset = level_offsets[level_idx]
    #         higher_offset = level_offsets[level_idx + 1]
            
    #         for lower_idx, higher_indices in lower_to_higher.items():
    #             for higher_idx in higher_indices:
    #                 # Lower -> Higher
    #                 cross_edges.append([lower_idx + lower_offset, higher_idx + higher_offset])
    #                 # Higher -> Lower
    #                 cross_edges.append([higher_idx + higher_offset, lower_idx + lower_offset])
            
    #         if cross_edges:
    #             cross_edge_index = torch.tensor(cross_edges, dtype=torch.long, device=device).t()
    #             unified_edges.append(cross_edge_index)
    #             edge_types.append(torch.full((cross_edge_index.size(1),), len(level_graphs) + level_idx, dtype=torch.long, device=device))
        
    #     # Combine all edges and edge types
    #     unified_edge_index = torch.cat(unified_edges, dim=1) if unified_edges else torch.zeros((2, 0), dtype=torch.long, device=device)
    #     unified_edge_type = torch.cat(edge_types, dim=0) if edge_types else torch.zeros((0,), dtype=torch.long, device=device)
        
    #     # Create unified graph
    #     unified_graph = Data(
    #         x=unified_x,
    #         edge_index=unified_edge_index,
    #         edge_type=unified_edge_type,
    #         node_level=unified_node_level,
    #         level_offsets=level_offsets
    #     )
        
    #     return unified_graph



















    def _compute_hier_aux_pair_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self, "hier_aux_unit_norm", False)):
            pred = F.normalize(pred, p=2.0, dim=-1, eps=1e-8)
            target = F.normalize(target, p=2.0, dim=-1, eps=1e-8)

        loss_mode = str(getattr(self, "hier_aux_loss_mode", "mse")).lower()
        if loss_mode in {"mse_norm", "nmse"}:
            sq_err = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
            denom = target.pow(2).mean(dim=-1).clamp_min(1e-8)
            return (sq_err / denom).mean()
        if loss_mode == "cosine":
            cos = F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
            return ((1.0 - cos) * 0.5).mean()
        return F.mse_loss(pred, target, reduction="mean")

    def _hier_aux_should_detach(self, low_level: int) -> bool:
        """Whether the aux reconstruction TARGET for this level is stop-grad. Default True
        (JEPA anti-collapse). `hier_aux_link_l0_target` un-detaches ONLY the L0 target so the
        aux gradient flows into the L0 (Hyena-pooled) features — a direct link of the aux
        objective to the Hyena pooling, while L1/L2/L3 stay detached to limit collapse risk."""
        if int(low_level) == 0 and bool(getattr(self, "hier_aux_link_l0_target", False)):
            return False
        return bool(getattr(self, "hier_aux_detach_target", True))

    def _compute_hier_aux_pair_loss_jepa(
        self,
        g,
        low_level: int,
        high_level: int,
        predictor_key: str,
    ) -> torch.Tensor:
        x = g.x
        node_level = getattr(g, "node_level", None)
        edge_index = getattr(g, "edge_index", None)
        device = x.device

        if node_level is None or edge_index is None:
            return torch.zeros((), device=device, dtype=x.dtype)

        predictors = getattr(self, "hier_aux_pair_predictors", None)
        if predictors is None or predictor_key not in predictors:
            return torch.zeros((), device=device, dtype=x.dtype)

        src, dst = edge_index
        mask_lh = (node_level[src] == int(low_level)) & (node_level[dst] == int(high_level))
        mask_hl = (node_level[src] == int(high_level)) & (node_level[dst] == int(low_level))

        child_idx = torch.cat([src[mask_lh], dst[mask_hl]], dim=0)
        parent_idx = torch.cat([dst[mask_lh], src[mask_hl]], dim=0)
        node_ar_time = getattr(g, "node_ar_time", None)
        if bool(getattr(self, "hier_ar_enable", False)) and node_ar_time is not None and node_ar_time.numel() > 0:
            parent_time = node_ar_time[parent_idx]
            child_time = node_ar_time[child_idx]
            if bool(getattr(self, "hier_aux_ar_strict", True)):
                keep_pair = parent_time < child_time
            else:
                keep_pair = parent_time <= child_time
            child_idx = child_idx[keep_pair]
            parent_idx = parent_idx[keep_pair]
        if child_idx.numel() == 0:
            return torch.zeros((), device=device, dtype=x.dtype)

        n_nodes, hidden = x.shape
        agg = torch.zeros(n_nodes, hidden, device=device, dtype=x.dtype)
        cnt = torch.zeros(n_nodes, 1, device=device, dtype=x.dtype)
        agg.index_add_(0, child_idx, x[parent_idx])
        cnt.index_add_(0, child_idx, torch.ones(child_idx.size(0), 1, device=device, dtype=x.dtype))

        low_mask = (node_level == int(low_level))
        matched_mask = cnt.squeeze(-1) > 0
        low_mask = low_mask & matched_mask
        if not bool(low_mask.any()):
            return torch.zeros((), device=device, dtype=x.dtype)

        pred_context = agg[low_mask] / cnt[low_mask].clamp_min(1.0)
        predictor = predictors[predictor_key]
        pred = predictor(pred_context)
        target = x[low_mask]
        if self._hier_aux_should_detach(low_level):
            target = target.detach()

        return self._compute_hier_aux_pair_loss(pred, target)

    def _compute_hier_aux_pair_loss_mean_causal(
        self,
        g,
        low_level: int,
        high_level: int,
    ) -> torch.Tensor:
        x = g.x
        node_level = getattr(g, "node_level", None)
        edge_index = getattr(g, "edge_index", None)
        device = x.device

        if node_level is None or edge_index is None:
            return torch.zeros((), device=device, dtype=x.dtype)

        src, dst = edge_index
        mask_lh = (node_level[src] == int(low_level)) & (node_level[dst] == int(high_level))
        mask_hl = (node_level[src] == int(high_level)) & (node_level[dst] == int(low_level))

        child_idx = torch.cat([src[mask_lh], dst[mask_hl]], dim=0)
        parent_idx = torch.cat([dst[mask_lh], src[mask_hl]], dim=0)
        node_ar_time = getattr(g, "node_ar_time", None)
        if bool(getattr(self, "hier_ar_enable", False)) and node_ar_time is not None and node_ar_time.numel() > 0:
            parent_time = node_ar_time[parent_idx]
            child_time = node_ar_time[child_idx]
            if bool(getattr(self, "hier_aux_ar_strict", True)):
                keep_pair = parent_time < child_time
            else:
                keep_pair = parent_time <= child_time
            child_idx = child_idx[keep_pair]
            parent_idx = parent_idx[keep_pair]
        if child_idx.numel() == 0:
            return torch.zeros((), device=device, dtype=x.dtype)

        n_nodes, hidden = x.shape
        agg = torch.zeros(n_nodes, hidden, device=device, dtype=x.dtype)
        cnt = torch.zeros(n_nodes, 1, device=device, dtype=x.dtype)
        agg.index_add_(0, child_idx, x[parent_idx])
        cnt.index_add_(0, child_idx, torch.ones(child_idx.size(0), 1, device=device, dtype=x.dtype))

        low_mask = (node_level == int(low_level))
        matched_mask = cnt.squeeze(-1) > 0
        low_mask = low_mask & matched_mask
        if not bool(low_mask.any()):
            return torch.zeros((), device=device, dtype=x.dtype)

        pred = agg[low_mask] / cnt[low_mask].clamp_min(1.0)
        target = x[low_mask]
        if self._hier_aux_should_detach(low_level):
            target = target.detach()

        return self._compute_hier_aux_pair_loss(pred, target)

    def _compute_hierarchy_aux_loss_runtime(self, g) -> torch.Tensor:
        mode = str(getattr(self, "hier_aux_mode", "mean_mse")).lower()
        if mode == "mean_mse":
            if not bool(getattr(self, "hier_ar_enable", False)):
                self._last_hier_aux_pair_losses = {}
                return compute_hierarchy_aux_loss(
                    g,
                    detach_target=bool(getattr(self, "hier_aux_detach_target", True)),
                    link_low0=bool(getattr(self, "hier_aux_link_l0_target", False)),
                    w_l2_from_l3=float(getattr(self, "hier_aux_w_l2_from_l3", 1.0)),
                    w_l1_from_l2=float(getattr(self, "hier_aux_w_l1_from_l2", 1.0)),
                    w_l0_from_l1=float(getattr(self, "hier_aux_w_l0_from_l1", 1.0)),
                    w_l0_from_l3=float(getattr(self, "hier_aux_w_l0_from_l3", 0.25)),
                    loss_mode=str(getattr(self, "hier_aux_loss_mode", "mse")),
                )

            pair_defs = [
                ("l2_from_l3", 2, 3, float(getattr(self, "hier_aux_w_l2_from_l3", 1.0))),
                ("l1_from_l2", 1, 2, float(getattr(self, "hier_aux_w_l1_from_l2", 1.0))),
                ("l0_from_l1", 0, 1, float(getattr(self, "hier_aux_w_l0_from_l1", 1.0))),
                ("l0_from_l3", 0, 3, float(getattr(self, "hier_aux_w_l0_from_l3", 0.25))),
            ]

            loss_total = torch.zeros((), device=g.x.device, dtype=g.x.dtype)
            pair_log: Dict[str, float] = {}
            for pair_key, low_lvl, high_lvl, weight in pair_defs:
                if float(weight) == 0.0:
                    continue
                if (
                    bool(getattr(self, "hier_ar_enable", False))
                    and bool(getattr(self, "hier_aux_ar_disable_l0_from_l3", True))
                    and str(pair_key) == "l0_from_l3"
                ):
                    if not bool(getattr(self, "_hier_aux_l0_from_l3_disabled_logged", False)):
                        logger.info("Hierarchy aux (AR strict): disabling l0_from_l3 pair to avoid future-context contamination")
                        self._hier_aux_l0_from_l3_disabled_logged = True
                    continue
                pair_loss = self._compute_hier_aux_pair_loss_mean_causal(
                    g,
                    low_level=int(low_lvl),
                    high_level=int(high_lvl),
                )
                loss_total = loss_total + (float(weight) * pair_loss)
                pair_log[pair_key] = float(pair_loss.detach().item())

            self._last_hier_aux_pair_losses = pair_log
            return loss_total

        pair_defs = [
            ("l2_from_l3", 2, 3, float(getattr(self, "hier_aux_w_l2_from_l3", 1.0))),
            ("l1_from_l2", 1, 2, float(getattr(self, "hier_aux_w_l1_from_l2", 1.0))),
            ("l0_from_l1", 0, 1, float(getattr(self, "hier_aux_w_l0_from_l1", 1.0))),
            ("l0_from_l3", 0, 3, float(getattr(self, "hier_aux_w_l0_from_l3", 0.25))),
        ]

        loss_total = torch.zeros((), device=g.x.device, dtype=g.x.dtype)
        pair_log: Dict[str, float] = {}
        for pair_key, low_lvl, high_lvl, weight in pair_defs:
            if float(weight) == 0.0:
                continue
            if (
                bool(getattr(self, "hier_ar_enable", False))
                and bool(getattr(self, "hier_aux_ar_disable_l0_from_l3", True))
                and str(pair_key) == "l0_from_l3"
            ):
                if not bool(getattr(self, "_hier_aux_l0_from_l3_disabled_logged", False)):
                    logger.info("Hierarchy aux (AR strict): disabling l0_from_l3 pair to avoid future-context contamination")
                    self._hier_aux_l0_from_l3_disabled_logged = True
                continue
            pair_loss = self._compute_hier_aux_pair_loss_jepa(
                g,
                low_level=int(low_lvl),
                high_level=int(high_lvl),
                predictor_key=str(pair_key),
            )
            loss_total = loss_total + (float(weight) * pair_loss)
            pair_log[pair_key] = float(pair_loss.detach().item())

        self._last_hier_aux_pair_losses = pair_log
        return loss_total

    def _queue_storage_dtype(self, vectors: torch.Tensor) -> torch.dtype:
        return torch.bfloat16 if vectors.dtype == torch.bfloat16 else torch.float32


    def _reveal_propagate_state(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        steps: int,
    ) -> torch.Tensor:
        if steps <= 0 or edge_index is None or edge_index.numel() == 0:
            return x

        src = edge_index[0]
        dst = edge_index[1]
        x_cur = x
        for _ in range(steps):
            agg = torch.zeros_like(x_cur)
            agg.index_add_(0, dst, x_cur[src])
            deg = torch.zeros(x_cur.size(0), device=x_cur.device, dtype=x_cur.dtype)
            deg.index_add_(0, dst, torch.ones(dst.size(0), device=x_cur.device, dtype=x_cur.dtype))
            nbr_mean = agg / deg.clamp_min(1.0).unsqueeze(-1)
            x_cur = 0.5 * x_cur + 0.5 * nbr_mean
        return x_cur

    def _reveal_transformer_propagate(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        node_level: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        steps: int,
        mode: str,
    ) -> torch.Tensor:
        if steps <= 0:
            return x
        if edge_index is None or edge_index.numel() == 0:
            return x
        if node_level is None or positions is None:
            return x

        if self.share_transformers:
            transformers_to_use = [m for mods in self.level_transformers for m in mods]
        elif self.refinement_transformers is not None:
            transformers_to_use = list(self.refinement_transformers)
        else:
            transformers_to_use = []

        if not transformers_to_use:
            return x

        x_cur = x
        edge_attr_cur = edge_attr
        use_edge_attr = self.use_edge_attr and edge_attr_cur is not None

        for _ in range(int(steps)):
            if mode == "transformer_layer":
                layers = [transformers_to_use[-1]]
            else:
                layers = transformers_to_use

            for layer in layers:
                x_cur, edge_attr_new = layer(
                    x_cur,
                    edge_index,
                    node_level,
                    level_offsets=None,
                    positions=positions,
                    edge_attr=edge_attr_cur if use_edge_attr else None,
                )
                if edge_attr_new is not None:
                    edge_attr_cur = edge_attr_new
                    use_edge_attr = self.use_edge_attr

            x_cur = self.layer_norm(x_cur)

        return x_cur

    def _apply_transformer_sampled(
        self,
        transformer,
        x_global: torch.Tensor,        # [N, H] on device
        data_cpu: Data,                # Graph structure on CPU
        pos_global: torch.Tensor,      # [N] positions on CPU
        edge_attr_global: Optional[torch.Tensor] = None,  # [E] on device
        input_nodes: Optional[torch.Tensor] = None,       # [K] on device or CPU
        nodes_per_sample: Optional[int] = None,
    ):
        """
        Apply one transformer layer using NeighborLoader mini-batches.
        Resamples neighbors fresh for this layer pass.
        """
        device = x_global.device
        N = x_global.size(0)

        if input_nodes is not None:
            if input_nodes.numel() == 0:
                return x_global, edge_attr_global
            input_nodes_cpu = input_nodes.detach().to("cpu", dtype=torch.long)
            input_nodes_cpu = input_nodes_cpu[(input_nodes_cpu >= 0) & (input_nodes_cpu < N)]
            if input_nodes_cpu.numel() == 0:
                return x_global, edge_attr_global
        else:
            input_nodes_cpu = torch.arange(N, dtype=torch.long)
        
        # Create loader - resamples neighbors for this layer
        loader = NeighborLoader(
            data_cpu,
            num_neighbors=self.num_neighbors,
            input_nodes=input_nodes_cpu,
            batch_size=self.sampling_batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        for batch in loader:
            n_id = batch.n_id          # Global indices in subgraph
            if n_id.numel() == 0:
                continue
            if bool(getattr(self, "sampler_debug_checks", False)) and (int(n_id.min().item()) < 0 or int(n_id.max().item()) >= N):
                raise RuntimeError(
                    "NeighborLoader returned invalid n_id: "
                    f"min={int(n_id.min().item())} max={int(n_id.max().item())} num_nodes={N}"
                )
            num_seed = batch.batch_size
            seed_global = n_id[:num_seed].to(device)
            e_id = getattr(batch, "e_id", None)
            e_id_device = None
            if self.use_edge_attr and edge_attr_global is not None:
                self._last_sampled_edge_attr_batches = int(getattr(self, "_last_sampled_edge_attr_batches", 0)) + 1
                if e_id is None:
                    self._last_sampled_edge_attr_missing_eid = int(getattr(self, "_last_sampled_edge_attr_missing_eid", 0)) + 1
            
            # Gather features for subgraph nodes
            x_sub = x_global[n_id]                              # [n_sub, H]
            pos_sub = pos_global[n_id].to(device)               # [n_sub]
            edge_index_sub = batch.edge_index.to(device)        # [2, E_sub]
            node_level_sub = data_cpu.node_level[n_id].to(device)
            
            # Edge attributes if used
            edge_attr_sub = None
            if (
                self.use_edge_attr
                and edge_attr_global is not None
                and e_id is not None
            ):
                e_id_device = e_id.to(device)
                edge_attr_sub = edge_attr_global[e_id_device]

            mp = getattr(transformer, "message_passing", None)
            if mp is not None:
                local_attn_cfg = getattr(self, "local_attn_config", {})
                use_l0_local = bool(self.l0_local_backend != "pyg" and int(self.l0_local_window) > 0)
                mp.l0_local_runtime_enable = bool(use_l0_local) and not bool(local_attn_cfg)
                mp.l0_local_runtime_causal = bool(self.hier_ar_enable and self.l0_ar_enable)
                mp.local_attn_runtime_enable = bool(local_attn_cfg)
                mp.local_attn_runtime_causal_gate = bool(self.hier_ar_enable)
                mp.local_attn_runtime_level_grid_shapes = _normalize_level_grid_shape_map(
                    getattr(data_cpu, "level_grid_shapes", None)
                )
                mp.local_attn_runtime_spatial_metric = str(getattr(self, "graph_spatial_metric", "chebyshev"))
                mp.local_attn_runtime_sampled = True
                if nodes_per_sample is not None and int(nodes_per_sample) > 0:
                    mp.local_attn_runtime_group = n_id.to(device=device, dtype=torch.long) // int(nodes_per_sample)
                else:
                    mp.local_attn_runtime_group = None

            # Forward through transformer
            if self.use_gradient_checkpointing and torch.is_grad_enabled():
                x_out, new_edge_attribute = checkpoint(
                    transformer, x_sub, edge_index_sub, node_level_sub,
                    None, pos_sub, edge_attr_sub,
                    use_reentrant=False,
                )
            else:
                x_out, new_edge_attribute = transformer(
                    x_sub, edge_index_sub, node_level_sub,
                    level_offsets=None, positions=pos_sub, edge_attr=edge_attr_sub,
                )

            
            if num_seed > 0:
                x_global[seed_global] = x_out[:num_seed]

            if (
                new_edge_attribute is not None
                and edge_attr_global is not None
                and e_id_device is not None
            ):
                edge_attr_global.index_copy_(0, e_id_device, new_edge_attribute)
                self._last_sampled_edge_attr_writebacks = int(getattr(self, "_last_sampled_edge_attr_writebacks", 0)) + 1

        return x_global, edge_attr_global


    def _resolve_neighbor_sampling_backend(self, device: torch.device) -> str:
        # Only the PyG NeighborLoader backend is supported in this build.
        self._neighbor_sampling_backend_resolved = "pyg"
        return "pyg"



    def _zipper_build_children_map(
        self,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        parent_level: int,
        child_level: int,
    ) -> Dict[int, List[int]]:
        src, dst = edge_index
        lvl_src = node_level[src]
        lvl_dst = node_level[dst]

        child_map: Dict[int, List[int]] = {}

        mask_pc = (lvl_src == parent_level) & (lvl_dst == child_level)
        if mask_pc.any():
            for p, c in zip(src[mask_pc].tolist(), dst[mask_pc].tolist()):
                child_map.setdefault(p, []).append(c)

        mask_cp = (lvl_src == child_level) & (lvl_dst == parent_level)
        if mask_cp.any():
            for c, p in zip(src[mask_cp].tolist(), dst[mask_cp].tolist()):
                child_map.setdefault(p, []).append(c)

        return child_map

    def _zipper_build_children_table(
        self,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        parent_level: int,
        child_level: int,
    ) -> Dict[str, torch.Tensor]:
        src, dst = edge_index
        lvl_src = node_level[src]
        lvl_dst = node_level[dst]
        device = edge_index.device
        num_nodes = int(node_level.numel())
        max_children = max(1, int(self.zip_max_children_per_parent))

        mask_pc = (lvl_src == parent_level) & (lvl_dst == child_level)
        mask_cp = (lvl_src == child_level) & (lvl_dst == parent_level)

        parents = torch.cat([src[mask_pc], dst[mask_cp]], dim=0)
        children = torch.cat([dst[mask_pc], src[mask_cp]], dim=0)
        if parents.numel() == 0:
            return {
                "parent_ids": torch.empty((0,), dtype=torch.long, device=device),
                "children": torch.empty((0, max_children), dtype=torch.long, device=device),
                "counts": torch.empty((0,), dtype=torch.long, device=device),
                "row_of_node": torch.full((num_nodes,), -1, dtype=torch.long, device=device),
            }

        pair_key = parents.to(torch.long) * int(num_nodes) + children.to(torch.long)
        pair_key = torch.unique(pair_key, sorted=True)
        parents = torch.div(pair_key, int(num_nodes), rounding_mode="floor")
        children = pair_key % int(num_nodes)

        parent_ids, inv, counts = torch.unique_consecutive(parents, return_inverse=True, return_counts=True)
        idx = torch.arange(parents.numel(), device=device, dtype=torch.long)
        starts = torch.cumsum(counts, dim=0) - counts
        rank = idx - torch.repeat_interleave(starts, counts)
        keep = rank < max_children

        inv_k = inv[keep]
        rank_k = rank[keep]
        children_k = children[keep]

        table = torch.full((parent_ids.numel(), max_children), -1, dtype=torch.long, device=device)
        table[inv_k, rank_k] = children_k
        counts_k = torch.minimum(counts, torch.full_like(counts, max_children))

        row_of_node = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        row_of_node[parent_ids] = torch.arange(parent_ids.numel(), device=device, dtype=torch.long)

        return {
            "parent_ids": parent_ids,
            "children": table,
            "counts": counts_k,
            "row_of_node": row_of_node,
        }

    def _zipper_select_top_pairs(
        self,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        edge_scores: torch.Tensor,
        level: int,
        max_pairs: int,
    ) -> torch.Tensor:
        src, dst = edge_index
        mask = (node_level[src] == level) & (node_level[dst] == level)
        if not mask.any():
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        scores = edge_scores[mask]
        if scores.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        topk = min(max_pairs, scores.numel())
        top_idx = torch.topk(scores, topk, largest=True).indices
        edges = edge_index[:, mask]
        return edges[:, top_idx]

    def _zipper_score_candidates(
        self,
        transformer,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty((0,), device=x.device, dtype=x.dtype)

        mode = getattr(self, "zip_score_mode", "mp_full")
        if mode == "fast_qk" and hasattr(transformer, "message_passing"):
            mp = transformer.message_passing
            if all(hasattr(mp, attr) for attr in ("q_proj", "k_proj", "level_embedding", "level_attn")):
                with torch.no_grad():
                    x_in = transformer.norm1(x) if hasattr(transformer, "norm1") else x
                    src, dst = edge_index

                    q_all = mp.q_proj(x_in).view(-1, mp.num_heads, mp.head_dim)
                    k_all = mp.k_proj(x_in).view(-1, mp.num_heads, mp.head_dim)
                    q_i = q_all[dst]
                    k_j = k_all[src]

                    score_heads = (q_i * k_j).sum(dim=-1) / math.sqrt(mp.head_dim)

                    level_emb = mp.level_embedding(node_level)
                    level_emb_i = level_emb[dst]
                    level_emb_j = level_emb[src]
                    level_concat = torch.cat([level_emb_i, level_emb_j], dim=-1)
                    level_weights = mp.level_attn(level_concat)
                    level_diff = torch.abs(level_emb_i[:, 0:1] - level_emb_j[:, 0:1])
                    level_scale = 1.0 / (1.0 + level_diff)
                    score_heads = score_heads + level_weights * level_scale

                    conf = score_heads.max(dim=-1, keepdim=True).values
                    effective = score_heads * conf
                    if hasattr(mp, "edge_combine"):
                        scores = mp.edge_combine(effective).squeeze(-1)
                        scores = torch.tanh(scores)
                    else:
                        scores = score_heads.mean(dim=-1)
                    return scores.to(dtype=x.dtype)

        x_in = transformer.norm1(x) if hasattr(transformer, "norm1") else x
        with torch.no_grad():
            _, new_edge_attr = transformer.message_passing(
                x_in,
                edge_index,
                node_level,
                level_offsets=None,
                positions=positions,
                edge_attr=None,
            )
        if new_edge_attr is None:
            return torch.zeros(edge_index.size(1), device=x.device, dtype=x.dtype)
        return new_edge_attr

    def _zipper_select_mask(self, scores: torch.Tensor, device: torch.device) -> torch.Tensor:
        if scores.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        mode = self.zip_edge_select_mode
        if mode == "absolute":
            return scores >= self.zip_edge_drop_threshold
        if mode == "topk":
            topk = min(self.zip_max_candidate_edges, scores.numel())
            idx = torch.topk(scores, topk, largest=True).indices
            keep = torch.zeros(scores.size(0), dtype=torch.bool, device=device)
            keep[idx] = True
            return keep
        if mode == "relative_max":
            max_score = scores.max()
            return scores >= (self.zip_edge_relative_ratio * max_score)

        k = int(max(1, math.ceil((1.0 - self.zip_edge_percentile) * scores.numel())))
        k = min(k, scores.numel())
        idx = torch.topk(scores, k, largest=True).indices
        keep = torch.zeros(scores.size(0), dtype=torch.bool, device=device)
        keep[idx] = True
        return keep

    def _zipper_build_ephemeral_candidates(
        self,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
    ) -> torch.Tensor:
        device = edge_index.device

        l3_to_l2 = self._zipper_build_children_map(edge_index, node_level, 3, 2)
        l2_to_l1 = self._zipper_build_children_map(edge_index, node_level, 2, 1)
        l1_to_l0 = self._zipper_build_children_map(edge_index, node_level, 1, 0) if self.zip_depth == "l0" else None

        def _dense_pairs(idx: torch.Tensor, cap: int) -> torch.Tensor:
            if idx.numel() <= 1:
                return torch.empty((0, 2), dtype=torch.long, device=device)
            pairs = torch.cartesian_prod(idx, idx)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            if pairs.size(0) > cap:
                pairs = pairs[:cap]
            return pairs

        l3_idx = (node_level == 3).nonzero(as_tuple=False).view(-1)
        l2_idx = (node_level == 2).nonzero(as_tuple=False).view(-1)
        top_l3 = _dense_pairs(l3_idx, int(self.zip_max_l3_pairs))
        top_l2 = _dense_pairs(l2_idx, int(self.zip_max_l2_pairs))

        def _sample_children(children: List[int]) -> List[int]:
            if len(children) <= self.zip_max_children_per_parent:
                return children
            return children[: self.zip_max_children_per_parent]

        candidates: List[List[int]] = []
        seen = set()

        def _append_pair(s: int, d: int) -> bool:
            key = (int(s), int(d))
            if key in seen:
                return False
            seen.add(key)
            candidates.append([key[0], key[1]])
            return len(candidates) < int(self.zip_max_candidate_edges)

        for s, d in top_l3.tolist():
            if not _append_pair(s, d):
                break
            c1 = _sample_children(l3_to_l2.get(int(s), []))
            c2 = _sample_children(l3_to_l2.get(int(d), []))
            for a in c1:
                for b in c2:
                    if not _append_pair(a, b):
                        break
                if len(candidates) >= int(self.zip_max_candidate_edges):
                    break
            if len(candidates) >= int(self.zip_max_candidate_edges):
                break

        if len(candidates) < int(self.zip_max_candidate_edges):
            for s, d in top_l2.tolist():
                if not _append_pair(s, d):
                    break
                c1 = _sample_children(l2_to_l1.get(int(s), []))
                c2 = _sample_children(l2_to_l1.get(int(d), []))
                for a in c1:
                    for b in c2:
                        if not _append_pair(a, b):
                            break
                    if len(candidates) >= int(self.zip_max_candidate_edges):
                        break

                if self.zip_depth == "l0" and l1_to_l0 is not None and len(candidates) < int(self.zip_max_candidate_edges):
                    for a in c1:
                        ca = _sample_children(l1_to_l0.get(int(a), []))
                        for b in c2:
                            cb = _sample_children(l1_to_l0.get(int(b), []))
                            for ia in ca:
                                for ib in cb:
                                    if not _append_pair(ia, ib):
                                        break
                                if len(candidates) >= int(self.zip_max_candidate_edges):
                                    break
                            if len(candidates) >= int(self.zip_max_candidate_edges):
                                break
                        if len(candidates) >= int(self.zip_max_candidate_edges):
                            break

                if len(candidates) >= int(self.zip_max_candidate_edges):
                    break

        if not candidates:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(candidates, device=device, dtype=torch.long).t()

    def _zipper_score_candidates_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_bnh.size(0)
        K = edge_index.size(1)
        if K == 0:
            return torch.empty((B, 0), device=x_bnh.device, dtype=x_bnh.dtype)

        mode = getattr(self, "zip_score_mode", "fast_qk")
        if mode != "fast_qk":
            scores = []
            for b in range(B):
                scores.append(self._zipper_score_candidates(transformer, x_bnh[b], edge_index, node_level, positions=None))
            return torch.stack(scores, dim=0)

        mp = transformer.message_passing
        if x_in is None:
            x_in = transformer.norm1(x_bnh) if hasattr(transformer, "norm1") else x_bnh
        with torch.no_grad():
            q_all = mp.q_proj(x_in).view(B, x_bnh.size(1), mp.num_heads, mp.head_dim)
            k_all = mp.k_proj(x_in).view(B, x_bnh.size(1), mp.num_heads, mp.head_dim)
            level_emb = mp.level_embedding(node_level)

        return self._zipper_score_from_qk_cache(
            transformer=transformer,
            edge_index=edge_index,
            node_level=node_level,
            _qk_cache=(q_all, k_all, level_emb),
            dtype=x_bnh.dtype,
        )

    def _zipper_score_from_qk_cache(
        self,
        transformer,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        _qk_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Score candidate edges using precomputed Q, K, and level embeddings.
        Avoids re-running q_proj/k_proj/level_embedding per stage."""
        q_all, k_all, level_emb = _qk_cache
        B = int(q_all.size(0))
        K = int(edge_index.size(1))
        if K == 0:
            return torch.empty((B, 0), device=q_all.device, dtype=dtype)

        mp = transformer.message_passing
        src, dst = edge_index

        with torch.no_grad():
            q_i = q_all[:, dst]
            k_j = k_all[:, src]
            score_heads = (q_i * k_j).sum(dim=-1) / math.sqrt(mp.head_dim)

            level_emb_i = level_emb[dst]
            level_emb_j = level_emb[src]
            level_concat = torch.cat([level_emb_i, level_emb_j], dim=-1)
            level_weights = mp.level_attn(level_concat)
            level_diff = torch.abs(level_emb_i[:, 0:1] - level_emb_j[:, 0:1])
            level_scale = 1.0 / (1.0 + level_diff)
            score_heads = score_heads + level_weights.unsqueeze(0) * level_scale.unsqueeze(0)

            conf = score_heads.max(dim=-1, keepdim=True).values
            effective = score_heads * conf
            if hasattr(mp, "edge_combine"):
                scores = mp.edge_combine(effective.reshape(B * K, mp.num_heads)).squeeze(-1).view(B, K)
                scores = torch.tanh(scores)
            else:
                scores = score_heads.mean(dim=-1)
        return scores.to(dtype=dtype)

    def _zipper_candidate_softmax_batched(
        self,
        scores_bk: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        B, K = scores_bk.shape
        if K == 0:
            return scores_bk
        device = scores_bk.device
        offsets = (torch.arange(B, device=device, dtype=torch.long) * int(num_nodes)).view(B, 1)
        index_flat = (dst_idx.view(1, -1).to(device=device, dtype=torch.long) + offsets).reshape(-1)
        weights_flat = softmax(scores_bk.reshape(-1), index=index_flat)
        return weights_flat.view(B, K)

    def _zipper_select_mask_batched(
        self,
        scores_bk: torch.Tensor,
        dst_idx: torch.Tensor,
        valid_bk: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K = scores_bk.shape
        device = scores_bk.device
        keep = torch.zeros((B, K), dtype=torch.bool, device=device)
        if K == 0:
            return keep

        if valid_bk is not None and valid_bk.shape != keep.shape:
            raise ValueError(
                f"zipper valid_bk shape mismatch: expected {tuple(keep.shape)}, got {tuple(valid_bk.shape)}"
            )

        if getattr(self, "zip_select_scope", "global") == "global":
            # Phase 4: vectorized global selection avoids per-batch Python loop
            mode = self.zip_edge_select_mode
            if B > 1 and mode == "percentile" and K > 0:
                # Vectorized percentile selection across all batches
                pctl = float(self.zip_edge_percentile)
                if valid_bk is not None:
                    # Mask invalid scores to -inf so they are never selected
                    masked_scores = scores_bk.clone()
                    masked_scores[~valid_bk] = float("-inf")
                    n_valid_per_b = valid_bk.sum(dim=1)  # [B]
                else:
                    masked_scores = scores_bk
                    n_valid_per_b = torch.full((B,), K, device=device, dtype=torch.long)

                # k per batch = ceil((1 - pctl) * n_valid), at least 1
                k_per_b = torch.clamp(
                    torch.ceil((1.0 - pctl) * n_valid_per_b.float()).to(torch.long),
                    min=1,
                )
                k_per_b = torch.minimum(k_per_b, n_valid_per_b)
                max_k = int(k_per_b.max().item()) if int(n_valid_per_b.max().item()) > 0 else 0

                if max_k > 0:
                    # Sort descending per row, pick top max_k
                    topvals, topidx = torch.topk(masked_scores, min(max_k, K), dim=1, largest=True, sorted=True)
                    # Build a rank mask: position < k_per_b[b]
                    ranks = torch.arange(topidx.size(1), device=device, dtype=torch.long).unsqueeze(0)  # [1, max_k]
                    rank_keep = ranks < k_per_b.unsqueeze(1)  # [B, max_k]
                    # Also exclude any -inf scores (from invalid masking)
                    rank_keep = rank_keep & (topvals > float("-inf"))
                    # Scatter back to keep
                    keep.scatter_(1, topidx, rank_keep)
                return keep
            else:
                # Fallback for B==1 or non-percentile modes
                for b in range(B):
                    valid_idx = None
                    if valid_bk is not None:
                        valid_idx = valid_bk[b].nonzero(as_tuple=False).view(-1)
                        if valid_idx.numel() == 0:
                            continue
                        score_vec = scores_bk[b, valid_idx]
                    else:
                        score_vec = scores_bk[b]
                    keep_local = self._zipper_select_mask(score_vec, device)
                    if valid_idx is None:
                        keep[b] = keep_local
                    else:
                        keep[b, valid_idx[keep_local]] = True
                return keep

        topk_per_dst = max(1, int(getattr(self, "zip_per_dst_topk", 1)))
        if K == 0:
            return keep

        dst_span = int(dst_idx.max().item()) + 1
        b_full = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(B, K).reshape(-1)
        k_full = torch.arange(K, device=device, dtype=torch.long).view(1, K).expand(B, K).reshape(-1)
        dst_full = dst_idx.view(1, K).expand(B, K).reshape(-1)
        score_full = scores_bk.reshape(-1)

        valid_flat = torch.ones((B * K,), dtype=torch.bool, device=device)
        if valid_bk is not None:
            valid_flat = valid_bk.reshape(-1)
        if not valid_flat.any():
            return keep

        cand = valid_flat.nonzero(as_tuple=False).view(-1)
        b_c = b_full[cand]
        k_c = k_full[cand]
        dst_c = dst_full[cand]
        s_c = score_full[cand]

        # Group by (batch, dst) and keep top-k per destination (vectorized).
        g_c = b_c * dst_span + dst_c
        perm_score = torch.argsort(s_c, descending=True, stable=True)
        g_s = g_c[perm_score]
        b_s = b_c[perm_score]
        k_s = k_c[perm_score]
        s_s = s_c[perm_score]

        perm_group = torch.argsort(g_s, descending=False, stable=True)
        g_s = g_s[perm_group]
        b_s = b_s[perm_group]
        k_s = k_s[perm_group]
        s_s = s_s[perm_group]

        _, counts = torch.unique_consecutive(g_s, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(g_s.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(starts, counts)
        keep_topdst = rank < topk_per_dst

        b_keep = b_s[keep_topdst]
        k_keep = k_s[keep_topdst]
        s_keep = s_s[keep_topdst]

        cap = int(self.zip_max_candidate_edges)
        if b_keep.numel() > 0 and cap > 0:
            # Final per-batch cap while preserving top-score preference.
            perm_score_b = torch.argsort(s_keep, descending=True, stable=True)
            b2 = b_keep[perm_score_b]
            k2 = k_keep[perm_score_b]

            perm_batch = torch.argsort(b2, descending=False, stable=True)
            b2 = b2[perm_batch]
            k2 = k2[perm_batch]

            _, b_counts = torch.unique_consecutive(b2, return_counts=True)
            b_starts = torch.cumsum(b_counts, dim=0) - b_counts
            b_rank = torch.arange(b2.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(b_starts, b_counts)
            keep_batch = b_rank < cap

            b_keep = b2[keep_batch]
            k_keep = k2[keep_batch]

        if b_keep.numel() > 0:
            keep[b_keep, k_keep] = True

        return keep

    def _zipper_inject_ephemeral_messages_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        edge_index: torch.Tensor,
        scores_bk: torch.Tensor,
        keep_bk: torch.Tensor,
        attn_bk: Optional[torch.Tensor] = None,
        x_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        selected = keep_bk.nonzero(as_tuple=False)
        if selected.numel() == 0:
            return x_bnh, 0

        device = x_bnh.device
        B, N, _ = x_bnh.shape
        src_all, dst_all = edge_index
        b_idx = selected[:, 0]
        k_idx = selected[:, 1]
        src_idx = src_all[k_idx]
        dst_idx = dst_all[k_idx]
        score_sel = scores_bk[b_idx, k_idx]

        group_idx = b_idx * N + dst_idx
        if attn_bk is not None:
            weights = attn_bk[b_idx, k_idx]
        else:
            weights = softmax(score_sel, group_idx)

        mass_budget = float(getattr(self, "zip_mass_budget_per_dst", 1.0))
        if mass_budget > 0.0:
            mass = scatter(weights, group_idx, dim=0, dim_size=B * N, reduce="sum")
            mass_scale = torch.clamp(mass_budget / (mass + 1e-8), max=1.0)
            weights = weights * mass_scale[group_idx]

        if bool(getattr(self, "zip_paramfree_gate", True)):
            conf_dst = scatter(weights, group_idx, dim=0, dim_size=B * N, reduce="max")
            tau = max(1e-6, float(getattr(self, "zip_paramfree_gate_tau", 0.1)))
            center = float(getattr(self, "zip_paramfree_gate_center", 0.2))
            gate_dst = torch.sigmoid((conf_dst - center) / tau)
            weights = weights * gate_dst[group_idx]

        mp = transformer.message_passing
        if x_in is None:
            x_in = transformer.norm1(x_bnh) if hasattr(transformer, "norm1") else x_bnh
        # Fat-V: wide value head dim on this zipper apply (== head_dim when off).
        v_hd = mp.sparse_v_head_dim
        v_all = mp.sparse_proj_v(x_in).view(B, N, mp.num_heads, v_hd)
        msg = v_all[b_idx, src_idx] * weights.view(-1, 1, 1)

        out_flat = torch.zeros(B * N, mp.num_heads, v_hd, device=device, dtype=x_bnh.dtype)
        out_flat.index_add_(0, group_idx, msg)
        m_zip = out_flat.view(B, N, mp.num_heads, v_hd).reshape(B, N, mp.num_heads * v_hd)
        m_zip = mp.sparse_out_proj(m_zip)

        if (not bool(getattr(self, "zip_paramfree_gate", True))) and bool(getattr(self, "zip_use_beta_gate", True)):
            beta_layer = getattr(transformer, "zip_lin_beta_attn", None)
            if beta_layer is not None:
                beta = torch.sigmoid(beta_layer(m_zip))
                zip_dropout = transformer.dropout(m_zip) if hasattr(transformer, "dropout") else m_zip
                m_zip = (1.0 - beta) * zip_dropout

        norm_clip = float(getattr(self, "zip_norm_clip_ratio", 0.0))
        if norm_clip > 0.0:
            x_norm = x_bnh.norm(dim=-1, keepdim=True)
            m_norm = m_zip.norm(dim=-1, keepdim=True)
            max_m = norm_clip * x_norm
            clip = torch.clamp(max_m / (m_norm + 1e-8), max=1.0)
            m_zip = m_zip * clip

        with torch.no_grad():
            x_norm = x_bnh.norm(dim=-1)
            m_norm = m_zip.norm(dim=-1)
            ratio = m_norm / (x_norm + 1e-8)
            self._last_zip_msg_norm_ratio_mean = float(ratio.mean().item())
            self._last_zip_msg_norm_ratio_max = float(ratio.max().item())

        return x_bnh + m_zip, int(selected.size(0))

    def _zipper_filter_ar_edges(
        self,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index
        if not (
            self.hier_ar_enable
            and self.hier_ar_filter_zip
            and node_ar_time is not None
        ):
            return edge_index

        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        if self.hier_ar_allow_same_time:
            keep_ar = node_ar_time[src_idx] <= node_ar_time[dst_idx]
        else:
            keep_ar = node_ar_time[src_idx] < node_ar_time[dst_idx]

        if not self.l0_ar_enable:
            both_l0 = (node_level[src_idx] == 0) & (node_level[dst_idx] == 0)
            keep_ar = keep_ar | both_l0

        return edge_index[:, keep_ar]

    def _zipper_filter_within_sample_edges(
        self,
        edge_index: torch.Tensor,
        nodes_per_sample: int,
    ) -> torch.Tensor:
        if edge_index.numel() == 0 or int(nodes_per_sample) <= 0:
            return edge_index
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        keep_same = (src_idx // int(nodes_per_sample)) == (dst_idx // int(nodes_per_sample))
        return edge_index[:, keep_same]

    def _zipper_dense_pairs_from_indices(
        self,
        idx: torch.Tensor,
        cap: int,
        device: torch.device,
    ) -> torch.Tensor:
        if idx.numel() <= 1 or cap <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        m = int(idx.numel())
        max_pairs = m * (m - 1)
        cap = int(cap)

        if max_pairs <= cap:
            pairs = torch.cartesian_prod(idx, idx)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            if pairs.size(0) == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            return pairs.t()

        k_per_src = min(m - 1, max(1, (cap + m - 1) // m))
        src_rep = idx.repeat_interleave(k_per_src)
        base = torch.arange(m, device=device, dtype=torch.long)
        offs = torch.arange(1, k_per_src + 1, device=device, dtype=torch.long)
        dst_pos = (base.unsqueeze(1) + offs.unsqueeze(0)) % m
        dst_rep = idx[dst_pos.reshape(-1)]

        pairs = torch.stack([src_rep, dst_rep], dim=0)
        if pairs.size(1) > cap:
            pairs = pairs[:, :cap]
        return pairs

    def _zipper_dense_pairs_from_level_per_sample(
        self,
        node_level: torch.Tensor,
        level: int,
        cap_per_sample: int,
        nodes_per_sample: int,
        device: torch.device,
    ) -> torch.Tensor:
        cap_per_sample = int(cap_per_sample)
        if cap_per_sample <= 0 or int(nodes_per_sample) <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        lvl_idx = (node_level == int(level)).nonzero(as_tuple=False).view(-1)
        if lvl_idx.numel() <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        sample_idx = torch.div(lvl_idx, int(nodes_per_sample), rounding_mode="floor")
        unique_samples = torch.unique(sample_idx, sorted=True)
        parts: List[torch.Tensor] = []
        for b in unique_samples.tolist():
            idx_b = lvl_idx[sample_idx == int(b)]
            pairs_b = self._zipper_dense_pairs_from_indices(
                idx=idx_b,
                cap=cap_per_sample,
                device=device,
            )
            if pairs_b.numel() > 0:
                parts.append(pairs_b)

        if not parts:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(parts, dim=1)

    def _zipper_score_select_stage_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        valid_bk: Optional[torch.Tensor],
        x_in: Optional[torch.Tensor],
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        device = x_bnh.device
        B = int(x_bnh.size(0))
        K = int(edge_index.size(1))
        stage_stats: Dict[str, int] = {
            "candidates": K,
            "selected_union": 0,
            "selected_total": 0,
        }
        if K == 0:
            return (
                torch.zeros((B, 0), dtype=torch.bool, device=device),
                torch.empty((B, 0), dtype=x_bnh.dtype, device=device),
                stage_stats,
            )

        if x_in is None:
            x_in = transformer.norm1(x_bnh) if hasattr(transformer, "norm1") else x_bnh

        if _qk_cache is not None:
            scores_bk = self._zipper_score_from_qk_cache(
                transformer=transformer,
                edge_index=edge_index,
                node_level=node_level,
                _qk_cache=_qk_cache,
                dtype=x_bnh.dtype,
            )
        else:
            scores_bk = self._zipper_score_candidates_batched(
                transformer=transformer,
                x_bnh=x_bnh,
                edge_index=edge_index,
                node_level=node_level,
                x_in=x_in,
            )

        attn_bk = self._zipper_candidate_softmax_batched(
            scores_bk=scores_bk,
            dst_idx=edge_index[1],
            num_nodes=x_bnh.size(1),
        )
        select_values = attn_bk if bool(getattr(self, "zip_select_from_softmax", True)) else scores_bk
        keep_bk = self._zipper_select_mask_batched(
            scores_bk=select_values,
            dst_idx=edge_index[1],
            valid_bk=valid_bk,
        )

        keep_union = keep_bk.any(dim=0)
        stage_stats["selected_union"] = int(keep_union.sum().item())
        stage_stats["selected_total"] = int(keep_bk.sum().item())
        return keep_bk, scores_bk, stage_stats

    def _zipper_score_candidates_flat(
        self,
        transformer,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty((0,), device=x_flat.device, dtype=x_flat.dtype)

        mp = transformer.message_passing
        src, dst = edge_index
        if _qk_cache is not None:
            q_all, k_all, level_emb = _qk_cache
        else:
            if x_in is None:
                x_in = transformer.norm1(x_flat) if hasattr(transformer, "norm1") else x_flat
            with torch.no_grad():
                q_all = mp.q_proj(x_in).view(-1, mp.num_heads, mp.head_dim)
                k_all = mp.k_proj(x_in).view(-1, mp.num_heads, mp.head_dim)
                level_emb = mp.level_embedding(node_level)

        with torch.no_grad():
            q_i = q_all[dst]
            k_j = k_all[src]
            score_heads = (q_i * k_j).sum(dim=-1) / math.sqrt(mp.head_dim)

            level_emb_i = level_emb[dst]
            level_emb_j = level_emb[src]
            level_concat = torch.cat([level_emb_i, level_emb_j], dim=-1)
            level_weights = mp.level_attn(level_concat)
            level_diff = torch.abs(level_emb_i[:, 0:1] - level_emb_j[:, 0:1])
            level_scale = 1.0 / (1.0 + level_diff)
            score_heads = score_heads + level_weights * level_scale

            conf = score_heads.max(dim=-1, keepdim=True).values
            effective = score_heads * conf
            if hasattr(mp, "edge_combine"):
                scores = mp.edge_combine(effective).squeeze(-1)
                scores = torch.tanh(scores)
            else:
                scores = score_heads.mean(dim=-1)
        return scores.to(dtype=x_flat.dtype)

    def _zipper_score_select_stage_flat(
        self,
        transformer,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        valid_bk: Optional[torch.Tensor],
        batch_size: int,
        nodes_per_sample: int,
        x_in: Optional[torch.Tensor] = None,
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        device = x_flat.device
        B = int(batch_size)
        K = int(edge_index.size(1))
        stage_stats: Dict[str, int] = {
            "candidates": K,
            "selected_union": 0,
            "selected_total": 0,
        }
        if K == 0:
            return (
                torch.zeros((B, 0), dtype=torch.bool, device=device),
                torch.empty((B, 0), dtype=x_flat.dtype, device=device),
                stage_stats,
            )

        scores = self._zipper_score_candidates_flat(
            transformer=transformer,
            x_flat=x_flat,
            edge_index=edge_index,
            node_level=node_level,
            x_in=x_in,
            _qk_cache=_qk_cache,
        )

        dst_idx = edge_index[1]
        weights = softmax(scores, index=dst_idx)
        select_values = weights if bool(getattr(self, "zip_select_from_softmax", True)) else scores

        scores_bk = select_values.view(1, K).expand(B, K).clone()
        if valid_bk is None:
            if int(nodes_per_sample) <= 0:
                valid_bk = torch.ones((B, K), dtype=torch.bool, device=device)
            else:
                sample_idx = torch.div(edge_index[0], int(nodes_per_sample), rounding_mode="floor")
                valid_bk = torch.zeros((B, K), dtype=torch.bool, device=device)
                in_range = (sample_idx >= 0) & (sample_idx < B)
                if bool(in_range.any()):
                    k_idx = torch.nonzero(in_range, as_tuple=False).view(-1)
                    valid_bk[sample_idx[in_range], k_idx] = True

        keep_bk = self._zipper_select_mask_batched(
            scores_bk=scores_bk,
            dst_idx=dst_idx,
            valid_bk=valid_bk,
        )

        score_bk = scores.view(1, K).expand(B, K).clone()
        keep_union = keep_bk.any(dim=0)
        stage_stats["selected_union"] = int(keep_union.sum().item())
        stage_stats["selected_total"] = int(keep_bk.sum().item())
        return keep_bk, score_bk, stage_stats

    def _zipper_collect_selected_stage_edges(
        self,
        edge_index: torch.Tensor,
        keep_bk: torch.Tensor,
        scores_bk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device
        if edge_index.numel() == 0 or keep_bk.numel() == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=scores_bk.dtype, device=device),
            )

        keep_union = keep_bk.any(dim=0)
        if not bool(keep_union.any()):
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=scores_bk.dtype, device=device),
            )

        edge_sel = edge_index[:, keep_union]
        score_sel = scores_bk[:, keep_union]
        keep_sel = keep_bk[:, keep_union]
        neg_inf = torch.full_like(score_sel, float("-inf"))
        score_union = torch.where(keep_sel, score_sel, neg_inf).max(dim=0).values
        finite = torch.isfinite(score_union)
        if not bool(finite.any()):
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=scores_bk.dtype, device=device),
            )
        return edge_sel[:, finite], score_union[finite]

    def _zipper_build_blockdiag_mutation_candidates_batched(
        self,
        transformer,
        x_flat: torch.Tensor,
        base_edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        batch_size: int,
        nodes_per_sample: int,
        enable_l3_stage: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        device = x_flat.device
        B = int(batch_size)
        N = int(nodes_per_sample)
        if B <= 0 or N <= 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=x_flat.dtype, device=device),
                {
                    "l3_candidates": 0,
                    "l3_selected_union": 0,
                    "l3_selected_total": 0,
                    "l2_candidates": 0,
                    "l2_selected_union": 0,
                    "l2_selected_total": 0,
                    "l1_candidates": 0,
                    "l1_selected_union": 0,
                    "l1_selected_total": 0,
                    "l0_candidates": 0,
                    "l0_selected_union": 0,
                    "l0_selected_total": 0,
                },
            )

        stage_stats: Dict[str, int] = {
            "l3_candidates": 0,
            "l3_selected_union": 0,
            "l3_selected_total": 0,
            "l2_candidates": 0,
            "l2_selected_union": 0,
            "l2_selected_total": 0,
            "l1_candidates": 0,
            "l1_selected_union": 0,
            "l1_selected_total": 0,
            "l0_candidates": 0,
            "l0_selected_union": 0,
            "l0_selected_total": 0,
        }

        l3_to_l2 = self._zipper_build_children_table_cached(base_edge_index, node_level, 3, 2)
        l2_to_l1 = self._zipper_build_children_table_cached(base_edge_index, node_level, 2, 1)
        l1_to_l0 = None
        if self.zip_depth == "l0":
            l1_to_l0 = self._zipper_build_children_table_cached(base_edge_index, node_level, 1, 0)

        x_in = transformer.norm1(x_flat) if hasattr(transformer, "norm1") else x_flat
        mp = transformer.message_passing
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        mode = getattr(self, "zip_score_mode", "fast_qk")
        if mode == "fast_qk" and all(hasattr(mp, a) for a in ("q_proj", "k_proj", "level_embedding", "level_attn")):
            with torch.no_grad():
                q_all = mp.q_proj(x_in).view(B * N, mp.num_heads, mp.head_dim)
                k_all = mp.k_proj(x_in).view(B * N, mp.num_heads, mp.head_dim)
                level_emb = mp.level_embedding(node_level)
            _qk_cache = (q_all, k_all, level_emb)

        edge_parts: List[torch.Tensor] = []
        score_parts: List[torch.Tensor] = []

        stage_l3 = torch.empty((2, 0), dtype=torch.long, device=device)
        if enable_l3_stage:
            stage_l3 = self._zipper_dense_pairs_from_level_per_sample(
                node_level=node_level,
                level=3,
                cap_per_sample=int(self.zip_l3_pairs_cap),
                nodes_per_sample=N,
                device=device,
        )
        stage_l3 = self._zipper_filter_ar_edges(stage_l3, node_level, node_ar_time)
        keep_l3_bk, scores_l3_bk, l3_stats = self._zipper_score_select_stage_flat(
            transformer=transformer,
            x_flat=x_flat,
            edge_index=stage_l3,
            node_level=node_level,
            valid_bk=None,
            batch_size=B,
            nodes_per_sample=N,
            x_in=x_in,
            _qk_cache=_qk_cache,
        )
        stage_stats["l3_candidates"] = int(l3_stats["candidates"])
        stage_stats["l3_selected_union"] = int(l3_stats["selected_union"])
        stage_stats["l3_selected_total"] = int(l3_stats["selected_total"])
        e_l3, s_l3 = self._zipper_collect_selected_stage_edges(stage_l3, keep_l3_bk, scores_l3_bk)
        if e_l3.numel() > 0:
            edge_parts.append(e_l3)
            score_parts.append(s_l3)

        stage_l2, valid_l2_bk, _ = self._zipper_build_stage_candidates_with_mask_fast(
            parent_edge_index=stage_l3,
            parent_keep_bk=keep_l3_bk,
            children_table=l3_to_l2,
            fallback_level=2,
            fallback_cap=0,
            node_level=node_level,
            node_ar_time=node_ar_time,
            device=device,
        )
        keep_l2_bk, scores_l2_bk, l2_stats = self._zipper_score_select_stage_flat(
            transformer=transformer,
            x_flat=x_flat,
            edge_index=stage_l2,
            node_level=node_level,
            valid_bk=valid_l2_bk,
            batch_size=B,
            nodes_per_sample=N,
            x_in=x_in,
            _qk_cache=_qk_cache,
        )
        stage_stats["l2_candidates"] = int(l2_stats["candidates"])
        stage_stats["l2_selected_union"] = int(l2_stats["selected_union"])
        stage_stats["l2_selected_total"] = int(l2_stats["selected_total"])
        e_l2, s_l2 = self._zipper_collect_selected_stage_edges(stage_l2, keep_l2_bk, scores_l2_bk)
        if e_l2.numel() > 0:
            edge_parts.append(e_l2)
            score_parts.append(s_l2)

        stage_l1, valid_l1_bk, _ = self._zipper_build_stage_candidates_with_mask_fast(
            parent_edge_index=stage_l2,
            parent_keep_bk=keep_l2_bk,
            children_table=l2_to_l1,
            fallback_level=1,
            fallback_cap=0,
            node_level=node_level,
            node_ar_time=node_ar_time,
            device=device,
        )
        keep_l1_bk, scores_l1_bk, l1_stats = self._zipper_score_select_stage_flat(
            transformer=transformer,
            x_flat=x_flat,
            edge_index=stage_l1,
            node_level=node_level,
            valid_bk=valid_l1_bk,
            batch_size=B,
            nodes_per_sample=N,
            x_in=x_in,
            _qk_cache=_qk_cache,
        )
        stage_stats["l1_candidates"] = int(l1_stats["candidates"])
        stage_stats["l1_selected_union"] = int(l1_stats["selected_union"])
        stage_stats["l1_selected_total"] = int(l1_stats["selected_total"])
        e_l1, s_l1 = self._zipper_collect_selected_stage_edges(stage_l1, keep_l1_bk, scores_l1_bk)
        if e_l1.numel() > 0:
            edge_parts.append(e_l1)
            score_parts.append(s_l1)

        if self.zip_depth == "l0" and l1_to_l0 is not None:
            stage_l0, valid_l0_bk, _ = self._zipper_build_stage_candidates_with_mask_fast(
                parent_edge_index=stage_l1,
                parent_keep_bk=keep_l1_bk,
                children_table=l1_to_l0,
                fallback_level=0,
                fallback_cap=0,
                node_level=node_level,
                node_ar_time=node_ar_time,
                device=device,
            )
            keep_l0_bk, scores_l0_bk, l0_stats = self._zipper_score_select_stage_flat(
                transformer=transformer,
                x_flat=x_flat,
                edge_index=stage_l0,
                node_level=node_level,
                valid_bk=valid_l0_bk,
                batch_size=B,
                nodes_per_sample=N,
                x_in=x_in,
                _qk_cache=_qk_cache,
            )
            stage_stats["l0_candidates"] = int(l0_stats["candidates"])
            stage_stats["l0_selected_union"] = int(l0_stats["selected_union"])
            stage_stats["l0_selected_total"] = int(l0_stats["selected_total"])
            e_l0, s_l0 = self._zipper_collect_selected_stage_edges(stage_l0, keep_l0_bk, scores_l0_bk)
            if e_l0.numel() > 0:
                edge_parts.append(e_l0)
                score_parts.append(s_l0)

        if not edge_parts:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=x_flat.dtype, device=device),
                stage_stats,
            )

        all_edges = torch.cat(edge_parts, dim=1)
        all_scores = torch.cat(score_parts, dim=0).to(dtype=x_flat.dtype)
        num_nodes_total = int(node_level.numel())
        key = all_edges[0].to(torch.long) * num_nodes_total + all_edges[1].to(torch.long)
        key, inv = torch.unique(key, sorted=True, return_inverse=True)
        score_red = scatter(all_scores, inv, dim=0, dim_size=key.numel(), reduce="max")
        dst = key % num_nodes_total
        src = torch.div(key, num_nodes_total, rounding_mode="floor")
        merged_edges = torch.stack([src, dst], dim=0)
        return merged_edges, score_red, stage_stats

    def _zipper_dense_pairs_from_level(
        self,
        node_level: torch.Tensor,
        level: int,
        cap: int,
        device: torch.device,
    ) -> torch.Tensor:
        lvl_idx = (node_level == int(level)).nonzero(as_tuple=False).view(-1)
        if lvl_idx.numel() <= 1 or cap <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        m = int(lvl_idx.numel())
        max_pairs = m * (m - 1)
        cap = int(cap)

        if max_pairs <= cap:
            pairs = torch.cartesian_prod(lvl_idx, lvl_idx)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            if pairs.size(0) == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            return pairs.t()

        # Efficient bounded dense fallback without materializing all O(m^2) pairs.
        k_per_src = min(m - 1, max(1, (cap + m - 1) // m))
        src_rep = lvl_idx.repeat_interleave(k_per_src)
        base = torch.arange(m, device=device, dtype=torch.long)
        offs = torch.arange(1, k_per_src + 1, device=device, dtype=torch.long)
        dst_pos = (base.unsqueeze(1) + offs.unsqueeze(0)) % m
        dst_rep = lvl_idx[dst_pos.reshape(-1)]

        pairs = torch.stack([src_rep, dst_rep], dim=0)
        if pairs.size(1) > cap:
            pairs = pairs[:, :cap]
        return pairs

    def _zipper_expand_descendant_pairs(
        self,
        parent_pairs: torch.Tensor,
        children_map: Dict[int, List[int]],
        cap: int,
        device: torch.device,
    ) -> torch.Tensor:
        if parent_pairs.numel() == 0 or cap <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        candidates: List[List[int]] = []
        seen = set()
        max_children = max(1, int(self.zip_max_children_per_parent))

        for src_parent, dst_parent in zip(parent_pairs[0].tolist(), parent_pairs[1].tolist()):
            src_children = children_map.get(int(src_parent), [])
            dst_children = children_map.get(int(dst_parent), [])
            if len(src_children) > max_children:
                src_children = src_children[:max_children]
            if len(dst_children) > max_children:
                dst_children = dst_children[:max_children]

            for src_child in src_children:
                for dst_child in dst_children:
                    if int(src_child) == int(dst_child):
                        continue
                    key = (int(src_child), int(dst_child))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append([key[0], key[1]])
                    if len(candidates) >= int(cap):
                        break
                if len(candidates) >= int(cap):
                    break
            if len(candidates) >= int(cap):
                break

        if not candidates:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(candidates, dtype=torch.long, device=device).t()

    def _zipper_build_stage_candidates_with_mask(
        self,
        parent_edge_index: torch.Tensor,
        parent_keep_bk: torch.Tensor,
        children_map: Dict[int, List[int]],
        fallback_level: int,
        fallback_cap: int,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B = parent_keep_bk.size(0)
        per_sample_pairs: List[List[Tuple[int, int]]] = []
        fallback_used = 0

        for b in range(B):
            sel_idx = parent_keep_bk[b].nonzero(as_tuple=False).view(-1)
            if sel_idx.numel() > 0:
                parent_pairs_b = parent_edge_index[:, sel_idx]
                pairs_b = self._zipper_expand_descendant_pairs(
                    parent_pairs=parent_pairs_b,
                    children_map=children_map,
                    cap=int(self.zip_max_candidate_edges),
                    device=device,
                )
                pairs_b = self._zipper_filter_ar_edges(pairs_b, node_level, node_ar_time)
            else:
                pairs_b = torch.empty((2, 0), dtype=torch.long, device=device)

            if pairs_b.numel() == 0:
                fallback_used += 1
                pairs_b = self._zipper_dense_pairs_from_level(
                    node_level=node_level,
                    level=int(fallback_level),
                    cap=int(fallback_cap),
                    device=device,
                )
                pairs_b = self._zipper_filter_ar_edges(pairs_b, node_level, node_ar_time)

            if pairs_b.numel() == 0:
                per_sample_pairs.append([])
            else:
                per_sample_pairs.append([(int(s), int(d)) for s, d in pairs_b.t().tolist()])

        edge_to_idx: Dict[Tuple[int, int], int] = {}
        stage_pairs: List[Tuple[int, int]] = []
        for pairs_b in per_sample_pairs:
            for pair in pairs_b:
                if pair not in edge_to_idx:
                    edge_to_idx[pair] = len(stage_pairs)
                    stage_pairs.append(pair)

        if not stage_pairs:
            empty_ei = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_mask = torch.zeros((B, 0), dtype=torch.bool, device=device)
            return empty_ei, empty_mask, int(fallback_used)

        stage_edge_index = torch.tensor(stage_pairs, dtype=torch.long, device=device).t()
        K = stage_edge_index.size(1)
        eligible_bk = torch.zeros((B, K), dtype=torch.bool, device=device)
        for b, pairs_b in enumerate(per_sample_pairs):
            if not pairs_b:
                continue
            idxs = [edge_to_idx[pair] for pair in pairs_b if pair in edge_to_idx]
            if idxs:
                eligible_bk[b, torch.tensor(idxs, dtype=torch.long, device=device)] = True

        return stage_edge_index, eligible_bk, int(fallback_used)

    def _zipper_build_stage_candidates_with_mask_fast(
        self,
        parent_edge_index: torch.Tensor,
        parent_keep_bk: torch.Tensor,
        children_table: Dict[str, torch.Tensor],
        fallback_level: int,
        fallback_cap: int,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B = int(parent_keep_bk.size(0))
        N = int(node_level.numel())

        row_of_node = children_table.get("row_of_node")
        child_table = children_table.get("children")
        child_counts = children_table.get("counts")

        cand_b = torch.empty((0,), dtype=torch.long, device=device)
        cand_src = torch.empty((0,), dtype=torch.long, device=device)
        cand_dst = torch.empty((0,), dtype=torch.long, device=device)

        selected = parent_keep_bk.nonzero(as_tuple=False)
        if (
            selected.numel() > 0
            and row_of_node is not None
            and child_table is not None
            and child_counts is not None
            and child_table.numel() > 0
        ):
            b_sel = selected[:, 0]
            k_sel = selected[:, 1]
            src_parent = parent_edge_index[0, k_sel]
            dst_parent = parent_edge_index[1, k_sel]

            row_src = row_of_node[src_parent]
            row_dst = row_of_node[dst_parent]
            valid_parent = (row_src >= 0) & (row_dst >= 0)
            if valid_parent.any():
                b_sel = b_sel[valid_parent]
                row_src = row_src[valid_parent]
                row_dst = row_dst[valid_parent]

                cnt_src = child_counts[row_src]
                cnt_dst = child_counts[row_dst]
                valid_parent = (cnt_src > 0) & (cnt_dst > 0)
                if valid_parent.any():
                    b_sel = b_sel[valid_parent]
                    row_src = row_src[valid_parent]
                    row_dst = row_dst[valid_parent]
                    cnt_src = cnt_src[valid_parent]
                    cnt_dst = cnt_dst[valid_parent]

                    C = int(child_table.size(1))
                    src_children = child_table[row_src]
                    dst_children = child_table[row_dst]

                    src_exp = src_children.unsqueeze(2).expand(-1, C, C)
                    dst_exp = dst_children.unsqueeze(1).expand(-1, C, C)
                    idx = torch.arange(C, device=device, dtype=torch.long)
                    valid_src = idx.view(1, C, 1) < cnt_src.view(-1, 1, 1)
                    valid_dst = idx.view(1, 1, C) < cnt_dst.view(-1, 1, 1)
                    valid = valid_src & valid_dst & (src_exp >= 0) & (dst_exp >= 0) & (src_exp != dst_exp)

                    b_exp = b_sel.view(-1, 1, 1).expand(-1, C, C)
                    cand_b = b_exp[valid]
                    cand_src = src_exp[valid]
                    cand_dst = dst_exp[valid]

        stage_cap = int(self.zip_max_candidate_edges)
        if cand_b.numel() > 0 and stage_cap > 0:
            order = torch.argsort(cand_b)
            b_sorted = cand_b[order]
            src_sorted = cand_src[order]
            dst_sorted = cand_dst[order]

            _, counts = torch.unique_consecutive(b_sorted, return_counts=True)
            starts = torch.cumsum(counts, dim=0) - counts
            rank = torch.arange(b_sorted.numel(), device=device, dtype=torch.long) - torch.repeat_interleave(starts, counts)
            keep = rank < stage_cap

            cand_b = b_sorted[keep]
            cand_src = src_sorted[keep]
            cand_dst = dst_sorted[keep]

        if cand_b.numel() > 0 and self.hier_ar_enable and self.hier_ar_filter_zip and node_ar_time is not None:
            if self.hier_ar_allow_same_time:
                keep_ar = node_ar_time[cand_src] <= node_ar_time[cand_dst]
            else:
                keep_ar = node_ar_time[cand_src] < node_ar_time[cand_dst]
            if not self.l0_ar_enable:
                both_l0 = (node_level[cand_src] == 0) & (node_level[cand_dst] == 0)
                keep_ar = keep_ar | both_l0
            cand_b = cand_b[keep_ar]
            cand_src = cand_src[keep_ar]
            cand_dst = cand_dst[keep_ar]

        if cand_b.numel() > 0:
            key_bs = (cand_b.to(torch.long) * N + cand_src.to(torch.long)) * N + cand_dst.to(torch.long)
            key_bs = torch.unique(key_bs, sorted=True)
            cand_dst = key_bs % N
            tmp = torch.div(key_bs, N, rounding_mode="floor")
            cand_src = tmp % N
            cand_b = torch.div(tmp, N, rounding_mode="floor")

        if cand_b.numel() > 0:
            sample_counts = torch.bincount(cand_b, minlength=B)
        else:
            sample_counts = torch.zeros((B,), dtype=torch.long, device=device)

        missing = (sample_counts == 0).nonzero(as_tuple=False).view(-1)
        fallback_used = int(missing.numel())

        if missing.numel() > 0:
            fb = self._zipper_dense_pairs_from_level(
                node_level=node_level,
                level=int(fallback_level),
                cap=int(fallback_cap),
                device=device,
            )
            fb = self._zipper_filter_ar_edges(fb, node_level, node_ar_time)
            if fb.numel() > 0:
                Kf = int(fb.size(1))
                fb_b = missing.repeat_interleave(Kf)
                fb_src = fb[0].repeat(int(missing.numel()))
                fb_dst = fb[1].repeat(int(missing.numel()))

                if cand_b.numel() == 0:
                    cand_b, cand_src, cand_dst = fb_b, fb_src, fb_dst
                else:
                    cand_b = torch.cat([cand_b, fb_b], dim=0)
                    cand_src = torch.cat([cand_src, fb_src], dim=0)
                    cand_dst = torch.cat([cand_dst, fb_dst], dim=0)

                key_bs = (cand_b.to(torch.long) * N + cand_src.to(torch.long)) * N + cand_dst.to(torch.long)
                key_bs = torch.unique(key_bs, sorted=True)
                cand_dst = key_bs % N
                tmp = torch.div(key_bs, N, rounding_mode="floor")
                cand_src = tmp % N
                cand_b = torch.div(tmp, N, rounding_mode="floor")

        if cand_b.numel() == 0:
            empty_ei = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_mask = torch.zeros((B, 0), dtype=torch.bool, device=device)
            return empty_ei, empty_mask, int(fallback_used)

        edge_key = cand_src.to(torch.long) * N + cand_dst.to(torch.long)
        edge_key, inv = torch.unique(edge_key, sorted=True, return_inverse=True)
        stage_dst = edge_key % N
        stage_src = torch.div(edge_key, N, rounding_mode="floor")
        stage_edge_index = torch.stack([stage_src, stage_dst], dim=0)

        K = int(stage_edge_index.size(1))
        eligible_bk = torch.zeros((B, K), dtype=torch.bool, device=device)
        eligible_bk[cand_b, inv] = True

        return stage_edge_index, eligible_bk, int(fallback_used)

    def _zipper_run_ephemeral_stage_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        valid_bk: Optional[torch.Tensor] = None,
        x_in: Optional[torch.Tensor] = None,
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        device = x_bnh.device
        B = x_bnh.size(0)
        stage_stats: Dict[str, int] = {
            "candidates": int(edge_index.size(1)),
            "selected_union": 0,
            "selected_total": 0,
        }
        if edge_index.numel() == 0:
            empty_keep = torch.zeros((B, 0), dtype=torch.bool, device=device)
            empty_attn = torch.zeros((B, 0), dtype=x_bnh.dtype, device=device)
            return empty_keep, empty_attn, stage_stats

        if x_in is None:
            x_in = transformer.norm1(x_bnh) if hasattr(transformer, "norm1") else x_bnh

        # Phase 3: use precomputed Q/K if available
        if _qk_cache is not None:
            scores_bk = self._zipper_score_from_qk_cache(
                transformer=transformer,
                edge_index=edge_index,
                node_level=node_level,
                _qk_cache=_qk_cache,
                dtype=x_bnh.dtype,
            )
        else:
            scores_bk = self._zipper_score_candidates_batched(
                transformer=transformer,
                x_bnh=x_bnh,
                edge_index=edge_index,
                node_level=node_level,
                x_in=x_in,
            )
        attn_bk = self._zipper_candidate_softmax_batched(
            scores_bk=scores_bk,
            dst_idx=edge_index[1],
            num_nodes=x_bnh.size(1),
        )
        select_values = attn_bk if bool(getattr(self, "zip_select_from_softmax", True)) else scores_bk
        keep_bk = self._zipper_select_mask_batched(
            scores_bk=select_values,
            dst_idx=edge_index[1],
            valid_bk=valid_bk,
        )

        keep_union = keep_bk.any(dim=0)
        stage_stats["selected_union"] = int(keep_union.sum().item())
        stage_stats["selected_total"] = int(keep_bk.sum().item())
        return keep_bk, attn_bk, stage_stats

    def _zipper_inject_selected_messages_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        b_idx: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        weights: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
        apply_zip_gates: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        if b_idx.numel() == 0:
            return x_bnh, 0

        device = x_bnh.device
        B, N, _ = x_bnh.shape
        group_idx = b_idx * N + dst_idx

        eta = float(getattr(self, "zip_msg_eta", 1.0))
        warmup_steps = max(0, int(getattr(self, "zip_warmup_steps", 0)))
        if warmup_steps > 0:
            self._zip_msg_step = int(getattr(self, "_zip_msg_step", 0)) + 1
            warm = min(1.0, float(self._zip_msg_step) / float(warmup_steps))
            eta = eta * warm
        if eta != 1.0:
            weights = weights * eta

        mp = transformer.message_passing
        if x_in is None:
            x_in = transformer.norm1(x_bnh) if hasattr(transformer, "norm1") else x_bnh
        # Fat-V: wide value head dim on this zipper apply (== head_dim when off).
        v_hd = mp.sparse_v_head_dim
        v_all = mp.sparse_proj_v(x_in).view(B, N, mp.num_heads, v_hd)
        msg = v_all[b_idx, src_idx] * weights.view(-1, 1, 1)

        # Phase 5: reuse workspace buffer only when autograd is off.
        flat_size = B * N
        ws_key = (flat_size, mp.num_heads, v_hd, str(device), str(x_bnh.dtype))
        use_workspace_cache = not torch.is_grad_enabled()
        ws = self._zipper_inject_workspace.get(ws_key) if use_workspace_cache else None
        if ws is not None and ws.shape == (flat_size, mp.num_heads, v_hd) and ws.device == device and ws.dtype == x_bnh.dtype:
            out_flat = ws.zero_()
        else:
            out_flat = torch.zeros(flat_size, mp.num_heads, v_hd, device=device, dtype=x_bnh.dtype)
            # Cache for future reuse only when gradients are disabled.
            if use_workspace_cache:
                if len(self._zipper_inject_workspace) >= self._zipper_inject_workspace_max_entries:
                    oldest = next(iter(self._zipper_inject_workspace))
                    del self._zipper_inject_workspace[oldest]
                self._zipper_inject_workspace[ws_key] = out_flat
        out_flat.index_add_(0, group_idx, msg)
        m_zip = out_flat.view(B, N, mp.num_heads, v_hd).reshape(B, N, mp.num_heads * v_hd)
        m_zip = mp.sparse_out_proj(m_zip)

        if bool(apply_zip_gates):
            eta = float(getattr(self, "zip_msg_eta", 1.0))
            warmup_steps = max(0, int(getattr(self, "zip_warmup_steps", 0)))
            if warmup_steps > 0:
                self._zip_msg_step = int(getattr(self, "_zip_msg_step", 0)) + 1
                warm = min(1.0, float(self._zip_msg_step) / float(warmup_steps))
                eta = eta * warm
            if eta != 1.0:
                m_zip = m_zip * eta

            norm_clip = float(getattr(self, "zip_norm_clip_ratio", 0.0))
            if norm_clip > 0.0:
                x_norm = x_bnh.norm(dim=-1, keepdim=True)
                m_norm = m_zip.norm(dim=-1, keepdim=True)
                max_m = norm_clip * x_norm
                clip = torch.clamp(max_m / (m_norm + 1e-8), max=1.0)
                m_zip = m_zip * clip

            if (not bool(getattr(self, "zip_paramfree_gate", True))) and bool(getattr(self, "zip_use_beta_gate", True)):
                beta_layer = getattr(transformer, "zip_lin_beta_attn", None)
                if beta_layer is not None:
                    beta = torch.sigmoid(beta_layer(m_zip))
                    zip_dropout = transformer.dropout(m_zip) if hasattr(transformer, "dropout") else m_zip
                    m_zip = (1.0 - beta) * zip_dropout

        with torch.no_grad():
            x_norm = x_bnh.norm(dim=-1)
            m_norm = m_zip.norm(dim=-1)
            ratio = m_norm / (x_norm + 1e-8)
            self._last_zip_msg_norm_ratio_mean = float(ratio.mean().item())
            self._last_zip_msg_norm_ratio_max = float(ratio.max().item())

        return x_bnh + m_zip, int(b_idx.numel())

    def _zipper_children_table_cache_key(
        self,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        parent_level: int,
        child_level: int,
    ) -> Tuple[Any, ...]:
        """Build a lightweight hashable key for children table caching."""
        # Use data_ptr + shape + device as proxy for identity (avoids hashing tensor contents)
        return (
            int(edge_index.data_ptr()), int(edge_index.shape[1]),
            int(node_level.data_ptr()), int(node_level.numel()),
            int(parent_level), int(child_level),
            str(edge_index.device),
        )

    def _zipper_build_children_table_cached(
        self,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        parent_level: int,
        child_level: int,
    ) -> Dict[str, torch.Tensor]:
        """Cached version of _zipper_build_children_table. Cache is keyed on
        tensor identity (data_ptr + shape) which is valid within a single
        forward pass where edge_index/node_level don't change."""
        cache = self._zipper_children_table_cache
        key = self._zipper_children_table_cache_key(edge_index, node_level, parent_level, child_level)
        if key in cache:
            self._zipper_children_cache_hits += 1
            return cache[key]
        self._zipper_children_cache_misses += 1
        result = self._zipper_build_children_table(edge_index, node_level, parent_level, child_level)
        # Evict oldest if over capacity
        max_entries = max(1, int(self._zipper_children_table_cache_max_entries))
        if len(cache) >= max_entries:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[key] = result
        return result

    def _hqd_effective_causal(self) -> bool:
        if self.hqd_causal is not None:
            return bool(self.hqd_causal)
        return bool(self.hier_ar_enable and self.l0_ar_enable)

    def _hqd_skeleton_cache_key(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
    ) -> Tuple[Any, ...]:
        return (
            int(num_nodes),
            int(edge_index.data_ptr()), int(edge_index.shape[1]), str(edge_index.device), str(edge_index.dtype),
            int(node_level.data_ptr()), int(node_level.numel()), str(node_level.device), str(node_level.dtype),
            0 if node_ar_time is None else int(node_ar_time.data_ptr()),
            0 if node_ar_time is None else int(node_ar_time.numel()),
            "none" if node_ar_time is None else str(node_ar_time.dtype),
            int(getattr(self, "zip_max_children_per_parent", 4)),
        )

    def _hqd_expand_children(
        self,
        parent_nodes: torch.Tensor,
        children_table: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        device = parent_nodes.device if torch.is_tensor(parent_nodes) else self.device
        if parent_nodes is None or parent_nodes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        row_of_node = children_table.get("row_of_node", None)
        child_table = children_table.get("children", None)
        if row_of_node is None or child_table is None or child_table.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        rows = row_of_node.index_select(0, parent_nodes.to(device=row_of_node.device, dtype=torch.long))
        rows = rows[rows >= 0]
        if rows.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        children = child_table.index_select(0, rows).reshape(-1)
        children = children[children >= 0]
        if children.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        return torch.unique(children.to(device=device, dtype=torch.long), sorted=False)

    def _hqd_children_table_is_disjoint(self, children_table: Dict[str, torch.Tensor]) -> bool:
        child_table = children_table.get("children", None)
        if child_table is None or child_table.numel() == 0:
            return True
        valid_children = child_table[child_table >= 0]
        if valid_children.numel() <= 1:
            return True
        return int(torch.unique(valid_children, sorted=False).numel()) == int(valid_children.numel())

    def _hqd_build_descendant_interval_cache(
        self,
        level_nodes: torch.Tensor,
        lower_min_cache: torch.Tensor,
        lower_max_cache: torch.Tensor,
        children_table: Dict[str, torch.Tensor],
        total_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = lower_min_cache.device
        min_cache = torch.full((int(total_nodes),), -1, dtype=torch.long, device=device)
        max_cache = torch.full((int(total_nodes),), -1, dtype=torch.long, device=device)
        if level_nodes is None or level_nodes.numel() == 0:
            return min_cache, max_cache

        parent_ids = children_table.get("parent_ids", None)
        child_table = children_table.get("children", None)
        if parent_ids is None or child_table is None or child_table.numel() == 0:
            return min_cache, max_cache

        if parent_ids.numel() == 0:
            return min_cache, max_cache

        parent_ids = parent_ids.to(device=device, dtype=torch.long)
        child_table = child_table.to(device=device, dtype=torch.long)
        child_valid = child_table >= 0
        if not bool(child_valid.any()):
            return min_cache, max_cache

        safe_children = child_table.clamp_min(0)
        child_min = lower_min_cache.index_select(0, safe_children.reshape(-1)).view_as(child_table)
        child_max = lower_max_cache.index_select(0, safe_children.reshape(-1)).view_as(child_table)

        valid_min = child_valid & (child_min >= 0)
        valid_max = child_valid & (child_max >= 0)
        huge = torch.full_like(child_min, torch.iinfo(torch.long).max)
        neg = torch.full_like(child_max, -1)
        row_min = torch.where(valid_min, child_min, huge).amin(dim=-1)
        row_max = torch.where(valid_max, child_max, neg).amax(dim=-1)
        has_any = row_max >= 0
        if bool(has_any.any()):
            min_cache[parent_ids[has_any]] = row_min[has_any]
            max_cache[parent_ids[has_any]] = row_max[has_any]

        return min_cache, max_cache

    def _hqd_local_window_candidates(
        self,
        query_idx: int,
        num_l0: int,
        window_size: int,
        causal: bool,
        allow_same_time: bool,
        device: torch.device,
    ) -> torch.Tensor:
        if int(num_l0) <= 0 or int(window_size) <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        q = int(query_idx)
        w = max(1, int(window_size))
        if causal:
            end = q + (1 if allow_same_time else 0)
            start = max(0, q - w)
        else:
            start = max(0, q - w)
            end = min(int(num_l0), q + w + 1)
        start = max(0, min(start, int(num_l0)))
        end = max(start, min(end, int(num_l0)))
        if end <= start:
            return torch.empty((0,), dtype=torch.long, device=device)
        return torch.arange(start, end, device=device, dtype=torch.long)

    def _hqd_score_topk_candidates(
        self,
        query_vec: torch.Tensor,
        key_bank: torch.Tensor,
        candidate_idx: torch.Tensor,
        topk: int,
        query_start: Optional[int] = None,
        candidate_start_cache: Optional[torch.Tensor] = None,
        causal: bool = False,
        allow_same_time: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = query_vec.device
        if candidate_idx is None or candidate_idx.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=query_vec.dtype, device=device),
            )

        cand = candidate_idx.to(device=device, dtype=torch.long).reshape(-1)
        if cand.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=query_vec.dtype, device=device),
            )

        if causal and candidate_start_cache is not None and query_start is not None:
            cand_starts = candidate_start_cache.index_select(0, cand.to(device=candidate_start_cache.device))
            if allow_same_time:
                keep = cand_starts <= int(query_start)
            else:
                keep = cand_starts < int(query_start)
            cand = cand[keep.to(device=device)]

        if cand.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=query_vec.dtype, device=device),
            )

        cand = torch.unique(cand, sorted=False)
        if cand.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=query_vec.dtype, device=device),
            )

        cand_keys = key_bank.index_select(0, cand.to(device=key_bank.device))
        head_dim = max(1, int(query_vec.size(-1)))
        num_heads = max(1, int(query_vec.size(-2)))
        scores = torch.einsum("hd,nhd->n", query_vec, cand_keys)
        scores = scores / float(num_heads * math.sqrt(float(head_dim)))

        k = min(max(1, int(topk)), int(cand.numel()))
        top_scores, top_pos = torch.topk(scores, k=k, largest=True, sorted=False)
        return cand.index_select(0, top_pos), top_scores

    def _hqd_expand_children_batched(
        self,
        parent_nodes: torch.Tensor,
        children_table: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row_of_node = children_table.get("row_of_node", None)
        child_table = children_table.get("children", None)
        if parent_nodes.numel() == 0 or row_of_node is None or child_table is None or child_table.numel() == 0:
            empty_nodes = torch.empty((*parent_nodes.shape, 0), dtype=torch.long, device=parent_nodes.device)
            empty_mask = torch.empty((*parent_nodes.shape, 0), dtype=torch.bool, device=parent_nodes.device)
            return empty_nodes, empty_mask

        safe_parent = parent_nodes.clamp(min=0)
        parent_rows = row_of_node.index_select(0, safe_parent.reshape(-1)).view_as(parent_nodes)
        valid_parent = (parent_nodes >= 0) & (parent_rows >= 0)
        safe_rows = parent_rows.clamp(min=0)
        children = child_table.index_select(0, safe_rows.reshape(-1)).view(*parent_nodes.shape, -1)
        valid_children = valid_parent.unsqueeze(-1) & (children >= 0)
        return children, valid_children

    def _hqd_static_descent_callable(self):
        """torch.compile the static descent core, lazily, with the same probation policy
        as _refresh_callable/_layer_callable: first runtime failure logs once and drops to
        eager permanently.

        This is the whole point of the fixed shapes. The stage-by-stage descent rebuilds
        candidate sets with data-dependent shapes, so dynamo graph-breaks at every level;
        the static core is one shape-stable graph, so the gather/mask/topk chains fuse.
        Unlike the layer compile this is NOT gated on self.training — the descent runs
        under no_grad in both modes and its shapes do not vary between them.
        """
        if not bool(getattr(self, "hqd_static_compile", False)):
            return self._hqd_descent_static_core
        fn = getattr(self, "_hqd_static_compiled", None)
        if fn is None:
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.cache_size_limit = max(int(_dynamo.config.cache_size_limit), 64)
                compiled = torch.compile(self._hqd_descent_static_core, dynamic=False)
            except Exception as exc:
                logger.warning("hqd_static_compile: torch.compile unavailable (%s); staying eager.", exc)
                self._hqd_static_compiled = self._hqd_descent_static_core
                return self._hqd_descent_static_core

            def _probation(*args, _c=compiled, **kwargs):
                try:
                    return _c(*args, **kwargs)
                except Exception as err:
                    logger.warning(
                        "hqd_static_compile: compiled descent failed at runtime (%s: %s); "
                        "falling back to eager permanently.", type(err).__name__, err,
                    )
                    self._hqd_static_compiled = self._hqd_descent_static_core
                    return self._hqd_descent_static_core(*args, **kwargs)

            self._hqd_static_compiled = _probation
            fn = _probation
        return fn

    def _hqd_descent_static_core(
        self,
        q_vec: torch.Tensor,
        k_all: torch.Tensor,
        l3_idx: torch.Tensor,
        tables: Tuple[Any, Any, Any],
        topks: Tuple[int, int, int, int],
        query_time: torch.Tensor,
        max_time: Any,
        causal: bool,
        allow_same_time: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed-shape 4-stage beam. Returns (nodes [B,Q,K], valid [B,Q,K]).

        Every shape here is a function of (B, Q, topks, children-table widths) only — no
        dedup, no boolean-mask compaction, no torch.cat — so the whole descent is a single
        static graph and a legal torch.compile target. The stage-by-stage path rebuilds
        candidate sets with data-dependent shapes at every level, which is what forces it
        into ~13 kernels per stage and blocks fusion.

        Selection matches the original when dedup is off: duplicate candidates only ever
        consume a top-k slot, they cannot change which nodes score highest.

        Levels are walked as a LOOP over (table, topk), not four unrolled blocks, so a
        deeper hierarchy is a longer tuple rather than new code.
        """
        num_heads = int(q_vec.size(-2))
        head_dim = int(q_vec.size(-1))
        scale = float(num_heads * math.sqrt(float(head_dim)))
        b_sz, q_sz = int(q_vec.size(0)), int(q_vec.size(1))

        # Root stage: score every L3 node. This level is small by construction; it is the
        # only stage that touches a whole level, and the reason depth must grow with N.
        k_l3 = k_all.index_select(1, l3_idx)
        scores = torch.einsum("bqhd,bkhd->bqk", q_vec, k_l3) / scale
        if causal and max_time is not None:
            t3 = max_time[3].index_select(0, l3_idx).view(1, 1, -1)
            ok = (t3 <= query_time.unsqueeze(-1)) if allow_same_time else (t3 < query_time.unsqueeze(-1))
            scores = scores.masked_fill(~(ok & (t3 >= 0)), float("-inf"))
        k3 = max(1, min(int(topks[0]), int(l3_idx.numel())))
        top_scores, top_pos = torch.topk(scores, k=k3, dim=-1, largest=True, sorted=False)
        sel = l3_idx.index_select(0, top_pos.reshape(-1)).view(b_sz, q_sz, k3)
        sel = torch.where(torch.isfinite(top_scores), sel, torch.full_like(sel, -1))

        # hqd_keep_stage_survivors: the descent already scores and ranks a top-k at EVERY
        # level and then throws all of it away except the final L0 set. Keeping the
        # survivors costs nothing to produce and turns the read from "32 distant raw
        # tokens" into a MIXED-LEVEL set: 4 L3 + 4 L2 + 4 L1 + 32 L0 = 44 rows whose
        # combined span is ~672 L0-token-equivalents instead of 32, for +37% rows.
        #
        # Motivated by bench_pack_read.py: an L0 query spends ~42% of its packed-window
        # mass on coarse rows in BOTH the windowed and full-window runs, but the
        # full-window run (the one that composes) spends it on distant L1 (0.288) while
        # the windowed one leans on L3 (0.092). It wants distant SUMMARIES, not only
        # distant tokens -- and it still puts 0.58 on L0, so fetching L1 alone
        # (hqd_stop_level: 1) would trade one starvation for another.
        #
        # Static-shape safe: every stage's top-k is a fixed [B,Q,k], so the cat is a
        # fixed [B,Q,sum(k)] and the whole descent stays one compilable graph.
        keep_survivors = bool(getattr(self, "hqd_keep_stage_survivors", False))
        survivors: List[torch.Tensor] = [sel] if keep_survivors else []

        # Descent: expand -> score -> narrow, identical five ops per level.
        for table, topk, lvl in zip(tables, topks[1:], (2, 1, 0)):
            cand, cand_mask = self._hqd_expand_children_batched(sel, table)
            cand = cand.reshape(b_sz, q_sz, -1)
            cand_mask = cand_mask.reshape(b_sz, q_sz, -1)
            cand = torch.where(cand_mask, cand, torch.full_like(cand, -1))
            # Dedup is sort + compare + where -- all fixed shape, so it stays inside the
            # static graph. Kept for exact parity with the reference descent; turn it off
            # with hqd_assume_disjoint_children to drop the sort.
            cand, cand_mask = self._hqd_maybe_dedup_candidates_batched(cand)
            scores = self._hqd_score_candidates_batched(
                query_vec=q_vec,
                key_bank=k_all,
                candidate_nodes=cand,
                candidate_mask=cand_mask,
                query_time=query_time,
                candidate_max_time_cache=(max_time[lvl] if max_time is not None else None),
                causal=causal,
                allow_same_time=allow_same_time,
            )
            sel, _, _ = self._hqd_topk_from_scores_batched(cand, scores, int(topk))
            if keep_survivors:
                survivors.append(sel)

        if keep_survivors:
            # Levels occupy disjoint node-id ranges, so the union needs no dedup: an L0
            # survivor can never collide with its own L1 parent.
            sel = torch.cat(survivors, dim=-1)
        return sel, sel >= 0

    def _hqd_dedup_sorted_candidates_batched(
        self,
        candidate_nodes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidate_nodes.numel() == 0:
            return candidate_nodes, torch.empty_like(candidate_nodes, dtype=torch.bool)

        sorted_nodes, _ = torch.sort(candidate_nodes, dim=-1)
        keep = sorted_nodes >= 0
        if sorted_nodes.size(-1) > 1:
            keep[..., 1:] = keep[..., 1:] & (sorted_nodes[..., 1:] != sorted_nodes[..., :-1])
        deduped = torch.where(keep, sorted_nodes, torch.full_like(sorted_nodes, -1))
        return deduped, keep

    def _hqd_maybe_dedup_candidates_batched(
        self,
        candidate_nodes: torch.Tensor,
        force: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidate_nodes.numel() == 0:
            return candidate_nodes, torch.empty_like(candidate_nodes, dtype=torch.bool)
        if bool(getattr(self, "hqd_assume_disjoint_children", False)) and not bool(force):
            keep = candidate_nodes >= 0
            return torch.where(keep, candidate_nodes, torch.full_like(candidate_nodes, -1)), keep
        return self._hqd_dedup_sorted_candidates_batched(candidate_nodes)

    def _hqd_local_window_candidates_batched(
        self,
        query_positions: torch.Tensor,
        l0_nodes: torch.Tensor,
        window_size: int,
        causal: bool,
        allow_same_time: bool,
    ) -> torch.Tensor:
        if int(window_size) <= 0 or query_positions.numel() == 0 or l0_nodes.numel() == 0:
            return torch.empty((*query_positions.shape, 0), dtype=torch.long, device=l0_nodes.device)

        w = max(1, int(window_size))
        if causal:
            offsets = torch.arange(-w, 1 if allow_same_time else 0, device=l0_nodes.device, dtype=torch.long)
        else:
            offsets = torch.arange(-w, w + 1, device=l0_nodes.device, dtype=torch.long)
        pos = query_positions.unsqueeze(-1) + offsets.unsqueeze(0)
        valid = (pos >= 0) & (pos < int(l0_nodes.numel()))
        safe_pos = pos.clamp(0, int(l0_nodes.numel()) - 1)
        nodes = l0_nodes.index_select(0, safe_pos.reshape(-1)).view(*safe_pos.shape)
        return torch.where(valid, nodes, torch.full_like(nodes, -1))

    def _hqd_score_candidates_batched(
        self,
        query_vec: torch.Tensor,
        key_bank: torch.Tensor,
        candidate_nodes: torch.Tensor,
        candidate_mask: torch.Tensor,
        query_time: torch.Tensor,
        candidate_max_time_cache: Optional[torch.Tensor],
        causal: bool,
        allow_same_time: bool,
    ) -> torch.Tensor:
        if candidate_nodes.numel() == 0:
            return torch.empty((*candidate_nodes.shape[:-1], 0), device=query_vec.device, dtype=query_vec.dtype)

        safe_nodes = candidate_nodes.clamp(min=0)
        B = int(query_vec.size(0))
        N = int(key_bank.size(1))
        head_dim = max(1, int(query_vec.size(-1)))
        num_heads = max(1, int(query_vec.size(-2)))
        Q = int(safe_nodes.size(1))
        n_cand = int(safe_nodes.size(-1))
        # The head-summed score is one full-width dot product per (query, candidate). Gathering
        # per-candidate KEY VECTORS ([B,Q,n,H,D] — gigabytes of traffic) to einsum them is far
        # slower than one dense GEMM over the whole bank ([B,Q,N] scores) + a score gather,
        # whenever the dense score matrix is affordable and the vector gather would move more.
        use_dense = (B * Q * N) <= 128_000_000 and (n_cand * num_heads * head_dim) > (2 * N)
        if use_dense:
            q_flat = query_vec.reshape(B, Q, num_heads * head_dim)
            k_flat = key_bank.reshape(B, N, num_heads * head_dim)
            dense = torch.bmm(q_flat, k_flat.transpose(1, 2))  # [B, Q, N]
            scores = dense.gather(-1, safe_nodes)
        else:
            flat_key_bank = key_bank.reshape(B * N, num_heads, head_dim)
            batch_offsets = (torch.arange(B, device=safe_nodes.device, dtype=torch.long).view(B, 1, 1) * N)
            gather_idx = (safe_nodes + batch_offsets).reshape(-1)
            cand_keys = flat_key_bank.index_select(0, gather_idx).view(*safe_nodes.shape, num_heads, head_dim)
            scores = torch.einsum("bqhd,bqnhd->bqn", query_vec, cand_keys)
        scores = scores / float(num_heads * math.sqrt(float(head_dim)))
        scores = scores.masked_fill(~candidate_mask, float("-inf"))

        if causal and candidate_max_time_cache is not None:
            cand_max_time = candidate_max_time_cache.index_select(0, safe_nodes.reshape(-1)).view_as(safe_nodes)
            if allow_same_time:
                causal_mask = cand_max_time <= query_time.unsqueeze(-1)
            else:
                causal_mask = cand_max_time < query_time.unsqueeze(-1)
            scores = scores.masked_fill(~(candidate_mask & causal_mask & (cand_max_time >= 0)), float("-inf"))

        return scores

    def _hqd_topk_from_scores_batched(
        self,
        candidate_nodes: torch.Tensor,
        scores: torch.Tensor,
        topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_nodes.numel() == 0:
            empty_nodes = torch.empty((*candidate_nodes.shape[:-1], 0), dtype=torch.long, device=candidate_nodes.device)
            empty_scores = torch.empty((*candidate_nodes.shape[:-1], 0), dtype=scores.dtype, device=scores.device)
            empty_mask = torch.empty((*candidate_nodes.shape[:-1], 0), dtype=torch.bool, device=candidate_nodes.device)
            return empty_nodes, empty_scores, empty_mask

        k = min(max(1, int(topk)), int(candidate_nodes.size(-1)))
        top_scores, top_pos = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False)
        top_nodes = candidate_nodes.gather(-1, top_pos)
        top_valid = torch.isfinite(top_scores)
        top_nodes = torch.where(top_valid, top_nodes, torch.full_like(top_nodes, -1))
        return top_nodes, top_scores, top_valid

    def _hqd_apply_per_batch_global_topk(
        self,
        scores: torch.Tensor,
        valid: torch.Tensor,
        global_topk: int,
    ) -> torch.Tensor:
        if int(global_topk) <= 0:
            return valid

        B = int(scores.size(0))
        flat_scores = scores.masked_fill(~valid, float("-inf")).reshape(B, -1)
        k = min(int(global_topk), int(flat_scores.size(1)))
        if k <= 0:
            return valid & False

        top_scores, top_pos = torch.topk(flat_scores, k=k, dim=-1, largest=True, sorted=False)
        keep_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
        keep_flat.scatter_(1, top_pos, torch.isfinite(top_scores))
        return keep_flat.view_as(valid) & valid

    def _hqd_expand_to_l0_batched(
        self,
        nodes: torch.Tensor,
        level: int,
        l2_to_l1: Dict[str, torch.Tensor],
        l1_to_l0: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        level = int(level)
        if nodes.numel() == 0:
            return nodes, torch.empty_like(nodes, dtype=torch.bool)
        if level == 0:
            mask = nodes >= 0
            return torch.where(mask, nodes, torch.full_like(nodes, -1)), mask
        if level == 1:
            child_nodes, child_valid = self._hqd_expand_children_batched(nodes, l1_to_l0)
            child_nodes = child_nodes.reshape(*nodes.shape[:-1], -1)
            child_valid = child_valid.reshape(*nodes.shape[:-1], -1)
            child_nodes = torch.where(child_valid, child_nodes, torch.full_like(child_nodes, -1))
            return self._hqd_maybe_dedup_candidates_batched(child_nodes, force=True)
        if level == 2:
            l1_nodes, l1_valid = self._hqd_expand_children_batched(nodes, l2_to_l1)
            l1_nodes = l1_nodes.reshape(*nodes.shape[:-1], -1)
            l1_valid = l1_valid.reshape(*nodes.shape[:-1], -1)
            l1_nodes = torch.where(l1_valid, l1_nodes, torch.full_like(l1_nodes, -1))
            l1_nodes, _ = self._hqd_maybe_dedup_candidates_batched(l1_nodes, force=True)
            l0_nodes, l0_valid = self._hqd_expand_children_batched(l1_nodes, l1_to_l0)
            l0_nodes = l0_nodes.reshape(*nodes.shape[:-1], -1)
            l0_valid = l0_valid.reshape(*nodes.shape[:-1], -1)
            l0_nodes = torch.where(l0_valid, l0_nodes, torch.full_like(l0_nodes, -1))
            return self._hqd_maybe_dedup_candidates_batched(l0_nodes, force=True)
        mask = nodes >= 0
        return torch.where(mask, nodes, torch.full_like(nodes, -1)), mask

    def _witness_l0_surprise(
        self,
        x_bnh: torch.Tensor,
        node_level: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Detached per-L0-node 'surprise' = reconstruction residual of each L0 node
        from the mean of its L1 parents (same signal as the hierarchy aux loss, but
        per-node and not reduced). Returns [B, N], zero for non-L0 / parentless nodes."""
        B, N, H = x_bnh.shape
        device = x_bnh.device
        surprise = torch.zeros(B, N, device=device, dtype=x_bnh.dtype)
        if edge_index is None or edge_index.numel() == 0:
            return surprise
        # The L0<->L1 edge extraction is pure topology — cache it per edge tensor (static
        # skeleton) instead of re-scanning every HQD layer.
        cache_key = (edge_index.data_ptr(), int(edge_index.size(1)), int(N), str(device))
        cache = getattr(self, "_witness_surprise_idx_cache", None)
        if cache is None:
            cache = {}
            self._witness_surprise_idx_cache = cache
        hit = cache.get(cache_key)
        if hit is None:
            src, dst = edge_index
            mask_lh = (node_level[src] == 0) & (node_level[dst] == 1)
            mask_hl = (node_level[src] == 1) & (node_level[dst] == 0)
            child_idx = torch.cat([src[mask_lh], dst[mask_hl]], dim=0)   # L0 nodes
            parent_idx = torch.cat([dst[mask_lh], src[mask_hl]], dim=0)  # L1 nodes
            if len(cache) > 8:
                cache.clear()
            cache[cache_key] = (child_idx, parent_idx)
        else:
            child_idx, parent_idx = hit
        if child_idx.numel() == 0:
            return surprise
        agg = torch.zeros(B, N, H, device=device, dtype=x_bnh.dtype)
        cnt = torch.zeros(B, N, 1, device=device, dtype=x_bnh.dtype)
        agg.index_add_(1, child_idx, x_bnh.index_select(1, parent_idx))
        ones = torch.ones(B, child_idx.numel(), 1, device=device, dtype=x_bnh.dtype)
        cnt.index_add_(1, child_idx, ones)
        has_parent = (cnt.squeeze(-1) > 0)
        pred = agg / cnt.clamp_min(1.0)
        resid = ((pred - x_bnh) ** 2).mean(dim=-1)  # [B, N]
        return (resid * has_parent.to(resid.dtype)).detach()

    def _hqd_compose_l0_children(
        self,
        parent_ids: torch.Tensor,
        level: int,
        child_tables: Dict[int, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compose a [P, C] table of all L0 descendant ids (-1 padded) for parents at
        `level`, by chaining the per-level children tables down to L0. Used to build
        cross-level (L2->L0, L3->L0) witness bags."""
        desc = parent_ids.view(-1, 1)                       # [P, 1]
        for lvl in range(int(level), 0, -1):
            table = child_tables.get(int(lvl))
            if table is None:
                return torch.full((int(parent_ids.numel()), 0), -1, dtype=torch.long, device=parent_ids.device)
            ch, cv = self._hqd_expand_children_batched(desc, table)
            ch = torch.where(cv, ch, torch.full_like(ch, -1))
            desc = ch.reshape(desc.size(0), -1)             # [P, C_so_far]
        return desc                                          # [P, C] L0 ids

    def _build_witness_l0_ids(
        self,
        q_all: torch.Tensor,
        k_all: torch.Tensor,
        child_tables: Dict[int, Dict[str, torch.Tensor]],
        level_indices: Dict[int, torch.Tensor],
        node_ar_time: Optional[torch.Tensor],
        x_bnh: torch.Tensor,
        node_level: torch.Tensor,
        edge_index: torch.Tensor,
        allow_same_time: bool,
        causal: bool,
    ) -> Optional[torch.Tensor]:
        """Build a [B, N, slots] L0-id bag table (-1 padded), populating the rows of each
        parent level in self.witness_levels. L1 uses its direct L0 children; L2/L3 compose
        their full L0 descendant set (cross-level bags). None if nothing to build.

        witness_reuse == "step": build once per forward (first HQD layer) and reuse across
        the step's remaining HQD layers — q/k drift slowly layer-to-layer and witness
        selection is a coarse top-k, so the table is stable; this cuts the per-layer rebuild
        to one. Ids are Long (no autograd); under gradient checkpointing the recompute hits
        the same epoch -> identical table -> consistent recomputed activations.

        hqd_packed_witness_source == "backbone": compose the table from the packed
        cross-level lane's captured direct-child attention top-k (near-free — the scores
        were already computed by the backbone) instead of re-scoring descendants with the
        current q/k. Requires cross_level_packed; falls back to the recompute build until
        the first capture of the forward lands (e.g. the layer-0 HQD build). Backbone
        tables are ALWAYS step-cached (keyed on the forward epoch): the capture comes from
        one specific earlier layer anyway, and under gradient checkpointing the epoch-keyed
        cache keeps recomputed activations identical."""
        B, N = int(q_all.size(0)), int(q_all.size(1))
        device = q_all.device
        epoch = int(getattr(self, "_witness_build_epoch", -1))
        latest = getattr(self, "_backbone_witness_latest", None)
        use_backbone = (
            str(getattr(self, "hqd_packed_witness_source", "graph")).lower() == "backbone"
            and not bool(self.summary_witness_per_head)  # capture is head-averaged
            and latest is not None
            and int(latest[0]) == epoch
            and isinstance(latest[1], dict)
            and isinstance(latest[2], dict)
        )
        mode = "backbone" if use_backbone else "recompute"
        if use_backbone or str(getattr(self, "witness_reuse", "layer")).lower() == "step":
            key = (mode, epoch, B, N, str(device))
            cached = getattr(self, "_witness_table_step_cache", None)
            if cached is not None and cached[0] == key:
                return cached[1]
        else:
            key = None
        if use_backbone:
            out = self._build_witness_l0_ids_backbone(
                cap_ids=latest[1], cap_scores=latest[2],
                level_indices=level_indices, node_ar_time=node_ar_time,
                x_bnh=x_bnh, node_level=node_level, edge_index=edge_index,
                allow_same_time=allow_same_time, causal=causal,
            )
            if out is not None:
                if key is not None:
                    self._witness_table_step_cache = (key, out)
                return out
            # Composition failed (e.g. a witness level missing from the capture) ->
            # fall through to the recompute build for this call.
        out: Optional[torch.Tensor] = None
        for level in self.witness_levels:
            if int(level) == 1:
                table = child_tables.get(1)
                if table is None:
                    continue
                parent_ids = table.get("parent_ids")
                children = table.get("children")
            else:
                parent_ids = level_indices.get(int(level))
                if parent_ids is None or parent_ids.numel() == 0:
                    continue
                children = self._hqd_compose_l0_children(parent_ids, int(level), child_tables)
            if parent_ids is None or children is None or parent_ids.numel() == 0 or children.size(-1) == 0:
                continue
            packed = self._build_witness_packet(
                parent_ids=parent_ids, children=children, q_all=q_all, k_all=k_all,
                node_ar_time=node_ar_time, x_bnh=x_bnh, node_level=node_level,
                edge_index=edge_index, allow_same_time=allow_same_time, causal=causal,
            )
            if packed is None:
                continue
            slots = int(packed.size(-1))
            if out is None:
                out = torch.full((B, N, slots), -1, dtype=torch.long, device=device)
            elif slots != out.size(-1):
                # Pad to a common slot width (levels share k config, so normally equal).
                w = max(slots, out.size(-1))
                if out.size(-1) < w:
                    out = torch.cat([out, torch.full((B, N, w - out.size(-1)), -1, dtype=torch.long, device=device)], dim=-1)
                if slots < w:
                    packed = torch.cat([packed, torch.full((B, int(parent_ids.numel()), w - slots), -1, dtype=torch.long, device=device)], dim=-1)
            out.index_copy_(1, parent_ids, packed)
        if key is not None:
            self._witness_table_step_cache = (key, out)
        return out

    def _build_witness_l0_ids_backbone(
        self,
        cap_ids: Dict[int, torch.Tensor],
        cap_scores: Dict[int, torch.Tensor],
        level_indices: Dict[int, torch.Tensor],
        node_ar_time: Optional[torch.Tensor],
        x_bnh: torch.Tensor,
        node_level: torch.Tensor,
        edge_index: torch.Tensor,
        allow_same_time: bool,
        causal: bool,
    ) -> Optional[torch.Tensor]:
        """Witness table from the packed backbone's captured attention (see
        _build_witness_l0_ids). A parent's L0 bag is a beam over the captured
        per-level direct-child top-k: an L3 row's top L2 children -> their top L1
        children -> their top L0 children, ranked by the summed attention logits
        along the path. Summary slots take the beam top-k; rare slots re-rank with
        the detached surprise term (parity with the recompute build). Returns
        [B, N, slots] (-1 padded) or None if a needed capture level is missing."""
        B, N, _ = x_bnh.shape
        device = x_bnh.device
        k_sum = int(self.witness_k_summary) if self.use_summary_witnesses else 0
        k_rare = int(self.witness_k_rare) if self.use_rare_witnesses else 0
        slots = k_sum + k_rare
        if slots == 0:
            return None
        neg = torch.finfo(torch.float32).min

        def _hop(ids: torch.Tensor, sc: torch.Tensor, tbl_ids: torch.Tensor, tbl_sc: torch.Tensor):
            # Expand each candidate by its captured children: [B,P,W] -> [B,P,W*k].
            Kb = int(tbl_ids.size(-1))
            P, W = int(ids.size(1)), int(ids.size(2))
            off = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1) * N
            gi = (ids.clamp(min=0) + off).reshape(-1)
            nxt = tbl_ids.reshape(B * N, Kb).index_select(0, gi).view(B, P, W, Kb)
            nsc = tbl_sc.reshape(B * N, Kb).index_select(0, gi).view(B, P, W, Kb).float()
            ok = (ids >= 0).unsqueeze(-1) & (nxt >= 0)
            nsc = torch.where(ok, sc.unsqueeze(-1) + nsc, torch.full_like(nsc, neg))
            nxt = torch.where(ok, nxt, torch.full_like(nxt, -1))
            return nxt.reshape(B, P, W * Kb), nsc.reshape(B, P, W * Kb)

        out: Optional[torch.Tensor] = None
        surprise: Optional[torch.Tensor] = None
        t_all = None
        if causal and node_ar_time is not None and node_ar_time.numel() >= N:
            t_all = node_ar_time.to(device=device, dtype=torch.long)
        for level in self.witness_levels:
            L = int(level)
            if any(l not in cap_ids or l not in cap_scores for l in range(1, L + 1)):
                return None
            parent_ids = level_indices.get(L) if isinstance(level_indices, dict) else None
            if parent_ids is None or parent_ids.numel() == 0:
                parent_ids = torch.nonzero(node_level == L, as_tuple=False).view(-1)
            if parent_ids.numel() == 0:
                continue
            parent_ids = parent_ids.to(device=device, dtype=torch.long)
            P = int(parent_ids.numel())
            ids = cap_ids[L].index_select(1, parent_ids)            # [B, P, k]
            sc = cap_scores[L].index_select(1, parent_ids).float()
            sc = torch.where(ids >= 0, sc, torch.full_like(sc, neg))
            for lvl in range(L - 1, 0, -1):                          # chain down to L0
                ids, sc = _hop(ids, sc, cap_ids[lvl], cap_scores[lvl])
            W = int(ids.size(-1))
            valid = ids >= 0
            if t_all is not None:
                pt = t_all.index_select(0, parent_ids).view(1, P, 1)
                ct = t_all[ids.clamp(min=0)]
                valid = valid & ((ct <= pt) if allow_same_time else (ct < pt))
            sc = torch.where(valid, sc, torch.full_like(sc, neg))
            # Beam paths overlap (a node can be reached via several parents at a
            # hop) — suppress all but the best-scoring occurrence of each id so the
            # top-k budget goes to unique witnesses (same trick as the per_head cap).
            eq = ids.unsqueeze(-1) == ids.unsqueeze(-2)              # [B, P, W, W]
            ar = torch.arange(W, device=device)
            k_lt_j = ar.unsqueeze(0) < ar.unsqueeze(1)
            beaten = eq & ((sc.unsqueeze(-2) > sc.unsqueeze(-1)) |
                           ((sc.unsqueeze(-2) == sc.unsqueeze(-1)) & k_lt_j))
            sc = torch.where(beaten.any(dim=-1), torch.full_like(sc, neg), sc)
            valid = valid & (sc > (neg * 0.5))

            id_parts: List[torch.Tensor] = []
            score_parts: List[torch.Tensor] = []
            if k_sum > 0:
                vals, idx = torch.topk(sc, k=min(k_sum, W), dim=-1)
                picked = ids.gather(-1, idx)
                id_parts.append(torch.where(vals > (neg * 0.5), picked, torch.full_like(picked, -1)))
                score_parts.append(vals)
            if k_rare > 0:
                if surprise is None:
                    surprise = self._witness_l0_surprise(x_bnh, node_level, edge_index)  # [B, N]
                s_c = surprise.gather(1, ids.clamp(min=0).reshape(B, -1)).view(B, P, W).float()
                vf = valid.float()
                denom = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
                mean_s = (s_c * vf).sum(dim=-1, keepdim=True) / denom
                var_s = (((s_c - mean_s) ** 2) * vf).sum(dim=-1, keepdim=True) / denom
                norm_s = (s_c - mean_s) / (var_s.sqrt() + 1e-5)
                rare = torch.where(valid, sc + float(self.witness_lambda_rare) * norm_s,
                                   torch.full_like(sc, neg))
                vals, idx = torch.topk(rare, k=min(k_rare, W), dim=-1)
                picked = ids.gather(-1, idx)
                id_parts.append(torch.where(vals > (neg * 0.5), picked, torch.full_like(picked, -1)))
                score_parts.append(vals)
            packed = torch.cat(id_parts, dim=-1)
            cap = int(getattr(self, "witness_max_per_parent", 0))
            if cap > 0 and cap < packed.size(-1):
                sp = torch.cat(score_parts, dim=-1)
                sp = torch.where(packed >= 0, sp, torch.full_like(sp, neg))
                top_v, top_i = torch.topk(sp, k=cap, dim=-1)
                packed = packed.gather(-1, top_i)
                packed = torch.where(top_v > (neg * 0.5), packed, torch.full_like(packed, -1))
            width = int(packed.size(-1))
            if out is None:
                out = torch.full((B, N, width), -1, dtype=torch.long, device=device)
            elif width != out.size(-1):
                w = max(width, out.size(-1))
                if out.size(-1) < w:
                    out = torch.cat([out, torch.full((B, N, w - out.size(-1)), -1, dtype=torch.long, device=device)], dim=-1)
                if width < w:
                    packed = torch.cat([packed, torch.full((B, P, w - width), -1, dtype=torch.long, device=device)], dim=-1)
            out.index_copy_(1, parent_ids, packed)
        return out

    def _build_witness_packet(
        self,
        parent_ids: torch.Tensor,
        children: torch.Tensor,
        q_all: torch.Tensor,
        k_all: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        x_bnh: torch.Tensor,
        node_level: torch.Tensor,
        edge_index: torch.Tensor,
        allow_same_time: bool,
        causal: bool,
    ) -> Optional[torch.Tensor]:
        """Core witness selection for one parent level. `children` is a [P, C] table of L0
        descendant ids (-1 padded). Returns packed [B, P, slots] of L0 ids it 'cannot forget':
          - summary witnesses: top-k children by responsibility (q_parent.k_child),
            head-averaged (flat) or top-k per head (summary_witness_per_head).
          - rare witnesses: top-k by responsibility + lambda * normalized detached surprise.
        None if there is nothing to build."""
        B, N, Hh, Dh = q_all.shape
        device = q_all.device
        if parent_ids is None or children is None or parent_ids.numel() == 0:
            return None

        k_sum = int(self.witness_k_summary) if self.use_summary_witnesses else 0
        k_rare = int(self.witness_k_rare) if self.use_rare_witnesses else 0
        per_head = bool(self.summary_witness_per_head)
        P, C = int(parent_ids.numel()), int(children.size(1))
        k_sum = min(k_sum, C)
        k_rare = min(k_rare, C)
        sum_slots = (k_sum * Hh) if per_head else k_sum
        slots = sum_slots + k_rare
        if slots == 0:
            return None

        child_safe = children.clamp(min=0)                     # [P, C]
        child_valid = children >= 0                            # [P, C]
        q_p = q_all.index_select(1, parent_ids)                # [B, P, Hh, Dh]
        # Same dense-GEMM-then-gather trick as _hqd_score_candidates_batched: scoring every
        # parent against the whole key bank ([B,P,N,Hh] scores) beats gathering per-child key
        # VECTORS ([B,P,C,Hh,Dh]) whenever the descendant table C is wide (L2/L3 bags).
        use_dense = (B * P * N * Hh) <= 128_000_000 and (C * Dh) > (2 * N)
        if use_dense:
            dense = torch.einsum("bphd,bnhd->bpnh", q_p, k_all) / math.sqrt(float(Dh))  # [B,P,N,Hh]
            gidx = child_safe.view(1, P, C, 1).expand(B, P, C, Hh)
            score_heads = dense.gather(2, gidx)                # [B, P, C, Hh]
        else:
            k_c = k_all.index_select(1, child_safe.reshape(-1)).view(B, P, C, Hh, Dh)
            score_heads = torch.einsum("bphd,bpchd->bpch", q_p, k_c) / math.sqrt(float(Dh))

        valid = child_valid.view(1, P, C).expand(B, P, C).clone()  # [B, P, C]
        if causal and node_ar_time is not None and node_ar_time.numel() >= N:
            t = node_ar_time.to(device=device, dtype=torch.long)
            pt = t.index_select(0, parent_ids).view(1, P, 1)
            ct = t.index_select(0, child_safe.reshape(-1)).view(1, P, C)
            valid = valid & ((ct <= pt) if allow_same_time else (ct < pt))
        neg = torch.finfo(score_heads.dtype).min
        score_heads = score_heads.masked_fill(~valid.unsqueeze(-1), neg)
        child_ids_bpc = child_safe.view(1, P, C).expand(B, P, C)

        id_parts: List[torch.Tensor] = []
        score_parts: List[torch.Tensor] = []
        if k_sum > 0:
            if per_head:
                sh = score_heads.permute(0, 1, 3, 2)            # [B, P, Hh, C]
                vals, idx = torch.topk(sh, k=k_sum, dim=-1)      # [B, P, Hh, k_sum]
                ids = child_safe.view(1, P, 1, C).expand(B, P, Hh, C).gather(-1, idx)
                ids = torch.where(vals > (neg * 0.5), ids, torch.full_like(ids, -1))
                id_parts.append(ids.reshape(B, P, Hh * k_sum))
                score_parts.append(vals.reshape(B, P, Hh * k_sum))
            else:
                vals, idx = torch.topk(score_heads.mean(dim=-1), k=k_sum, dim=-1)
                ids = child_ids_bpc.gather(-1, idx)
                id_parts.append(torch.where(vals > (neg * 0.5), ids, torch.full_like(ids, -1)))
                score_parts.append(vals)

        if k_rare > 0:
            score_mean = score_heads.mean(dim=-1)               # [B, P, C], masked -> neg
            surprise = self._witness_l0_surprise(x_bnh, node_level, edge_index)  # [B, N]
            sc = surprise.index_select(1, child_safe.reshape(-1)).view(B, P, C).to(score_mean.dtype)
            vf = valid.to(score_mean.dtype)
            denom = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_s = (sc * vf).sum(dim=-1, keepdim=True) / denom
            var_s = (((sc - mean_s) ** 2) * vf).sum(dim=-1, keepdim=True) / denom
            norm_s = (sc - mean_s) / (var_s.sqrt() + 1e-5)
            rare_score = torch.where(valid, score_mean + float(self.witness_lambda_rare) * norm_s,
                                     torch.full_like(score_mean, neg))
            vals, idx = torch.topk(rare_score, k=k_rare, dim=-1)
            ids = child_ids_bpc.gather(-1, idx)
            id_parts.append(torch.where(vals > (neg * 0.5), ids, torch.full_like(ids, -1)))
            score_parts.append(vals)

        packed = torch.cat(id_parts, dim=-1)                     # [B, P, slots]
        # Optional max-topk cap: bound the packet width (and thus the downstream candidate
        # set) regardless of per_head*num_heads, keeping the read linear with a small constant.
        cap = int(getattr(self, "witness_max_per_parent", 0))
        if cap > 0 and cap < packed.size(-1):
            sp = torch.cat(score_parts, dim=-1)
            sp = torch.where(packed >= 0, sp, torch.full_like(sp, neg))
            if per_head:
                # Per-head picks overlap heavily, so the cap would spend slots on duplicates.
                # Suppress all but the highest-scoring occurrence of each id (tie -> lowest
                # index) so the cap budget goes to unique witnesses. Skipped for the
                # head-averaged path, which produces no intra-parent duplicates -> no compute.
                S = packed.size(-1)
                eq = packed.unsqueeze(-1) == packed.unsqueeze(-2)          # [B, P, S, S]
                ar = torch.arange(S, device=device)
                k_lt_j = ar.unsqueeze(0) < ar.unsqueeze(1)                 # [S, S]: competitor k < j
                beaten = eq & ((sp.unsqueeze(-2) > sp.unsqueeze(-1)) |
                               ((sp.unsqueeze(-2) == sp.unsqueeze(-1)) & k_lt_j))
                sp = torch.where(beaten.any(dim=-1), torch.full_like(sp, neg), sp)
            top_v, top_i = torch.topk(sp, k=cap, dim=-1)
            packed = packed.gather(-1, top_i)
            packed = torch.where(top_v > (neg * 0.5), packed, torch.full_like(packed, -1))
        return packed                                          # [B, P, slots]

    def _hierarchical_query_descent_ephemeral_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        base_edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        x_in: Optional[torch.Tensor] = None,
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Dict[str, int]]:
        return self._hierarchical_query_descent_vectorized_batched(
            transformer=transformer,
            x_bnh=x_bnh,
            base_edge_index=base_edge_index,
            node_level=node_level,
            node_ar_time=node_ar_time,
            x_in=x_in,
            _qk_cache=_qk_cache,
        )

    def _hierarchical_query_descent_packed_witness_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        base_edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        q_all: torch.Tensor,
        k_all: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        """L0->L3 selection, then captured graph-witness lookup, returned as packed L0 ids.

        This is the efficient variant of shallow L3 witness reads: hierarchy attention from
        the current message-passing layer has already stored direct child witnesses per
        parent, so HQD only scores L0 queries against L3 summaries and composes
        L3->L2->L1->L0 witness tables. The returned packed candidate tensor is consumed by
        message passing without flattening into a sparse L0 edge list.
        """
        if int(getattr(self, "hqd_query_level", 0)) != 0 or int(getattr(self, "hqd_shallow_read_level", 0)) != 3:
            return None
        mp = getattr(transformer, "message_passing", None)
        if str(getattr(self, "hqd_packed_witness_source", "graph")).lower() == "backbone":
            # Same table format as the scatter-path capture, but harvested (for free)
            # from the packed cross-level lane of an earlier layer this forward.
            latest = getattr(self, "_backbone_witness_latest", None)
            fresh = latest is not None and int(latest[0]) == int(getattr(self, "_witness_build_epoch", -1))
            direct = latest[1] if fresh else None
        else:
            direct = getattr(mp, "_last_graph_witness_ids", None) if mp is not None else None
        if not isinstance(direct, dict) or not all(level in direct for level in (1, 2, 3)):
            return None

        device = x_bnh.device
        B, N, _ = x_bnh.shape
        causal = self._hqd_effective_causal()
        allow_same_time = bool(getattr(self, "hier_ar_allow_same_time", False))
        profile_enabled = bool(getattr(self, "hqd_debug", False))
        t0 = time.monotonic() if profile_enabled else 0.0
        profile_stats: Dict[str, float] = {}

        l0_idx = torch.nonzero(node_level == 0, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        l1_idx = torch.nonzero(node_level == 1, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        l2_idx = torch.nonzero(node_level == 2, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        l3_idx = torch.nonzero(node_level == 3, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        if l0_idx.numel() == 0 or l1_idx.numel() == 0 or l2_idx.numel() == 0 or l3_idx.numel() == 0:
            return None

        if node_ar_time is not None and node_ar_time.numel() >= int(N):
            t_all = node_ar_time.to(device=device, dtype=torch.long)
            l0_time = t_all.index_select(0, l0_idx)
            l3_time = t_all.index_select(0, l3_idx)
        else:
            l0_time = torch.arange(l0_idx.numel(), device=device, dtype=torch.long)
            # Conservative fallback: if explicit AR time is absent, use descendant intervals.
            l3_to_l2 = self._zipper_build_children_table_cached(base_edge_index, node_level, 3, 2)
            l2_to_l1 = self._zipper_build_children_table_cached(base_edge_index, node_level, 2, 1)
            l1_to_l0 = self._zipper_build_children_table_cached(base_edge_index, node_level, 1, 0)
            min0 = torch.full((int(N),), -1, dtype=torch.long, device=device)
            max0 = torch.full((int(N),), -1, dtype=torch.long, device=device)
            min0[l0_idx] = l0_time
            max0[l0_idx] = l0_time
            min1, max1 = self._hqd_build_descendant_interval_cache(l1_idx, min0, max0, l1_to_l0, int(N))
            min2, max2 = self._hqd_build_descendant_interval_cache(l2_idx, min1, max1, l2_to_l1, int(N))
            _, max3 = self._hqd_build_descendant_interval_cache(l3_idx, min2, max2, l3_to_l2, int(N))
            l3_time = max3.index_select(0, l3_idx)
            t_all = max0

        q_l0 = q_all.index_select(1, l0_idx)
        k_l3 = k_all.index_select(1, l3_idx)
        head_dim = max(1, int(q_l0.size(-1)))
        num_heads = max(1, int(q_l0.size(-2)))
        scores3 = torch.einsum("bqhd,bkhd->bqk", q_l0, k_l3) / float(num_heads * math.sqrt(float(head_dim)))
        if causal:
            valid3 = (l3_time.view(1, 1, -1) <= l0_time.view(1, -1, 1)) if allow_same_time else (l3_time.view(1, 1, -1) < l0_time.view(1, -1, 1))
            valid3 = valid3 & (l3_time.view(1, 1, -1) >= 0)
            scores3 = scores3.masked_fill(~valid3, float("-inf"))
        sel3, _, sel3_valid = self._hqd_topk_from_scores_batched(
            l3_idx.view(1, 1, -1).expand(int(B), int(l0_idx.numel()), -1),
            scores3,
            int(getattr(self, "hqd_topk_l3", 4)),
        )

        def _lookup(table: torch.Tensor, nodes: torch.Tensor, valid: torch.Tensor, cap: int) -> Tuple[torch.Tensor, torch.Tensor]:
            slots = int(table.size(-1))
            safe = nodes.clamp(min=0)
            batch_offsets = torch.arange(int(B), device=device, dtype=torch.long).view(int(B), 1, 1) * int(N)
            flat = table.to(device=device, dtype=torch.long).reshape(int(B) * int(N), slots)
            out = flat.index_select(0, (safe + batch_offsets).reshape(-1)).view(int(B), nodes.size(1), nodes.size(2), slots)
            out = torch.where(valid.unsqueeze(-1), out, torch.full_like(out, -1))
            out = out.reshape(int(B), nodes.size(1), -1)
            out, mask = self._hqd_maybe_dedup_candidates_batched(out, force=True)
            out = torch.where(mask, out, torch.full_like(out, -1))
            out, _ = torch.sort(out, dim=-1, descending=True)
            mask = out >= 0
            keep = min(max(1, int(cap)), int(out.size(-1)))
            out = out[..., :keep]
            mask = mask[..., :keep] & (out >= 0)
            return out, mask

        l2_nodes, l2_valid = _lookup(direct[3], sel3, sel3_valid, int(getattr(self, "hqd_topk_l2", 4)))
        l1_nodes, l1_valid = _lookup(direct[2], l2_nodes, l2_valid, int(getattr(self, "hqd_topk_l1", 4)))
        l0_nodes, l0_valid = _lookup(direct[1], l1_nodes, l1_valid, int(getattr(self, "hqd_topk_l0", 8)))

        if causal:
            cand_t = t_all.index_select(0, l0_nodes.clamp(min=0).reshape(-1)).view_as(l0_nodes)
            causal_ok = (cand_t <= l0_time.view(1, -1, 1)) if allow_same_time else (cand_t < l0_time.view(1, -1, 1))
            l0_valid = l0_valid & causal_ok & (cand_t >= 0)
        l0_nodes = torch.where(l0_valid, l0_nodes, torch.full_like(l0_nodes, -1))
        added_total = int(l0_valid.sum().item())
        if added_total <= 0:
            return None

        stage_stats = {
            "queries": int(B * l0_idx.numel()),
            "l3_selected_total": int(sel3_valid.sum().item()),
            "l2_selected_total": int(l2_valid.sum().item()),
            "l1_selected_total": int(l1_valid.sum().item()),
            "l0_selected_total": int(added_total),
            "final_selected_total": int(added_total),
            "final_candidates": int(added_total),
            "packed_witness_l0": int(added_total),
        }
        if profile_enabled:
            profile_stats["packed_witness_select_ms"] = (time.monotonic() - t0) * 1000.0
        return {
            "edges": None,
            "packed_l0": (l0_idx, l0_nodes),
            "added_total": int(added_total),
            "stage_stats": stage_stats,
            "profile_stats": profile_stats if profile_enabled else None,
        }

    def _hierarchical_query_descent_recomputed_packed_witness_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        base_edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        q_all: torch.Tensor,
        k_all: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        """Use the existing recomputed witness table, but consume it as packed L0 reads."""
        if int(getattr(self, "hqd_query_level", 0)) != 0 or int(getattr(self, "hqd_shallow_read_level", 0)) != 3:
            return None
        if not bool(getattr(self, "use_witness_packets", False)):
            return None
        if 3 not in getattr(self, "witness_levels", [1]):
            return None

        device = x_bnh.device
        B, N, _ = x_bnh.shape
        causal = self._hqd_effective_causal()
        allow_same_time = bool(getattr(self, "hier_ar_allow_same_time", False))
        profile_enabled = bool(getattr(self, "hqd_debug", False))
        t0 = time.monotonic() if profile_enabled else 0.0
        profile_stats: Dict[str, float] = {}

        l0_idx = torch.nonzero(node_level == 0, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        l1_idx = torch.nonzero(node_level == 1, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        l2_idx = torch.nonzero(node_level == 2, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        l3_idx = torch.nonzero(node_level == 3, as_tuple=False).view(-1).to(device=device, dtype=torch.long)
        if l0_idx.numel() == 0 or l1_idx.numel() == 0 or l2_idx.numel() == 0 or l3_idx.numel() == 0:
            return None

        l3_to_l2 = self._zipper_build_children_table_cached(base_edge_index, node_level, 3, 2)
        l2_to_l1 = self._zipper_build_children_table_cached(base_edge_index, node_level, 2, 1)
        l1_to_l0 = self._zipper_build_children_table_cached(base_edge_index, node_level, 1, 0)
        if node_ar_time is not None and node_ar_time.numel() >= int(N):
            t_all = node_ar_time.to(device=device, dtype=torch.long)
        else:
            t_all = torch.full((int(N),), -1, dtype=torch.long, device=device)
            t_all[l0_idx] = torch.arange(l0_idx.numel(), device=device, dtype=torch.long)
            _, max1 = self._hqd_build_descendant_interval_cache(l1_idx, t_all, t_all, l1_to_l0, int(N))
            _, max2 = self._hqd_build_descendant_interval_cache(l2_idx, max1, max1, l2_to_l1, int(N))
            _, max3 = self._hqd_build_descendant_interval_cache(l3_idx, max2, max2, l3_to_l2, int(N))
            t_all = torch.maximum(torch.maximum(torch.maximum(t_all, max1), max2), max3)
        l0_time = t_all.index_select(0, l0_idx)
        l3_time = t_all.index_select(0, l3_idx)

        witness_table = self._build_witness_l0_ids(
            q_all=q_all,
            k_all=k_all,
            child_tables={1: l1_to_l0, 2: l2_to_l1, 3: l3_to_l2},
            level_indices={1: l1_idx, 2: l2_idx, 3: l3_idx},
            node_ar_time=node_ar_time,
            x_bnh=x_bnh,
            node_level=node_level,
            edge_index=base_edge_index,
            allow_same_time=allow_same_time,
            causal=causal,
        )
        if witness_table is None or witness_table.numel() == 0:
            return None

        q_l0 = q_all.index_select(1, l0_idx)
        k_l3 = k_all.index_select(1, l3_idx)
        head_dim = max(1, int(q_l0.size(-1)))
        num_heads = max(1, int(q_l0.size(-2)))
        scores3 = torch.einsum("bqhd,bkhd->bqk", q_l0, k_l3) / float(num_heads * math.sqrt(float(head_dim)))
        if causal:
            valid3 = (l3_time.view(1, 1, -1) <= l0_time.view(1, -1, 1)) if allow_same_time else (l3_time.view(1, 1, -1) < l0_time.view(1, -1, 1))
            valid3 = valid3 & (l3_time.view(1, 1, -1) >= 0)
            scores3 = scores3.masked_fill(~valid3, float("-inf"))
        sel3, _, sel3_valid = self._hqd_topk_from_scores_batched(
            l3_idx.view(1, 1, -1).expand(int(B), int(l0_idx.numel()), -1),
            scores3,
            int(getattr(self, "hqd_topk_l3", 4)),
        )

        slots = int(witness_table.size(-1))
        wflat = witness_table.to(device=device, dtype=torch.long).reshape(int(B) * int(N), slots)
        boff = torch.arange(int(B), device=device, dtype=torch.long).view(int(B), 1, 1) * int(N)
        wit = wflat.index_select(0, (sel3.clamp(min=0) + boff).reshape(-1)).view(int(B), sel3.size(1), sel3.size(2), slots)
        wit = torch.where(sel3_valid.unsqueeze(-1), wit, torch.full_like(wit, -1)).reshape(int(B), sel3.size(1), -1)
        wit, wit_mask = self._hqd_maybe_dedup_candidates_batched(wit, force=True)
        wit = torch.where(wit_mask, wit, torch.full_like(wit, -1))
        wit, _ = torch.sort(wit, dim=-1, descending=True)
        wit_mask = wit >= 0
        if causal:
            cand_t = t_all.index_select(0, wit.clamp(min=0).reshape(-1)).view_as(wit)
            causal_ok = (cand_t <= l0_time.view(1, -1, 1)) if allow_same_time else (cand_t < l0_time.view(1, -1, 1))
            wit_mask = wit_mask & causal_ok & (cand_t >= 0)
            wit = torch.where(wit_mask, wit, torch.full_like(wit, -1))

        scores0 = self._hqd_score_candidates_batched(
            query_vec=q_l0,
            key_bank=k_all,
            candidate_nodes=wit,
            candidate_mask=wit_mask,
            query_time=l0_time.view(1, -1).expand(int(B), -1),
            candidate_max_time_cache=t_all,
            causal=causal,
            allow_same_time=allow_same_time,
        )
        top_nodes, _, top_valid = self._hqd_topk_from_scores_batched(wit, scores0, int(getattr(self, "hqd_topk_l0", 8)))
        top_nodes = torch.where(top_valid, top_nodes, torch.full_like(top_nodes, -1))
        added_total = int(top_valid.sum().item())
        if added_total <= 0:
            return None

        stage_stats = {
            "queries": int(B * l0_idx.numel()),
            "l3_selected_total": int(sel3_valid.sum().item()),
            "l0_selected_total": int(added_total),
            "final_selected_total": int(added_total),
            "final_candidates": int(added_total),
            "packed_witness_l0": int(added_total),
            "packed_witness_source_recompute": 1,
        }
        if profile_enabled:
            profile_stats["packed_witness_select_ms"] = (time.monotonic() - t0) * 1000.0
        return {
            "edges": None,
            "packed_l0": (l0_idx, top_nodes),
            "added_total": int(added_total),
            "stage_stats": stage_stats,
            "profile_stats": profile_stats if profile_enabled else None,
        }

    def _hierarchical_query_descent_vectorized_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        base_edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        x_in: Optional[torch.Tensor] = None,
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Dict[str, int]]:
        device = x_bnh.device
        B, N, _ = x_bnh.shape
        causal = self._hqd_effective_causal()
        allow_same_time = bool(getattr(self, "hier_ar_allow_same_time", False))
        stage_stats: Dict[str, int] = {
            "queries": 0,
            "l3_candidates": 0,
            "l3_selected_total": 0,
            "l2_candidates": 0,
            "l2_selected_total": 0,
            "l1_candidates": 0,
            "l1_selected_total": 0,
            "l0_candidates": 0,
            "l0_selected_total": 0,
            "final_candidates": 0,
            "final_candidate_total": 0,
            "final_selected_total": 0,
            "local_window_total": 0,
            "local_window_overlap": 0,
            "causal_violations_pre": 0,
            "causal_violations_post": 0,
        }
        profile_enabled = bool(self.hqd_debug)
        collect_stats = bool(self.hqd_debug)
        profile_stats: Dict[str, float] = {}
        t0 = time.monotonic() if profile_enabled else 0.0

        def _profile_start() -> float:
            return time.monotonic() if profile_enabled else 0.0

        def _profile_add(key: str, start_time: float) -> None:
            if profile_enabled:
                profile_stats[key] = float(profile_stats.get(key, 0.0) + (time.monotonic() - start_time) * 1000.0)

        _qk_t0 = _profile_start()
        mp = transformer.message_passing
        use_shared_qk = bool(self.hqd_use_existing_zipper_projections and _qk_cache is not None and len(_qk_cache) >= 2)
        if x_in is None and not use_shared_qk:
            x_in = transformer.norm1(x_bnh) if hasattr(transformer, "norm1") else x_bnh
        if use_shared_qk:
            q_all = _qk_cache[0]
            k_all = _qk_cache[1]
        else:
            with torch.no_grad():
                q_all = mp.q_proj(x_in).view(B, N, mp.num_heads, mp.head_dim)
                k_all = mp.k_proj(x_in).view(B, N, mp.num_heads, mp.head_dim)
        _profile_add("qk_proj_ms", _qk_t0)

        _skeleton_t0 = _profile_start()
        skeleton_key = self._hqd_skeleton_cache_key(
            num_nodes=int(N),
            edge_index=base_edge_index,
            node_level=node_level,
            node_ar_time=node_ar_time,
        )
        skeleton_cache = self._hqd_skeleton_cache
        skeleton = skeleton_cache.get(skeleton_key, None)

        if skeleton is None:
            l0_idx = (node_level == 0).nonzero(as_tuple=False).view(-1)
            l1_idx = (node_level == 1).nonzero(as_tuple=False).view(-1)
            l2_idx = (node_level == 2).nonzero(as_tuple=False).view(-1)
            l3_idx = (node_level == 3).nonzero(as_tuple=False).view(-1)

            l3_to_l2 = self._zipper_build_children_table_cached(base_edge_index, node_level, 3, 2)
            l2_to_l1 = self._zipper_build_children_table_cached(base_edge_index, node_level, 2, 1)
            l1_to_l0 = self._zipper_build_children_table_cached(base_edge_index, node_level, 1, 0)

            if bool(getattr(self, "hqd_validate_disjoint_children", False)):
                disjoint_ok = (
                    self._hqd_children_table_is_disjoint(l3_to_l2)
                    and self._hqd_children_table_is_disjoint(l2_to_l1)
                    and self._hqd_children_table_is_disjoint(l1_to_l0)
                )
                if not disjoint_ok and bool(getattr(self, "hqd_assume_disjoint_children", False)):
                    logger.warning(
                        "HQD disjoint-child validation failed; disabling hqd_assume_disjoint_children for this model."
                    )
                    self.hqd_assume_disjoint_children = False

            level_min_time_cache: Dict[int, torch.Tensor] = {
                0: torch.full((int(N),), -1, dtype=torch.long, device=device),
            }
            level_max_time_cache: Dict[int, torch.Tensor] = {
                0: torch.full((int(N),), -1, dtype=torch.long, device=device),
            }

            if l0_idx.numel() > 0:
                if node_ar_time is not None and node_ar_time.numel() >= int(N):
                    l0_time = node_ar_time.index_select(0, l0_idx.to(device=node_ar_time.device)).to(device=device, dtype=torch.long)
                else:
                    l0_time = torch.arange(l0_idx.numel(), device=device, dtype=torch.long)
                level_min_time_cache[0][l0_idx] = l0_time
                level_max_time_cache[0][l0_idx] = l0_time

            if node_ar_time is not None and node_ar_time.numel() >= int(N):
                node_ar_time_long = node_ar_time.to(device=device, dtype=torch.long)
                level_max_time_cache[1] = node_ar_time_long.clone()
                level_max_time_cache[2] = node_ar_time_long.clone()
                level_max_time_cache[3] = node_ar_time_long.clone()
                level_min_time_cache[1] = node_ar_time_long.clone()
                level_min_time_cache[2] = node_ar_time_long.clone()
                level_min_time_cache[3] = node_ar_time_long.clone()
            else:
                level_min_time_cache[1], level_max_time_cache[1] = self._hqd_build_descendant_interval_cache(
                    l1_idx,
                    level_min_time_cache[0],
                    level_max_time_cache[0],
                    l1_to_l0,
                    int(N),
                )
                level_min_time_cache[2], level_max_time_cache[2] = self._hqd_build_descendant_interval_cache(
                    l2_idx,
                    level_min_time_cache[1],
                    level_max_time_cache[1],
                    l2_to_l1,
                    int(N),
                )
                level_min_time_cache[3], level_max_time_cache[3] = self._hqd_build_descendant_interval_cache(
                    l3_idx,
                    level_min_time_cache[2],
                    level_max_time_cache[2],
                    l3_to_l2,
                    int(N),
                )

            skeleton = {
                "l0_idx": l0_idx,
                "l1_idx": l1_idx,
                "l2_idx": l2_idx,
                "l3_idx": l3_idx,
                "l3_to_l2": l3_to_l2,
                "l2_to_l1": l2_to_l1,
                "l1_to_l0": l1_to_l0,
                "level_min_time_cache": level_min_time_cache,
                "level_max_time_cache": level_max_time_cache,
            }
            max_entries = max(1, int(self._hqd_skeleton_cache_max_entries))
            if len(skeleton_cache) >= max_entries:
                oldest_key = next(iter(skeleton_cache))
                del skeleton_cache[oldest_key]
            skeleton_cache[skeleton_key] = skeleton
        else:
            l0_idx = skeleton["l0_idx"]
            l1_idx = skeleton["l1_idx"]
            l2_idx = skeleton["l2_idx"]
            l3_idx = skeleton["l3_idx"]
            l3_to_l2 = skeleton["l3_to_l2"]
            l2_to_l1 = skeleton["l2_to_l1"]
            l1_to_l0 = skeleton["l1_to_l0"]

        level_max_time_cache = skeleton["level_max_time_cache"]
        level_min_time_cache = skeleton["level_min_time_cache"]
        _profile_add("skeleton_ms", _skeleton_t0)

        hqd_topk_l3 = int(self.hqd_topk_l3)
        hqd_topk_l2 = int(self.hqd_topk_l2)
        hqd_topk_l1 = int(self.hqd_topk_l1)
        hqd_topk_l0 = int(self.hqd_topk_l0)
        hqd_query_level = int(getattr(self, "hqd_query_level", 0))
        hqd_stop_level = int(getattr(self, "hqd_stop_level", 0))
        hqd_handoff_to_l0 = bool(getattr(self, "hqd_handoff_to_l0", False)) and hqd_query_level > 0 and hqd_stop_level > 0
        hqd_global_topk = int(getattr(self, "hqd_global_topk", 0))
        # Descent shortcut ("bag instead of descend"): the last level the scored descent
        # runs. 0 = full (score L3,L2,L1,L0). 2 = score L3,L2 then take ALL L1 children of
        # the selected L2 nodes unscored (skip the L1 scoring stage); the final L0 stage
        # still scores + prunes to topk_l0, so attention stays bounded. Changes selection
        # (behaviour-affecting), query_level 0 only.
        desc_stop_at = int(getattr(self, "hqd_descent_stop_at", 0)) if hqd_query_level == 0 and hqd_stop_level == 0 else 0
        # Shallow bag read ("L0 queries L3, bags down"): stop the SCORED descent at this
        # level and reach L0 only through the selected coarse nodes' precomputed bags
        # (cross-level witnesses), skipping the L2/L1/L0 scoring stages — the real cost.
        # 0 = off (full descent). 3 = score L3 only. 2 = score L3,L2. Requires the stop
        # level to be in witness_levels. query_level 0 only; causal (bag = causal key).
        shallow_read = int(getattr(self, "hqd_shallow_read_level", 0)) if hqd_query_level == 0 and hqd_stop_level == 0 else 0
        if shallow_read in (2, 3) and shallow_read not in getattr(self, "witness_levels", [1]) and not bool(getattr(self, "_hqd_shallow_warned", False)):
            logger.warning(
                "hqd_shallow_read_level=%d but level %d is not in witness_levels=%s: the shallow "
                "read reaches L0 only via that level's bag, so add %d to witness_levels or the "
                "candidate set will be empty.", shallow_read, shallow_read,
                getattr(self, "witness_levels", [1]), shallow_read,
            )
            self._hqd_shallow_warned = True
        local_window_size = int(self.hqd_local_window_size if self.hqd_local_window_size > 0 else getattr(self, "l0_local_window", 0))
        level_indices = {0: l0_idx, 1: l1_idx, 2: l2_idx, 3: l3_idx}
        query_idx = level_indices.get(hqd_query_level, l0_idx)
        query_chunk_size = min(max(1, int(self.hqd_query_chunk_size)), max(1, int(query_idx.numel())))

        # Witness packets: per-L1-node L0 pointers, recomputed each call from current
        # features (NOT cached in the feature-agnostic skeleton). Force-included into the
        # final-stage candidate scores so important/rare children survive top-k pruning.
        read_coarse_levels = {int(l) for l in getattr(self, "hqd_read_levels", [0]) if int(l) in (1, 2, 3)}
        # Per-level window bag (lateral reach): accumulate the L0 routing's per-level
        # selections, then scatter them back to each level's own nodes after the loop.
        bag_levels = [int(l) for l in getattr(self, "hqd_window_bag_levels", []) if int(l) in (1, 2, 3)] if hqd_query_level == 0 else []
        bag_routing = str(getattr(self, "hqd_window_bag_routing", "global"))
        if bag_levels and bag_routing == "global" and causal and not bool(getattr(self, "_hqd_bag_global_causal_warned", False)):
            logger.warning(
                "hqd_window_bag_routing='global' is NOT strictly AR-causal: the shared top-k bag "
                "is selected over all positions' scores, so future queries influence the bag a "
                "past coarse node attends to (leak ~1e-7, below bf16 precision but real). Use "
                "'per_position' for strict causal AR; 'global' is intended for masked-diffusion / "
                "bidirectional training."
            )
            self._hqd_bag_global_causal_warned = True
        bag_accum: Dict[int, Dict[str, List[torch.Tensor]]] = {l: {"nodes": [], "scores": []} for l in bag_levels}
        # per_position routing: map each L0 node to its level-X ancestor so a coarse node's
        # bag = its own descendants' selections (re-targeted up). Structural, O(N) to build.
        bag_ancestors: Dict[int, torch.Tensor] = {}
        if bag_levels and bag_routing == "per_position":
            def _invert_children(table: Dict[str, torch.Tensor]) -> torch.Tensor:
                parent_of = torch.full((int(N),), -1, dtype=torch.long, device=device)
                pid = table.get("parent_ids")
                ch = table.get("children")
                if pid is not None and ch is not None and pid.numel() > 0:
                    v = ch >= 0
                    parent_of[ch[v]] = pid.view(-1, 1).expand_as(ch)[v]
                return parent_of
            p_l0_l1 = _invert_children(l1_to_l0)
            p_l1_l2 = _invert_children(l2_to_l1)
            p_l2_l3 = _invert_children(l3_to_l2)
            anc1 = p_l0_l1
            anc2 = torch.where(anc1 >= 0, p_l1_l2.index_select(0, anc1.clamp(min=0)), torch.full_like(anc1, -1))
            anc3 = torch.where(anc2 >= 0, p_l2_l3.index_select(0, anc2.clamp(min=0)), torch.full_like(anc2, -1))
            bag_ancestors = {1: anc1, 2: anc2, 3: anc3}
        # per_position cap: 0 = natural bound (emit all descendant selections in-loop), >0 =
        # hard per-coarse-node budget (accumulate, then per-ancestor top-k after the loop).
        bag_cap = int(getattr(self, "hqd_window_bag_topk", 0)) if bag_routing == "per_position" else 0
        pp_accum: Dict[int, Dict[str, List[torch.Tensor]]] = (
            {l: {"nodes": [], "scores": [], "valid": []} for l in bag_levels} if (bag_levels and bag_cap > 0) else {}
        )
        pp_qpos: List[torch.Tensor] = []
        witness_table: Optional[torch.Tensor] = None
        witness_active = (
            bool(getattr(self, "use_witness_packets", False))
            and hqd_query_level == 0
            and (bool(self.use_summary_witnesses) or bool(self.use_rare_witnesses))
            and (int(self.witness_k_summary) > 0 or int(self.witness_k_rare) > 0)
        )
        if witness_active:
            witness_table = self._build_witness_l0_ids(
                q_all=q_all,
                k_all=k_all,
                child_tables={1: l1_to_l0, 2: l2_to_l1, 3: l3_to_l2},
                level_indices={1: l1_idx, 2: l2_idx, 3: l3_idx},
                node_ar_time=node_ar_time,
                x_bnh=x_bnh,
                node_level=node_level,
                edge_index=base_edge_index,
                allow_same_time=allow_same_time,
                causal=causal,
            )
            witness_active = witness_table is not None

        if self.hqd_debug and not bool(self._hqd_runtime_logged):
            logger.info(
                "HQD enabled: causal=%s query_level=%d stop_level=%d handoff_l0=%s global_topk=%d topk=(l3=%d,l2=%d,l1=%d,l0=%d,l0_topk_enable=%s) chunk=%d local_window=%d include_local=%s granularity=%s every_n=%d reuse_prev=%s reuse_max_age=%d",
                bool(causal),
                int(hqd_query_level),
                int(hqd_stop_level),
                bool(hqd_handoff_to_l0),
                int(hqd_global_topk),
                int(hqd_topk_l3),
                int(hqd_topk_l2),
                int(hqd_topk_l1),
                int(hqd_topk_l0),
                bool(self.hqd_l0_topk_enable),
                int(query_chunk_size),
                int(local_window_size),
                bool(self.hqd_include_local_window),
                str(self.hqd_granularity),
                int(self.hqd_every_n),
                bool(self.hqd_reuse_previous),
                int(self.hqd_reuse_max_age),
            )
            self._hqd_runtime_logged = True

        if l0_idx.numel() == 0 or l1_idx.numel() == 0 or l2_idx.numel() == 0 or l3_idx.numel() == 0 or query_idx.numel() == 0:
            self._last_hqd_added_total = 0
            self._last_hqd_selected_total = 0
            self._last_hqd_avg_l0 = None
            self._last_hqd_stage_stats = dict(stage_stats)
            self._last_hqd_profile_stats = profile_stats if profile_enabled else None
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty, empty, 0, stage_stats

        stage_stats["queries"] = int(B * query_idx.numel())
        batch_idx = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)

        # ~~~~ Static-shape descent (hqd_static_descent) ~~~~
        # Runs the plain 4-stage beam as ONE fixed-shape graph. Only the vanilla
        # configuration is handled; anything using the extra routing machinery falls
        # through to the stage-by-stage path below, which stays the reference.
        if bool(getattr(self, "hqd_static_descent", False)):
            simple = (
                hqd_query_level == 0
                and hqd_stop_level == 0
                and shallow_read == 0
                and hqd_global_topk == 0
                and not hqd_handoff_to_l0
                and not bool(getattr(self, "hqd_include_local_window", False))
                and int(getattr(self, "hqd_local_window_size", 0)) <= 0
                and not [l for l in getattr(self, "hqd_coarse_route_levels", []) or [] if int(l) in (1, 2, 3)]
            )
            if not simple:
                if not bool(getattr(self, "_hqd_static_fallback_logged", False)):
                    logger.warning(
                        "hqd_static_descent is on but this HQD configuration uses the extended "
                        "routing path (query_level=%d stop_level=%d shallow=%d global_topk=%d); "
                        "falling back to the stage-by-stage descent.",
                        hqd_query_level, hqd_stop_level, shallow_read, hqd_global_topk,
                    )
                    self._hqd_static_fallback_logged = True
            else:
                with torch.no_grad():
                    # Chunked exactly like the reference path: each chunk is its own fixed
                    # shape, so staticness is per-chunk and compile sees one graph per size.
                    sel_parts: List[torch.Tensor] = []
                    n_q = int(query_idx.numel())
                    for c0 in range(0, n_q, int(query_chunk_size)):
                        q_nodes_c = query_idx[c0 : c0 + int(query_chunk_size)]
                        q_vec_c = q_all.index_select(1, q_nodes_c)
                        if node_ar_time is not None and node_ar_time.numel() >= int(N):
                            qt_c = node_ar_time.index_select(
                                0, q_nodes_c.to(device=node_ar_time.device)
                            ).to(device=device, dtype=torch.long)
                        else:
                            qt_c = level_max_time_cache[0].index_select(0, q_nodes_c)
                        sel_c, _ = self._hqd_static_descent_callable()(
                            q_vec=q_vec_c,
                            k_all=k_all,
                            l3_idx=l3_idx,
                            tables=(l3_to_l2, l2_to_l1, l1_to_l0),
                            topks=(hqd_topk_l3, hqd_topk_l2, hqd_topk_l1, hqd_topk_l0),
                            query_time=qt_c.unsqueeze(0).expand(B, -1),
                            max_time=level_max_time_cache,
                            causal=causal,
                            allow_same_time=allow_same_time,
                        )
                        sel_parts.append(sel_c)
                    sel_s = sel_parts[0] if len(sel_parts) == 1 else torch.cat(sel_parts, dim=1)
                    valid_s = sel_s >= 0
                    # One compaction at the boundary (the stage path did this per level).
                    dst_s = query_idx.view(1, -1, 1).expand(B, -1, sel_s.size(-1))
                    b_s = batch_idx.expand_as(sel_s)
                    b_out = b_s[valid_s]
                    src_out = sel_s[valid_s]
                    dst_out = dst_s[valid_s]
                added = int(b_out.numel())
                # Reach diagnostics. HQD has no xq-style exclude_local, so the useful
                # numbers are how FAR selections land and how many duplicate what the
                # local window already covers. Reduced on GPU and left as 0-dim tensors;
                # gate_monitor pays the sync, and only on its own cadence.
                if added > 0:
                    _d = (src_out.float() - dst_out.float()).abs()
                    self._last_hqd_mean_dist = _d.mean().detach()
                    _half = float(getattr(self, "local_pack_window", 0) or 0) / 2.0
                    self._last_hqd_local_frac = (
                        (_d <= _half).float().mean().detach() if _half > 0 else None
                    )
                else:
                    self._last_hqd_mean_dist = None
                    self._last_hqd_local_frac = None
                if added > 0:
                    self._hqd_reuse_cache = {
                        "b_idx": b_out.detach(), "src_idx": src_out.detach(),
                        "dst_idx": dst_out.detach(), "B": int(B), "N": int(N),
                        "device": str(device),
                    }
                else:
                    self._hqd_reuse_cache = None
                stage_stats["final_selected_total"] = added
                self._last_hqd_added_total = added
                self._last_hqd_selected_total = added
                self._last_hqd_avg_l0 = None
                self._last_hqd_stage_stats = dict(stage_stats)
                self._last_hqd_profile_stats = profile_stats if profile_enabled else None
                return b_out, src_out, dst_out, added, stage_stats

        selected_b: List[torch.Tensor] = []
        selected_src: List[torch.Tensor] = []
        selected_dst: List[torch.Tensor] = []

        with torch.no_grad():
            for q_start_idx in range(0, int(query_idx.numel()), int(query_chunk_size)):
                q_end = min(int(query_idx.numel()), q_start_idx + int(query_chunk_size))
                query_nodes = query_idx[q_start_idx:q_end]
                query_positions = torch.arange(q_start_idx, q_end, device=device, dtype=torch.long)
                q_vec = q_all[:, query_nodes]
                if node_ar_time is not None and node_ar_time.numel() >= int(N):
                    query_time = node_ar_time.index_select(0, query_nodes.to(device=node_ar_time.device)).to(device=device, dtype=torch.long)
                else:
                    query_time = level_max_time_cache[hqd_query_level].index_select(0, query_nodes)
                query_time = query_time.unsqueeze(0).expand(B, -1)

                # Stage 3: chosen-level queries -> L3 summaries.
                _stage_t0 = _profile_start()
                cand3_nodes = l3_idx.view(1, 1, -1).expand(B, q_vec.size(1), -1)
                k_l3 = k_all.index_select(1, l3_idx)
                head_dim = max(1, int(q_vec.size(-1)))
                num_heads = max(1, int(q_vec.size(-2)))
                scores3 = torch.einsum("bqhd,bkhd->bqk", q_vec, k_l3)
                scores3 = scores3 / float(num_heads * math.sqrt(float(head_dim)))
                cand3_mask = torch.ones_like(cand3_nodes, dtype=torch.bool)
                if causal:
                    l3_max_time = level_max_time_cache[3].index_select(0, l3_idx)
                    if allow_same_time:
                        valid3 = l3_max_time.view(1, 1, -1) <= query_time.unsqueeze(-1)
                    else:
                        valid3 = l3_max_time.view(1, 1, -1) < query_time.unsqueeze(-1)
                    valid3 = valid3 & (l3_max_time.view(1, 1, -1) >= 0)
                    cand3_mask = valid3
                    scores3 = scores3.masked_fill(~valid3, float("-inf"))
                if collect_stats:
                    stage_stats["l3_candidates"] += int(cand3_mask.sum().item())
                sel3_nodes, sel3_scores, sel3_valid = self._hqd_topk_from_scores_batched(cand3_nodes, scores3, hqd_topk_l3)
                if hqd_global_topk > 0:
                    sel3_valid = self._hqd_apply_per_batch_global_topk(sel3_scores, sel3_valid, hqd_global_topk)
                if collect_stats:
                    stage_stats["l3_selected_total"] += int(sel3_valid.sum().item())
                _profile_add("l3_stage_ms", _stage_t0)

                final_nodes = sel3_nodes
                final_mask = sel3_valid
                final_level = 3

                # Stage 2: expand L3 -> L2.
                _stage_t0 = _profile_start()
                cand2_nodes_raw, cand2_valid_raw = self._hqd_expand_children_batched(sel3_nodes, l3_to_l2)
                cand2_nodes_raw = cand2_nodes_raw.reshape(B, q_vec.size(1), -1)
                cand2_valid_raw = cand2_valid_raw.reshape(B, q_vec.size(1), -1)
                cand2_nodes_raw = torch.where(cand2_valid_raw, cand2_nodes_raw, torch.full_like(cand2_nodes_raw, -1))
                cand2_nodes, cand2_mask = self._hqd_maybe_dedup_candidates_batched(cand2_nodes_raw)
                if collect_stats:
                    stage_stats["l2_candidates"] += int(cand2_mask.sum().item())
                if shallow_read == 3:
                    # Shallow read at L3: skip L2 scoring; sel2 empty -> reach L0 via L3 bags.
                    Qc = int(q_vec.size(1))
                    sel2_nodes = torch.empty(B, Qc, 0, dtype=torch.long, device=device)
                    sel2_scores = torch.empty(B, Qc, 0, dtype=q_vec.dtype, device=device)
                    sel2_valid = torch.empty(B, Qc, 0, dtype=torch.bool, device=device)
                else:
                    scores2 = self._hqd_score_candidates_batched(
                        query_vec=q_vec,
                        key_bank=k_all,
                        candidate_nodes=cand2_nodes,
                        candidate_mask=cand2_mask,
                        query_time=query_time,
                        candidate_max_time_cache=level_max_time_cache[2],
                        causal=causal,
                        allow_same_time=allow_same_time,
                    )
                    sel2_nodes, sel2_scores, sel2_valid = self._hqd_topk_from_scores_batched(cand2_nodes, scores2, hqd_topk_l2)
                    if hqd_global_topk > 0:
                        sel2_valid = self._hqd_apply_per_batch_global_topk(sel2_scores, sel2_valid, hqd_global_topk)
                if collect_stats:
                    stage_stats["l2_selected_total"] += int(sel2_valid.sum().item())
                final_nodes = sel2_nodes
                final_mask = sel2_valid
                final_level = 2
                _profile_add("l2_stage_ms", _stage_t0)
                if hqd_stop_level == 2:
                    top_nodes = final_nodes
                    top_valid = final_mask
                    if hqd_handoff_to_l0:
                        src_l0_nodes, src_l0_mask = self._hqd_expand_to_l0_batched(top_nodes, final_level, l2_to_l1, l1_to_l0)
                        dst_parent = query_nodes.view(1, -1, 1).expand(B, -1, 1)
                        dst_l0_nodes, dst_l0_mask = self._hqd_expand_to_l0_batched(dst_parent, hqd_query_level, l2_to_l1, l1_to_l0)
                        dst_flat = dst_l0_nodes.reshape(B, -1)
                        dst_mask_flat = dst_l0_mask.reshape(B, -1)
                        if dst_flat.numel() == 0 or src_l0_nodes.numel() == 0:
                            continue
                        safe_dst = dst_flat.clamp(min=0)
                        q_l0_vec = q_all.gather(1, safe_dst.view(B, -1, 1, 1).expand(-1, -1, q_all.size(2), q_all.size(3)))
                        src_exp = src_l0_nodes.unsqueeze(2).expand(B, src_l0_nodes.size(1), dst_l0_nodes.size(-1), src_l0_nodes.size(-1)).reshape(B, -1, src_l0_nodes.size(-1))
                        src_mask_exp = src_l0_mask.unsqueeze(2).expand(B, src_l0_mask.size(1), dst_l0_nodes.size(-1), src_l0_mask.size(-1)).reshape(B, -1, src_l0_mask.size(-1))
                        src_mask_exp = src_mask_exp & dst_mask_flat.unsqueeze(-1)
                        dst_time = level_max_time_cache[0].index_select(0, safe_dst.reshape(-1)).view_as(safe_dst)
                        scores0 = self._hqd_score_candidates_batched(q_l0_vec, k_all, src_exp, src_mask_exp, dst_time, level_max_time_cache[0], causal, allow_same_time)
                        top_nodes, top_scores, top_valid = self._hqd_topk_from_scores_batched(src_exp, scores0, hqd_topk_l0)
                        if hqd_global_topk > 0:
                            top_valid = self._hqd_apply_per_batch_global_topk(top_scores, top_valid, hqd_global_topk)
                        if collect_stats:
                            stage_stats["l0_candidates"] += int(src_mask_exp.sum().item())
                            stage_stats["final_candidate_total"] += int(src_mask_exp.sum().item())
                            stage_stats["l0_selected_total"] += int(top_valid.sum().item())
                            stage_stats["final_selected_total"] += int(top_valid.sum().item())
                            stage_stats["final_candidates"] += int(top_valid.sum().item())
                        keep = top_valid
                        selected_b.append(torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1).expand_as(top_nodes)[keep])
                        selected_src.append(top_nodes[keep])
                        selected_dst.append(dst_flat.unsqueeze(-1).expand_as(top_nodes)[keep])
                    else:
                        if collect_stats:
                            stage_stats["final_candidate_total"] += int(top_valid.numel())
                            stage_stats["final_selected_total"] += int(top_valid.sum().item())
                            stage_stats["final_candidates"] += int(top_valid.sum().item())
                        keep = top_valid
                        selected_b.append(batch_idx.expand_as(top_nodes)[keep])
                        selected_src.append(top_nodes[keep])
                        selected_dst.append(query_nodes.view(1, -1, 1).expand(B, -1, top_nodes.size(-1))[keep])
                    continue

                # Stage 1: expand L2 -> L1.
                _stage_t0 = _profile_start()
                cand1_nodes_raw, cand1_valid_raw = self._hqd_expand_children_batched(sel2_nodes, l2_to_l1)
                cand1_nodes_raw = cand1_nodes_raw.reshape(B, q_vec.size(1), -1)
                cand1_valid_raw = cand1_valid_raw.reshape(B, q_vec.size(1), -1)
                cand1_nodes_raw = torch.where(cand1_valid_raw, cand1_nodes_raw, torch.full_like(cand1_nodes_raw, -1))
                cand1_nodes, cand1_mask = self._hqd_maybe_dedup_candidates_batched(cand1_nodes_raw)
                if collect_stats:
                    stage_stats["l1_candidates"] += int(cand1_mask.sum().item())
                if shallow_read in (2, 3):
                    # Shallow read: skip L1 scoring; sel1 empty -> reach L0 via the selected
                    # L2/L3 nodes' bags (final stage scores + prunes them).
                    Qc = int(q_vec.size(1))
                    sel1_nodes = torch.empty(B, Qc, 0, dtype=torch.long, device=device)
                    sel1_scores = torch.empty(B, Qc, 0, dtype=q_vec.dtype, device=device)
                    sel1_valid = torch.empty(B, Qc, 0, dtype=torch.bool, device=device)
                elif desc_stop_at >= 2:
                    # Shortcut: skip the L1 scoring/top-k. All L1 children of the selected
                    # L2 nodes become "selected", causally masked (L1 max_time vs query).
                    # The final L0 stage scores + prunes, so the read stays bounded.
                    sel1_nodes = cand1_nodes
                    l1_max_t = level_max_time_cache[1].index_select(0, cand1_nodes.clamp(min=0).reshape(-1)).view_as(cand1_nodes)
                    if causal:
                        cmask = (l1_max_t <= query_time.unsqueeze(-1)) if allow_same_time else (l1_max_t < query_time.unsqueeze(-1))
                        sel1_valid = cand1_mask & cmask & (l1_max_t >= 0)
                    else:
                        sel1_valid = cand1_mask
                    sel1_scores = torch.zeros_like(sel1_nodes, dtype=q_vec.dtype)
                else:
                    scores1 = self._hqd_score_candidates_batched(
                        query_vec=q_vec,
                        key_bank=k_all,
                        candidate_nodes=cand1_nodes,
                        candidate_mask=cand1_mask,
                        query_time=query_time,
                        candidate_max_time_cache=level_max_time_cache[1],
                        causal=causal,
                        allow_same_time=allow_same_time,
                    )
                    sel1_nodes, sel1_scores, sel1_valid = self._hqd_topk_from_scores_batched(cand1_nodes, scores1, hqd_topk_l1)
                    if hqd_global_topk > 0:
                        sel1_valid = self._hqd_apply_per_batch_global_topk(sel1_scores, sel1_valid, hqd_global_topk)
                if collect_stats:
                    stage_stats["l1_selected_total"] += int(sel1_valid.sum().item())
                final_nodes = sel1_nodes
                final_mask = sel1_valid
                final_level = 1
                _profile_add("l1_stage_ms", _stage_t0)
                if hqd_stop_level == 1:
                    top_nodes = final_nodes
                    top_valid = final_mask
                    if hqd_handoff_to_l0:
                        src_l0_nodes, src_l0_mask = self._hqd_expand_to_l0_batched(top_nodes, final_level, l2_to_l1, l1_to_l0)
                        dst_parent = query_nodes.view(1, -1, 1).expand(B, -1, 1)
                        dst_l0_nodes, dst_l0_mask = self._hqd_expand_to_l0_batched(dst_parent, hqd_query_level, l2_to_l1, l1_to_l0)
                        dst_flat = dst_l0_nodes.reshape(B, -1)
                        dst_mask_flat = dst_l0_mask.reshape(B, -1)
                        if dst_flat.numel() == 0 or src_l0_nodes.numel() == 0:
                            continue
                        safe_dst = dst_flat.clamp(min=0)
                        q_l0_vec = q_all.gather(1, safe_dst.view(B, -1, 1, 1).expand(-1, -1, q_all.size(2), q_all.size(3)))
                        src_exp = src_l0_nodes.unsqueeze(2).expand(B, src_l0_nodes.size(1), dst_l0_nodes.size(-1), src_l0_nodes.size(-1)).reshape(B, -1, src_l0_nodes.size(-1))
                        src_mask_exp = src_l0_mask.unsqueeze(2).expand(B, src_l0_mask.size(1), dst_l0_nodes.size(-1), src_l0_mask.size(-1)).reshape(B, -1, src_l0_mask.size(-1))
                        src_mask_exp = src_mask_exp & dst_mask_flat.unsqueeze(-1)
                        dst_time = level_max_time_cache[0].index_select(0, safe_dst.reshape(-1)).view_as(safe_dst)
                        scores0 = self._hqd_score_candidates_batched(q_l0_vec, k_all, src_exp, src_mask_exp, dst_time, level_max_time_cache[0], causal, allow_same_time)
                        top_nodes, top_scores, top_valid = self._hqd_topk_from_scores_batched(src_exp, scores0, hqd_topk_l0)
                        if hqd_global_topk > 0:
                            top_valid = self._hqd_apply_per_batch_global_topk(top_scores, top_valid, hqd_global_topk)
                        if collect_stats:
                            stage_stats["l0_candidates"] += int(src_mask_exp.sum().item())
                            stage_stats["final_candidate_total"] += int(src_mask_exp.sum().item())
                            stage_stats["l0_selected_total"] += int(top_valid.sum().item())
                            stage_stats["final_selected_total"] += int(top_valid.sum().item())
                            stage_stats["final_candidates"] += int(top_valid.sum().item())
                        keep = top_valid
                        selected_b.append(torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1).expand_as(top_nodes)[keep])
                        selected_src.append(top_nodes[keep])
                        selected_dst.append(dst_flat.unsqueeze(-1).expand_as(top_nodes)[keep])
                    else:
                        if collect_stats:
                            stage_stats["final_candidate_total"] += int(top_valid.numel())
                            stage_stats["final_selected_total"] += int(top_valid.sum().item())
                            stage_stats["final_candidates"] += int(top_valid.sum().item())
                        keep = top_valid
                        selected_b.append(batch_idx.expand_as(top_nodes)[keep])
                        selected_src.append(top_nodes[keep])
                        selected_dst.append(query_nodes.view(1, -1, 1).expand(B, -1, top_nodes.size(-1))[keep])
                    continue

                # Final stage: expand L1 -> L0 and optionally union a local window.
                _stage_t0 = _profile_start()
                cand0_nodes_raw, cand0_valid_raw = self._hqd_expand_children_batched(sel1_nodes, l1_to_l0)
                cand0_nodes_raw = cand0_nodes_raw.reshape(B, q_vec.size(1), -1)
                cand0_valid_raw = cand0_valid_raw.reshape(B, q_vec.size(1), -1)
                cand0_nodes_raw = torch.where(cand0_valid_raw, cand0_nodes_raw, torch.full_like(cand0_nodes_raw, -1))
                cand0_nodes, cand0_mask = self._hqd_maybe_dedup_candidates_batched(cand0_nodes_raw)
                if collect_stats:
                    stage_stats["l0_candidates"] += int(cand0_mask.sum().item())

                final_nodes = cand0_nodes
                final_mask = cand0_mask
                if hqd_query_level == 0 and self.hqd_include_local_window and local_window_size > 0:
                    local_nodes = self._hqd_local_window_candidates_batched(
                        query_positions=query_positions,
                        l0_nodes=l0_idx,
                        window_size=local_window_size,
                        causal=causal,
                        allow_same_time=allow_same_time,
                    )
                    local_nodes = local_nodes.unsqueeze(0).expand(B, -1, -1)
                    combined_nodes = torch.cat([final_nodes, local_nodes], dim=-1)
                    final_nodes, final_mask = self._hqd_maybe_dedup_candidates_batched(combined_nodes, force=True)
                    if collect_stats:
                        local_valid_count = int((local_nodes >= 0).sum().item())
                        stage_stats["local_window_total"] += local_valid_count
                        stage_stats["local_window_overlap"] += max(0, int(cand0_mask.sum().item()) + local_valid_count - int(final_mask.sum().item()))

                # Witness packets: union the selected coarse parents' L0 bag ids into the
                # candidate pool. Each active witness level (L1/L2/L3) contributes the bags
                # of its selected nodes. Scored normally (causal -inf survivors stay
                # excluded) but get a +bias so the final top-k keeps them. The L2/L3 bags
                # are cross-level (L0 descendants), letting a coarse selection reach L0
                # without descending through it.
                wit_ids = None
                if witness_active:
                    slots_w = int(witness_table.size(-1))
                    wflat = witness_table.reshape(B * N, slots_w)
                    boff = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1) * N
                    wit_parts: List[torch.Tensor] = []
                    for _wl, _wn, _wv in ((1, sel1_nodes, sel1_valid),
                                          (2, sel2_nodes, sel2_valid),
                                          (3, sel3_nodes, sel3_valid)):
                        if _wl not in self.witness_levels or _wn.numel() == 0:
                            continue
                        gidx = (_wn.clamp(min=0) + boff).reshape(-1)
                        wp = wflat.index_select(0, gidx).view(B, _wn.size(1), _wn.size(2), slots_w)
                        wp = torch.where(_wv.unsqueeze(-1), wp, torch.full_like(wp, -1))
                        wit_parts.append(wp.reshape(B, _wn.size(1), -1))
                    if wit_parts:
                        wit_ids = torch.cat(wit_parts, dim=-1)  # [B, Q, Σ topk_lX*slots]
                        combined = torch.cat([final_nodes, wit_ids], dim=-1)
                        final_nodes, final_mask = self._hqd_maybe_dedup_candidates_batched(combined, force=True)

                if collect_stats:
                    stage_stats["final_candidate_total"] += int(final_mask.sum().item())

                scores0 = self._hqd_score_candidates_batched(
                    query_vec=q_vec,
                    key_bank=k_all,
                    candidate_nodes=final_nodes,
                    candidate_mask=final_mask,
                    query_time=query_time,
                    candidate_max_time_cache=level_max_time_cache[0],
                    causal=causal,
                    allow_same_time=allow_same_time,
                )
                if witness_active and wit_ids is not None:
                    # Membership of each final candidate in the per-query witness set, via
                    # searchsorted (O(B*Q*F*logW)) rather than a [B,Q,F,W] broadcast — the
                    # latter explodes at long context (Q ~ seq_len) with per-head witnesses.
                    ws, _ = torch.sort(wit_ids, dim=-1)
                    pos = torch.searchsorted(ws, final_nodes).clamp(max=ws.size(-1) - 1)
                    is_wit = (ws.gather(-1, pos) == final_nodes) & (final_nodes >= 0)
                    scores0 = scores0 + float(self.witness_score_bias) * is_wit.to(scores0.dtype)
                if bool(self.hqd_l0_topk_enable):
                    top_nodes, top_scores, top_valid = self._hqd_topk_from_scores_batched(final_nodes, scores0, hqd_topk_l0)
                else:
                    top_nodes = final_nodes
                    top_scores = scores0
                    top_valid = final_mask & torch.isfinite(scores0)
                if hqd_global_topk > 0:
                    top_valid = self._hqd_apply_per_batch_global_topk(top_scores, top_valid, hqd_global_topk)
                if collect_stats:
                    stage_stats["l0_selected_total"] += int(top_valid.sum().item())
                    stage_stats["final_selected_total"] += int(top_valid.sum().item())
                    stage_stats["final_candidates"] += int(top_valid.sum().item())
                _profile_add("l0_stage_ms", _stage_t0)

                keep = top_valid
                selected_b.append(batch_idx.expand_as(top_nodes)[keep])
                selected_src.append(top_nodes[keep])
                selected_dst.append(query_nodes.view(1, -1, 1).expand(B, -1, top_nodes.size(-1))[keep])

                # Mix selected coarse summary nodes (L1/L2/L3) into the read, per
                # hqd_read_levels. Their validity masks already encode causal filtering
                # (a coarse node is valid only if all its descendants precede the query),
                # so this stays causal-safe at query_level 0. Level-aware scoring in
                # message_passing handles the cross-level q.k.
                for _lvl, _nodes, _valid in (
                    (1, sel1_nodes, sel1_valid),
                    (2, sel2_nodes, sel2_valid),
                    (3, sel3_nodes, sel3_valid),
                ):
                    if _lvl in read_coarse_levels and _nodes.numel() > 0:
                        selected_b.append(batch_idx.expand_as(_nodes)[_valid])
                        selected_src.append(_nodes[_valid])
                        selected_dst.append(query_nodes.view(1, -1, 1).expand(B, -1, _nodes.size(-1))[_valid])

                # per_position window-bag: re-target each query's level-X selections to the
                # query's level-X ancestor (dst). Causal by construction: a query's selections
                # are <= its own time <= the ancestor's max time, so no future-of-ancestor info
                # enters its bag. src_t <= dst_t is then automatic; we mask it defensively.
                if bag_levels and bag_routing == "per_position":
                    if bag_cap <= 0:
                        for _bl, _bn, _bv in ((1, sel1_nodes, sel1_valid), (2, sel2_nodes, sel2_valid), (3, sel3_nodes, sel3_valid)):
                            if _bl in bag_levels and _bn.numel() > 0:
                                anc = bag_ancestors[_bl].index_select(0, query_nodes)        # [Qc]
                                src_t = level_max_time_cache[_bl].index_select(0, _bn.clamp(min=0).reshape(-1)).view_as(_bn)
                                dst_t = level_max_time_cache[_bl].index_select(0, anc.clamp(min=0)).view(1, -1, 1)
                                ok = _bv & (anc.view(1, -1, 1) >= 0) & (src_t >= 0) & (src_t <= dst_t)
                                dst = anc.view(1, -1, 1).expand(B, int(anc.numel()), _bn.size(-1))
                                selected_b.append(batch_idx.expand_as(_bn)[ok])
                                selected_src.append(_bn[ok])
                                selected_dst.append(dst[ok])
                    else:
                        pp_qpos.append(query_nodes)
                        for _bl, _bn, _bs, _bv in ((1, sel1_nodes, sel1_scores, sel1_valid),
                                                   (2, sel2_nodes, sel2_scores, sel2_valid),
                                                   (3, sel3_nodes, sel3_scores, sel3_valid)):
                            if _bl in pp_accum and _bn.numel() > 0:
                                pp_accum[_bl]["nodes"].append(_bn)
                                pp_accum[_bl]["scores"].append(_bs)
                                pp_accum[_bl]["valid"].append(_bv)

                # Window-bag: stash this chunk's per-level selections (masked invalid -> -inf
                # score) so the union can be scattered to each level's nodes after the loop.
                if bag_levels and bag_routing == "global":
                    for _lvl, _nodes, _scores, _valid in (
                        (1, sel1_nodes, sel1_scores, sel1_valid),
                        (2, sel2_nodes, sel2_scores, sel2_valid),
                        (3, sel3_nodes, sel3_scores, sel3_valid),
                    ):
                        if _lvl in bag_accum and _nodes.numel() > 0:
                            sc = _scores.masked_fill(~_valid, float("-inf"))
                            bag_accum[_lvl]["nodes"].append(_nodes.reshape(B, -1))
                            bag_accum[_lvl]["scores"].append(sc.reshape(B, -1))

        # Window-bag scatter (global routing): every level-L node attends to the top bag_topk
        # level-L nodes the L0 routing surfaced, masked causal via src_max <= dst_min so a
        # coarse node never sees a sibling whose span reaches its own future (avoids the
        # query_level>=1 leak). per_position routing is a follow-up.
        if bag_levels and bag_routing == "global" and selected_b:
            bag_topk = int(getattr(self, "hqd_window_bag_topk", 0)) or 16
            for L in bag_levels:
                if not bag_accum[L]["nodes"]:
                    continue
                dst_nodes = level_indices.get(L)
                if dst_nodes is None or dst_nodes.numel() == 0:
                    continue
                nodes = torch.cat(bag_accum[L]["nodes"], dim=1)        # [B, M]
                scores = torch.cat(bag_accum[L]["scores"], dim=1)      # [B, M]
                k = min(bag_topk, int(nodes.size(1)))
                bv, bi = torch.topk(scores, k=k, dim=1)                # [B, k]
                bag_nodes = nodes.gather(1, bi)                        # [B, k]
                bag_ok = torch.isfinite(bv)                           # [B, k]
                src_t = level_max_time_cache[L].index_select(0, bag_nodes.reshape(-1)).view(B, k)
                dst_min = level_min_time_cache[L].index_select(0, dst_nodes)        # [n_L]
                nL = int(dst_nodes.numel())
                src_e = bag_nodes.view(B, 1, k).expand(B, nL, k)
                ok = (bag_ok.view(B, 1, k) & (src_t.view(B, 1, k) >= 0)
                      & (src_t.view(B, 1, k) <= dst_min.view(1, nL, 1)))
                dst_e = dst_nodes.view(1, nL, 1).expand(B, nL, k)
                bidx_e = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1).expand(B, nL, k)
                selected_b.append(bidx_e[ok])
                selected_src.append(src_e[ok])
                selected_dst.append(dst_e[ok])

        # per_position capped: per coarse node, gather its descendants' selections (nodes +
        # the descent's own scores), take the top bag_cap by score, emit as bag edges. Causal:
        # descendants are all <= the coarse node's max time, so the bag carries no future info.
        if bag_levels and bag_routing == "per_position" and bag_cap > 0 and pp_qpos:
            query_order = torch.cat(pp_qpos, dim=0)
            Qtot = int(query_order.numel())
            inv_query = torch.full((int(N),), -1, dtype=torch.long, device=device)
            inv_query[query_order] = torch.arange(Qtot, device=device, dtype=torch.long)
            child_tables = {1: l1_to_l0, 2: l2_to_l1, 3: l3_to_l2}
            for X in bag_levels:
                if not pp_accum[X]["nodes"]:
                    continue
                nodes_X = level_indices.get(X)
                if nodes_X is None or nodes_X.numel() == 0:
                    continue
                selN = torch.cat(pp_accum[X]["nodes"], dim=1)     # [B, Qtot, kX]
                selS = torch.cat(pp_accum[X]["scores"], dim=1)
                selV = torch.cat(pp_accum[X]["valid"], dim=1)
                kX = int(selN.size(-1)); nX = int(nodes_X.numel())
                desc = nodes_X.view(1, nX)
                for lvl in range(X, 0, -1):
                    ch, cv = self._hqd_expand_children_batched(desc, child_tables[lvl])
                    ch = ch.reshape(1, nX, -1); cv = cv.reshape(1, nX, -1)
                    desc = torch.where(cv, ch, torch.full_like(ch, -1))
                D = int(desc.size(-1))                            # [1, nX, D] L0 ids
                qp = inv_query.index_select(0, desc.clamp(min=0).reshape(-1)).view(1, nX, D)
                qp_valid = (desc >= 0) & (qp >= 0)
                boff = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1) * Qtot
                gidx = (qp.clamp(min=0).expand(B, nX, D) + boff).reshape(-1)
                pn = selN.reshape(B * Qtot, kX).index_select(0, gidx).view(B, nX, D, kX)
                ps = selS.reshape(B * Qtot, kX).index_select(0, gidx).view(B, nX, D, kX)
                pv = selV.reshape(B * Qtot, kX).index_select(0, gidx).view(B, nX, D, kX)
                pv = pv & qp_valid.view(1, nX, D, 1) & (pn >= 0)
                P = D * kX
                pn = pn.reshape(B, nX, P)
                ps = ps.reshape(B, nX, P).masked_fill(~pv.reshape(B, nX, P), float("-inf"))
                k = min(int(bag_cap), P)
                bv, bi = torch.topk(ps, k=k, dim=-1)
                bag = torch.where(torch.isfinite(bv), pn.gather(-1, bi), torch.full_like(pn[..., :k], -1))
                bag, _ = self._hqd_maybe_dedup_candidates_batched(bag, force=True)
                k2 = int(bag.size(-1))
                src_t = level_max_time_cache[X].index_select(0, bag.clamp(min=0).reshape(-1)).view(B, nX, k2)
                dst_t = level_max_time_cache[X].index_select(0, nodes_X).view(1, nX, 1)
                bok = (bag >= 0) & (src_t >= 0) & (src_t <= dst_t)
                dst_e = nodes_X.view(1, nX, 1).expand(B, nX, k2)
                bidx_e = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1).expand(B, nX, k2)
                selected_b.append(bidx_e[bok])
                selected_src.append(bag[bok])
                selected_dst.append(dst_e[bok])

        # Option 2 — bidirectional coarse routing: each coarse node attends to its top-r
        # same-level neighbours (cheap: few nodes), and matched pairs surface bag<->bag L0
        # edges. NOT AR-causal (a coarse node queries with its full-span vector), so it is
        # gated to non-causal (masked-diffusion / bidirectional) runs only.
        coarse_route_levels = [int(l) for l in getattr(self, "hqd_coarse_route_levels", []) if int(l) in (1, 2, 3)]
        if coarse_route_levels and causal:
            if not bool(getattr(self, "_hqd_coarse_route_causal_warned", False)):
                logger.warning(
                    "hqd_coarse_route_levels is set but the graph is causal (AR): coarse-as-query "
                    "leaks future info (~7e-3). Coarse routing is disabled under AR; use it only "
                    "for masked-diffusion / bidirectional training."
                )
                self._hqd_coarse_route_causal_warned = True
            coarse_route_levels = []
        if coarse_route_levels and witness_table is not None:
            route_r = int(getattr(self, "hqd_coarse_route_topk", 8))
            sw = int(witness_table.size(-1))
            for L in coarse_route_levels:
                if L not in self.witness_levels:
                    continue
                nodes = level_indices.get(L)
                if nodes is None or int(nodes.numel()) < 2:
                    continue
                nL = int(nodes.numel())
                qL = q_all.index_select(1, nodes)                    # [B, nL, H, D]
                kL = k_all.index_select(1, nodes)
                sc = torch.einsum("bnhd,bmhd->bnm", qL, kL) / float(qL.size(-2) * math.sqrt(float(qL.size(-1))))
                r = min(route_r, nL)
                _, ri = torch.topk(sc, k=r, dim=-1)                  # [B, nL, r] -> positions in `nodes`
                bags = witness_table.index_select(1, nodes)          # [B, nL, sw] L0 ids (-1 pad)
                nbr_bags = torch.gather(
                    bags, 1, ri.reshape(B, nL * r, 1).expand(B, nL * r, sw)
                ).view(B, nL, r, sw)                                 # [B, nL, r, sw]
                dst_e = bags.view(B, nL, 1, sw, 1).expand(B, nL, r, sw, sw)   # node i's bag = dst
                src_e = nbr_bags.view(B, nL, r, 1, sw).expand(B, nL, r, sw, sw)  # neighbour j's bag = src
                valid = (dst_e >= 0) & (src_e >= 0)
                bidx_e = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1, 1, 1).expand_as(dst_e)
                selected_b.append(bidx_e[valid])
                selected_src.append(src_e[valid])
                selected_dst.append(dst_e[valid])

        if not selected_b:
            self._hqd_reuse_cache = None
            self._last_hqd_added_total = 0
            self._last_hqd_selected_total = 0
            self._last_hqd_avg_l0 = None
            self._last_hqd_stage_stats = dict(stage_stats)
            self._last_hqd_profile_stats = profile_stats if profile_enabled else None
            if profile_enabled:
                profile_stats["total_ms"] = (time.monotonic() - t0) * 1000.0
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty, empty, 0, stage_stats

        _cat_t0 = _profile_start()
        b_idx = torch.cat(selected_b, dim=0)
        src_idx = torch.cat(selected_src, dim=0)
        dst_idx = torch.cat(selected_dst, dim=0)
        if b_idx.numel() == 0:
            self._hqd_reuse_cache = None
            self._last_hqd_added_total = 0
            self._last_hqd_selected_total = 0
            self._last_hqd_avg_l0 = None
            self._last_hqd_stage_stats = dict(stage_stats)
            self._last_hqd_profile_stats = profile_stats if profile_enabled else None
            if profile_enabled:
                profile_stats["total_ms"] = (time.monotonic() - t0) * 1000.0
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty, empty, 0, stage_stats

        self._hqd_reuse_cache = {
            # Cache the flattened batched edge list as-is; reuse must not broadcast it again.
            "b_idx": b_idx.detach(),
            "src_idx": src_idx.detach(),
            "dst_idx": dst_idx.detach(),
            "B": int(B),
            "N": int(N),
            "device": str(device),
        }
        _profile_add("cat_cache_ms", _cat_t0)

        added_total = int(b_idx.numel())
        stage_stats["causal_violations_post"] = 0
        self._last_hqd_added_total = int(added_total)
        self._last_hqd_selected_total = int(added_total)
        self._last_hqd_stage_stats = dict(stage_stats)
        self._last_hqd_profile_stats = profile_stats if profile_enabled else None
        if profile_enabled:
            profile_stats["total_ms"] = (time.monotonic() - t0) * 1000.0

        if self.hqd_debug:
            logger.info(
                "HQD stats: queries=%d final=%d l3=%d l2=%d l1=%d l0=%d local=%d overlap=%d",
                int(stage_stats["queries"]),
                int(stage_stats["final_candidates"]),
                int(stage_stats["l3_selected_total"]),
                int(stage_stats["l2_selected_total"]),
                int(stage_stats["l1_selected_total"]),
                int(stage_stats["l0_selected_total"]),
                int(stage_stats["local_window_total"]),
                int(stage_stats["local_window_overlap"]),
            )

        return b_idx, src_idx, dst_idx, int(added_total), stage_stats

    def _zipper_apply_staged_ephemeral_batched(
        self,
        transformer,
        x_bnh: torch.Tensor,
        base_edge_index: torch.Tensor,
        node_level: torch.Tensor,
        node_ar_time: Optional[torch.Tensor],
        x_in: Optional[torch.Tensor] = None,
        _qk_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, int]]:
        device = x_bnh.device
        max_candidates = int(self.zip_max_candidate_edges)
        profile_enabled = bool(getattr(self, "zip_log_enable", False))
        profile_stats: Dict[str, float] = {}
        stage_stats: Dict[str, int] = {
            "l3_candidates": 0,
            "l3_selected_union": 0,
            "l3_selected_total": 0,
            "l2_candidates": 0,
            "l2_selected_union": 0,
            "l2_selected_total": 0,
            "l2_fallback_used": 0,
            "l1_candidates": 0,
            "l1_selected_union": 0,
            "l1_selected_total": 0,
            "l1_fallback_used": 0,
            "l0_candidates": 0,
            "l0_selected_union": 0,
            "l0_selected_total": 0,
            "l0_fallback_used": 0,
        }

        _t0 = time.monotonic() if profile_enabled else 0.0

        # Phase 2: cached children tables
        l3_to_l2 = self._zipper_build_children_table_cached(base_edge_index, node_level, 3, 2)
        l2_to_l1 = self._zipper_build_children_table_cached(base_edge_index, node_level, 2, 1)
        l1_to_l0 = None
        if self.zip_depth == "l0":
            l1_to_l0 = self._zipper_build_children_table_cached(base_edge_index, node_level, 1, 0)

        if profile_enabled:
            _t1 = time.monotonic()
            profile_stats["children_table_ms"] = (_t1 - _t0) * 1000.0

        if x_in is None:
            x_in = transformer.norm1(x_bnh) if hasattr(transformer, "norm1") else x_bnh

        # Phase 3: compute Q/K once and reuse across all stages
        mp = transformer.message_passing
        B, N = x_bnh.size(0), x_bnh.size(1)
        qk_cache_local = _qk_cache
        mode = getattr(self, "zip_score_mode", "fast_qk")
        if qk_cache_local is None and mode == "fast_qk" and all(hasattr(mp, a) for a in ("q_proj", "k_proj", "level_embedding", "level_attn")):
            with torch.no_grad():
                q_all = mp.q_proj(x_in).view(B, N, mp.num_heads, mp.head_dim)
                k_all = mp.k_proj(x_in).view(B, N, mp.num_heads, mp.head_dim)
                level_emb = mp.level_embedding(node_level)
            qk_cache_local = (q_all, k_all, level_emb)

        if profile_enabled:
            _t2 = time.monotonic()
            profile_stats["qk_proj_ms"] = (_t2 - _t1) * 1000.0

        selected_b: List[torch.Tensor] = []
        selected_src: List[torch.Tensor] = []
        selected_dst: List[torch.Tensor] = []
        selected_w: List[torch.Tensor] = []

        def _append_stage_selected(stage_edge_index: torch.Tensor, keep_bk: torch.Tensor, attn_bk: torch.Tensor) -> None:
            selected = keep_bk.nonzero(as_tuple=False)
            if selected.numel() == 0:
                return
            b_idx = selected[:, 0]
            k_idx = selected[:, 1]
            selected_b.append(b_idx)
            selected_src.append(stage_edge_index[0, k_idx])
            selected_dst.append(stage_edge_index[1, k_idx])
            selected_w.append(attn_bk[b_idx, k_idx])

        # --- L3 stage ---
        if profile_enabled:
            _ts = time.monotonic()
        stage_l3 = self._zipper_dense_pairs_from_level(
            node_level=node_level,
            level=3,
            cap=int(self.zip_max_l3_pairs),
            device=device,
        )
        stage_l3 = self._zipper_filter_ar_edges(stage_l3, node_level, node_ar_time)
        valid_l3 = torch.ones(
            (x_bnh.size(0), stage_l3.size(1)),
            dtype=torch.bool,
            device=device,
        )
        keep_l3_bk, attn_l3_bk, l3_stats = self._zipper_run_ephemeral_stage_batched(
            transformer=transformer,
            x_bnh=x_bnh,
            edge_index=stage_l3,
            node_level=node_level,
            valid_bk=valid_l3,
            x_in=x_in,
            _qk_cache=qk_cache_local,
        )
        _append_stage_selected(stage_l3, keep_l3_bk, attn_l3_bk)
        stage_stats["l3_candidates"] = int(l3_stats["candidates"])
        stage_stats["l3_selected_union"] = int(l3_stats["selected_union"])
        stage_stats["l3_selected_total"] = int(l3_stats["selected_total"])
        if profile_enabled:
            profile_stats["l3_stage_ms"] = (time.monotonic() - _ts) * 1000.0

        # --- L2 stage ---
        if profile_enabled:
            _ts = time.monotonic()
        stage_l2, valid_l2_bk, l2_fallback_count = self._zipper_build_stage_candidates_with_mask_fast(
            parent_edge_index=stage_l3,
            parent_keep_bk=keep_l3_bk,
            children_table=l3_to_l2,
            fallback_level=2,
            fallback_cap=int(self.zip_max_l2_pairs),
            node_level=node_level,
            node_ar_time=node_ar_time,
            device=device,
        )
        keep_l2_bk, attn_l2_bk, l2_stats = self._zipper_run_ephemeral_stage_batched(
            transformer=transformer,
            x_bnh=x_bnh,
            edge_index=stage_l2,
            node_level=node_level,
            valid_bk=valid_l2_bk,
            x_in=x_in,
            _qk_cache=qk_cache_local,
        )
        _append_stage_selected(stage_l2, keep_l2_bk, attn_l2_bk)
        stage_stats["l2_candidates"] = int(l2_stats["candidates"])
        stage_stats["l2_selected_union"] = int(l2_stats["selected_union"])
        stage_stats["l2_selected_total"] = int(l2_stats["selected_total"])
        stage_stats["l2_fallback_used"] = int(l2_fallback_count)
        if profile_enabled:
            profile_stats["l2_stage_ms"] = (time.monotonic() - _ts) * 1000.0

        # --- L1 stage ---
        if profile_enabled:
            _ts = time.monotonic()
        stage_l1, valid_l1_bk, l1_fallback_count = self._zipper_build_stage_candidates_with_mask_fast(
            parent_edge_index=stage_l2,
            parent_keep_bk=keep_l2_bk,
            children_table=l2_to_l1,
            fallback_level=1,
            fallback_cap=max_candidates,
            node_level=node_level,
            node_ar_time=node_ar_time,
            device=device,
        )
        keep_l1_bk, attn_l1_bk, l1_stats = self._zipper_run_ephemeral_stage_batched(
            transformer=transformer,
            x_bnh=x_bnh,
            edge_index=stage_l1,
            node_level=node_level,
            valid_bk=valid_l1_bk,
            x_in=x_in,
            _qk_cache=qk_cache_local,
        )
        _append_stage_selected(stage_l1, keep_l1_bk, attn_l1_bk)
        stage_stats["l1_candidates"] = int(l1_stats["candidates"])
        stage_stats["l1_selected_union"] = int(l1_stats["selected_union"])
        stage_stats["l1_selected_total"] = int(l1_stats["selected_total"])
        stage_stats["l1_fallback_used"] = int(l1_fallback_count)
        if profile_enabled:
            profile_stats["l1_stage_ms"] = (time.monotonic() - _ts) * 1000.0

        # --- L0 stage (optional) ---
        if self.zip_depth == "l0" and l1_to_l0 is not None:
            if profile_enabled:
                _ts = time.monotonic()
            stage_l0, valid_l0_bk, l0_fallback_count = self._zipper_build_stage_candidates_with_mask_fast(
                parent_edge_index=stage_l1,
                parent_keep_bk=keep_l1_bk,
                children_table=l1_to_l0,
                fallback_level=0,
                fallback_cap=max_candidates,
                node_level=node_level,
                node_ar_time=node_ar_time,
                device=device,
            )
            keep_l0_bk, attn_l0_bk, l0_stats = self._zipper_run_ephemeral_stage_batched(
                transformer=transformer,
                x_bnh=x_bnh,
                edge_index=stage_l0,
                node_level=node_level,
                valid_bk=valid_l0_bk,
                x_in=x_in,
                _qk_cache=qk_cache_local,
            )
            _append_stage_selected(stage_l0, keep_l0_bk, attn_l0_bk)
            stage_stats["l0_candidates"] = int(l0_stats["candidates"])
            stage_stats["l0_selected_union"] = int(l0_stats["selected_union"])
            stage_stats["l0_selected_total"] = int(l0_stats["selected_total"])
            stage_stats["l0_fallback_used"] = int(l0_fallback_count)
            if profile_enabled:
                profile_stats["l0_stage_ms"] = (time.monotonic() - _ts) * 1000.0

        # --- Injection ---
        if profile_enabled:
            _ts = time.monotonic()
        if selected_b:
            b_idx = torch.cat(selected_b, dim=0)
            src_idx = torch.cat(selected_src, dim=0)
            dst_idx = torch.cat(selected_dst, dim=0)
            weights = torch.cat(selected_w, dim=0).to(dtype=x_bnh.dtype)
            x_bnh, zip_added_total = self._zipper_inject_selected_messages_batched(
                transformer=transformer,
                x_bnh=x_bnh,
                b_idx=b_idx,
                src_idx=src_idx,
                dst_idx=dst_idx,
                weights=weights,
                x_in=x_in,
            )
        else:
            zip_added_total = 0
        if profile_enabled:
            profile_stats["inject_ms"] = (time.monotonic() - _ts) * 1000.0
            profile_stats["total_ms"] = (time.monotonic() - _t0) * 1000.0

        self._last_zip_profile_stats = profile_stats if profile_enabled else None
        return x_bnh, int(zip_added_total), stage_stats

    def _zipper_apply_gate(
        self,
        scores: torch.Tensor,
        hard_mask: torch.Tensor,
    ) -> torch.Tensor:
        m_soft = torch.sigmoid((scores - self.zip_gate_center) / self.zip_gate_tau)
        m_hard = hard_mask.to(dtype=scores.dtype)
        return (m_hard - m_soft).detach() + m_soft

    def _zipper_dense_l3_pairs(
        self,
        l3_idx: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attn_agg: Optional[torch.Tensor],
        p_update: Optional[torch.Tensor],
        cap: int,
    ) -> torch.Tensor:
        if l3_idx.numel() <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=l3_idx.device)

        pairs = torch.cartesian_prod(l3_idx, l3_idx)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        total = pairs.size(0)
        if total == 0:
            return torch.empty((2, 0), dtype=torch.long, device=l3_idx.device)

        if total <= cap or not self.zip_l3_all_pairs_sample:
            return pairs.t()

        weights = None
        if self.zip_l3_pair_sampling_mode == "attn_entropy":
            w_attn = self.zip_l3_pair_attn_weight
            w_ent = self.zip_l3_pair_entropy_weight
            attn_score = torch.zeros(total, device=l3_idx.device, dtype=torch.float)
            if edge_attn_agg is not None:
                l3_set = set(l3_idx.tolist())
                src, dst = edge_index
                edge_map = {}
                for s, d, a in zip(src.tolist(), dst.tolist(), edge_attn_agg.tolist()):
                    if s in l3_set and d in l3_set:
                        edge_map[(s, d)] = a
                for i, (u, v) in enumerate(pairs.tolist()):
                    attn_score[i] = edge_map.get((u, v), 0.0)
            ent_score = torch.zeros(total, device=l3_idx.device, dtype=torch.float)
            if p_update is not None:
                ent_score = (p_update[pairs[:, 0]] + p_update[pairs[:, 1]]) * 0.5
            weights = w_attn * attn_score + w_ent * ent_score

        if weights is None or weights.sum() <= 0:
            perm = torch.randperm(total, device=l3_idx.device)[:cap]
            return pairs[perm].t()

        weights = weights.clamp_min(0.0)
        if weights.sum() <= 0:
            perm = torch.randperm(total, device=l3_idx.device)[:cap]
            return pairs[perm].t()

        idx = torch.multinomial(weights, cap, replacement=False)
        return pairs[idx].t()
    




    def _set_true_batch_nozip_tail_metrics(
        self,
        cycles_used: int,
        zip_added_total: Optional[int] = None,
        zip_stage_stats: Optional[Dict[str, int]] = None,
        hqd_added_total: Optional[int] = None,
        hqd_stage_stats: Optional[Dict[str, int]] = None,
        hqd_profile_stats: Optional[Dict[str, float]] = None,
        hqd_avg_l0: Optional[float] = None,
        attn_saved_pct: Optional[float] = None,
        win_dense_pct: Optional[float] = None,
        sparse_dense_pct: Optional[float] = None,
        graph_l0_edges_per_sample: Optional[int] = None,
        l0_attn_is_causal: Optional[bool] = None,
    ) -> None:
        self._last_zip_added_total = None if zip_added_total is None else int(zip_added_total)
        self._last_zip_stage_stats = None if zip_stage_stats is None else dict(zip_stage_stats)
        self._last_hqd_added_total = None if hqd_added_total is None else int(hqd_added_total)
        self._last_hqd_stage_stats = None if hqd_stage_stats is None else dict(hqd_stage_stats)
        self._last_hqd_profile_stats = None if hqd_profile_stats is None else dict(hqd_profile_stats)
        self._last_hqd_avg_l0 = None if hqd_avg_l0 is None else float(hqd_avg_l0)
        self._last_attn_saved_pct = None if attn_saved_pct is None else float(attn_saved_pct)
        self._last_win_dense_pct = None if win_dense_pct is None else float(win_dense_pct)
        self._last_sparse_dense_pct = None if sparse_dense_pct is None else float(sparse_dense_pct)
        self._last_graph_l0_edges = None if graph_l0_edges_per_sample is None else int(graph_l0_edges_per_sample)
        self._last_l0_attn_is_causal = None if l0_attn_is_causal is None else bool(l0_attn_is_causal)
        if zip_added_total is None:
            self._last_zip_msg_norm_ratio_mean = None
            self._last_zip_msg_norm_ratio_max = None
        self._last_cycles_used = int(cycles_used)


        self._last_auto_prob_cycles_ema = None
        self._last_auto_prob_min_prob = None
        self._last_auto_prob_max_cycle_rate = None

    def _compute_true_batch_aux_loss(
        self,
        x_bnh: torch.Tensor,
        base_ei: torch.Tensor,
        base_nl: torch.Tensor,
        base_lo: Optional[torch.Tensor],
        base_ar_time: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from torch_geometric.data import Data

        device = x_bnh.device
        if not (self.training and self.use_aux_loss):
            return torch.zeros((), device=device, dtype=x_bnh.dtype)

        B, N, H = x_bnh.shape
        aux_mode = getattr(self, "true_batch_aux_mode", "per_sample_mean")

        if aux_mode == "per_sample_mean" or B == 1:
            losses = []
            for b in range(B):
                g_aux = Data(x=x_bnh[b], edge_index=base_ei, node_level=base_nl)
                if base_lo is not None:
                    g_aux.level_offsets = base_lo
                if base_ar_time is not None:
                    g_aux.node_ar_time = base_ar_time
                losses.append(self._compute_hierarchy_aux_loss_runtime(g_aux))
            if not losses:
                return torch.zeros((), device=device, dtype=x_bnh.dtype)
            return torch.stack(losses).mean()

        # exact_blockdiag debug mode: match block-diagonal aux topology exactly
        batch_offsets = (
            torch.arange(B, device=device, dtype=base_ei.dtype).view(B, 1, 1) * N
        )
        ei_merged = (base_ei.unsqueeze(0) + batch_offsets).transpose(0, 1).reshape(2, -1)
        nl_merged = base_nl.repeat(B)

        g_aux = Data(x=x_bnh.reshape(B * N, H), edge_index=ei_merged, node_level=nl_merged)
        if base_lo is not None:
            g_aux.level_offsets = base_lo
        if base_ar_time is not None:
            g_aux.node_ar_time = base_ar_time.repeat(B)
        return self._compute_hierarchy_aux_loss_runtime(g_aux)

    def _lap_pe_raw_on_device(self, lap_pe: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (
            int(lap_pe.data_ptr()),
            tuple(int(v) for v in lap_pe.shape),
            str(device),
            str(dtype),
        )
        cached = getattr(self, "_lap_pe_device_cache", None)
        if cached is not None and getattr(self, "_lap_pe_device_cache_key", None) == key:
            return cached
        lap_pe_device = lap_pe.to(device=device, dtype=dtype, non_blocking=True)
        self._lap_pe_device_cache = lap_pe_device
        self._lap_pe_device_cache_key = key
        return lap_pe_device

    def _refine_step_true_batch_native(
        self,
        transformer,
        x_bnh: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        level_offsets: Optional[torch.Tensor],
        pos_local: torch.Tensor,
        edge_attr_work: Optional[torch.Tensor],
        edge_type_work: Optional[torch.Tensor] = None,
        hqd_b_idx: Optional[torch.Tensor] = None,
        hqd_src_idx: Optional[torch.Tensor] = None,
        hqd_dst_idx: Optional[torch.Tensor] = None,
        active_levels: Optional[List[int]] = None,
        xq_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hqd_edges = None
        if hqd_b_idx is not None and hqd_b_idx.numel() > 0:
            hqd_edges = (hqd_b_idx, hqd_src_idx, hqd_dst_idx)
        flash_local_checkpoint_reentrant = bool(
            self.l0_local_backend == "flash"
            and int(getattr(self, "l0_local_window", 0)) > 0
            and bool(getattr(self, "local_attn_config", {}))
        )
        checkpoint_reentrant = bool(hqd_edges is not None or flash_local_checkpoint_reentrant)

        def _forward_with_hqd(
            x_in: torch.Tensor,
            edge_index_in: torch.Tensor,
            node_level_in: torch.Tensor,
            level_offsets_in: Optional[torch.Tensor],
            pos_local_in: torch.Tensor,
            edge_attr_work_in: Optional[torch.Tensor],
            edge_type_work_in: Optional[torch.Tensor],
            hqd_b_idx_in: torch.Tensor,
            hqd_src_idx_in: torch.Tensor,
            hqd_dst_idx_in: torch.Tensor,
            xq_gate_in: torch.Tensor,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            hqd_edges_in = None
            if hqd_b_idx_in.numel() > 0:
                hqd_edges_in = (hqd_b_idx_in, hqd_src_idx_in, hqd_dst_idx_in)
            edge_type_arg = edge_type_work_in if edge_type_work_in.numel() > 0 else None
            # xq per-query relevance gate: MUST enter as an explicit checkpoint arg, not a
            # closure/module-attr capture — the reentrant path below calls autograd.backward
            # internally, and a live-graph closure tensor makes that inner backward walk out
            # into (and free) the main graph -> "backward through the graph a second time".
            # Setting the hook attr HERE also means backward recompute re-establishes the
            # correct per-layer gate snapshot (multi-round re-nomination safe).
            mp_in = getattr(transformer, "message_passing", None)
            if mp_in is not None:
                mp_in._hqd_out_query_gate = xq_gate_in if xq_gate_in.numel() > 0 else None
            try:
                layer_fn = self._layer_callable(transformer)
                return layer_fn(
                    x_in,
                    edge_index_in,
                    node_level_in,
                    level_offsets_in,
                    pos_local_in,
                    edge_attr_work_in,
                    hqd_edges_in,
                    active_levels=active_levels,
                    edge_type=edge_type_arg,
                )
            finally:
                if mp_in is not None:
                    mp_in._hqd_out_query_gate = None

        if self.use_gradient_checkpointing and torch.is_grad_enabled():
            # HQD/PyG and flash local attention have shown non-reentrant replay issues.
            # Keep non-reentrant for the simpler paths, use reentrant only when needed.
            x_next, new_edge_attr = torch.utils.checkpoint.checkpoint(
                _forward_with_hqd,
                x_bnh,
                edge_index,
                node_level,
                level_offsets if level_offsets is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                pos_local,
                edge_attr_work,
                edge_type_work if edge_type_work is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                hqd_b_idx if hqd_b_idx is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                hqd_src_idx if hqd_src_idx is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                hqd_dst_idx if hqd_dst_idx is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                xq_gate if xq_gate is not None else torch.empty(0, device=x_bnh.device, dtype=x_bnh.dtype),
                use_reentrant=checkpoint_reentrant,
            )
        else:
            x_next, new_edge_attr = _forward_with_hqd(
                x_bnh,
                edge_index,
                node_level,
                level_offsets if level_offsets is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                pos_local,
                edge_attr_work,
                edge_type_work if edge_type_work is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                hqd_b_idx if hqd_b_idx is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                hqd_src_idx if hqd_src_idx is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                hqd_dst_idx if hqd_dst_idx is not None else torch.empty(0, device=x_bnh.device, dtype=torch.long),
                xq_gate if xq_gate is not None else torch.empty(0, device=x_bnh.device, dtype=x_bnh.dtype),
            )
        return x_next, new_edge_attr

    def _refine_step_blockdiag(
        self,
        transformer,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        node_level: torch.Tensor,
        pos_flat: torch.Tensor,
        edge_attr_input: Optional[torch.Tensor],
        keep_grad: bool,
        level_grid_shapes: Optional[Union[List[Any], Tuple[Any, ...], Dict[int, Tuple[int, int]]]] = None,
        spatial_metric: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mp = getattr(transformer, "message_passing", None)
        if mp is not None:
            local_attn_cfg = getattr(self, "local_attn_config", {})
            use_l0_local = bool(self.l0_local_backend != "pyg" and int(self.l0_local_window) > 0)
            mp.l0_local_runtime_enable = bool(use_l0_local) and not bool(local_attn_cfg)
            mp.l0_local_runtime_causal = bool(self.hier_ar_enable and self.l0_ar_enable)
            mp.local_attn_runtime_enable = bool(local_attn_cfg)
            mp.local_attn_runtime_causal_gate = bool(self.hier_ar_enable)
            mp.local_attn_runtime_group = None
            mp.local_attn_runtime_level_grid_shapes = _normalize_level_grid_shape_map(level_grid_shapes)
            mp.local_attn_runtime_spatial_metric = str(
                spatial_metric if spatial_metric is not None else getattr(self, "graph_spatial_metric", "chebyshev")
            )
            mp.local_attn_runtime_sampled = False

        args = (x_flat, edge_index, node_level, None, pos_flat, edge_attr_input)
        if not keep_grad and self.TRM:
            with torch.no_grad():
                x_next, new_edge_attr = transformer(*args)
                x_next = x_next.detach()
        else:
            if self.use_gradient_checkpointing and torch.is_grad_enabled():
                x_next, new_edge_attr = torch.utils.checkpoint.checkpoint(
                    transformer,
                    *args,
                    use_reentrant=False,
                )
            else:
                x_next, new_edge_attr = transformer(*args)
        return x_next, new_edge_attr

    def _apply_unified_refinement_true_batch_nozip(
        self,
        unified_graph,
        num_cycles: int,
        cond_vec: Optional[torch.Tensor] = None,
        per_layer_hook=None,
    ):
        # per_layer_hook(x_bnh, layer_idx, entry_idx) -> x_bnh: optional callback invoked
        # after EACH refinement layer (used by memory co-evolution to run one memory
        # message-passing round in lockstep with every native layer). Must return x in
        # the same [B, N, H] layout. None = no-op (legacy).
        from torch_geometric.data import Data

        # New forward pass => new witness-table epoch (witness_reuse == "step" caches the
        # table under this epoch so all HQD layers of THIS pass share one build; checkpoint
        # recomputes keep the epoch and therefore the identical table).
        self._witness_build_epoch = int(getattr(self, "_witness_build_epoch", 0)) + 1

        x = unified_graph.x
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, H = x.shape
        device = x.device

        base_ei = unified_graph.edge_index.to(device)
        base_nl = unified_graph.node_level.to(device)
        base_lo = getattr(unified_graph, "level_offsets", None)
        base_ar_time = getattr(unified_graph, "node_ar_time", None)
        base_branch = getattr(unified_graph, "node_branch", None)
        base_ea = getattr(unified_graph, "edge_attr", None)
        base_et = getattr(unified_graph, "edge_type", None)
        base_level_grid_shapes = getattr(unified_graph, "level_grid_shapes", None)
        if base_lo is not None:
            base_lo = torch.as_tensor(base_lo, device=device)
        if base_ar_time is not None:
            base_ar_time = base_ar_time.to(device=device, dtype=torch.long)
        if base_branch is not None:
            base_branch = torch.as_tensor(base_branch, device=device, dtype=torch.long)
            if base_branch.numel() != N:
                base_branch = None

        # Multi-level local attention: determine which levels have active local attn
        use_l0_local = bool(self.l0_local_backend != "pyg" and int(self.l0_local_window) > 0)
        local_attn_cfg = getattr(self, "local_attn_config", {})
        active_local_levels = set(local_attn_cfg.keys()) if local_attn_cfg else set()
        # Backward compat: if only legacy L0 path is active, still add level 0
        if use_l0_local and 0 not in active_local_levels:
            active_local_levels.add(0)
        use_multi_local = bool(active_local_levels)

        refine_ei = base_ei
        refine_ea = base_ea
        refine_et = base_et.to(device=device, dtype=torch.long) if base_et is not None else None
        drop_cross_level = bool(getattr(self, "drop_static_cross_level_edges", False))
        if use_multi_local or drop_cross_level:
            prune_key = (
                int(base_ei.data_ptr()),
                int(base_nl.data_ptr()),
                int(refine_et.data_ptr()) if refine_et is not None else 0,
                tuple(sorted(int(lvl) for lvl in active_local_levels)),
                bool(drop_cross_level),
                str(device),
            )
            prune_cache = getattr(self, "_local_attn_pruned_edge_cache", {})
            cached_prune = prune_cache.get(prune_key)
            if cached_prune is None:
                src_ref, dst_ref = base_ei
                src_levels = base_nl.index_select(0, src_ref)
                dst_levels = base_nl.index_select(0, dst_ref)
                prune_mask = torch.zeros(src_ref.size(0), dtype=torch.bool, device=device)
                non_self = src_ref != dst_ref
                for lvl in active_local_levels:
                    same_level = (src_levels == int(lvl)) & (dst_levels == int(lvl))
                    if int(lvl) == 0:
                        prune_mask |= same_level
                    else:
                        prune_mask |= same_level & non_self
                if drop_cross_level:
                    # cleaner-pinball mode: retire the static vertical scatter backbone; the
                    # up/down refreshes + cross-query carry vertical flow. Dynamic (HQD/zipper)
                    # edges are injected separately at attention time and are unaffected.
                    prune_mask |= src_levels != dst_levels
                keep_edges = ~prune_mask
                cached_prune = {
                    "keep_edges": keep_edges,
                    "edge_index": base_ei[:, keep_edges],
                    "edge_type": refine_et[keep_edges] if refine_et is not None and refine_et.numel() == keep_edges.numel() else refine_et,
                }
                if len(prune_cache) > 8:
                    prune_cache.clear()
                prune_cache[prune_key] = cached_prune
                self._local_attn_pruned_edge_cache = prune_cache
            keep_edges = cached_prune["keep_edges"]
            refine_ei = cached_prune["edge_index"]
            refine_et = cached_prune["edge_type"]
            if base_ea is not None:
                if base_ea.dim() == 1 and base_ea.size(0) == keep_edges.numel():
                    refine_ea = base_ea[keep_edges]
                elif base_ea.dim() == 2:
                    if base_ea.size(0) == keep_edges.numel():
                        refine_ea = base_ea[keep_edges]
                    elif base_ea.size(1) == keep_edges.numel():
                        refine_ea = base_ea[:, keep_edges]
                elif base_ea.dim() == 3 and base_ea.size(1) == keep_edges.numel():
                    refine_ea = base_ea[:, keep_edges, :]
        # HQD/zipper walk the hierarchy STRUCTURE through parent<->child edges (children
        # tables, descent). When drop_static_cross_level_edges removes those from the
        # ATTENTION edge set, hand the structural consumers the full set instead — their
        # attention contribution is ephemeral-only either way.
        struct_ei = base_ei if drop_cross_level else refine_ei

        pos_local = getattr(unified_graph, "node_pos_local", None)
        if pos_local is not None:
            pos_local = torch.as_tensor(pos_local, device=device, dtype=torch.long)
        else:
            pos_local = torch.arange(N, device=device)
            if base_lo is not None:
                pos_local = pos_local - base_lo[base_nl]

        lap_pe = getattr(unified_graph, "lap_pe_raw_cpu", None)
        if lap_pe is not None and self.lap_pe_proj is not None:
            lap_pe = self._lap_pe_raw_on_device(lap_pe, device=device, dtype=x.dtype)
            x = x + self.lap_pe_proj(lap_pe).unsqueeze(0)

        if self.share_transformers:
            transformers_to_use = [m for level_mods in self.level_transformers for m in level_mods]
        elif self.refinement_transformers is not None:
            transformers_to_use = list(self.refinement_transformers)
        else:
            transformers_to_use = []
        if len(transformers_to_use) == 0:
            # An empty refinement stack means the body never runs: x passes through
            # unchanged and only the embeddings receive gradient (the model plateaus at
            # the unigram cross-entropy floor). This is always a misconfiguration, so fail
            # loudly instead of silently training an identity model. The usual cause is
            # share_transformers=True with num_layers=[0,...] (depth lives in
            # num_refinement_layers, which only builds the dedicated refinement stack when
            # share_transformers=False).
            raise RuntimeError(
                "Pinball refinement has zero transformer layers "
                f"(share_transformers={self.share_transformers}, "
                f"num_refinement_layers={getattr(self, 'num_refinement_layers', '?')}, "
                f"level_transformers depth={[len(m) for m in self.level_transformers]}). "
                "Set qkv_sharing accordingly and put depth in num_refinement_layers, or use "
                "share_transformers=True only with num_layers>0."
            )

        x = self.pinball_work_in(x)

        l0_mask = (base_nl == 0)
        l0_idx = torch.nonzero(l0_mask, as_tuple=False).view(-1)
        apply_l0_alpha = bool(getattr(self, "l0_alpha_enable", True) and hasattr(self, "alpha"))
        x_l0_orig = x.index_select(1, l0_idx).clone() if apply_l0_alpha else None

        edge_attr_work = base_ea if bool(self.use_edge_attr) else None

        if self.hqd_every_n > 0:
            num_active_layers = len(transformers_to_use)
            if self.hqd_every_n > num_active_layers:
                logger.warning(
                    "HQD every_n=%d exceeds active refinement layers (%d); will run at most once per cycle.",
                    self.hqd_every_n, num_active_layers,
                )
                self.hqd_every_n = num_active_layers
        if use_multi_local and bool(self.use_edge_attr):
            edge_attr_work = refine_ea

        for transformer in transformers_to_use:
            mp = getattr(transformer, "message_passing", None)
            if mp is not None:
                # Legacy L0 path (backward compat)
                mp.l0_local_runtime_enable = bool(use_l0_local) and not bool(local_attn_cfg)
                mp.l0_local_runtime_causal = bool(self.hier_ar_enable and self.l0_ar_enable)
                # Multi-level path
                mp.local_attn_runtime_enable = bool(use_multi_local and local_attn_cfg)
                mp.local_attn_runtime_causal_gate = bool(self.hier_ar_enable)
                mp.local_attn_runtime_group = base_branch
                mp.local_attn_runtime_sampled = False
                if isinstance(base_level_grid_shapes, (list, tuple)):
                    level_shape_map: Dict[int, Tuple[int, int]] = {}
                    for lvl_idx, gs in enumerate(base_level_grid_shapes):
                        if gs is None or len(gs) != 2:
                            continue
                        gh = int(gs[0])
                        gw = int(gs[1])
                        if gh > 0 and gw > 0:
                            level_shape_map[int(lvl_idx)] = (gh, gw)
                    mp.local_attn_runtime_level_grid_shapes = level_shape_map
                else:
                    mp.local_attn_runtime_level_grid_shapes = {}
                mp.local_attn_runtime_spatial_metric = str(getattr(self, "graph_spatial_metric", "chebyshev"))
                mp.hqd_sparse_project_active_only = bool(getattr(self, "hqd_sparse_project_active_only", False))
                mp.hqd_tiled_apply = bool(getattr(self, "hqd_tiled_apply", False))
                mp.hqd_tile_size = int(getattr(self, "hqd_tile_size", 64))
                mp.hqd_tile_topk = int(getattr(self, "hqd_tile_topk", 64))
                mp.hqd_attn_impl = str(getattr(self, "hqd_attn_impl", "scatter"))
                mp.hqd_dense_backend = str(getattr(self, "hqd_dense_backend", "sdpa"))
                mp.hqd_profile_enable = bool(getattr(self, "hqd_debug", False))
                mp.hqd_graph_witness_enable = bool(getattr(self, "hqd_graph_witness_enable", False))
                mp.hqd_graph_witness_topk = int(getattr(self, "hqd_graph_witness_topk", 4))
                mp.hqd_packed_witness_chunk_size = int(getattr(self, "hqd_packed_witness_chunk_size", 2048))
                # Backbone witness harvest: only worth capturing when HQD will consume it,
                # and only possible when the packed cross-level lane is the one running.
                mp.hqd_backbone_witness_capture = bool(
                    getattr(self, "hierarchical_query_descent_enable", False)
                    and str(getattr(self, "hqd_packed_witness_source", "graph")).lower() == "backbone"
                    and getattr(self, "cross_level_packed", False)
                )
                mp.hqd_backbone_witness_topk = int(getattr(self, "hqd_graph_witness_topk", 4))

        if use_multi_local and not bool(getattr(self, "_l0_local_runtime_logged", False)):
            for lvl in sorted(active_local_levels):
                cfg = local_attn_cfg.get(lvl, {"window": self.l0_local_window, "causal": (self.hier_ar_enable and self.l0_ar_enable), "backend": self.l0_local_backend})
                cfg_causal = bool(cfg.get("causal", False))
                effective_causal = cfg_causal and bool(self.hier_ar_enable)
                if int(lvl) == 0:
                    effective_causal = cfg_causal and bool(self.hier_ar_enable and self.l0_ar_enable)
                logger.info(
                    "Local attn active in true_batch_nozip: level=%d backend=%s window=%d causal_cfg=%s causal_effective=%s",
                    lvl,
                    str(cfg.get("backend", self.l0_local_backend)),
                    int(cfg.get("window", self.l0_local_window)),
                    cfg_causal,
                    effective_causal,
                )
            self._l0_local_runtime_logged = True


        zip_ephemeral_enable = bool(getattr(self, "zip_enable", False)) and getattr(self, "zip_execution_mode", "edge_mutation") == "ephemeral_msg"
        zip_added_total = 0
        zip_stage_stats_total: Optional[Dict[str, int]] = None
        hqd_enable = bool(getattr(self, "hierarchical_query_descent_enable", False))
        hqd_added_total = 0
        hqd_search_added_total = 0
        hqd_reuse_added_total = 0
        hqd_reused_layers = 0
        hqd_hit_layers = 0
        hqd_stage_stats_total: Optional[Dict[str, int]] = None
        hqd_profile_stats_total: Optional[Dict[str, float]] = None
        self._last_zip_added_total = None
        self._last_zip_stage_stats = None
        self._last_zip_profile_stats = None
        self._last_zip_msg_norm_ratio_mean = None
        self._last_zip_msg_norm_ratio_max = None
        self._last_hqd_added_total = None
        self._last_hqd_selected_total = None
        self._last_hqd_stage_stats = None
        self._last_hqd_profile_stats = None
        self._last_hqd_avg_l0 = None
        self._hqd_reuse_cache = None
        hqd_cached_step = -1
        layer_step = 0
        self._xq_nom_edges = None  # per-forward; filled once layer_step reaches _xq_nom_after
        self._xq_query_gate_full = None  # per-forward; the per-query relevance gate (with grad)
        # Clear LAST forward's gate hooks now, UNCONDITIONALLY (not per-layer in the loop:
        # gradient checkpointing re-reads the attr during backward, so it must outlive the
        # forward; and not guarded on gate_enable: a runtime toggle-off would otherwise
        # leave stale armed gates whose graph is freed -> backward crash / wrong grads).
        # Each fed layer's mp re-arms below; a layer fed twice in one forward (weight-tied
        # multi-cycle schedules) would keep only the last round's gate.
        # Packed cross-level local attention: build (or fetch the skeleton-cached) spec once
        # per forward and inject it on every layer's message passing; None when off/ineligible
        # clears any stale spec from a previous forward (specs are shape-guarded layer-side).
        _local_pack_spec_fw = None
        if (
            bool(getattr(self, "local_pack_cross_level", False))
            and base_lo is not None
            and base_ar_time is not None
            and str(getattr(self, "graph_geometry_mode", "sequence")).lower() == "sequence"
        ):
            _local_pack_spec_fw = self._local_pack_build_spec(base_lo, base_nl, base_ar_time)
        self._last_local_pack_spec_stats = (
            {
                "len": int(_local_pack_spec_fw["num_nodes"]),
                "queries": int(_local_pack_spec_fw["query_sel"].numel()),
                "window": int(_local_pack_spec_fw["window"]),
                "lane_len": int(_local_pack_spec_fw["lane_perm"].numel())
                if "lane_perm" in _local_pack_spec_fw else 0,
                "lane_window": int(_local_pack_spec_fw.get("lane_window", 0)),
            }
            if _local_pack_spec_fw is not None
            else None
        )
        for _t in getattr(self, "refinement_transformers", []) or []:
            _mp = getattr(_t, "message_passing", None)
            if _mp is None:
                continue
            if getattr(_mp, "_hqd_out_query_gate", None) is not None:
                _mp._hqd_out_query_gate = None
            _mp._local_pack_spec = _local_pack_spec_fw
        _xq_nom_layers_used = 0
        _xq_nom_calls = 0
        _xq_last_nom_step = -1
        _xq_raw_after = getattr(self, "xq_nominate_after_layer", "auto")
        _xq_nom_after = (
            max(1, len(transformers_to_use) // 2)
            if str(_xq_raw_after).lower() == "auto"
            else max(1, int(_xq_raw_after))
        )
        _xq_every = int(getattr(self, "xq_nominate_every", 0))
        multirate_active = bool(
            getattr(self, "pinball_multirate_enable", False)
            and (
                getattr(self, "pinball_upper_refiner", None) is not None
                or getattr(self, "pinball_upper_cross_refiner", None) is not None
                or len(getattr(self, "pinball_cross_query_refiners", [])) > 0
                or getattr(self, "pinball_top_refiner", None) is not None
            )
        )
        pinball_cycle_active = bool(getattr(self, "pinball_level_cycle_enable", False)) and not multirate_active
        all_pinball_levels = list(range(int(getattr(self, "num_hier_levels", 0))))
        schedule_entries = self._pinball_layer_schedule(int(num_cycles), len(transformers_to_use))
        if multirate_active:
            schedule_entries = [
                (cycle, layer_idx, None)
                for cycle in range(max(0, int(num_cycles)))
                for layer_idx in range(len(transformers_to_use))
            ]
        self._last_pinball_active_levels = []
        self._last_pinball_multirate_stats = None
        if pinball_cycle_active:
            active_hist: Dict[str, int] = {}
            for _, _, scheduled_active in schedule_entries:
                active_key = "all" if scheduled_active is None else ",".join(str(int(level)) for level in scheduled_active)
                active_hist[active_key] = int(active_hist.get(active_key, 0) + 1)
            level_node_counts = {
                int(level): int((base_nl == int(level)).sum().item())
                for level in all_pinball_levels
            }
            self._last_pinball_schedule_stats = {
                "total_layer_calls": int(len(schedule_entries)),
                "active_hist": dict(active_hist),
                "level_node_counts": dict(level_node_counts),
                "cycle_mode": str(getattr(self, "pinball_level_cycle_mode", "extra_cycles")),
                "active_compute": str(getattr(self, "pinball_graph_active_compute", "all")),
            }
            if not bool(getattr(self, "_pinball_schedule_stats_logged", False)):
                logger.info(
                    "Pinball level schedule: calls=%d hist=%s nodes=%s mode=%s active_compute=%s",
                    int(len(schedule_entries)),
                    dict(active_hist),
                    dict(level_node_counts),
                    str(getattr(self, "pinball_level_cycle_mode", "extra_cycles")),
                    str(getattr(self, "pinball_graph_active_compute", "all")),
                )
                self._pinball_schedule_stats_logged = True
        else:
            self._last_pinball_schedule_stats = None

        def _copy_active_nodes(x_base: torch.Tensor, x_update: torch.Tensor, active_idx: Optional[torch.Tensor]) -> torch.Tensor:
            if active_idx is None or active_idx.numel() == 0:
                return x_update
            x_next = x_base.clone()
            x_next.index_copy_(1, active_idx, x_update.index_select(1, active_idx))
            return x_next

        def _accum_hqd_profile(profile: Optional[Dict[str, float]]) -> None:
            nonlocal hqd_profile_stats_total
            if not profile:
                return
            if hqd_profile_stats_total is None:
                hqd_profile_stats_total = {str(k): float(v) for k, v in profile.items()}
            else:
                for key, value in profile.items():
                    hqd_profile_stats_total[str(key)] = float(hqd_profile_stats_total.get(str(key), 0.0) + float(value))

        def _accum_hqd_stage(stage_stats: Optional[Dict[str, int]]) -> None:
            nonlocal hqd_stage_stats_total
            if not stage_stats:
                return
            if hqd_stage_stats_total is None:
                hqd_stage_stats_total = {k: int(v) for k, v in stage_stats.items()}
            else:
                for key, value in stage_stats.items():
                    hqd_stage_stats_total[key] = int(hqd_stage_stats_total.get(key, 0) + int(value))

        # ---- multirate abstract block (called per cycle) ----
        _mr_call_count: int = 0
        _mr_touched_all: List[int] = []
        _mr_graph_ms: float = 0.0
        _mr_cross_ms: float = 0.0
        _mr_upper_ms: float = 0.0
        _mr_top_ms: float = 0.0
        _mr_generic_pairs: int = 0
        _mr_generic_selected: int = 0
        _mr_generic_updated: int = 0
        _mr_cross_l2_selected: int = 0
        _mr_cross_l2_updated: int = 0
        _mr_upper_steps: int = 0
        _mr_cross_steps: int = 0
        _mr_top_steps: int = 0
        _mr_runtime_logged: bool = False

        def _run_multirate_block(x_in: torch.Tensor, query_keep_by_level: Optional[Dict[int, Tuple[torch.Tensor, int]]] = None) -> torch.Tensor:
            nonlocal _mr_call_count, _mr_touched_all, _mr_graph_ms, _mr_cross_ms, _mr_upper_ms, _mr_top_ms
            nonlocal _mr_generic_pairs, _mr_generic_selected, _mr_generic_updated, _mr_runtime_logged
            nonlocal _mr_cross_l2_selected, _mr_cross_l2_updated
            nonlocal _mr_upper_steps, _mr_cross_steps, _mr_top_steps
            _mr_call_count = int(_mr_call_count) + 1
            call_count = int(getattr(self, "_pinball_multirate_call_count", 0)) + 1
            self._pinball_multirate_call_count = call_count
            upper_should_run = getattr(self, "pinball_upper_refiner", None) is not None and (call_count % int(self.pinball_upper_refine_every) == 0)
            cross_should_run = getattr(self, "pinball_upper_cross_refiner", None) is not None and (call_count % int(self.pinball_upper_refine_every) == 0)
            generic_cross_should_run = len(getattr(self, "pinball_cross_query_refiners", [])) > 0 and (call_count % int(self.pinball_upper_refine_every) == 0)
            top_should_run = getattr(self, "pinball_top_refiner", None) is not None and (call_count % int(self.pinball_top_refine_every) == 0)
            touched: List[int] = []
            x_out = x_in
            # KV-cached decode: cached pre-tail coarse rows enter as frozen context —
            # per-level causal prefixes for the within-level refiners, extra memory rows
            # for the cross-attention refiners. None (the default) is a strict no-op.
            gen_prefix = getattr(self, "_gen_mr_level_prefix", None)
            gen_extra_mem = getattr(self, "_gen_mr_extra_memory", None)
            if upper_should_run:
                t0 = time.monotonic()
                x_out = self.pinball_upper_refiner(x_out, base_nl, steps=self.pinball_upper_refine_steps, extra_prefix=gen_prefix)
                _mr_upper_ms = float(_mr_upper_ms + (time.monotonic() - t0) * 1000.0)
                _mr_upper_steps = int(_mr_upper_steps + int(self.pinball_upper_refine_steps))
                touched.extend([2, 3])
            if cross_should_run:
                t0 = time.monotonic()
                x_out, cross_stats = self.pinball_upper_cross_refiner(
                    x_out, base_nl, node_ar_time=base_ar_time,
                    extra_memory=None if gen_extra_mem is None else gen_extra_mem.get(int(self.pinball_upper_cross_refiner.memory_level)),
                )
                _mr_cross_ms = float(_mr_cross_ms + (time.monotonic() - t0) * 1000.0)
                _mr_cross_steps = int(_mr_cross_steps + int(self.pinball_upper_cross_attn_steps))
                _mr_cross_l2_selected = int(_mr_cross_l2_selected + int(cross_stats.get("l2_selected", 0)))
                _mr_cross_l2_updated = int(_mr_cross_l2_updated + int(cross_stats.get("l2_updated", 0)))
                touched.append(3)
                if int(cross_stats.get("l2_updated", 0)) > 0:
                    touched.append(2)
            if generic_cross_should_run:
                for refiner in self.pinball_cross_query_refiners:
                    t0 = time.monotonic()
                    q_keep = None if query_keep_by_level is None else query_keep_by_level.get(int(refiner.query_level))
                    x_out, pair_stats = refiner(
                        x_out, base_nl, node_ar_time=base_ar_time, query_keep=q_keep,
                        extra_memory=None if gen_extra_mem is None else gen_extra_mem.get(int(refiner.memory_level)),
                    )
                    _mr_cross_ms = float(_mr_cross_ms + (time.monotonic() - t0) * 1000.0)
                    _mr_generic_pairs = int(_mr_generic_pairs + 1)
                    _mr_generic_selected = int(_mr_generic_selected + int(pair_stats.get("memory_selected", 0)))
                    _mr_generic_updated = int(_mr_generic_updated + int(pair_stats.get("memory_updated", 0)))
                    touched.append(int(pair_stats.get("query_level", -1)))
                    if int(pair_stats.get("memory_updated", 0)) > 0:
                        touched.append(int(pair_stats.get("memory_level", -1)))
            if top_should_run:
                t0 = time.monotonic()
                x_out = self.pinball_top_refiner(x_out, base_nl, steps=self.pinball_top_refine_steps, extra_prefix=gen_prefix)
                _mr_top_ms = float(_mr_top_ms + (time.monotonic() - t0) * 1000.0)
                _mr_top_steps = int(_mr_top_steps + int(self.pinball_top_refine_steps))
                touched.append(3)
            touched = sorted(set(t for t in touched if t >= 0))
            _mr_touched_all.extend(touched)
            if bool(getattr(self, "pinball_multirate_debug", False)) and not _mr_runtime_logged:
                logger.info(
                    "Pinball multirate cycle: upper_ms=%.2f cross_ms=%.2f top_ms=%.2f pairs=%d touched=%s",
                    float(_mr_upper_ms),
                    float(_mr_cross_ms),
                    float(_mr_top_ms),
                    int(_mr_generic_pairs),
                    str(touched),
                )
                _mr_runtime_logged = True
            return x_out

        mr_schedule = str(getattr(self, "pinball_multirate_schedule", "after_full_stack")).lower().replace("-", "_")
        mr_repeats = max(1, int(getattr(self, "pinball_multirate_midpoint_repeats", 1)))
        raw_midpoint_layer = getattr(self, "pinball_multirate_midpoint_layer", "auto")
        if str(raw_midpoint_layer).lower() == "auto":
            mr_midpoint_after = max(1, len(transformers_to_use) // 2)
        else:
            mr_midpoint_after = max(1, int(raw_midpoint_layer))
        mr_midpoint_after = min(max(1, len(transformers_to_use)), int(mr_midpoint_after))
        graph_refine_t0 = time.monotonic()
        graph_segment_t0 = graph_refine_t0
        midpoint_ran = False

        mr_adaptive = bool(getattr(self, "pinball_multirate_adaptive", False))
        mr_energy_tau = float(getattr(self, "pinball_multirate_energy_tau", 0.05))
        mr_max_cycles = max(1, int(getattr(self, "pinball_multirate_max_cycles", 4)))
        _mr_adaptive_cycles: int = 0
        _mr_adaptive_energy: List[float] = []
        _mr_adaptive_frozen_frac: float = 0.0

        def _run_multirate_repeats(x_in: torch.Tensor, repeats: int) -> torch.Tensor:
            nonlocal _mr_graph_ms, graph_segment_t0
            nonlocal _mr_adaptive_cycles, _mr_adaptive_frozen_frac
            _mr_graph_ms = float(_mr_graph_ms + (time.monotonic() - graph_segment_t0) * 1000.0)
            x_out = x_in
            # KV-cached decode: settled context rows are pre-frozen for the whole episode —
            # they hold their (cache-injected) episode-entry values while only frontier rows
            # iterate, and the active-rows gather skips computing them from cycle 1. The
            # post-episode divergence this causes on context rows is repaired by the
            # _gen_mr_hook overwrite below. None (the default) is a strict no-op.
            gen_prefrozen = getattr(self, "_gen_mr_prefrozen", None)  # [B, N] bool
            gen_prefrozen_keep = getattr(self, "_gen_mr_prefrozen_keep", None)  # {level: (mask, n_act)}
            # Per-CYCLE hook: settled rows evolve within the episode in the true forward, so
            # boundary injection alone is not enough — the decode driver captures every
            # cycle's state on the refill forward and re-injects settled rows after every
            # cycle here (exact by causality). Also fires in capture mode. None = no-op.
            gen_cycle_hook = getattr(self, "_gen_mr_cycle_hook", None)
            if not mr_adaptive:
                if gen_prefrozen is None and gen_cycle_hook is None:
                    for _ in range(max(1, int(repeats))):
                        x_out = _run_multirate_block(x_out)
                else:
                    for rep_i in range(max(1, int(repeats))):
                        x_new = _run_multirate_block(x_out, query_keep_by_level=gen_prefrozen_keep)
                        if gen_prefrozen is not None:
                            x_new = torch.where(gen_prefrozen.unsqueeze(-1), x_out, x_new)
                        if gen_cycle_hook is not None:
                            x_new = gen_cycle_hook(x_new, int(rep_i))
                        x_out = x_new
            else:
                # Energy-gated settling: iterate the block until every node's relative
                # update falls below tau, up to max_cycles. Halting must be PER NODE on
                # the node's own energy — any global "batch is settled" halt would let
                # future windows' settle time modulate how many updates past rows
                # receive (an AR leak through the amount of compute). Frozen rows keep
                # their settled values, so breaking once all rows are frozen is a
                # bit-exact no-op. The block writes L0 too (0:q cross-query pairs), so
                # the freeze mask covers ALL rows, not just coarse levels.
                call0 = int(getattr(self, "_pinball_multirate_call_count", 0))
                mr_active_rows = bool(getattr(self, "pinball_multirate_active_rows", True))
                frozen: Optional[torch.Tensor] = gen_prefrozen  # [B, N] bool
                active_keep: Optional[Dict[int, Tuple[torch.Tensor, int]]] = gen_prefrozen_keep
                keep_level_idx: Optional[Dict[int, torch.Tensor]] = None
                if mr_active_rows:
                    keep_levels = sorted({int(r.query_level) for r in (getattr(self, "pinball_cross_query_refiners", []) or [])})
                    keep_level_idx = {l: torch.nonzero(base_nl == l, as_tuple=False).view(-1) for l in keep_levels}
                for _cyc_i in range(mr_max_cycles):
                    x_new = _run_multirate_block(x_out, query_keep_by_level=active_keep)
                    if frozen is not None:
                        x_new = torch.where(frozen.unsqueeze(-1), x_out, x_new)
                    if gen_cycle_hook is not None:
                        # decode: overwrite settled rows with their true (captured) state for
                        # this cycle BEFORE the energy step, so the freeze dynamics of the
                        # frontier rows see exactly what the full forward's would.
                        x_new = gen_cycle_hook(x_new, int(_cyc_i))
                    with torch.no_grad():
                        delta = (x_new - x_out).float().norm(dim=-1)
                        ref = x_out.float().norm(dim=-1).clamp_min(1e-6)
                        energy = delta / ref
                        settled = energy <= mr_energy_tau
                        frozen = settled if frozen is None else (frozen | settled)
                        level_masks: Dict[int, torch.Tensor] = {}
                        if keep_level_idx:
                            keep_all = ~frozen
                            level_masks = {l: keep_all.index_select(1, idx) for l, idx in keep_level_idx.items()}
                        # ONE host sync per cycle: mean energy, frozen frac (1.0 = break),
                        # and each level's max active-row count for the gather path.
                        packed = torch.cat(
                            [energy.mean().view(1), frozen.float().mean().view(1)]
                            + [mk.sum(dim=1).max().to(torch.float32).view(1) for mk in level_masks.values()]
                        ).tolist()
                    _mr_adaptive_energy.append(float(packed[0]))
                    _mr_adaptive_frozen_frac = float(packed[1])
                    x_out = x_new
                    _mr_adaptive_cycles = int(_mr_adaptive_cycles) + 1
                    if packed[1] >= 1.0:
                        break
                    # Active-rows: from the next cycle on, the cross-query refiners only
                    # compute unfrozen query rows (exact — queries are independent there;
                    # frozen rows would be discarded by the where() above anyway). Only
                    # gather when a solid majority is frozen; below that the gather
                    # overhead outweighs the saved compute.
                    active_keep = None
                    if level_masks:
                        gathered = {}
                        for j, (l, mk) in enumerate(level_masks.items()):
                            lvl_n_act = int(packed[2 + j])
                            if lvl_n_act <= int(0.75 * mk.size(1)):
                                gathered[l] = (mk, lvl_n_act)
                        active_keep = gathered or None
                # The persistent call counter gates the sub-refiner `_every` schedule;
                # pin it to the max so a content-dependent early break cannot shift
                # which refiners run in later invocations (schedule stays deterministic).
                self._pinball_multirate_call_count = call0 + mr_max_cycles
            # KV-cached decode: capture (prompt) or re-inject (decode) the POST-episode
            # state — the layer-boundary hook alone can't see it, and the next layer
            # reads context rows at exactly this state. No-op when unset.
            gen_mr_hook = getattr(self, "_gen_mr_hook", None)
            if gen_mr_hook is not None:
                x_out = gen_mr_hook(x_out)
            graph_segment_t0 = time.monotonic()
            return x_out

        for entry_idx, (cycle, layer_idx, scheduled_active_levels) in enumerate(schedule_entries):
                transformer = transformers_to_use[int(layer_idx)]
                active_levels = all_pinball_levels if scheduled_active_levels is None else list(scheduled_active_levels)
                active_node_mask_bnh = None
                active_node_idx = None
                if pinball_cycle_active:
                    self._last_pinball_active_levels.append(list(active_levels))
                    active_node_mask = self._pinball_level_mask(base_nl, active_levels)
                    if not bool(active_node_mask.any()):
                        layer_step += 1
                        continue
                    active_node_idx = torch.nonzero(active_node_mask, as_tuple=False).view(-1)
                    active_node_mask_bnh = active_node_mask.view(1, N, 1)
                x_before_layer = x
                x = self._apply_refinement_conditioning_bnh(x, cond_vec)

                run_zip_step = zip_ephemeral_enable and (
                    self.zip_granularity == "per_layer"
                    or (self.zip_granularity == "per_cycle" and layer_idx == 0)
                )
                _hqd_off = int(getattr(self, "hqd_every_n_offset", 0))
                run_hqd_step = hqd_enable and (
                    (self.hqd_every_n > 0 and layer_idx >= _hqd_off and (layer_idx - _hqd_off) % self.hqd_every_n == 0)
                    or (self.hqd_every_n <= 0 and self.hqd_granularity == "per_layer")
                    or (self.hqd_every_n <= 0 and self.hqd_granularity == "per_cycle" and layer_idx == 0)
                )
                select_hqd_inside_mp = bool(run_hqd_step and getattr(self, "hqd_select_inside_message_passing", False))
                run_hqd_reuse_step = False
                if hqd_enable and (not run_hqd_step) and bool(self.hqd_reuse_previous):
                    if (
                        self._hqd_reuse_cache is not None
                        and "b_idx" in self._hqd_reuse_cache
                        and "src_idx" in self._hqd_reuse_cache
                        and "dst_idx" in self._hqd_reuse_cache
                    ):
                        cache_age = layer_step - int(hqd_cached_step)
                        within_age = self.hqd_reuse_max_age <= 0 or cache_age <= int(self.hqd_reuse_max_age)
                        b_cached = self._hqd_reuse_cache["b_idx"]
                        src_cached = self._hqd_reuse_cache["src_idx"]
                        dst_cached = self._hqd_reuse_cache["dst_idx"]
                        cache_batch_size = int(self._hqd_reuse_cache.get("B", x.size(0)))
                        cache_num_nodes = int(self._hqd_reuse_cache.get("N", x.size(1)))
                        cache_device = str(self._hqd_reuse_cache.get("device", x.device))
                        run_hqd_reuse_step = bool(
                            within_age
                            and b_cached.numel() > 0
                            and src_cached.numel() == b_cached.numel()
                            and dst_cached.numel() == b_cached.numel()
                            and cache_batch_size == int(x.size(0))
                            and cache_num_nodes == int(x.size(1))
                            and cache_device == str(x.device)
                        )

                shared_zip_qk_cache = None
                shared_hqd_qk_cache = None
                shared_x_in = None

                if (run_zip_step or (run_hqd_step and not select_hqd_inside_mp)) and hasattr(transformer, "message_passing"):
                    mp = transformer.message_passing
                    shared_x_in = transformer.norm1(x) if hasattr(transformer, "norm1") else x
                    if hasattr(mp, "q_proj") and hasattr(mp, "k_proj"):
                        with torch.no_grad():
                            q_all = mp.q_proj(shared_x_in).view(x.size(0), x.size(1), mp.num_heads, mp.head_dim)
                            k_all = mp.k_proj(shared_x_in).view(x.size(0), x.size(1), mp.num_heads, mp.head_dim)
                            level_emb = mp.level_embedding(base_nl) if hasattr(mp, "level_embedding") else None
                        shared_hqd_qk_cache = (q_all, k_all)
                        if level_emb is not None:
                            shared_zip_qk_cache = (q_all, k_all, level_emb)

                # ~~~~ HQD edge selection (before transformer, so attention fuses inside) ~~~~
                hqd_b_idx = None
                hqd_src_idx = None
                hqd_dst_idx = None
                mp_runtime = getattr(transformer, "message_passing", None)
                if mp_runtime is not None:
                    mp_runtime.hqd_runtime_selector = None
                if select_hqd_inside_mp and mp_runtime is not None:
                    def _runtime_hqd_selector(
                        q_runtime: torch.Tensor,
                        k_runtime: torch.Tensor,
                        _transformer=transformer,
                        _x=x,
                        _refine_ei=struct_ei,
                        _base_nl=base_nl,
                        _base_ar_time=base_ar_time,
                    ):
                        if bool(getattr(self, "hqd_packed_witness_read", False)):
                            if str(getattr(self, "hqd_packed_witness_source", "graph")) == "recompute":
                                packed = self._hierarchical_query_descent_recomputed_packed_witness_batched(
                                    transformer=_transformer,
                                    x_bnh=_x,
                                    base_edge_index=_refine_ei,
                                    node_level=_base_nl,
                                    node_ar_time=_base_ar_time,
                                    q_all=q_runtime,
                                    k_all=k_runtime,
                                )
                            else:
                                packed = self._hierarchical_query_descent_packed_witness_batched(
                                    transformer=_transformer,
                                    x_bnh=_x,
                                    base_edge_index=_refine_ei,
                                    node_level=_base_nl,
                                    node_ar_time=_base_ar_time,
                                    q_all=q_runtime,
                                    k_all=k_runtime,
                                )
                            if packed is not None:
                                return packed
                            return {
                                "edges": None,
                                "packed_l0": None,
                                "added_total": 0,
                                "stage_stats": {"packed_witness_l0": 0},
                                "profile_stats": None,
                            }
                        b_sel, src_sel, dst_sel, added, stage = self._hierarchical_query_descent_ephemeral_batched(
                            transformer=_transformer,
                            x_bnh=_x,
                            base_edge_index=_refine_ei,
                            node_level=_base_nl,
                            node_ar_time=_base_ar_time,
                            x_in=None,
                            _qk_cache=(q_runtime, k_runtime),
                        )
                        profile = getattr(self, "_last_hqd_profile_stats", None)
                        return b_sel, src_sel, dst_sel, int(added), stage, profile
                    mp_runtime.hqd_runtime_selector = _runtime_hqd_selector
                elif run_hqd_step:
                    b_sel_idx, src_sel_idx, dst_sel_idx, added_hqd, hqd_stage_stats = self._hierarchical_query_descent_ephemeral_batched(
                        transformer=transformer,
                        x_bnh=x,
                        base_edge_index=struct_ei,
                        node_level=base_nl,
                        node_ar_time=base_ar_time,
                        x_in=shared_x_in,
                        _qk_cache=shared_hqd_qk_cache,
                    )
                    _accum_hqd_profile(getattr(self, "_last_hqd_profile_stats", None))
                    if b_sel_idx.numel() > 0:
                        hqd_b_idx, hqd_src_idx, hqd_dst_idx = b_sel_idx, src_sel_idx, dst_sel_idx
                    hqd_added_total += int(added_hqd)
                    hqd_hit_layers += 1
                    hqd_search_added_total += int(added_hqd)
                    hqd_cached_step = int(layer_step)
                    _accum_hqd_stage(hqd_stage_stats)
                elif run_hqd_reuse_step and self._hqd_reuse_cache is not None:
                    b_cached = self._hqd_reuse_cache["b_idx"].to(device=x.device, dtype=torch.long)
                    src_cached = self._hqd_reuse_cache["src_idx"].to(device=x.device, dtype=torch.long)
                    dst_cached = self._hqd_reuse_cache["dst_idx"].to(device=x.device, dtype=torch.long)
                    if b_cached.numel() > 0:
                        hqd_b_idx, hqd_src_idx, hqd_dst_idx = b_cached, src_cached, dst_cached
                        hqd_added_total += int(b_cached.numel())
                        hqd_hit_layers += 1
                        hqd_reuse_added_total += int(b_cached.numel())
                        hqd_reused_layers += 1

                # HQD v2: cross-query-guided nominations (computed once at the multirate
                # midpoint) feed subsequent layers through the same ephemeral-edge path, up to
                # the xq_nominate_max_layers budget (the sparse path costs per layer).
                # v1 HQD, when enabled, takes precedence on layers where it selected.
                _xq_gate_cur = None
                if hqd_b_idx is None and getattr(self, "_xq_nom_edges", None) is not None:
                    _xq_budget = int(getattr(self, "xq_nominate_max_layers", 0))
                    if _xq_budget <= 0 or _xq_nom_layers_used < _xq_budget:
                        hqd_b_idx, hqd_src_idx, hqd_dst_idx = self._xq_nom_edges
                        _xq_nom_layers_used += 1
                        # per-query relevance gate: carried as an EXPLICIT tensor argument
                        # through the checkpoint boundary (see _forward_with_hqd — the inner
                        # reentrant checkpoint double-backwards through attr-captured live
                        # graphs) and snapshotted per layer for re-nomination correctness.
                        _xq_gate_cur = self._xq_query_gate_full

                # ~~~~ Transformer step (HQD attention fused in, if edges provided) ~~~~
                self._hqd_inside_mp_active = bool(select_hqd_inside_mp)
                try:
                    active_compute_levels = None
                    if (
                        pinball_cycle_active
                        and str(getattr(self, "pinball_graph_active_compute", "all")) == "destination_only"
                        and len(active_levels) < len(all_pinball_levels)
                    ):
                        active_compute_levels = active_levels
                    _use_ckpt = (
                        bool(getattr(self, "use_gradient_checkpointing", False))
                        and self.training
                        and torch.is_grad_enabled()
                    )
                    if _use_ckpt and not getattr(self, "_flash_backend_warmed", False):
                        # Warm the flash-backend lru_cache HERE (grad-enabled, outside the
                        # checkpoint). The picker's one-time smoke test does a .backward(); if its
                        # first call instead lands inside the checkpoint's recompute (run under
                        # no_grad), that nested backward fails and the cache is poisoned to a
                        # non-flash backend permanently -> "flash unavailable" for level 0.
                        try:
                            from .layers.hierarchical_message_passing import pick_attention_backend
                            pick_attention_backend(getattr(x, "device", None))
                        except Exception:
                            pass
                        self._flash_backend_warmed = True
                    if _use_ckpt:
                        # Checkpoint the per-layer refinement step: free its internal
                        # activations (attention/FFN) and recompute them in backward.
                        # Loop-varying args are bound as defaults so the backward
                        # recompute uses THIS layer's snapshot (edge_attr_work is
                        # reassigned later in the loop).
                        def _ckpt_refine(
                            x_in,
                            _t=transformer, _ei=refine_ei, _nl=base_nl, _lo=base_lo,
                            _pl=pos_local, _ea=edge_attr_work, _et=refine_et,
                            _bi=hqd_b_idx, _si=hqd_src_idx, _di=hqd_dst_idx,
                            _al=active_compute_levels, _gate=_xq_gate_cur,
                        ):
                            return self._refine_step_true_batch_native(
                                transformer=_t, x_bnh=x_in, edge_index=_ei,
                                node_level=_nl, level_offsets=_lo, pos_local=_pl,
                                edge_attr_work=_ea, edge_type_work=_et,
                                hqd_b_idx=_bi, hqd_src_idx=_si, hqd_dst_idx=_di,
                                active_levels=_al, xq_gate=_gate,
                            )
                        x, new_edge_attr = torch.utils.checkpoint.checkpoint(
                            _ckpt_refine, x, use_reentrant=False,
                        )
                    else:
                        x, new_edge_attr = self._refine_step_true_batch_native(
                            transformer=transformer,
                            x_bnh=x,
                            edge_index=refine_ei,
                            node_level=base_nl,
                            level_offsets=base_lo,
                            pos_local=pos_local,
                            edge_attr_work=edge_attr_work,
                            edge_type_work=refine_et,
                            hqd_b_idx=hqd_b_idx,
                            hqd_src_idx=hqd_src_idx,
                            hqd_dst_idx=hqd_dst_idx,
                            active_levels=active_compute_levels,
                            xq_gate=_xq_gate_cur,
                        )
                finally:
                    self._hqd_inside_mp_active = False

                if pinball_cycle_active:
                    x = _copy_active_nodes(x_before_layer, x, active_node_idx)
                # Harvest the packed lane's backbone witness capture (detached ids/logits),
                # stamped with this forward's witness epoch so later HQD layers can build
                # their table from it and a stale capture (previous step) is never used.
                # Levels 1..3 must all be present (a partial-active-levels layer may not
                # produce every level's block).
                mp_h = getattr(transformer, "message_passing", None)
                if mp_h is not None and bool(getattr(mp_h, "hqd_backbone_witness_capture", False)):
                    _cap_ids = getattr(mp_h, "_last_backbone_witness_ids", None)
                    if isinstance(_cap_ids, dict) and all(l in _cap_ids for l in (1, 2, 3)):
                        self._backbone_witness_latest = (
                            int(getattr(self, "_witness_build_epoch", 0)),
                            _cap_ids,
                            getattr(mp_h, "_last_backbone_witness_scores", None),
                        )
                if bool(getattr(self, "hqd_debug", False)) and hqd_b_idx is not None and hasattr(transformer, "message_passing"):
                    apply_ms = getattr(transformer.message_passing, "_last_hqd_apply_ms", None)
                    if apply_ms is not None:
                        _accum_hqd_profile({"apply_ms": float(apply_ms)})
                if select_hqd_inside_mp and mp_runtime is not None:
                    added_hqd = getattr(mp_runtime, "_last_hqd_runtime_added_total", None)
                    if added_hqd is not None:
                        hqd_added_total += int(added_hqd)
                        hqd_hit_layers += 1
                        hqd_search_added_total += int(added_hqd)
                        hqd_cached_step = int(layer_step)
                    _accum_hqd_stage(getattr(mp_runtime, "_last_hqd_runtime_stage_stats", None))
                    _accum_hqd_profile(getattr(mp_runtime, "_last_hqd_runtime_profile_stats", None))
                    if bool(getattr(self, "hqd_debug", False)):
                        apply_ms = getattr(mp_runtime, "_last_hqd_apply_ms", None)
                        if apply_ms is not None:
                            _accum_hqd_profile({"apply_ms": float(apply_ms)})
                    mp_runtime.hqd_runtime_selector = None

                if new_edge_attr is not None:
                    edge_attr_work = new_edge_attr

                # ~~~~ Zipper (unchanged, runs after transformer) ~~~~
                if run_zip_step:
                    x_before_zip = x
                    x, added, stage_stats = self._zipper_apply_staged_ephemeral_batched(
                        transformer=transformer,
                        x_bnh=x,
                        base_edge_index=struct_ei,
                        node_level=base_nl,
                        node_ar_time=base_ar_time,
                        x_in=shared_x_in,
                        _qk_cache=shared_zip_qk_cache,
                    )
                    zip_added_total += int(added)
                    if zip_stage_stats_total is None:
                        zip_stage_stats_total = {k: int(v) for k, v in stage_stats.items()}
                    else:
                        for key, value in stage_stats.items():
                            zip_stage_stats_total[key] = int(zip_stage_stats_total.get(key, 0) + int(value))
                    if pinball_cycle_active:
                        x = _copy_active_nodes(x_before_zip, x, active_node_idx)

                if apply_l0_alpha and ((not pinball_cycle_active) or 0 in active_levels):
                    a = self.alpha
                    x_l0 = x.index_select(1, l0_idx)
                    x.index_copy_(1, l0_idx, a * x_l0 + (1.0 - a) * x_l0_orig)

                if pinball_cycle_active:
                    x_normed = self.pinball_refinement_norm(x.index_select(1, active_node_idx))
                    x = x.clone()
                    x.index_copy_(1, active_node_idx, x_normed.to(dtype=x.dtype))
                else:
                    x = self.pinball_refinement_norm(x)
                layer_step += 1

                # Per-layer upward refresh: ascend this layer's L0 state into the coarse levels
                # NOW, so the next layer's downward reads see a current abstraction.
                if (
                    getattr(self, "hier_upward_refresh", False)
                    and base_lo is not None
                    and (layer_step % self.hier_upward_refresh_every) == 0
                ):
                    x = self._apply_upward_refresh(x, base_lo)

                # Per-layer downward refresh: broadcast each coarser level's freshest closed
                # summary back down (dense gather replacement for the bridges scatter edges).
                if (
                    getattr(self, "hier_downward_refresh", False)
                    and base_lo is not None
                    and base_ar_time is not None
                    and (layer_step % self.hier_downward_refresh_every) == 0
                ):
                    x = self._apply_downward_refresh(x, base_lo, base_ar_time)

                # Per-layer co-evolution hook: run one memory round in lockstep with this
                # native layer (memory graph evolves at the same depth as the native graph).
                if per_layer_hook is not None:
                    x = per_layer_hook(x, int(layer_idx), int(entry_idx))

                if multirate_active:
                    is_cycle_end = bool(
                        entry_idx == len(schedule_entries) - 1
                        or int(schedule_entries[entry_idx + 1][0]) != int(cycle)
                    )
                    is_final_cycle = bool(entry_idx == len(schedule_entries) - 1)
                    layers_completed_in_cycle = int(layer_idx) + 1
                    if mr_schedule == "midpoint" and (not midpoint_ran) and layers_completed_in_cycle >= int(mr_midpoint_after):
                        x = _run_multirate_repeats(x, mr_repeats)
                        midpoint_ran = True
                    elif mr_schedule == "midpoint_each_cycle" and layers_completed_in_cycle == int(mr_midpoint_after):
                        x = _run_multirate_repeats(x, mr_repeats)
                    elif mr_schedule == "after_full_cycle" and is_cycle_end:
                        if (not bool(getattr(self, "pinball_multirate_skip_after_last_cycle", True))) or (not is_final_cycle):
                            x = _run_multirate_repeats(x, 1)

                # HQD v2 nomination: first fires after xq_nominate_after_layer layers ("auto"
                # = mid-stack), on the post-layer/post-multirate x — independent of the
                # multirate schedule. With xq_nominate_every > 0 it then RE-SELECTS every k
                # layers with the evolved queries (iterative / multi-hop retrieval), resetting
                # the per-interval fed-layer budget. Later layers consume the edges (see the
                # injection hook).
                if (
                    getattr(self, "xq_nominate_enable", False)
                    and base_lo is not None
                    and base_ar_time is not None
                    and layer_step >= _xq_nom_after
                    # no consumer left after the final layer — don't pay a wasted selection
                    # (with after=6, every=2 on 12 layers the round at step 12 fed nothing)
                    and layer_step < len(schedule_entries)
                    and (
                        _xq_nom_calls == 0
                        or (_xq_every > 0 and (layer_step - _xq_last_nom_step) >= _xq_every)
                    )
                ):
                    self._xq_nom_edges = self._cross_query_nominate(
                        x, base_lo, base_ar_time, block=transformer)
                    _xq_nom_calls += 1
                    _xq_last_nom_step = layer_step
                    _xq_nom_layers_used = 0
                    self._last_xq_nom_calls = _xq_nom_calls

        if multirate_active and mr_schedule == "after_full_stack":
            x = _run_multirate_repeats(x, 1)
        if multirate_active:
            _mr_graph_ms = float(_mr_graph_ms + (time.monotonic() - graph_segment_t0) * 1000.0)

        graph_refine_ms = (time.monotonic() - graph_refine_t0) * 1000.0

        if multirate_active:
            touched_levels = sorted(set(int(level) for level in _mr_touched_all if int(level) >= 0))
            multirate_stats: Dict[str, Any] = {
                "graph_ms": float(_mr_graph_ms if _mr_call_count > 0 else graph_refine_ms),
                "upper_ms": float(_mr_upper_ms),
                "cross_ms": float(_mr_cross_ms),
                "top_ms": float(_mr_top_ms),
                "upper_steps": int(_mr_upper_steps),
                "cross_steps": int(_mr_cross_steps),
                "top_steps": int(_mr_top_steps),
                "call": int(getattr(self, "_pinball_multirate_call_count", 0)),
                "calls_this_forward": int(_mr_call_count),
                "schedule": str(mr_schedule),
                "l2_nodes": int((base_nl == 2).sum().item()),
                "l3_nodes": int((base_nl == 3).sum().item()),
                "workspace_tokens": int(self.pinball_l3_workspace_tokens),
                "cross_l2_selected": int(_mr_cross_l2_selected),
                "cross_l2_updated": int(_mr_cross_l2_updated),
                "generic_cross_pairs": int(_mr_generic_pairs),
                "generic_cross_selected": int(_mr_generic_selected),
                "generic_cross_updated": int(_mr_generic_updated),
                "touched_levels": list(touched_levels),
                "adaptive": bool(mr_adaptive),
                "adaptive_cycles": int(_mr_adaptive_cycles),
                "adaptive_energy": [round(float(e), 5) for e in _mr_adaptive_energy],
                "adaptive_frozen_frac": float(_mr_adaptive_frozen_frac),
            }
            self._last_pinball_multirate_stats = multirate_stats
            self._last_mr_adaptive_cycles = int(_mr_adaptive_cycles)
            if mr_adaptive:
                prev_ema = float(getattr(self, "_mr_adaptive_cycles_ema", 0.0))
                cur = float(_mr_adaptive_cycles)
                self._mr_adaptive_cycles_ema = cur if prev_ema <= 0.0 else (0.98 * prev_ema + 0.02 * cur)
                self._mr_adaptive_cycles_max = max(int(getattr(self, "_mr_adaptive_cycles_max", 0)), int(_mr_adaptive_cycles))
            if bool(getattr(self, "pinball_multirate_debug", False)) and not bool(getattr(self, "_pinball_multirate_runtime_logged", False)):
                logger.info(
                    "Pinball multirate runtime: graph_ms=%.2f upper_ms=%.2f cross_ms=%.2f top_ms=%.2f upper_steps=%d cross_steps=%d top_steps=%d l2=%d l2_selected=%d l2_updated=%d l3=%d workspace=%d",
                    float(multirate_stats["graph_ms"]),
                    float(multirate_stats["upper_ms"]),
                    float(multirate_stats["cross_ms"]),
                    float(multirate_stats["top_ms"]),
                    int(multirate_stats["upper_steps"]),
                    int(multirate_stats["cross_steps"]),
                    int(multirate_stats["top_steps"]),
                    int(multirate_stats["l2_nodes"]),
                    int(multirate_stats["cross_l2_selected"]),
                    int(multirate_stats["cross_l2_updated"]),
                    int(multirate_stats["l3_nodes"]),
                    int(multirate_stats["workspace_tokens"]),
                )
                self._pinball_multirate_runtime_logged = True

        x_out = self.pinball_work_out(x)
        g_out = Data(x=x_out, edge_index=refine_ei, node_level=base_nl)
        # Full (pre-prune) edge set for the hierarchy aux loss: the aux derives its
        # child<->parent map from cross-level edges, which drop_static_cross_level_edges
        # removes from refine_ei (and local-attn pruning thins even without it).
        g_out.edge_index_full = base_ei
        if refine_et is not None:
            g_out.edge_type = refine_et
        if edge_attr_work is not None:
            g_out.edge_attr = edge_attr_work
        g_out.zip_added_total = int(zip_added_total) if zip_ephemeral_enable else None
        g_out.hqd_added_total = int(hqd_added_total) if hqd_enable else None
        g_out.hqd_search_added_total = int(hqd_search_added_total) if hqd_enable else None
        g_out.hqd_reuse_added_total = int(hqd_reuse_added_total) if hqd_enable else None
        g_out.hqd_reused_layers = int(hqd_reused_layers) if hqd_enable else None
        n_l0 = max(1, int((base_nl == 0).sum().item()))
        n_active = max(1, hqd_hit_layers) if hqd_enable else 1
        if hqd_enable and hqd_hit_layers > 0:
            g_out.hqd_avg_l0 = float(hqd_added_total) / float(B) / float(n_l0) / float(n_active)
        else:
            g_out.hqd_avg_l0 = None
        # Attention savings vs full dense at L0
        is_causal = bool(self.hier_ar_enable and self.l0_ar_enable)
        l0_window = max(1, int(self.l0_local_window))
        l0_mask_bool = (base_nl == 0)
        if refine_ei is not None and refine_ei.numel() > 0:
            src_l0 = l0_mask_bool[refine_ei[0]]
            dst_l0 = l0_mask_bool[refine_ei[1]]
            n_l0_graph_total = int((src_l0 & dst_l0).sum().item())
        else:
            n_l0_graph_total = 0
        n_l0_graph_per_sample = n_l0_graph_total
        if is_causal:
            local_win_per_sample = float(n_l0 * l0_window)
            dense_per_layer = n_l0 * (n_l0 + 1) // 2
        else:
            local_win_per_sample = float(n_l0 * 2 * l0_window)
            dense_per_layer = n_l0 * n_l0
        hqd_per_layer_per_sample = float(hqd_added_total) / float(B) / float(max(1, n_active))
        total_actual = local_win_per_sample + float(n_l0_graph_per_sample) + hqd_per_layer_per_sample
        g_out.attn_saved_pct = 100.0 * (1.0 - total_actual / float(dense_per_layer)) if dense_per_layer > 0 else 0.0
        g_out.win_dense_pct = 100.0 * local_win_per_sample / float(dense_per_layer) if dense_per_layer > 0 else 0.0
        g_out.sparse_dense_pct = 100.0 * (float(n_l0_graph_per_sample) + hqd_per_layer_per_sample) / float(dense_per_layer) if dense_per_layer > 0 else 0.0
        g_out.graph_l0_edges_per_sample = int(n_l0_graph_per_sample)
        g_out.l0_attn_is_causal = bool(is_causal)
        g_out.zip_stage_stats = (
            dict(zip_stage_stats_total)
            if zip_stage_stats_total is not None
            else None
        )
        g_out.hqd_stage_stats = dict(hqd_stage_stats_total) if hqd_stage_stats_total is not None else None
        g_out.hqd_profile_stats = dict(hqd_profile_stats_total) if hqd_profile_stats_total is not None else None
        if base_ar_time is not None:
            g_out.node_ar_time = base_ar_time
        if base_lo is not None:
            g_out.level_offsets = base_lo
        if hasattr(unified_graph, "ae_decoder_l0_slice"):
            g_out.ae_decoder_l0_slice = tuple(unified_graph.ae_decoder_l0_slice)
        if hasattr(unified_graph, "node_pos_local"):
            g_out.node_pos_local = unified_graph.node_pos_local
        if hasattr(unified_graph, "level_grid_shapes"):
            g_out.level_grid_shapes = list(unified_graph.level_grid_shapes)
        if base_branch is not None:
            g_out.node_branch = base_branch
        return g_out

    def _pinball_staggered_flush_steps(self, cycles_by_level: List[int], level: int) -> set:
        cycles = max(0, int(cycles_by_level[int(level)]))
        if cycles <= 0:
            return set()
        max_level_cycles = max(1, max(max(0, int(x)) for x in cycles_by_level))
        last_step = (max_level_cycles - 1) + (len(cycles_by_level) - 1 - int(level))
        if cycles == 1:
            return {last_step}
        return {min(last_step, int(round(i * last_step / float(cycles - 1)))) for i in range(cycles)}

    def _pinball_spread_steps(self, cycles: int, max_cycles: int) -> set:
        cycles = max(0, int(cycles))
        max_cycles = max(1, int(max_cycles))
        if cycles <= 0:
            return set()
        if cycles >= max_cycles:
            return set(range(max_cycles))
        return {min(max_cycles - 1, int(i * max_cycles // cycles)) for i in range(cycles)}

    def _pinball_scheduled_total_steps(self, requested_cycles: int) -> int:
        if not bool(getattr(self, "pinball_level_cycle_enable", False)):
            return int(requested_cycles)
        if str(getattr(self, "pinball_level_cycle_mode", "extra_cycles")) == "destination_only":
            return int(requested_cycles)
        cycles_by_level = list(getattr(self, "pinball_level_cycles", []))
        if not cycles_by_level:
            return int(requested_cycles)
        return max(1, max(max(0, int(x)) for x in cycles_by_level))

    def _pinball_layer_schedule(
        self,
        requested_cycles: int,
        num_layers: int,
    ) -> List[Tuple[int, int, Optional[List[int]]]]:
        num_layers = max(0, int(num_layers))
        if num_layers <= 0:
            return []
        all_levels = list(range(int(getattr(self, "num_hier_levels", 0))))
        if not bool(getattr(self, "pinball_level_cycle_enable", False)):
            return [
                (cycle, layer_idx, None)
                for cycle in range(max(0, int(requested_cycles)))
                for layer_idx in range(num_layers)
            ]

        mode = str(getattr(self, "pinball_level_cycle_mode", "extra_cycles"))
        if mode == "destination_only":
            total_steps = max(1, int(requested_cycles) * num_layers)
            schedule: List[Tuple[int, int, Optional[List[int]]]] = []
            for step in range(total_steps):
                active = self._pinball_active_levels_for_step(step, total_steps)
                schedule.append((step // num_layers, step % num_layers, active))
            return schedule

        cycles_by_level = [max(0, int(x)) for x in list(getattr(self, "pinball_level_cycles", []))]
        if not cycles_by_level:
            return [
                (cycle, layer_idx, all_levels)
                for cycle in range(max(0, int(requested_cycles)))
                for layer_idx in range(num_layers)
            ]

        total_passes = max(1, max(cycles_by_level))
        schedule: List[Tuple[int, int, Optional[List[int]]]] = []
        for pass_idx in range(total_passes):
            chunk_by_layer: Dict[int, List[int]] = {}
            full_levels: List[int] = []
            for level, cycles in enumerate(cycles_by_level):
                if cycles <= 0:
                    continue
                if cycles >= total_passes:
                    full_levels.append(int(level))
                    continue
                start = int(math.floor(float(pass_idx * cycles * num_layers) / float(total_passes)))
                end = int(math.floor(float((pass_idx + 1) * cycles * num_layers) / float(total_passes)))
                for layer_linear_idx in range(max(0, start), max(0, end)):
                    layer_idx = int(layer_linear_idx % num_layers)
                    chunk_by_layer.setdefault(layer_idx, []).append(int(level))

            # Staggered flush: advance low-cycle levels by their next layer chunk,
            # then run full-stack passes for max-cycle levels (for [1,1,1,4],
            # this is one quarter of L0-L2 followed by one full L3 pass).
            for layer_idx in sorted(chunk_by_layer):
                active = sorted(set(chunk_by_layer[layer_idx]))
                if active:
                    schedule.append((pass_idx, int(layer_idx), active))
            if full_levels:
                full_active = sorted(set(full_levels))
                for layer_idx in range(num_layers):
                    schedule.append((pass_idx, int(layer_idx), full_active))
        return schedule

    def _pinball_active_levels_for_step(self, step: int, max_steps: int) -> List[int]:
        if not bool(getattr(self, "pinball_level_cycle_enable", False)):
            return list(range(int(getattr(self, "num_hier_levels", 0))))
        if str(getattr(self, "pinball_level_cycle_mode", "extra_cycles")) == "destination_only":
            cycles_by_level = list(getattr(self, "pinball_level_cycles", []))
            total_layers = max(1, int(max_steps))
            schedule = str(getattr(self, "pinball_cycle_schedule", "spread"))
            max_level_cycles = max(1, max(max(0, int(x)) for x in cycles_by_level)) if cycles_by_level else 1
            if schedule == "staggered_flush":
                source_steps = max_level_cycles + len(cycles_by_level) - 1
            else:
                source_steps = max_level_cycles
            active: List[int] = []
            for level, cycles in enumerate(cycles_by_level):
                cycles = int(cycles)
                if cycles <= 0:
                    continue
                if schedule == "frontloaded":
                    source_active = set(range(min(cycles, source_steps)))
                elif schedule == "staggered_flush":
                    source_active = self._pinball_staggered_flush_steps(cycles_by_level, level)
                else:
                    source_active = self._pinball_spread_steps(cycles, source_steps)
                mapped_active = set()
                for source_step in source_active:
                    if source_steps <= 1 or total_layers <= 1:
                        mapped_active.add(0)
                    else:
                        mapped_active.add(int(round(float(source_step) * float(total_layers - 1) / float(source_steps - 1))))
                if int(step) in mapped_active:
                    active.append(int(level))
            return active
        cycles_by_level = list(getattr(self, "pinball_level_cycles", []))
        schedule = str(getattr(self, "pinball_cycle_schedule", "spread"))
        active: List[int] = []
        for level, cycles in enumerate(cycles_by_level):
            cycles = int(cycles)
            if cycles <= 0:
                continue
            if schedule == "frontloaded":
                runs = int(step) < cycles
            elif schedule == "staggered_flush":
                runs = int(step) in self._pinball_staggered_flush_steps(cycles_by_level, level)
            else:
                runs = int(step) in self._pinball_spread_steps(cycles, max_steps)
            if runs:
                active.append(int(level))
        return active

    def _pinball_level_mask(self, node_level: torch.Tensor, active_levels: List[int]) -> torch.Tensor:
        if not active_levels:
            return torch.zeros_like(node_level, dtype=torch.bool)
        mask = torch.zeros_like(node_level, dtype=torch.bool)
        for level in active_levels:
            mask = mask | (node_level == int(level))
        return mask


    def _apply_unified_refinement(
        self,
        unified_graph,
        num_cycles: int,
        reveal_target_ids: Optional[torch.Tensor] = None,
        reveal_mask: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
        per_layer_hook=None,
    ): #_gemini
            from torch_geometric.data import Data

            # KV-cached decode: the generation driver installs its capture/inject hook as an
            # attribute (the enhanced forward doesn't plumb per_layer_hook through). Explicit
            # arguments win; unset attribute = existing behavior.
            if per_layer_hook is None:
                per_layer_hook = getattr(self, "_gen_layer_hook", None)

            # 1. Detect Batching
            x = unified_graph.x
            if x.dim() == 2: x = x.unsqueeze(0) # [1, N, H]
            B, N, H = x.shape
            device = x.device

            requested_mode = getattr(self, "refinement_batch_mode", "blockdiag")
            active_mode = requested_mode
            fallback_reason = None
            if requested_mode == "true_batch_nozip":
                if bool(getattr(self, "zip_enable", False)) and getattr(self, "zip_execution_mode", "edge_mutation") != "ephemeral_msg":
                    fallback_reason = "TB001: true_batch_nozip requires zip_execution_mode='ephemeral_msg' when zip_enable=True"
                elif bool(getattr(self, "zip_enable", False)) and getattr(self, "zip_score_mode", "fast_qk") != "fast_qk":
                    fallback_reason = "TB007: true_batch_nozip zipper currently supports zip_score_mode='fast_qk' only"
                elif bool(getattr(self, "use_neighbor_sampling", False)) and getattr(self, "num_neighbors", [-1]) != [-1]:
                    fallback_reason = "TB002: true_batch_nozip requested but neighbor sampling is enabled"

                if fallback_reason is not None:
                    if bool(getattr(self, "true_batch_strict", False)):
                        raise RuntimeError(fallback_reason)
                    logger.warning(fallback_reason)
                    active_mode = "blockdiag"
                else:
                    active_mode = "true_batch_nozip"

            self._last_refinement_batch_mode = active_mode
            self._last_refinement_batch_fallback = fallback_reason
            if bool(getattr(self, "pinball_work_projection_effective", False)) and active_mode != "true_batch_nozip":
                raise RuntimeError("pinball_work_dim != hidden_dim currently requires refinement_batch_mode='true_batch_nozip' without fallback")
            if not bool(getattr(self, "_pinball_refinement_runtime_logged", False)):
                logger.info(
                    "Pinball refinement runtime: requested=%s active=%s fallback=%s hetero_effective=%s level_cycles=%s graph_active_compute=%s cycle_mode=%s",
                    str(requested_mode),
                    str(active_mode),
                    "none" if fallback_reason is None else str(fallback_reason),
                    False,
                    bool(getattr(self, "pinball_level_cycle_enable", False)),
                    str(getattr(self, "pinball_graph_active_compute", "all")),
                    str(getattr(self, "pinball_level_cycle_mode", "extra_cycles")),
                )
                self._pinball_refinement_runtime_logged = True
            x_flat = x.reshape(B * N, H)

            # 2. Prepare Base Topology
            base_ei = unified_graph.edge_index.to(device) # [2, E]
            base_nl = unified_graph.node_level.to(device) # [N]
            base_pos_local = getattr(unified_graph, "node_pos_local", None)
            if base_pos_local is not None:
                base_pos_local = torch.as_tensor(base_pos_local, device=device, dtype=torch.long)
            base_ea = getattr(unified_graph, "edge_attr", None)
            base_lo = getattr(unified_graph, "level_offsets", None)
            base_ar_time = getattr(unified_graph, "node_ar_time", None)
            base_level_grid_shapes = getattr(unified_graph, "level_grid_shapes", None)
            #lo = getattr(unified_graph, "level_offsets", None)      
            if base_lo is not None: base_lo = torch.as_tensor(base_lo, device=device)
            if base_ar_time is not None:
                base_ar_time = base_ar_time.to(device=device, dtype=torch.long)

            if active_mode == "true_batch_nozip":
                need_pre_state = False
                x_tb_pre = None
                if need_pre_state:
                    x_tb_pre = unified_graph.x
                    if x_tb_pre.dim() == 2:
                        x_tb_pre = x_tb_pre.unsqueeze(0)
                    x_tb_pre = x_tb_pre.clone()
                tb_graph = self._apply_unified_refinement_true_batch_nozip(
                    unified_graph,
                    num_cycles,
                    cond_vec=cond_vec,
                    per_layer_hook=per_layer_hook,
                )
                x_tb = tb_graph.x if tb_graph.x.dim() == 3 else tb_graph.x.unsqueeze(0)
                tb_lo = getattr(tb_graph, "level_offsets", None)
                if tb_lo is not None:
                    tb_lo = torch.as_tensor(tb_lo, device=device)
                self._last_hier_aux_loss = self._compute_true_batch_aux_loss(
                    x_bnh=x_tb,
                    base_ei=getattr(tb_graph, "edge_index_full", tb_graph.edge_index).to(device),
                    base_nl=tb_graph.node_level.to(device),
                    base_lo=tb_lo,
                    base_ar_time=getattr(tb_graph, "node_ar_time", None),
                )
                self._last_hier_predaux_loss = self._compute_predictive_aux_loss(x_tb, tb_lo)
                self._set_true_batch_nozip_tail_metrics(
                    cycles_used=num_cycles,
                    zip_added_total=getattr(tb_graph, "zip_added_total", None),
                    zip_stage_stats=getattr(tb_graph, "zip_stage_stats", None),
                    hqd_added_total=getattr(tb_graph, "hqd_added_total", None),
                    hqd_stage_stats=getattr(tb_graph, "hqd_stage_stats", None),
                    hqd_profile_stats=getattr(tb_graph, "hqd_profile_stats", None),
                    hqd_avg_l0=getattr(tb_graph, "hqd_avg_l0", None),
                    attn_saved_pct=getattr(tb_graph, "attn_saved_pct", None),
                    win_dense_pct=getattr(tb_graph, "win_dense_pct", None),
                    sparse_dense_pct=getattr(tb_graph, "sparse_dense_pct", None),
                    graph_l0_edges_per_sample=getattr(tb_graph, "graph_l0_edges_per_sample", None),
                    l0_attn_is_causal=getattr(tb_graph, "l0_attn_is_causal", None),
                )
                tb_nl_base = tb_graph.node_level.to(device)
                tb_lo = getattr(tb_graph, "level_offsets", None)
                if tb_lo is not None:
                    tb_lo = torch.as_tensor(tb_lo, device=device)
                tb_pos_local = getattr(tb_graph, "node_pos_local", None)
                if tb_pos_local is not None:
                    tb_pos_local = torch.as_tensor(tb_pos_local, device=device, dtype=torch.long)

                tb_nl = tb_nl_base
                if x_tb.size(0) > 1:
                    tb_nl = tb_nl.repeat(int(x_tb.size(0)))

                return tb_graph

            # 3. Vectorized Construction
            if B == 1:
                ei_merged = base_ei
                nl_merged = base_nl
                ea_merged = base_ea
                ar_time_merged = base_ar_time
                # RoPE: simple 0..N-1
                if base_pos_local is not None:
                    pos_flat = base_pos_local
                else:
                    pos_flat = torch.arange(N, device=device)
                    if base_lo is not None:
                        pos_flat = pos_flat - base_lo[base_nl]
            else:
                # --- Offsets for Edge Index ---
                # [0, N, 2N...]
                batch_offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
                
                # Tile Edge Index: [2, B*E]
                # (1,2,E) + (B,1,1) -> (B,2,E) -> (2, B*E)
                ei_merged = (base_ei.unsqueeze(0) + batch_offsets).transpose(0, 1).reshape(2, -1)

                # Tile Node Levels: [B*N]
                nl_merged = base_nl.repeat(B)

                # Tile Edge Attributes: [B*E, D] or None
                if base_ea is not None:
                    ea_merged = base_ea.repeat(B, 1)
                else:
                    ea_merged = None

                if base_ar_time is not None:
                    ar_time_merged = base_ar_time.repeat(B)
                else:
                    ar_time_merged = None

                # --- RoPE Position Reset ---
                # Standard arange(B*N) is WRONG for RoPE (monotonically increases).
                # We need [0..N-1, 0..N-1] repeated.
                local_idx = torch.arange(N, device=device).repeat(B)
                if base_pos_local is not None:
                    pos_flat = base_pos_local.repeat(B)
                elif base_lo is not None:
                    pos_flat = local_idx - base_lo[nl_merged]
                else:
                    pos_flat = local_idx

                # RoPE safety guard for B>1 block-diagonal path: positions must repeat per sample.
                expected_local = torch.arange(N, device=device).repeat(B)
                if not torch.equal(local_idx, expected_local):
                    rope_reason = "TBROPE01: invalid repeated local positions in blockdiag path"
                    if bool(getattr(self, "true_batch_strict", False)):
                        raise RuntimeError(rope_reason)
                    logger.warning(rope_reason)

            # 4. LapPE Tiling
            lap_pe = getattr(unified_graph, "lap_pe_raw_cpu", None)
            if lap_pe is not None:
                lap_pe = lap_pe.to(device).repeat(B, 1)

            # 5. Build Graph
            g_work = Data(x=x_flat, edge_index=ei_merged, edge_attr=ea_merged, node_level=nl_merged)
            if ar_time_merged is not None:
                g_work.node_ar_time = ar_time_merged
            
            # Inject LapPE
            if lap_pe is not None and self.lap_pe_proj is not None:
                lap_pe = self._lap_pe_raw_on_device(lap_pe, device=device, dtype=g_work.x.dtype)
                g_work.x = g_work.x + self.lap_pe_proj(lap_pe)
            if lap_pe is None and self.lap_pe_proj is not None and not self._lap_pe_missing_warned:
                logger.warning("Skipping LapPE due to absence in unified graph.")
                self._lap_pe_missing_warned = True


            # 6. Run Refinement
            l0_mask = (nl_merged == 0)
            apply_l0_alpha = bool(getattr(self, "l0_alpha_enable", True) and hasattr(self, "alpha"))
            x_l0_orig = g_work.x[l0_mask].clone() if apply_l0_alpha else None
            l1_mask = (nl_merged == 1)
            l2_mask = (nl_merged == 2)
            l3_mask = (nl_merged == 3)


            zip_enable = getattr(self, "zip_enable", False)
            zip_added_total = 0
            if zip_enable:
                self._last_zip_added_total = 0
                self._last_zip_stage_stats = None
            base_edge_count = g_work.edge_index.size(1)
            edge_attn_agg = None

            if zip_enable:
                edge_attn_agg = torch.zeros(base_edge_count, device=device, dtype=g_work.x.dtype)

            if not zip_enable:
                self._last_zip_added_total = None

            use_sampled_path = bool(
                getattr(self, "use_neighbor_sampling", False)
                and self.num_neighbors != [-1]
            )
            trm_requires_forced_grad_cycle = bool(self.TRM and not use_sampled_path)
            force_grad_next_cycle = False
            break_after_forced_grad = False


            # ---------- Optional level-projection init (reintroduce hierarchy pooling) ----------
            # Only if we have both level_offsets and level_projections
            #if hasattr(self, "level_projections") and self.level_projections is not None:
            #    max_level = int(unified_graph.node_level.max().item())
            #    if unified_graph.level_offsets is not None:
                    # For single-sample: g_work.x is [N,H]
                    # For batched:      g_work.x is [B*N,H] -> reshape to [B,N,H]
                    #print("Applying hierarchical level init...")
                    #logger.info("Applying hierarchical level init...")
            #         if batched:
            #             # reuse B, N, H from above
            #             x_tmp = g_work.x.view(B, N, H)
            #             x_tmp = self._hierarchical_level_init_inplace(
            #                 x_tmp,
            #                 level_offsets=base_lo,   # offsets per sample
            #                 max_level=max_level,
            #             )
            #             g_work.x = x_tmp.view(B * N, H)
            #         else:
            #        x_tmp = self._hierarchical_level_init_inplace(
            #        g_work.x,                # [N,H]
            #        level_offsets=lo,
            #        max_level=max_level,
            #        )
            #        g_work.x = x_tmp
                    #print("Applied hierarchical level init.")
            #    else:
            #        logger.debug("No level_offsets; skipping hierarchical level init.")
            #        print("No level_offsets; skipping hierarchical level init.")
            
            # Cache data on CPU for NeighborLoader (created once, reused)
            _data_cpu = None
            _pos_cpu = None
            cycles_used = 0
            scheduled_steps = self._pinball_scheduled_total_steps(int(num_cycles))

            if self.share_transformers:
                transformers_to_use = [m for level_mods in self.level_transformers for m in level_mods]
            elif self.refinement_transformers is not None:
                transformers_to_use = list(self.refinement_transformers)
            else:
                logger.warning("No transformers specified for unified refinement cycles. Stopping.")
                transformers_to_use = []
            self._last_pinball_active_levels = []

            for cycle in range(scheduled_steps):
                if not transformers_to_use:
                    break
                cycles_used = cycle + 1
                active_levels = self._pinball_active_levels_for_step(cycle, scheduled_steps)
                self._last_pinball_active_levels.append(list(active_levels))
                active_node_mask = self._pinball_level_mask(nl_merged, active_levels)
                if not bool(active_node_mask.any()):
                    continue
                cycle_x_start = g_work.x.clone() if bool(getattr(self, "pinball_level_cycle_enable", False)) else None
                #transformers = self.refinement_transformers # or self.level_transformers logic
                keep_grad = bool(force_grad_next_cycle or (cycle >= scheduled_steps - 1))
                if force_grad_next_cycle:
                    force_grad_next_cycle = False
                for layer_idx, transformer in enumerate(transformers_to_use):
                    g_work.x = self._apply_refinement_conditioning_flat(
                        x_flat=g_work.x,
                        batch_size=B,
                        nodes_per_sample=N,
                        cond_vec=cond_vec,
                    )
                    # Set last attention value as edge attribute 
                    edge_attr_input = g_work.edge_attr if hasattr(g_work, "edge_attr") and self.use_edge_attr else None
                    new_edge_attr = None
                    
                    
                    #layerwarmedup = layer_idx > 1 if cycle == 0 else True
                    #if cycle == num_cycles - 1 and layer_idx == len(transformers_to_use) - 2:
                    #    layerwarmedup = False
                    layerwarmedup = True # disable warmup logic for now, as it’s not working well and adds complexity
                    seed_nodes = None

                    # === SAMPLED PATH ===
                    if getattr(self, 'use_neighbor_sampling', False) and self.num_neighbors != [-1]:
                        edge_attr_global = None
                        if self.use_edge_attr or zip_enable:
                            edge_attr_global = edge_attr_input
                            if edge_attr_global is None:
                                edge_attr_global = torch.zeros(
                                    g_work.edge_index.size(1),
                                    device=g_work.edge_index.device,
                                    dtype=g_work.x.dtype,
                                )
                        # Lazy init CPU data (once) for PyG NeighborLoader
                        if _data_cpu is None:
                            if bool(getattr(self, "sampler_debug_checks", False)) and g_work.edge_index.numel() > 0:
                                min_edge = int(g_work.edge_index.min().detach().item())
                                max_edge = int(g_work.edge_index.max().detach().item())
                                if min_edge < 0 or max_edge >= int(g_work.x.size(0)):
                                    raise RuntimeError(
                                        "NeighborLoader received invalid edge_index: "
                                        f"min={min_edge} max={max_edge} num_nodes={int(g_work.x.size(0))}"
                                    )
                            _data_cpu = Data(
                                edge_index=g_work.edge_index.cpu(),
                                node_level=g_work.node_level.cpu(),
                                num_nodes=g_work.x.size(0),
                            )
                            _data_cpu.level_grid_shapes = list(base_level_grid_shapes) if base_level_grid_shapes is not None else []
                            _pos_cpu = pos_flat.cpu()
                            # Precompute sparse adjacency to avoid O(E) CSR rebuild
                            # inside every NeighborLoader / NeighborSampler construction.
                            if int(_data_cpu.edge_index.size(1)) > 0:
                                try:
                                    from torch_geometric.transforms import ToSparseTensor
                                    _data_cpu = ToSparseTensor(remove_edge_index=False)(_data_cpu)
                                    _precompute_sparse = getattr(_data_cpu, "adj_t", None)
                                except Exception:
                                    pass
                        
                        # PyG NeighborLoader neighbor sampling
                        g_work.x, edge_attr_global = self._apply_transformer_sampled(
                            transformer=transformer,
                            x_global=g_work.x,
                            data_cpu=_data_cpu,
                            pos_global=_pos_cpu,
                            edge_attr_global=edge_attr_global,
                            input_nodes=seed_nodes,
                            nodes_per_sample=int(N),
                        )
                        if edge_attr_global is not None:
                            g_work.edge_attr = edge_attr_global

                        # entropy/tick updates are handled inside _apply_transformer_sampled
                    else:
                        # Backend layer step (blockdiag)
                        x_before = g_work.x
                        g_work.x, new_edge_attr = self._refine_step_blockdiag(
                            transformer=transformer,
                            x_flat=g_work.x,
                            edge_index=g_work.edge_index,
                            node_level=nl_merged,
                            pos_flat=pos_flat,
                            edge_attr_input=edge_attr_input,
                            keep_grad=keep_grad,
                            level_grid_shapes=base_level_grid_shapes,
                            spatial_metric=getattr(self, "graph_spatial_metric", "chebyshev"),
                        )

                        if bool(getattr(self, "pinball_level_cycle_enable", False)):
                            g_work.x = torch.where(active_node_mask.unsqueeze(-1), g_work.x, x_before)

                    if new_edge_attr is not None and (self.use_edge_attr or zip_enable):
                        g_work.edge_attr = new_edge_attr

                    if zip_enable and g_work.edge_attr is not None:
                        if edge_attn_agg is None:
                            edge_attn_agg = torch.zeros_like(g_work.edge_attr)
                        elif edge_attn_agg.numel() < g_work.edge_attr.numel():
                            pad = g_work.edge_attr.numel() - edge_attn_agg.numel()
                            if pad > 0:
                                edge_attn_agg = torch.cat(
                                    [edge_attn_agg, torch.zeros(pad, device=device, dtype=g_work.edge_attr.dtype)]
                                )

                        if self.zip_attn_agg == "ema":
                            edge_attn_agg = (
                                self.zip_attn_ema_beta * edge_attn_agg
                                + (1.0 - self.zip_attn_ema_beta) * g_work.edge_attr
                            )
                        else:
                            edge_attn_agg = torch.maximum(edge_attn_agg, g_work.edge_attr)

                        if self.zip_persist == "layer_only":
                            if g_work.edge_index.size(1) > base_edge_count:
                                g_work.edge_index = g_work.edge_index[:, :base_edge_count]
                                g_work.edge_attr = g_work.edge_attr[:base_edge_count]
                                edge_attn_agg = edge_attn_agg[:base_edge_count]
                        elif self.zip_persist == "cycle_only" and layer_idx == 0:
                            if g_work.edge_index.size(1) > base_edge_count:
                                g_work.edge_index = g_work.edge_index[:, :base_edge_count]
                                g_work.edge_attr = g_work.edge_attr[:base_edge_count]
                                edge_attn_agg = edge_attn_agg[:base_edge_count]
                        elif self.zip_persist == "decay" and g_work.edge_index.size(1) > base_edge_count:
                            dyn_slice = slice(base_edge_count, g_work.edge_attr.size(0))
                            g_work.edge_attr[dyn_slice] = g_work.edge_attr[dyn_slice] * self.zip_edge_decay
                            if self.zip_edge_drop_threshold > 0:
                                dyn_keep = g_work.edge_attr[dyn_slice] >= self.zip_edge_drop_threshold
                                if not dyn_keep.all():
                                    keep_mask = torch.ones(g_work.edge_attr.size(0), dtype=torch.bool, device=device)
                                    keep_mask[dyn_slice] = dyn_keep
                                    g_work.edge_index = g_work.edge_index[:, keep_mask]
                                    g_work.edge_attr = g_work.edge_attr[keep_mask]
                                    edge_attn_agg = edge_attn_agg[keep_mask]

                        if self.zip_granularity == "per_layer" or (
                            self.zip_granularity == "per_cycle" and layer_idx == 0
                        ):
                            if edge_attn_agg.numel() > 0:
                                edge_index_candidate, cand_scores, zip_stage_stats = self._zipper_build_blockdiag_mutation_candidates_batched(
                                    transformer=transformer,
                                    x_flat=g_work.x,
                                    base_edge_index=g_work.edge_index,
                                    node_level=nl_merged,
                                    node_ar_time=ar_time_merged,
                                    batch_size=B,
                                    nodes_per_sample=N,
                                    enable_l3_stage=(layer_idx % max(1, int(self.zip_l3_all_pairs_every)) == 0),
                                )
                                self._last_zip_stage_stats = dict(zip_stage_stats)

                                if edge_index_candidate.numel() > 0:
                                    # Safety: only mutate within each sample block.
                                    if B > 1:
                                        src_c = edge_index_candidate[0]
                                        dst_c = edge_index_candidate[1]
                                        keep_same = (src_c // int(N)) == (dst_c // int(N))
                                        if not bool(keep_same.all()):
                                            edge_index_candidate = edge_index_candidate[:, keep_same]
                                            cand_scores = cand_scores[keep_same]

                                if edge_index_candidate.numel() > 0:
                                    dyn_budget_per_sample = int(getattr(self, "zip_max_dyn_edges_per_sample", 0))
                                    if dyn_budget_per_sample > 0:
                                        dyn_budget_total = dyn_budget_per_sample * int(B)
                                    else:
                                        dyn_budget_total = int(self.zip_max_dyn_edges_total)

                                    dyn_limit = max(
                                        0,
                                        int(dyn_budget_total) - int(g_work.edge_index.size(1) - base_edge_count),
                                    )
                                    if dyn_limit > 0 and cand_scores.numel() > 0:
                                        if cand_scores.numel() > dyn_limit:
                                            top_idx = torch.topk(cand_scores, dyn_limit, largest=True).indices
                                            edge_index_candidate = edge_index_candidate[:, top_idx]
                                            cand_scores = cand_scores[top_idx]

                                        g_work.edge_index = torch.cat(
                                            [g_work.edge_index, edge_index_candidate], dim=1
                                        )
                                        g_work.edge_attr = torch.cat(
                                            [g_work.edge_attr, cand_scores], dim=0
                                        )
                                        edge_attn_agg = torch.cat(
                                            [edge_attn_agg, cand_scores], dim=0
                                        )
                                        zip_added_total += edge_index_candidate.size(1)

                    
                    if new_edge_attr is not None:
                        g_work.edge_attr = new_edge_attr

                    if apply_l0_alpha and ((not bool(getattr(self, "pinball_level_cycle_enable", False))) or 0 in active_levels):
                        a = self.alpha
                        if (
                            getattr(self, 'use_neighbor_sampling', False)
                            and seed_nodes is not None
                            and seed_nodes.numel() > 0
                        ):
                            updated_mask = torch.zeros_like(l0_mask, dtype=torch.bool)
                            updated_mask[seed_nodes] = True
                            l0_idx = torch.nonzero(l0_mask, as_tuple=False).view(-1)
                            l0_updated_pos = torch.nonzero(updated_mask[l0_idx], as_tuple=False).view(-1)
                            l0_updated_idx = l0_idx.index_select(0, l0_updated_pos)
                            g_work.x[l0_updated_idx] = (
                                a * g_work.x[l0_updated_idx]
                                + (1.0 - a) * x_l0_orig.index_select(0, l0_updated_pos)
                            )
                        else:
                            g_work.x[l0_mask] = a * g_work.x[l0_mask] + (1.0 - a) * x_l0_orig

                g_work.x = self.layer_norm(g_work.x)
                if cycle_x_start is not None:
                    g_work.x = torch.where(active_node_mask.unsqueeze(-1), g_work.x, cycle_x_start)
                if break_after_forced_grad and keep_grad:
                    break

            if not zip_enable:
                self._last_zip_stage_stats = None
            self._last_zip_msg_norm_ratio_mean = None
            self._last_zip_msg_norm_ratio_max = None
            if zip_enable and zip_added_total > 0:
                self._last_zip_added_total = int(zip_added_total)
                if self.zip_log_enable:
                    logger.info(
                        f"[ZIP] Added {zip_added_total} dynamic edges in unified refinement."
                    )
            elif zip_enable:
                self._last_zip_added_total = 0

            self._last_cycles_used = int(cycles_used)

            aux_loss = None
            if self.training and self.use_aux_loss:
                try:
                    aux_loss = self._compute_hierarchy_aux_loss_runtime(g_work)
                except Exception as e:
                    logger.warning(f"Hierarchy aux loss failed: {e}")
                    aux_loss = None

            self._last_hier_aux_loss = aux_loss
            self._last_hier_predaux_loss = None  # predictive aux runs on the fast path only

            # 7. Reshape and Restore
            unified_graph.x = g_work.x.view(B, N, H)# if B > 1 else g_work.x.view(N, H)
            
            return unified_graph


    
    
    def add_next_token_node(self, refined_graph, token_embedding):
        """
        Add a node for the next token to the graph.
        
        Args:
            refined_graph: The current graph
            token_embedding: Embedding for the next token
            
        Returns:
            updated_graph: Graph with next token node added
            next_token_idx: Index of the next token node
        """
        device = refined_graph.x.device
        
        # Get index for the new node
        next_token_idx = refined_graph.x.size(0)
        
        # Add the node features
        refined_graph.x = torch.cat([refined_graph.x, token_embedding.unsqueeze(0)], dim=0)
        
        # Add node level (L0)
        refined_graph.node_level = torch.cat([
            refined_graph.node_level,
            torch.zeros(1, dtype=torch.long, device=device)
        ], dim=0)
        
        # Create connections to recent tokens
        if hasattr(refined_graph, 'level_offsets') and len(refined_graph.level_offsets) > 1:
            l0_end = refined_graph.level_offsets[1]
        else:
            # Handle case where level_offsets might not be available or complete
            l0_end = refined_graph.x.size(0) - 1  # All nodes except the one we just added
        
        context_window = min(1, l0_end)  # Use last 5 tokens or fewer
        
        # Create edges to recent tokens
        edges = []
        
        # Connect to recent tokens
        for i in range(max(0, l0_end - context_window), l0_end):
            edges.append([i, next_token_idx])
            edges.append([next_token_idx, i])
        
        # Add edges to graph
        if edges:
            edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).t()
            new_edge_index = torch.cat([refined_graph.edge_index, edge_tensor], dim=1)
            
            # Add edge types (all 0 for token-level connections)
            num_new_edges = edge_tensor.size(1)
            edge_types = torch.zeros(num_new_edges, dtype=torch.long, device=device)
            new_edge_type = torch.cat([refined_graph.edge_type, edge_types])
            
            # Generate edge features for new edges if we're using edge attributes
            if hasattr(refined_graph, 'edge_attr') and hasattr(self, 'use_edge_attr') and self.use_edge_attr:
                # Use edge feature generator if available
                if hasattr(self, 'edge_feature_generator'):
                    # Create temporary graph with all nodes and only new edges for feature generation
                    temp_graph = Data(
                        x=refined_graph.x,
                        edge_index=edge_tensor,
                        edge_type=edge_types
                    )
                    
                    # Generate features for new edges
                    new_edge_attr = self.edge_feature_generator(
                        temp_graph.x, 
                        temp_graph.edge_index, 
                        temp_graph.edge_type
                    )
                    
                    # Combine existing and new edge attributes
                    new_edge_attr_full = torch.cat([refined_graph.edge_attr, new_edge_attr], dim=0)
                    
                    # Update graph with new edges and attributes
                    refined_graph.edge_index = new_edge_index
                    refined_graph.edge_type = new_edge_type
                    refined_graph.edge_attr = new_edge_attr_full
                else:
                    # If no edge feature generator, just create zero-initialized edge features
                    new_edge_attr = torch.zeros((num_new_edges, refined_graph.edge_attr.size(1)), 
                                            dtype=torch.float, device=device)
                    refined_graph.edge_attr = torch.cat([refined_graph.edge_attr, new_edge_attr], dim=0)
                    refined_graph.edge_index = new_edge_index
                    refined_graph.edge_type = new_edge_type
            else:
                # Update graph without edge attributes
                refined_graph.edge_index = new_edge_index
                refined_graph.edge_type = new_edge_type
        
        # Update level offsets if they exist
        if hasattr(refined_graph, 'level_offsets') and len(refined_graph.level_offsets) > 1:
            refined_graph.level_offsets[1] += 1
        
        return refined_graph, next_token_idx

    def process_next_token(self, refined_graph, next_token_idx, num_cycles=2):
        """
        Process graph with next token node to get predictions.
        
        Args:
            refined_graph: Graph with next token node
            next_token_idx: Index of next token node
            num_cycles: Number of refinement cycles
            
        Returns:
            processed_graph: Updated graph after processing
        """
        # Check if the graph is valid before processing
        if refined_graph is None or next_token_idx is None:
            raise ValueError("Invalid graph or token index")
        
        if not hasattr(refined_graph, 'x') or refined_graph.x is None:
            raise ValueError("Graph missing node features")
            
        if not hasattr(refined_graph, 'edge_index') or refined_graph.edge_index is None:
            raise ValueError("Graph missing edge index")
        
        # Process with refinement cycles
        for cycle in range(num_cycles):
            if hasattr(self, 'share_transformers') and self.share_transformers:
                # Use all level transformers in sequence
                for level_idx, level_transformers in enumerate(self.level_transformers):
                    for transformer in level_transformers:
                        # Apply transformer layer
                        edge_attr = None
                        if hasattr(refined_graph, 'edge_attr') and hasattr(self, 'use_edge_attr') and self.use_edge_attr:
                            edge_attr = refined_graph.edge_attr
                        
                        refined_graph.x = transformer(
                            refined_graph.x,
                            refined_graph.edge_index,
                            refined_graph.node_level,
                            edge_attr=edge_attr
                        )
                        
                        # Apply layer normalization
                        refined_graph.x = self.layer_norm(refined_graph.x)
            else:
                # Use dedicated refinement transformers
                for transformer in self.refinement_transformers:
                    # Apply transformer layer
                    edge_attr = None
                    if hasattr(refined_graph, 'edge_attr') and hasattr(self, 'use_edge_attr') and self.use_edge_attr:
                        edge_attr = refined_graph.edge_attr
                    
                    refined_graph.x = transformer(
                        refined_graph.x,
                        refined_graph.edge_index,
                        refined_graph.node_level,
                        edge_attr=edge_attr
                    )
                    
                    # Apply layer normalization
                    refined_graph.x = self.layer_norm(refined_graph.x)
        
        return refined_graph

    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hierarchical_features: bool = False,
        return_token_features: Optional[bool] = None,
        num_cycles: int = None,
        use_level_prediction: bool = False,  # New parameter
        reveal_target_ids: Optional[torch.Tensor] = None,
        reveal_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the unified hierarchical graph transformer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Optional position IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_hierarchical_features: Whether to return features from all levels
            num_cycles: Number of hierarchical processing cycles
            use_level_prediction: Whether to use level projection (like UnifiedHierarchicalGAT)
            
        Returns:
            logits: Output token prediction logits [batch_size, seq_len, vocab_size]
            features: Optional hierarchical features if return_hierarchical_features=True
        """
        batch_size, seq_len = input_ids.shape

        # Handle batching - process each example separately for now
        # A more advanced implementation could process the whole batch at once
        # if batch_size > 1:
        #     all_logits = []
        #     all_features = []
            
        #     for i in range(batch_size):
        #         if return_hierarchical_features:
        #             logits, features = self.forward(
        #                 input_ids[i:i+1],
        #                 position_ids[i:i+1] if position_ids is not None else None,
        #                 attention_mask[i:i+1] if attention_mask is not None else None,
        #                 return_hierarchical_features=True,
        #                 num_cycles=num_cycles,
        #                 use_level_prediction=use_level_prediction,
        #             )
        #             all_logits.append(logits)
        #             all_features.append(features)
        #         else:
        #             logits = self.forward(
        #                 input_ids[i:i+1],
        #                 position_ids[i:i+1] if position_ids is not None else None,
        #                 attention_mask[i:i+1] if attention_mask is not None else None,
        #                 return_hierarchical_features=False,
        #                 num_cycles=num_cycles,
        #                 use_level_prediction=use_level_prediction,
        #             )
        #             all_logits.append(logits)
            
        #     # Combine results
        #     combined_logits = torch.cat(all_logits, dim=0)
            
        #     if return_hierarchical_features:
        #         return combined_logits, all_features
        #     else:
        #         return combined_logits
        
        # Use refinement cycles if specified, otherwise use default
        cycles = num_cycles if num_cycles is not None else self.refinement_cycles
        
        # Reset level mappings
        self.level_mappings = []
        
        # Get token embeddings with positional information
        token_embeddings = self._get_embeddings(input_ids, position_ids, max_seq_len=seq_len)
        token_embeddings = token_embeddings.view(-1, self.hidden_dim)
        
        
        # Step 1: Build and process L0 (token level)
        l0_graph = self._build_level_graph(token_embeddings, level_idx=0)
        l0_processed = self._process_level(l0_graph, level_idx=0)
        del l0_graph  # Free memory
        # Store all processed level graphs
        level_graphs = [l0_processed]
        
        # Step 2: Build L1 from processed L0
        if len(self.compression_ratios) > 0 and len(self.level_transformers) > 1:
            l1_graph, l0_l1_mapping = self._create_next_level(
                l0_processed, 
                level_idx=1,
                compression_ratio=self.compression_ratios[0],
                overlap_ratio=self.overlap_ratios[0]
            )
            
            # Process L1
            l1_processed = self._process_level(l1_graph, level_idx=1)
            del l1_graph  # Free memory
            # Store mapping and processed graph
            self.level_mappings.append(l0_l1_mapping)
            level_graphs.append(l1_processed)
            
            # Step 3: Build L2 from processed L1
            if len(self.compression_ratios) > 1 and len(self.level_transformers) > 2:
                l2_graph, l1_l2_mapping = self._create_next_level(
                    l1_processed,
                    level_idx=2,
                    compression_ratio=self.compression_ratios[1],
                    overlap_ratio=self.overlap_ratios[1]
                )
                
                # Process L2
                l2_processed = self._process_level(l2_graph, level_idx=2)
                del l2_graph  # Free memory
                # Store mapping and processed graph
                self.level_mappings.append(l1_l2_mapping)
                level_graphs.append(l2_processed)
                
                # Step 4: Build L3 from processed L2
                if len(self.compression_ratios) > 2 and len(self.level_transformers) > 3:
                    l3_graph, l2_l3_mapping = self._create_next_level(
                        l2_processed,
                        level_idx=3,
                        compression_ratio=self.compression_ratios[2],
                        overlap_ratio=self.overlap_ratios[2]
                    )
                    
                    # Process L3
                    l3_processed = self._process_level(l3_graph, level_idx=3)
                    del l3_graph  # Free memory
                    # Store mapping and processed graph
                    self.level_mappings.append(l2_l3_mapping)
                    level_graphs.append(l3_processed)
        
        # Step 5: Build unified graph connecting all processed levels
        unified_graph = self._build_unified_graph(level_graphs, self.level_mappings)
        del level_graphs  # Free memory
        # Step 6: Apply bidirectional refinement cycles
        refined_graph = self._apply_refinement_cycles(unified_graph, cycles)
        del unified_graph  # Free memory
        # Extract features for output
        if use_level_prediction:
            # Add level_to_token_projection if needed
            if not hasattr(self, 'level_to_token_projection'):
                self.level_to_token_projection = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                # Ensure it's on the same device
                self.level_to_token_projection = self.level_to_token_projection.to(refined_graph.x.device)
            
            # Use highest level features for prediction
            highest_level_idx = len(refined_graph.level_offsets) - 2
            highest_level_start = refined_graph.level_offsets[highest_level_idx]
            highest_level_end = refined_graph.level_offsets[highest_level_idx + 1]
            highest_level_features = refined_graph.x[highest_level_start:highest_level_end]
            
            # Project highest level features to token level
            global_context = self.level_to_token_projection(highest_level_features.mean(dim=0))
            global_context = global_context.unsqueeze(0).expand(seq_len, -1)
            
            # Combine with token features
            token_features = refined_graph.x[:seq_len] + global_context
        elif self.use_final_layer_for_prediction:
            # Use refined token-level features
            token_features = refined_graph.x[:seq_len]
            # Choose a specific level for prediction
            #token_features = refined_graph.x[refined_graph.level_offsets[1]:refined_graph.level_offsets[2]]

        else:
            # Use token-level features directly
            token_features = l0_processed.x

        token_features = token_features.view(batch_size, seq_len, self.hidden_dim)
        emit_token_features = bool(
            getattr(self, "return_token_features", False)
            if return_token_features is None
            else return_token_features
        )
        if emit_token_features:
            if return_hierarchical_features:
                hierarchical_features = []
                for i in range(len(refined_graph.level_offsets) - 1):
                    start = refined_graph.level_offsets[i]
                    end = refined_graph.level_offsets[i + 1]
                    hierarchical_features.append(refined_graph.x[:, start:end] if refined_graph.x.dim() == 3 else refined_graph.x[start:end])
                return token_features, hierarchical_features
            return token_features
        
        # Project to vocabulary
        logits = self.output_projection(token_features).view(batch_size, seq_len, -1)
        
        if return_hierarchical_features:
            # Extract features for each level
            hierarchical_features = []
            for i in range(len(refined_graph.level_offsets) - 1):
                start = refined_graph.level_offsets[i]
                end = refined_graph.level_offsets[i + 1]
                hierarchical_features.append(refined_graph.x[start:end])
            
            return logits, hierarchical_features

        return logits

    # ------------------------------------------------------------------
    # Incremental KV-cached decode (Tier 5). ADDITIVE + opt-in: the normal
    # generate() path is untouched and is the always-available fallback.
    #
    # Staged build:
    #   S1 (this): dedicated entry point + verify harness + tunable frontier
    #              width plumbing. Correct by construction (full forward/token).
    #   S2: per-(refinement layer, level) K/V cache populated on the prompt forward.
    #   S3: refinement "decode mode" that recomputes only the frontier node set
    #       (last `frontier_width` L0 + open coarse + parents) while reading cached
    #       K/V for settled neighbours -> the actual O(1)/token speedup.
    #   S4: frontier bookkeeping (settle closed windows). S5: graph-edge fallback.
    #
    # `frontier_width` is the tunable correctness/speed knob (user choice):
    #   None -> auto (= max local window + max compression, the measured drift span)
    #   0    -> approx (recompute only the new token; coarse treated as settled)
    #   big  -> exact (recompute enough recent context to match full forward in bf16)
    # ------------------------------------------------------------------
    def _gen_graph_meta(self, l0_len: int):
        """(level_sizes, level_offsets, node_level, node_ar_time) of the sequence-mode packed
        graph for l0_len tokens, computed directly from the window rule (parent ar_time =
        last child's ar_time; start = min(i*stride, n_child-1), end clamped like
        _xq_children_tables). Skeleton-free so the decode driver can build row maps before
        any forward has cached that length; asserted equal to the skeleton in the probe."""
        sizes = [int(s) for s in self._predict_level_sizes(int(l0_len))]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)
        node_level = torch.cat([torch.full((s,), lvl, dtype=torch.long) for lvl, s in enumerate(sizes)])
        ar_chunks = [torch.arange(sizes[0], dtype=torch.long)]
        child_t = ar_chunks[0]
        for lvl in range(1, len(sizes)):
            comp = int(self.compression_ratios[lvl - 1])
            stride = max(1, int(comp * (1.0 - self.overlap_ratios[lvl - 1])))
            n_child, n_parent = sizes[lvl - 1], sizes[lvl]
            starts = torch.clamp(torch.arange(n_parent, dtype=torch.long) * stride, max=max(0, n_child - 1))
            ends = torch.maximum(torch.clamp(starts + comp, max=n_child), starts + 1)
            parent_t = child_t[ends - 1]
            ar_chunks.append(parent_t)
            child_t = parent_t
        return sizes, offsets, node_level, torch.cat(ar_chunks)

    def _gen_kv_guard_reasons(self, batch_size: int) -> List[str]:
        """Config features the cached-tail decode does not cover. Non-empty -> the caller
        falls back to the full-rebuild loop (never silently wrong, just slower)."""
        reasons: List[str] = []
        if int(batch_size) != 1:
            reasons.append("batch_size>1")
        if str(getattr(self, "graph_geometry_mode", "sequence")).lower() != "sequence":
            reasons.append("grid2d geometry")
        if not bool(getattr(self, "enable_unified_skeleton_cache", False)):
            reasons.append("skeleton cache off")
        if str(getattr(self, "refinement_batch_mode", "")) != "true_batch_nozip":
            reasons.append("refinement_batch_mode != true_batch_nozip")
        if not bool(getattr(self, "drop_static_cross_level_edges", False)):
            reasons.append("static cross-level scatter edges on (cleaner mode only)")
        if bool(getattr(self, "hierarchical_query_descent_enable", False)):
            reasons.append("HQD on (data-dependent far edges)")
        if bool(getattr(self, "zip_enable", False)):
            reasons.append("zipper on")
        if bool(getattr(self, "xq_nominate_enable", False)):
            reasons.append("xq_nominate on")
        if bool(getattr(self, "local_pack_cross_level", False)):
            reasons.append("local_pack_cross_level on (packed tail decode not built)")
        if getattr(self, "token_unet", None) is not None:
            reasons.append("token_unet on (length-dependent stem)")
        if str(getattr(self, "autoenc_graph_mode", "off")).lower() not in ("off", "none", ""):
            reasons.append("autoenc graph nodes")
        if bool(getattr(self, "hier_copredict_l0", False)):
            reasons.append("hier_copredict_l0 on")
        if bool(getattr(self, "pinball_level_cycle_enable", False)):
            reasons.append("pinball_level_cycle on")
        return reasons

    def _generate_incremental(
        self,
        current_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        use_level_prediction: bool = False,
        num_cycles: Optional[int] = None,
        frontier_width: Optional[int] = None,
        verify: bool = False,
    ) -> torch.Tensor:
        """S3: cached-tail incremental decode ("reuse the previous graph").

        Causality makes every node with ar_time behind the frontier BIT-IDENTICAL across
        appends, so a full forward every REFILL_CHUNK tokens captures the packed state at
        every layer boundary (plus the post-episode state), and each decode step re-runs the
        forward only on a stride-aligned TAIL of the sequence, overwriting all settled rows
        from the cache at every boundary. The tail is sized to cover the widest backbone
        local-attention span, so everything a frontier row reads is either computed in-tail
        or cache-injected. The multirate episode additionally gets the cached pre-tail coarse
        rows as frozen extra memory (cross-query refiners, topk=0 long-range channel) and as
        causal prefixes (upper/top level refiners); settled rows are pre-frozen there. The
        single approximation: pre-tail rows hold their episode-ENTRY state during episode
        cycles >= 2 instead of their (sub-tau, soon-frozen) mid-episode updates — measured by
        `verify`, and repaired at the next boundary by the post-episode hook.
        """
        device = current_ids.device
        cycles = num_cycles if num_cycles is not None else self.refinement_cycles
        B = int(current_ids.size(0))

        reasons = self._gen_kv_guard_reasons(B)
        if reasons:
            raise RuntimeError("kv-decode unsupported here: " + "; ".join(reasons))

        import contextlib
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        use_amp = device.type == "cuda"
        eos_token_id = min(self.vocab_size - 1, int(getattr(self, "eos_token_id", self.vocab_size - 1)))
        pad_id = int(getattr(self, "pad_token_id", None) or getattr(self, "mask_token_id", 0) or 0)

        # ---- geometry ----
        comps = [int(c) for c in (getattr(self, "compression_ratios", []) or [])]
        ovs = list(getattr(self, "overlap_ratios", []) or [])
        stride_l0 = [1]
        for i, c in enumerate(comps):
            ov = float(ovs[i]) if i < len(ovs) else 0.0
            stride_l0.append(stride_l0[-1] * max(1, int(c * (1.0 - ov))))
        lookahead = int(self._gen_frontier_lookahead())
        REFILL_CHUNK = 256
        ALIGN = REFILL_CHUNK  # multiple of every stride_l0 entry (powers chain up)
        for s in stride_l0:
            if ALIGN % s != 0:
                ALIGN = s * ((ALIGN + s - 1) // s)

        # Widest backbone within-level local-attention span in L0 units: reads the tail
        # must contain because the backbone gets no memory injection (unlike the episode).
        span = int(getattr(self, "l0_local_window", 128) or 128)
        for lvl, cfg in (getattr(self, "local_attn_config", {}) or {}).items():
            if int(lvl) < len(stride_l0):
                span = max(span, int(cfg.get("window", 0)) * stride_l0[int(lvl)])
        if frontier_width is not None and int(frontier_width) > 0:
            span = int(frontier_width)  # user override: smaller = faster + more approximate
        elif frontier_width is not None and int(frontier_width) == 0:
            span = 512
        self._kv_frontier_width = int(span)

        def _align_up(v: int, a: int) -> int:
            return ((int(v) + a - 1) // a) * a

        # ---- per-refill state (rebuilt by _refill) ----
        state: Dict[str, object] = {}
        n_layer_entries = 0

        def _capture_layer_hook(x, layer_idx, entry_idx):
            state["layer_cache"][int(entry_idx)] = x.detach().clone()
            state["last_entry"] = int(entry_idx)
            return x

        def _capture_mr_hook(x):
            # fires right after a multirate episode; the episode's ENTRY state is the
            # per-layer capture that ran immediately before it (hook precedes the episode).
            state["mr_cache"].append(x.detach().clone())
            state["mr_after_entry"].append(int(state.get("last_entry", -1)))
            return x

        def _capture_mr_cycle_hook(x, cycle_idx):
            # fires after every episode CYCLE; ordinal = episodes completed so far (the
            # episode-end hook above increments the list length AFTER its cycles ran)
            state["mr_cycles"].setdefault(len(state["mr_after_entry"]), []).append(x.detach().clone())
            return x

        def _forward_padded(ids_full: torch.Tensor, proj_idx: int):
            ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                return self.forward(
                    ids_full,
                    num_cycles=cycles,
                    use_level_prediction=use_level_prediction,
                    logits_last_index=int(proj_idx),
                )

        def _clear_hooks():
            self._gen_layer_hook = None
            self._gen_mr_hook = None
            self._gen_mr_cycle_hook = None
            self._gen_mr_prefrozen = None
            self._gen_mr_prefrozen_keep = None
            self._gen_mr_extra_memory = None
            self._gen_mr_level_prefix = None

        def _prewarm_skeleton(l0_len: int):
            sizes = self._predict_level_sizes(int(l0_len))
            self._runtime_l0_grid_shape = None
            if self._unified_skeleton_cache.get(self._unified_cache_key(sizes), None) is None:
                dummy = torch.full((B, int(l0_len)), pad_id, dtype=current_ids.dtype, device=device)
                call_save = int(getattr(self, "_pinball_multirate_call_count", 0))
                _forward_padded(dummy, 0)
                self._pinball_multirate_call_count = call_save  # side-effect-free prewarm

        def _refill(ids_now: torch.Tensor):
            """Full padded capture forward at the current length; rebuilds bucket, caches,
            row maps and per-step hook tensors. Returns this step's next-token logits."""
            L_now = int(ids_now.size(1))
            bucket = min(int(self.max_seq_len), _align_up(L_now + lookahead + REFILL_CHUNK, ALIGN))
            # Anchor the tail at the FRONTIER, not the bucket end: every position the frontier
            # reads directly (widest local-attention span) must exist as a tail row, and the
            # tail extends to the bucket end so the frontier stays inside it for a full refill
            # chunk of growth. Alignment keeps every coarse level's windows on full-graph
            # window boundaries (t0 divisible by all level strides).
            t0 = max(0, L_now - span - 64)
            t0 = (t0 // ALIGN) * ALIGN
            tail_len = bucket - t0

            _prewarm_skeleton(bucket)
            if tail_len != bucket:
                _prewarm_skeleton(tail_len)

            padded = torch.full((B, bucket), pad_id, dtype=ids_now.dtype, device=device)
            padded[:, :L_now] = ids_now

            _, off_f, _, ar_f = self._gen_graph_meta(bucket)
            sizes_t, off_t, nl_t, ar_t = self._gen_graph_meta(tail_len)
            ar_f = ar_f.to(device)
            ar_t = ar_t.to(device)
            nl_t = nl_t.to(device)

            # tail row -> full row map (stride-aligned slice of every level)
            maps = []
            for lvl in range(len(sizes_t)):
                shift = t0 // stride_l0[lvl] if lvl < len(stride_l0) else 0
                n_t_lvl = off_t[lvl + 1] - off_t[lvl]
                maps.append(torch.arange(n_t_lvl, dtype=torch.long, device=device) + int(off_f[lvl] + shift))
            tail_to_full = torch.cat(maps)

            state.clear()
            state["layer_cache"] = {}
            state["mr_cache"] = []
            state["mr_after_entry"] = []
            state["mr_cycles"] = {}
            self._gen_layer_hook = _capture_layer_hook
            self._gen_mr_hook = _capture_mr_hook
            self._gen_mr_cycle_hook = _capture_mr_cycle_hook
            try:
                logits = _forward_padded(padded, L_now - 1)
            finally:
                _clear_hooks()

            # settled rows: everything whose receptive window closed before this refill.
            # Static until the next refill; younger rows are recomputed in-tail each step.
            settle_t = L_now - 1
            inject_tail = ((ar_t + t0) <= settle_t).view(1, -1, 1)  # [1, N_t, 1]
            prefrozen = inject_tail.view(1, -1).expand(B, -1).contiguous()

            # pre-gather every cache entry to tail rows (static per refill)
            layer_tail = {e: t.index_select(1, tail_to_full) for e, t in state["layer_cache"].items()}
            mr_tail = [t.index_select(1, tail_to_full) for t in state["mr_cache"]]
            mr_cycles_tail = {
                o: [t.index_select(1, tail_to_full) for t in cyc]
                for o, cyc in state["mr_cycles"].items()
            }

            # episode context: pre-tail coarse rows, PER CYCLE (they evolve inside the
            # episode; the block at cycle k reads their state after cycle k-1, entry state
            # at k=0). extra memory for the cross-attention refiners (memory_level rows),
            # causal prefixes for the within-level upper/top refiners.
            mem_levels = set()
            for r in getattr(self, "pinball_cross_query_refiners", []) or []:
                mem_levels.add(int(r.memory_level))
            if getattr(self, "pinball_upper_cross_refiner", None) is not None:
                mem_levels.add(int(self.pinball_upper_cross_refiner.memory_level))
            prefix_specs: Dict[int, int] = {}
            for ref_name, win_attr in (("pinball_upper_refiner", "pinball_upper_refine_window"),
                                       ("pinball_top_refiner", "pinball_top_refine_window")):
                ref = getattr(self, ref_name, None)
                if ref is not None:
                    w = int(getattr(self, win_attr, 0) or 0)
                    for lvl in getattr(ref, "levels", []):
                        want = w * max(1, int(getattr(ref, "max_steps", 1))) if w > 0 else 1 << 30
                        prefix_specs[int(lvl)] = max(prefix_specs.get(int(lvl), 0), want)

            def _episode_ctx_from(src):
                extra_l, prefix_l = {}, {}
                if src is not None:
                    for lvl in range(1, len(sizes_t)):
                        shift = t0 // stride_l0[lvl]
                        if shift <= 0:
                            continue
                        f_lo, f_hi = int(off_f[lvl]), int(off_f[lvl] + shift)
                        if lvl in mem_levels:
                            extra_l[lvl] = (src[:, f_lo:f_hi, :], ar_f[f_lo:f_hi] - t0)
                        if lvl in prefix_specs:
                            keep = min(int(prefix_specs[lvl]), f_hi - f_lo)
                            if keep > 0:
                                prefix_l[lvl] = src[:, f_hi - keep : f_hi, :]
                return (extra_l or None, prefix_l or None)

            # mr_ctx_by_cycle[episode][k] = (extra_memory, level_prefix) the block reads at
            # cycle k: entry state for k=0, then the captured state after cycle k-1.
            mr_ctx_by_cycle = []
            for o, after_entry in enumerate(state["mr_after_entry"]):
                srcs = [state["layer_cache"].get(int(after_entry), None)] + state["mr_cycles"].get(o, [])
                mr_ctx_by_cycle.append([_episode_ctx_from(s) for s in srcs])

            # per-level active-row counts for the episode's pre-freeze gather (one host
            # sync per refill, none per step)
            keep_levels = sorted({int(r.query_level) for r in (getattr(self, "pinball_cross_query_refiners", []) or [])})
            prefrozen_keep = None
            if keep_levels:
                keep_all = ~prefrozen
                masks = {l: keep_all[:, (nl_t == l).nonzero(as_tuple=False).view(-1)] for l in keep_levels}
                counts = {l: int(m.sum(dim=1).max().item()) for l, m in masks.items()}
                prefrozen_keep = {
                    l: (masks[l], counts[l]) for l in keep_levels
                    if 0 < counts[l] <= int(0.75 * masks[l].size(1))
                }
                prefrozen_keep = prefrozen_keep or None

            state.update(
                padded=padded, bucket=bucket, t0=t0, tail_len=tail_len, refill_len=L_now,
                layer_tail=layer_tail, mr_tail=mr_tail, mr_cycles_tail=mr_cycles_tail,
                inject_tail=inject_tail, prefrozen=prefrozen, prefrozen_keep=prefrozen_keep,
                mr_ctx_by_cycle=mr_ctx_by_cycle,
            )
            return logits

        def _inject_layer_hook(x, layer_idx, entry_idx):
            cached = state["layer_tail"].get(int(entry_idx), None)
            if cached is None:
                return x
            return torch.where(state["inject_tail"], cached.to(dtype=x.dtype), x)

        def _set_episode_ctx(ordinal: int, cycle: int):
            ctx = state["mr_ctx_by_cycle"]
            if 0 <= ordinal < len(ctx) and ctx[ordinal]:
                extra, prefix = ctx[ordinal][min(int(cycle), len(ctx[ordinal]) - 1)]
                self._gen_mr_extra_memory = extra
                self._gen_mr_level_prefix = prefix

        def _inject_mr_cycle_hook(x, cycle_idx):
            o = int(state.get("mr_ordinal", 0))
            cyc = state["mr_cycles_tail"].get(o, [])
            _set_episode_ctx(o, int(cycle_idx) + 1)  # what the block reads NEXT cycle
            if not cyc:
                return x
            k = min(int(cycle_idx), len(cyc) - 1)
            return torch.where(state["inject_tail"], cyc[k].to(dtype=x.dtype), x)

        def _inject_mr_hook(x):
            i = int(state.get("mr_ordinal", 0))
            state["mr_ordinal"] = i + 1
            mr_tail = state["mr_tail"]
            _set_episode_ctx(i + 1, 0)  # arm the NEXT episode (multi-episode schedules)
            if i >= len(mr_tail):
                return x
            return torch.where(state["inject_tail"], mr_tail[i].to(dtype=x.dtype), x)

        def _decode_step(t_len: int):
            """Next-token logits for position t_len given padded[:, :t_len] real tokens."""
            t0 = int(state["t0"])
            tail_ids = state["padded"][:, t0 : t0 + int(state["tail_len"])]
            state["mr_ordinal"] = 0
            if not bool(getattr(self, "_kv_disable_inject", False)):  # diagnostic ablation:
                # True = raw sliding-window decode (context beyond the tail is DROPPED).
                self._gen_layer_hook = _inject_layer_hook
                self._gen_mr_hook = _inject_mr_hook
                self._gen_mr_cycle_hook = _inject_mr_cycle_hook
                self._gen_mr_prefrozen = state["prefrozen"]
                self._gen_mr_prefrozen_keep = state["prefrozen_keep"]
                _set_episode_ctx(0, 0)
            try:
                return _forward_padded(tail_ids, t_len - 1 - t0)
            finally:
                _clear_hooks()

        # ---- decode loop ----
        max_verify_err = 0.0
        was_training = self.training
        self.eval()
        try:
            logits = None
            for _ in range(int(max_new_tokens)):
                t_len = int(current_ids.size(1))
                if t_len >= int(self.max_seq_len):
                    break
                need_refill = (
                    "padded" not in state
                    or t_len + lookahead >= int(state["t0"]) + int(state["tail_len"])
                    or t_len + lookahead >= int(state["bucket"])
                    or t_len - int(state["refill_len"]) >= REFILL_CHUNK
                )
                if need_refill:
                    logits = _refill(current_ids)
                else:
                    state["padded"][:, t_len - 1] = current_ids[:, t_len - 1]
                    logits = _decode_step(t_len)
                    if verify:
                        call_save = int(getattr(self, "_pinball_multirate_call_count", 0))
                        ref = _forward_padded(state["padded"][:, : int(state["bucket"])], t_len - 1)
                        self._pinball_multirate_call_count = call_save  # side-effect-free reference
                        err = float((logits[:, -1, :].float() - ref[:, -1, :].float()).abs().max().item())
                        max_verify_err = max(max_verify_err, err)
                next_token = self._safe_sampling(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    current_ids=current_ids,
                    do_sample=do_sample,
                )
                current_ids = torch.cat([current_ids, next_token], dim=1)
                if next_token.item() == eos_token_id:
                    break
        finally:
            _clear_hooks()
            if was_training:
                self.train()

        if verify:
            logger.info("[KV-CACHE] verify_incremental max|logit err| = %.3e (cached-tail vs full forward)", max_verify_err)
            self._kv_last_verify_err = float(max_verify_err)
        return current_ids

    def _gen_frontier_lookahead(self) -> int:
        """L0 span of a top-level coarse node = how many tokens ahead a coarse window that
        overlaps the frontier needs before it CLOSES. Padding the AR prefix by this much lets
        the frontier token see exactly the closed-window coarse context it saw in the training
        forward (open windows get causal-cut instead of truncated-and-read). Cached."""
        cached = getattr(self, "_gen_frontier_lookahead_cached", None)
        if cached is not None:
            return int(cached)
        comps = list(getattr(self, "compression_ratios", []) or [])
        ovs = list(getattr(self, "overlap_ratios", []) or [])
        stride_l0, window_l0 = 1, 1
        for i, comp in enumerate(comps):
            ov = float(ovs[i]) if i < len(ovs) else 0.0
            node_stride = max(1, int(int(comp) * (1.0 - ov)))
            window_l0 = (int(comp) - 1) * stride_l0 + window_l0
            stride_l0 = node_stride * stride_l0
        look = max(0, int(window_l0) - 1)
        self._gen_frontier_lookahead_cached = look
        return look

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        use_level_prediction: bool = False,
        use_direct_prediction: bool = False,
        rebuild_graph: bool = False,  # New option to rebuild the graph for each token
        num_cycles: int = None,  # Allow overriding cycles
        use_kv_cache: bool = True,         # cached-tail incremental decode (2.6-3.8x at 2k-8k,
                                           # verified exact-mod-bf16); guarded configs fall back
                                           # to the rebuild loop automatically
        kv_frontier_width: Optional[int] = None,  # None = auto (exact span); smaller = faster but
                                           # approximate (truncates coarse local-attn reach)
        verify_incremental: bool = False,  # log max |kv logits - full-forward logits| (debug)
    ) -> torch.Tensor:
        """
        Generate text using hierarchical flow with multiple generation options.
        Improved with robust error handling and fallbacks.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample from the distribution
            top_k: Top-k filtering parameter
            top_p: Top-p (nucleus) filtering parameter
            repetition_penalty: Penalty for repeating tokens
            use_level_prediction: Whether to use level projection for prediction
            use_direct_prediction: Whether to use direct next token prediction
            rebuild_graph: Whether to rebuild the graph for each token (like training)
            num_cycles: Optional override for refinement cycles
            
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len + new_tokens]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        cycles = num_cycles if num_cycles is not None else self.refinement_cycles
        
        # Start with input sequence
        current_ids = input_ids.clone()
        
        # Initialize level_mappings if needed
        if not hasattr(self, 'level_mappings') or self.level_mappings is None:
            self.level_mappings = []

        # CASE 0: Incremental KV-cached decode (opt-in, additive). Falls back to the normal
        # rebuild loop on any error so the default path is never compromised.
        if use_kv_cache:
            try:
                return self._generate_incremental(
                    current_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_level_prediction=use_level_prediction,
                    num_cycles=cycles,
                    frontier_width=kv_frontier_width,
                    verify=verify_incremental,
                )
            except Exception as e:
                logger.warning("[KV-CACHE] incremental decode failed (%s); falling back to full rebuild.", e)
                rebuild_graph = True

        # CASE 1: Rebuild Graph approach
        if rebuild_graph:
            try:
                # Rebuild graph approach - most similar to training
                # Frontier-consistent decode: a raw prefix builds TRUNCATED coarse windows, so the
                # frontier token reads its own open parent — but in training that parent's full
                # window extends past it and is causal-cut. Padding the prefix forward by the top-
                # level window span closes those windows (with causally-invisible pad tokens), so
                # the frontier sees the SAME closed-window context as training. Verified bit-exact.
                frontier_consistent = bool(getattr(self, "gen_frontier_consistent", True))
                pad_id = int(getattr(self, "pad_token_id", None) or getattr(self, "mask_token_id", 0) or 0)
                lookahead = self._gen_frontier_lookahead() if frontier_consistent else 0
                # bf16 autocast: flash local attention rejects fp32 — without this, flash
                # configs silently fell through to the emergency generation path.
                import contextlib
                _gen_amp = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else contextlib.nullcontext()
                )
                for _ in range(max_length):
                    # Check if we've reached maximum sequence length
                    if current_ids.size(1) >= self.max_seq_len:
                        break

                    # Get next token logits by calling forward, completely rebuilding the graph
                    with torch.no_grad(), _gen_amp:
                        cur_len = int(current_ids.size(1))
                        pad_n = min(lookahead, int(self.max_seq_len) - cur_len)
                        if frontier_consistent and pad_n > 0:
                            padded = torch.cat(
                                [current_ids, torch.full((current_ids.size(0), pad_n), pad_id,
                                                         dtype=current_ids.dtype, device=current_ids.device)],
                                dim=1)
                            logits = self.forward(
                                padded,
                                num_cycles=cycles,
                                use_level_prediction=use_level_prediction,
                                logits_last_index=cur_len - 1,  # the true frontier, not the padding
                            )
                        else:
                            # Use the forward method directly - like in training
                            logits = self.forward(
                                current_ids,
                                num_cycles=cycles,
                                use_level_prediction=use_level_prediction,
                                logits_last_only=True,
                            )
                        # Get logits for the last token (projection already returned [B,1,vocab])
                        next_token_logits = logits[:, -1, :]
                    
                    # Use safe sampling method
                    next_token = self._safe_sampling(
                        next_token_logits, 
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        current_ids=current_ids,
                        do_sample=do_sample
                    )
                    
                    # Append next token to the sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    
                    # Check for EOS token (use safe comparison)
                    eos_token_id = min(self.vocab_size - 1, getattr(self, 'eos_token_id', self.vocab_size - 1))
                    if next_token.item() == eos_token_id:
                        break
                
                return current_ids
            except Exception as e:
                print(f"Error in rebuild generation: {str(e)}. Falling back to direct prediction.")
                use_direct_prediction = True
        
        # CASE 2: Direct prediction approach     
        if use_direct_prediction:
            try:
                # Direct next token prediction approach (faster but less context-aware)
                for _ in range(max_length):
                    # Check if we've reached maximum sequence length
                    if current_ids.size(1) >= self.max_seq_len:
                        break
                    
                    # Get next token logits from forward pass
                    with torch.no_grad():
                        logits = self.forward(
                            current_ids,
                            num_cycles=cycles,
                            use_level_prediction=use_level_prediction,
                            logits_last_only=True,
                        )
                        next_token_logits = logits[:, -1, :]
                    
                    # Use safe sampling method
                    next_token = self._safe_sampling(
                        next_token_logits, 
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        current_ids=current_ids,
                        do_sample=do_sample
                    )
                    
                    # Append next token to the sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    
                    # Check for EOS token (use safe comparison)
                    eos_token_id = min(self.vocab_size - 1, getattr(self, 'eos_token_id', self.vocab_size - 1))
                    if next_token.item() == eos_token_id:
                        break
                
                return current_ids
            except Exception as e:
                print(f"Error in direct generation: {str(e)}. Trying graph generation with safer approach.")
        
        # CASE 3: Graph approach (original) - with extremely defensive programming
        try:
            # Simple graph approach that doesn't rely on complex graph manipulation
            for _ in range(max_length):
                # Check if we've reached maximum sequence length
                if current_ids.size(1) >= self.max_seq_len:
                    break
                
                # Get next token logits by rebuilding the graph and forward pass
                with torch.no_grad():
                    # Process the current sequence
                    logits = self.forward(
                        current_ids,
                        num_cycles=cycles,
                        use_level_prediction=use_level_prediction,
                        logits_last_only=True,
                    )
                    # Get logits for the last token
                    next_token_logits = logits[:, -1, :]
                
                # Use safe sampling method
                next_token = self._safe_sampling(
                    next_token_logits, 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    current_ids=current_ids,
                    do_sample=do_sample
                )
                
                # Append next token to the sequence - always by rebuilding
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Check for EOS token (use safe comparison)
                eos_token_id = min(self.vocab_size - 1, getattr(self, 'eos_token_id', self.vocab_size - 1))
                if next_token.item() == eos_token_id:
                    break
            
            return current_ids
                
        except Exception as e:
            print(f"Error in all generation methods: {str(e)}. Using emergency direct generation.")
            # Emergency fallback - simplest possible generation
            return self._emergency_generation(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p
            )

    def _emergency_generation(self, input_ids, max_length=100, temperature=1.0, do_sample=True, top_k=50, top_p=0.9):
        """Emergency fallback generation method that uses minimal functionality."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        current_ids = input_ids.clone()
        
        # Simplified token-by-token generation
        for _ in range(max_length):
            if current_ids.size(1) >= self.max_seq_len:
                break
                
            # Get embeddings for current sequence
            with torch.no_grad():
                # Get token embeddings
                embeddings = self.token_embedding(current_ids)
                
                # Forward through output projection immediately
                logits = self.output_projection(embeddings[:, -1])
                
                # Apply safe sampling with minimal processing
                try:
                    # Apply temperature
                    next_token_logits = logits / max(temperature, 1e-8)
                    
                    # Basic sampling
                    if do_sample:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Safety check for token id
                    if next_token.item() >= self.vocab_size:
                        next_token = torch.tensor([[0]], device=device)  # Use token 0 as fallback
                except Exception:
                    # Ultimate fallback - just use token 0
                    next_token = torch.tensor([[0]], device=device)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop at EOS or last token in vocab (safe)
                eos_token_id = min(self.vocab_size - 1, getattr(self, 'eos_token_id', self.vocab_size - 1))
                if next_token.item() == eos_token_id:
                    break
                    
        return current_ids
