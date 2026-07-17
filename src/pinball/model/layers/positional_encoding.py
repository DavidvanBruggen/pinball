# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math


logger = logging.getLogger(__name__)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding with optional axial 2D support.

    Defaults to auto mode:
    - 1D RoPE for scalar positions
    - axial 2D RoPE for (row, col) positions
    """

    def __init__(self, dim, max_seq_len=131072, rope_mode: str = "auto", axial_split_ratio: float = 0.5):
        super().__init__()
        self.dim = dim if dim % 2 == 0 else dim - 1
        if self.dim != dim:
             print(f"RoPE Warning: dim reduced from {dim} to {self.dim} to be even.")
        self.max_seq_len = max_seq_len
        self.rope_mode = str(rope_mode).lower()
        if self.rope_mode not in {"auto", "1d", "2d", "2d_axial", "axial", "grid2d"}:
            self.rope_mode = "auto"
        self.axial_split_ratio = float(axial_split_ratio)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)

        self.cos_cached = None
        self.sin_cached = None
        self.seq_len_cached = None
        self._rope_runtime_logged_keys = set()

    def _precompute_sincos(self, max_pos, device):
        safe_max_pos = min(max_pos, self.max_seq_len)
        inv_freq_bf = self.inv_freq.to(device).unsqueeze(0)
        position = torch.arange(safe_max_pos, device=device).unsqueeze(1)
        position_enc = torch.matmul(position.float(), inv_freq_bf)
        self.cos_cached = torch.cos(position_enc)
        self.sin_cached = torch.sin(position_enc)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _apply_rotary_pos_emb_1d(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        num_nodes, num_heads, head_dim = x.shape
        if torch.compiler.is_compiling():
            # Traceable path for torch.compile: the cached-table sizing below needs
            # positions.max().item(), which graph-breaks (GPU->CPU sync) inside every
            # compiled layer. Compute cos/sin directly from the positions instead — the
            # same float32 product the table build uses, so the values are identical;
            # inductor fuses it with the rotation. Clamp mirrors the table's max_seq_len cap.
            pos_f = positions.clamp(0, self.max_seq_len - 1).float()
            enc = pos_f.unsqueeze(1) * self.inv_freq.to(device=x.device).unsqueeze(0)
            cos = torch.cos(enc).to(dtype=x.dtype).unsqueeze(1)
            sin = torch.sin(enc).to(dtype=x.dtype).unsqueeze(1)
        else:
            max_pos_needed = int(positions.max().item()) + 1
            if (self.cos_cached is None or
                max_pos_needed > self.cos_cached.size(0) or
                self.cos_cached.device != x.device):
                self._precompute_sincos(max_pos_needed, x.device)

            cached_len = self.cos_cached.size(0)
            clamped_positions = positions.clamp(0, cached_len - 1)
            cos = self.cos_cached[clamped_positions].to(dtype=x.dtype).unsqueeze(1)
            sin = self.sin_cached[clamped_positions].to(dtype=x.dtype).unsqueeze(1)

        dim_rotary = self.dim
        x_rotated = x.clone()
        x_even = x[..., 0:dim_rotary:2]
        x_odd = x[..., 1:dim_rotary:2]
        x_rotated[..., 0:dim_rotary:2] = x_even * cos - x_odd * sin
        x_rotated[..., 1:dim_rotary:2] = x_odd * cos + x_even * sin
        if dim_rotary < head_dim:
            x_rotated[..., dim_rotary:] = x[..., dim_rotary:]
        return x_rotated

    def _apply_rotary_pos_emb_2d(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self._apply_rotary_pos_emb_nd(x, positions)

    def _apply_rotary_pos_emb_nd(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() == 3:
            if positions.size(-1) < 1:
                return self._apply_rotary_pos_emb_1d(x, positions.reshape(-1).long())
            positions = positions.reshape(-1, positions.size(-1))

        if positions.dim() != 2:
            return self._apply_rotary_pos_emb_1d(x, positions.reshape(-1).long())

        num_axes = int(positions.size(-1))
        if num_axes <= 1:
            return self._apply_rotary_pos_emb_1d(x, positions.reshape(-1).long())

        num_nodes, num_heads, head_dim = x.shape
        rotary_dim = head_dim - (head_dim % (2 * num_axes))
        if rotary_dim < 2 * num_axes:
            raise ValueError(
                f"Multi-axis RoPE needs at least {2 * num_axes} rotary dims, got head_dim={head_dim}"
            )

        axis_dim = rotary_dim // num_axes
        if axis_dim % 2 != 0:
            axis_dim -= 1
        if axis_dim < 2:
            raise ValueError(
                f"Multi-axis RoPE axis dimension became too small for {num_axes} axes (head_dim={head_dim})"
            )

        x_rotated = x.clone()
        inv_freq_axis = 1.0 / (
            10000 ** (torch.arange(0, axis_dim, 2, device=x.device, dtype=torch.float32) / axis_dim)
        )

        for axis_idx in range(num_axes):
            axis_pos = positions[:, axis_idx].to(device=x.device, dtype=torch.float32)
            axis_enc = torch.matmul(axis_pos.unsqueeze(1), inv_freq_axis.unsqueeze(0))
            axis_cos = torch.cos(axis_enc).to(dtype=x.dtype).unsqueeze(1)
            axis_sin = torch.sin(axis_enc).to(dtype=x.dtype).unsqueeze(1)

            axis_start = axis_idx * axis_dim
            axis_end = axis_start + axis_dim
            axis_slice = x[..., axis_start:axis_end]
            axis_even = axis_slice[..., 0:axis_dim:2]
            axis_odd = axis_slice[..., 1:axis_dim:2]
            axis_rot = axis_slice.clone()
            axis_rot[..., 0:axis_dim:2] = axis_even * axis_cos - axis_odd * axis_sin
            axis_rot[..., 1:axis_dim:2] = axis_odd * axis_cos + axis_even * axis_sin
            x_rotated[..., axis_start:axis_end] = axis_rot

        if rotary_dim < head_dim:
            x_rotated[..., rotary_dim:] = x[..., rotary_dim:]
        return x_rotated

    def _apply_rotary_pos_emb(self, x: torch.Tensor):
        positions = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
        return self._apply_rotary_pos_emb_1d(x, positions)

    def forward(self, x, seq_len=None):
        try:
            mode = str(getattr(self, "rope_mode", "auto")).lower()
            if mode not in {"auto", "1d"}:
                raise ValueError(
                    "RotaryPositionalEncoding.forward only supports 1D embeddings; use apply_rotary_pos_emb for multi-axis positions."
                )

            squeeze_batch = False
            if x.dim() == 2:
                x = x.unsqueeze(0)
                squeeze_batch = True
            elif x.dim() != 3:
                raise ValueError(f"RotaryPositionalEncoding.forward expects [B,T,D] or [T,D], got {tuple(x.shape)}")

            seq_len = x.shape[1] if seq_len is None else int(seq_len)
            seq_len = min(seq_len, self.max_seq_len)
            if seq_len != x.shape[1]:
                if seq_len > x.shape[1]:
                    raise ValueError(
                        f"seq_len={seq_len} exceeds input sequence length {x.shape[1]}"
                    )
                x = x[:, :seq_len, :]

            batch_size, token_len, hidden_dim = x.shape
            positions = torch.arange(token_len, device=x.device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(batch_size, token_len).reshape(-1)
            x_flat = x.reshape(batch_size * token_len, 1, hidden_dim)
            x_rot = self._apply_rotary_pos_emb_1d(x_flat, positions).reshape(batch_size, token_len, hidden_dim)
            return x_rot.squeeze(0) if squeeze_batch else x_rot
        except Exception as e:
            logger.exception("Error in positional encoding")
            raise

    def apply_rotary_pos_emb(self, x, positions):
        try:
            num_nodes, num_heads, head_dim = x.shape
            if positions is None:
                return x

            if not torch.is_tensor(positions):
                positions = torch.as_tensor(positions, device=x.device)
            else:
                positions = positions.to(device=x.device)

            if str(getattr(self, "rope_mode", "auto")).lower() == "1d" and positions.dim() >= 2 and int(positions.size(-1)) > 1:
                raise ValueError(
                    f"rope_mode='1d' received multi-axis positions with shape {tuple(positions.shape)}"
                )

            grid_shapes = getattr(self, "local_attn_runtime_level_grid_shapes", {})
            try:
                grid_key = tuple(
                    sorted(
                        (int(lvl), int(shape[0]), int(shape[1]))
                        for lvl, shape in grid_shapes.items()
                        if shape is not None and len(shape) == 2
                    )
                )
            except Exception:
                grid_key = ()
            axes = int(positions.size(-1)) if positions.dim() >= 2 else 1
            use_nd_rope = positions.dim() >= 2 and axes >= 2 and self.rope_mode != "1d"
            rope_log_key = (
                f"{axes}d" if use_nd_rope else "1d",
                str(self.rope_mode),
                tuple(int(v) for v in positions.shape),
                grid_key,
            )
            if rope_log_key not in self._rope_runtime_logged_keys:
                if use_nd_rope:
                    logger.debug(
                        "[HMP:ROPE] using %dD axial RoPE mode=%s positions_shape=%s grid_shapes=%s",
                        int(axes),
                        str(self.rope_mode),
                        tuple(int(v) for v in positions.shape),
                        grid_shapes,
                    )
                else:
                    logger.debug(
                        "[HMP:ROPE] using 1D RoPE mode=%s positions_shape=%s grid_shapes=%s",
                        str(self.rope_mode),
                        tuple(int(v) for v in positions.shape),
                        grid_shapes,
                    )
                self._rope_runtime_logged_keys.add(rope_log_key)

            if positions.dim() >= 2 and positions.size(-1) >= 2 and self.rope_mode != "1d":
                if positions.size(-1) == 2:
                    return self._apply_rotary_pos_emb_2d(x, positions)
                return self._apply_rotary_pos_emb_nd(x, positions)

            if positions.dim() > 1:
                positions = positions.reshape(-1)
            return self._apply_rotary_pos_emb_1d(x, positions.long())
        except Exception as e:
            logger.exception(
                "Error applying RoPE. Shapes - x: %s, positions: %s",
                tuple(x.shape),
                tuple(positions.shape) if torch.is_tensor(positions) else getattr(positions, "shape", None),
            )
            raise


class LagrangianPositionalEncoding(MessagePassing):
    """
    Structure-aware positional encoding based on relative positions in the graph.
    
    This encoding captures the structural information of nodes within the graph,
    complementing sequence-based positional encodings.
    """
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')
        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index, pos=None):
        """
        Apply Lagrangian positional encoding.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            pos: Node positions (if None, use x as positions)
            
        Returns:
            encoded_x: Position-encoded features
        """
        pos = x if pos is None else pos
        return self.propagate(edge_index, x=x, pos=pos)
    
    def message(self, x_j, pos_i, pos_j):
        """
        Compute messages based on relative positions.
        
        Args:
            x_j: Source node features
            pos_i: Target node positions
            pos_j: Source node positions
            
        Returns:
            messages: Position-aware messages
        """
        # Compute relative position encoding
        rel_pos = pos_j - pos_i
        pos_encoding = self.pos_mlp(rel_pos)
        
        # Add position encoding to node features
        return x_j + pos_encoding
