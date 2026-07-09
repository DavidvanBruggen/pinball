# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
import time
import math
from dataclasses import dataclass

from typing import Optional, List, Tuple, Dict, Union, Any
from transformers import PreTrainedTokenizerBase

#from .hierarchical_flow_gat_cached import HierarchicalFlowGAT
from .hierarchical_flow_gat_cached_batch import HierarchicalFlowGAT
from .layers.hierarchical_message_passing import HierarchicalTransformerLayer
from .layers.normalization import make_norm
from .batched_layer_executor import BatchedLayerExecutor

logger = logging.getLogger(__name__)


@dataclass
class RecurrentPinballState:
    """Mutable graph state for recurrent Pinball inference."""

    x: torch.Tensor
    token_ids: torch.Tensor
    edge_index: torch.Tensor
    edge_type: Optional[torch.Tensor]
    node_level: torch.Tensor
    level_offsets: torch.Tensor
    node_ar_time: torch.Tensor
    node_pos_local: Optional[torch.Tensor]
    level_grid_shapes: Optional[List[Any]]
    time: int
    active_len: int
    current_l0_idx: int

def safe_sample_from_logits(
        logits: torch.Tensor,                # Input logits (should be 1D: [vocab_size])
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        previous_ids: Optional[torch.Tensor] = None,
        fallback_token_id: int = 0,
        debug_id: str = ""
    ) -> torch.Tensor:                      # Returns a 0D tensor (scalar)
    """
    Safely samples a token ID from 1D logits, applying penalty, temperature,
    top-k, top-p. Includes clamping for stability. Returns 0D tensor.
    """
    try:
        if logits.dim() != 1:
             # Simplified check/fix for common case from generation loop
             if logits.dim() == 2 and logits.size(0) == 1: logits = logits.squeeze(0)
             elif logits.numel() == logits.size(-1): logits = logits.flatten()
             else: raise ValueError(f"[{debug_id}] Logits must be 1D, got shape {logits.shape}")
        vocab_size = logits.size(0)

        # --- Use float32 for stability during manipulation ---
        original_dtype = logits.dtype
        logits = logits.to(torch.float32).clone() # Work on float32 copy

        # Guard: check initial finite state
        if not torch.isfinite(logits).all():
            logger.warning(f"⚠️ [{debug_id}] Initial logits non-finite. Using fallback.")
            return torch.tensor(fallback_token_id, device=logits.device, dtype=torch.long)

        # --- 1. Apply Repetition Penalty ---
        if repetition_penalty != 1.0 and previous_ids is not None and previous_ids.numel() > 0:
            if repetition_penalty > 0:
                 unique_previous_ids = torch.unique(previous_ids.view(-1).to(logits.device))
                 for token_id in unique_previous_ids:
                      if 0 <= token_id < vocab_size:
                           if logits[token_id] < 0: logits[token_id] *= repetition_penalty
                           else: logits[token_id] /= repetition_penalty
            # else: logger.warning(...) # Warn invalid penalty

        # --- 2. Apply Temperature ---
        effective_temp = temperature if do_sample and temperature > 1e-8 else 1.0
        if effective_temp != 1.0:
             logits = logits / effective_temp

        # --- CLAMP logits after penalty/temp before filtering ---
        # Prevent extreme values causing issues in topk/softmax
        logits.clamp_(min=-1e4, max=1e4) # Clamp in-place
        if not torch.isfinite(logits).all(): # Final check after clamp
            logger.warning(f"⚠️ [{debug_id}] Logits still non-finite after clamp. Using fallback.")
            return torch.tensor(fallback_token_id, device=logits.device, dtype=torch.long)
        # ---

        # --- 3. Apply Top-K Filtering (only if sampling) ---
        if top_k > 0 and do_sample:
            _top_k = min(top_k, vocab_size)
            if _top_k > 0 and _top_k < vocab_size:
                top_k_values, top_k_indices = torch.topk(logits, _top_k)
                filter_mask = torch.ones_like(logits, dtype=torch.bool)
                filter_mask[top_k_indices] = False
                logits.masked_fill_(filter_mask, float('-inf'))
        # --- End Top-K ---

        # --- 4. Apply Top-P Filtering (only if sampling) ---
        if 0.0 < top_p < 1.0 and do_sample:
            # Check if any valid logits remain
            if not torch.isneginf(logits).all():
                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                 # Use float32 for stable softmax/cumsum
                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                 # Filter logic
                 sorted_indices_to_remove = cumulative_probs > top_p
                 if sorted_indices_to_remove.numel() > 1:
                     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                 sorted_indices_to_remove[..., 0] = 0
                 if sorted_indices_to_remove.any():
                     indices_to_remove = sorted_indices[sorted_indices_to_remove]
                     logits.scatter_(0, indices_to_remove, float('-inf'))
            # else: logger.warning(...) # Log if top-p skipped due to all -inf
        # --- End Top-P ---

        # --- 5. Sample or Greedy Decode ---
        if torch.isneginf(logits).all(): # Check again after all filtering
            logger.warning(f"⚠️ [{debug_id}] All logits filtered out. Using fallback.")
            return torch.tensor(fallback_token_id, device=logits.device, dtype=torch.long)

        if do_sample:
            probs = F.softmax(logits, dim=-1) # Calculate probs from potentially filtered logits
            if not (probs.sum() > 1e-9): # Check if probs sum is valid
                 logger.warning(f"⚠️ [{debug_id}] Prob sum zero after filtering. Using argmax fallback.")
                 return torch.argmax(logits, dim=-1)

            try: # Sample using multinomial
                sampled_token_id = torch.multinomial(probs, num_samples=1)
                return sampled_token_id.squeeze() # Return 0D tensor
            except Exception as e:
                logger.error(f"⚠️ [{debug_id}] Multinomial sampling failed: {e}. Using argmax fallback.")
                return torch.argmax(logits, dim=-1)
        else: # Greedy decoding
            return torch.argmax(logits, dim=-1)

    except Exception as e:
        logger.error(f"⚠️ [{debug_id}] Safe sampling failed unexpectedly: {e}", exc_info=True)
        return torch.tensor(fallback_token_id, device=logits.device, dtype=torch.long)
    

    
# def safe_sample_from_logits(
#         logits: torch.Tensor,
#         temperature: float = 1.0,
#         top_k: int = 50,
#         top_p: float = 0.95,
#         do_sample: bool = True,
#         fallback_token_id: int = 0,
#         debug_id: str = ""
#     ) -> torch.Tensor:
#     # ... (robust implementation from previous answer) ...
#     try:
#         # Guard: check for NaNs or Infs
#         if not torch.isfinite(logits).all():
#             logger.warning(f"⚠️ [{debug_id}] Non-finite logits detected BEFORE sampling. Replacing with fallback.")
#             return torch.tensor([fallback_token_id], device=logits.device, dtype=torch.long)

#         # Apply temperature
#         effective_temp = temperature
#         if not do_sample: # Greedy decoding
#             effective_temp = 1.0 # Apply filters but use argmax later
#         elif effective_temp <= 0: # Avoid division by zero/invalid ops if temp is <= 0 for sampling
#              effective_temp = 1.0
#              logger.warning(f"⚠️ [{debug_id}] Temperature is <= 0 but do_sample is True. Using temp=1.0.")

#         if effective_temp != 1.0:
#              # Check for extreme values before division
#              if (logits / effective_temp).abs().max() > 1e5: # Heuristic threshold
#                   logger.warning(f"⚠️ [{debug_id}] Potential overflow/underflow with temp={effective_temp}. Clamping logits.")
#                   logits = torch.clamp(logits, min=-1e4*effective_temp, max=1e4*effective_temp)
#              logits = logits / effective_temp


#         # Optional: Top-k
#         if top_k > 0:
#             _top_k = min(top_k, logits.size(-1))
#             if _top_k > 0:
#                 top_k_values, top_k_indices = torch.topk(logits, _top_k)
#                 mask = torch.full_like(logits, float('-inf'))
#                 mask.scatter_(0, top_k_indices, top_k_values)
#                 logits = mask

#         # Optional: Top-p
#         if 0.0 < top_p < 1.0:
#             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#             sorted_mask = cumulative_probs > top_p
#             # Ensure mask is not empty and has more than one element before slicing
#             if sorted_mask.numel() > 1:
#               sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
#               sorted_mask[..., 0] = 0
#               # Ensure indices are valid before using them
#               if sorted_mask.any():
#                   remove_indices = sorted_indices[sorted_mask]
#                   logits[remove_indices] = float('-inf')

#         # Guard: all logits -inf
#         if (logits == float('-inf')).all():
#             logger.warning(f"⚠️ [{debug_id}] All logits were filtered out. Using fallback.")
#             if 'top_k_indices' in locals() and top_k_indices.numel() == 1:
#                 return top_k_indices.clone().detach()
#             return torch.tensor([fallback_token_id], device=logits.device, dtype=torch.long)

#         # Convert to probs
#         probs = F.softmax(logits, dim=-1)

#         # Guard: Invalid probs
#         probs_sum = probs.sum()
#         if not torch.isfinite(probs).all() or probs_sum <= 1e-9:
#             logger.warning(f"⚠️ [{debug_id}] Invalid probs detected (sum={probs_sum.item()}). Trying greedy fallback.")
#             if (logits > float('-inf')).any():
#                  return torch.argmax(logits, dim=-1, keepdim=True)
#             logger.warning(f"⚠️ [{debug_id}] Greedy fallback failed. Using fallback token id.")
#             return torch.tensor([fallback_token_id], device=logits.device, dtype=torch.long)

#         # Sample or greedy
#         if do_sample:
#             probs = probs / probs.sum().clamp(min=1e-9) # Ensure sum to 1
#             if (probs <= 0).all():
#                  logger.warning(f"⚠️ [{debug_id}] All probabilities became zero or less after normalization. Using fallback.")
#                  return torch.tensor([fallback_token_id], device=logits.device, dtype=torch.long)
#             return torch.multinomial(probs, num_samples=1)
#         else:
#             return torch.argmax(logits, dim=-1, keepdim=True) # Greedy from filtered logits

#     except Exception as e:
#         logger.error(f"⚠️ [{debug_id}] Safe sampling failed: {e}", exc_info=True)
#         return torch.tensor([fallback_token_id], device=logits.device, dtype=torch.long)


class EnhancedHierarchicalFlowGAT(HierarchicalFlowGAT):
    """
    Enhanced Hierarchical Flow GAT with internal contextualization cycles at each level,
    a dedicated prediction transformer stack, and support for token imputation.
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
        max_seq_len: int = 4096,
        use_final_layer_for_prediction: bool = True,
        add_self_loops: bool = True,
        add_long_range_edges: bool = True,
        long_range_distance: int = 3,
        iterative_refinement_cycles: int = 3, # Stage 1 cycles
        unified_refinement_cycles: int = 0,   # Stage 2 cycles
        refinement_cycles: int = 2,
        use_edge_attr: bool = True,
        share_transformers: bool = True,
        lap_pe_k: int = 16,
        #refinement_style: str = "iterative_level",
        # Args specifically for Enhanced Class
        l0_cycles=8,
        internal_cycles_per_level=[4, 4, 4, 4],
        #prediction_layers=2,
        train_with_imputation=False,
        *args,
        **kwargs
    ):
        super().__init__(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        compression_ratios=compression_ratios,
        overlap_ratios=overlap_ratios,
        max_seq_len=max_seq_len,
        use_final_layer_for_prediction=use_final_layer_for_prediction,
        add_self_loops=add_self_loops,
        add_long_range_edges=add_long_range_edges,
        long_range_distance=long_range_distance,
        iterative_refinement_cycles=iterative_refinement_cycles, # Pass Stage 1 cycles
        unified_refinement_cycles=unified_refinement_cycles, # Pass Stage 2 cycles
        refinement_cycles=refinement_cycles,
        use_edge_attr=use_edge_attr,
        share_transformers=share_transformers,
        lap_pe_k=lap_pe_k,
        #refinement_style=refinement_style,
        *args, 
        **kwargs,
        )
        self.l0_cycles = l0_cycles
        self.internal_cycles_per_level = internal_cycles_per_level
        #self.prediction_layers = prediction_layers
        self.train_with_imputation = train_with_imputation
        # Add pre-output normalization
        self.pre_output_norm = make_norm(self.hidden_dim, norm_type=self.norm_type, eps=self.norm_eps)
        # unified skeleton cache (enhanced-only)
        self.enable_unified_skeleton_cache = True
        self._unified_skeleton_cache = {}   # key -> skeleton dict
        self._unified_skeleton_device_cache = {}  # (ukey, device_type, device_idx) -> device tensors
        # FIFO bounds: training reuses a fixed length (stays hot); generation cycles through many
        # lengths, so without a bound the device cache grows until OOM. Device cache holds GPU
        # tensors -> keep it small; host skeleton cache is CPU RAM -> a looser bound is fine.
        self._skeleton_cache_max = 16
        self._skeleton_device_cache_max = 4
        self._uf_cache_fast_hits: int = 0
        self._uf_cache_fast_misses: int = 0
        self._uf_cache_seed_count: int = 0
        self._uf_cache_restart_count: int = 0
        self._uf_cache_last_seq_len: Optional[int] = None
        self._uf_cache_last_key: Optional[tuple] = None
        self._uf_cache_last_build_ms: Optional[float] = None
        self._uf_cache_last_rehydrate_ms: Optional[float] = None
        self._uf_cache_last_miss_reason: Optional[str] = None
        self._last_forward_graph_log_key: Optional[tuple] = None
        self.upper_init = getattr(self, "upper_init", "mask")  # "mask" or "zeros"
        # Track current device for the executor
        self.device = next(self.parameters()).device  # robust after .to(device)
        self._batched_executor = BatchedLayerExecutor(use_ckpt=bool(getattr(self, "use_gradient_checkpointing", False)))
        self.emit_features_only:bool = False

        # Layer / RMS normalization before final projection
        self.final_norm = make_norm(self.hidden_dim, norm_type=self.norm_type, eps=self.norm_eps)

        # # Create dedicated prediction transformer layers
        # self.prediction_transformers = nn.ModuleList()
        # for _ in range(prediction_layers):
        #     self.prediction_transformers.append(
        #         HierarchicalTransformerLayer(
        #             hidden_dim=self.hidden_dim,
        #             num_heads=self.num_heads,
        #             dropout=kwargs.get('dropout', 0.1),
        #             edge_dim=self.hidden_dim if kwargs.get('use_edge_attr', True) else None,
        #             use_edge_attr=kwargs.get('use_edge_attr', True)
        #         )
        #     )
        
        #self.highest_to_token_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        logger.info(f"Initialized Enhanced HFGAT. L0 Cycles={self.l0_cycles}, Internal Cycles={self.internal_cycles_per_level}, Iterative Cycles={self.iterative_refinement_cycles}, Unified Cycles={self.unified_refinement_cycles}")

        #logger.info(f"Initialized Enhanced HFGAT. Refinement: {self.refinement_style}")


        
        # Log initialization message
        # msg = f"Initialized Enhanced Hierarchical Flow GAT with {self.l0_cycles} L0 cycles, " \
        #       f"{self.internal_cycles_per_level} internal cycles per level, and " \
        #       f"{self.prediction_layers} prediction transformer layers"
        # if train_with_imputation:
        #     msg += " (using imputation for training)"
        # logger.info(msg)

    # Add a layer norm to use after refined graph

            
    
    def _pack_unified_batch_old(
        self,
        sk_list: list,            # list of skeleton dicts (from _skeletonize_unified)
        x_list: list,             # list of [N_i, H] tensors per sample (L0..Ln concatenated per sample)
        device: torch.device,     # model/device (only used to shape/validate)
    ):
        """
        Pack multiple per-sample unified skeletons and features into a single big graph.
        Returns: (batched_cpu_graph, slices)
        """
        from torch_geometric.data import Data
        import torch.nn.functional as F

        assert len(sk_list) == len(x_list), "sk_list and x_list must be same length"
        B = len(sk_list)

        # 1) per-sample node counts + offsets
        n_nodes = [
            int(sk["num_nodes"]) if "num_nodes" in sk else int(x_list[i].size(0))
            for i, sk in enumerate(sk_list)
        ]
        offsets = [0]
        for n in n_nodes:
            offsets.append(offsets[-1] + n)
        total_N = offsets[-1]

        H = x_list[0].size(1)
        x_cat = torch.empty((total_N, H), dtype=x_list[0].dtype, device="cpu")
        start = 0
        for i, x in enumerate(x_list):
            n = n_nodes[i]
            x_cat[start:start+n].copy_(x.detach().to("cpu", non_blocking=True))
            start += n

        # 2) concat skeleton parts with index shifts
        edge_index_list = []
        node_level_list = []
        edge_type_list  = []

        def _norm(lo):
            return tuple(int(v) for v in (lo or []))

        layout_sig = _norm(sk_list[0].get("level_offsets", [])) if B > 0 else ()
        same_layout = all(_norm(sk.get("level_offsets", [])) == layout_sig for sk in sk_list)

        l0_slices = []  # (start,end) for L0 per sample

        for i, sk in enumerate(sk_list):
            off = offsets[i]
            # edge_index
            ei = sk["edge_index_cpu"].clone()
            ei += off
            edge_index_list.append(ei)

            # node_level
            nl = sk.get("node_level_cpu", None)
            if nl is not None:
                node_level_list.append(nl.clone())

            # edge_type
            et = sk.get("edge_type_cpu", None)
            if et is not None:
                edge_type_list.append(et.clone())

            # record L0 slice (assumes layout_sig[0]=0, layout_sig[1]=nL0)
            if same_layout and len(layout_sig) >= 2:
                l0_len = layout_sig[1] - layout_sig[0]
                l0_slices.append((off + layout_sig[0], off + layout_sig[1]))
            else:
                # fallback: first T nodes are L0
                # NOTE: caller must pass T separately if you choose this
                pass

        edge_index_cat = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2,0), dtype=torch.long)
        node_level_cat = torch.cat(node_level_list, dim=0) if node_level_list else None
        edge_type_cat  = torch.cat(edge_type_list,  dim=0) if edge_type_list  else None

        batched_cpu = Data(
            x=x_cat,  # convenience; used as x_init
            edge_index=edge_index_cat.contiguous().cpu(),
            node_level=(node_level_cat.contiguous().cpu() if node_level_cat is not None else None),
            edge_type=(edge_type_cat.contiguous().cpu() if edge_type_cat is not None else None),
            num_nodes=total_N,
        )

        # Merge level_offsets if all samples share the same layout:
        if same_layout and layout_sig:
            # layout_sig is something like [0, nL0, nL0+nL1, ..., sum]
            merged_offsets = []
            for i in range(B):
                base = offsets[i]
                # append all elements shifted by base (including 0 for each sample)
                merged_offsets.extend([base + v for v in layout_sig])
            # Store as a CPU tensor so your executor can .to(device) each step
            batched_cpu.level_offsets = torch.tensor(merged_offsets, dtype=torch.long)  # CPU
        else:
            batched_cpu.level_offsets = None  # fall back to positions=n_idexes if needed

        # Pack LapPE (projected) if present per sample
        if any(("lap_pe_proj_cpu" in sk and sk["lap_pe_proj_cpu"] is not None) for sk in sk_list):
            pe_cat = torch.zeros_like(x_cat)
            start = 0
            for i, sk in enumerate(sk_list):
                n = n_nodes[i]
                pe = sk.get("lap_pe_proj_cpu", None)
                if pe is not None:
                    assert pe.size(0) == n, "LapPE length mismatch"
                    pe_cat[start:start+n].copy_(pe)
                start += n
            batched_cpu.lap_pe_proj_cpu = pe_cat.contiguous().cpu()

        slices = {
            "offsets": offsets,          # cumulative offsets
            "n_nodes": n_nodes,
            "l0_slices": l0_slices if l0_slices else None,
        }
        return batched_cpu, slices
    
    def _pack_unified_batch(self, sk_list, x_list, device):
        from torch_geometric.data import Data
        assert len(sk_list) == len(x_list)
        B = len(sk_list)

        # sizes/offsets
        n_nodes, offsets = [], [0]
        for i, sk in enumerate(sk_list):
            n = int(sk["num_nodes"]) if "num_nodes" in sk else int(x_list[i].size(0))
            n_nodes.append(n); offsets.append(offsets[-1] + n)
        total_N = offsets[-1]

        H = x_list[0].size(1)
        x_cat = torch.empty((total_N, H), dtype=x_list[0].dtype, device="cpu")
        start = 0
        for i, x in enumerate(x_list):
            n = n_nodes[i]
            x_cat[start:start+n].copy_(x.detach().to("cpu", non_blocking=True))
            start += n

        # concat graph attrs (+shift)
        edge_index_list, node_level_list, edge_type_list = [], [], []
        def _norm(lo): return tuple(int(v) for v in (lo or []))
        layout_sig = _norm(sk_list[0].get("level_offsets", [])) if B>0 else ()
        same_layout = all(_norm(sk.get("level_offsets", [])) == layout_sig for sk in sk_list)
        def _norm_grid(gs):
            if gs is None:
                return None
            return tuple(None if s is None else (int(s[0]), int(s[1])) for s in gs)
        first_grid_shapes = sk_list[0].get("level_grid_shapes", None) if B > 0 else None
        layout_grid_shapes = list(first_grid_shapes) if first_grid_shapes is not None else None
        same_grid_shapes = (
            layout_grid_shapes is not None
            and all(_norm_grid(sk.get("level_grid_shapes", None)) == _norm_grid(layout_grid_shapes) for sk in sk_list)
        )

        l0_slices = []  # (start,end) inclusive-exclusive per sample

        for i, sk in enumerate(sk_list):
            off = offsets[i]
            ei = sk["edge_index_cpu"].clone(); ei += off
            edge_index_list.append(ei)

            nl = sk.get("node_level_cpu", None)
            if nl is not None: node_level_list.append(nl.clone())
            et = sk.get("edge_type_cpu", None)
            if et is not None: edge_type_list.append(et.clone())

        edge_index_cat = torch.cat(edge_index_list, 1) if edge_index_list else torch.empty((2,0), dtype=torch.long)
        node_level_cat = torch.cat(node_level_list, 0) if node_level_list else None
        edge_type_cat  = torch.cat(edge_type_list,  0) if edge_type_list  else None

        batched_cpu = Data(
            x=x_cat,
            edge_index=edge_index_cat.contiguous().cpu(),
            node_level=(node_level_cat.contiguous().cpu() if node_level_cat is not None else None),
            edge_type=(edge_type_cat.contiguous().cpu() if edge_type_cat is not None else None),
            num_nodes=total_N,
        )

        # merged level_offsets only if same layout
        if same_layout and layout_sig:
            merged = []
            for i in range(B):
                base = offsets[i]
                merged.extend([base + v for v in layout_sig])
            batched_cpu.level_offsets = torch.tensor(merged, dtype=torch.long)  # CPU
        else:
            batched_cpu.level_offsets = None

        if same_grid_shapes and layout_grid_shapes is not None:
            batched_cpu.level_grid_shapes = list(layout_grid_shapes)
        else:
            batched_cpu.level_grid_shapes = None

        # ---- NEW: robust L0 slice discovery (works even if layouts differ) ----
        if node_level_cat is not None:
            for i in range(B):
                lo, hi = offsets[i], offsets[i+1]
                seg = node_level_cat[lo:hi]
                idx = (seg == 0).nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    # fallback: assume first T nodes are L0 (caller must handle)
                    l0_slices.append((lo, lo))  # empty; will be checked later
                else:
                    s = lo + idx[0].item()
                    e = lo + idx[-1].item() + 1
                    l0_slices.append((s, e))
        else:
            # no node_level — last resort: leave empty and let caller slice by seq_len
            l0_slices = None
        # ----------------------------------------------------------------------

        # ---- NEW: L3 slice discovery for per-sample memory retrieval ----
        l3_slices = []
        if node_level_cat is not None:
            for i in range(B):
                lo, hi = offsets[i], offsets[i+1]
                seg_levels = node_level_cat[lo:hi]
                l3_mask = (seg_levels == 3)
                if l3_mask.any():
                    l3_idx_local = l3_mask.nonzero(as_tuple=False).view(-1)
                    l3_start = lo + l3_idx_local[0].item()
                    l3_end = lo + l3_idx_local[-1].item() + 1
                    l3_slices.append((l3_start, l3_end))
                else:
                    l3_slices.append((lo, lo))  # Empty
        else:
            l3_slices = None
        # ----------------------------------------------------------------------

        # ---- NEW: L2 slice discovery for multi-level memory injection (Phase 10.6) ----
        l2_slices = []
        if node_level_cat is not None:
            for i in range(B):
                lo, hi = offsets[i], offsets[i+1]
                seg_levels = node_level_cat[lo:hi]
                l2_mask = (seg_levels == 2)
                if l2_mask.any():
                    l2_idx_local = l2_mask.nonzero(as_tuple=False).view(-1)
                    l2_start = lo + l2_idx_local[0].item()
                    l2_end = lo + l2_idx_local[-1].item() + 1
                    l2_slices.append((l2_start, l2_end))
                else:
                    l2_slices.append((lo, lo))  # Empty
        else:
            l2_slices = None
        # ----------------------------------------------------------------------

        # ---- NEW: L1 slice discovery for multi-level memory injection (Phase 10.6) ----
        l1_slices = []
        if node_level_cat is not None:
            for i in range(B):
                lo, hi = offsets[i], offsets[i+1]
                seg_levels = node_level_cat[lo:hi]
                l1_mask = (seg_levels == 1)
                if l1_mask.any():
                    l1_idx_local = l1_mask.nonzero(as_tuple=False).view(-1)
                    l1_start = lo + l1_idx_local[0].item()
                    l1_end = lo + l1_idx_local[-1].item() + 1
                    l1_slices.append((l1_start, l1_end))
                else:
                    l1_slices.append((lo, lo))  # Empty
        else:
            l1_slices = None
        # ----------------------------------------------------------------------


        # ---- LapPE pack (raw, optional) ----
        if any(("lap_pe_raw_cpu" in sk and sk["lap_pe_raw_cpu"] is not None) for sk in sk_list):
            # k can vary across configs—get it from first non-None
            first_pe = next(sk["lap_pe_raw_cpu"] for sk in sk_list if sk.get("lap_pe_raw_cpu", None) is not None)
            k = first_pe.size(1)
            pe_cat = torch.zeros((total_N, k), dtype=first_pe.dtype, device="cpu")
            start = 0
            for i, sk in enumerate(sk_list):
                n = n_nodes[i]
                pe = sk.get("lap_pe_raw_cpu", None)
                if pe is not None:
                    assert pe.size(0) == n, "LapPE length mismatch for sample {i}"
                    pe_cat[start:start+n].copy_(pe)
                start += n
            batched_cpu.lap_pe_raw_cpu = pe_cat.contiguous()
        else:
            batched_cpu.lap_pe_raw_cpu = None

        # # LapPE pack (optional)
        # if any(("lap_pe_proj_cpu" in sk and sk["lap_pe_proj_cpu"] is not None) for sk in sk_list):
        #     pe_cat = torch.zeros_like(x_cat)
        #     start = 0
        #     for i, sk in enumerate(sk_list):
        #         n = n_nodes[i]
        #         pe = sk.get("lap_pe_proj_cpu", None)
        #         if pe is not None:
        #             pe_cat[start:start+n].copy_(pe)
        #         start += n
        #     batched_cpu.lap_pe_proj_cpu = pe_cat.contiguous().cpu()

        # --- NEW: precompute per-node pos_in_level on CPU ---
        pos_in_level_cat = None
        if node_level_cat is not None:
            pos_list = []
            for i, sk in enumerate(sk_list):
                lo, hi = offsets[i], offsets[i+1]
                seg_levels = node_level_cat[lo:hi]               # [Ni]
                # local ids 0..Ni-1
                local = torch.arange(hi - lo, dtype=torch.long)
                # If layouts are aligned across the batch, use layout_sig; otherwise derive per-sample starts.
                if same_layout and layout_sig:
                    # layout_sig = [L0_start, L1_start, ..., sum]
                    # per-level start (in local coords) is layout_sig[l]
                    pos_local = local.clone()
                    for l in range(len(layout_sig) - 1):
                        m = (seg_levels == l)
                        if m.any():
                            pos_local[m] = pos_local[m] - layout_sig[l]
                else:
                    # derive starts from the first occurrence of each level in this sample
                    pos_local = local.clone()
                    for l in torch.unique(seg_levels).tolist():
                        idx = (seg_levels == l).nonzero(as_tuple=False).view(-1)
                        if idx.numel() > 0:
                            start_l = idx[0].item()
                            pos_local[idx] = pos_local[idx] - start_l
                pos_list.append(pos_local)
            pos_in_level_cat = torch.cat(pos_list, dim=0)        # [N] long

        # attach to Data
        batched_cpu.pos_in_level = (pos_in_level_cat if pos_in_level_cat is not None else None)


        slices = {
            "offsets": offsets,
            "n_nodes": n_nodes,
            "l0_slices": l0_slices,
            "l1_slices": l1_slices,  # Phase 10.6
            "l2_slices": l2_slices,  # Phase 10.6
            "l3_slices": l3_slices,
        }
        return batched_cpu, slices


    from torch_geometric.data import Data
    
    def _build_and_cache_unified_skeleton_from_embeddings(self, token_embeddings: torch.Tensor):
        from torch_geometric.data import Data
        import torch.nn.functional as F

        T = token_embeddings.size(0)
        level_sizes = self._predict_level_sizes(T)
        ukey = self._unified_cache_key(level_sizes)
        self._uf_cache_last_seq_len = int(T)
        self._uf_cache_last_key = ukey
        if ukey in self._unified_skeleton_cache:
            return ukey, self._unified_skeleton_cache[ukey]

        t_build_start = time.time()
        logger.info(
            "[EHFGAT:HIER] building skeleton layout l0_len=%d level_sizes=%s",
            int(T),
            tuple(int(s) for s in level_sizes),
        )

        self.level_mappings = []
        l0 = self._build_level_graph(token_embeddings, level_idx=0)
        level_graphs = [l0]
        cur = l0
        for level_idx in range(1, len(level_sizes)):
            cr = self.compression_ratios[level_idx-1]
            or_ = self.overlap_ratios[level_idx-1]
            nxt, mapping = self._create_next_level(cur, level_idx=level_idx, compression_ratio=cr, overlap_ratio=or_)
            self.level_mappings.append(mapping)
            nxt = self._process_level(nxt, level_idx=level_idx)
            level_graphs.append(nxt)
            cur = nxt

        unified_graph = self._build_unified_graph(level_graphs, self.level_mappings)

        sk = {
            "num_nodes": int(unified_graph.num_nodes) if hasattr(unified_graph, "num_nodes") else int(unified_graph.x.size(0)),
            "edge_index_cpu": unified_graph.edge_index.detach().to("cpu"),
            "node_level_cpu": getattr(unified_graph, "node_level", None).detach().to("cpu")
                                if getattr(unified_graph, "node_level", None) is not None else None,
            "node_branch_cpu": getattr(unified_graph, "node_branch", None).detach().to("cpu")
                                if getattr(unified_graph, "node_branch", None) is not None else None,
            "edge_type_cpu": getattr(unified_graph, "edge_type", None).detach().to("cpu")
                                if getattr(unified_graph, "edge_type", None) is not None else None,
            "node_ar_time_cpu": getattr(unified_graph, "node_ar_time", None).detach().to("cpu")
                                if getattr(unified_graph, "node_ar_time", None) is not None else None,
            "level_offsets": list(getattr(unified_graph, "level_offsets", [])),
            "level_grid_shapes": list(getattr(unified_graph, "level_grid_shapes", []))
                                if getattr(unified_graph, "level_grid_shapes", None) is not None else None,
        }
        # LapPE (raw) — compute once on CPU for this layout
        if self.lap_pe_transform is not None and (self.lap_pe_k or 0) > 0:
            g_cpu_tmp = Data(edge_index=sk["edge_index_cpu"], num_nodes=sk["num_nodes"])
            g_cpu_tmp = self.lap_pe_transform(g_cpu_tmp)  # CPU op
            sk["lap_pe_raw_cpu"] = g_cpu_tmp.lap_pe.detach().to("cpu").contiguous()
        else:
            sk["lap_pe_raw_cpu"] = None
            logger.debug("LapPE transform disabled; skipping LapPE skeleton caching.")

        # Optional: LapPE (projected) snapshot on CPU with functional linear
        # if self.lap_pe_transform is not None and getattr(self, "lap_pe_k", 0) > 0:
        #     try:
        #         g_cpu_tmp = Data(edge_index=sk["edge_index_cpu"], num_nodes=sk["num_nodes"])
        #         g_cpu_tmp = self.lap_pe_transform(g_cpu_tmp)
        #         lap_raw = g_cpu_tmp.lap_pe.to(dtype=torch.float32, device="cpu").contiguous()
        #         sk["lap_pe_raw_cpu"] = lap_raw
        #         # if hasattr(self, "lap_pe_proj"):
        #         #     with torch.no_grad():
        #         #         W = self.lap_pe_proj.weight.detach().to("cpu", non_blocking=True)
        #         #         b = (self.lap_pe_proj.bias.detach().to("cpu", non_blocking=True)
        #         #             if getattr(self.lap_pe_proj, "bias", None) is not None else None)
        #         #         sk["lap_pe_proj_cpu"] = F.linear(lap_raw, W, b).contiguous()
        #         # else:
        #         sk["lap_pe_proj_cpu"] = None
        #     except Exception:
        #         sk["lap_pe_proj_cpu"] = None

        self._unified_skeleton_cache[ukey] = sk
        stale_dev_keys = [k for k in self._unified_skeleton_device_cache.keys() if k[0] == ukey]
        for stale_key in stale_dev_keys:
            self._unified_skeleton_device_cache.pop(stale_key, None)
        # Bound the host skeleton cache (FIFO) so generation's many lengths can't grow it without
        # bound. Must populate unconditionally: forward()'s RESTART_FORWARD path depends on the
        # just-built entry being present on the immediate recursive retry.
        self._evict_skeleton_cache(self._unified_skeleton_cache, self._skeleton_cache_max)
        self._uf_cache_seed_count += 1
        self._uf_cache_last_build_ms = (time.time() - t_build_start) * 1000.0
        logger.info(
            "[EHFGAT:HIER] cached skeleton built level_sizes=%s unified_nodes=%d unified_edges=%d build_ms=%.1f",
            tuple(int(s) for s in level_sizes),
            int(sk["num_nodes"]),
            int(sk["edge_index_cpu"].size(1)),
            float(self._uf_cache_last_build_ms),
        )
        return ukey, sk
    
    @staticmethod
    def _evict_skeleton_cache(cache: dict, max_entries: int) -> None:
        """FIFO-evict the oldest entries so `cache` holds at most `max_entries`. Dropped device
        packs lose their last reference and free their GPU tensors back to the allocator."""
        if max_entries <= 0:
            return
        while len(cache) > max_entries:
            cache.pop(next(iter(cache)), None)

    def _forward_batch_slow(self, input_ids, position_ids=None, attention_mask=None):
        """
        Called only when at least one sample in the batch has no cached unified skeleton.
        Seeds the caches for ALL samples in the batch, then executes the batched fast path once.
        Returns the same type your forward() would return in the fast path (features/logits).
        """
        device = next(self.parameters()).device
        #B, T = input_ids.shape
        # Determine batch size and sequence length early
        #print(f"Input IDs shape: {input_ids.shape}")
        if input_ids.ndim == 3:
            B, T, _ = input_ids.shape
        else:
            B, T = input_ids.shape

        sk_list = []
        x_list  = []

        for b in range(B):
            # Build per-sample embeddings
            tok_emb = self._get_embeddings(
                input_ids[b:b+1],
                position_ids[b:b+1] if position_ids is not None else None
            ).reshape(-1, self.hidden_dim)  # [T,H]

            # Ensure skeleton exists (and LapPE cached) for this layout
            ukey, sk = self._build_and_cache_unified_skeleton_from_embeddings(tok_emb)
            if sk is None:
                raise RuntimeError("Failed to build unified skeleton for a batch sample.")

            # Build per-sample concatenated x: L0 + zeros/mask for upper levels
            level_sizes = self._predict_level_sizes(tok_emb.size(0))
            xs = [tok_emb]
            for lvl in range(1, len(level_sizes)):
                n = level_sizes[lvl]
                if self.input_mode == "tokens" and getattr(self, "upper_init", "mask") in ("mask", "pooled"):
                    mask_vec = self.token_embedding(torch.tensor([self.mask_token_id], device=tok_emb.device))
                    xs.append(mask_vec.repeat(n, 1))
                else:
                    xs.append(torch.zeros(n, self.hidden_dim, device=tok_emb.device))
            x_cat = torch.cat(xs, dim=0)  # [sum(level_sizes), H]

            sk_list.append(sk)
            x_list.append(x_cat)

        # Now that every sample has a skeleton, reuse the already-written fast batched path:
        # (We pack, refine once, slice, and return.)
        batched_cpu, slices = self._pack_unified_batch(sk_list, x_list, device)

        x_init = batched_cpu.x  # CPU, can be pinned
        executor = getattr(self, "_batched_executor", None)
        if executor is None:
            from .batched_layer_executor import BatchedLayerExecutor
            self._batched_executor = BatchedLayerExecutor(use_ckpt=bool(getattr(self, "use_gradient_checkpointing", False)))
            executor = self._batched_executor

        # Cache the built graph and slices for memory graph merging
        self._last_built_graph = batched_cpu
        self._last_graph_slices = slices

        x_final = self._apply_unified_refinement_batched(
            unified_graph_cpu=batched_cpu,
            x_init=x_init,
            cycles=self.unified_refinement_cycles,
            batch_executor=executor,
            sweep_L0=True, sweep_L1=True, L1_iters=1, L2L3_iters=1,
        )  # CPU [N_total,H]

        if self.final_norm.weight.device != x_final.device:
            self.final_norm = self.final_norm.to(x_final.device)
                
        x_final = self.final_norm(x_final)

        # Slice L0 per sample -> [B,T,H]
        feats_bt = []
        for b in range(B):
            start = slices["offsets"][b]
            end   = start + T
            feats_bt.append(x_final[start:end].to(device, non_blocking=True))
        feats_bt = torch.stack([f.view(T, self.hidden_dim) for f in feats_bt], dim=0)  # [B,T,H]

        # Respect your output-mode logic
        #mode = self.get_output_mode()
        want_features = False#(self.training)
        want_logits   = True#(not self.training)

        if want_logits:
            logits = self.output_projection(feats_bt.view(-1, self.hidden_dim)).view(B, T, -1)

        if self.training:
            return feats_bt if want_features else logits
        else:
            return logits#{"features": feats_bt, "logits": logits}

    def _apply_unified_refinement_batched_ori(
        self,
        unified_graph_cpu,        # torch_geometric.data.Data on CPU (x can be None)
        x_init,                   # [N, H] CPU tensor (pin if possible)
        cycles: int,
        batch_executor: BatchedLayerExecutor,
        sweep_L0: bool = True,
        sweep_L1: bool = True,
        L1_iters: int = 1,
        L2L3_iters: int = 1,
    ):
        """
        Layer-wise refinement with exact 1-hop subgraphs streamed from CPU:
          - Repeat for `cycles`
          - For each transformer layer in the unified stack:
              * (optional) run a few dense passes over L2/L3
              * sweep L0 in batches
              * sweep L1 in batches
        Returns:
          x_global (CPU) after all cycles/layers, shape [N, H]
        """
        data = unified_graph_cpu
        #assert data.edge_index.device.type == "cpu", "unified_graph_cpu.edge_index must be on CPU"

        # Global store on CPU
        x_global = x_init
        if not isinstance(x_global, torch.Tensor):
            raise ValueError("x_init must be a CPU tensor [N, H]")

        # Level masks
        node_level = data.node_level  # CPU [N]
        L0_idx = (node_level == 0).nonzero(as_tuple=False).view(-1)
        L1_idx = (node_level == 1).nonzero(as_tuple=False).view(-1)
        L2_idx = (node_level == 2).nonzero(as_tuple=False).view(-1) if (node_level.max() >= 2) else torch.empty(0, dtype=torch.long)
        L3_idx = (node_level == 3).nonzero(as_tuple=False).view(-1) if (node_level.max() >= 3) else torch.empty(0, dtype=torch.long)
        # make a all node index
        All_idx = torch.arange(node_level.size(0), dtype=torch.long)
        split_levels = False
        bs_l0 = 16384*1024*2
        bs_l1 = bs_l0#1024
        bs_l2l3 = bs_l0#max(16, (L2_idx.numel() + L3_idx.numel()) ) 

        # Flatten the transformer stack used for unified refinement
        layers = []
        if self.share_transformers:
            for mods in self.level_transformers:
                layers.extend(list(mods))
        elif getattr(self, "refinement_transformers", None) is not None:
            layers = list(self.refinement_transformers)
        else:
            logger.warning("No transformers configured for unified refinement; returning x_init.")
            return x_global

        extra = {"level_offsets": getattr(data, "level_offsets", None)}

        for _ in range(max(1, int(cycles))):
            for layer in layers:
                # if split levels or first or last layer, do level-wise sweeps
                if split_levels:# or layer == layers[0] or layer == layers[-1]:
                    # # (a) sweep L0
                    # if sweep_L0 and L0_idx.numel() > 0:
                    #     x_global = batch_executor.run_one_layer(
                    #         layer=layer,
                    #         data=data,
                    #         x_global=x_global,
                    #         batch_size=bs_l0,
                    #         seed_nodes=L0_idx,
                    #         extra_kwargs=extra,
                    #     )

                    # # (b) sweep L1
                    # if sweep_L1 and L1_idx.numel() > 0:
                    #    x_global = batch_executor.run_one_layer(
                    #        layer=layer,
                    #        data=data,
                    #        x_global=x_global,
                    #        seed_nodes=L1_idx,
                    #        extra_kwargs=extra,
                    #    )

                    # (c) small dense passes on upper levels (global-ish mixing)
                    for _ in range(max(1, int(L1_iters))):

                        if sweep_L1 and L1_idx.numel() > 0:
                            x_global = batch_executor.run_one_layer(
                                layer=layer,
                                data=data,
                                x_global=x_global,
                                batch_size=bs_l1,
                                seed_nodes=L1_idx,
                                extra_kwargs=extra,
                            )
                            
                        
                        for _ in range(max(1, int(L2L3_iters))):
                            for U in (L2_idx, L3_idx):
                                if U.numel() == 0:
                                    continue
                                x_global = batch_executor.run_one_layer(
                                    layer=layer,
                                    data=data,
                                    x_global=x_global,
                                    batch_size=bs_l2l3,
                                    seed_nodes=U,
                                    extra_kwargs=extra,
                                )


                    # (d) sweep L1
                    if sweep_L1 and L1_idx.numel() > 0:
                        x_global = batch_executor.run_one_layer(
                            layer=layer,
                            data=data,
                            x_global=x_global,
                            batch_size=bs_l1,
                            seed_nodes=L1_idx,
                            extra_kwargs=extra,
                        )

                    # (e) sweep L0
                    if sweep_L0 and L0_idx.numel() > 0:
                        x_global = batch_executor.run_one_layer(
                            layer=layer,
                            data=data,
                            x_global=x_global,
                            batch_size=bs_l0,
                            seed_nodes=L0_idx,
                            extra_kwargs=extra,
                        )
                else:
                    # Single pass over all nodes
                    x_global = batch_executor.run_one_layer(
                        layer=layer,
                        data=data,
                        x_global=x_global,
                        batch_size=bs_l0,
                        seed_nodes=All_idx,
                        extra_kwargs=extra,
                    )

        return x_global
    

    def _apply_unified_refinement_batched(
        self,
        unified_graph_cpu,
        x_init,
        cycles: int,
        batch_executor: BatchedLayerExecutor,
        sweep_L0: bool = True,
        sweep_L1: bool = True,
        L1_iters: int = 1,
        L2L3_iters: int = 1,
    ):
        data = unified_graph_cpu
        x_global = x_init
        if not isinstance(x_global, torch.Tensor):
            raise ValueError("x_init must be a CPU tensor [N, H]")

        node_level = data.node_level  # CPU [N]
        L0_idx = (node_level == 0).nonzero(as_tuple=False).view(-1)
        L1_idx = (node_level == 1).nonzero(as_tuple=False).view(-1)
        L2_idx = (node_level == 2).nonzero(as_tuple=False).view(-1) if (node_level.max() >= 2) else torch.empty(0, dtype=torch.long)
        L3_idx = (node_level == 3).nonzero(as_tuple=False).view(-1) if (node_level.max() >= 3) else torch.empty(0, dtype=torch.long)
        All_idx = torch.arange(node_level.size(0), dtype=torch.long)
   
        # --- NEW: pass edge_feature_generator (if any) + level_offsets ---
        extra = {#"level_offsets": getattr(data, "level_offsets", None),
                "level_offsets": None,
                 "lap_pe_proj": getattr(self, "lap_pe_proj", None)}
        edge_gen = getattr(self, "edge_feature_generator", None)
        if edge_gen is not None:
            extra["edge_feature_generator"] = edge_gen  # executor will pick this up

        split_levels = False
        bs_l0 = 16384*1024*2
        bs_l1 = bs_l0
        bs_l2l3 = bs_l0

        layers = []
        if self.share_transformers:
            for mods in self.level_transformers:
                layers.extend(list(mods))
        elif getattr(self, "refinement_transformers", None) is not None:
            layers = list(self.refinement_transformers)
        else:
            logger.warning("No transformers configured for unified refinement; returning x_init.")
            return x_global

        #extra = {"level_offsets": getattr(data, "level_offsets", None)}

        # ------------------- NORMAL CYCLES (unchanged) -------------------
        for _ in range(max(1, int(cycles))):
            for layer in layers:
                
                # run the first and last layer with split levels
                if split_levels:# or layer == layers[0] or layer == layers[-1]:
                    # (L1/L2/L3 “dense-ish”), then L1, then L0 sweeps...
                    for _ in range(max(1, int(L1_iters))):
                        if sweep_L1 and L1_idx.numel() > 0:
                            x_global = batch_executor.run_one_layer(
                                layer=layer, data=data, x_global=x_global,
                                batch_size=bs_l1, seed_nodes=L1_idx, extra_kwargs=extra,
                            )
                        for _ in range(max(1, int(L2L3_iters))):
                            for U in (L2_idx, L3_idx):
                                if U.numel() == 0:
                                    continue
                                x_global = batch_executor.run_one_layer(
                                    layer=layer, data=data, x_global=x_global,
                                    batch_size=bs_l2l3, seed_nodes=U, extra_kwargs=extra,
                                )
                    if sweep_L1 and L1_idx.numel() > 0:
                        x_global = batch_executor.run_one_layer(
                            layer=layer, data=data, x_global=x_global,
                            batch_size=bs_l1, seed_nodes=L1_idx, extra_kwargs=extra,
                        )
                    if sweep_L0 and L0_idx.numel() > 0:
                        x_global = batch_executor.run_one_layer(
                            layer=layer, data=data, x_global=x_global,
                            batch_size=bs_l0, seed_nodes=L0_idx, extra_kwargs=extra,
                        )
                else:
                    # single sweep over all nodes
                    x_global = batch_executor.run_one_layer(
                        layer=layer, data=data, x_global=x_global,
                        batch_size=bs_l0, seed_nodes=All_idx, extra_kwargs=extra,
                    )

        # ------------------- OPTIONAL RECONSTRUCTION CYCLE -------------------
        # Trigger with 50% chance; only if we ran ≥1 normal cycle
        #do_recon = (cycles >= 1) and (torch.rand(1).item() < 0.5)
        # ------------------- OPTIONAL RECONSTRUCTION CYCLE -------------------
        do_recon = (
            self.training            # <--- only when model is in training mode
            and (cycles >= 1)
            and (torch.rand(1).item() < 0)#.1)
        )

        if do_recon:
            # Infer t (mask severity) from data if provided; else random fallback.
            # Expectation: caller can set `data.l0_mask_positions` as a boolean mask over the L0 block.
            t = None
            l0_mask_pos = getattr(data, "l0_mask_positions", None)  # torch.bool of shape [len(L0)]
            if l0_mask_pos is not None and l0_mask_pos.numel() == L0_idx.numel():
                # t := fraction of L0 that was masked
                t = float(l0_mask_pos.float().mean().item())
            if t is None:
                t = torch.rand(1).item()  # fallback: random severity
            t = t*0.5
            p_h = max(0.0, min(1.0, 0.5 - t))  # inverse-t for hierarchy masking

            # # Build a per-node boolean mask for zeroing inputs this extra cycle:
            # N = x_global.size(1)
            # # squeeze to add batch dim if needed
            # recon_mask = torch.zeros(N, dtype=torch.bool).s
            # # Always drop L0:
            # recon_mask[L0_idx] = True
            # # Bernoulli mask on higher levels with prob p_h:
            # if L1_idx.numel() > 0:
            #     recon_mask[L1_idx] = torch.rand(L1_idx.numel()) < p_h
            # if L2_idx.numel() > 0:
            #     recon_mask[L2_idx] = torch.rand(L2_idx.numel()) < p_h
            # if L3_idx.numel() > 0:
            #     recon_mask[L3_idx] = torch.rand(L3_idx.numel()) < p_h

            # # Prepare masked input (avoid in-place on x_global to keep autograd clean)
            # x_in = x_global.clone()
            # x_in[recon_mask] = 0#0.0 #Keep original masking values
            # x_global is [B, N, H] now
            B, N, H = x_global.shape
            device = x_global.device

            # 1D mask over nodes, shared across batch
            recon_mask = torch.zeros(N, dtype=torch.bool, device=device)

            # Always drop L0:
            recon_mask[L0_idx] = True

            # Bernoulli mask on higher levels with prob p_h:
            if L1_idx.numel() > 0:
                recon_mask[L1_idx] = torch.rand(L1_idx.numel(), device=device) < p_h
            if L2_idx.numel() > 0:
                recon_mask[L2_idx] = torch.rand(L2_idx.numel(), device=device) < p_h
            if L3_idx.numel() > 0:
                recon_mask[L3_idx] = torch.rand(L3_idx.numel(), device=device) < p_h

            # Prepare masked input (avoid in-place on x_global to keep autograd clean)
            x_in = x_global.clone()
            # Apply mask on node dimension, broadcast over batch + hidden
            x_in[:, recon_mask, :] = 0.0
            #print("running l0 masking recon cycle, t=", t, " p_h=", p_h, " masked nodes=", recon_mask.sum().item())
            # Run one more full pass of the same layer stack as an "extra cycle"
            for layer in layers:
                
                # if split levels or first or last layer, do level-wise sweeps
                if split_levels:# or layer == layers[0] or layer == layers[-1]:
                    # Mirror the split path if you use it; otherwise reuse the simple full sweep.
                    x_in = batch_executor.run_one_layer(
                        layer=layer, data=data, x_global=x_in,
                        batch_size=bs_l1, seed_nodes=L1_idx, extra_kwargs=extra,
                    )
                    for U in (L2_idx, L3_idx):
                        if U.numel() == 0:
                            continue
                        x_in = batch_executor.run_one_layer(
                            layer=layer, data=data, x_global=x_in,
                            batch_size=bs_l2l3, seed_nodes=U, extra_kwargs=extra,
                        )
                    if sweep_L1 and L1_idx.numel() > 0:
                        x_in = batch_executor.run_one_layer(
                            layer=layer, data=data, x_global=x_in,
                            batch_size=bs_l1, seed_nodes=L1_idx, extra_kwargs=extra,
                        )
                    if sweep_L0 and L0_idx.numel() > 0:
                        x_in = batch_executor.run_one_layer(
                            layer=layer, data=data, x_global=x_in,
                            batch_size=bs_l0, seed_nodes=L0_idx, extra_kwargs=extra,
                        )
                else:
                    x_in = batch_executor.run_one_layer(
                        layer=layer, data=data, x_global=x_in,
                        batch_size=bs_l0, seed_nodes=All_idx, extra_kwargs=extra,
                    )

            # Commit the reconstruction-updated state
            x_global = x_in


        return x_global
    
    def _predict_level_sizes(self, l0_len: int) -> list:
        geom_mode = str(getattr(self, "graph_geometry_mode", "sequence")).lower()
        if geom_mode == "grid2d":
            runtime = getattr(self, "_runtime_l0_grid_shape", None)
            if runtime is not None:
                gh = int(runtime[0])
                gw = int(runtime[1])
            else:
                gh = int(getattr(self, "graph_grid_height", 0))
                gw = int(getattr(self, "graph_grid_width", 0))
            if gh <= 0 or gw <= 0 or gh * gw != int(l0_len):
                side = int(round(math.sqrt(max(1, int(l0_len)))))
                if side * side == int(l0_len):
                    gh, gw = side, side
                else:
                    gh, gw = int(l0_len), 1
            sizes = [int(gh * gw)]
            ds_default = max(1, int(getattr(self, "graph_downsample_factor", 2)))
            for i in range(len(self.compression_ratios)):
                comp = max(1, int(self.compression_ratios[i]))
                ov = max(0.0, min(0.99, float(self.overlap_ratios[i])))
                kernel_2d = max(ds_default, int(round(math.sqrt(comp))))
                stride_2d = max(1, int(round(float(kernel_2d) * (1.0 - ov))))
                gh = max(1, (gh + stride_2d - 1) // stride_2d)
                gw = max(1, (gw + stride_2d - 1) // stride_2d)
                sizes.append(int(gh * gw))
            return sizes

        sizes = [l0_len]
        for i in range(len(self.compression_ratios)):
            comp = int(self.compression_ratios[i])
            stride = max(1, int(comp * (1.0 - self.overlap_ratios[i])))
            higher = max(1, (sizes[-1] - 1) // stride + 1)
            sizes.append(higher)
        return sizes

    def _unified_strides_for(self, level_sizes: list) -> list:
        return [max(1, int(self.compression_ratios[i] * (1.0 - self.overlap_ratios[i])))
                for i in range(len(level_sizes) - 1)]

    def _unified_cache_key(self, level_sizes: list) -> tuple:
        runtime = getattr(self, "_runtime_l0_grid_shape", None)
        if runtime is not None:
            gh = int(runtime[0])
            gw = int(runtime[1])
        else:
            gh = int(getattr(self, "graph_grid_height", 0))
            gw = int(getattr(self, "graph_grid_width", 0))
        return (
            tuple(level_sizes),
            tuple(self._unified_strides_for(level_sizes)),
            len(level_sizes),
            str(getattr(self, "graph_geometry_mode", "sequence")),
            int(gh),
            int(gw),
            str(getattr(self, "graph_spatial_metric", "chebyshev")),
            int(getattr(self, "graph_downsample_factor", 2)),
            bool(self.use_edge_attr),
            bool(self.add_self_loops),
            int(self.long_range_distance or 0),
            bool(getattr(self, "hier_ar_enable", False)),
            bool(getattr(self, "hier_ar_allow_same_time", True)),
            bool(getattr(self, "l0_ar_enable", False)),
            bool(getattr(self, "enable_l0_parent_edges", True)),
            bool(getattr(self, "l0_parent_edges_bidirectional", True)),
            bool(getattr(self, "ensure_l0_past_l1_edges", True)),
            bool(getattr(self, "ensure_past_hier_edges_all_levels", False)),
            bool(getattr(self, "ensure_l0_past_parent_edges", False)),
            int(getattr(self, "l0_past_parent_min_level", 1)),
            -1 if getattr(self, "l0_past_parent_max_level", None) is None else int(self.l0_past_parent_max_level),
            str(getattr(self, "autoenc_graph_mode", "off")),
            bool(getattr(self, "autoenc_coupled_feedback", True)),
        )

    def _skeletonize_unified(self, g: Data) -> dict:
        return {
            "num_nodes": int(g.num_nodes) if hasattr(g, "num_nodes") else int(g.x.size(0)),
            "edge_index_cpu": g.edge_index.detach().to("cpu"),
            "node_level_cpu": getattr(g, "node_level", None).detach().to("cpu"),
            "node_branch_cpu": getattr(g, "node_branch", None).detach().to("cpu")
                                if getattr(g, "node_branch", None) is not None else None,
            "edge_type_cpu": getattr(g, "edge_type", None).detach().to("cpu"),
            "node_ar_time_cpu": getattr(g, "node_ar_time", None).detach().to("cpu")
                                if getattr(g, "node_ar_time", None) is not None else None,
            "level_offsets": list(getattr(g, "level_offsets", [])),
            "level_grid_shapes": list(getattr(g, "level_grid_shapes", []))
                                if getattr(g, "level_grid_shapes", None) is not None else None,
            "lap_pe_raw_cpu": getattr(g, "lap_pe_raw_cpu", None),
            "node_pos_local_cpu": getattr(g, "node_pos_local", None).detach().to("cpu")
                                if getattr(g, "node_pos_local", None) is not None else None,
            "ae_decoder_l0_slice": tuple(getattr(g, "ae_decoder_l0_slice", (0, 0))),
        }
    

    def _rehydrate_unified(
        self,
        sk: dict,
        x_cat: torch.Tensor,
        device: torch.device,
        cache_key: Optional[tuple] = None,
    ) -> Data:
        device_pack = None
        if cache_key is not None and bool(getattr(self, "unified_skeleton_device_cache_enable", True)):
            dev_idx = int(device.index) if device.index is not None else -1
            dkey = (cache_key, str(device.type), dev_idx)
            device_pack = self._unified_skeleton_device_cache.get(dkey, None)
            if device_pack is None:
                device_pack = {
                    "edge_index": sk["edge_index_cpu"].to(device, non_blocking=True),
                    "node_level": sk["node_level_cpu"].to(device, non_blocking=True) if sk["node_level_cpu"] is not None else None,
                    "node_branch": sk["node_branch_cpu"].to(device, non_blocking=True) if sk.get("node_branch_cpu", None) is not None else None,
                    "edge_type": sk["edge_type_cpu"].to(device, non_blocking=True) if sk["edge_type_cpu"] is not None else None,
                    "node_ar_time": sk["node_ar_time_cpu"].to(device, non_blocking=True) if sk.get("node_ar_time_cpu", None) is not None else None,
                    "level_offsets": (
                        torch.tensor(sk["level_offsets"], device=device, dtype=torch.long)
                        if sk.get("level_offsets", None) is not None
                        else None
                    ),
                    "level_grid_shapes": list(sk["level_grid_shapes"]) if sk.get("level_grid_shapes", None) is not None else None,
                    "lap_pe_raw": sk["lap_pe_raw_cpu"].to(device, non_blocking=True) if sk.get("lap_pe_raw_cpu", None) is not None else None,
                    "node_pos_local": sk["node_pos_local_cpu"].to(device, non_blocking=True) if sk.get("node_pos_local_cpu", None) is not None else None,
                }
                self._unified_skeleton_device_cache[dkey] = device_pack
                # Bound the GPU device cache: generation runs at many sequence lengths, each
                # seeding a resident device pack. Evict oldest (FIFO) so it can't grow without
                # bound and OOM. Keep enough for training's fixed length(s) to stay hot.
                self._evict_skeleton_cache(self._unified_skeleton_device_cache, self._skeleton_device_cache_max)

        if device_pack is not None:
            g = Data(
                x=x_cat,
                edge_index=device_pack["edge_index"],
                node_level=device_pack["node_level"],
                node_branch=device_pack["node_branch"],
                edge_type=device_pack["edge_type"],
                node_ar_time=device_pack["node_ar_time"],
                level_offsets=device_pack["level_offsets"],
                level_grid_shapes=device_pack.get("level_grid_shapes", None),
                lap_pe_raw_cpu=device_pack["lap_pe_raw"],
            )
            if device_pack.get("node_pos_local", None) is not None:
                g.node_pos_local = device_pack["node_pos_local"]
        else:
            g = Data(
                x=x_cat,
                edge_index=sk["edge_index_cpu"].to(device, non_blocking=True),
                node_level=sk["node_level_cpu"].to(device, non_blocking=True) if sk["node_level_cpu"] is not None else None,
                node_branch=sk["node_branch_cpu"].to(device, non_blocking=True) if sk.get("node_branch_cpu", None) is not None else None,
                edge_type=sk["edge_type_cpu"].to(device, non_blocking=True) if sk["edge_type_cpu"] is not None else None,
                node_ar_time=sk["node_ar_time_cpu"].to(device, non_blocking=True) if sk.get("node_ar_time_cpu", None) is not None else None,
                level_offsets=list(sk["level_offsets"]) if sk["level_offsets"] is not None else None,
                level_grid_shapes=list(sk["level_grid_shapes"]) if sk.get("level_grid_shapes", None) is not None else None,
                lap_pe_raw_cpu=sk["lap_pe_raw_cpu"].to(device, non_blocking=True) if sk.get("lap_pe_raw_cpu", None) is not None else None,
            )
            if sk.get("node_pos_local_cpu", None) is not None:
                g.node_pos_local = sk["node_pos_local_cpu"].to(device, non_blocking=True)
        if sk.get("ae_decoder_l0_slice", (0, 0)) != (0, 0):
            g.ae_decoder_l0_slice = tuple(sk.get("ae_decoder_l0_slice", (0, 0)))
        #print(sk["lap_pe_raw_cpu"])
        # keep LapPE on CPU; we’ll slice & move per mini-batch
        #g.lap_pe_raw_cpu  = sk.get("lap_pe_raw_cpu", None)    # [N,k] or None
        #g.lap_pe_proj_cpu = sk.get("lap_pe_proj_cpu", None)   # [N,H] or None
        return g   

    def _hierarchical_level_init_inplace_simple(
        self,
        x_ref: torch.Tensor,        # [N, H] (single) OR [B, N, H] (batched local)
        level_offsets: torch.Tensor, # 1D [L+1] or [L] depending on your convention
        max_level: int,
    ):
        """
        In-place, projection-based initialization of levels > 0.

        - Assumes nodes are laid out contiguously per level:
            L0: [offset[0] : offset[1])
            L1: [offset[1] : offset[2])
            ...
        - Uses self.compression_ratios[level] and self.overlap_ratios[level]
        to reconstruct the sliding windows.
        - Uses self.level_projections[level] as in the original _create_next_level.
        """
        import math
        import torch.nn.functional as F

        if not hasattr(self, "level_projections") or self.level_projections is None:
            return  # nothing to do

        # Normalize offsets to tensor on x_ref.device
        level_offsets = torch.as_tensor(level_offsets, device=x_ref.device, dtype=torch.long)

        # Determine if batched or single
        if x_ref.dim() == 2:
            # [N, H] -> treat as [1, N, H] for uniform logic
            x_ref = x_ref.unsqueeze(0)  # [1, N, H]
            squeeze_back = True
        elif x_ref.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(f"_hierarchical_level_init_inplace: unsupported x_ref shape {x_ref.shape}")

        B, N, H = x_ref.shape

        # For each level > 0, recompute higher-level features from lower level
        # using the same compression + overlap logic as _create_next_level.
        for lvl in range(1, max_level + 1):
            lower = lvl - 1

            # Sanity checks
            if lower >= len(self.compression_ratios) or lower >= len(self.overlap_ratios):
                continue
            if lower >= len(self.level_projections):
                continue

            cr = int(self.compression_ratios[lower])
            ov = float(self.overlap_ratios[lower])
            stride = max(1, int(cr * (1.0 - ov)))

            # Level index ranges (local to each sample)
            lower_start = level_offsets[lower].item()
            lower_end   = level_offsets[lower + 1].item() if (lower + 1) <= max_level else N
            num_lower   = lower_end - lower_start

            higher_start = level_offsets[lvl].item()
            higher_end   = level_offsets[lvl + 1].item() if (lvl + 1) <= max_level else N
            num_higher   = higher_end - higher_start

            if num_lower <= 0 or num_higher <= 0:
                continue

            proj_layer = self.level_projections[lower]   # nn.Linear(H, H) or similar

            # Loop over higher-level nodes; #nodes is small so a Python loop is fine
            for h_idx in range(num_higher):
                # Sliding window in the lower level
                start_idx_local = min(h_idx * stride, max(0, num_lower - 1))
                end_idx_local   = min(start_idx_local + cr, num_lower)

                lower_slice = slice(lower_start + start_idx_local,
                                    lower_start + end_idx_local)
                higher_idx  = higher_start + h_idx

                if start_idx_local >= end_idx_local:
                    # degenerate; fall back to single lower node
                    lower_slice = slice(lower_start + start_idx_local,
                                        lower_start + start_idx_local + 1)

                # x_ref: [B, N, H] -> take [B, window, H]
                lower_feats = x_ref[:, lower_slice, :]      # [B, W, H]
                # project each token in window
                proj = proj_layer(lower_feats)              # [B, W, H]
                # mean pool
                pooled = proj.mean(dim=1)                   # [B, H]

                # assign into higher-level node across all batches
                x_ref[:, higher_idx, :] = pooled            # in-place

        if squeeze_back:
            # back to [N, H]
            return x_ref.squeeze(0)
        else:
            return x_ref

    def _hierarchical_level_init_inplace(
        self,
        x_ref: torch.Tensor,
        level_offsets: torch.Tensor,
        max_level: int,
        within_level_edges: dict = None,
    ):
        """
        Process-then-pool initialization.
        Checkpointing-safe version - avoids in-place ops on input tensor.
        """
        if not hasattr(self, "level_projections") or self.level_projections is None:
            return x_ref

        level_offsets = torch.as_tensor(level_offsets, device=x_ref.device, dtype=torch.long)

        # Handle batching
        if x_ref.dim() == 2:
            x_ref = x_ref.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        B, N, H = x_ref.shape
        device = x_ref.device

        # KEY FIX: Clone to avoid in-place modification of original tensor
        x_out = x_ref.clone()

        for lvl in range(max_level + 1):
            start = level_offsets[lvl].item()
            end = level_offsets[lvl + 1].item() if (lvl + 1) <= max_level else N
            num_nodes = end - start

            if num_nodes <= 0:
                continue

            # ===== PROCESS THIS LEVEL =====
            level_x = x_out[:, start:end, :].clone().reshape(B * num_nodes, H)

            if within_level_edges is not None and lvl in within_level_edges:
                level_edges = within_level_edges[lvl].to(device)
                if B > 1:
                    level_edges = torch.cat([level_edges + b * num_nodes for b in range(B)], dim=1)
            else:
                seq = torch.arange(num_nodes - 1, device=device)
                level_edges = torch.stack([seq, seq + 1], dim=0)
                level_edges = torch.cat([level_edges, level_edges.flip(0)], dim=1)
                if B > 1:
                    level_edges = torch.cat([level_edges + b * num_nodes for b in range(B)], dim=1)

            level_pos = torch.arange(num_nodes, device=device).repeat(B)
            level_nl = torch.full((B * num_nodes,), lvl, dtype=torch.long, device=device)

            for transformer in self.level_transformers[lvl]:
                if self.use_gradient_checkpointing and self.training:
                    level_x = torch.utils.checkpoint.checkpoint(
                        transformer, level_x, level_edges, level_nl,
                        None, level_pos, None, use_reentrant=False
                    )
                else:
                    level_x = transformer(
                        level_x, level_edges, level_nl,
                        level_offsets=None, positions=level_pos, edge_attr=None
                    )

            # Store processed features (now safe - writing to clone)
            x_out[:, start:end, :] = self.layer_norm(level_x).view(B, num_nodes, H)

            # ===== POOL TO NEXT LEVEL =====
            if lvl < max_level:
                lower = lvl
                higher_lvl = lvl + 1

                if lower >= len(self.compression_ratios):
                    continue

                cr = int(self.compression_ratios[lower])
                ov = float(self.overlap_ratios[lower])
                stride = max(1, int(cr * (1.0 - ov)))

                higher_start = level_offsets[higher_lvl].item()
                higher_end = level_offsets[higher_lvl + 1].item() if (higher_lvl + 1) <= max_level else N
                num_higher = higher_end - higher_start

                if num_higher <= 0:
                    continue

                proj_layer = self.level_projections[lower]

                # Vectorized pooling to avoid per-node loop
                pooled_all = self._pool_level_vectorized(
                    x_out[:, start:end, :],  # [B, num_lower, H]
                    num_higher, cr, stride, proj_layer
                )  # [B, num_higher, H]

                x_out[:, higher_start:higher_start + num_higher, :] = pooled_all

        if squeeze_back:
            return x_out.squeeze(0)
        return x_out


    def _pool_level_vectorized(self, lower_feats, num_higher, cr, stride, proj_layer):
        """
        Vectorized pooling from lower level to higher level.
        Avoids Python loop over higher nodes.
        
        Args:
            lower_feats: [B, num_lower, H]
            num_higher: number of higher-level nodes
            cr: compression ratio (window size)
            stride: stride between windows
            proj_layer: projection layer
        
        Returns:
            pooled: [B, num_higher, H]
        """
        B, num_lower, H = lower_feats.shape
        device = lower_feats.device

        # Compute window indices for each higher node
        h_indices = torch.arange(num_higher, device=device)
        win_starts = (h_indices * stride).clamp(max=num_lower - 1)
        win_ends = (win_starts + cr).clamp(max=num_lower)

        # Pad lower_feats for easy window extraction
        # Max window size is cr, so we need to handle variable window sizes
        pooled = torch.zeros(B, num_higher, H, device=device, dtype=lower_feats.dtype)
        counts = torch.zeros(B, num_higher, 1, device=device, dtype=lower_feats.dtype)

        # Use scatter_add for efficiency
        for h_idx in range(num_higher):
            ws = win_starts[h_idx].item()
            we = win_ends[h_idx].item()
            if ws >= we:
                we = ws + 1  # At least one element

            window = lower_feats[:, ws:we, :]  # [B, W, H]
            projected = proj_layer(window)      # [B, W, H]
            pooled[:, h_idx, :] = projected.mean(dim=1)

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hierarchical_features: bool = False, # Note: This flag is hard to support reliably now
        num_cycles: int = None,
        use_level_prediction: bool = False, # This flag is handled inside _process_hierarchical_graph
        imputation_mode: bool = False,
        imputation_idx: int = None,
        reveal_target_ids: Optional[torch.Tensor] = None,
        reveal_mask: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        logits_last_only: bool = False,  # AR generation: project only the final position
                                  # retrieval: build the bundle from THIS forward's base L3
                                  # (no separate no-grad forward, which perturbs training).
        logits_last_index: Optional[int] = None,  # frontier-consistent AR gen: project only this
                                  # position (the true frontier when the input is padded past it),
                                  # so the padded/settled hierarchy matches the training forward.
    ):
        """
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Optional position IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_hierarchical_features: Whether to return features from all levels
            num_cycles: Number of hierarchical processing cycles
            use_level_prediction: Whether to use level projection
            imputation_mode: Whether to run in token imputation mode
            imputation_idx: Index of token to impute (defaults to last token)
        
        Returns:
            If imputation_mode=False (normal mode):
                logits: Output token prediction logits [batch_size, seq_len, vocab_size]
                features: Optional hierarchical features if return_hierarchical_features=True
            If imputation_mode=True:
                token_logits: Logits for the imputed token [vocab_size]
                token_features: Features for the imputed token [hidden_dim]
        Forward pass. Handles standard prediction and single-token imputation for generation.
        Uses the appropriate refinement style defined during init.
        """
        # Determine batch size and sequence length early
        #print(f"Input IDs shape: {input_ids.shape}")
        if input_ids.ndim == 3:
            batch_size, seq_len, _ = input_ids.shape
        else:
            batch_size, seq_len = input_ids.shape
        self._runtime_l0_grid_shape = None

        self._last_autoenc_logits = None
        self._last_autoenc_query_logits = None
        self._last_token_unet_lookahead_logits = None

        device = input_ids.device
        
        
        #batch_size, seq_len = input_ids.shape
        #device = next(self.parameters()).device

        # if batch_size > 10000 and getattr(self, "enable_unified_skeleton_cache", False):
        #     # 1) build per-sample features & fetch skeletons
        #     sk_list = []
        #     x_list  = []
        #     total_levels = 1 + len(self.compression_ratios)

        #     for b in range(batch_size):
        #         # embeddings per sample
        #         tok_emb = self._get_embeddings(input_ids[b:b+1], position_ids[b:b+1] if position_ids is not None else None)
        #         tok_emb = tok_emb.reshape(-1, self.hidden_dim)  # [T,H]
        #         level_sizes = self._predict_level_sizes(tok_emb.size(0))
        #         ukey = self._unified_cache_key(level_sizes)
        #         sk = self._unified_skeleton_cache.get(ukey, None)

        #         if sk is None:
        #             # Build once (slow path) then cache skeleton. You already do this elsewhere; you can call that function.
        #             # For batched fast path, you can (optionally) bail out to your old slow path until cached.
        #             return self._forward_batch_slow(input_ids, position_ids, attention_mask)

        #         # Build per-sample concatenated x: L0 + zeros/mask for upper levels
        #         xs = [tok_emb]
        #         for lvl in range(1, total_levels):
        #             n = level_sizes[lvl]
        #             if self.input_mode == "tokens" and self.upper_init == "mask":
        #                 mask_vec = self.token_embedding(torch.tensor([self.mask_token_id], device=tok_emb.device))
        #                 xs.append(mask_vec.repeat(n, 1))
        #             else:
        #                 xs.append(torch.zeros(n, self.hidden_dim, device=tok_emb.device))
        #         x_cat = torch.cat(xs, dim=0)  # [sum(level_sizes), H]

        #         sk_list.append(sk)
        #         x_list.append(x_cat)

        #     # 2) pack into one big CPU graph
        #     batched_cpu, slices = self._pack_unified_batch(sk_list, x_list, device)

        #     # 3) run your **existing** batched refinement once
        #     x_init = batched_cpu.x  # CPU [N_total, H], can pin if you want
        #     executor = getattr(self, "_batched_executor", None)
        #     if executor is None:
        #         from .batched_layer_executor import BatchedLayerExecutor
        #         #self._batched_executor = BatchedLayerExecutor(default_batch_size=8192, pin_memory=True, num_workers=0, use_ckpt=self.use_gradient_checkpointing, use_amp=True)
        #         self._batched_executor = BatchedLayerExecutor()
        #         executor = self._batched_executor

        #     x_final = self._apply_unified_refinement_batched(
        #         unified_graph_cpu=batched_cpu,
        #         x_init=x_init,
        #         cycles=self.unified_refinement_cycles,
        #         batch_executor=executor,
        #         sweep_L0=True, sweep_L1=True, L1_iters=1, L2L3_iters=1,
        #     )
        #     # x_final: CPU [N_total, H]
        #     if self.final_norm.weight.device != x_final.device:
        #         self.final_norm = self.final_norm.to(x_final.device)

        #     x_final = self.final_norm(x_final)

        #     # # 4) slice L0 per sample and assemble [B,T,H]
        #     # feats_bt = []
        #     # for b in range(batch_size):
        #     #     # In your skeleton, L0 is always the first block of size seq_len
        #     #     start = slices["offsets"][b]
        #     #     end   = start + seq_len
        #     #     feats_bt.append(x_final[start:end].to(device, non_blocking=True))
        #     # feats_bt = torch.stack([f.view(seq_len, self.hidden_dim) for f in feats_bt], dim=0)  # [B,T,H]

            

        #     # 4) slice L0 per sample and assemble [B,T,H]
        #     feats_bt = []
        #     l0_slices = slices.get("l0_slices", None)

        #     for b in range(batch_size):
        #         if l0_slices is not None and len(l0_slices) == batch_size:
        #             s, e = l0_slices[b]
        #             # sanity: if empty (no level=0 detected), fall back to first seq_len
        #             if e <= s:
        #                 s = slices["offsets"][b]
        #                 e = min(s + seq_len, slices["offsets"][b+1])
        #         else:
        #             # last resort: assume first seq_len are L0 within this sample’s block
        #             s = slices["offsets"][b]
        #             e = min(s + seq_len, slices["offsets"][b+1])

        #         l0_feat = x_final[s:e].to(device, non_blocking=True)

        #         # align to seq_len if needed
        #         cur_len = e - s
        #         if cur_len > seq_len:
        #             l0_feat = l0_feat[:seq_len]
        #         elif cur_len < seq_len:
        #             pad = torch.zeros((seq_len - cur_len, l0_feat.size(1)), device=l0_feat.device, dtype=l0_feat.dtype)
        #             l0_feat = torch.cat([l0_feat, pad], dim=0)

        #         feats_bt.append(l0_feat)

        #     feats_bt = torch.stack(feats_bt, dim=0)  # [B,T,H]

        #     # 5) Output-mode logic (features/logits/both)
        #     #mode = self.get_output_mode()
        #     want_features = False#(self.training)
        #     want_logits   = True#(not self.training)

        #     if want_logits:
        #         logits = self.output_projection(feats_bt.view(-1, self.hidden_dim)).view(batch_size, seq_len, -1)

        #     if self.training:
        #         return feats_bt if want_features else logits
        #     else:
        #         return logits#{"features": feats_bt, "logits": logits}

        
        # --- Handle Batching (Recursive Calls) ---
        # Note: This recursive approach is simple but might be inefficient.
        # A proper batch implementation would process all items simultaneously.
        # Also, needs careful checking if imputation logic handles batch > 1 correctly internally.
        # if batch_size > 1:
        #     all_outputs = []
        #     for i in range(batch_size):
        #          # Recursive call for single item
        #          output = self.forward(
        #              input_ids[i:i+1],
        #              position_ids[i:i+1] if position_ids is not None else None,
        #              attention_mask[i:i+1] if attention_mask is not None else None,
        #              return_hierarchical_features=False, # Avoid collecting features in loop
        #              num_cycles=num_cycles,
        #              use_level_prediction=use_level_prediction, # Pass flag down
        #              imputation_mode=imputation_mode, # Pass flag down
        #              imputation_idx=imputation_idx # Pass index down
        #          )

        #          # Check return type to handle standard vs imputation mode returns
        #          if isinstance(output, tuple): # Imputation returns (logits, features)
        #              all_outputs.append(output[0]) # Append only logits for the single token
        #          elif output is not None: # Standard mode returns logits tensor [1, seq_len, vocab]
        #              all_outputs.append(output)
        #          # Else: Handle None return from error case?

        #     # Combine results if any outputs were generated
        #     if not all_outputs:
        #          logger.error("Batch processing yielded no outputs.")
        #          # Return something sensible, e.g., zeros or None
        #          # Shape for standard mode: [batch_size, seq_len, vocab_size]
        #          # Shape for imputation mode is tricky - maybe error?
        #          # Let's return None and let caller handle? Or zeros matching standard shape.
        #          return torch.zeros((batch_size, seq_len, self.vocab_size), device=device)


        #     # Concatenation depends on what single-item call returns in each mode
        #     if imputation_mode:
        #          # Each output is [vocab_size]. Stack them -> [batch_size, vocab_size]
        #          try:
        #               combined_output = torch.stack(all_outputs, dim=0)
        #               # Return format for batched imputation is ambiguous.
        #               # Maybe return logits + dummy features?
        #               # Let's return just the stacked logits for now.
        #               return combined_output
        #          except Exception as e:
        #               logger.error(f"Error stacking imputation outputs: {e}")
        #               return None # Indicate error
        #     else:
        #          # Each output is [1, seq_len, vocab_size]. Concatenate along batch dim.
        #          try:
        #               combined_logits = torch.cat(all_outputs, dim=0)
        #               if return_hierarchical_features:
        #                   # Cannot reliably get features from this batched approach
        #                   return combined_logits, []
        #               else:
        #                   return combined_logits
        #          except Exception as e:
        #               logger.error(f"Error concatenating standard outputs: {e}")
        #               # Return None or zeros matching shape
        #               return torch.zeros((batch_size, seq_len, self.vocab_size), device=device)

        # --- Processing for Batch Size = 1 ---
        
        # Use refinement cycles from init if not overridden
        cycles = num_cycles if num_cycles is not None else self.refinement_cycles
        #print(input_ids.shape, " shape of input ids")
        #print(input_ids.dtype, " dtype of input ids")
        #print(input_ids[0,:10], " sample input ids")
        # --- Get Initial Embeddings ---
        if input_ids.ndim == 3:                  # we already have embeddings
            token_embeddings_dense = self._get_embeddings(input_ids, position_ids)         # (B, L, H)
        else:                                    # classic token IDs
            token_embeddings_dense = self._get_embeddings(input_ids, position_ids)#, max_len=seq_len)  # (B, L, H)

        cond_vec = self._compute_conditioning_vector(
            batch_size=int(token_embeddings_dense.size(0)),
            device=token_embeddings_dense.device,
            class_labels=class_labels,
            timesteps=timesteps,
        )
        token_embeddings_dense = self._apply_film_conditioning(token_embeddings_dense, cond_vec)

        # Get batch size and sequence lengths.
        batch_size, seq_len_dense, _ = token_embeddings_dense.shape
        dense_grid_shape = self._resolve_l0_grid_shape_for_tokens(int(seq_len_dense))
        token_embeddings_bt, token_unet_decode_context = self._token_unet_encode_for_graph(
            token_embeddings_dense,
            grid_shape=dense_grid_shape,
        )
        seq_len_graph = int(token_embeddings_bt.size(1))
        runtime_graph_shape = dense_grid_shape
        if isinstance(token_unet_decode_context, dict):
            coarse_grid = token_unet_decode_context.get("graph_grid_shape", None)
            if coarse_grid is None:
                coarse_grid = token_unet_decode_context.get("coarse_grid_shape", None)
            if coarse_grid is not None:
                gh = int(coarse_grid[0])
                gw = int(coarse_grid[1])
                if gh > 0 and gw > 0 and gh * gw == int(seq_len_graph):
                    runtime_graph_shape = (gh, gw)
        if runtime_graph_shape is not None:
            self._runtime_l0_grid_shape = (int(runtime_graph_shape[0]), int(runtime_graph_shape[1]))
        else:
            self._runtime_l0_grid_shape = None

        forward_graph_key = (
            int(seq_len_dense),
            int(seq_len_graph),
            tuple(int(v) for v in dense_grid_shape) if dense_grid_shape is not None else None,
            tuple(int(v) for v in runtime_graph_shape) if runtime_graph_shape is not None else None,
            bool(getattr(self, "token_unet_is_2d", False)),
            bool(getattr(self, "enable_unified_skeleton_cache", False)),
        )
        if (
            dense_grid_shape is not None
            or runtime_graph_shape is not None
            or bool(getattr(self, "token_unet_is_2d", False))
        ) and forward_graph_key != getattr(self, "_last_forward_graph_log_key", None):
            logger.info(
                "[EHFGAT:FWD] layout dense=%s graph=%s seq_len_dense=%d seq_len_graph=%d token_unet_2d=%s cache_enabled=%s",
                dense_grid_shape,
                runtime_graph_shape,
                int(seq_len_dense),
                int(seq_len_graph),
                bool(getattr(self, "token_unet_is_2d", False)),
                bool(getattr(self, "enable_unified_skeleton_cache", False)),
            )
            self._last_forward_graph_log_key = forward_graph_key

        reveal_target_ids_graph = reveal_target_ids
        reveal_mask_graph = reveal_mask
        if token_unet_decode_context is not None:
            reveal_target_ids_graph, reveal_mask_graph, _ = self._map_reveal_targets_to_coarse(
                reveal_target_ids=reveal_target_ids,
                reveal_mask=reveal_mask,
                coarse_len=seq_len_graph,
            )

        #print(token_embeddings_bt.shape, " shape of token embeddings")
        # Flatten for graph processing (legacy/slow path expects this shape)
        token_embeddings = token_embeddings_bt.reshape(-1, self.hidden_dim)
        
        # === FAST PATH (Enhanced): reuse cached unified skeleton and skip Stage-1 ===
        if getattr(self, "enable_unified_skeleton_cache", False):
            level_sizes = self._predict_level_sizes(seq_len_graph)
            ukey = self._unified_cache_key(level_sizes)
            sk = self._unified_skeleton_cache.get(ukey, None)
            self._uf_cache_last_seq_len = int(seq_len_graph)
            self._uf_cache_last_key = ukey

            if sk is not None:
                self._uf_cache_fast_hits += 1
                # Build x for all levels (L0 = tokens, upper = mask/zeros)
                device = token_embeddings.device
                
                x_cat = [token_embeddings_bt[0, :seq_len_graph, :]]
                for lvl in range(1, len(level_sizes)):
                    n = level_sizes[lvl]
                    if self.input_mode == "tokens" and self.upper_init in ("mask", "pooled"):
                        mask_vec = self.token_embedding(torch.tensor([self.mask_token_id], device=device))
                        x_cat.append(mask_vec.repeat(n, 1))
                    else:
                        x_cat.append(torch.zeros(n, self.hidden_dim, device=device))

                ae_slice = sk.get("ae_decoder_l0_slice", (0, 0)) if isinstance(sk, dict) else (0, 0)
                if ae_slice != (0, 0) and len(level_sizes) >= 3:
                    for n in (level_sizes[0], level_sizes[1], level_sizes[2]):
                        if self.input_mode == "tokens" and self.upper_init in ("mask", "pooled"):
                            mask_vec = self.token_embedding(torch.tensor([self.mask_token_id], device=device))
                            x_cat.append(mask_vec.repeat(int(n), 1))
                        else:
                            x_cat.append(torch.zeros(int(n), self.hidden_dim, device=device))
                x_cat = torch.cat(x_cat, dim=0)
                #print(f"x_cat shape: {x_cat.shape}")

            
                # torch.cuda.reset_peak_memory_stats(device)
                # torch.cuda.synchronize(device)
                # torch.cuda.empty_cache()
                # alloc = torch.cuda.memory_allocated(device) / 1e9
                # peak  = torch.cuda.max_memory_allocated(device) / 1e9
                # print(f"[MEM] start: alloc={alloc:.2f} GB, peak={peak:.2f} GB")

                # Rehydrate on CPU (important)
                #unified_graph = self._rehydrate_unified(sk, x_cat[:0], device=torch.device("cpu"))
                with torch.no_grad():
                    #unified_graph = self._rehydrate_unified(sk, None, device=torch.device("cpu"))
                    #print(device, " device for rehydration")
                    #print("Rehydrating unified graph from skeleton cache...")
                    t_rehydrate_start = time.time()
                    unified_graph = self._rehydrate_unified(sk, None, device=device, cache_key=ukey)
                    self._uf_cache_last_rehydrate_ms = (time.time() - t_rehydrate_start) * 1000.0
                    unified_graph.num_nodes = x_cat.size(0)
                    unified_graph.x = None  # do not store full x in Data (we keep it as x_init on CPU)

                    # torch.cuda.reset_peak_memory_stats(device)
                    # torch.cuda.synchronize(device)
                    # torch.cuda.empty_cache()
                    # alloc = torch.cuda.memory_allocated(device) / 1e9
                    # peak  = torch.cuda.max_memory_allocated(device) / 1e9
                    # print(f"[MEM] post-hydrate: alloc={alloc:.2f} GB, peak={peak:.2f} GB")

                    # CPU x_init (pinned if possible)
                    #x_init = x_cat.detach().cpu()
                #x_init = x_cat#.to(torch.bfloat16)#.cpu()#.detach() # removed detach
                #print(f"x_init shape before expand: {x_init.shape}")
                # Expand per-batch graph features and inject per-sample L0 tokens.
                x_cat = x_cat.unsqueeze(0).repeat(batch_size, 1, 1)
                x_cat[:, :seq_len_graph, :] = token_embeddings_bt[:, :seq_len_graph, :]
                if self.upper_init == "pooled":
                    # Seed coarse levels bottom-up from the per-sample tokens (gated; causal:
                    # a coarse node's ar_time is its window END). Gives every refinement layer
                    # informative coarse nodes instead of filling the pyramid layer-by-layer.
                    x_cat = self._apply_pooled_upper_seed(x_cat, level_sizes)
                #print(f"x_init reshaped to: {x_init.shape}")
                # Now add token_embeddings_full to the L0 slice, by taking the token_embeddings_full seq_len and replacing x_init[:,:seq_len,:]
                # Use view to expand token_embeddings to match batch size
                #token_embeddings_full = token_embeddings.view(batch_size, seq_len, self.hidden_dim)
                #x_init[:, :seq_len, :] += token_embeddings_full.to(x_init.device, non_blocking=True)
                # try:
                #     x_init = x_init.pin_memory()
                # except RuntimeError:
                #     pass
                #if self.use_edge_attr and getattr(self, "edge_feature_generator", None) is not None \
                #and unified_graph.edge_index.numel() > 0:
                #    try:
                        # Edge feature generator only needs x stats if it is x-dependent.
                        # If independent (type-only), pass a dummy (e.g., zeros) or reuse x_init on CPU.
                        # Here we assume it uses x; pass x_init on CPU.
                        # get gpu device for edge feature generation
                        #edge_device = next(self.edge_feature_generator.parameters()).device
                        #unified_graph.edge_attr = self.edge_feature_generator(
                        #    x_init.to(edge_device), unified_graph.edge_index.to(edge_device), unified_graph.edge_type.to(edge_device)
                        #).to("cpu")
                        #edge_device = next(self.edge_feature_generator.parameters()).device
                        #unified_graph.edge_attr = self.edge_feature_generator(
                        #    x_init, unified_graph.edge_index, unified_graph.edge_type
                        #)#.to("cpu")
                    #except Exception as e:
                    #    logger.warning(f"[EHFGAT:BATCH] edge_attr regen on CPU failed: {e}", exc_info=True)
                # torch.cuda.reset_peak_memory_stats(device)
                # torch.cuda.synchronize(device)
                # torch.cuda.empty_cache()
                # alloc = torch.cuda.memory_allocated(device) / 1e9
                # peak  = torch.cuda.max_memory_allocated(devi#ce) / 1e9
                # print(f"[MEM] post-x_init: alloc={alloc:.2f} GB, peak={peak:.2f} GB")


                # Batched, layer-wise unified refinement
                # Apply unified refinement
                unified_graph.x = x_cat  # CPU [N_total, H]
                #print(unified_graph.x.shape," shape of x_init going into refinement")
                refined_graph = self._apply_unified_refinement(
                    unified_graph,
                    self.unified_refinement_cycles,
                    reveal_target_ids=reveal_target_ids_graph,
                    reveal_mask=reveal_mask_graph,
                    cond_vec=cond_vec,
                )
                x_final = refined_graph.x  # CPU [N_total, H]


                # x_final = self._apply_unified_refinement_batched(
                #     unified_graph_cpu=unified_graph,
                #     x_init=x_init,
                #     cycles=self.unified_refinement_cycles if num_cycles is None else num_cycles,
                #     batch_executor=self._batched_executor,
                #     sweep_L0=True,
                #     sweep_L1=True,
                #     L1_iters=1,
                #     L2L3_iters=1,
                # )

                # if self.final_norm.weight.device != x_final.device:
                #     self.final_norm = self.final_norm.to(x_final.device)
                
                # x_final = self.final_norm(x_final)
                # torch.cuda.reset_peak_memory_stats(device)
                # torch.cuda.synchronize(device)
                # torch.cuda.empty_cache()
                # alloc = torch.cuda.memory_allocated(device) / 1e9
                # peak  = torch.cuda.max_memory_allocated(device) / 1e9
                # print(f"[MEM] post-refinement: alloc={alloc:.2f} GB, peak={peak:.2f} GB")
                # Predict from L0 slice (x_final is CPU)
                #print(f"x_final shape: {x_final.shape}")
                #print(seq_len, " seq_len value")
                token_features_graph = x_final[:, :seq_len_graph, :]
                token_features = token_features_graph.to(self.output_projection.weight.device, non_blocking=True)
                # Coarse co-prediction: fold the strictly-past L1/L2/L3 summaries into L0's
                # pre-head features (causal, gate-init 0 = identity). At graph resolution so the
                # gather aligns with node_ar_time (before any token-unet decode to dense).
                if getattr(self, "hier_copredict_l0", False):
                    token_features = self._copredict_inject(
                        token_features,
                        x_final,
                        getattr(refined_graph, "level_offsets", None),
                        getattr(refined_graph, "node_ar_time", None),
                        int(seq_len_graph),
                    )
                token_unet_lookahead_features = None
                need_token_unet_lookahead_logits = bool(getattr(self, "_token_unet_emit_lookahead_logits", False))
                if token_unet_decode_context is not None:
                    token_features, token_unet_lookahead_features = self._token_unet_decode_from_graph(
                        token_features,
                        token_unet_decode_context,
                    )
                    if need_token_unet_lookahead_logits and token_unet_lookahead_features is not None:
                        lookahead_logits_full = self.output_projection(token_unet_lookahead_features).view(batch_size, seq_len_dense, -1)
                        self._last_token_unet_lookahead_logits = lookahead_logits_full

                ae_slice = getattr(refined_graph, "ae_decoder_l0_slice", None)
                if ae_slice is None and isinstance(sk, dict):
                    ae_slice = sk.get("ae_decoder_l0_slice", (0, 0))
                if ae_slice is not None:
                    ds, de = int(ae_slice[0]), int(ae_slice[1])
                    if de > ds:
                        dec_feats = x_final[:, ds:de, :].to(self.output_projection.weight.device, non_blocking=True)
                        if token_unet_decode_context is not None:
                            dec_feats, _ = self._token_unet_decode_from_graph(dec_feats, token_unet_decode_context)
                        dec_len = int(dec_feats.size(1))
                        if dec_len > seq_len_dense:
                            dec_feats = dec_feats[:, :seq_len_dense, :]
                        elif dec_len < seq_len_dense:
                            pad = torch.zeros(
                                (batch_size, seq_len_dense - dec_len, dec_feats.size(-1)),
                                device=dec_feats.device,
                                dtype=dec_feats.dtype,
                            )
                            dec_feats = torch.cat([dec_feats, pad], dim=1)
                        ae_logits_full = self.output_projection(dec_feats).view(batch_size, seq_len_dense, -1)
                        self._last_autoenc_logits = ae_logits_full
                        if ae_logits_full.size(1) > 0:
                            self._last_autoenc_query_logits = ae_logits_full[:, -1, :]

                # 4) slice L0 per sample and assemble [B,T,H]
                # feats_bt = []
                # for b in range(batch_size):
                #     if slices["l0_slices"] is not None:
                #         s, e = slices["l0_slices"][b]
                #         feats_bt.append(x_final[s:e].to(device, non_blocking=True))
                #     else:
                #         # fallback: assume first T nodes of each sample are L0
                #         start = slices["offsets"][b]
                #         end   = start + seq_len
                #         feats_bt.append(x_final[start:end].to(device, non_blocking=True))

                # feats_bt = torch.stack([f.view(-1, self.hidden_dim) for f in feats_bt], dim=0)  # [B,T,H]

                #if getattr(self, "emit_features_only", False):
                #if self.training:
                    # Return features shaped [B, T, H] to the trainer
                    #print(token_features.shape)
                    #feats = token_features.view(batch_size, seq_len, self.hidden_dim)
                    #print("Returning token features from EHFGAT fast path.")
                    #return feats
                #print(batch_size, seq_len, self.hidden_dim)
                # Chunked-CE loss head: when the trainer flags return_token_features (set only
                # for the chunked-CE training forward, reset right after — so validation and
                # generation are unaffected) hand back the [B,T,hidden] features instead of
                # projecting the full [B,T,vocab] logits here. The trainer then projects +
                # cross-entropies in seq-chunks, never materializing the full logits (+ its
                # softmax/CE backward) -> ~3.3GB saved at B8/T1024/V50257. Not for generation
                # (logits_last_only), hierarchical-feature returns, or the AE decode head.
                if (bool(getattr(self, "return_token_features", False))
                        and not logits_last_only and not return_hierarchical_features
                        and getattr(self, "_force_decode_head", None) != "ae"):
                    return token_features.reshape(batch_size, seq_len_dense, self.hidden_dim)
                if logits_last_only or (logits_last_index is not None):
                    # AR generation reads only one position's logits; projecting the whole sequence
                    # each step is O(T*vocab) compute + a [B,T,vocab] tensor (~1.5 GiB at T=4608)
                    # wasted. Project just that position -> [B,1,vocab]. logits_last_index picks the
                    # true frontier when the input is padded past it (frontier-consistent decode).
                    _proj_idx = int(seq_len_dense) - 1 if logits_last_index is None else int(logits_last_index)
                    _proj_idx = max(0, min(_proj_idx, int(seq_len_dense) - 1))
                    logits = self.output_projection(token_features[:, _proj_idx:_proj_idx + 1, :]).view(batch_size, 1, -1)
                else:
                    logits = self.output_projection(token_features).view(batch_size, seq_len_dense, -1)
                if getattr(self, "_force_decode_head", None) == "ae":
                    ae_logits = getattr(self, "_last_autoenc_logits", None)
                    if ae_logits is not None and tuple(ae_logits.shape[:2]) == tuple(logits.shape[:2]):
                        logits = ae_logits.to(device=logits.device, dtype=logits.dtype)
                self._maybe_log_gate_monitor()
                #print(logits.shape, " logits shape after projection")
                if not getattr(self, "_fp_log_once", False):
                    logger.info(
                        "[EHFGAT:FAST] Reused skeleton seq_len_graph=%d level_sizes=%s runtime_grid=%s cache_hits=%d seeds=%d last_build_ms=%s",
                        int(seq_len_graph),
                        tuple(int(s) for s in level_sizes),
                        runtime_graph_shape,
                        int(self._uf_cache_fast_hits),
                        int(self._uf_cache_seed_count),
                        "n/a" if self._uf_cache_last_build_ms is None else f"{float(self._uf_cache_last_build_ms):.1f}",
                    )
                    self._fp_log_once = True

                if return_hierarchical_features:
                    # If you need level features, you can slice via sk["level_offsets"]
                    feats = []
                    loff = sk.get("level_offsets", None)
                    if loff and len(loff) > 1:
                        for i in range(len(loff)-1):
                            s, e = loff[i], loff[i+1]
                            feats.append(x_final[s:e])
                    else:
                        feats = [x_final[:seq_len_graph]]
                    return logits, feats

                return logits

            self._uf_cache_fast_misses += 1
            self._uf_cache_last_miss_reason = "key_not_found"

        # === END FAST PATH ===


        # --- Get Initial Embeddings ---
        # Note: input_ids might contain the <MASK> token at the end if in imputation mode
        actual_len = int(seq_len_graph)
        #print(f" actual_len: {actual_len}")
        #token_embeddings = self._get_embeddings(input_ids, position_ids)
        # Reshape for graph processing: [1, actual_len, hidden_dim] -> [actual_len, hidden_dim]
        #token_embeddings = token_embeddings.view(-1, self.hidden_dim)
        # ---

        # --- Process Graph ---
        # _process_hierarchical_graph now performs imputation pooling if needed,
        # applies the chosen refinement style (iterative or unified),
        # and returns the FINAL processed token features (potentially fused with high-level context).
        # Expected return shape: [final_num_tokens, hidden_dim] (usually final_num_tokens == actual_len)
        #print(f" token_embeddings shape before graph processing: {token_embeddings.shape}")
        # reshape batch into token embeddings
        token_embeddings = token_embeddings_bt.view(batch_size, actual_len, self.hidden_dim)
        final_token_features = self._process_hierarchical_graph(
            token_embeddings=token_embeddings, # Shape: [actual_len, hidden_dim]
            #cycles=cycles, # Pass refinement cycles count
            use_level_prediction=use_level_prediction # Pass flag
        )
        # Check if the slow path just built the cache and requests a restart
        if final_token_features == "RESTART_FORWARD":
            self._uf_cache_restart_count += 1
            # Clear memory used by graph building before restarting
            del token_embeddings
            
            # Recursive call: This will now hit the "FAST PATH" because the cache exists
            return self.forward(
                input_ids, position_ids, attention_mask,
                return_hierarchical_features, num_cycles,
                use_level_prediction, imputation_mode, imputation_idx,
                reveal_target_ids, reveal_mask,
                class_labels=class_labels,
                timesteps=timesteps,
                logits_last_only=logits_last_only,
                logits_last_index=logits_last_index,
            )
        # ---

        # --- Handle Potential Errors from Graph Processing ---
        # if final_token_features is None:
        #     logger.error(f"Graph processing returned None for input shape {input_ids.shape}. Cannot generate logits.")
        #     if imputation_mode:
        #          # Need to return tuple for generator
        #          return torch.empty(0, device=device), torch.empty(0, device=device)
        #     else:
        #          # Return zero logits matching expected output shape
        #          return torch.zeros((batch_size, seq_len, self.vocab_size), device=device)
        # # ---

        # # --- Output Generation ---
        # if imputation_mode:
        #     # This mode is used by _generate_with_imputation for inference/sampling
        #     if imputation_idx is None: imputation_idx = -1 # Default to last token (the mask)

        #     # Select features for the target index
        #     try:
        #         # final_token_features should have shape [actual_len, hidden_dim]
        #         imputed_token_features = final_token_features[imputation_idx] # Shape: [hidden_dim]
        #     except IndexError:
        #         logger.error(f"imputation_idx {imputation_idx} out of bounds for final_token_features size {final_token_features.size(0)}")
        #         return torch.empty(0, device=device), torch.empty(0, device=device) # Return empty tuple

        #     # Project selected features to vocabulary
        #     imputed_token_logits = self.output_projection(imputed_token_features) # Shape: [vocab_size]

        #     # Return format expected by _generate_with_imputation: (logits, features)
        #     return imputed_token_logits, imputed_token_features

        # else:
        #     # Standard mode: Predict for all original positions
        #     # final_token_features shape: [actual_len, hidden_dim]
        #     # Need to ensure length matches original seq_len if different
        #     processed_len = final_token_features.size(1)
        #     if processed_len != seq_len:
        #          logger.warning(f"Output feature length ({processed_len}) differs from input seq_len ({seq_len}). Truncating/Padding output.")
        #          if processed_len > seq_len:
        #               final_token_features = final_token_features[:seq_len, :]
        #          else: # processed_len < seq_len
        #               # This shouldn't happen if graph processing is correct
        #               # Pad with zeros? Risky. Error might be better.
        #               padding = torch.zeros((seq_len - processed_len, self.hidden_dim), device=device, dtype=final_token_features.dtype)
        #               final_token_features = torch.cat([final_token_features, padding], dim=0)
        #     # If caller wants features, return [B,T,H] and stop
        #     #if getattr(self, "emit_features_only", False):
        #     if self.training:
        #         #print(batch_size, seq_len, self.hidden_dim)
        #         feats = final_token_features.view(batch_size, seq_len, self.hidden_dim)
        #         #print("Returning token features from EHFGAT.")
        #         # logger.info("Returning token features from EHFGAT slow path.")  # optional
        #         #return feats
        #     # Project all token features to vocabulary
        #     logits = self.output_projection(final_token_features) # Shape: [seq_len, vocab_size]

        #     # Reshape to [batch_size, seq_len, vocab_size]
        #     logits = logits.view(batch_size, seq_len, self.vocab_size)

        #     # Handle return_hierarchical_features (difficult with new structure)
        #     if return_hierarchical_features:
        #          logger.warning("return_hierarchical_features=True is not reliably supported with current architecture.")
        #          # Cannot easily return hierarchical_features list from here
        #          return logits, [] # Return empty list placeholder
        #     else:
        #          return logits # Return only logits
    
    def _process_hierarchical_graph(self, token_embeddings, use_level_prediction):
        """
        Processes the graph using imputation pooling and the two-stage refinement.
        'cycles' argument is ignored, uses self.iterative_refinement_cycles and self.unified_refinement_cycles.
        """
        with torch.no_grad():
            token_embeddings_full = token_embeddings
            #print(token_embeddings_full.shape, " shape of token_embeddings_full in _process_hierarchical_graph")
            self.level_mappings = [] # Reset for this forward pass
            if token_embeddings.ndim == 3:
                batch_size, seq_len, _ = token_embeddings.shape
            else:
                # add batch dimension
                #token_embeddings = token_embeddings.unsqueeze(0)
                batch_size, seq_len = token_embeddings.shape
            token_embeddings = token_embeddings[0,:,:] if token_embeddings.ndim == 3 else token_embeddings
            #print(token_embeddings.shape, " shape of token_embeddings in _process_hierarchical_graph")
            # --- Step 1: Build L0 & L0 Cycles ---
            l0_graph = self._build_level_graph(token_embeddings, level_idx=0)
            if self.l0_cycles > 0: logger.debug(f"Running {self.l0_cycles} L0 cycles...")
            for _ in range(self.l0_cycles):
                l0_graph = self._process_level(l0_graph, level_idx=0)
            level_graphs = [l0_graph]
            current_processed_graph = l0_graph
            # --- End Step 1 ---

            # --- Step 2: Build Hierarchy & Impute Levels ---
            for level_idx in range(1, self.num_hier_levels):
                if level_idx >= len(self.level_transformers) or level_idx - 1 >= len(self.compression_ratios) or level_idx - 1 >= len(self.overlap_ratios):
                    logger.warning(f"Configuration insufficient for level {level_idx}. Stopping hierarchy build.")
                    break

                # 2a. Create next level initialized with MASK
                next_level_graph_init, level_mapping = self._create_next_level(
                    current_processed_graph, level_idx=level_idx,
                    compression_ratio=self.compression_ratios[level_idx-1],
                    overlap_ratio=self.overlap_ratios[level_idx-1]
                )
                self.level_mappings.append(level_mapping)

                # 2b. Build subgraph connecting L_i and L_{i+1}_init
                subgraph = self._build_level_subgraph(current_processed_graph, next_level_graph_init, level_mapping)

                # 2c. Run Imputation Cycles
                internal_cycles = self.internal_cycles_per_level[level_idx-1] if level_idx-1 < len(self.internal_cycles_per_level) else 1
                if internal_cycles > 0 : logger.debug(f"Running {internal_cycles} imputation cycles for L{level_idx}")
                for _ in range(internal_cycles):
                    # Ensure _process_subgraph exists and uses appropriate transformers
                    subgraph = self._process_subgraph(subgraph, level_idx)

                # 2d. Extract Refined Features for L_{i+1}
                next_level_offset = current_processed_graph.num_nodes
                if subgraph.x.size(0) < next_level_offset + next_level_graph_init.num_nodes:
                    logger.error(f"Subgraph feature tensor size error L{level_idx}.")
                    break

                next_level_graph_refined = Data(
                    x=subgraph.x[next_level_offset:],
                    edge_index=next_level_graph_init.edge_index,
                    node_level=next_level_graph_init.node_level,
                    num_nodes=next_level_graph_init.num_nodes, # Add num_nodes
                    **( {attr: getattr(next_level_graph_init, attr) for attr in next_level_graph_init.keys() if attr not in ['x', 'edge_index', 'node_level', 'num_nodes']} )
                )
                level_graphs.append(next_level_graph_refined)
                current_processed_graph = next_level_graph_refined
            # --- End Step 2 ---


            # Stage 1 (legacy iterative-level refinement) removed; unified refinement is the path.
            hierarchical_features = [g.x for g in level_graphs]
            for g in level_graphs:
                g.x = None # free memory


            # === STEP 4: STAGE 2 - Optional Unified Graph Refinement ===
            if self.unified_refinement_cycles > 0:
                logger.debug(f"Running Stage 2: Unified graph refinement ({self.unified_refinement_cycles} cycles)...")
                if not hasattr(self, '_build_unified_graph') or not hasattr(self, '_apply_unified_refinement_batched'):
                    raise NotImplementedError("Unified refinement helpers are missing")

                # Rebuild Graph Objects with refined features from Stage 1
                refined_level_graphs = []
                for i, features in enumerate(hierarchical_features):
                    if i < len(level_graphs):
                        level_graph = level_graphs[i]
                        temp_graph = Data(
                            x=features,                                 # likely on CUDA
                            edge_index=level_graph.edge_index,          # follow level_graphs device
                            node_level=level_graph.node_level,
                            num_nodes=features.size(0),
                            **{
                                attr: getattr(level_graph, attr)
                                for attr in level_graph.keys()
                                if attr not in ["x", "edge_index", "node_level", "num_nodes"]
                            },
                        )
                        refined_level_graphs.append(temp_graph)
                    else:
                        logger.error(f"Mismatch feature/graph list length at index {i}")
                        return None

                # Build unified graph (likely resides on CUDA right now)
                unified_graph = self._build_unified_graph(refined_level_graphs, self.level_mappings)
                
                # -------- NEW: precompute LapPE and cache on CPU --------
                if self.lap_pe_transform is not None and getattr(self, "lap_pe_k", 0) > 0:
                    try:
                        # 1) apply LapPE transform on CPU
                        g_cpu_tmp = Data(
                            edge_index=unified_graph.edge_index,
                            num_nodes=unified_graph.num_nodes,
                        )
                        g_cpu_tmp = self.lap_pe_transform(g_cpu_tmp)
                        lap_raw = g_cpu_tmp.lap_pe.detach().to(dtype=torch.float32, device="cpu").contiguous()
                        #lap_raw = g_cpu_tmp.lap_pe.to(dtype=torch.float32).contiguous()

                        # 2) store both raw and projected (optional)
                        unified_graph.lap_pe_raw_cpu = lap_raw
                        #if hasattr(self, "lap_pe_proj"):
                            # unified_graph_cpu.lap_pe_proj_cpu = (
                            #     self.lap_pe_proj.to("cpu")(lap_raw).detach().contiguous()
                            # )
                        #    Wdev = self.lap_pe_proj.weight.device  # e.g., cuda:1
                            #with torch.no_grad():
                        #    pe_proj = self.lap_pe_proj(lap_raw.to(Wdev, non_blocking=True))  # compute on CUDA
                            #unified_graph_cpu.lap_pe_proj_cpu = pe_proj.to("cpu", non_blocking=True).contiguous()
                        #    unified_graph_cpu.lap_pe_proj_cpu = pe_proj.to(Wdev).contiguous()
                    except Exception as e:
                        logger.warning(f"[LapPE cache] failed: {e}")

                # Seed unified skeleton cache for this level layout
                if getattr(self, "enable_unified_skeleton_cache", False):
                    try:
                        level_sizes = [g.x.size(0) for g in refined_level_graphs]
                        ukey = self._unified_cache_key(level_sizes)
                        if ukey not in self._unified_skeleton_cache:
                            self._unified_skeleton_cache[ukey] = self._skeletonize_unified(unified_graph)
                            self._evict_skeleton_cache(self._unified_skeleton_cache, self._skeleton_cache_max)
                            self._uf_cache_seed_count += 1
                            self._uf_cache_last_build_ms = None
                            try:
                                # Unpack the cache key with correct indices based on _unified_cache_key tuple structure
                                # Indices: 0=level_sizes,1=strides,2=num_levels,3=graph_geometry_mode,4=gh,5=gw,
                                # 6=graph_spatial_metric,7=graph_downsample_factor,8=use_edge_attr,9=add_self_loops,
                                # 10=long_range_distance,11=hier_ar_enable,12=hier_ar_allow_same_time,13=l0_ar_enable,
                                # 14=enable_l0_parent_edges,15=l0_parent_edges_bidirectional,16=ensure_l0_past_l1_edges,
                                # 17=ensure_past_hier_edges_all_levels,18=autoenc_graph_mode,19=autoenc_coupled_feedback
                                if len(ukey) >= 20:
                                    if self.verbose:
                                        logger.info(
                                            "[EHFGAT:CACHE] Seeded unified skeleton: "
                                            "level_sizes=%s strides=%s num_levels=%d graph_geometry_mode=%s grid_height=%d grid_width=%d "
                                            "graph_spatial_metric=%s graph_downsample_factor=%d use_edge_attr=%s add_self_loops=%s "
                                            "long_range_distance=%d hier_ar_enable=%s hier_ar_allow_same_time=%s "
                                            "l0_ar_enable=%s enable_l0_parent_edges=%s l0_parent_edges_bidirectional=%s "
                                            "ensure_l0_past_l1_edges=%s ensure_past_hier_edges_all_levels=%s autoenc_graph_mode=%s autoenc_coupled_feedback=%s",
                                            ukey[0],
                                            ukey[1],
                                            int(ukey[2]),
                                            str(ukey[3]),
                                            int(ukey[4]),
                                            int(ukey[5]),
                                            str(ukey[6]),
                                            int(ukey[7]),
                                            bool(ukey[8]),
                                            bool(ukey[9]),
                                            int(ukey[10]),
                                            bool(ukey[11]),
                                            bool(ukey[12]),
                                            bool(ukey[13]),
                                            bool(ukey[14]),
                                            bool(ukey[15]),
                                            bool(ukey[16]),
                                            bool(ukey[17]),
                                            str(ukey[18]),
                                            bool(ukey[19]),
                                        )
                                else:
                                    if self.verbose:
                                        logger.info("[EHFGAT:CACHE] Seeded unified skeleton (key too short for detailed logging)")
                            except Exception:
                                if self.verbose:
                                    logger.info("[EHFGAT:CACHE] Seeded unified skeleton (detailed key unavailable)")
                    except Exception as e:
                        logger.debug(f"[EHFGAT:CACHE] seeding skipped: {e}")
                del g, refined_level_graphs

                del unified_graph#, refined_level_graphs
                return "RESTART_FORWARD"
                # for g in refined_level_graphs:
                #    g.x = None  # free memory
                # -------- NEW: CPU-ize the skeleton + pin x --------
                # 1) Build a CPU-only skeleton (NO x on GPU here)
        #         unified_graph_cpu = Data(
        #             edge_index=unified_graph.edge_index.detach().to("cpu", non_blocking=True),
        #             node_level=unified_graph.node_level.detach().to("cpu", non_blocking=True)
        #                 if hasattr(unified_graph, "node_level") and unified_graph.node_level is not None else None,
        #             edge_type=unified_graph.edge_type.detach().to("cpu", non_blocking=True)
        #                 if hasattr(unified_graph, "edge_type") and unified_graph.edge_type is not None else None,
        #             num_nodes=int(unified_graph.num_nodes) if hasattr(unified_graph, "num_nodes")
        #                 else int(unified_graph.x.size(0)),
        #             level_offsets=list(getattr(unified_graph, "level_offsets", [])),
        #             lap_pe_raw_cpu=unified_graph.lap_pe_raw_cpu if hasattr(unified_graph, "lap_pe_raw_cpu") and unified_graph.lap_pe_raw_cpu is not None else None,
        #         ).detach().to("cpu")
        #         unified_graph_cpu.x = None  # do not store full x in Data (we keep it as x_init on CPU)
        #         # target_device = next(self.parameters()).device
        #         # unified_graph_cpu = Data(
        #         #     edge_index=unified_graph.edge_index.detach().to(target_device, non_blocking=True),
        #         #     node_level=unified_graph.node_level.detach().to(target_device, non_blocking=True)
        #         #         if hasattr(unified_graph, "node_level") and unified_graph.node_level is not None else None,
        #         #     edge_type=unified_graph.edge_type.detach().to(target_device, non_blocking=True)
        #         #         if hasattr(unified_graph, "edge_type") and unified_graph.edge_type is not None else None,
        #         #     num_nodes=int(unified_graph.num_nodes) if hasattr(unified_graph, "num_nodes")
        #         #         else int(unified_graph.x.size(0)),
        #         #     level_offsets=list(getattr(unified_graph, "level_offsets", [])),
        #         #     lap_pe_raw_cpu=unified_graph.lap_pe_raw_cpu if hasattr(unified_graph, "lap_pe_raw_cpu") and unified_graph.lap_pe_raw_cpu is not None else None,
        #         # )#.to(target_device, non_blocking=True)
        #         # unified_graph_cpu.x = None  # do not store full x in Data (we keep it as x_init on CPU)

        #         #unified_graph_cpu = Data(
        #         #    edge_index=unified_graph.edge_index.detach().to(unified_graph.device, non_blocking=True),
        #         #    node_level=unified_graph.node_level.detach().to(unified_graph.device, non_blocking=True)
        #         #        if hasattr(unified_graph, "node_level") and unified_graph.node_level is not None else None,
        #         #    edge_type=unified_graph.edge_type.detach().to(unified_graph.device, non_blocking=True)
        #         #        if hasattr(unified_graph, "edge_type") and unified_graph.edge_type is not None else None,
        #         #    num_nodes=int(unified_graph.num_nodes) if hasattr(unified_graph, "num_nodes")
        #         #        else int(unified_graph.x.size(0)),
        #         #    level_offsets=list(getattr(unified_graph, "level_offsets", [])),
        #         #)
                
        #         #print(unified_graph_cpu.lap_pe_raw_cpu)
        #         # N = unified_graph_cpu.num_nodes

        #         # # global positional index 0..N-1
        #         # unified_graph_cpu.pos_global_cpu = torch.arange(N, dtype=torch.long)

        #         # # local per-level positions: 0..(size_of_level-1) for each level, concatenated in the unified layout
        #         # if getattr(unified_graph_cpu, "level_offsets", None):
        #         #     offsets = unified_graph_cpu.level_offsets
        #         #     pos_local = torch.empty(N, dtype=torch.long)
        #         #     for li in range(len(offsets) - 1):
        #         #         s, e = offsets[li], offsets[li + 1]
        #         #         pos_local[s:e] = torch.arange(e - s, dtype=torch.long)
        #         #     unified_graph_cpu.pos_local_cpu = pos_local
        #         # else:
        #         #     # Fallback if offsets missing: treat as single level
        #         #     unified_graph_cpu.pos_local_cpu = torch.arange(N, dtype=torch.long)

        #         # 2) Move node features to CPU (pinned if possible). Keep separate from skeleton.
        #         #x_init = unified_graph.x.to("cpu", non_blocking=True).detach()
        #         #x_init = unified_graph.x.detach().to("cpu", non_blocking=True)
        #         # Make torch zeros tensor to start x_init, matching unfied_graph.x
        #         x_init = torch.zeros((unified_graph.num_nodes, self.hidden_dim), dtype=torch.float32, device="cpu")
        #         unified_graph.x = None  # release GPU memory *immediately*
        # # rebatch x_init so we can add token_embeddings_full to it directly
        # # Get batch from token_embeddings_full
        # if token_embeddings_full.ndim == 3:
        #     batch_size, seq_len, _ = token_embeddings_full.shape
        #     # Expand x_init along the batch dimension
        #     x_init = x_init.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        #     #print(f"x_init reshaped to: {x_init.shape}")
        #     # Now add token_embeddings_full to the L0 slice, by taking the token_embeddings_full seq_len and replacing x_init[:,:seq_len,:]
        #     x_init[:, :seq_len, :] += token_embeddings_full.to(x_init.device, non_blocking=True)


        # #x_init = unified_graph.x.detach()
        # #try:
        # #    x_init = x_init.pin_memory()
        # #except RuntimeError:
        # #    pass  # pinning can fail on some systems; it's optional

        # # 3) Rebuild edge_attr on CPU if you use edge features
        # #if self.use_edge_attr and getattr(self, "edge_feature_generator", None) is not None \
        # #and unified_graph_cpu.edge_index.numel() > 0:
        # #    try:
        #         # Edge feature generator only needs x stats if it is x-dependent.
        #         # If independent (type-only), pass a dummy (e.g., zeros) or reuse x_init on CPU.
        #         # Here we assume it uses x; pass x_init on CPU.
        #         # get gpu device for edge feature generation
        #         #edge_device = next(self.edge_feature_generator.parameters()).device
        #         #unified_graph_cpu.edge_attr = self.edge_feature_generator(
        #         #    x_init.to(edge_device), unified_graph_cpu.edge_index.to(edge_device), unified_graph_cpu.edge_type.to(edge_device)
        #         #).to("cpu")
        #         #edge_device = next(self.edge_feature_generator.parameters()).device
        # #        unified_graph_cpu.edge_attr = self.edge_feature_generator(
        # #            x_init, unified_graph_cpu.edge_index, unified_graph_cpu.edge_type
        # #        ).to("cpu")
        # #    except Exception as e:
        # #        logger.warning(f"[EHFGAT:BATCH] edge_attr regen on CPU failed: {e}", exc_info=True)

        # # 4) Prepare the batched executor (target CUDA device inferred from model weights)
        # target_device = next(self.parameters()).device
        # # executor = getattr(self, "_batched_executor", None)
        # # if executor is None:
        # #     from .batched_layer_executor import BatchedLayerExecutor
        # #     # Choose your batch size; 8192/16384 are good starting points
        # #     self._batched_executor = BatchedLayerExecutor(
        # #         device=target_device, #pin_memory=False
        # #     )
        # #     executor = self._batched_executor
        # # else:
        # #     # Make sure the executor is aligned with current device
        # #     executor.device = target_device
        # # #print(f"X_init shape: {x_init.shape}, device: {x_init.device}, target_device: {target_device}")
        # # # 5) Run the batched, layer-wise refinement on CPU skeleton
        # # x_final = self._apply_unified_refinement_batched(
        # #     unified_graph_cpu=unified_graph_cpu,
        # #     x_init=x_init,                               # CPU, (pinned if possible)
        # #     cycles=self.unified_refinement_cycles,
        # #     batch_executor=executor,
        # #     sweep_L0=True,
        # #     sweep_L1=True,
        # #     L1_iters=1,
        # #     L2L3_iters=1,
        # # )
        # # clean memory before refinement to avoid GPU OOM
        # del unified_graph, current_processed_graph, level_graphs, level_mapping, subgraph, next_level_graph_init, next_level_graph_refined, \
        #     l0_graph, token_embeddings, token_embeddings_full
        # torch.cuda.empty_cache()
        # unified_graph_cpu = unified_graph_cpu.to(target_device)
        # unified_graph_cpu.x = x_init.to(target_device)  # CPU [N_total, H]
        # #print(target_device, " target device for unified refinement")
        # refined_graph = self._apply_unified_refinement(unified_graph_cpu, self.unified_refinement_cycles)
        # x_final = refined_graph.x  # CPU [N_total, H]
        # # if self.final_norm.weight.device != x_final.device:
        # #     self.final_norm = self.final_norm.to(x_final.device)
        # # # send to gpu instead
        # # self.final_norm = self.final_norm.to(target_device)
        # # #print(f"x_final shape: {x_final.shape}, device: {x_final.device}, target_device: {target_device}")
        # # x_final = self.final_norm(x_final.to(target_device, non_blocking=True))
        # # x_final is a CPU tensor [N, H] with updated features
        # # Slice L0 and continue:
        # token_features = x_final[:,:hierarchical_features[0].size(0)]
        # hierarchical_features[0] = token_features.to(target_device, non_blocking=True)
        # #else:
        # if self.unified_refinement_cycles == 0:
        #     logger.debug("Skipping Stage 2: Unified graph refinement (0 cycles).")
        # # === END STAGE 2 ===


        # # --- Step 5: Feature Extraction & Final Projection ---
        # if not hierarchical_features or hierarchical_features[0] is None or hierarchical_features[0].numel() == 0:
        #     logger.error("Invalid or empty L0 features detected after refinement!")
        #     return None # Signal error

        # token_features = hierarchical_features[0] # Get final L0 features

        # final_token_features = token_features
        # if use_level_prediction and len(hierarchical_features) > 1:
        #      highest_level_features = hierarchical_features[-1]
        #      if hasattr(self, 'highest_to_token_projection') and highest_level_features is not None and highest_level_features.numel() > 0 and \
        #         not (torch.isnan(highest_level_features).any() or torch.isinf(highest_level_features).any()):
        #           global_context = self.highest_to_token_projection(highest_level_features.mean(dim=0))
        #           num_tokens_final = final_token_features.size(0)
        #           global_context = global_context.unsqueeze(0).expand(num_tokens_final, -1)
        #           final_token_features = final_token_features + global_context
        #      # else: warnings handled?
        
        # # Return final token features (for forward() method)
        # return final_token_features
    
    # OLD
    # def _process_hierarchical_graph(self, token_embeddings, cycles, use_level_prediction):
    #     """
    #     Process the token embeddings through the entire hierarchical graph.
    #     This function encapsulates the core graph processing logic shared between
    #     normal forward and imputation modes.
        
    #     Args:
    #         token_embeddings: Embedded tokens [num_tokens, hidden_dim]
    #         cycles: Number of refinement cycles
    #         use_level_prediction: Whether to use level projection
            
    #     Returns:
    #         results: Dictionary with processed graph, token features, and hierarchical features
    #     """

    #     self.level_mappings = [] 
        
    #     # Step 1: Build L0 (token level) graph
    #     l0_graph = self._build_level_graph(token_embeddings, level_idx=0)
        
    #     # Enhanced contextualization for L0
    #     for cycle in range(self.l0_cycles):
    #         l0_graph = self._process_level(l0_graph, level_idx=0)
        
    #     # Store all processed level graphs
    #     level_graphs = [l0_graph]
    #     current_graph = l0_graph
        
    #     # Progressively build and refine higher levels
    #     for level_idx in range(1, len(self.compression_ratios) + 1):
    #         if level_idx >= len(self.level_transformers):
    #             break
                
    #         # Use existing hierarchy builder to create next level
    #         next_level_graph_init, level_mapping = self._create_next_level(
    #             current_graph,
    #             level_idx=level_idx,
    #             compression_ratio=self.compression_ratios[level_idx-1],
    #             overlap_ratio=self.overlap_ratios[level_idx-1]
    #         )
            
    #         # Internal contextualization cycles for this level
    #         #internal_cycles = self.internal_cycles_per_level[level_idx-1] if level_idx-1 < len(self.internal_cycles_per_level) else 2
            
    #         # Create a unified level-specific subgraph with connections between current and next level
    #         subgraph = self._build_level_subgraph(current_graph, next_level_graph_init, level_mapping)
            
    #         # Run internal refinement cycles on this subgraph
    #         #for cycle in range(internal_cycles):
    #         #    subgraph = self._process_subgraph(subgraph, level_idx)

    #         # 3. Run IMPUTATION cycles on the subgraph
    #         #    This refines the masked nodes based on lower-level context
    #         internal_cycles = self.internal_cycles_per_level[level_idx-1] if level_idx-1 < len(self.internal_cycles_per_level) else 1 # Default to 1 cycle if not specified
    #         logger.debug(f"  Running {internal_cycles} imputation cycles...")
    #         for _ in range(internal_cycles):
    #             # _process_subgraph applies transformer layers to update subgraph.x
    #             # It should ideally use level_idx related transformers
    #             subgraph = self._process_subgraph(subgraph, level_idx)

    #         # Extract the REFINED/IMPUTED summary features from the subgraph
    #         next_level_offset = current_graph.x.size(0)
    #         # Check if subgraph processing yielded expected feature tensor size
    #         if subgraph.x.size(0) < next_level_offset + next_level_graph_init.x.size(0):
    #              print(f"Error: Subgraph feature tensor size ({subgraph.x.size(0)}) is smaller than expected "
    #                    f"({next_level_offset + next_level_graph_init.x.size(0)}) after imputation cycles for L{level_idx}. Stopping hierarchy build.")
    #              # Maybe return partial results or raise an error
    #              break # Exit loop
    #         # OLD
    #         #next_level_graph.x = subgraph.x[next_level_offset:]
            
    #         # Create the final graph object for this level using the imputed features
    #         # and the original structure (edges, level) from the initialized graph
    #         next_level_graph_refined = Data(
    #              x=subgraph.x[next_level_offset:], # Take the imputed features
    #              edge_index=next_level_graph_init.edge_index, # Keep original internal edges
    #              node_level=next_level_graph_init.node_level, # Keep original node levels
    #              # Copy other necessary attributes from next_level_graph_init if they exist
    #              **( {attr: getattr(next_level_graph_init, attr) for attr in next_level_graph_init.keys() if attr not in ['x', 'edge_index', 'node_level']} )
    #         )
    #         logger.debug(f"  L{level_idx} imputed features extracted, shape: {next_level_graph_refined.x.shape}")

    #         # Process the next level with its own transformers
    #         #next_level_graph = self._process_level(next_level_graph, level_idx=level_idx)
            
    #         # Store mapping and the *refined* graph
    #         self.level_mappings.append(level_mapping)
    #         level_graphs.append(next_level_graph_refined) # Add the graph with imputed features

    #         # Update current graph for the next iteration
    #         current_graph = next_level_graph_refined

    #     num_refinement_cycles = self.refinement_cycles # Use cycles from init
    #     hierarchical_features = [] # To store final features

    #     if self.refinement_style == "iterative_level":
    #          logger.debug(f"Using iterative level refinement ({num_refinement_cycles} cycles)...")
    #          if not hasattr(self, '_apply_iterative_level_refinement'):
    #               raise NotImplementedError("_apply_iterative_level_refinement not found")
    #          # Pass level_graphs (containing structure and initial refined features)
    #          final_features_list = self._apply_iterative_level_refinement(
    #              level_graphs, self.level_mappings, num_refinement_cycles
    #          )
    #          if not final_features_list: return {"token_features": None, "hierarchical_features": []}
    #          token_features = final_features_list[0]
    #          hierarchical_features = final_features_list # Store the list

    #     elif self.refinement_style == "unified":
    #          logger.debug(f"Using unified graph refinement ({num_refinement_cycles} cycles)...")
    #          if not hasattr(self, '_build_unified_graph') or not hasattr(self, '_apply_unified_refinement'):
    #                raise NotImplementedError("_build_unified_graph or _apply_unified_refinement not found")
    #          unified_graph = self._build_unified_graph(level_graphs, self.level_mappings)
    #          refined_graph = self._apply_unified_refinement(unified_graph, num_refinement_cycles)
    #          # Extract L0 features
    #          num_tokens = level_graphs[0].num_nodes
    #          token_features = refined_graph.x[:num_tokens]
    #          # Extract all level features
    #          for i in range(self.num_hier_levels):
    #              start = refined_graph.level_offsets[i]
    #              end = refined_graph.level_offsets[i+1]
    #              hierarchical_features.append(refined_graph.x[start:end])

    #     else:
    #         raise ValueError(f"Unknown refinement_style: {self.refinement_style}")
        
    #     # Step 5: Build unified graph connecting all processed levels
    #     #unified_graph = self._build_unified_graph(level_graphs, self.level_mappings)
        
    #     # Step 6: Apply bidirectional refinement cycles
    #     #refined_graph = self._apply_refinement_cycles(unified_graph, cycles)
        
    #     # --- Step 4: Feature Extraction & Final Projection ---
    #     if token_features is None or torch.isnan(token_features).any() or torch.isinf(token_features).any():
    #         logger.error("Invalid token features detected after refinement!")
    #         return {"token_features": None, "hierarchical_features": hierarchical_features}

    #     final_token_features = token_features
    #     if use_level_prediction and len(hierarchical_features) > 1:
    #          highest_level_features = hierarchical_features[-1]
    #          if hasattr(self, 'highest_to_token_projection') and \
    #             not (torch.isnan(highest_level_features).any() or torch.isinf(highest_level_features).any()):
    #               global_context = self.highest_to_token_projection(highest_level_features.mean(dim=0))
    #               num_tokens_final = final_token_features.size(0)
    #               global_context = global_context.unsqueeze(0).expand(num_tokens_final, -1)
    #               final_token_features = final_token_features + global_context
    #          # else: Warnings handled in base class check maybe
    #     return final_token_features
        # return {
        #     "token_features": final_token_features,
        #     "hierarchical_features": hierarchical_features
        # }
    
    # OLD
    # def _build_level_subgraph(self, lower_graph, higher_graph, level_mapping):
    #     """
    #     Build a connected subgraph between two adjacent levels.
        
    #     Args:
    #         lower_graph: Graph of the lower level
    #         higher_graph: Graph of the higher level
    #         level_mapping: Tuple of (lower_to_higher, higher_to_lower) mappings
            
    #     Returns:
    #         subgraph: Connected subgraph with both levels
    #     """
    #     device = lower_graph.x.device
    #     lower_to_higher, higher_to_lower = level_mapping
        
    #     # Combine node features
    #     combined_x = torch.cat([lower_graph.x, higher_graph.x], dim=0)
        
    #     # Combine node levels
    #     combined_node_level = torch.cat([
    #         lower_graph.node_level,
    #         higher_graph.node_level
    #     ], dim=0)
        
    #     # Copy within-level edges
    #     lower_edges = lower_graph.edge_index.clone()
    #     higher_edges = higher_graph.edge_index.clone() + lower_graph.x.size(0)  # Offset
        
    #     # Create cross-level edges
    #     cross_edges = []
        
    #     # Add connections between levels based on the mapping
    #     num_lower_nodes = lower_graph.x.size(0)
        
    #     for lower_idx, higher_indices in lower_to_higher.items():
    #         for higher_idx in higher_indices:
    #             higher_offset = num_lower_nodes + higher_idx
    #             # Lower -> Higher
    #             cross_edges.append([lower_idx, higher_offset])
    #             # Higher -> Lower
    #             cross_edges.append([higher_offset, lower_idx])
        
    #     # Convert to tensor
    #     if cross_edges:
    #         cross_edge_index = torch.tensor(cross_edges, dtype=torch.long, device=device).t()
    #     else:
    #         cross_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
    #     # Combine all edges
    #     combined_edge_index = torch.cat([lower_edges, higher_edges, cross_edge_index], dim=1)
        
    #     # Create combined graph
    #     from torch_geometric.data import Data
    #     subgraph = Data(
    #         x=combined_x,
    #         edge_index=combined_edge_index,
    #         node_level=combined_node_level
    #     )
        
    #     return subgraph
    
    # def _build_level_subgraph_old(self, lower_graph, higher_graph, level_mapping):
        
    #     device = lower_graph.x.device
    #     lower_to_higher, higher_to_lower = level_mapping
    #     combined_x = torch.cat([lower_graph.x, higher_graph.x], dim=0)
    #     # Ensure node_level exists on both graphs
    #     lower_nl = getattr(lower_graph, 'node_level', torch.zeros(lower_graph.num_nodes, dtype=torch.long, device=device))
    #     higher_nl = getattr(higher_graph, 'node_level', torch.ones(higher_graph.num_nodes, dtype=torch.long, device=device)) # Assume level 1 if missing? Needs care
    #     combined_node_level = torch.cat([lower_nl, higher_nl], dim=0)
    #     # Ensure edge_index exists
    #     lower_edges = getattr(lower_graph, 'edge_index', torch.empty((2,0), dtype=torch.long, device=device))
    #     higher_edges = getattr(higher_graph, 'edge_index', torch.empty((2,0), dtype=torch.long, device=device)) + lower_graph.num_nodes

    #     cross_edges = []
    #     num_lower_nodes = lower_graph.num_nodes
    #     num_higher_nodes = higher_graph.num_nodes
    #     for lower_idx, higher_indices in lower_to_higher.items():
    #         for higher_idx in higher_indices:
    #             # Add boundary checks
    #             if lower_idx < num_lower_nodes and higher_idx < num_higher_nodes:
    #                 higher_offset = num_lower_nodes + higher_idx
    #                 cross_edges.append([lower_idx, higher_offset])
    #                 cross_edges.append([higher_offset, lower_idx])

    #     cross_edge_index = torch.tensor(cross_edges, dtype=torch.long, device=device).t() if cross_edges else torch.zeros((2, 0), dtype=torch.long, device=device)
    #     combined_edge_index = torch.cat([lower_edges, higher_edges, cross_edge_index], dim=1)

    #     # TODO: Handle edge_type and edge_attr concatenation if they exist and are used
    #     subgraph = Data(x=combined_x, edge_index=combined_edge_index, node_level=combined_node_level)
    #     return subgraph

    def _build_level_subgraph(self, lower_graph, higher_graph, level_mapping, level_idx=None):
        device = lower_graph.x.device
        lower_to_higher, higher_to_lower = level_mapping

        # Features
        combined_x = torch.cat([lower_graph.x, higher_graph.x], dim=0).contiguous()

        # Node levels (explicit if missing)
        lower_nl  = getattr(lower_graph, 'node_level', None)
        higher_nl = getattr(higher_graph, 'node_level', None)
        if lower_nl is None:
            assert level_idx is not None, "Need level_idx to stamp node_level."
            lower_nl  = torch.full((lower_graph.num_nodes,),  level_idx,     dtype=torch.long, device=device)
        if higher_nl is None:
            assert level_idx is not None, "Need level_idx to stamp node_level."
            higher_nl = torch.full((higher_graph.num_nodes,), level_idx + 1, dtype=torch.long, device=device)
        combined_node_level = torch.cat([lower_nl, higher_nl], dim=0).contiguous()

        # Intra-level edges (shift higher by lower size)
        lower_edges  = getattr(lower_graph,  'edge_index', torch.empty((2,0), dtype=torch.long, device=device))
        higher_edges = getattr(higher_graph, 'edge_index', torch.empty((2,0), dtype=torch.long, device=device))
        if higher_edges.numel() > 0:
            higher_edges = higher_edges + lower_graph.num_nodes

        # Cross edges (bidirectional), vectorized
        pairs = [(l, h) for l, H in lower_to_higher.items() for h in H]
        if pairs:
            lh = torch.tensor(pairs, device=device, dtype=torch.long)               # [K,2]
            up   = torch.stack([lh[:,0], lh[:,1] + lower_graph.num_nodes], dim=0)
            down = torch.stack([lh[:,1] + lower_graph.num_nodes, lh[:,0]], dim=0)
            cross_edge_index = torch.cat([up, down], dim=1)
        else:
            cross_edge_index = torch.empty((2,0), dtype=torch.long, device=device)

        combined_edge_index = torch.cat([lower_edges, higher_edges, cross_edge_index], dim=1)

        # (Optional) undirect + de-dup
        # from torch_geometric.utils import to_undirected
        # combined_edge_index = to_undirected(combined_edge_index, num_nodes=combined_x.size(0))
        combined_edge_index = torch.unique(combined_edge_index.t(), dim=0).t().contiguous()

        # (Optional) self-loops
        # from torch_geometric.utils import add_self_loops
        # combined_edge_index, _ = add_self_loops(combined_edge_index, num_nodes=combined_x.size(0))

        subgraph = Data(
            x=combined_x,
            edge_index=combined_edge_index,
            node_level=combined_node_level,
            num_nodes=combined_x.size(0),
        )
        return subgraph


    def _process_subgraph(self, subgraph, level_idx):
        # This applies transformers to the subgraph.
        # Assumes level_idx indicates which transformers to use (e.g., L_i transformers for L_i<>L_{i+1})
        # It needs access to level_transformers.
        if level_idx < 0 or level_idx >= len(self.level_transformers):
            logger.warning(f"Invalid level_idx {level_idx} provided to _process_subgraph. Skipping processing.")
            return subgraph

        processed_x = subgraph.x
        # Calculate necessary inputs for transformer based on subgraph structure
        # RoPE positions might need careful handling here depending on how subgraph nodes map to original levels/positions
        # For simplicity, maybe disable RoPE inside _process_subgraph for now?
        positions_input = None # Or calculate based on subgraph.node_level?

        for transformer in self.level_transformers[level_idx]:
            # Pass attributes available on the subgraph object
            edge_attr_input = subgraph.edge_attr if hasattr(subgraph, 'edge_attr') and self.use_edge_attr else None
            node_level_input = subgraph.node_level if hasattr(subgraph, 'node_level') else None
            level_offsets_input = None # Offsets generally not applicable to subgraph

            if node_level_input is None:
                 logger.warning("_process_subgraph: node_level missing from subgraph. Cannot proceed.")
                 return subgraph # Return unchanged

            result = transformer(
                 processed_x, subgraph.edge_index, node_level_input,
                 level_offsets=level_offsets_input, positions=positions_input, edge_attr=edge_attr_input
            )
            processed_x = result[0] if isinstance(result, tuple) else result
            # No outer LayerNorm here? Assume it's inside the transformer.
        subgraph.x = processed_x
        return subgraph

    # OLD
    # def _process_subgraph(self, subgraph, level_idx):
    #     """
    #     Process a subgraph with transformers from specified level.
        
    #     Args:
    #         subgraph: PyG Data object with the subgraph
    #         level_idx: Level index for transformer selection
            
    #     Returns:
    #         processed_subgraph: Updated subgraph after processing
    #     """
    #     # Apply transformers from the specified level
    #     if level_idx < len(self.level_transformers):
    #         for transformer in self.level_transformers[level_idx]:
    #             subgraph.x = transformer(
    #                 subgraph.x,
    #                 subgraph.edge_index,
    #                 subgraph.node_level,
    #                 edge_attr=None if not hasattr(subgraph, 'edge_attr') else subgraph.edge_attr
    #             )
    #             subgraph.x = self.layer_norm(subgraph.x)
        
    #     return subgraph

    def _recurrent_default_cycles(self, num_cycles: Optional[int] = None) -> int:
        if num_cycles is not None:
            return max(1, int(num_cycles))
        unified_cycles = int(getattr(self, "unified_refinement_cycles", 0) or 0)
        if unified_cycles > 0:
            return unified_cycles
        return max(1, int(getattr(self, "refinement_cycles", 1) or 1))

    def init_recurrent_state(
        self,
        batch_size: int = 1,
        l0_window: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> RecurrentPinballState:
        """
        Initialize a fixed-topology recurrent graph state.

        The active L0 window is kept in chronological physical order. Once full,
        each consumed token shifts L0 left and writes into the final L0 slot.
        Higher-level nodes persist and are updated only by normal refinement.
        """
        if str(getattr(self, "input_mode", "tokens")).lower() != "tokens":
            raise ValueError("Recurrent Pinball generation currently supports input_mode='tokens' only")
        if str(getattr(self, "graph_geometry_mode", "sequence")).lower() != "sequence":
            raise ValueError("Recurrent Pinball generation currently supports sequence graph geometry only")

        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        batch_size = max(1, int(batch_size))
        l0_window = int(l0_window if l0_window is not None else getattr(self, "max_seq_len", 512))
        l0_window = max(1, l0_window)

        dummy_l0 = torch.zeros((l0_window, self.hidden_dim), device=device, dtype=dtype)
        with torch.no_grad():
            cache_key, skeleton = self._build_and_cache_unified_skeleton_from_embeddings(dummy_l0)
            graph = self._rehydrate_unified(skeleton, None, device=device, cache_key=cache_key)

        level_offsets = getattr(graph, "level_offsets", None)
        if level_offsets is None:
            raise RuntimeError("Recurrent skeleton is missing level_offsets")
        level_offsets = torch.as_tensor(level_offsets, device=device, dtype=torch.long)
        if level_offsets.numel() < 2:
            raise RuntimeError("Recurrent skeleton has invalid level_offsets")

        node_level = getattr(graph, "node_level", None)
        if node_level is None:
            raise RuntimeError("Recurrent skeleton is missing node_level")
        node_level = node_level.to(device=device, dtype=torch.long)
        edge_index = graph.edge_index.to(device=device, dtype=torch.long)
        edge_type = getattr(graph, "edge_type", None)
        if edge_type is not None:
            edge_type = edge_type.to(device=device, dtype=torch.long)

        num_nodes = int(level_offsets[-1].item())
        pad_id = getattr(self, "pad_token_id", None)
        if pad_id is None and hasattr(self, "tokenizer"):
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None and hasattr(self, "tokenizer"):
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = 0 if pad_id is None else int(pad_id)
        token_ids = torch.full((batch_size, l0_window), pad_id, device=device, dtype=torch.long)

        pad_vec = self.token_embedding(torch.tensor([pad_id], device=device)).to(dtype=dtype)
        x = pad_vec.view(1, 1, -1).repeat(batch_size, num_nodes, 1)

        # Match the normal full-context path for initial higher-level slots when configured.
        if self.input_mode == "tokens" and getattr(self, "upper_init", "mask") in ("mask", "pooled") and level_offsets.numel() > 2:
            mask_vec = self.token_embedding(torch.tensor([self.mask_token_id], device=device)).to(dtype=dtype)
            upper_start = int(level_offsets[1].item())
            if upper_start < num_nodes:
                x[:, upper_start:, :] = mask_vec.view(1, 1, -1)

        future_time = max(1, l0_window)
        node_ar_time = torch.full((num_nodes,), future_time, device=device, dtype=torch.long)
        if level_offsets.numel() > 2:
            node_ar_time[int(level_offsets[1].item()):] = 0

        node_pos_local = getattr(graph, "node_pos_local", None)
        if node_pos_local is not None:
            node_pos_local = torch.as_tensor(node_pos_local, device=device, dtype=torch.long)

        return RecurrentPinballState(
            x=x,
            token_ids=token_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            node_level=node_level,
            level_offsets=level_offsets,
            node_ar_time=node_ar_time,
            node_pos_local=node_pos_local,
            level_grid_shapes=list(getattr(graph, "level_grid_shapes", [])) if getattr(graph, "level_grid_shapes", None) is not None else None,
            time=0,
            active_len=0,
            current_l0_idx=0,
        )

    def recurrent_consume(
        self,
        state: RecurrentPinballState,
        token_ids: torch.Tensor,
        num_cycles: Optional[int] = None,
        detach_state: bool = False,
    ) -> Tuple[RecurrentPinballState, torch.Tensor]:
        """Consume token t, refine the carried graph state, and return logits for t+1."""
        if token_ids.dim() == 0:
            token_ids = token_ids.view(1)
        if token_ids.dim() == 2 and token_ids.size(1) == 1:
            token_ids = token_ids[:, 0]
        token_ids = token_ids.to(device=state.x.device, dtype=torch.long).view(-1)
        if int(token_ids.size(0)) != int(state.x.size(0)):
            raise ValueError(f"Recurrent consume batch mismatch: tokens={token_ids.size(0)} state={state.x.size(0)}")

        l0_start = int(state.level_offsets[0].item())
        l0_end = int(state.level_offsets[1].item())
        l0_window = max(1, l0_end - l0_start)
        functional_update = bool(torch.is_grad_enabled() and not detach_state)

        if state.active_len < l0_window:
            slot = int(state.active_len)
            state.active_len += 1
            x_next = state.x.clone() if functional_update else state.x
            token_ids_next = state.token_ids.clone() if functional_update else state.token_ids
            ar_time_next = state.node_ar_time.clone() if functional_update else state.node_ar_time
        else:
            # Keep physical L0 order chronological for current local-attention kernels.
            if functional_update:
                x_next = state.x.clone()
                token_ids_next = state.token_ids.clone()
                ar_time_next = state.node_ar_time.clone()
                x_next[:, l0_start:l0_end - 1, :] = state.x[:, l0_start + 1:l0_end, :]
                token_ids_next[:, :-1] = state.token_ids[:, 1:]
                ar_time_next[l0_start:l0_end - 1] = state.node_ar_time[l0_start + 1:l0_end]
            else:
                state.x[:, l0_start:l0_end - 1, :] = state.x[:, l0_start + 1:l0_end, :].clone()
                state.token_ids[:, :-1] = state.token_ids[:, 1:].clone()
                state.node_ar_time[l0_start:l0_end - 1] = state.node_ar_time[l0_start + 1:l0_end].clone()
                x_next = state.x
                token_ids_next = state.token_ids
                ar_time_next = state.node_ar_time
            slot = l0_window - 1

        abs_slot = l0_start + slot
        token_emb = self.token_embedding(token_ids).to(device=state.x.device, dtype=state.x.dtype)
        x_next[:, abs_slot, :] = token_emb
        token_ids_next[:, slot] = token_ids
        ar_time_next[abs_slot] = int(state.time)
        state.x = x_next
        state.token_ids = token_ids_next
        state.node_ar_time = ar_time_next
        state.current_l0_idx = int(abs_slot)
        if state.level_offsets.numel() > 2:
            upper_start = int(state.level_offsets[1].item())
            if functional_update:
                state.node_ar_time = state.node_ar_time.clone()
            state.node_ar_time[upper_start:] = int(state.time)

        graph = Data(
            x=state.x,
            edge_index=state.edge_index,
            edge_type=state.edge_type,
            node_level=state.node_level,
            level_offsets=state.level_offsets,
            node_ar_time=state.node_ar_time,
        )
        if state.node_pos_local is not None:
            graph.node_pos_local = state.node_pos_local
        if state.level_grid_shapes is not None:
            graph.level_grid_shapes = list(state.level_grid_shapes)
        # Do not pass cached additive LapPE here: recurrent state.x already carries
        # refined features forward, and re-adding static LapPE each token would drift.

        refined = self._apply_unified_refinement(
            graph,
            self._recurrent_default_cycles(num_cycles),
        )
        refined_x = refined.x if refined.x.dim() == 3 else refined.x.unsqueeze(0)
        state.x = refined_x.detach() if detach_state else refined_x

        state.time += 1

        logits = self.output_projection(state.x[:, abs_slot, :])
        return state, logits

    def recurrent_chunk_forward(
        self,
        state: RecurrentPinballState,
        input_ids_chunk: torch.Tensor,
        num_cycles: Optional[int] = None,
        detach_state: bool = False,
    ) -> Tuple[RecurrentPinballState, torch.Tensor]:
        """Consume a chunk of tokens in parallel and return logits for that chunk."""
        if input_ids_chunk.dim() == 1:
            input_ids_chunk = input_ids_chunk.unsqueeze(0)
        if input_ids_chunk.dim() != 2:
            raise ValueError("recurrent_chunk_forward expects input_ids_chunk shaped [batch, chunk_len]")
        token_ids = input_ids_chunk.to(device=state.x.device, dtype=torch.long)
        if int(token_ids.size(0)) != int(state.x.size(0)):
            raise ValueError(f"Recurrent chunk batch mismatch: tokens={token_ids.size(0)} state={state.x.size(0)}")

        chunk_len = int(token_ids.size(1))
        if chunk_len <= 0:
            raise ValueError("recurrent_chunk_forward received an empty chunk")

        l0_start = int(state.level_offsets[0].item())
        l0_end = int(state.level_offsets[1].item())
        l0_window = max(1, l0_end - l0_start)
        if chunk_len > l0_window:
            raise ValueError(
                f"recurrent_chunk_forward chunk_len={chunk_len} exceeds recurrent_l0_window={l0_window}"
            )

        functional_update = bool(torch.is_grad_enabled() and not detach_state)
        x_next = state.x.clone() if functional_update else state.x
        token_ids_next = state.token_ids.clone() if functional_update else state.token_ids
        ar_time_next = state.node_ar_time.clone() if functional_update else state.node_ar_time

        token_emb = self.token_embedding(token_ids).to(device=state.x.device, dtype=state.x.dtype)
        cur_active = int(state.active_len)
        if cur_active + chunk_len <= l0_window:
            insert_start = cur_active
            insert_end = cur_active + chunk_len
            abs_start = l0_start + insert_start
            abs_end = abs_start + chunk_len
            x_next[:, abs_start:abs_end, :] = token_emb
            token_ids_next[:, insert_start:insert_end] = token_ids
            ar_time_next[abs_start:abs_end] = torch.arange(
                int(state.time),
                int(state.time) + chunk_len,
                device=state.x.device,
                dtype=torch.long,
            )
            state.active_len = insert_end
        else:
            keep_old = l0_window - chunk_len
            if keep_old > 0:
                old_start = cur_active - keep_old
                x_next[:, l0_start:l0_start + keep_old, :] = state.x[:, l0_start + old_start:l0_start + cur_active, :]
                token_ids_next[:, :keep_old] = state.token_ids[:, old_start:cur_active]
                ar_time_next[l0_start:l0_start + keep_old] = state.node_ar_time[
                    l0_start + old_start:l0_start + cur_active
                ]
            abs_start = l0_start + keep_old
            abs_end = abs_start + chunk_len
            x_next[:, abs_start:abs_end, :] = token_emb
            token_ids_next[:, keep_old:keep_old + chunk_len] = token_ids
            ar_time_next[abs_start:abs_end] = torch.arange(
                int(state.time),
                int(state.time) + chunk_len,
                device=state.x.device,
                dtype=torch.long,
            )
            state.active_len = l0_window

        if state.level_offsets.numel() > 2:
            upper_start = int(state.level_offsets[1].item())
            ar_time_next[upper_start:] = int(state.time) + chunk_len - 1

        state.time += chunk_len
        state.x = x_next
        state.token_ids = token_ids_next
        state.node_ar_time = ar_time_next
        state.current_l0_idx = int(abs_end - 1)

        graph = Data(
            x=state.x,
            edge_index=state.edge_index,
            edge_type=state.edge_type,
            node_level=state.node_level,
            level_offsets=state.level_offsets,
            node_ar_time=state.node_ar_time,
        )
        if state.node_pos_local is not None:
            graph.node_pos_local = state.node_pos_local
        if state.level_grid_shapes is not None:
            graph.level_grid_shapes = list(state.level_grid_shapes)

        refined = self._apply_unified_refinement(
            graph,
            self._recurrent_default_cycles(num_cycles),
        )
        refined_x = refined.x if refined.x.dim() == 3 else refined.x.unsqueeze(0)
        state.x = refined_x.detach() if detach_state else refined_x

        logits = self.output_projection(state.x[:, abs_start:abs_end, :])
        return state, logits

    def detach_recurrent_state(self, state: RecurrentPinballState) -> RecurrentPinballState:
        """Detach differentiable recurrent tensors at a truncated-BPTT boundary."""
        state.x = state.x.detach()
        return state

    def recurrent_teacher_forced_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_l0_window: Optional[int] = None,
        unroll_len: int = 64,
        detach_every: int = 64,
        loss_stride: int = 1,
        warmup_tokens: int = 0,
        num_cycles: Optional[int] = None,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, Union[int, float, torch.Tensor]]]:
        """
        Teacher-forced recurrent next-token loss.

        Consumes x_t through the recurrent graph cell and supervises the returned
        logits against x_{t+1}. The state is detached at chunk boundaries for
        truncated BPTT when detach_every > 0.
        """
        if input_ids.dim() != 2:
            raise ValueError("recurrent_teacher_forced_loss expects input_ids shaped [B, T]")
        B, T = input_ids.shape
        device = input_ids.device
        if T < 2:
            zero = torch.zeros((), device=device, dtype=next(self.parameters()).dtype)
            return zero, {"tokens": 0, "correct": 0, "loss": zero}

        unroll_len = max(1, int(unroll_len))
        detach_every = max(0, int(detach_every))
        loss_stride = max(1, int(loss_stride))
        warmup_tokens = max(0, int(warmup_tokens))
        state = self.init_recurrent_state(
            batch_size=int(B),
            l0_window=recurrent_l0_window,
            device=device,
        )

        total_numer = torch.zeros((), device=device, dtype=torch.float32)
        total_denom = 0
        total_correct = 0
        steps_since_detach = 0

        for t in range(T - 1):
            state, logits = self.recurrent_consume(
                state,
                input_ids[:, t],
                num_cycles=num_cycles,
                detach_state=False,
            )
            steps_since_detach += 1

            supervise = bool(t >= warmup_tokens and ((t - warmup_tokens) % loss_stride == 0))
            if supervise:
                target = input_ids[:, t + 1]
                if attention_mask is not None:
                    valid = attention_mask[:, t + 1].to(device=device, dtype=torch.bool)
                else:
                    valid = torch.ones((B,), device=device, dtype=torch.bool)
                if bool(valid.any()):
                    ce = F.cross_entropy(
                        logits.float(),
                        target,
                        reduction="none",
                        label_smoothing=max(0.0, float(label_smoothing)),
                    )
                    total_numer = total_numer + (ce * valid.to(dtype=ce.dtype)).sum()
                    count = int(valid.sum().item())
                    total_denom += count
                    pred = logits.argmax(dim=-1)
                    total_correct += int(((pred == target) & valid).sum().item())

            detach_boundary = bool(detach_every > 0 and steps_since_detach >= detach_every)
            unroll_boundary = bool((t + 1) % unroll_len == 0)
            if detach_boundary or unroll_boundary:
                state = self.detach_recurrent_state(state)
                steps_since_detach = 0

        if total_denom <= 0:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return zero, {"tokens": 0, "correct": 0, "loss": zero}
        loss = total_numer / float(total_denom)
        return loss, {
            "tokens": int(total_denom),
            "correct": int(total_correct),
            "loss": loss,
        }

    @torch.no_grad()
    def generate_recurrent(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        recurrent_l0_window: Optional[int] = None,
        num_cycles: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation using Pinball as a recurrent graph cell."""
        if input_ids.dim() != 2:
            raise ValueError("generate_recurrent expects input_ids shaped [batch, seq]")
        if input_ids.size(0) != 1:
            raise ValueError("generate_recurrent currently supports batch_size=1")

        state = self.init_recurrent_state(
            batch_size=int(input_ids.size(0)),
            l0_window=recurrent_l0_window,
            device=input_ids.device,
        )
        current_ids = input_ids.clone()
        logits = None
        for pos in range(int(input_ids.size(1))):
            state, logits = self.recurrent_consume(
                state,
                input_ids[:, pos],
                num_cycles=num_cycles,
                detach_state=True,
            )

        target_total = max(int(max_length), int(current_ids.size(1)))
        eos_token_id = getattr(self, "eos_token_id", None)
        if eos_token_id is None and hasattr(self, "tokenizer"):
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        while int(current_ids.size(1)) < target_total:
            if logits is None:
                logits = self.output_projection(state.x[:, state.current_l0_idx, :])
            next_token = self._safe_sampling(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                current_ids=current_ids,
                do_sample=do_sample,
            )
            current_ids = torch.cat([current_ids, next_token.to(device=current_ids.device, dtype=torch.long)], dim=1)
            if eos_token_id is not None and bool((next_token.view(-1) == int(eos_token_id)).all()):
                break
            state, logits = self.recurrent_consume(
                state,
                next_token,
                num_cycles=num_cycles,
                detach_state=True,
            )

        return current_ids

    @torch.no_grad()
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
        rebuild_graph: bool = True,  # Default to rebuild approach for better coherence
        num_cycles: int = None,      # Allow overriding cycles
        use_imputation: bool = False, # New parameter to use token imputation
        use_autoenc_query: bool = False,
        use_autoenc_ar: bool = False,
        force_autoregressive: bool = False,
        use_recurrent: bool = False,
        recurrent_l0_window: Optional[int] = None,
        use_kv_cache: bool = True,   # cached-tail decode (see base class); safe auto-fallback
        kv_frontier_width: Optional[int] = None,
        verify_incremental: bool = False,
    ) -> torch.Tensor:
        """
        Generate text with support for different strategies including token imputation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample from the distribution
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            repetition_penalty: Penalty for repeating tokens
            use_level_prediction: Whether to use level projection for prediction
            use_direct_prediction: Whether to use direct next token prediction
            rebuild_graph: Whether to rebuild the graph for each token
            num_cycles: Optional override for refinement cycles
            use_imputation: Whether to use token imputation for generation
            use_autoenc_query: Whether to generate via decoder L0' query logits
            use_autoenc_ar: Whether to generate via decoder L0' AR last-position logits
            force_autoregressive: If true, bypass imputation routing and use AR generation
            
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len + new_tokens]
        """
        autoenc_mode = str(getattr(self, "autoenc_graph_mode", "off")).lower()

        if use_recurrent:
            return self.generate_recurrent(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recurrent_l0_window=recurrent_l0_window,
                num_cycles=num_cycles,
            )

        if (not force_autoregressive) and use_autoenc_ar:
            if autoenc_mode != "twin_shared_l3":
                logger.warning(
                    "use_autoenc_ar requested but autoenc_graph_mode=%s; falling back to default generation.",
                    autoenc_mode,
                )
            else:
                prev_decode_head = getattr(self, "_force_decode_head", None)
                self._force_decode_head = "ae"
                try:
                    return super().generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_level_prediction=use_level_prediction,
                        use_direct_prediction=use_direct_prediction,
                        rebuild_graph=rebuild_graph,
                        num_cycles=num_cycles,
                    )
                finally:
                    self._force_decode_head = prev_decode_head

        if (not force_autoregressive) and use_autoenc_query:
            if autoenc_mode != "twin_shared_l3":
                logger.warning(
                    "use_autoenc_query requested but autoenc_graph_mode=%s; falling back to default generation.",
                    autoenc_mode,
                )
            else:
                return self._generate_with_autoenc_query(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_level_prediction=use_level_prediction,
                    num_cycles=num_cycles,
                )

        # If imputation is requested or the model was trained with imputation,
        # use the imputation-based generation by default
        if (not force_autoregressive) and (use_imputation or self.train_with_imputation):
            return self._generate_with_imputation(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_level_prediction=use_level_prediction,
                num_cycles=num_cycles
            )
        
        # Otherwise use the original generation method
        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_level_prediction=use_level_prediction,
            use_direct_prediction=use_direct_prediction,
            rebuild_graph=rebuild_graph,
            num_cycles=num_cycles,
            use_kv_cache=use_kv_cache,
            kv_frontier_width=kv_frontier_width,
            verify_incremental=verify_incremental,
        )

    def _generate_with_autoenc_ar(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        use_level_prediction: bool = False,
        num_cycles: int = None,
    ) -> torch.Tensor:
        """
        Generate autoregressively from decoder L0' logits at the current last position.
        """
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        cycles = num_cycles if num_cycles is not None else self.refinement_cycles
        current_ids = input_ids.clone()

        eos_token_id = getattr(self, "eos_token_id", None)
        if eos_token_id is None and hasattr(self, "tokenizer"):
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self, "pad_token_id", 0)
        eos_token_id = int(min(self.vocab_size - 1, int(eos_token_id)))

        with torch.no_grad():
            for _ in range(max(0, max_length - prompt_len)):
                if current_ids.size(1) >= self.max_seq_len:
                    break

                forward_out = self.forward(
                    input_ids=current_ids,
                    num_cycles=cycles,
                    use_level_prediction=use_level_prediction,
                    imputation_mode=False,
                    imputation_idx=None,
                )

                ae_logits = getattr(self, "_last_autoenc_logits", None)
                if ae_logits is not None and ae_logits.dim() == 3 and ae_logits.size(1) > 0:
                    next_logits = ae_logits[:, -1, :]
                else:
                    logits = forward_out[0] if isinstance(forward_out, tuple) else forward_out
                    if logits is None or logits.dim() != 3 or logits.size(1) == 0:
                        logger.warning("Autoenc-AR generation received invalid logits; stopping.")
                        break
                    next_logits = logits[:, -1, :]

                if hasattr(self, "_safe_sampling"):
                    next_token = self._safe_sampling(
                        next_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        current_ids=current_ids,
                        do_sample=do_sample,
                    )
                else:
                    sampled_tokens = []
                    for b in range(next_logits.size(0)):
                        sampled = safe_sample_from_logits(
                            logits=next_logits[b],
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=do_sample,
                            repetition_penalty=repetition_penalty,
                            previous_ids=current_ids[b],
                            fallback_token_id=eos_token_id,
                            debug_id=f"ae_ar_b{b}_t{current_ids.size(1)}",
                        )
                        sampled_tokens.append(sampled.to(device=device, dtype=torch.long).view(1))
                    next_token = torch.stack(sampled_tokens, dim=0)
                current_ids = torch.cat([current_ids, next_token], dim=1)

                if next_token.numel() == 1:
                    if int(next_token.item()) == eos_token_id:
                        break
                elif bool((next_token.view(-1) == eos_token_id).all()):
                    break

        return current_ids

    def _generate_with_autoenc_query(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        use_level_prediction: bool = False,
        num_cycles: int = None,
    ) -> torch.Tensor:
        """
        Generate by appending one empty query position and sampling from decoder L0' logits.
        """
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        cycles = num_cycles if num_cycles is not None else self.refinement_cycles

        mask_token_id = getattr(self, "mask_token_id", None)
        if mask_token_id is None and hasattr(self, "tokenizer"):
            mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self, "pad_token_id", 0)

        current_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max(0, max_length - prompt_len)):
                if current_ids.size(1) >= self.max_seq_len:
                    break

                query_col = torch.full(
                    (batch_size, 1),
                    int(mask_token_id),
                    dtype=torch.long,
                    device=device,
                )
                query_ids = torch.cat([current_ids, query_col], dim=1)

                forward_out = self.forward(
                    input_ids=query_ids,
                    num_cycles=cycles,
                    use_level_prediction=use_level_prediction,
                    imputation_mode=False,
                    imputation_idx=None,
                )

                query_logits = getattr(self, "_last_autoenc_query_logits", None)
                if query_logits is None:
                    ae_logits = getattr(self, "_last_autoenc_logits", None)
                    if ae_logits is not None and ae_logits.dim() == 3 and ae_logits.size(1) > 0:
                        query_logits = ae_logits[:, -1, :]
                if query_logits is None:
                    logits = forward_out[0] if isinstance(forward_out, tuple) else forward_out
                    if logits is None or logits.dim() != 3 or logits.size(1) == 0:
                        logger.warning("Autoenc-query generation received invalid logits; stopping.")
                        break
                    query_logits = logits[:, -1, :]

                if query_logits.dim() == 1:
                    query_logits = query_logits.unsqueeze(0)

                sampled_tokens = []
                for b in range(query_logits.size(0)):
                    sampled = safe_sample_from_logits(
                        logits=query_logits[b],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                        previous_ids=current_ids[b],
                        fallback_token_id=int(mask_token_id),
                        debug_id=f"ae_query_b{b}_t{current_ids.size(1)}",
                    )
                    sampled_tokens.append(sampled.to(device=device, dtype=torch.long).view(1))

                next_token = torch.stack(sampled_tokens, dim=0)
                current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids
    
    def _generate_with_imputation(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        use_level_prediction: bool = False,
        num_cycles: int = None,
    ) -> torch.Tensor:
        """
        Generate text using token imputation through the enhanced forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample from the distribution
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            repetition_penalty: Penalty for repeating tokens
            use_level_prediction: Whether to use level projection for prediction
            num_cycles: Optional override for refinement cycles
            
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len + new_tokens]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        cycles = num_cycles if num_cycles is not None else self.refinement_cycles
        
        # Only support batch size of 1 for now
        if batch_size > 1:
            raise ValueError("Token imputation currently only supports batch size of 1")
        
        mask_token_id = getattr(self, 'mask_token_id', None) # Check if model stores it
        if mask_token_id is None and hasattr(self, 'tokenizer'): # Check tokenizer if attached
            mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)

        if mask_token_id is None:
            # Fallback if mask token is somehow unavailable
            logger.warning("Mask token ID not found for generation, falling back to pad/0.")
            mask_token_id = getattr(self, 'pad_token_id', 0) # Use padding or 0


        # Start with input sequence
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            # Generate tokens one by one
            for _ in range(max_length - seq_len):
                # Check if we've reached maximum sequence length
                if current_ids.size(1) >= self.max_seq_len:
                    break
                
                # Append the actual MASK token ID for imputation
                imputation_ids = torch.cat([
                    current_ids,
                    torch.tensor([[mask_token_id]], dtype=torch.long, device=device) # Use mask_token_id
                ], dim=1)

                # Run forward pass in imputation mode
                # Need to make sure self.forward handles imputation_mode=True correctly
                #token_logits, _, _ = self.forward(
                imputation_result = self.forward(
                    input_ids=imputation_ids, # Pass the sequence with mask at the end
                    num_cycles=cycles,
                    use_level_prediction=use_level_prediction,
                    imputation_mode=True,
                    imputation_idx=-1 # Target the last token (the mask)
                )

                # Check return value before unpacking
                if not isinstance(imputation_result, tuple) or len(imputation_result) < 2:
                     logger.error(f"Unexpected return value from forward in imputation mode: {type(imputation_result)}. Stopping generation.")
                     break
                
                token_logits = imputation_result[0]
                token_features = imputation_result[1]
                # Check if token_logits is valid
                if token_logits is None or token_logits.numel() == 0:
                     logger.error("Received invalid logits during imputation generation. Stopping.")
                     break
                
                # --- Sampling Logic (using _safe_sampling if available) ---
                if hasattr(self, '_safe_sampling'):
                     next_token = self._safe_sampling(
                         token_logits.unsqueeze(0), # Add batch dim for safe_sampling
                         temperature=temperature,
                         top_k=top_k,
                         top_p=top_p,
                         repetition_penalty=repetition_penalty,
                         current_ids=current_ids,
                         do_sample=do_sample
                     )
                else:
                    # Fallback sampling if _safe_sampling isn't available
                    next_token_logits = token_logits / temperature

                    # Apply temperature
                    #next_token_logits = token_logits / temperature
                    
                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        for token_id in set(current_ids[0].tolist()):
                            if next_token_logits[token_id] > 0:
                                next_token_logits[token_id] /= repetition_penalty
                            else:
                                next_token_logits[token_id] *= repetition_penalty
                    
                    # Apply sampling or greedy decoding
                    if do_sample:
                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                            next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                        
                        # Apply top-p filtering
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            # Remove tokens with cumulative probability above threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        # Sample from the filtered distribution
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                    probs = F.softmax(next_token_logits, dim=-1)
                    if do_sample:
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                # Append the selected token to the sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Check for EOS token
                eos_token_id = getattr(self.tokenizer if hasattr(self, 'tokenizer') else self, 'eos_token_id', self.vocab_size -1)
                if next_token.item() == eos_token_id:
                    break

        return current_ids
    
    # def _generate_with_diffusion_from_prompt(
    #     self,
    #     input_ids: torch.Tensor,
    #     generate_length: int = 256,
    #     block_size: int = 64,
    #     overlap: int = 16,
    #     temperature: float = 1.0,
    #     do_sample: bool = True,
    #     top_k: int = 50,
    #     top_p: float = 0.9,
    #     repetition_penalty: float = 1.0,  # Not used in this version, but placeholder
    #     use_level_prediction: bool = False,
    #     num_cycles: int = 10,
    #     remask_threshold: float = 0.8,
    # ) -> torch.Tensor:
    #     """
    #     Diffusion-style multi-block generation with overlap and safe sampling.
    #     """
    #     batch_size, prompt_len = input_ids.shape
    #     device = input_ids.device

    #     mask_token_id = getattr(self, 'mask_token_id', None)
    #     if mask_token_id is None and hasattr(self, 'tokenizer'):
    #         mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)
    #     if mask_token_id is None:
    #         mask_token_id = getattr(self, 'unk_token_id', 0)
    #     if mask_token_id is None:
    #         mask_token_id = getattr(self, 'pad_token_id', 0)

    #     vocab_size = getattr(self, 'vocab_size', None)
    #     if vocab_size is None:
    #         vocab_size = self.get_output_embeddings().weight.shape[0]

    #     current_ids = input_ids.clone()
    #     total_generated = []

    #     # How many blocks to generate
    #     total_blocks = (generate_length + block_size - 1) // block_size

    #     with torch.no_grad():
    #         for block_idx in range(total_blocks):
    #             this_block_size = min(block_size, generate_length - block_idx * block_size)

    #             mask_block = torch.full((batch_size, this_block_size), mask_token_id, dtype=torch.long, device=device)

    #             if block_idx == 0:
    #                 context_part = current_ids  # Full prompt for the first block
    #             else:
    #                 context_part = current_ids[:, -overlap:]  # Just overlap for later blocks

    #             input_with_mask = torch.cat([context_part, mask_block], dim=1)


    #             # Append to current_ids for tracking full length (only updated after acceptance)
    #             #extended_input = torch.cat([current_ids, mask_block], dim=1)

    #             for step in range(num_cycles):
    #                 imputation_result = self.forward(
    #                     input_ids=input_with_mask,
    #                     num_cycles=self.refinement_cycles,
    #                     use_level_prediction=use_level_prediction,
    #                     imputation_mode=True,
    #                     imputation_idx=None
    #                 )

    #                 if not isinstance(imputation_result, tuple) or len(imputation_result) < 2:
    #                     logger.error(f"Unexpected return from forward during block {block_idx}")
    #                     break

    #                 token_logits = imputation_result[0]

    #                 if token_logits is None or token_logits.numel() == 0:
    #                     logger.error("Empty logits received. Aborting generation.")
    #                     break

    #                 sampled_ids = torch.full_like(input_with_mask, mask_token_id)

    #                 for b in range(batch_size):
    #                     for t in range(input_with_mask.size(1)):
    #                         if t < overlap:
    #                             sampled_ids[b, t] = input_with_mask[b, t]  # Keep context
    #                             continue

    #                         #token_logit = token_logits[b, t]
    #                         if token_logits.ndim == 3:
    #                             token_logit = token_logits[b, t]
    #                         elif token_logits.ndim == 2:
    #                             token_logit = token_logits[t]  # assume batch=1
    #                         else:
    #                             token_logit = token_logits  # assume [vocab] already
    #                         sampled_token = safe_sample_from_logits(
    #                             token_logit,
    #                             temperature=temperature,
    #                             top_k=top_k,
    #                             top_p=top_p,
    #                             fallback_token_id=mask_token_id,
    #                             debug_id=f"block{block_idx}_cycle{step}_b{b}_t{t}"
    #                         )
    #                         sampled_ids[b, t] = sampled_token

    #                 # Confidence
    #                 logits_softmax = F.softmax(token_logits, dim=-1)
    #                 max_probs, _ = logits_softmax.max(dim=-1)

    #                 # Accept or remask
    #                 is_mask = torch.arange(input_with_mask.size(1), device=device) >= overlap
    #                 accept_mask = (max_probs >= remask_threshold) & is_mask.unsqueeze(0)
    #                 remask_mask = ~accept_mask & is_mask.unsqueeze(0)

    #                 input_with_mask[accept_mask] = sampled_ids[accept_mask]
    #                 input_with_mask[remask_mask] = mask_token_id
    #                 # At the final cycle
    #                 if step == num_cycles - 1:
    #                     accepted = accept_mask.sum().item()
    #                     total_masked = is_mask.sum().item()
    #                     logger.info(f"Final cycle for block {block_idx}: {accepted} out of {total_masked} tokens accepted above threshold, forcing fill for all remaining masked tokens...")
                        

    #                     for b in range(batch_size):
    #                         for t in range(input_with_mask.size(1)):
    #                             if input_with_mask[b, t] == mask_token_id:
    #                                 if token_logits.ndim == 3:
    #                                     token_logit = token_logits[b, t]
    #                                 elif token_logits.ndim == 2:  # maybe batchless
    #                                     token_logit = token_logits[t]
    #                                 else:  # flat logits, fallback
    #                                     token_logit = token_logits

    #                                 sampled_token = safe_sample_from_logits(
    #                                     token_logit,
    #                                     temperature=temperature,
    #                                     top_k=top_k,
    #                                     top_p=top_p,
    #                                     fallback_token_id=mask_token_id,
    #                                     debug_id=f"final_forcefill_b{b}_t{t}"
    #                                 )

    #                                 input_with_mask[b, t] = sampled_token

    #             final_block = input_with_mask[:, overlap:]
    #             current_ids = torch.cat([current_ids, final_block], dim=1)
    #             total_generated.append(final_block)

    #     return torch.cat([input_ids] + total_generated, dim=1)

    #4o does not work
    # def _generate_with_diffusion_from_prompt(
    #         self,
    #         input_ids: torch.Tensor,
    #         generate_length: int = 256,
    #         block_size: int = 64,
    #         overlap: int = 16,
    #         temperature: float = 1.0,
    #         do_sample: bool = True,
    #         top_k: int = 50,
    #         top_p: float = 0.9,
    #         use_safe_sampling: bool = True,
    #         repetition_penalty: float = 1.0,
    #         use_level_prediction: bool = False,
    #         num_cycles: int = 20,
    #         initial_threshold: float = 0.2,
    #         final_threshold: float = 0.01,
    #     ) -> torch.Tensor:
    #     """
    #     Diffusion-style multi-block generation with overlap and dynamic masking.
    #     """
    #     import logging
    #     logger = logging.getLogger(__name__)

    #     batch_size, prompt_len = input_ids.shape
    #     device = input_ids.device

    #     mask_token_id = getattr(self, 'mask_token_id', None)
    #     if mask_token_id is None and hasattr(self, 'tokenizer'):
    #         mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)
    #     if mask_token_id is None:
    #         mask_token_id = getattr(self, 'pad_token_id', 0)
    #     #Find vocab size
    #     vocab_size = getattr(self, 'vocab_size', None)
    #     if vocab_size is None:
    #         vocab_size = self.get_output_embeddings().weight.shape[0]

    #     current_ids = input_ids.clone()
    #     total_generated = []

    #     total_blocks = (generate_length + block_size - 1) // block_size

    #     with torch.no_grad():
    #         for block_idx in range(total_blocks):
    #             this_block_size = min(block_size, generate_length - block_idx * block_size)

    #             context_overlap = current_ids[:, -overlap:] if overlap > 0 else current_ids
    #             mask_block = torch.full((batch_size, this_block_size), mask_token_id, dtype=torch.long, device=device)
    #             input_with_mask = torch.cat([context_overlap, mask_block], dim=1)

    #             for step in range(num_cycles):
    #                 current_threshold = initial_threshold + (final_threshold - initial_threshold) * (step / (num_cycles - 1))

    #                 imputation_result = self.forward(
    #                     input_ids=input_with_mask,
    #                     num_cycles=self.refinement_cycles,
    #                     use_level_prediction=use_level_prediction,
    #                     imputation_mode=True,
    #                     imputation_idx=None
    #                 )

    #                 if not isinstance(imputation_result, tuple) or len(imputation_result) < 2:
    #                     break

    #                 token_logits = imputation_result[0]  # Unscaled logits

    #                 logits_softmax = F.softmax(token_logits, dim=-1)
    #                 max_probs, _ = logits_softmax.max(dim=-1)

    #                 sampled_ids = torch.full_like(input_with_mask, mask_token_id)

    #                 for b in range(batch_size):
    #                     for t in range(input_with_mask.size(1)):
    #                         if t < overlap:
    #                             sampled_ids[b, t] = input_with_mask[b, t]
    #                             continue

    #                         raw_logits = token_logits[b, t] if token_logits.ndim == 3 else token_logits[t]

    #                         if use_safe_sampling:
    #                             sampled_token = safe_sample_from_logits(
    #                                 raw_logits,
    #                                 temperature=temperature,
    #                                 top_k=top_k,
    #                                 top_p=top_p,
    #                                 fallback_token_id=mask_token_id,
    #                                 debug_id=f"block{block_idx}_cycle{step}_b{b}_t{t}"
    #                             )
    #                         else:
    #                             logits = raw_logits / temperature
    #                             probs = F.softmax(logits, dim=-1)
    #                             sampled_token = torch.multinomial(probs, 1).squeeze(0) if do_sample else torch.argmax(probs)

    #                         sampled_ids[b, t] = sampled_token

    #                 is_mask = torch.arange(input_with_mask.size(1), device=device) >= overlap
    #                 accept_mask = (max_probs >= current_threshold) & is_mask.unsqueeze(0)
    #                 remask_mask = ~accept_mask & is_mask.unsqueeze(0)

    #                 num_accepted = accept_mask.sum().item()
    #                 logger.info(f"[Block {block_idx}, Cycle {step}] Tokens accepted: {num_accepted} / {batch_size * this_block_size}")

    #                 input_with_mask[accept_mask] = sampled_ids[accept_mask]
    #                 input_with_mask[remask_mask] = mask_token_id

    #             final_block = input_with_mask[:, overlap:]
    #             current_ids = torch.cat([current_ids, final_block], dim=1)
    #             total_generated.append(final_block)

    #     return torch.cat([input_ids] + total_generated, dim=1)
    

    
    @torch.no_grad()
    def _generate_with_diffusion_from_prompt(
        self,
        input_ids: torch.Tensor,
        generate_length: int = 512,
        chunk_size: int = 32,
        temperature: float = 1.0,
        sort_conf: bool = True,
        do_sample: bool = True,
        top_k: int = 45,
        top_p: float = 0.95,
        use_safe_sampling: bool = True,
        guidance_scale: float = 0,
        repetition_penalty: float = 1.0,
        use_level_prediction: bool = False,
        num_cycles: int = 16,#8,
        max_total_cycles: int = 32,#32,
        force_last_step: bool = False,
        initial_threshold: float = 0.0,#0.4,#.3,
        final_threshold: float = 0.00,#.1,
        random_remask_prob: float = 0.0,
        random_remask_cutoff: float = 0.0,#.5,
        min_temperature: float = 1,
        use_autoenc_head: bool = False,
    ) -> torch.Tensor:
        """Diffusion‑denoise generator (VRAM‑safe).

        Added **early random remasking**:
        * During the first `random_remask_cutoff` fraction of diffusion cycles
        we randomly re‑mask `random_remask_prob` of still‑masked tokens to
        encourage exploration and break out of high‑confidence loops.
        * Parameters are user‑tunable.
        * Logger now outputs how many tokens were randomly remasked each step.
        """
        #do_sample = False
        forward_cycles = max(1, int(getattr(self, "unified_refinement_cycles", getattr(self, "refinement_cycles", 1))))
        sampling_name = "safe_sample" if use_safe_sampling else "multinomial"
        logger.info(
            f"Diffusion‑gen start | prompt={input_ids.shape[1]} new={generate_length} "
            f"chunk={chunk_size} denoise_steps={num_cycles} forward_cycles={forward_cycles} temp={temperature} "
            f"sample={sampling_name} cfg={guidance_scale} do_sample={do_sample} "
            f"head={'ae' if use_autoenc_head else 'main'}")

        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        ae_head_warned = False
        if generate_length <= 0:
            return input_ids.clone()

        mask_token_id = getattr(self, 'mask_token_id', 0)
        fallback_token_id = getattr(self, 'eos_token_id', None)
        if fallback_token_id is None:
            fallback_token_id = getattr(self, 'pad_token_id', None)
        if fallback_token_id is None or int(fallback_token_id) == int(mask_token_id):
            fallback_token_id = 0
        vocab_size = getattr(self, 'vocab_size', None)
        if vocab_size is None: vocab_size = self.get_output_embeddings().weight.shape[0]
        model_max_len = getattr(self, 'max_seq_len', 8192)

        current_ids = torch.cat([
            input_ids,
            torch.full((batch_size, generate_length), mask_token_id, dtype=torch.long, device=device)
        ], 1)
        target_total_len = prompt_len + generate_length

        start_t = time.perf_counter()
        generated = 0; chunks = 0; fwd_calls = 0

        while generated < generate_length:
            c_start = prompt_len + generated
            c_end   = min(c_start + chunk_size, target_total_len)
            c_len   = c_end - c_start
            if c_len <= 0:
                break
            chunks += 1
            logger.debug(f"Chunk {chunks}: idx [{c_start}:{c_end}] len={c_len}")

            #for step in range(num_cycles):
            step = 0
            while step < max_total_cycles:
                ctx_end   = c_end
                ctx_start = max(0, ctx_end - model_max_len)
                ctx_ids   = current_ids[:, ctx_start:ctx_end]
                seq_ctx   = ctx_ids.size(1)
                c_off     = c_start - ctx_start

                def reshape(raw):
                    nonlocal fwd_calls
                    fwd_calls += 1
                    if raw.dim() == 2:
                        if raw.size(0) == seq_ctx:  # [seq_ctx,V]
                            raw = raw.unsqueeze(0)
                        else:                        # [B,V]
                            raw = raw.unsqueeze(1).expand(-1, seq_ctx, -1)
                    elif raw.dim() == 3 and raw.size(0) != batch_size:
                        raw = raw.permute(1,0,2)
                    return raw  # [B,seq_ctx,V]

                cond_out = self.forward(
                    input_ids=ctx_ids,
                    num_cycles=forward_cycles,
                    use_level_prediction=use_level_prediction,
                    imputation_mode=False,
                    imputation_idx=None,
                )
                cond_raw = cond_out[0] if isinstance(cond_out, tuple) else cond_out
                if use_autoenc_head:
                    ae_raw = getattr(self, "_last_autoenc_logits", None)
                    if (
                        ae_raw is not None
                        and ae_raw.dim() == 3
                        and ae_raw.size(0) == batch_size
                        and ae_raw.size(1) == seq_ctx
                    ):
                        cond_raw = ae_raw.to(device=cond_raw.device, dtype=cond_raw.dtype)
                    elif not ae_head_warned:
                        logger.warning("Diffusion AE-head requested but AE logits unavailable/mismatched; using main logits.")
                        ae_head_warned = True
                cond_logits = reshape(cond_raw)[:, c_off:c_off + c_len, :]

                if guidance_scale > 0:
                    pre_prompt = max(0, prompt_len - ctx_start)
                    null_ids = torch.cat([
                        torch.full((batch_size, pre_prompt), mask_token_id, dtype=torch.long, device=device),
                        ctx_ids[:, pre_prompt:]
                    ], 1)
                    null_out = self.forward(
                        input_ids=null_ids,
                        num_cycles=forward_cycles,
                        use_level_prediction=use_level_prediction,
                        imputation_mode=False,
                        imputation_idx=None,
                    )
                    null_raw = null_out[0] if isinstance(null_out, tuple) else null_out
                    if use_autoenc_head:
                        ae_raw = getattr(self, "_last_autoenc_logits", None)
                        if (
                            ae_raw is not None
                            and ae_raw.dim() == 3
                            and ae_raw.size(0) == batch_size
                            and ae_raw.size(1) == seq_ctx
                        ):
                            null_raw = ae_raw.to(device=null_raw.device, dtype=null_raw.dtype)
                        elif not ae_head_warned:
                            logger.warning("Diffusion AE-head requested but AE logits unavailable/mismatched; using main logits.")
                            ae_head_warned = True
                    null_logits = reshape(null_raw)[:, c_off:c_off + c_len, :]
                    chunk_logits = null_logits + guidance_scale * (cond_logits - null_logits)
                else:
                    chunk_logits = cond_logits

                if 0 <= int(mask_token_id) < int(chunk_logits.size(-1)):
                    chunk_logits = chunk_logits.clone()
                    chunk_logits[..., int(mask_token_id)] = torch.finfo(chunk_logits.dtype).min

                temp_now = temperature if do_sample else 1.0
                prog    = (step +1 ) / max(1, num_cycles - 1)
                temp_now = temperature + (min_temperature - temperature) * prog
                probs = F.softmax(chunk_logits / temp_now, dim=-1)

                if do_sample:
                    if use_safe_sampling:
                            
                        flat_logits = chunk_logits.view(-1, vocab_size)
                        sampled_ids = []
                        # make a *mutable* tensor we can append to
                        history_ids = current_ids[0, max(0, c_start - 128):c_start].clone()
                        #history_ids = current_ids[0, max(0, c_start - 32):c_start+chunk_size].clone()
                        for i in range(flat_logits.size(0)):              # position-by-position
                            new_id = safe_sample_from_logits(
                                flat_logits[i],
                                temperature=temp_now,
                                top_k=top_k,
                                top_p=top_p,
                                do_sample=True,
                                repetition_penalty=repetition_penalty,
                                previous_ids=history_ids,                 # ★ includes everything so far
                                fallback_token_id=int(fallback_token_id),
                            )
                            sampled_ids.append(new_id)
                            history_ids = torch.cat([history_ids, new_id.unsqueeze(0)])   # ★ grow window
                            if history_ids.numel() > 128:                                 # keep window len
                                history_ids = history_ids[-128:]
                        samp_ids = torch.stack(sampled_ids).view(batch_size, c_len)

                        # history_window = current_ids[0, c_start-128: ]
                        # history_ids_for_penalty = history_window.flatten()
                        # flat_logits = chunk_logits.view(-1, vocab_size)
                        # sampled = [safe_sample_from_logits(flat_logits[i], temp_now, top_k, top_p, True,
                        #                                 repetition_penalty, history_ids_for_penalty, mask_token_id)
                        #         for i in range(flat_logits.size(0))]
                        # samp_ids = torch.stack(sampled).view(batch_size, c_len)
                    else:
                        # B, c_len, V = probs.shape
                        # flat_probs = probs.view(-1, V)  # (c_len, V) since B==1
                        # sampled_ids = []
                        # # make a *mutable* tensor we can append to
                        # history_ids = current_ids[0, max(0, c_start - 256):c_start].clone()
                        
                        # for i in range(flat_probs.size(0)):
                        #     row = flat_probs[i].clone()

                        #     if repetition_penalty != 1.0 and history_ids.numel() > 0:
                        #         hist_unique = torch.unique(history_ids)
                        #         row[hist_unique] = row[hist_unique] / repetition_penalty
                        #         # add a local window repetition penalty
                        #         #local_hist = history_ids[-32:]  # last 32 tokens and future tokens
                        #         #local_unique = torch.unique(local_hist)
                        #         #row[local_unique] = row[local_unique] / (repetition_penalty * 2)

                        #     # renorm
                        #     row_sum = row.sum()
                        #     if row_sum <= 0 or torch.isnan(row_sum) or torch.isinf(row_sum):
                        #         # fallback: argmax from *original* (unpenalized) row to stay deterministic
                        #         new_id = torch.argmax(flat_probs[i])
                        #     else:
                        #         row = row / row_sum

                        #         # guard rails
                        #         if not torch.isfinite(row).all() or (row < 0).any():
                        #             new_id = torch.argmax(flat_probs[i])
                        #         else:
                        #             new_id = torch.multinomial(row, num_samples=1).squeeze(0)

                        #     sampled_ids.append(new_id)

                        #     # grow & trim history
                        #     history_ids = torch.cat([history_ids, new_id.unsqueeze(0)])
                        #     if history_ids.numel() > 256:
                        #         history_ids = history_ids[-256:]
                        #samp_ids = torch.stack(sampled_ids).view(batch_size, c_len)
                        # --- Path B: Use multinomial ---
                        flat_logits = probs.view(-1, vocab_size)
                        #flat_logits = probs.reshape(-1, vocab_size); flat_logits = flat_logits / flat_logits.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                        # Build small history window (match safe path window length if desired)
                        history_ids = current_ids[0, max(0, c_start - 128):c_start+chunk_size]

                        if repetition_penalty != 1.0 and history_ids.numel() > 0:
                            # unique tokens in history
                            hist_unique = torch.unique(history_ids)
                            # scale down their probs
                            flat_logits = flat_logits.clone()  # avoid in-place on shared tensor
                            flat_logits[:, hist_unique] = flat_logits[:, hist_unique] / repetition_penalty
                            # re-normalize rows
                            flat_logits = flat_logits / flat_logits.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                        # else: fall through, use original probs
                        sampled_ids = []
                        try:
                            if not torch.isfinite(flat_logits).all(): raise ValueError("Non-finite probs");
                            if (flat_logits < 0).any(): raise ValueError("Negative probs");
                            if not (flat_logits.sum(dim=-1) > 1e-7).all(): raise ValueError("Zero-sum probs row");
                            sampled_ids_flat_chunk = torch.multinomial(flat_logits, num_samples=1)
                        except ValueError as e:
                            logger.error(f"Multinomial sampling error: {e}. Using argmax instead.")
                            sampled_ids_flat_chunk = torch.argmax(chunk_logits, dim=-1, keepdim=True)
                        
                        samp_ids = sampled_ids_flat_chunk.view(batch_size, c_len)
                else:
                    samp_ids = torch.argmax(chunk_logits, dim=-1)

                samp_p  = torch.gather(probs, -1, samp_ids.unsqueeze(-1)).squeeze(-1)
                conf_th = initial_threshold + (final_threshold - initial_threshold) * prog
                #if step >= num_cycles * 0.5 or step < 2:
                #    conf_th = 00
                low_conf = samp_p < conf_th

                bud = int(c_len * (1.0 - prog)) 
                budget_mask = torch.zeros_like(low_conf)
                if bud > 0:
                    _, order = torch.sort(samp_p, dim=-1, descending=not sort_conf)
                    budget_mask.scatter_(1, order[:, :bud], True)
                remask = low_conf | budget_mask

                # ----- RANDOM REMASK (early cycles) -----------------------
                if step < int(num_cycles * random_remask_cutoff):
                    random_mask = (torch.rand_like(remask, dtype=torch.float32) < random_remask_prob)
                    added = (random_mask & ~remask).sum().item()
                    remask = remask | random_mask
                    logger.debug(
                        f"  step {step:02d}/{num_cycles-1} | random_remask added {added} tokens")

                current_ids[:, c_start:c_end] = torch.where(remask,
                                                            torch.tensor(mask_token_id, device=device),
                                                            samp_ids)
                logger.debug(
                    f"  step {step:02d}/{num_cycles-1} | accept={(~remask).sum().item()} "
                    f"remask={remask.sum().item()} conf_thr={conf_th:.3f}"
                    f" temperature={temp_now:.2f} ")

                if not (current_ids[:, c_start:c_end] == mask_token_id).any():
                    break

                any_masked = (current_ids[:, c_start:c_end] == mask_token_id).any()
                if any_masked and step >= num_cycles - 1:
                    logger.debug(
                        f"  step {step:02d}/{num_cycles} | still masked, continuing...")
                    num_cycles += 1  # still bump num_cycles if needed
                    #if temp_now > 0.6:
                    #    min_temperature -= 0.01
                    if conf_th > 0.0008:
                        final_threshold -= 0.002
                        if final_threshold < 0.0008:
                            final_threshold = 0.00008
                    if force_last_step and remask.sum().item() <= 5:
                        final_threshold = 0
                

                step += 1
                
            else:
                logger.warning(f"Hit maximum allowed steps ({max_total_cycles}), some tokens may remain masked.")
            unresolved = current_ids[:, c_start:c_end] == mask_token_id
            if unresolved.any():
                greedy_ids = torch.argmax(chunk_logits, dim=-1)
                current_ids[:, c_start:c_end].masked_scatter_(unresolved, greedy_ids[unresolved])
                logger.info(
                        f"  step {step:02d}/{num_cycles-1} | greedily filling in unresolved tokens")
            generated += c_len

        elapsed = time.perf_counter() - start_t
        logger.info(f"Finished: new={generated}/{generate_length} chunks={chunks} "
                    f"fwd_calls={fwd_calls} time={elapsed:.2f}s")

        return current_ids[:, :prompt_len + generated]

    




    @torch.no_grad() # Decorator ensures no gradients calculated
    def _generate_with_diffusion_from_prompt_new( # Simplified name back
                self,
                input_ids: torch.Tensor,
                generate_length: int = 128,
                chunk_size: int = 32, # Renamed from block_size
                temperature: float = 1.0,
                sort_conf : bool = True,
                do_sample: bool = True,
                top_k: int = 50,
                top_p: float = 0.9,
                use_safe_sampling: bool = True, # Default to multinomial as requested
                guidance_scale: float = 2, # CFG strength (0 disables)
                repetition_penalty: float = 1.0, # Implemented
                use_level_prediction: bool = False,
                num_cycles: int = 32,
                initial_threshold: float = 0.2, # Not used for acceptance
                final_threshold: float = 0.0,
            ) -> torch.Tensor:
        """
        Diffusion-style generation processing in chunks, conditioning on full history.
        Uses budget annealing based on LLaDA/V1 for acceptance within the current chunk.

        Args:
            input_ids: Tensor [batch_size, prompt_len]
            generate_length: Total number of NEW tokens to generate.
            chunk_size: Size of the segment actively refined in each outer step.
            use_safe_sampling: If True, use safe_sample (with top-k/top-p);
                               If False, use torch.multinomial.
            guidance_scale: CFG strength (1.0 disables).
            num_cycles: Diffusion refinement steps *per chunk*.
            ... (other args) ...
        """
        #do_sample = False
        cfg_enabled = guidance_scale > 0
        sampling_method = "safe_sample (Top-K/P)" if use_safe_sampling else "multinomial"
        logger.info(f"Starting diffusion generation (Chunked/FullHist Cond., CFG {'Enabled' if cfg_enabled else 'Disabled'}, Sampling: {sampling_method}). Prompt: {input_ids.shape[1]}, Target New: {generate_length}, Chunk: {chunk_size}, Guidance: {guidance_scale}, Cycles: {num_cycles}")

        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        if generate_length <= 0: return input_ids.clone()

        mask_token_id = getattr(self, 'mask_token_id', 0)
        fallback_token_id = getattr(self, 'eos_token_id', None)
        if fallback_token_id is None:
            fallback_token_id = getattr(self, 'pad_token_id', None)
        if fallback_token_id is None or int(fallback_token_id) == int(mask_token_id):
            fallback_token_id = 0
        vocab_size = getattr(self, 'vocab_size', None)
        if vocab_size is None: vocab_size = self.get_output_embeddings().weight.shape[0]
        forward_internal_cycles = 1
        model_max_len = getattr(self, 'max_seq_len', 1024)

        # --- Prepare initial full sequence state (Prompt + All Masks) ---
        target_total_length = prompt_len + generate_length
        full_mask_block = torch.full((batch_size, generate_length), mask_token_id, dtype=torch.long, device=device)
        # current_ids holds the evolving state
        current_ids = torch.cat([input_ids, full_mask_block], dim=1)

        start_time = time.perf_counter(); total_forward_passes = 0; blocks_processed = 0; generated_count_in_chunks = 0

        with torch.no_grad():
            while generated_count_in_chunks < generate_length:
                blocks_processed += 1
                # --- Define current chunk indices in the *absolute* sequence ---
                chunk_start_idx = prompt_len + generated_count_in_chunks
                chunk_end_idx = min(chunk_start_idx + chunk_size, target_total_length)
                current_chunk_len = chunk_end_idx - chunk_start_idx

                if current_chunk_len <= 0: logger.warning("Chunk len <= 0. Stopping."); break

                logger.debug(f"Processing Chunk {blocks_processed}: Absolute Indices [{chunk_start_idx}:{chunk_end_idx}] (Length {current_chunk_len})")

                # --- Diffusion Loop for the Current Chunk ---
                for step in range(num_cycles):
                    # --- Prepare FULL input for forward, respecting max_len ---
                    # The context for prediction includes everything up to the end of the *current chunk*
                    context_end_idx = chunk_end_idx
                    context_start_idx = max(0, context_end_idx - model_max_len)
                    forward_input_ids = current_ids[:, context_start_idx:context_end_idx].clone() # Use current state
                    current_input_len = forward_input_ids.size(1)
                    
                    # --- CFG Forward Passes (Conditional and Unconditional) ---
                    try:
                        # logger.debug(f"[Chunk{blocks_processed} C{step}] Running conditional forward (Input Len {current_input_len})...")
                        imputation_result_prompt = self.forward(
                            input_ids=forward_input_ids,
                            num_cycles=forward_internal_cycles,
                            use_level_prediction=use_level_prediction,
                            imputation_mode=False,
                            imputation_idx=None,
                        )
                        token_logits_prompt = imputation_result_prompt[0] if isinstance(imputation_result_prompt, tuple) else imputation_result_prompt
                        total_forward_passes += 1
                        # Validation...
                        expected_shape = (batch_size, current_input_len, vocab_size);
                        if token_logits_prompt is None or token_logits_prompt.shape != expected_shape: raise ValueError(f"Prompt logits shape error")
                        if not torch.isfinite(token_logits_prompt).all(): logger.warning(f"[C{blocks_processed} C{step}] Clamping non-finite prompt logits."); token_logits_prompt = torch.nan_to_num(token_logits_prompt, nan=-1e9, posinf=1e4, neginf=-1e4)
                    except Exception as e: logger.error(f"[Chunk{blocks_processed} C{step}] Cond forward failed: {e}", exc_info=True); break # Exit inner loop

                    final_token_logits_fullview = token_logits_prompt # Logits correspond to forward_input_ids
                    if cfg_enabled:
                        try:
                            # logger.debug(f"[Chunk{blocks_processed} C{step}] Running unconditional forward...")
                            prompt_len_in_view = max(0, prompt_len - context_start_idx)
                            null_context_part = torch.full((batch_size, prompt_len_in_view), mask_token_id, dtype=torch.long, device=device)
                            generation_part_in_view = forward_input_ids[:, prompt_len_in_view:]
                            null_forward_input_ids = torch.cat([null_context_part, generation_part_in_view], dim=1)
                            imputation_result_null = self.forward(
                                input_ids=null_forward_input_ids,
                                num_cycles=forward_internal_cycles,
                                use_level_prediction=use_level_prediction,
                                imputation_mode=False,
                                imputation_idx=None,
                            )
                            token_logits_null = imputation_result_null[0] if isinstance(imputation_result_null, tuple) else imputation_result_null
                            total_forward_passes += 1
                            # Validation...
                            if token_logits_null is None or token_logits_null.shape != expected_shape: raise ValueError(f"Null logits shape error")
                            if not torch.isfinite(token_logits_null).all(): logger.warning(f"[C{blocks_processed} C{step}] Clamping non-finite null logits."); token_logits_null = torch.nan_to_num(token_logits_null, nan=-1e9, posinf=1e4, neginf=-1e4)
                            final_token_logits_fullview = token_logits_null + guidance_scale * (token_logits_prompt - token_logits_null)
                        except Exception as e: logger.error(f"[Chunk{blocks_processed} C{step}] Uncond forward failed: {e}. Using prompt logits only.", exc_info=True)

                    # --- Process & Sample ONLY for the CURRENT CHUNK ---
                    if not torch.isfinite(final_token_logits_fullview).all(): logger.warning(f"[C{blocks_processed} C{step}] Clamping final logits!"); final_token_logits_fullview = torch.nan_to_num(final_token_logits_fullview, nan=-1e9, posinf=1e4, neginf=-1e4)

                    # Calculate start/end indices of the current chunk *within the forward_input_ids view*
                    chunk_start_in_view = max(0, chunk_start_idx - context_start_idx)
                    chunk_end_in_view   = chunk_start_in_view + current_chunk_len

                    # Slice logits to get only those corresponding to the current chunk
                    chunk_logits = final_token_logits_fullview[:, chunk_start_in_view:chunk_end_in_view, :]#.to(torch.float64)
                    if 0 <= int(mask_token_id) < int(chunk_logits.size(-1)):
                        chunk_logits = chunk_logits.clone()
                        chunk_logits[..., int(mask_token_id)] = torch.finfo(chunk_logits.dtype).min
                    #print("chunk_logits", chunk_logits.shape)
                    adjusted_temperature = temperature + (( temperature - 0.2) - temperature) * step / max(1, num_cycles - 1) # Annealing temperature
                    chunk_temp = adjusted_temperature if do_sample else 1.0
                    logits_for_sampling = chunk_logits / (chunk_temp if chunk_temp > 0 else 1.0)
                    chunk_probs = F.softmax(logits_for_sampling, dim=-1)
                    

                    if do_sample:
                        _logits_flat_chunk = chunk_logits.reshape(-1, vocab_size) # Raw guided logits for the chunk
                        if use_safe_sampling:
                             # --- Path A: Use safe_sample ---
                             sampled_ids_flat_chunk = torch.full((_logits_flat_chunk.size(0), 1), mask_token_id, dtype=torch.long, device=device)
                             #history_ids_for_penalty = current_ids[0, :chunk_start_idx]
                             history_ids_for_penalty = current_ids[0, chunk_start_idx-50: ]
                             for i in range(_logits_flat_chunk.size(0)):
                                  dbg_id=f"Safe_Chk{blocks_processed}T{i}_C{step}"; sampled_ids_flat_chunk[i] = safe_sample_from_logits(_logits_flat_chunk[i], adjusted_temperature, top_k, top_p, True, repetition_penalty, history_ids_for_penalty, int(fallback_token_id), dbg_id)
                        else:
                             # --- Path B: Use multinomial ---
                             probs_flat_chunk = chunk_probs.reshape(-1, vocab_size); probs_flat_chunk = probs_flat_chunk / probs_flat_chunk.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                             try:
                                 if not torch.isfinite(probs_flat_chunk).all(): raise ValueError("Non-finite probs");
                                 if (probs_flat_chunk < 0).any(): raise ValueError("Negative probs");
                                 if not (probs_flat_chunk.sum(dim=-1) > 1e-7).all(): raise ValueError("Zero-sum probs row");
                                 sampled_ids_flat_chunk = torch.multinomial(probs_flat_chunk, num_samples=1)
                             except Exception as e:
                                 logger.error(f"Multinomial fail Chk{blocks_processed} C{step}: {e}. Falling back.", exc_info=False)
                                 # Fallback uses safe_sample
                                 sampled_ids_flat_chunk = torch.full((_logits_flat_chunk.size(0), 1), mask_token_id, dtype=torch.long, device=device)
                                 for i in range(_logits_flat_chunk.size(0)):
                                     dbg_id = f"Fallback_Chk{blocks_processed}T{i}_C{step}"
                                     sampled_ids_flat_chunk[i] = safe_sample_from_logits(
                                         _logits_flat_chunk[i],
                                         temperature,
                                         top_k,
                                         top_p,
                                         True,
                                         repetition_penalty,
                                         None,
                                         int(fallback_token_id),
                                         dbg_id,
                                     )

                        sampled_chunk_ids = sampled_ids_flat_chunk.view(batch_size, current_chunk_len)
                    else: # Greedy for the chunk
                        sampled_chunk_ids = torch.argmax(chunk_logits, dim=-1)
                    

                    # Use original chunk_probs for confidence before any penalty/filtering inside safe_sample affects it
                    #original_chunk_logits = final_token_logits_fullview[:, chunk_start_idx:chunk_end_idx, :].to(torch.float32)
                    #original_chunk_probs = F.softmax(original_chunk_logits / (temperature if temperature > 0 else 1.0), dim=-1)
                    # Confidence = probability of the token that was actually sampled (whether greedy or sampled)
                    chunk_max_probs, _ = chunk_probs.max(dim=-1)
                    confidence_scores_chunk = torch.gather(chunk_probs, -1, sampled_chunk_ids.unsqueeze(-1)).squeeze(-1)
                    # # Confidence scores for the current chunk
                    # chunk_max_probs, _ = chunk_probs.max(dim=-1)
                    # confidence_scores_chunk = chunk_max_probs.copy() # Copy to avoid in-place operation
                    # --- Mask confidence scores of non-masked tokens ---
                    chunk_max_probs = torch.where(
                    current_ids[:, chunk_start_idx:chunk_end_idx] == mask_token_id,
                    chunk_max_probs,
                    torch.tensor(float('-inf'), device=device) # Set to -inf for non-masked tokens
                    )

                    # --- LLaDA/V1 Budget Annealing Acceptance (Applied ONLY to CURRENT CHUNK) ---
                    initial_budget = current_chunk_len
                    progress = step / max(1, num_cycles - 1)
                    # Quadratic budget increase
                    #current_budget = int(initial_budget * (1 - progress ** 2))
                    #linear
                    current_budget = int(initial_budget * (1.0 - progress))
                    current_budget = max(0, min(current_budget, current_chunk_len))

                    if current_budget > 0 and current_chunk_len > 0:
                        sorted_chunk_conf, sorted_chunk_indices = torch.sort(chunk_max_probs, dim=-1, descending=True)
                        if not sort_conf:
                            batch_size = sorted_chunk_indices.size(0)
                            chunk_len = sorted_chunk_indices.size(1)
                            # Create batch_size permutations
                            sorted_chunk_indices = torch.stack([
                                torch.randperm(chunk_len, device=device) for _ in range(batch_size)
                            ], dim=0)
                            #sorted_chunk_indices = torch.randperm(sorted_chunk_indices.size(1), device=device)
                        tokens_to_force_remask_indices = sorted_chunk_indices[:, :current_budget]
                        # force_remask_mask_chunk is TRUE for tokens in the chunk that MUST BE MASKED
                        force_remask_mask_chunk = torch.zeros_like(chunk_max_probs, dtype=torch.bool)
                        force_remask_mask_chunk.scatter_(1, tokens_to_force_remask_indices, True)
                    else:
                        force_remask_mask_chunk = torch.zeros_like(chunk_max_probs, dtype=torch.bool)

                    current_chunk_ids = current_ids[:, chunk_start_idx:chunk_end_idx]
                    initially_masked_in_chunk = (current_chunk_ids == mask_token_id) # Where was it masked at step start?

                    # Accept if NOT forced to remask by budget
                    final_accept_mask_chunk = initially_masked_in_chunk & (~force_remask_mask_chunk)
                    #final_accept_mask_chunk = ~force_remask_mask_chunk
                    
                    
                    #if step < num_cycles - 2 and step > num_cycles * 0.2: # Only apply confidence threshold if last 20% of cycles or the first 20 percent of the cycles
                    if step > num_cycles * 0.4 and step < num_cycles * 0.6: # Only apply confidence threshold if last 20% of cycles or the first 20 percent of the cycles
                        # --- Confidence Threshold Re-masking ---
                        confidence_threshold = initial_threshold + (final_threshold - initial_threshold) * progress # Annealing threshold
                    else:
                        confidence_threshold = 0.00
                    low_confidence_mask = (confidence_scores_chunk < confidence_threshold)
                    # --- End Confidence Threshold ---


                    # --- Combine Re-masking Conditions ---
                    # Force remask if EITHER selected by budget OR below confidence threshold
                    force_remask_final = force_remask_mask_chunk | low_confidence_mask
                    # --- End Combine ---


                    # --- Determine final ACCEPTANCE mask ---
                    # Accept = NOT forced to remask by either condition
                    final_accept_mask_chunk = ~force_remask_final
                    # ---


                    # --- Update current_ids using torch.where and SAMPLED IDs ---
                    current_chunk_ids = current_ids[:, chunk_start_idx:chunk_end_idx]
                    mask_token_tensor = torch.tensor(mask_token_id, device=device, dtype=torch.long)

                    # If accepted (final_accept_mask_chunk is True), use the sampled token.
                    # If not accepted (force_remask_final is True), use the mask token.
                    new_chunk_ids = torch.where(
                        final_accept_mask_chunk,
                        sampled_chunk_ids,   # <<< USE THE SAMPLED IDS HERE <<<
                        mask_token_tensor
                    )
                    current_ids[:, chunk_start_idx:chunk_end_idx] = new_chunk_ids


                    # # Update current_ids:
                    # # 1. Put sampled IDs into accepted positions
                    # current_ids[:, chunk_start_idx:chunk_end_idx][final_accept_mask_chunk] = sampled_ids_to_accept
                    # # 2. Put MASK ID into positions that were masked but rejected by budget
                    # current_ids[:, chunk_start_idx:chunk_end_idx][indices_to_remask_in_chunk] = mask_token_id

                    # # --- Update ONLY the current chunk in the main current_ids tensor ---
                    # new_chunk_part = torch.where(
                    #     final_accept_mask_chunk,
                    #     sampled_chunk_ids,
                    #     torch.tensor(mask_token_id, device=device, dtype=torch.long)
                    # )
                    # current_ids[:, chunk_start_idx:chunk_end_idx] = new_chunk_part

                    # Logging for the chunk
                    num_accepted_this_step = final_accept_mask_chunk.sum().item()
                    num_remasked_this_step = force_remask_final.sum().item()
                    #num_remasked_this_step = indices_to_remask_in_chunk.sum().item()
                    logger.debug(f"[Chunk{blocks_processed} C{step}/{num_cycles-1}] Budget: {current_budget}, Accepted: {num_accepted_this_step}, Forced Remask: {num_remasked_this_step}, Confidence Cutoff: {confidence_threshold:.2f}, Current Temperature: {adjusted_temperature:.2f}")

                    if not (current_ids[:, chunk_start_idx:chunk_end_idx] == mask_token_id).any():
                        logger.debug(f"Chunk {blocks_processed} completed at cycle {step}.")
                        break # Early exit inner loop for this chunk

                # --- End of Diffusion Loop for Chunk ---
                if (current_ids[:, chunk_start_idx:chunk_end_idx] == mask_token_id).all():
                     logger.warning(f"Chunk {blocks_processed} failed: All tokens still masks after {num_cycles} cycles. Stopping."); break # Stop outer loop
                # --- Finalize the chunk: Fill any remaining unresolved tokens ---
                unresolved = current_ids.eq(mask_token_id)[:, chunk_start_idx:chunk_end_idx]
                if unresolved.any():
                    logits = final_token_logits_fullview[:, chunk_end_idx:chunk_end_idx]
                    current_ids[:, chunk_end_idx:chunk_end_idx].masked_scatter_(unresolved, logits.argmax(-1)[unresolved])
                    logger.debug(f"Chunk {blocks_processed} had unresolved tokens. Filling them with logits.")
                generated_count_in_chunks += current_chunk_len
                logger.debug(f"Finished refining chunk {blocks_processed}. Total generated tokens in chunks: {generated_count_in_chunks}/{generate_length}")

            # --- End of While Loop (Chunk Processing) ---

            # --- Timing & Stats Calculation & Logging ---
            end_time = time.perf_counter(); elapsed_time = end_time - start_time
            # Use generated_count_in_chunks for accuracy if loop broke early
            actual_generated_tokens = min(generated_count_in_chunks, generate_length)
            final_len = prompt_len + actual_generated_tokens
            tokens_per_second = actual_generated_tokens / elapsed_time if elapsed_time > 1e-6 and actual_generated_tokens > 0 else 0.0
            logger.info(f"--- Generation Summary ---"); logger.info(f"Target generated length: {generate_length}"); logger.info(f"Actual generated tokens: {actual_generated_tokens}"); logger.info(f"Final total length: {final_len}")
            logger.info(f"Chunks processed: {blocks_processed}"); logger.info(f"Total forward passes: {total_forward_passes}"); logger.info(f"Total generation time: {elapsed_time:.3f} seconds"); logger.info(f"Tokens per second: {tokens_per_second:.2f} T/s"); logger.info(f"-------------------------")

            return current_ids[:, :final_len] # Return the final state up to generated length

    #4o simple, does work well
    def _generate_with_diffusion_from_prompt_old(
            self,
            input_ids: torch.Tensor,
            generate_length: int = 100,
            temperature: float = 1.0,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.9,
            repetition_penalty: float = 1.0,
            use_level_prediction: bool = False,
            num_cycles: int = 100,
            initial_threshold: float = 0.9,
            final_threshold: float = 0.6,
        ) -> torch.Tensor:
        """
        Diffusion-style sampling: keep prompt fixed, append masks, iteratively generate into masked block
        with dynamic confidence threshold and mask budget annealing.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        #do_sample = False
        # Find mask token
        mask_token_id = getattr(self, 'mask_token_id', None)
        if mask_token_id is None and hasattr(self, 'tokenizer'):
            mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)
        if mask_token_id is None:
            mask_token_id = getattr(self, 'pad_token_id', 0)

        # Find vocab size
        vocab_size = getattr(self, 'vocab_size', None)
        if vocab_size is None:
            vocab_size = self.get_output_embeddings().weight.shape[0]

        # Prepare masked generation block
        mask_block = torch.full((batch_size, generate_length), mask_token_id, dtype=torch.long, device=device)
        current_ids = torch.cat([input_ids, mask_block], dim=1)

        prompt_len = seq_len

        with torch.no_grad():
            for step in range(num_cycles):
                # 1. Dynamic remask threshold
                current_threshold = initial_threshold + (final_threshold - initial_threshold) * (step / (num_cycles - 1))

                # 2. Forward pass
                imputation_result = self.forward(
                    input_ids=current_ids,
                    num_cycles=self.refinement_cycles,
                    use_level_prediction=use_level_prediction,
                    imputation_mode=False,
                    imputation_idx= None
                )

                # Make safe
                if isinstance(imputation_result, tuple):
                    token_logits = imputation_result[0]
                else:
                    token_logits = imputation_result  # Single output, assume it's logits

                if token_logits is None or token_logits.numel() == 0:
                    logger.error("Invalid logits during diffusion generation. Stopping.")
                    break
                if token_logits.dim() < 2:
                    logger.error(f"Unexpected logits dimension: {token_logits.shape}")
                    break
                #print(token_logits.shape, "token_logits shape")
                # 3. Sampling
                logits = token_logits / temperature
                probs = F.softmax(logits, dim=-1)

                if do_sample:
                    sampled_ids = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(batch_size, -1)
                else:
                    sampled_ids = torch.argmax(probs, dim=-1)

                max_probs, _ = probs.max(dim=-1)  # Confidence scores

                # Only apply updates to masked block (after prompt)
                masked_block_positions = torch.arange(current_ids.size(1), device=device) >= prompt_len

                # 4. Dynamic confidence-based acceptance
                accept_mask = (max_probs >= current_threshold) & masked_block_positions.unsqueeze(0)
                remask_positions = (~accept_mask) & masked_block_positions.unsqueeze(0)

                # 5. Mask budget annealing
                initial_budget = (current_ids.size(1) - prompt_len)
                final_budget = 0
                current_budget = int(initial_budget * (1 - step / (num_cycles - 1)))

                masked_confidences = max_probs[:, prompt_len:]  # Only in masked block
                sorted_confidences, sorted_indices = torch.sort(masked_confidences, dim=-1)  # Ascending

                # Tokens to keep masked (lowest confidence)
                tokens_to_remask = sorted_indices[:, :current_budget]

                # Create remask mask
                full_remask_mask = torch.zeros_like(masked_confidences, dtype=torch.bool)
                #full_remask_mask.scatter_(1, tokens_to_remask, 1)
                full_remask_mask.scatter_(1, tokens_to_remask, 1)

                absolute_remask_mask = torch.zeros_like(current_ids, dtype=torch.bool)
                absolute_remask_mask[:, prompt_len:] = full_remask_mask  # Apply to full sequence

                # Final acceptance: accept all tokens not in remask mask
                final_accept_mask = masked_block_positions.unsqueeze(0) & (~absolute_remask_mask)

                # 6. Update tokens
                current_ids[final_accept_mask] = sampled_ids[final_accept_mask]
                current_ids[absolute_remask_mask] = mask_token_id

        return current_ids

    

    def _safe_sampling(self, logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, 
                    current_ids=None, do_sample=True):
        """
        Safe sampling method with additional checks for vocab size issues
        """
        # Apply temperature
        next_token_logits = logits / max(temperature, 1e-8)  # avoid division by zero
        
        # Apply repetition penalty
        if repetition_penalty != 1.0 and current_ids is not None:
            for token_id in set(current_ids[0].tolist()):
                # Make sure token_id is within vocab range
                if 0 <= token_id < next_token_logits.size(-1):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
        
        # Apply top-k filtering
        if top_k > 0:
            # Cap top_k to avoid out of bounds
            top_k = min(top_k, next_token_logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Apply top-p filtering
        if top_p < 1.0:
            try:
                # Sort logits and compute cumulative probabilities
                sorted_logits, sorted_indices = torch.sort(next_token_logits, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices to original logits
                for batch_idx in range(next_token_logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            except Exception as e:
                # If there's an error with top-p, fall back to just using softmax
                print(f"Warning: Error in top-p filtering ({e}), falling back to basic sampling")
        
        # Apply softmax to get probabilities
        try:
            probs = F.softmax(next_token_logits, dim=-1)
        except Exception as e:
            # Handle any issues with softmax
            print(f"Warning: Error in softmax computation ({e}), falling back to basic normalization")
            # Basic fallback normalization 
            exp_logits = torch.exp(next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0])
            probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        
        # Either sample from the distribution or take the most likely token
        if do_sample:
            try:
                next_token = torch.multinomial(probs, num_samples=1)
            except Exception as e:
                # If sampling fails, fall back to argmax
                print(f"Warning: Error in sampling ({e}), falling back to argmax")
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        
        # Final check to ensure token is within vocab range
        if (next_token >= self.vocab_size).any():
            print(f"Warning: Generated token {next_token.item()} exceeds vocab size {self.vocab_size}, replacing with UNK")
            # Replace out-of-bounds tokens with a safe token (usually UNK or the last valid token)
            safe_token = min(0, self.vocab_size - 1)  # Use either 0 or the last valid token
            next_token = torch.where(next_token >= self.vocab_size, 
                                    torch.tensor(safe_token, device=next_token.device), 
                                    next_token)
            
        return next_token
