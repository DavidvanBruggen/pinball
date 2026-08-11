#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""
Fixed version of hybrid_training.py that handles device correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import time
from contextlib import nullcontext
from tqdm import tqdm
import math
from typing import Any, Dict, Optional, Tuple

from ..utils.amp import amp_dtype, bf16_supported


logger = logging.getLogger(__name__)

COPY_TASK_SPECIAL_TOKENS = ["<copy_src>", "</copy_src>", "<copy_dst>", "</copy_dst>"]
COPY_TASK_LOG_BUCKET_EDGES = (32, 64, 128, 256, 512, 1024, 2048)


def ensure_copy_task_tokens(tokenizer) -> int:
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    to_add = [tok for tok in COPY_TASK_SPECIAL_TOKENS if tok not in existing]
    if not to_add:
        return 0
    return int(tokenizer.add_special_tokens({"additional_special_tokens": to_add}))


def get_copy_task_marker_ids(tokenizer):
    ids = {
        "copy_src_open": tokenizer.convert_tokens_to_ids("<copy_src>"),
        "copy_src_close": tokenizer.convert_tokens_to_ids("</copy_src>"),
        "copy_dst_open": tokenizer.convert_tokens_to_ids("<copy_dst>"),
        "copy_dst_close": tokenizer.convert_tokens_to_ids("</copy_dst>"),
    }
    if any(v is None or int(v) < 0 for v in ids.values()):
        raise ValueError("Copy-task marker tokens were not resolved in tokenizer vocabulary")
    return {k: int(v) for k, v in ids.items()}


def apply_copy_task_to_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    marker_ids,
    apply_prob: float,
    src_len_min: int,
    src_len_max: int,
    min_gap: int,
    max_gap: int,
):
    if apply_prob <= 0.0:
        B, T = input_ids.shape
        meta = {
            "applied": torch.zeros((B,), dtype=torch.bool, device=input_ids.device),
            "distance": torch.full((B,), -1, dtype=torch.long, device=input_ids.device),
            "dst_payload_mask": torch.zeros((B, T), dtype=torch.bool, device=input_ids.device),
        }
        return input_ids, meta

    out = input_ids.clone()
    B, T = out.shape
    device = out.device

    dst_payload_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    applied = torch.zeros((B,), dtype=torch.bool, device=device)
    distance = torch.full((B,), -1, dtype=torch.long, device=device)

    src_len_min = max(1, int(src_len_min))
    src_len_max = max(src_len_min, int(src_len_max))
    min_gap = max(0, int(min_gap))
    max_gap = int(max_gap)

    for b in range(B):
        if float(torch.rand((), device=device).item()) > float(apply_prob):
            continue

        if attention_mask is not None:
            valid_len = int(attention_mask[b].sum().item())
        else:
            valid_len = T
        valid_len = min(valid_len, T)

        max_src_feasible = (valid_len - min_gap - 4) // 2
        max_src = min(src_len_max, int(max_src_feasible))
        if max_src < src_len_min:
            continue

        src_len = int(torch.randint(src_len_min, max_src + 1, (1,), device=device).item())
        max_gap_feasible = valid_len - (2 * src_len + 4)
        if max_gap_feasible < min_gap:
            continue
        gap_hi = max_gap_feasible if max_gap <= 0 else min(max_gap, max_gap_feasible)
        if gap_hi < min_gap:
            continue
        gap = int(torch.randint(min_gap, gap_hi + 1, (1,), device=device).item())

        total_span = 2 * src_len + gap + 4
        start_hi = valid_len - total_span
        if start_hi < 0:
            continue
        start = int(torch.randint(0, start_hi + 1, (1,), device=device).item())

        pick_hi = valid_len - src_len
        if pick_hi < 0:
            continue
        src_pick = int(torch.randint(0, pick_hi + 1, (1,), device=device).item())
        payload = out[b, src_pick:src_pick + src_len].clone()

        src_open = start
        src_payload_start = src_open + 1
        src_payload_end = src_payload_start + src_len
        src_close = src_payload_end

        dst_open = src_close + 1 + gap
        dst_payload_start = dst_open + 1
        dst_payload_end = dst_payload_start + src_len
        dst_close = dst_payload_end
        if dst_close >= valid_len:
            continue

        out[b, src_open] = int(marker_ids["copy_src_open"])
        out[b, src_payload_start:src_payload_end] = payload
        out[b, src_close] = int(marker_ids["copy_src_close"])

        out[b, dst_open] = int(marker_ids["copy_dst_open"])
        out[b, dst_payload_start:dst_payload_end] = payload
        out[b, dst_close] = int(marker_ids["copy_dst_close"])

        dst_payload_mask[b, dst_payload_start:dst_payload_end] = True
        applied[b] = True
        distance[b] = int(gap)

    meta = {
        "applied": applied,
        "distance": distance,
        "dst_payload_mask": dst_payload_mask,
    }
    return out, meta


def log_bucket_label(distance: int, edges=COPY_TASK_LOG_BUCKET_EDGES) -> str:
    d = int(distance)
    prev = 0
    for edge in edges:
        if d <= int(edge):
            return f"d{prev}_{int(edge)}"
        prev = int(edge) + 1
    return f"d{prev}_plus"


def copy_log_bucket_sort_key(label: str):
    if not isinstance(label, str) or not label.startswith("d"):
        return (10**12, 10**12, str(label))
    body = label[1:]
    if body.endswith("_plus"):
        low = body[:-5]
        try:
            low_i = int(low)
        except Exception:
            low_i = 10**12
        return (low_i, 10**12, label)
    parts = body.split("_", 1)
    if len(parts) != 2:
        return (10**12, 10**12, label)
    try:
        low_i = int(parts[0])
    except Exception:
        low_i = 10**12
    try:
        high_i = int(parts[1])
    except Exception:
        high_i = 10**12
    return (low_i, high_i, label)


def hierarchy_thresholds_from_compression(compression_ratios):
    if not compression_ratios:
        return ()
    thresholds = []
    prod = 1
    for ratio in compression_ratios:
        try:
            r = int(ratio)
        except Exception:
            continue
        if r <= 0:
            continue
        prod *= r
        thresholds.append(prod)
    return tuple(thresholds)


def hierarchy_bucket_label(distance: int, thresholds) -> str:
    d = int(distance)
    if not thresholds:
        return "h_unknown"
    if d <= int(thresholds[0]):
        return "h_l0"
    if len(thresholds) >= 2 and d <= int(thresholds[1]):
        return "h_l1"
    if len(thresholds) >= 3 and d <= int(thresholds[2]):
        return "h_l2"
    return "h_l3_plus"

#############################################
# Hybrid Masking Implementation
#############################################

def hybrid_mask_tokens(input_ids, tokenizer, mask_prob=0.15):
    """
    Apply hybrid masking: random tokens + always mask the last token.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: Tokenizer with mask token
        mask_prob: Probability of masking random tokens
        
    Returns:
        masked_input_ids: Input with masked tokens
        target_tokens: Ground truth tokens at masked positions (-100 elsewhere)
    """
    masked_input_ids = input_ids.clone()
    target_tokens = torch.full_like(input_ids, -100)
    
    # Get mask token ID or use UNK as fallback
    mask_token_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, 'unk_token_id', None)
    if mask_token_id is None:
        raise ValueError("Tokenizer does not have a mask_token_id or unk_token_id.")
    
    # 1. Randomly mask tokens in the sequence (except the last one)
    for i in range(input_ids.size(0)):  # For each sample in batch
        seq_len = input_ids.size(1)
        if seq_len <= 2:  # Skip if sequence is too short
            continue
            
        # Only mask positions 1 to seq_len-2 (leave first and last for now)
        for pos in range(1, seq_len-1):
            if torch.rand(1).item() < mask_prob:
                target_tokens[i, pos] = input_ids[i, pos]  # Store original token
                masked_input_ids[i, pos] = mask_token_id   # Replace with mask
    
    # 2. Always mask the last token (if sequence is long enough)
    for i in range(input_ids.size(0)):
        seq_len = input_ids.size(1)
        if seq_len > 1:  # Make sure there's at least 2 tokens
            last_pos = seq_len - 1
            target_tokens[i, last_pos] = input_ids[i, last_pos]  # Store original token
            masked_input_ids[i, last_pos] = mask_token_id        # Replace with mask
    
    return masked_input_ids, target_tokens

def _infer_grid_shape(seq_len: int, grid_shape: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    if grid_shape is not None:
        gh = int(grid_shape[0])
        gw = int(grid_shape[1])
        if gh > 0 and gw > 0 and gh * gw == int(seq_len):
            return (gh, gw)
        return None
    side = int(round(math.sqrt(max(1, int(seq_len)))))
    if side * side == int(seq_len):
        return (side, side)
    return None


def _sample_block_mask_2d(
    bsz: int,
    gh: int,
    gw: int,
    mask_prob: float,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((bsz, gh, gw), dtype=torch.bool, device=device)
    total = int(gh * gw)
    target = max(1, int(round(float(mask_prob) * total)))
    blk = max(1, int(block_size))
    max_tries = max(8, target * 4)
    for b in range(bsz):
        covered = 0
        tries = 0
        while covered < target and tries < max_tries:
            tries += 1
            h0 = int(torch.randint(0, max(1, gh), (1,), device=device).item())
            w0 = int(torch.randint(0, max(1, gw), (1,), device=device).item())
            h1 = min(gh, h0 + blk)
            w1 = min(gw, w0 + blk)
            prev = int(mask[b].sum().item())
            mask[b, h0:h1, w0:w1] = True
            covered = int(mask[b].sum().item())
            if covered == prev:
                continue
    return mask.view(bsz, gh * gw)


def _sample_path_mask_2d(
    bsz: int,
    gh: int,
    gw: int,
    mask_prob: float,
    path_len: int,
    device: torch.device,
) -> torch.Tensor:
    total = int(gh * gw)
    target = max(1, int(round(float(mask_prob) * total)))
    step_limit = max(target, int(path_len))
    mask = torch.zeros((bsz, gh, gw), dtype=torch.bool, device=device)
    dirs = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=device, dtype=torch.long)
    for b in range(bsz):
        y = int(torch.randint(0, max(1, gh), (1,), device=device).item())
        x = int(torch.randint(0, max(1, gw), (1,), device=device).item())
        for _ in range(step_limit * 2):
            mask[b, y, x] = True
            if int(mask[b].sum().item()) >= target:
                break
            d_idx = int(torch.randint(0, 4, (1,), device=device).item())
            dy = int(dirs[d_idx, 0].item())
            dx = int(dirs[d_idx, 1].item())
            y = max(0, min(gh - 1, y + dy))
            x = max(0, min(gw - 1, x + dx))
    return mask.view(bsz, gh * gw)


def hybrid_diffusion_mask_tokens(
    input_ids,
    tokenizer,
    val=False,
    mode: str = "random",
    grid_shape: Optional[Tuple[int, int]] = None,
    block_size: int = 4,
    path_length: int = 64,
    return_mask_prob: bool = False,
):
    """
    Apply diffusion-style hybrid masking: random tokens + always mask the last token.
    Mask ratio is sampled randomly each time.
    """
    masked_input_ids = input_ids.clone()
    target_tokens = torch.full_like(input_ids, -100)
    
    # Get mask token ID or fallback
    mask_token_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, 'unk_token_id', None)
    if mask_token_id is None:
        raise ValueError("Tokenizer does not have a mask_token_id or unk_token_id.")
    
    # Sample random mask probability [0, 1] per batch
    mask_prob = torch.rand(1).item()
    if val:
        mask_prob = 0.15
    
    # # Randomly mask tokens
    # for i in range(input_ids.size(0)):
    #     seq_len = input_ids.size(1)
    #     print(input_ids.size(), input_ids.size(1), "input ids size")
    #     if seq_len <= 2:
    #         continue

    #     for pos in range(0, seq_len):  # Mask entire sequence
    #         if torch.rand(1).item() < mask_prob:
    #             target_tokens[i, pos] = input_ids[i, pos]
    #             masked_input_ids[i, pos] = mask_token_id
    # # Randomly mask tokens but mask in one batch not iterating over the batch
    # seq_len = input_ids.size(1)
    # if seq_len <= 2:
    #     return masked_input_ids, target_tokens
    # for pos in range(0, seq_len):  # Mask entire sequence
    #     if torch.rand(1).item() < mask_prob:
    #         target_tokens[:, pos] = input_ids[:, pos]
    #         masked_input_ids[:, pos] = mask_token_id

    B, T = input_ids.shape
    mode_norm = str(mode).lower()
    if mode_norm not in {"random", "block", "path"}:
        mode_norm = "random"

    if mode_norm == "random":
        mask = torch.rand_like(input_ids, dtype=torch.float) < mask_prob  # [B, T] boolean
    else:
        inferred = _infer_grid_shape(T, grid_shape=grid_shape)
        if inferred is None:
            mask = torch.rand_like(input_ids, dtype=torch.float) < mask_prob
        else:
            gh, gw = inferred
            if mode_norm == "block":
                mask = _sample_block_mask_2d(B, gh, gw, mask_prob, block_size=block_size, device=input_ids.device)
            else:
                mask = _sample_path_mask_2d(B, gh, gw, mask_prob, path_len=path_length, device=input_ids.device)
    
    # Optionally avoid masking special tokens:
    # mask &= (input_ids != pad_id) & (input_ids != cls_id) & ...

    target_tokens[mask] = input_ids[mask]
    masked_input_ids[mask] = mask_token_id


        # for pos in range(0, seq_len-1):  # Mask entire sequence except last token
        #     if torch.rand(1).item() < mask_prob:
        #         target_tokens[i, pos] = input_ids[i, pos]
        #         masked_input_ids[i, pos] = mask_token_id
    
        # Always mask the last token
    #for i in range(input_ids.size(0)):
    #    seq_len = input_ids.size(1)
    #    if seq_len > 1:
    #        last_pos = seq_len - 1#64
    #        target_tokens[i, last_pos] = input_ids[i, last_pos]
    #        masked_input_ids[i, last_pos] = mask_token_id
    
    if return_mask_prob:
        return masked_input_ids, target_tokens, float(mask_prob)
    return masked_input_ids, target_tokens

def hybrid_diffusion_mask_tokens_newest_suffix(input_ids, tokenizer, val=False):
    masked_input_ids = input_ids.clone()
    target_tokens    = torch.full_like(input_ids, -100)

    mask_token_id = getattr(tokenizer, 'mask_token_id', None) or getattr(tokenizer, 'unk_token_id', None)
    if mask_token_id is None:
        raise ValueError("Tokenizer does not have a mask_token_id or unk_token_id.")

    device = input_ids.device
    B, T   = input_ids.shape

    # keep torch.rand
    mask_prob = 0.15 if val else torch.rand(1, device=device).item()

    # ---- suffix mask (shared) ----
    suffix_len      = min(64, T)
    suffix_mask_1T  = torch.zeros(T, dtype=torch.bool, device=device)
    if suffix_len > 0:
        suffix_mask_1T[-suffix_len:] = True

    # ---- random prefix mask (shared) using torch.rand ----
    prefix_len      = T - suffix_len
    rand_mask_1T    = torch.zeros(T, dtype=torch.bool, device=device)
    if prefix_len > 0:
        # Bernoulli via torch.rand
        bern = torch.rand(prefix_len, device=device) < mask_prob
        rand_mask_1T[:prefix_len] = bern

        # # (optional exact-K version, also torch.rand-based)
        # K = max(1, int(round(mask_prob * prefix_len)))
        # sel = torch.randperm(prefix_len, device=device)[:K]
        # rand_mask_1T.zero_()
        # rand_mask_1T[sel] = True

    # final shared mask and broadcast
    mask_1T = rand_mask_1T | suffix_mask_1T          # [T]
    mask_BT = mask_1T.unsqueeze(0).expand(B, T)      # [B,T]

    target_tokens[mask_BT]    = input_ids[mask_BT]
    masked_input_ids[mask_BT] = mask_token_id
    return masked_input_ids, target_tokens

def train_with_hybrid_masking(model, batch, criterion, optimizer, tokenizer,
                             gradient_accumulation_steps=1, mixed_precision=False, scaler=None, device=None, variable_cycles=None, use_bf16=True,
                             objective_mode="masked", lambda_masked=1.0, lambda_ar=0.1, lambda_base_ce=1.0, lambda_copy=0.0, lambda_unmasked=0.0, lambda_autoenc=0.0, lambda_autoenc_next=0.0,
                             ce_label_smoothing=0.0,
                              lambda_token_unet_lookahead_ce=0.0,
                              chunked_ce_enable=False,
                              chunked_ce_seq_chunk=0,
                              train_feature_chunked_ce_enable=False,
                              diffusion_mask_mode="random",
                             diffusion_mask_block_size=4,
                             diffusion_mask_path_length=64,
                             diffusion_grid_shape: Optional[Tuple[int, int]] = None,
                             autoenc_training_policy="auxiliary",
                             llada_loss_weighting=False):
    """
    Process a batch with hybrid masking (random + last token).
    
    Args:
        model: The hierarchical graph model
        batch: Training batch with input_ids and labels
        criterion: Loss criterion
        optimizer: Optimizer
        tokenizer: Tokenizer with mask token
        gradient_accumulation_steps: Steps for gradient accumulation
        mixed_precision: Whether to use mixed precision training
        scaler: Gradient scaler for mixed precision
        device: Device to use (if None, will try to determine from model)
    
    Returns:
        loss: Training loss
    """
    # Determine device if not provided
    if device is None:
        device = next(model.parameters()).device
    
    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    labels = batch.get("labels", None)
    if labels is not None:
        labels = labels.to(device)
    else:
        labels = torch.full_like(input_ids, -100)
        if input_ids.size(1) > 1:
            labels[:, :-1] = input_ids[:, 1:]
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    copy_dst_mask = batch.get("copy_dst_mask", None)
    if copy_dst_mask is not None:
        copy_dst_mask = copy_dst_mask.to(device=device, dtype=torch.bool)
    copy_force_mask_dst = bool(batch.get("copy_force_mask_dst", False))
    copy_mask_dst_in_ar = bool(batch.get("copy_mask_dst_in_ar", False))
    copy_mask_token_id = batch.get("copy_mask_token_id", None)
    
    objective_mode = str(objective_mode).lower()
    if objective_mode not in {"masked", "ar", "hybrid"}:
        objective_mode = "masked"
    lambda_masked = float(lambda_masked)
    lambda_ar = float(lambda_ar)
    lambda_copy = float(lambda_copy)
    lambda_unmasked = float(lambda_unmasked)
    lambda_autoenc = float(lambda_autoenc)
    lambda_autoenc_next = float(lambda_autoenc_next)
    lambda_token_unet_lookahead_ce = float(lambda_token_unet_lookahead_ce)
    chunked_ce_enable = bool(chunked_ce_enable)
    chunked_ce_seq_chunk = max(0, int(chunked_ce_seq_chunk))
    train_feature_chunked_ce_enable = bool(train_feature_chunked_ce_enable)
    diffusion_mask_mode = str(diffusion_mask_mode).lower()
    if diffusion_mask_mode not in {"random", "block", "path"}:
        diffusion_mask_mode = "random"
    diffusion_mask_block_size = max(1, int(diffusion_mask_block_size))
    diffusion_mask_path_length = max(1, int(diffusion_mask_path_length))
    model_mod_for_mode = getattr(model, "module", None)
    autoenc_graph_mode = str(
        getattr(
            model,
            "autoenc_graph_mode",
            getattr(model_mod_for_mode, "autoenc_graph_mode", "off"),
        )
    ).lower()
    autoenc_training_policy = str(autoenc_training_policy).lower()
    if autoenc_training_policy not in {"auxiliary", "autoenc_only", "autoenc_only_diffusion"}:
        autoenc_training_policy = "auxiliary"
    autoenc_only_mode = (
        autoenc_graph_mode == "twin_shared_l3"
        and autoenc_training_policy == "autoenc_only"
    )
    autoenc_only_diffusion_mode = (
        autoenc_graph_mode == "twin_shared_l3"
        and autoenc_training_policy == "autoenc_only_diffusion"
    )
    effective_objective_mode = "masked" if autoenc_only_diffusion_mode else objective_mode
    lambda_base_ce = float(lambda_base_ce)
    ce_label_smoothing = max(0.0, float(ce_label_smoothing))

    def _set_train_loss_stats(base_ce_loss_value=None, copy_ce_loss_value=None, copy_tokens_value=0, autoenc_ce_loss_value=None, autoenc_next_ce_loss_value=None, token_unet_lookahead_ce_loss_value=None, objective_loss_value=None):
        targets = [model]
        model_mod = getattr(model, "module", None)
        if model_mod is not None:
            targets.append(model_mod)
        for tgt in targets:
            tgt._last_ce_loss = None if base_ce_loss_value is None else float(base_ce_loss_value)
            tgt._last_copy_dst_ce_loss = None if copy_ce_loss_value is None else float(copy_ce_loss_value)
            tgt._last_copy_dst_token_count = int(copy_tokens_value)
            tgt._last_autoenc_ce_loss = None if autoenc_ce_loss_value is None else float(autoenc_ce_loss_value)
            tgt._last_autoenc_next_ce_loss = None if autoenc_next_ce_loss_value is None else float(autoenc_next_ce_loss_value)
            tgt._last_token_unet_lookahead_ce_loss = None if token_unet_lookahead_ce_loss_value is None else float(token_unet_lookahead_ce_loss_value)
            tgt._last_objective_loss = None if objective_loss_value is None else float(objective_loss_value)

    def _set_objective_loss_stat(objective_loss_value=None):
        targets = [model]
        model_mod = getattr(model, "module", None)
        if model_mod is not None:
            targets.append(model_mod)
        for tgt in targets:
            tgt._last_objective_loss = None if objective_loss_value is None else float(objective_loss_value)

    def _set_token_unet_runtime_flags(emit_lookahead_logits: bool):
        targets = [model]
        model_mod = getattr(model, "module", None)
        if model_mod is not None:
            targets.append(model_mod)
        for tgt in targets:
            setattr(tgt, "_token_unet_emit_lookahead_logits", bool(emit_lookahead_logits))

    def _chunked_ce_mean(
        logits_btv: torch.Tensor,
        targets_bt: torch.Tensor,
        mask_bt: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        if logits_btv is None or logits_btv.numel() == 0 or logits_btv.dim() != 3:
            zero = torch.zeros((), device=device)
            return zero, 0

        B, T, _ = logits_btv.shape
        if T <= 0:
            zero = torch.zeros((), device=logits_btv.device, dtype=logits_btv.dtype)
            return zero, 0

        use_chunk = bool(chunked_ce_enable and chunked_ce_seq_chunk > 0 and T > chunked_ce_seq_chunk)
        chunk = int(chunked_ce_seq_chunk) if use_chunk else T

        numer = torch.zeros((), device=logits_btv.device, dtype=logits_btv.dtype)
        denom = 0

        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            logits_slice = logits_btv[:, s:e, :]
            target_slice = targets_bt[:, s:e]
            # Flattened 2D CE: the [B, V, T] spatial variant dispatches a kernel that is
            # ~7x slower at LM vocab sizes (measured 256ms vs 34ms per step at T=8192).
            ce_slice = F.cross_entropy(
                logits_slice.reshape(-1, logits_slice.size(-1)),
                target_slice.reshape(-1),
                reduction="none",
                label_smoothing=ce_label_smoothing,
                ignore_index=(int(ignore_index) if ignore_index is not None else -100),
            ).view_as(target_slice)

            if mask_bt is not None:
                valid = mask_bt[:, s:e].to(device=ce_slice.device, dtype=torch.bool)
            elif ignore_index is not None:
                valid = (target_slice != int(ignore_index))
            else:
                valid = torch.ones_like(target_slice, dtype=torch.bool, device=ce_slice.device)

            if not bool(valid.any()):
                continue

            numer = numer + (ce_slice * valid.to(dtype=ce_slice.dtype)).sum()
            denom += int(valid.sum().item())

        if denom <= 0:
            zero = torch.zeros((), device=logits_btv.device, dtype=logits_btv.dtype)
            return zero, 0
        return numer / float(denom), int(denom)

    def _model_module():
        return getattr(model, "module", model)

    def _output_projection_module():
        return getattr(_model_module(), "output_projection", None)

    def _is_token_feature_tensor(value: torch.Tensor) -> bool:
        proj = _output_projection_module()
        if proj is None or value is None or value.dim() != 3:
            return False
        in_features = getattr(proj, "in_features", None)
        return in_features is not None and int(value.size(-1)) == int(in_features)

    def _project_token_features(features_bth: torch.Tensor) -> torch.Tensor:
        proj = _output_projection_module()
        if proj is None:
            raise RuntimeError("Model returned token features but has no output_projection for CE")
        return proj(features_bth)

    def _chunked_feature_ce_mean(
        features_bth: torch.Tensor,
        targets_bt: torch.Tensor,
        mask_bt: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        if features_bth is None or features_bth.numel() == 0 or features_bth.dim() != 3:
            zero = torch.zeros((), device=device)
            return zero, 0
        B, T, _ = features_bth.shape
        if T <= 0:
            zero = torch.zeros((), device=features_bth.device, dtype=features_bth.dtype)
            return zero, 0
        chunk = int(chunked_ce_seq_chunk) if int(chunked_ce_seq_chunk) > 0 else T
        # Auto-scale the position-chunk UP for small batch: memory per chunk ~ B*chunk*vocab,
        # so a fixed position count fragments a long sequence at batch 1 into hundreds of tiny
        # launch-bound GEMMs. Keep >= ~token_floor tokens (B*chunk) per chunk -> chunk count
        # stays small and memory stays bounded. B>=token_floor/chunk is unchanged.
        _token_floor = 2048
        if chunk > 0:
            chunk = max(chunk, (_token_floor + B - 1) // max(1, B))
        numer = torch.zeros((), device=features_bth.device, dtype=features_bth.dtype)
        denom = 0
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            logits_slice = _project_token_features(features_bth[:, s:e, :])
            target_slice = targets_bt[:, s:e]
            # Flattened 2D CE (see _chunked_ce_mean): avoids the slow [B, V, T] spatial kernel.
            ce_slice = F.cross_entropy(
                logits_slice.reshape(-1, logits_slice.size(-1)),
                target_slice.reshape(-1),
                reduction="none",
                label_smoothing=ce_label_smoothing,
                ignore_index=(int(ignore_index) if ignore_index is not None else -100),
            ).view_as(target_slice)
            if mask_bt is not None:
                valid = mask_bt[:, s:e].to(device=ce_slice.device, dtype=torch.bool)
            elif ignore_index is not None:
                valid = target_slice != int(ignore_index)
            else:
                valid = torch.ones_like(target_slice, dtype=torch.bool, device=ce_slice.device)
            if not bool(valid.any()):
                continue
            numer = numer + (ce_slice * valid.to(dtype=ce_slice.dtype)).sum()
            denom += int(valid.sum().item())
        if denom <= 0:
            zero = torch.zeros((), device=features_bth.device, dtype=features_bth.dtype)
            return zero, 0
        return numer / float(denom), int(denom)

    _set_token_unet_runtime_flags(emit_lookahead_logits=(lambda_token_unet_lookahead_ce > 0.0))

    diffusion_mask_prob = 1.0
    if effective_objective_mode == "ar":
        masked_ids = input_ids
        target_tokens = torch.full_like(input_ids, -100)
        # AR-aligned reveal target: predict next-token embedding at each position (t -> t+1).
        reveal_target_ids_for_model = input_ids.clone()
        if input_ids.size(1) > 1:
            reveal_target_ids_for_model[:, :-1] = input_ids[:, 1:]
        reveal_mask_for_model = torch.zeros_like(input_ids, dtype=torch.bool)
        if input_ids.size(1) > 1:
            reveal_mask_for_model[:, :-1] = True
            if attention_mask is not None:
                reveal_mask_for_model[:, :-1] = attention_mask[:, 1:].to(dtype=torch.bool)
    else:
        if llada_loss_weighting:
            masked_ids, target_tokens, diffusion_mask_prob = hybrid_diffusion_mask_tokens(
                input_ids,
                tokenizer,
                mode=diffusion_mask_mode,
                grid_shape=diffusion_grid_shape,
                block_size=diffusion_mask_block_size,
                path_length=diffusion_mask_path_length,
                return_mask_prob=True,
            )
        else:
            masked_ids, target_tokens = hybrid_diffusion_mask_tokens(
                input_ids,
                tokenizer,
                mode=diffusion_mask_mode,
                grid_shape=diffusion_grid_shape,
                block_size=diffusion_mask_block_size,
                path_length=diffusion_mask_path_length,
            )
            diffusion_mask_prob = 1.0
        reveal_target_ids_for_model = input_ids
        reveal_mask_for_model = (target_tokens != -100)

    diffusion_mask_loss_weight = 1.0
    if llada_loss_weighting and effective_objective_mode in {"masked", "hybrid"}:
        diffusion_mask_loss_weight = 1.0 / max(float(diffusion_mask_prob), 1e-3)

    if (
        copy_force_mask_dst
        and copy_dst_mask is not None
        and copy_mask_token_id is not None
        and bool(copy_dst_mask.any())
        and (effective_objective_mode != "ar" or copy_mask_dst_in_ar)
    ):
        masked_ids = masked_ids.clone()
        masked_ids[copy_dst_mask] = int(copy_mask_token_id)
        target_tokens = target_tokens.clone()
        target_tokens[copy_dst_mask] = input_ids[copy_dst_mask]
        if effective_objective_mode != "ar":
            reveal_mask_for_model = (target_tokens != -100)
    
    # sample logits for loss to reduce memory
    sample_logits = False
    sampl_size = 512

    # Memory-augmented forward pass helper
    def _forward_with_memory(input_ids_arg, attention_mask_arg, reveal_target_ids_arg, reveal_mask_arg):
        """Standard training forward (memory subsystem removed from the public build)."""
        model.train()
        logits_out = model(
            input_ids_arg,
            attention_mask=attention_mask_arg,
            reveal_target_ids=reveal_target_ids_arg,
            reveal_mask=reveal_mask_arg,
        )
        return logits_out

    # Forward pass with mixed precision if enabled
    # if mixed_precision is true, and scaler OR bf16 is true, use autocast
    if mixed_precision and (scaler is not None or use_bf16):
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16 if use_bf16 else torch.float16):

            # get random int for cycles ranging from 3 to variable_cycles_ceiling but predominantly choose small cycles
            if variable_cycles:
                variable_cycles_ceiling = variable_cycles
                # Generate a float between 0 and 1, skew toward 0 using a power (e.g., 2)
                r = torch.rand(1).item()
                k = 2
                skewed = int(r ** k * (variable_cycles_ceiling - 3)) + 3
                variable_cycles = max(3, min(skewed, variable_cycles_ceiling - 1))
            
            # before training epoch
            model_mod_flags = _model_module()
            prev_return_features = bool(getattr(model_mod_flags, "return_token_features", False))
            setattr(model_mod_flags, "return_token_features", bool(train_feature_chunked_ce_enable or prev_return_features))
            # Forward pass with masked inputs (memory-augmented if enabled)
            logits = _forward_with_memory(
                masked_ids,
                attention_mask,
                reveal_target_ids_for_model,
                reveal_mask_for_model,
            )
            logits_are_features = _is_token_feature_tensor(logits)
            token_features_for_ce = logits if logits_are_features else None
            setattr(model_mod_flags, "return_token_features", prev_return_features)
            if logits_are_features and not train_feature_chunked_ce_enable:
                logits = _project_token_features(logits)
            if autoenc_only_diffusion_mode:
                ae_logits_for_base = getattr(model, "_last_autoenc_logits", None)
                model_mod_local = getattr(model, "module", None)
                if ae_logits_for_base is None and model_mod_local is not None:
                    ae_logits_for_base = getattr(model_mod_local, "_last_autoenc_logits", None)
                if (
                    ae_logits_for_base is not None
                    and hasattr(logits, "shape")
                    and tuple(ae_logits_for_base.shape[:2]) == tuple(logits.shape[:2])
                ):
                    logits = ae_logits_for_base.to(device=logits.device, dtype=logits.dtype)
            #feats = model(masked_ids, attention_mask=attention_mask)  # [B,T,H]
            # print(logits.shape, "logits shape")
            # print(input_ids.shape, "input ids shape")
            # print(target_tokens.shape, "target tokens shape")
            # print(masked_ids.shape, "masked ids shape")
            
            need_ar_loss = effective_objective_mode in {"ar", "hybrid"}
            need_masked_loss = effective_objective_mode in {"masked", "hybrid"}
            need_copy_loss = (
                lambda_copy > 0.0
                and copy_dst_mask is not None
                and bool(copy_dst_mask.any())
            )
            if autoenc_only_mode:
                need_ar_loss = False
                need_masked_loss = False
                need_copy_loss = False

            if need_ar_loss or need_copy_loss:
                shift_labels = input_ids[..., 1:].contiguous()
                if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                    shift_logits = None
                    loss_ar, _ = _chunked_feature_ce_mean(token_features_for_ce[:, :-1, :], shift_labels)
                    token_ce = None
                    shift_logits_flat = None
                    shift_labels_flat = None
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                        loss_ar, _ = _chunked_ce_mean(shift_logits, shift_labels)
                        token_ce = None
                        shift_logits_flat = None
                        shift_labels_flat = None
                    else:
                        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels_flat = shift_labels.view(-1)
                        token_ce = F.cross_entropy(
                            shift_logits_flat,
                            shift_labels_flat,
                            reduction="none",
                            label_smoothing=ce_label_smoothing,
                        )
            else:
                shift_logits = None
                shift_labels = None
                shift_logits_flat = None
                shift_labels_flat = None
                token_ce = None

            if need_ar_loss:
                if token_ce is not None:
                    loss_ar = token_ce.mean()
            else:
                loss_ar = torch.zeros((), device=logits.device, dtype=logits.dtype)

            loss_copy = torch.zeros((), device=logits.device, dtype=logits.dtype)
            copy_token_count = 0
            if need_copy_loss and shift_labels is not None:
                copy_shift_mask = copy_dst_mask[:, 1:]
                if attention_mask is not None:
                    copy_shift_mask = copy_shift_mask & attention_mask[..., 1:].to(dtype=torch.bool)
                if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                    loss_copy, copy_token_count = _chunked_feature_ce_mean(
                        token_features_for_ce[:, :-1, :],
                        shift_labels,
                        mask_bt=copy_shift_mask,
                    )
                elif token_ce is not None:
                    copy_flat = copy_shift_mask.reshape(-1)
                    if bool(copy_flat.any()):
                        copy_token_count = int(copy_flat.sum().item())
                        loss_copy = token_ce[copy_flat].mean()
                elif shift_logits is not None:
                    loss_copy, copy_token_count = _chunked_ce_mean(
                        shift_logits,
                        shift_labels,
                        mask_bt=copy_shift_mask,
                    )

            loss_autoenc = torch.zeros((), device=logits.device, dtype=logits.dtype)
            autoenc_token_count = 0
            loss_autoenc_next = torch.zeros((), device=logits.device, dtype=logits.dtype)
            autoenc_next_token_count = 0
            loss_token_unet_lookahead = torch.zeros((), device=logits.device, dtype=logits.dtype)
            token_unet_lookahead_token_count = 0
            if (lambda_autoenc > 0.0 or lambda_autoenc_next > 0.0) and not autoenc_only_diffusion_mode:
                ae_logits = getattr(model, "_last_autoenc_logits", None)
                model_mod_local = getattr(model, "module", None)
                if ae_logits is None and model_mod_local is not None:
                    ae_logits = getattr(model_mod_local, "_last_autoenc_logits", None)
                if ae_logits is not None:
                    ae_logits = ae_logits.to(device=logits.device, dtype=logits.dtype)
                    if attention_mask is not None:
                        ae_mask = attention_mask.to(device=logits.device, dtype=torch.bool)
                    else:
                        ae_mask = torch.ones_like(input_ids, dtype=torch.bool, device=logits.device)
                    if bool(ae_mask.any()):
                        if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                            loss_autoenc, autoenc_token_count = _chunked_ce_mean(
                                ae_logits,
                                input_ids,
                                mask_bt=ae_mask,
                            )
                        else:
                            ae_token_loss = F.cross_entropy(
                                ae_logits.reshape(-1, ae_logits.size(-1)),
                                input_ids.reshape(-1),
                                reduction="none",
                                label_smoothing=ce_label_smoothing,
                            ).view_as(input_ids)
                            autoenc_token_count = int(ae_mask.sum().item())
                            loss_autoenc = (ae_token_loss * ae_mask.to(ae_token_loss.dtype)).sum() / ae_mask.sum().clamp_min(1)
                    if lambda_autoenc_next > 0.0 and ae_logits.size(1) > 1:
                        ae_next_logits = ae_logits[:, :-1, :]
                        ae_next_targets = input_ids[:, 1:]
                        if attention_mask is not None:
                            ae_next_mask = attention_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
                        else:
                            ae_next_mask = torch.ones_like(ae_next_targets, dtype=torch.bool, device=logits.device)
                        if bool(ae_next_mask.any()):
                            if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                                loss_autoenc_next, autoenc_next_token_count = _chunked_ce_mean(
                                    ae_next_logits,
                                    ae_next_targets,
                                    mask_bt=ae_next_mask,
                                )
                            else:
                                ae_next_loss = F.cross_entropy(
                                    ae_next_logits.reshape(-1, ae_next_logits.size(-1)),
                                    ae_next_targets.reshape(-1),
                                    reduction="none",
                                    label_smoothing=ce_label_smoothing,
                                ).view_as(ae_next_targets)
                                autoenc_next_token_count = int(ae_next_mask.sum().item())
                                loss_autoenc_next = (
                                    ae_next_loss * ae_next_mask.to(ae_next_loss.dtype)
                                ).sum() / ae_next_mask.sum().clamp_min(1)

            if lambda_token_unet_lookahead_ce > 0.0:
                unet_lookahead_logits = getattr(model, "_last_token_unet_lookahead_logits", None)
                model_mod_local = getattr(model, "module", None)
                if unet_lookahead_logits is None and model_mod_local is not None:
                    unet_lookahead_logits = getattr(model_mod_local, "_last_token_unet_lookahead_logits", None)
                if unet_lookahead_logits is not None:
                    unet_lookahead_logits = unet_lookahead_logits.to(device=logits.device, dtype=logits.dtype)
                    if attention_mask is not None:
                        unet_lookahead_mask = attention_mask.to(device=logits.device, dtype=torch.bool)
                    else:
                        unet_lookahead_mask = torch.ones_like(input_ids, dtype=torch.bool, device=logits.device)
                    if bool(unet_lookahead_mask.any()):
                        if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                            loss_token_unet_lookahead, token_unet_lookahead_token_count = _chunked_ce_mean(
                                unet_lookahead_logits,
                                input_ids,
                                mask_bt=unet_lookahead_mask,
                            )
                        else:
                            unet_lookahead_token_loss = F.cross_entropy(
                                unet_lookahead_logits.reshape(-1, unet_lookahead_logits.size(-1)),
                                input_ids.reshape(-1),
                                reduction="none",
                                label_smoothing=ce_label_smoothing,
                            ).view_as(input_ids)
                            token_unet_lookahead_token_count = int(unet_lookahead_mask.sum().item())
                            loss_token_unet_lookahead = (
                                unet_lookahead_token_loss * unet_lookahead_mask.to(unet_lookahead_token_loss.dtype)
                            ).sum() / unet_lookahead_mask.sum().clamp_min(1)

            mask_has = False
            loss_impute = torch.zeros((), device=logits.device, dtype=logits.dtype)
            if need_masked_loss:
                target_tokens_flat = target_tokens.view(-1)
                mask = target_tokens_flat != -100
                mask_has = bool(mask.any())
                if mask_has:
                    logits_for_masked_loss = logits
                    mask_token_id_for_loss = getattr(tokenizer, "mask_token_id", None)
                    if (
                        mask_token_id_for_loss is not None
                        and not (token_features_for_ce is not None and train_feature_chunked_ce_enable)
                        and 0 <= int(mask_token_id_for_loss) < int(logits.size(-1))
                    ):
                        logits_for_masked_loss = logits.clone()
                        logits_for_masked_loss[..., int(mask_token_id_for_loss)] = torch.finfo(logits_for_masked_loss.dtype).min
                    if sample_logits:
                        masked_logits = logits_for_masked_loss.view(-1, logits_for_masked_loss.size(-1))
                        masked_indices = mask.nonzero(as_tuple=True)[0]
                        num_masked = masked_indices.size(0)
                        sample_size = min(sampl_size, num_masked)
                        sampled_indices = masked_indices[torch.randperm(num_masked, device=masked_indices.device)[:sample_size]]
                        loss_impute = F.cross_entropy(
                            masked_logits[sampled_indices],
                            target_tokens_flat[sampled_indices],
                            label_smoothing=ce_label_smoothing,
                        )
                    else:
                        if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                            loss_impute, _ = _chunked_feature_ce_mean(
                                token_features_for_ce,
                                target_tokens,
                                ignore_index=-100,
                            )
                        elif bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                            loss_impute, _ = _chunked_ce_mean(
                                logits_for_masked_loss,
                                target_tokens,
                                ignore_index=-100,
                            )
                        else:
                            loss_impute = F.cross_entropy(
                                logits_for_masked_loss.reshape(-1, logits_for_masked_loss.size(-1)),
                                target_tokens.reshape(-1),
                                ignore_index=-100,
                                reduction="mean",
                                label_smoothing=ce_label_smoothing,
                            )
                    loss_impute = loss_impute * float(diffusion_mask_loss_weight)

            loss_unmasked = torch.zeros((), device=logits.device, dtype=logits.dtype)
            if need_masked_loss and lambda_unmasked > 0.0:
                unmasked_mask = (target_tokens == -100)
                if attention_mask is not None:
                    unmasked_mask = unmasked_mask & attention_mask.to(dtype=torch.bool)
                if bool(unmasked_mask.any()):
                    if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                        loss_unmasked, _ = _chunked_feature_ce_mean(
                            token_features_for_ce,
                            input_ids,
                            mask_bt=unmasked_mask,
                        )
                    elif bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                        loss_unmasked, _ = _chunked_ce_mean(
                            logits,
                            input_ids,
                            mask_bt=unmasked_mask,
                        )
                    else:
                        unmasked_token_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            input_ids.reshape(-1),
                            reduction="none",
                            label_smoothing=ce_label_smoothing,
                        ).view_as(input_ids)
                        loss_unmasked = (
                            unmasked_token_loss * unmasked_mask.to(unmasked_token_loss.dtype)
                        ).sum() / unmasked_mask.sum().clamp_min(1)

            #loss = (0.8995 * loss_impute) + (0.0995 * loss_next) + (0.001 * loss_unmasked)
            #loss = (0.9 * loss_impute) + (0.1 * loss_next) + (0.001 * loss_unmasked)
            #loss = (0.4975 * loss_impute) + (0.4975 * loss_next) + (0.05 * loss_unmasked)
            #loss = (0.9 * loss_impute) + (0.09 * loss_next) + (0.01 * loss_unmasked)
            #loss = (0.5 * loss_impute) + (0.49 * loss_next) + (0.01 * loss_unmasked)
            #loss = (0.99 * loss_impute) + (0.01 * loss_next) #+ (0.01 * loss_unmasked)
            #loss = (0.9 * loss_impute) + (0.1 * loss_next)
            ####loss = (0.8 * loss_impute) + (0.18 * loss_suffix) + (0.01 * loss_unmasked) + (0.01 * loss_next)
            #loss = (0.8* loss_impute.mean() ) + ( 0.1 * loss_suffix.mean() ) + ( 0.1 * loss_next.mean() )
            has_autoenc_self = (lambda_autoenc > 0.0 and autoenc_token_count > 0)
            has_autoenc_next = (lambda_autoenc_next > 0.0 and autoenc_next_token_count > 0)
            if autoenc_only_mode and not (has_autoenc_self or has_autoenc_next):
                _set_train_loss_stats(
                    base_ce_loss_value=None,
                    copy_ce_loss_value=None,
                    copy_tokens_value=0,
                    autoenc_ce_loss_value=float(loss_autoenc.detach().item()) if autoenc_token_count > 0 else None,
                    autoenc_next_ce_loss_value=float(loss_autoenc_next.detach().item()) if autoenc_next_token_count > 0 else None,
                    objective_loss_value=None,
                )
                return None

            if autoenc_only_mode:
                base_ce_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
                base_ce_loss_report = None
            elif effective_objective_mode == "ar":
                base_ce_loss = lambda_base_ce * loss_ar
                base_ce_loss_report = loss_ar
            elif effective_objective_mode == "hybrid":
                base_ce_loss = (lambda_masked * loss_impute.mean()) + (lambda_ar * loss_ar) + (lambda_unmasked * loss_unmasked)
                base_ce_loss_report = base_ce_loss
            else:
                if not mask_has:
                    _set_train_loss_stats(base_ce_loss_value=None, copy_ce_loss_value=None, copy_tokens_value=0, objective_loss_value=None)
                    return None
                base_ce_loss = loss_impute.mean() + (lambda_unmasked * loss_unmasked)
                base_ce_loss_report = base_ce_loss
            loss = base_ce_loss
            if (not autoenc_only_mode) and need_copy_loss and copy_token_count > 0:
                loss = loss + (lambda_copy * loss_copy)
            if has_autoenc_self and not autoenc_only_diffusion_mode:
                loss = loss + (lambda_autoenc * loss_autoenc)
            if has_autoenc_next and not autoenc_only_diffusion_mode:
                loss = loss + (lambda_autoenc_next * loss_autoenc_next)
            if token_unet_lookahead_token_count > 0:
                loss = loss + (lambda_token_unet_lookahead_ce * loss_token_unet_lookahead)
            #loss = (0.9 * loss_impute)# + (0.01 * loss_unmasked) + (0.09 * loss_next)
            _set_train_loss_stats(
                base_ce_loss_value=None if base_ce_loss_report is None else float(base_ce_loss_report.detach().item()),
                copy_ce_loss_value=float(loss_copy.detach().item()) if copy_token_count > 0 else None,
                copy_tokens_value=copy_token_count,
                autoenc_ce_loss_value=(None if autoenc_only_diffusion_mode else (float(loss_autoenc.detach().item()) if autoenc_token_count > 0 else None)),
                autoenc_next_ce_loss_value=(None if autoenc_only_diffusion_mode else (float(loss_autoenc_next.detach().item()) if autoenc_next_token_count > 0 else None)),
                token_unet_lookahead_ce_loss_value=(float(loss_token_unet_lookahead.detach().item()) if token_unet_lookahead_token_count > 0 else None),
                objective_loss_value=float(loss.detach().item()),
            )

            aux = getattr(model, "_last_hier_aux_loss", None)

            if aux is not None:
                loss = loss + model.lambda_hier_aux * aux.mean()

            predaux = getattr(model, "_last_hier_predaux_loss", None)
            if predaux is not None:
                loss = loss + model.lambda_hier_predaux * predaux.mean()

            # torch.cuda.reset_peak_memory_stats(device)
            # torch.cuda.synchronize(device)
            # torch.cuda.empty_cache()
            # alloc = torch.cuda.memory_allocated(device) / 1e9
            # peak  = torch.cuda.max_memory_allocated(device) / 1e9
            # print(f"[MEM] just got logits: alloc={alloc:.2f} GB, peak={peak:.2f} GB")
            #loss = loss_impute
            # ------------------------------------------------------------------
            # LOGITS READY ABOVE:
            #   logits: [B, T, V]
            #   input_ids: [B, T]
            #   target_tokens: [B, T] with -100 for non-supervised positions
            # ------------------------------------------------------------------
            #   feats: [B, T, H]  (from model with return_token_features=True)
            #   input_ids: [B, T]
            #   target_tokens: [B, T] with -100 where we don’t supervise
        #     proj_w, proj_b = model.output_projection.weight, model.output_projection.bias
        #     B, T, H = feats.shape
        #     #print(feats.shape,"feats shape")
        #     V = proj_w.size(0)

        #     # ---- knobs (tune per GPU) ----
        #     MAX_MASKED_PER_STEP = 4096   # global cap on supervised masked tokens (0 to disable)
        #     PER_SEQ_CAP         = 512    # per-sequence cap (0 to disable)
        #     SUFFIX              = 64     # keep suffix short
        #     SUFFIX_STEP         = 16     # compute CE over tiny time chunks
        #     W_IMPUTE            = 0.90
        #     W_SUFFIX            = 0.10
        #     IGNORE_INDEX        = -100
        #     # ------------------------------

        #     # 1) Masked imputation — supervise only a capped subset
        #     mask = (target_tokens != IGNORE_INDEX)                 # [B,T]
        #     if mask.any():
        #         b_all, t_all = mask.nonzero(as_tuple=True)

        #         # per-seq cap (spreads supervision across batch)
        #         if PER_SEQ_CAP > 0:
        #             keep_chunks = []
        #             # NOTE: vectorizing this is possible; this is simple and fast enough.
        #             for b in range(B):
        #                 idx_b = (b_all == b).nonzero(as_tuple=True)[0]
        #                 if idx_b.numel() == 0:
        #                     continue
        #                 if idx_b.numel() > PER_SEQ_CAP:
        #                     sel = torch.randint(0, idx_b.numel(), (PER_SEQ_CAP,), device=idx_b.device)
        #                     idx_b = idx_b[sel]
        #                 keep_chunks.append(idx_b)
        #             if keep_chunks:
        #                 keep_idx = torch.cat(keep_chunks, dim=0)
        #                 b_all = b_all[keep_idx]
        #                 t_all = t_all[keep_idx]

        #         # global cap
        #         if MAX_MASKED_PER_STEP > 0 and b_all.numel() > MAX_MASKED_PER_STEP:
        #             sel = torch.randint(0, b_all.numel(), (MAX_MASKED_PER_STEP,), device=b_all.device)
        #             b_all = b_all[sel]
        #             t_all = t_all[sel]

        #         # project on demand: [K,H] -> [K,V]
        #         feat_masked = feats[b_all, t_all, :]                      # [K,H]
        #         if feat_masked.dtype != proj_w.dtype:
        #             feat_masked = feat_masked.to(proj_w.dtype)                # cast activations to weight dtype (usually float32)
        #         logits_masked = torch.nn.functional.linear(feat_masked, proj_w, proj_b)  # [K,V]
        #         targets_masked = target_tokens[b_all, t_all]              # [K]

        #         loss_impute = torch.nn.functional.cross_entropy(
        #             logits_masked, targets_masked,
        #             reduction="mean", ignore_index=IGNORE_INDEX
        #         )

        #     else:
        #         loss_impute = feats.new_zeros(())

                

        #     # 2) Suffix next-token — tiny windows only
        #     Ls = min(SUFFIX, T)
        #     loss_suffix_sum = None
        #     count = 0
        #     if Ls > 0:
        #         # select suffix features: [B, Ls, H] -> process in time chunks
        #         for t0 in range(T - Ls, T, SUFFIX_STEP):
        #             t1 = min(t0 + SUFFIX_STEP, T)
        #             f_chunk = feats[:, t0:t1, :].reshape(-1, H)                 # [B*step, H]
        #             y_chunk = input_ids[:, t0:t1].reshape(-1)                   # [B*step]
        #             # project: [B*step, H] -> [B*step, V]
        #             if f_chunk.dtype != proj_w.dtype:
        #                 f_chunk = f_chunk.to(proj_w.dtype)
        #             l_chunk = torch.nn.functional.linear(f_chunk, proj_w, proj_b)
        #             ce = torch.nn.functional.cross_entropy(
        #                 l_chunk, y_chunk, reduction="mean", ignore_index=IGNORE_INDEX
        #             )
        #             loss_suffix_sum = ce if loss_suffix_sum is None else (loss_suffix_sum + ce)
        #             count += (t1 - t0) * B

        #     loss_suffix = (loss_suffix_sum / max(1, count)) if loss_suffix_sum is not None else feats.new_zeros(())

        #     # 3) Combine
        #     #loss = W_IMPUTE * loss_impute + W_SUFFIX * loss_suffix
        #     # (Optional) free suffix slices early
        #     #del suffix_logits, suffix_labels

        #     # -------------------------
        #     # 3) Combine and backprop
        #     # -------------------------
        #     loss = W_IMPUTE * loss_impute + W_SUFFIX * loss_suffix
        #     # torch.cuda.reset_peak_memory_stats(device)
        #     # torch.cuda.synchronize(device)
        #     # torch.cuda.empty_cache()
        #     # alloc = torch.cuda.memory_allocated(device) / 1e9
        #     # peak  = torch.cuda.max_memory_allocated(device) / 1e9
        #     # print(f"[MEM] pre scale: alloc={alloc:.2f} GB, peak={peak:.2f} GB")
        #     # If you do grad accumulation / AMP outside, keep as-is:
        #     # scaled_loss = loss / gradient_accumulation_steps
        #     # scaler.scale(scaled_loss).backward()
        # # Scale loss for gradient accumulation
        # scaled_loss = loss / gradient_accumulation_steps
        # # torch.cuda.reset_peak_memory_stats(device)
        # # torch.cuda.synchronize(device)
        # # torch.cuda.empty_cache()
        # # alloc = torch.cuda.memory_allocated(device) / 1e9
        # # peak  = torch.cuda.max_memory_allocated(device) / 1e9
        # # print(f"[MEM] post scale: alloc={alloc:.2f} GB, peak={peak:.2f} GB")
        # # Backward pass with scaling
        # scaler.scale(scaled_loss).backward()
        # # torch.cuda.reset_peak_memory_stats(device)
        # # torch.cuda.synchronize(device)
        # # torch.cuda.empty_cache()
        # # alloc = torch.cuda.memory_allocated(device) / 1e9
        # # peak  = torch.cuda.max_memory_allocated(device) / 1e9
        # # print(f"[MEM] post backward: alloc={alloc:.2f} GB, peak={peak:.2f} GB")
        # #import torch
        # #import torch.nn.functional as F

        # def diffusion_masked_loss(
        #     feats: torch.Tensor,           # [B,T,H]
        #     target_tokens: torch.Tensor,   # [B,T] with IGNORE_INDEX at non-supervised
        #     proj_w: torch.Tensor,          # [V,H]
        #     proj_b: torch.Tensor,          # [V] or None
        #     *,
        #     per_seq_cap: int = 0,        # max supervised positions per sequence (0 = no cap)
        #     global_cap: int = 4096,        # max supervised positions across batch (0 = no cap)
        #     shard_size: int = 4096,        # project in shards to bound peak mem
        #     label_smoothing: float = 0.1,  # 0.0 to disable
        #     ignore_index: int = -100,
        # ) -> torch.Tensor:
        #     """
        #     Diffusion-style objective: supervise ONLY masked positions (no AR shift).
        #     Keeps torch.rand/torch.randint sampling; normalizes by total supervised K.
        #     """
        #     B, T, H = feats.shape
        #     device = feats.device

        #     # 1) all supervised coordinates
        #     mask = (target_tokens != ignore_index)            # [B,T]
        #     if not mask.any():
        #         return feats.new_zeros(())                    # scalar 0 (no grads)

        #     b_all, t_all = mask.nonzero(as_tuple=True)        # K raw

        #     # 2) per-sequence cap (keeps torch.randint randomness)
        #     if per_seq_cap and per_seq_cap > 0:
        #         keep = []
        #         # NOTE: this loop is cheap (B is small). Keeps full randomness per sequence.
        #         for b in range(B):
        #             idx = (b_all == b).nonzero(as_tuple=True)[0]
        #             if idx.numel() == 0:
        #                 continue
        #             if idx.numel() > per_seq_cap:
        #                 sel = torch.randint(0, idx.numel(), (per_seq_cap,), device=device)
        #                 idx = idx[sel]
        #             keep.append(idx)
        #         if keep:
        #             keep = torch.cat(keep, dim=0)
        #             b_all = b_all[keep]
        #             t_all = t_all[keep]

        #     # 3) global cap (keeps torch.randint randomness)
        #     if global_cap and b_all.numel() > global_cap:
        #         sel = torch.randint(0, b_all.numel(), (global_cap,), device=device)
        #         b_all = b_all[sel]
        #         t_all = t_all[sel]

        #     K = b_all.numel()
        #     if K == 0:
        #         return feats.new_zeros(())

        #     # 4) shard to keep peak memory flat; SUM per shard, then divide by total K
        #     loss_sum = feats.new_zeros(())
        #     n_shards = (K + shard_size - 1) // shard_size

        #     for s in range(n_shards):
        #         i0 = s * shard_size
        #         i1 = min((s + 1) * shard_size, K)

        #         bs = b_all[i0:i1]
        #         ts = t_all[i0:i1]

        #         f = feats[bs, ts, :]                    # [k,H]
        #         y = target_tokens[bs, ts]               # [k]
        #         if f.dtype != proj_w.dtype:
        #             f = f.to(proj_w.dtype)                # cast activations to weight dtype (usually float32)
        #         # project on demand: [k,H] -> [k,V]
        #         logits = F.linear(f, proj_w, proj_b)

        #         # SUM here; normalize once by K after loop (scale-stable)
        #         ce_sum = F.cross_entropy(
        #             logits, y,
        #             reduction="sum",
        #             ignore_index=ignore_index,
        #             label_smoothing=label_smoothing
        #         )
        #         loss_sum = loss_sum + ce_sum

        #     loss = loss_sum / float(K)                  # final normalization
        #     return loss
        
        # # # feats: [B,T,H] from the model (your “features mode” in train)
        # # proj_w, proj_b = model.output_projection.weight, model.output_projection.bias

        # # loss = diffusion_masked_loss(
        # #     feats, target_tokens,
        # #     proj_w, proj_b,
        # #     per_seq_cap=512,        # tune
        # #     global_cap=4096,        # tune
        # #     shard_size=4096,        # tune to clamp peak mem
        # #     label_smoothing=0.0,    # try 0.05–0.1 if training is jittery
        # #     ignore_index=-100,
        # # )

        # # scaled = loss / gradient_accumulation_steps
        # # if scaler is not None:
        # #     scaler.scale(scaled).backward()
        # # else:
        # #     scaled.backward()

        # ---------------------------------------
        # Diffusion-style masked training + suffix-masked head
        # ---------------------------------------
        # proj_w, proj_b = model.output_projection.weight, model.output_projection.bias
        # B, T, H = feats.shape
        # IGNORE_INDEX = -100

        # # knobs (tune as you like)
        # PER_SEQ_CAP_MAIN   = 0#1024     # max supervised positions per sequence for main masked loss
        # GLOBAL_CAP_MAIN    = 0#1024    # global cap across batch for main masked loss
        # PER_SEQ_CAP_SUFFIX = 0#256     # (usually smaller) cap for suffix-only masked loss
        # GLOBAL_CAP_SUFFIX  = 0#2048
        # SHARD_SIZE         = 131072#4096    # clamp peak memory during projection
        # W_IMPUTE           = 1#0.90
        # W_SUFFIX           = 0#.10
        # SUFFIX             = 64      # suffix window length

        # device = feats.device
        # is_feats = False
        # # -------- 1) Main diffusion masked loss (all masked positions) --------
        # mask_all = (target_tokens != IGNORE_INDEX)                      # [B,T]
        # if mask_all.any():
        #     b_all, t_all = mask_all.nonzero(as_tuple=True)              # K positions

        #     # per-seq cap
        #     if PER_SEQ_CAP_MAIN and PER_SEQ_CAP_MAIN > 0:
        #         keep_chunks = []
        #         for b in range(B):
        #             idx_b = (b_all == b).nonzero(as_tuple=True)[0]
        #             if idx_b.numel() == 0:
        #                 continue
        #             if idx_b.numel() > PER_SEQ_CAP_MAIN:
        #                 sel = torch.randint(0, idx_b.numel(), (PER_SEQ_CAP_MAIN,), device=device)
        #                 idx_b = idx_b[sel]
        #             keep_chunks.append(idx_b)
        #         if keep_chunks:
        #             keep_idx = torch.cat(keep_chunks, dim=0)
        #             b_all = b_all[keep_idx]
        #             t_all = t_all[keep_idx]

        #     # global cap
        #     if GLOBAL_CAP_MAIN and b_all.numel() > GLOBAL_CAP_MAIN:
        #         sel = torch.randint(0, b_all.numel(), (GLOBAL_CAP_MAIN,), device=device)
        #         b_all = b_all[sel]
        #         t_all = t_all[sel]

        #     K = b_all.numel()
        #     loss_impute_sum = feats.new_zeros(())
        #     n_shards = (K + SHARD_SIZE - 1) // SHARD_SIZE if K > 0 else 0

        #     for s in range(n_shards):
        #         i0 = s * SHARD_SIZE
        #         i1 = min((s + 1) * SHARD_SIZE, K)
        #         bs = b_all[i0:i1]
        #         ts = t_all[i0:i1]

        #         f = feats[bs, ts, :]                         # [k,H]
        #         if f.dtype != proj_w.dtype:                  # BF16/FP32 harmonization
        #             f = f.to(proj_w.dtype)
        #         y = target_tokens[bs, ts] 
        #         if is_feats:                   # [k]
        #             logits = torch.nn.functional.linear(f, proj_w, proj_b)  # [k,V]
        #         else:
        #             logits = f
        #         ce = torch.nn.functional.cross_entropy(
        #             logits, y, reduction="mean", ignore_index=IGNORE_INDEX
        #         )
        #         loss_impute_sum = loss_impute_sum + ce

        #     loss_impute = (loss_impute_sum / max(1, n_shards)) if K > 0 else feats.new_zeros(())
        # else:
        #     loss_impute = feats.new_zeros(())

        # # -------- 2) Suffix masked loss (ONLY masked tokens in last SUFFIX window) --------
        # Ls = min(SUFFIX, T)
        # if Ls > 0:
        #     suffix_start = T - Ls
        #     suffix_mask = (target_tokens != IGNORE_INDEX).clone()       # [B,T]
        #     suffix_mask[:, :suffix_start] = False                       # keep only last Ls
        #     if suffix_mask.any():
        #         b_suf, t_suf = suffix_mask.nonzero(as_tuple=True)

        #         # per-seq cap for suffix
        #         if PER_SEQ_CAP_SUFFIX and PER_SEQ_CAP_SUFFIX > 0:
        #             keep_chunks = []
        #             for b in range(B):
        #                 idx_b = (b_suf == b).nonzero(as_tuple=True)[0]
        #                 if idx_b.numel() == 0:
        #                     continue
        #                 if idx_b.numel() > PER_SEQ_CAP_SUFFIX:
        #                     sel = torch.randint(0, idx_b.numel(), (PER_SEQ_CAP_SUFFIX,), device=device)
        #                     idx_b = idx_b[sel]
        #                 keep_chunks.append(idx_b)
        #             if keep_chunks:
        #                 keep_idx = torch.cat(keep_chunks, dim=0)
        #                 b_suf = b_suf[keep_idx]
        #                 t_suf = t_suf[keep_idx]

        #         # global cap for suffix
        #         if GLOBAL_CAP_SUFFIX and b_suf.numel() > GLOBAL_CAP_SUFFIX:
        #             sel = torch.randint(0, b_suf.numel(), (GLOBAL_CAP_SUFFIX,), device=device)
        #             b_suf = b_suf[sel]
        #             t_suf = t_suf[sel]

        #         K2 = b_suf.numel()
        #         loss_suffix_sum = feats.new_zeros(())
        #         n_shards2 = (K2 + SHARD_SIZE - 1) // SHARD_SIZE if K2 > 0 else 0

        #         for s in range(n_shards2):
        #             i0 = s * SHARD_SIZE
        #             i1 = min((s + 1) * SHARD_SIZE, K2)
        #             bs = b_suf[i0:i1]
        #             ts = t_suf[i0:i1]

        #             f = feats[bs, ts, :]                     # [k,H]
        #             if f.dtype != proj_w.dtype:              # BF16/FP32 harmonization
        #                 f = f.to(proj_w.dtype)
        #             y = target_tokens[bs, ts]                # [k]
        #             if is_feats:                   # [k]
        #                 logits = torch.nn.functional.linear(f, proj_w, proj_b)
        #             else:
        #                 logits = f
        #             ce = torch.nn.functional.cross_entropy(
        #                 logits, y, reduction="mean", ignore_index=IGNORE_INDEX
        #             )
        #             loss_suffix_sum = loss_suffix_sum + ce

        #         loss_suffix = (loss_suffix_sum / max(1, n_shards2)) if K2 > 0 else feats.new_zeros(())
        #     else:
        #         loss_suffix = feats.new_zeros(())
        # else:
        #     loss_suffix = feats.new_zeros(())

        # # -------- 3) Combine & backprop (with your grad accumulation / AMP outside) --------
        # loss = W_IMPUTE * loss_impute + W_SUFFIX * loss_suffix

        _set_objective_loss_stat(float(loss.detach().item()))
        scaled_loss = loss / gradient_accumulation_steps

        #if not scaled_loss.requires_grad:
        # Skip this micro-batch cleanly (or tie a zero to graph if you prefer)
        # Option A: skip
        #    return None  # caller should ignore/continue this micro-batch
        # Option B: tie zero to graph (if your caller expects a Tensor back)
        # scaled_loss = sum(p.sum() for p in model.parameters()) * 0.0
        _bwd_prof = bool(os.environ.get("MEM_PROFILE"))
        if _bwd_prof and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_bwd = time.perf_counter()
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if _bwd_prof:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[MEM_PROFILE] backward=%.1fms", (time.perf_counter() - _t_bwd) * 1000.0)
    else:
        # Forward pass with masked inputs (memory-augmented if enabled)
        model_mod_flags = _model_module()
        prev_return_features = bool(getattr(model_mod_flags, "return_token_features", False))
        setattr(model_mod_flags, "return_token_features", bool(train_feature_chunked_ce_enable or prev_return_features))
        logits = _forward_with_memory(
            masked_ids,
            attention_mask,
            reveal_target_ids_for_model,
            reveal_mask_for_model,
        )
        logits_are_features = _is_token_feature_tensor(logits)
        token_features_for_ce = logits if logits_are_features else None
        setattr(model_mod_flags, "return_token_features", prev_return_features)
        if logits_are_features and not train_feature_chunked_ce_enable:
            logits = _project_token_features(logits)
        if autoenc_only_diffusion_mode:
            ae_logits_for_base = getattr(model, "_last_autoenc_logits", None)
            model_mod_local = getattr(model, "module", None)
            if ae_logits_for_base is None and model_mod_local is not None:
                ae_logits_for_base = getattr(model_mod_local, "_last_autoenc_logits", None)
            if (
                ae_logits_for_base is not None
                and hasattr(logits, "shape")
                and tuple(ae_logits_for_base.shape[:2]) == tuple(logits.shape[:2])
            ):
                logits = ae_logits_for_base.to(device=logits.device, dtype=logits.dtype)

        need_ar_loss = effective_objective_mode in {"ar", "hybrid"}
        need_masked_loss = effective_objective_mode in {"masked", "hybrid"}
        need_copy_loss = (
            lambda_copy > 0.0
            and copy_dst_mask is not None
            and bool(copy_dst_mask.any())
        )
        if autoenc_only_mode:
            need_ar_loss = False
            need_masked_loss = False
            need_copy_loss = False

        if need_ar_loss or need_copy_loss:
            shift_labels = input_ids[..., 1:].contiguous()
            if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                shift_logits = None
                loss_ar, _ = _chunked_feature_ce_mean(token_features_for_ce[:, :-1, :], shift_labels)
                token_ce = None
                shift_logits_flat = None
                shift_labels_flat = None
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                    loss_ar, _ = _chunked_ce_mean(shift_logits, shift_labels)
                    token_ce = None
                    shift_logits_flat = None
                    shift_labels_flat = None
                else:
                    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels_flat = shift_labels.view(-1)
                    token_ce = F.cross_entropy(
                        shift_logits_flat,
                        shift_labels_flat,
                        reduction="none",
                        label_smoothing=ce_label_smoothing,
                    )
        else:
            shift_logits = None
            shift_labels = None
            shift_logits_flat = None
            shift_labels_flat = None
            token_ce = None

        if need_ar_loss:
            if token_ce is not None:
                loss_ar = token_ce.mean()
        else:
            loss_ar = torch.zeros((), device=logits.device, dtype=logits.dtype)

        loss_copy = torch.zeros((), device=logits.device, dtype=logits.dtype)
        copy_token_count = 0
        if need_copy_loss and shift_labels is not None:
            copy_shift_mask = copy_dst_mask[:, 1:]
            if attention_mask is not None:
                copy_shift_mask = copy_shift_mask & attention_mask[..., 1:].to(dtype=torch.bool)
            if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                loss_copy, copy_token_count = _chunked_feature_ce_mean(
                    token_features_for_ce[:, :-1, :],
                    shift_labels,
                    mask_bt=copy_shift_mask,
                )
            elif token_ce is not None:
                copy_flat = copy_shift_mask.reshape(-1)
                if bool(copy_flat.any()):
                    copy_token_count = int(copy_flat.sum().item())
                    loss_copy = token_ce[copy_flat].mean()
            elif shift_logits is not None:
                loss_copy, copy_token_count = _chunked_ce_mean(
                    shift_logits,
                    shift_labels,
                    mask_bt=copy_shift_mask,
                )

        loss_autoenc = torch.zeros((), device=logits.device, dtype=logits.dtype)
        autoenc_token_count = 0
        loss_autoenc_next = torch.zeros((), device=logits.device, dtype=logits.dtype)
        autoenc_next_token_count = 0
        loss_token_unet_lookahead = torch.zeros((), device=logits.device, dtype=logits.dtype)
        token_unet_lookahead_token_count = 0
        if (lambda_autoenc > 0.0 or lambda_autoenc_next > 0.0) and not autoenc_only_diffusion_mode:
            ae_logits = getattr(model, "_last_autoenc_logits", None)
            model_mod_local = getattr(model, "module", None)
            if ae_logits is None and model_mod_local is not None:
                ae_logits = getattr(model_mod_local, "_last_autoenc_logits", None)
            if ae_logits is not None:
                ae_logits = ae_logits.to(device=logits.device, dtype=logits.dtype)
                if attention_mask is not None:
                    ae_mask = attention_mask.to(device=logits.device, dtype=torch.bool)
                else:
                    ae_mask = torch.ones_like(input_ids, dtype=torch.bool, device=logits.device)
                if bool(ae_mask.any()):
                    if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                        loss_autoenc, autoenc_token_count = _chunked_ce_mean(
                            ae_logits,
                            input_ids,
                            mask_bt=ae_mask,
                        )
                    else:
                        ae_token_loss = F.cross_entropy(
                            ae_logits.reshape(-1, ae_logits.size(-1)),
                            input_ids.reshape(-1),
                            reduction="none",
                            label_smoothing=ce_label_smoothing,
                        ).view_as(input_ids)
                        autoenc_token_count = int(ae_mask.sum().item())
                        loss_autoenc = (ae_token_loss * ae_mask.to(ae_token_loss.dtype)).sum() / ae_mask.sum().clamp_min(1)
                if lambda_autoenc_next > 0.0 and ae_logits.size(1) > 1:
                    ae_next_logits = ae_logits[:, :-1, :]
                    ae_next_targets = input_ids[:, 1:]
                    if attention_mask is not None:
                        ae_next_mask = attention_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
                    else:
                        ae_next_mask = torch.ones_like(ae_next_targets, dtype=torch.bool, device=logits.device)
                    if bool(ae_next_mask.any()):
                        if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                            loss_autoenc_next, autoenc_next_token_count = _chunked_ce_mean(
                                ae_next_logits,
                                ae_next_targets,
                                mask_bt=ae_next_mask,
                            )
                        else:
                            ae_next_loss = F.cross_entropy(
                                ae_next_logits.reshape(-1, ae_next_logits.size(-1)),
                                ae_next_targets.reshape(-1),
                                reduction="none",
                                label_smoothing=ce_label_smoothing,
                            ).view_as(ae_next_targets)
                            autoenc_next_token_count = int(ae_next_mask.sum().item())
                            loss_autoenc_next = (
                                ae_next_loss * ae_next_mask.to(ae_next_loss.dtype)
                            ).sum() / ae_next_mask.sum().clamp_min(1)

        if lambda_token_unet_lookahead_ce > 0.0:
            unet_lookahead_logits = getattr(model, "_last_token_unet_lookahead_logits", None)
            model_mod_local = getattr(model, "module", None)
            if unet_lookahead_logits is None and model_mod_local is not None:
                unet_lookahead_logits = getattr(model_mod_local, "_last_token_unet_lookahead_logits", None)
            if unet_lookahead_logits is not None:
                unet_lookahead_logits = unet_lookahead_logits.to(device=logits.device, dtype=logits.dtype)
                if attention_mask is not None:
                    unet_lookahead_mask = attention_mask.to(device=logits.device, dtype=torch.bool)
                else:
                    unet_lookahead_mask = torch.ones_like(input_ids, dtype=torch.bool, device=logits.device)
                if bool(unet_lookahead_mask.any()):
                    if bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                        loss_token_unet_lookahead, token_unet_lookahead_token_count = _chunked_ce_mean(
                            unet_lookahead_logits,
                            input_ids,
                            mask_bt=unet_lookahead_mask,
                        )
                    else:
                        unet_lookahead_token_loss = F.cross_entropy(
                            unet_lookahead_logits.reshape(-1, unet_lookahead_logits.size(-1)),
                            input_ids.reshape(-1),
                            reduction="none",
                            label_smoothing=ce_label_smoothing,
                        ).view_as(input_ids)
                        token_unet_lookahead_token_count = int(unet_lookahead_mask.sum().item())
                        loss_token_unet_lookahead = (
                            unet_lookahead_token_loss * unet_lookahead_mask.to(unet_lookahead_token_loss.dtype)
                        ).sum() / unet_lookahead_mask.sum().clamp_min(1)

        mask_has = False
        loss_impute = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if need_masked_loss:
            target_tokens_flat = target_tokens.view(-1)
            mask = target_tokens_flat != -100
            mask_has = bool(mask.any())
            if mask_has:
                logits_for_masked_loss = logits
                mask_token_id_for_loss = getattr(tokenizer, "mask_token_id", None)
                if (
                    mask_token_id_for_loss is not None
                    and not (token_features_for_ce is not None and train_feature_chunked_ce_enable)
                    and 0 <= int(mask_token_id_for_loss) < int(logits.size(-1))
                ):
                    logits_for_masked_loss = logits.clone()
                    logits_for_masked_loss[..., int(mask_token_id_for_loss)] = torch.finfo(logits_for_masked_loss.dtype).min
                if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                    loss_impute, _ = _chunked_feature_ce_mean(
                        token_features_for_ce,
                        target_tokens,
                        ignore_index=-100,
                    )
                elif bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                    loss_impute, _ = _chunked_ce_mean(
                        logits_for_masked_loss,
                        target_tokens,
                        ignore_index=-100,
                    )
                else:
                    masked_logits = logits_for_masked_loss.view(-1, logits_for_masked_loss.size(-1))
                    loss_impute = F.cross_entropy(
                        masked_logits[mask],
                        target_tokens_flat[mask],
                        label_smoothing=ce_label_smoothing,
                    )
                loss_impute = loss_impute * float(diffusion_mask_loss_weight)

        loss_unmasked = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if need_masked_loss and lambda_unmasked > 0.0:
            unmasked_mask = (target_tokens == -100)
            if attention_mask is not None:
                unmasked_mask = unmasked_mask & attention_mask.to(dtype=torch.bool)
            if bool(unmasked_mask.any()):
                if token_features_for_ce is not None and train_feature_chunked_ce_enable:
                    loss_unmasked, _ = _chunked_feature_ce_mean(
                        token_features_for_ce,
                        input_ids,
                        mask_bt=unmasked_mask,
                    )
                elif bool(chunked_ce_enable and chunked_ce_seq_chunk > 0):
                    loss_unmasked, _ = _chunked_ce_mean(
                        logits,
                        input_ids,
                        mask_bt=unmasked_mask,
                    )
                else:
                    unmasked_token_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        input_ids.reshape(-1),
                        reduction="none",
                        label_smoothing=ce_label_smoothing,
                    ).view_as(input_ids)
                    loss_unmasked = (
                        unmasked_token_loss * unmasked_mask.to(unmasked_token_loss.dtype)
                    ).sum() / unmasked_mask.sum().clamp_min(1)
        
        # Unmasked token consistency loss
        #inverse_mask = (target_tokens_flat != -100) & (~mask)  # <- Only valid & unmasked
        #if inverse_mask.any():
        #    loss_unmasked = criterion(masked_logits[inverse_mask], target_tokens_flat[inverse_mask])
        #else:
        #    loss_unmasked = torch.tensor(0.0, device=logits.device, requires_grad=True)

        #loss = (0.8995 * loss_impute) + (0.0995 * loss_next) + (0.001 * loss_unmasked)
        #loss = (0.9 * loss_impute) + (0.1 * loss_next) + (0.001 * loss_unmasked)
        #loss = (0.4975 * loss_impute) + (0.4975 * loss_next) + (0.05 * loss_unmasked)
        #loss = (0.9 * loss_impute) + (0.09 * loss_next) + (0.01 * loss_unmasked)
        #loss = (0.5 * loss_impute) + (0.49 * loss_next) + (0.01 * loss_unmasked)
        #loss = (0.99 * loss_impute) + (0.01 * loss_next) #+ (0.01 * loss_unmasked)
        #loss = (0.9 * loss_impute) + (0.1 * loss_next)
        has_autoenc_self = (lambda_autoenc > 0.0 and autoenc_token_count > 0)
        has_autoenc_next = (lambda_autoenc_next > 0.0 and autoenc_next_token_count > 0)
        if autoenc_only_mode and not (has_autoenc_self or has_autoenc_next):
            _set_train_loss_stats(
                base_ce_loss_value=None,
                copy_ce_loss_value=None,
                copy_tokens_value=0,
                autoenc_ce_loss_value=float(loss_autoenc.detach().item()) if autoenc_token_count > 0 else None,
                autoenc_next_ce_loss_value=float(loss_autoenc_next.detach().item()) if autoenc_next_token_count > 0 else None,
                objective_loss_value=None,
            )
            return None

        if autoenc_only_mode:
            base_ce_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
            base_ce_loss_report = None
        elif effective_objective_mode == "ar":
            base_ce_loss = lambda_base_ce * loss_ar
            base_ce_loss_report = loss_ar
        elif effective_objective_mode == "hybrid":
            base_ce_loss = (lambda_masked * loss_impute.mean()) + (lambda_ar * loss_ar) + (lambda_unmasked * loss_unmasked)
            base_ce_loss_report = base_ce_loss
        else:
            if not mask_has:
                _set_train_loss_stats(base_ce_loss_value=None, copy_ce_loss_value=None, copy_tokens_value=0, objective_loss_value=None)
                return None
            base_ce_loss = loss_impute.mean() + (lambda_unmasked * loss_unmasked)
            base_ce_loss_report = base_ce_loss
        loss = base_ce_loss
        if (not autoenc_only_mode) and need_copy_loss and copy_token_count > 0:
            loss = loss + (lambda_copy * loss_copy)
        if has_autoenc_self and not autoenc_only_diffusion_mode:
            loss = loss + (lambda_autoenc * loss_autoenc)
        if has_autoenc_next and not autoenc_only_diffusion_mode:
            loss = loss + (lambda_autoenc_next * loss_autoenc_next)
        if token_unet_lookahead_token_count > 0:
            loss = loss + (lambda_token_unet_lookahead_ce * loss_token_unet_lookahead)
        _set_train_loss_stats(
            base_ce_loss_value=None if base_ce_loss_report is None else float(base_ce_loss_report.detach().item()),
            copy_ce_loss_value=float(loss_copy.detach().item()) if copy_token_count > 0 else None,
            copy_tokens_value=copy_token_count,
            autoenc_ce_loss_value=(None if autoenc_only_diffusion_mode else (float(loss_autoenc.detach().item()) if autoenc_token_count > 0 else None)),
            autoenc_next_ce_loss_value=(None if autoenc_only_diffusion_mode else (float(loss_autoenc_next.detach().item()) if autoenc_next_token_count > 0 else None)),
            token_unet_lookahead_ce_loss_value=(float(loss_token_unet_lookahead.detach().item()) if token_unet_lookahead_token_count > 0 else None),
            objective_loss_value=float(loss.detach().item()),
        )
        #loss = loss_impute
        # Scale loss for gradient accumulation
        _set_objective_loss_stat(float(loss.detach().item()))
        scaled_loss = loss / gradient_accumulation_steps
        
        # Backward pass
        (scaler.scale(scaled_loss) if scaler else scaled_loss).backward()
        
    return loss.item()  # Return unscaled final objective for metrics


#############################################
# Incremental Graph Functions
#############################################

def add_next_token_node(self, graph, token_embedding):
    """
    Add a node for the next token to the graph with optimized edge connections.
    
    Args:
        graph: The current graph
        token_embedding: Embedding for the next token
        
    Returns:
        updated_graph: Graph with next token node added
        next_token_idx: Index of the next token node
    """
    device = graph.x.device
    
    # Get index for the new node
    next_token_idx = graph.x.size(0)
    
    # Add the node features
    graph.x = torch.cat([graph.x, token_embedding.view(1, -1)], dim=0)
    
    # Add node level (L0)
    graph.node_level = torch.cat([
        graph.node_level,
        torch.zeros(1, dtype=torch.long, device=device)
    ], dim=0)
    
    # Get token level nodes (L0)
    if hasattr(graph, 'level_offsets') and len(graph.level_offsets) > 1:
        l0_end = graph.level_offsets[1]
    else:
        # Handle case where level_offsets might not be available
        l0_mask = (graph.node_level == 0)
        l0_indices = torch.where(l0_mask)[0]
        l0_end = l0_indices.size(0) if l0_indices.size(0) > 0 else graph.x.size(0) - 1
    
    # Define connection pattern - connect to recent tokens and important nodes
    edges = []
    edge_types = []
    
    # Connect to recent tokens (recency-based connections)
    context_window = min(1, l0_end)  # Use last 5 tokens or fewer
    for i in range(max(0, l0_end - context_window), l0_end):
        # Bidirectional connections
        edges.append([i, next_token_idx])
        edges.append([next_token_idx, i])
        edge_types.extend([0, 0])  # Token-level edge type
    
    # Connect to higher-level nodes (L1) if they exist (skip-connections)
    if hasattr(graph, 'level_offsets') and len(graph.level_offsets) > 2:
        l1_start = graph.level_offsets[1]
        l1_end = graph.level_offsets[2]
        
        # Connect to the last few L1 nodes (summary nodes)
        l1_window = min(2, l1_end - l1_start)
        for i in range(max(l1_start, l1_end - l1_window), l1_end):
            edges.append([i, next_token_idx])
            edges.append([next_token_idx, i])
            edge_types.extend([4, 5])  # Cross-level edge types
    
    # Add edges to graph (if any)
    if edges:
        edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).t()
        edge_types_tensor = torch.tensor(edge_types, dtype=torch.long, device=device)
        
        # Combine with existing edges
        graph.edge_index = torch.cat([graph.edge_index, edge_tensor], dim=1)
        graph.edge_type = torch.cat([graph.edge_type, edge_types_tensor])
        
        # Generate edge features if using edge attributes
        if hasattr(graph, 'edge_attr') and hasattr(self, 'edge_feature_generator'):
            # Generate features for new edges
            new_edge_attr = self.edge_feature_generator(
                graph.x, 
                edge_tensor,
                edge_types_tensor
            )
            
            # Combine with existing edge attributes
            graph.edge_attr = torch.cat([graph.edge_attr, new_edge_attr], dim=0)
    
    # Update level offsets if they exist
    if hasattr(graph, 'level_offsets') and len(graph.level_offsets) > 1:
        graph.level_offsets[1] += 1
    
    return graph, next_token_idx


def process_next_token(self, graph, token_idx, num_cycles=2):
    """
    Process graph with next token node using localized updates.

    Args:
        graph: Graph with next token node
        token_idx: Index of next token node
        num_cycles: Number of update cycles

    Returns:
        processed_graph: Updated graph after processing
    """
    local_nodes = set()
    local_nodes.add(token_idx)
    for edge_idx in range(graph.edge_index.size(1)):
        source, target = graph.edge_index[:, edge_idx]
        if source.item() == token_idx:
            local_nodes.add(target.item())
        elif target.item() == token_idx:
            local_nodes.add(source.item())
    local_nodes_list = list(local_nodes)
    local_nodes_tensor = torch.tensor(local_nodes_list, device=graph.x.device)
    neighborhood_mask = torch.zeros(graph.x.size(0), dtype=torch.bool, device=graph.x.device)
    neighborhood_mask[local_nodes_tensor] = True

    for cycle in range(num_cycles):
        if hasattr(self, 'share_transformers') and self.share_transformers:
            for level_idx, level_transformers in enumerate(self.level_transformers):
                for idx, transformer in enumerate(level_transformers):
                    if idx >= min(2, len(level_transformers)):
                        break

                    edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None

                    all_updated_features = transformer(
                        graph.x,
                        graph.edge_index,
                        graph.node_level,
                        edge_attr=edge_attr
                    )
                    #with torch.no_grad():
                         #graph.x[neighborhood_mask] = all_updated_features[neighborhood_mask]
                    graph.x = all_updated_features

                    graph.x = self.layer_norm(graph.x)
        else:
            transformers_to_use = getattr(self, 'refinement_transformers',
                                         self.level_transformers[0][:min(2, len(self.level_transformers[0]))] if self.level_transformers else [])

            for transformer in transformers_to_use:
                edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None

                all_updated_features = transformer(
                    graph.x,
                    graph.edge_index,
                    graph.node_level,
                    edge_attr=edge_attr
                )
                with torch.no_grad():
                     graph.x[neighborhood_mask] = all_updated_features[neighborhood_mask]

                graph.x = self.layer_norm(graph.x)

    return graph


def generate_with_graph_updates(self, input_ids, max_length=100, temperature=1.0, 
                               top_k=50, top_p=0.95, repetition_penalty=1.2, 
                               num_cycles=None, use_level_prediction=False):
    """
    Generate text by incrementally updating the graph for each new token.
    
    Args:
        input_ids: Starting input tokens [batch_size, seq_len]
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        repetition_penalty: Penalty for repeating tokens
        num_cycles: Number of graph processing cycles after adding each token
        use_level_prediction: Whether to use level projection
        
    Returns:
        generated_ids: Complete generated sequence
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    cycles = num_cycles if num_cycles is not None else self.refinement_cycles
    
    # Only support batch size of 1 for now
    if batch_size > 1:
        raise ValueError("Incremental generation currently only supports batch size of 1")
    
    # Start with input sequence
    current_ids = input_ids.clone()
    
    # Initial graph build - we do this once and then update incrementally
    with torch.no_grad():
        # Get token embeddings
        token_embeddings = self._get_embeddings(current_ids)
        token_embeddings = token_embeddings.view(-1, self.hidden_dim)
        
        # Process hierarchical graph
        hierarchy_results = self._process_hierarchical_graph(
            token_embeddings, cycles, use_level_prediction
        )
        
        # Store the refined graph state
        current_graph = hierarchy_results["refined_graph"]
        token_features = hierarchy_results["token_features"]
        
        # Generate tokens one by one
        for _ in range(max_length - seq_len):
            # Check if we've reached maximum sequence length
            if current_ids.size(1) >= self.max_seq_len:
                break
            
            # Get logits for next token prediction
            last_token_features = token_features[-1:] 
            next_token_logits = self.output_projection(last_token_features).squeeze(0) / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for token_id in set(current_ids[0].tolist()):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty
            
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
            
            # Add the token to the current sequence
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS token
            if hasattr(self, 'tokenizer') and next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Get embedding for the new token
            new_token_embedding = self.token_embedding(next_token)
            
            # Update the graph with the new token (instead of rebuilding)
            updated_graph, next_token_idx = add_next_token_node(
                self, current_graph, new_token_embedding
            )
            
            # Process the updated graph with a few cycles to integrate the new token
            updated_graph = process_next_token(
                self, 
                updated_graph, 
                next_token_idx, 
                num_cycles=min(2, cycles)  # Use fewer cycles for efficiency
            )
            
            # Update the graph for next iteration
            current_graph = updated_graph
            
            # Update token features by extracting from updated graph
            if use_level_prediction:
                # Apply level projection for the new token
                highest_level_idx = len(current_graph.level_offsets) - 2
                highest_level_start = current_graph.level_offsets[highest_level_idx]
                highest_level_end = current_graph.level_offsets[highest_level_idx + 1]
                highest_level_features = current_graph.x[highest_level_start:highest_level_end]
                
                # Project highest level features
                global_context = self.highest_to_token_projection(highest_level_features.mean(dim=0))
                global_context = global_context.unsqueeze(0)  # [1, hidden_dim]
                
                # New token features with level projection
                new_token_features = current_graph.x[next_token_idx:next_token_idx+1] + global_context
            else:
                # Use token features directly
                new_token_features = current_graph.x[next_token_idx:next_token_idx+1]
            
            # Append new token features
            token_features = torch.cat([token_features, new_token_features], dim=0)
    
    return current_ids


#############################################
# Enhanced Hierarchical Trainer
#############################################

class EnhancedHierarchicalTrainer:
    """
    Enhanced trainer for hierarchical models with hybrid masking and incremental generation.
    """
    
    def __init__(
        self,
        model,
        ema_model,
        optimizer=None,
        lr_scheduler=None,
        tokenizer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        checkpoint_dir="./checkpoints",
        log_interval=10000,
        eval_interval=10000,
        mixed_precision=False,
        variable_cycles=None,
        unified_refinement_cycles=None,
        progress_view="full",#"compact", rotate, or full
        progress_max_fields=8,
        progress_rotate_every=100,
        progress_update_every=1,
        progress_alias=True,
        progress_detail_interval=None,
        train_objective_mode="ar", # "masked", "ar", or "hybrid"
        lambda_masked_loss=1.0,
        lambda_ar_loss=1,#0.1,
        lambda_base_ce_loss=1.0,
        lambda_copy_loss=1.0,
        eval_ppl_ignore_prefix_tokens=0,
        eval_report_truncated_ppl=True,
        longctx_diag_every=0,        # >0: every N validations, log next-token PPL bucketed by
                                     # in-context recurrence distance (cheap, reuses val batches)
        longctx_diag_max_seqs=64,    # cap sequences scanned for the diagnostic (keeps it cheap)
        ce_label_smoothing_train=0.0,
        copy_task_enable=False,
        copy_task_train_prob=0.1,
        copy_task_val_prob=1.0,
        copy_task_src_len_min=8,
        copy_task_src_len_max=64,
        copy_task_min_gap=64,
        copy_task_max_gap=0,
        copy_task_mask_dst_in_ar=False,
        lambda_unmasked_loss=0.0,
        lambda_autoenc_loss=0.0,
        lambda_autoenc_next_loss=0.0,
        lambda_token_unet_lookahead_ce=0.0,
        chunked_ce_enable=False,
        chunked_ce_seq_chunk=0,
        train_feature_chunked_ce_enable=False,
        diffusion_mask_mode="random",
        diffusion_mask_block_size=4,
        diffusion_mask_path_length=64,
        autoenc_training_policy="auxiliary",
        llada_loss_weighting=False,
        modality="text",
        image_token_mode="latent",
        image_size=256,
        image_patch_size=16,
        image_latent_model_name="stabilityai/sd-vae-ft-mse",
        image_latent_scaling_factor=0.18215,
        image_latent_channels=4,
        image_latent_downsample=8,
        image_rgb_unet_token_dim=64,
        image_rgb_unet_downsample=16,
        image_rgb_unet_base_channels=64,
        image_rgb_unet_kernel_size=5,
        image_rgb_unet_decode_kernel_size=3,
        image_rgb_unet_decode_separable=True,
        image_rgb_unet_max_channels=512,
        image_diffusion_target="auto",
        image_diffusion_schedule="cosine",
        image_diffusion_prediction="auto",
        image_rgb_centered_diffusion=True,
        image_diffusion_min_snr_gamma=0.0,
        image_num_classes=1000,
        image_diffusion_steps=1000,
        image_diffusion_beta_start=1e-4,
        image_diffusion_beta_end=2e-2,
        image_diffusion_val_fixed_timesteps=None,
        image_fid_enable=False,
        image_fid_num_samples=2048,
        image_fid_guidance_scale=3.0,
        image_fid_diffusion_steps=0,
        image_fid_save_examples=False,
        image_fid_examples_per_eval=8,
        image_fid_examples_dir="",
        image_preview_enable=False,
        image_preview_num_samples=8,
        image_preview_guidance_scale=3.0,
        image_preview_diffusion_steps=0,
        image_preview_examples_dir="",
        image_diffusion_strict_shapes=True,
        image_sampling_legacy_update=False,
        image_sampling_respace_timesteps=True,
        image_sampling_clamp_x0=True,
        image_objective="diffusion",
        image_maskgit_variant="continuous",
        image_maskgit_vq_model_name="",
        image_maskgit_vq_tokenizer=None,
        image_maskgit_train_mask_min=0.10,
        image_maskgit_train_mask_max=1.00,
        image_maskgit_val_mask_ratio=0.50,
        image_maskgit_mask_value="zero",
        image_maskgit_unmasked_weight=0.0,
        image_maskgit_steps=12,
        image_maskgit_schedule="cosine",
        image_maskgit_confidence="stability",
        image_maskgit_use_timestep_cond=False,
        image_maskgit_temperature_anneal=True,
        image_maskgit_temperature_start=1.0,
        image_maskgit_temperature_end=0.1,
        image_roundtrip_check=True,
        image_roundtrip_check_num_samples=4,
        image_roundtrip_check_fail_fast=True,
        recurrent_training_enable=False,
        recurrent_val_enable=False,
        recurrent_l0_window=0,
        recurrent_unroll_len=64,
        recurrent_detach_every=64,
        recurrent_loss_stride=1,
        recurrent_warmup_tokens=0,
        train_metrics_callback=None,
        train_metrics_interval=0,
        # Memory training parameters
    ):
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.mixed_precision = mixed_precision
        # Prefer bf16 only where the GPU supports it NATIVELY (Ampere sm_80+). On bf16-less
        # GPUs (e.g. T4 / Turing sm_75) fall back to fp16, which then gets a GradScaler below.
        self.use_bf16 = bf16_supported()
        if self.mixed_precision:
            logger.info(
                "Mixed precision autocast dtype: %s (native_bf16=%s)",
                "bfloat16" if self.use_bf16 else "float16",
                self.use_bf16,
            )
        self.unified_refinement_cycles = unified_refinement_cycles
        self.progress_view = str(progress_view).lower()
        if self.progress_view not in {"full", "compact", "rotate"}:
            self.progress_view = "compact"
        self.progress_max_fields = max(4, int(progress_max_fields))
        self.progress_rotate_every = max(1, int(progress_rotate_every))
        self.progress_update_every = max(1, int(progress_update_every))
        self.progress_alias = bool(progress_alias)
        if progress_detail_interval is None:
            self.progress_detail_interval = max(1, int(log_interval))
        else:
            self.progress_detail_interval = max(1, int(progress_detail_interval))
        self.train_objective_mode = str(train_objective_mode).lower()
        if self.train_objective_mode not in {"masked", "ar", "hybrid"}:
            self.train_objective_mode = "masked"
        self.lambda_masked_loss = float(lambda_masked_loss)
        self.lambda_ar_loss = float(lambda_ar_loss)
        self.lambda_base_ce_loss = float(lambda_base_ce_loss)
        self.lambda_copy_loss = float(lambda_copy_loss)
        self.eval_ppl_ignore_prefix_tokens = max(0, int(eval_ppl_ignore_prefix_tokens))
        self.eval_report_truncated_ppl = bool(eval_report_truncated_ppl)
        self.longctx_diag_every = max(0, int(longctx_diag_every))
        self.longctx_diag_max_seqs = max(1, int(longctx_diag_max_seqs))
        self._longctx_diag_calls = 0
        self.ce_label_smoothing_train = max(0.0, float(ce_label_smoothing_train))
        self.copy_task_enable = bool(copy_task_enable)
        self.copy_task_train_prob = max(0.0, min(1.0, float(copy_task_train_prob)))
        self.copy_task_val_prob = max(0.0, min(1.0, float(copy_task_val_prob)))
        self.copy_task_src_len_min = max(1, int(copy_task_src_len_min))
        self.copy_task_src_len_max = max(self.copy_task_src_len_min, int(copy_task_src_len_max))
        self.copy_task_min_gap = max(0, int(copy_task_min_gap))
        self.copy_task_max_gap = int(copy_task_max_gap)
        self.copy_task_mask_dst_in_ar = bool(copy_task_mask_dst_in_ar)
        self.lambda_unmasked_loss = float(lambda_unmasked_loss)
        self.lambda_autoenc_loss = float(lambda_autoenc_loss)
        self.lambda_autoenc_next_loss = float(lambda_autoenc_next_loss)
        self.lambda_token_unet_lookahead_ce = float(lambda_token_unet_lookahead_ce)
        self.chunked_ce_enable = bool(chunked_ce_enable)
        self.chunked_ce_seq_chunk = max(0, int(chunked_ce_seq_chunk))
        self.train_feature_chunked_ce_enable = bool(train_feature_chunked_ce_enable)
        self.diffusion_mask_mode = str(diffusion_mask_mode).lower()
        if self.diffusion_mask_mode not in {"random", "block", "path"}:
            self.diffusion_mask_mode = "random"
        self.diffusion_mask_block_size = max(1, int(diffusion_mask_block_size))
        self.diffusion_mask_path_length = max(1, int(diffusion_mask_path_length))
        self.autoenc_training_policy = str(autoenc_training_policy).lower()
        if self.autoenc_training_policy not in {"auxiliary", "autoenc_only", "autoenc_only_diffusion"}:
            self.autoenc_training_policy = "auxiliary"
        self.llada_loss_weighting = bool(llada_loss_weighting)
        self.modality = str(modality).lower()
        if self.modality not in {"text", "image"}:
            self.modality = "text"
        self.image_token_mode = str(image_token_mode).lower()
        if self.image_token_mode not in {"latent", "raw_rgb_patches", "rgb_unet"}:
            self.image_token_mode = "latent"
        self.image_size = max(8, int(image_size))
        self.image_patch_size = max(1, int(image_patch_size))
        self.image_latent_model_name = str(image_latent_model_name)
        self.image_latent_scaling_factor = float(image_latent_scaling_factor)
        self.image_latent_channels = max(1, int(image_latent_channels))
        self.image_latent_downsample = max(1, int(image_latent_downsample))
        self.image_rgb_unet_token_dim = max(1, int(image_rgb_unet_token_dim))
        self.image_rgb_unet_downsample = max(1, int(image_rgb_unet_downsample))
        self.image_rgb_unet_base_channels = max(16, int(image_rgb_unet_base_channels))
        self.image_rgb_unet_kernel_size = max(1, int(image_rgb_unet_kernel_size))
        self.image_rgb_unet_decode_kernel_size = max(1, int(image_rgb_unet_decode_kernel_size))
        self.image_rgb_unet_decode_separable = bool(image_rgb_unet_decode_separable)
        self.image_rgb_unet_max_channels = max(self.image_rgb_unet_base_channels, int(image_rgb_unet_max_channels))
        self.image_diffusion_target = str(image_diffusion_target).lower()
        if self.image_diffusion_target not in {"auto", "token_epsilon", "rgb_epsilon"}:
            self.image_diffusion_target = "auto"
        if self.image_diffusion_target == "auto":
            self.image_diffusion_target_runtime = "rgb_epsilon" if self.image_token_mode == "rgb_unet" else "token_epsilon"
        else:
            self.image_diffusion_target_runtime = self.image_diffusion_target
        self.image_diffusion_schedule = str(image_diffusion_schedule).lower()
        if self.image_diffusion_schedule not in {"linear", "cosine"}:
            self.image_diffusion_schedule = "cosine"
        self.image_diffusion_prediction = str(image_diffusion_prediction).lower()
        if self.image_diffusion_prediction not in {"auto", "epsilon", "v", "x0"}:
            self.image_diffusion_prediction = "auto"
        if self.image_diffusion_prediction == "auto":
            self.image_diffusion_prediction_runtime = "v"
        else:
            self.image_diffusion_prediction_runtime = self.image_diffusion_prediction
        self.image_rgb_centered_diffusion = bool(image_rgb_centered_diffusion)
        self.image_diffusion_min_snr_gamma = max(0.0, float(image_diffusion_min_snr_gamma))
        self.image_num_classes = max(1, int(image_num_classes))
        self.image_diffusion_steps = max(2, int(image_diffusion_steps))
        self.image_diffusion_beta_start = float(image_diffusion_beta_start)
        self.image_diffusion_beta_end = float(image_diffusion_beta_end)
        # Validation timestep policy. Empty/None = the legacy seeded-random draw over the FULL
        # 0..steps range (comparable across epochs but dominated by the near-irreducible
        # high-noise term). A non-empty list pins validation to those fixed timesteps (cycled
        # across the batch), turning the val loss into a low-variance progress signal that
        # isolates informative noise levels. Clamped to [0, steps-1].
        _fixed_ts = image_diffusion_val_fixed_timesteps
        if _fixed_ts is None:
            _fixed_ts = []
        elif isinstance(_fixed_ts, (int, float)):
            _fixed_ts = [_fixed_ts]
        self.image_diffusion_val_fixed_timesteps = [
            max(0, min(int(self.image_diffusion_steps) - 1, int(v))) for v in _fixed_ts
        ]
        self.image_fid_enable = bool(image_fid_enable)
        self.image_fid_num_samples = max(64, int(image_fid_num_samples))
        self.image_fid_guidance_scale = float(image_fid_guidance_scale)
        self.image_fid_diffusion_steps = max(0, int(image_fid_diffusion_steps))
        self.image_fid_save_examples = bool(image_fid_save_examples)
        self.image_fid_examples_per_eval = max(1, int(image_fid_examples_per_eval))
        self.image_fid_examples_dir = str(image_fid_examples_dir or "")
        self.image_preview_enable = bool(image_preview_enable)
        self.image_preview_num_samples = max(1, int(image_preview_num_samples))
        self.image_preview_guidance_scale = float(image_preview_guidance_scale)
        self.image_preview_diffusion_steps = max(0, int(image_preview_diffusion_steps))
        self.image_preview_examples_dir = str(image_preview_examples_dir or "")
        self.image_diffusion_strict_shapes = bool(image_diffusion_strict_shapes)
        self.image_sampling_legacy_update = bool(image_sampling_legacy_update)
        self.image_sampling_respace_timesteps = bool(image_sampling_respace_timesteps)
        self.image_sampling_clamp_x0 = bool(image_sampling_clamp_x0)
        self.image_objective = str(image_objective).lower()
        if self.image_objective not in {"diffusion", "maskgit"}:
            self.image_objective = "diffusion"
        self.image_maskgit_variant = str(image_maskgit_variant).lower()
        if self.image_maskgit_variant not in {"continuous", "discrete"}:
            self.image_maskgit_variant = "continuous"
        self.image_maskgit_vq_model_name = str(image_maskgit_vq_model_name or "").strip()
        self._image_maskgit_vq_tokenizer = image_maskgit_vq_tokenizer
        self.image_maskgit_codebook_size = int(getattr(image_maskgit_vq_tokenizer, "codebook_size", 0)) if image_maskgit_vq_tokenizer is not None else 0
        self.image_maskgit_mask_token_id = int(getattr(image_maskgit_vq_tokenizer, "mask_token_id", -1)) if image_maskgit_vq_tokenizer is not None else -1
        self.image_maskgit_train_mask_min = max(0.0, min(1.0, float(image_maskgit_train_mask_min)))
        self.image_maskgit_train_mask_max = max(self.image_maskgit_train_mask_min, min(1.0, float(image_maskgit_train_mask_max)))
        self.image_maskgit_val_mask_ratio = max(0.0, min(1.0, float(image_maskgit_val_mask_ratio)))
        self.image_maskgit_mask_value = str(image_maskgit_mask_value).lower()
        if self.image_maskgit_mask_value not in {"zero", "noise"}:
            self.image_maskgit_mask_value = "zero"
        self.image_maskgit_unmasked_weight = max(0.0, float(image_maskgit_unmasked_weight))
        self.image_maskgit_steps = max(2, int(image_maskgit_steps))
        self.image_maskgit_schedule = str(image_maskgit_schedule).lower()
        if self.image_maskgit_schedule not in {"cosine", "linear"}:
            self.image_maskgit_schedule = "cosine"
        self.image_maskgit_confidence = str(image_maskgit_confidence).lower()
        if self.image_maskgit_confidence not in {"stability", "random"}:
            self.image_maskgit_confidence = "stability"
        self.image_maskgit_use_timestep_cond = bool(image_maskgit_use_timestep_cond)
        self.image_maskgit_temperature_anneal = bool(image_maskgit_temperature_anneal)
        self.image_maskgit_temperature_start = max(1e-6, float(image_maskgit_temperature_start))
        self.image_maskgit_temperature_end = max(1e-6, float(image_maskgit_temperature_end))
        self.image_roundtrip_check = bool(image_roundtrip_check)
        self.image_roundtrip_check_num_samples = max(1, int(image_roundtrip_check_num_samples))
        self.image_roundtrip_check_fail_fast = bool(image_roundtrip_check_fail_fast)
        self.recurrent_training_enable = bool(recurrent_training_enable)
        self.recurrent_val_enable = bool(recurrent_val_enable)
        self.recurrent_l0_window = max(0, int(recurrent_l0_window or 0))
        self.recurrent_unroll_len = max(1, int(recurrent_unroll_len))
        self.recurrent_detach_every = max(0, int(recurrent_detach_every))
        self.recurrent_loss_stride = max(1, int(recurrent_loss_stride))
        self.recurrent_warmup_tokens = max(0, int(recurrent_warmup_tokens))
        self._image_vae = None
        self._image_diffusion_cache = {}
        self._image_runtime_grid_shape = None
        self._image_runtime_feature_dim = None
        self._image_shape_mismatch_warned = False
        self._image_maskgit_rgb_unet_warned = False
        self._image_sampling_schedule_logged_keys = set()
        resolved_mode = self._resolve_autoenc_runtime_mode(model)
        logger.info(
            "Autoenc policy active: %s (graph_mode=%s)",
            resolved_mode.get("autoenc_training_policy", self.autoenc_training_policy),
            resolved_mode.get("autoenc_graph_mode", "off"),
        )
        if bool(resolved_mode.get("autoenc_only_diffusion_mode", False)):
            logger.info("Autoenc-only-diffusion: masked objective forced on L0' logits; diffusion auto uses AE head.")
        logger.info(
            "Base CE scale (AR mode only): %.6f",
            float(self.lambda_base_ce_loss),
        )
        logger.info(
            "CE label smoothing (train CE branches): %.6f",
            float(self.ce_label_smoothing_train),
        )
        logger.info(
            "Chunked CE (train-only): enable=%s seq_chunk=%d",
            bool(self.chunked_ce_enable),
            int(self.chunked_ce_seq_chunk),
        )
        logger.info(
            "Diffusion masker: mode=%s block_size=%d path_length=%d",
            str(self.diffusion_mask_mode),
            int(self.diffusion_mask_block_size),
            int(self.diffusion_mask_path_length),
        )
        if self.modality == "image":
            logger.info(
                "Image mode active: objective=%s variant=%s token_mode=%s target=%s pred=%s schedule=%s image_size=%d patch=%d latent_model=%s rgb_unet_token_dim=%d rgb_unet_downsample=%d rgb_unet_kernel=%d rgb_unet_decode_kernel=%d rgb_unet_decode_sep=%s rgb_unet_max_channels=%d centered_rgb=%s min_snr_gamma=%.3f diffusion_steps=%d classes=%d",
                self.image_objective,
                self.image_maskgit_variant,
                self.image_token_mode,
                self.image_diffusion_target_runtime,
                self.image_diffusion_prediction_runtime,
                self.image_diffusion_schedule,
                int(self.image_size),
                int(self.image_patch_size),
                self.image_latent_model_name,
                int(self.image_rgb_unet_token_dim),
                int(self.image_rgb_unet_downsample),
                int(self.image_rgb_unet_kernel_size),
                int(self.image_rgb_unet_decode_kernel_size),
                bool(self.image_rgb_unet_decode_separable),
                int(self.image_rgb_unet_max_channels),
                bool(self.image_rgb_centered_diffusion),
                float(self.image_diffusion_min_snr_gamma),
                int(self.image_diffusion_steps),
                int(self.image_num_classes),
            )
            if self.image_objective == "maskgit":
                logger.info(
                    "Image MaskGIT config: variant=%s mask_ratio=[%.3f, %.3f] val_mask_ratio=%.3f mask_value=%s steps=%d schedule=%s confidence=%s timestep_cond=%s temp_anneal=%s temp=[%.3f, %.3f] unmasked_weight=%.3f vq_model=%s",
                    str(self.image_maskgit_variant),
                    float(self.image_maskgit_train_mask_min),
                    float(self.image_maskgit_train_mask_max),
                    float(self.image_maskgit_val_mask_ratio),
                    str(self.image_maskgit_mask_value),
                    int(self.image_maskgit_steps),
                    str(self.image_maskgit_schedule),
                    str(self.image_maskgit_confidence),
                    bool(self.image_maskgit_use_timestep_cond),
                    bool(self.image_maskgit_temperature_anneal),
                    float(self.image_maskgit_temperature_start),
                    float(self.image_maskgit_temperature_end),
                    float(self.image_maskgit_unmasked_weight),
                    str(self.image_maskgit_vq_model_name or "<none>"),
                )
                if self.image_maskgit_variant == "discrete" and not self.image_maskgit_vq_model_name:
                    logger.warning("Image objective=maskgit with variant=discrete requires image_maskgit_vq_model_name; loading will fail until provided.")
                if self.image_maskgit_variant == "continuous" and self.image_token_mode == "rgb_unet":
                    logger.warning(
                        "Image objective=maskgit with image_token_mode=rgb_unet will train in feature space, but preview/FID will fall back to diffusion sampling because rgb_unet decode needs encoder context."
                    )
                if self.image_maskgit_variant == "discrete" and self.image_maskgit_use_timestep_cond:
                    logger.warning("Image objective=maskgit with variant=discrete ignores pseudo timestep conditioning; timesteps are not used in discrete MaskGIT.")
            logger.info(
                "Image diffusion runtime controls: strict_shapes=%s legacy_sampler=%s respace_timesteps=%s",
                bool(self.image_diffusion_strict_shapes),
                bool(self.image_sampling_legacy_update),
                bool(self.image_sampling_respace_timesteps),
            )
            if self.image_token_mode == "rgb_unet" and self.image_diffusion_target_runtime == "token_epsilon":
                logger.warning(
                    "rgb_unet + token_epsilon selected: decoder is not directly supervised; prefer rgb_epsilon for visual quality/FID."
                )
            if self.image_fid_enable:
                logger.info(
                    "Image FID config: samples=%d guidance=%.3f fid_steps=%d save_examples=%s examples_per_eval=%d examples_dir=%s",
                    int(self.image_fid_num_samples),
                    float(self.image_fid_guidance_scale),
                    int(self.image_fid_diffusion_steps if self.image_fid_diffusion_steps > 0 else self.image_diffusion_steps),
                    bool(self.image_fid_save_examples),
                    int(self.image_fid_examples_per_eval),
                    str(self.image_fid_examples_dir or "<checkpoint_dir>/fid_examples"),
                )
            if self.image_preview_enable:
                logger.info(
                    "Image preview config: samples=%d guidance=%.3f preview_steps=%d examples_dir=%s",
                    int(self.image_preview_num_samples),
                    float(self.image_preview_guidance_scale),
                    int(self.image_preview_diffusion_steps if self.image_preview_diffusion_steps > 0 else self.image_diffusion_steps),
                    str(self.image_preview_examples_dir or "<checkpoint_dir>/preview_examples"),
                )
            self._validate_image_feature_mode_io()
        model_mod = getattr(model, "module", None)
        l0_local_mode = self._resolve_l0_local_runtime_mode(model)
        logger.info(
            "L0 local backend config: backend=%s window=%d active=%s",
            l0_local_mode.get("backend", "pyg"),
            int(l0_local_mode.get("window", 0)),
            bool(l0_local_mode.get("active", False)),
        )
        la_cfg = l0_local_mode.get("local_attn_config", {})
        if la_cfg:
            for lvl in sorted(la_cfg.keys()):
                c = la_cfg[lvl]
                logger.info(
                    "Multi-level local attn: level=%d backend=%s window=%d causal=%s",
                    int(lvl), str(c.get("backend", "sdpa")), int(c.get("window", 0)), bool(c.get("causal", False)),
                )
        token_unet_enable = bool(getattr(model, "token_unet_enable", getattr(model_mod, "token_unet_enable", False)))
        token_unet_mode = str(getattr(model, "token_unet_mode", getattr(model_mod, "token_unet_mode", "stem")))
        token_unet_scale = int(getattr(model, "token_unet_scale", getattr(model_mod, "token_unet_scale", 1)))
        token_unet_kernel = int(getattr(model, "token_unet_kernel_size", getattr(model_mod, "token_unet_kernel_size", 5)))
        token_unet_dropout = float(getattr(model, "token_unet_dropout", getattr(model_mod, "token_unet_dropout", 0.0)))
        token_unet_right_edge = bool(getattr(model, "token_unet_right_edge_targets", getattr(model_mod, "token_unet_right_edge_targets", True)))
        token_unet_lookahead = bool(getattr(model, "token_unet_lookahead_decode_enable", getattr(model_mod, "token_unet_lookahead_decode_enable", False)))
        token_unet_lookahead_kernel = int(getattr(model, "token_unet_lookahead_kernel_size", getattr(model_mod, "token_unet_lookahead_kernel_size", 5)))
        token_unet_lookahead_blocks = int(getattr(model, "token_unet_lookahead_blocks", getattr(model_mod, "token_unet_lookahead_blocks", 2)))
        logger.info(
            "Token U-Net config: enable=%s mode=%s scale=%d kernel=%d dropout=%.3f right_edge_targets=%s lookahead=%s lookahead_kernel=%d lookahead_blocks=%d lambda_lookahead_ce=%.6f",
            token_unet_enable,
            token_unet_mode,
            token_unet_scale,
            token_unet_kernel,
            token_unet_dropout,
            token_unet_right_edge,
            token_unet_lookahead,
            token_unet_lookahead_kernel,
            token_unet_lookahead_blocks,
            float(self.lambda_token_unet_lookahead_ce),
        )
        self._cache_prev_hits = 0
        self._cache_prev_misses = 0
        self._cache_prev_restarts = 0
        self._cache_prev_seeds = 0
        self.train_metrics_callback = train_metrics_callback
        self.train_metrics_interval = max(0, int(train_metrics_interval))
        self._collapse_warning_cooldown = 0
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup for mixed precision training if enabled, but don't use scaler with BF16
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() and not self.use_bf16 else None

        def chunked_cross_entropy(
            logits,            # [N, V] or [B, T, V]
            targets,           # [N] or [B, T]
            chunk_size=1024,
            ignore_index=-100,
            reduction="mean",
        ):
            # Accept [B,T,V] by flattening to [N,V]
            if logits.dim() == 3:
                B, T, V = logits.shape
                logits = logits.reshape(B*T, V)
                targets = targets.reshape(B*T)

            N = logits.size(0)
            device = logits.device
            dtype  = logits.dtype

            loss_sum = logits.new_zeros(())  # scalar on same device/dtype, keeps grad
            count    = logits.new_zeros(())  # scalar on same device (float)

            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)

                t = targets[i:j]
                if ignore_index is not None:
                    keep = (t != ignore_index)
                    if keep.any():
                        l = F.cross_entropy(
                            logits[i:j][keep], t[keep],
                            reduction="sum",
                            ignore_index=-100  # safe: no -100s remain, but fine
                        )
                        loss_sum = loss_sum + l
                        count    = count + keep.sum().to(dtype=loss_sum.dtype)
                else:
                    l = F.cross_entropy(
                        logits[i:j], t,
                        reduction="sum"
                    )
                    loss_sum = loss_sum + l
                    count    = count + (j - i)

            # Avoid div-by-zero: if everything was ignored, return zero (no grad)
            if count.item() == 0:
                return logits.new_zeros(())

            return loss_sum / count
    
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        #self.criterion = chunked_cross_entropy
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.global_step = 0

        # Memory training parameters

        self.copy_task_marker_ids = None
        self.copy_task_mask_token_id = None
        self.copy_task_hierarchy_thresholds = ()
        if self.copy_task_enable:
            try:
                self.copy_task_marker_ids = get_copy_task_marker_ids(self.tokenizer)
                mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
                if mask_token_id is None:
                    mask_token_id = getattr(self.tokenizer, "unk_token_id", None)
                if mask_token_id is None:
                    logger.warning("Copy-task enabled but tokenizer has no mask/unk token; disabling copy-task masking")
                    self.copy_task_enable = False
                else:
                    self.copy_task_mask_token_id = int(mask_token_id)
                    ratios = getattr(self.model, "compression_ratios", None)
                    self.copy_task_hierarchy_thresholds = hierarchy_thresholds_from_compression(ratios)
                    logger.info(
                        "Copy-task enabled: train_prob=%.3f val_prob=%.3f src_len=[%d,%d] gap=[%d,%s] mask_dst_in_ar=%s hier_thresh=%s",
                        self.copy_task_train_prob,
                        self.copy_task_val_prob,
                        self.copy_task_src_len_min,
                        self.copy_task_src_len_max,
                        self.copy_task_min_gap,
                        str(self.copy_task_max_gap) if self.copy_task_max_gap > 0 else "auto",
                        str(self.copy_task_mask_dst_in_ar),
                        str(self.copy_task_hierarchy_thresholds),
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize copy-task markers; disabling copy-task: {e}")
                self.copy_task_enable = False
        
        # Patch model with incremental generation
        self._patch_model_with_incremental_generation()
    
    def _patch_model_with_incremental_generation(self):
        """Patch the model with the incremental generation method if needed."""
        # Store reference to tokenizer in model if not already there
        if not hasattr(self.model, 'tokenizer') and self.tokenizer is not None:
            self.model.tokenizer = self.tokenizer
            
        # Only patch methods if they don't already exist
        if not hasattr(self.model, 'add_next_token_node'):
            self.model.add_next_token_node = add_next_token_node.__get__(self.model, type(self.model))
            
        if not hasattr(self.model, 'process_next_token'):
            self.model.process_next_token = process_next_token.__get__(self.model, type(self.model))
            
        if not hasattr(self.model, 'generate_with_graph_updates'):
            self.model.generate_with_graph_updates = generate_with_graph_updates.__get__(self.model, type(self.model))
        
        logger.info("Model patched with incremental generation methods")

    def _image_expected_grid_shape(self) -> Tuple[int, int]:
        if self.image_token_mode == "raw_rgb_patches":
            gh = int(self.image_size // self.image_patch_size)
            gw = int(self.image_size // self.image_patch_size)
            return max(1, gh), max(1, gw)
        if self.image_token_mode == "rgb_unet":
            gh = int(self.image_size // self.image_rgb_unet_downsample)
            gw = int(self.image_size // self.image_rgb_unet_downsample)
            return max(1, gh), max(1, gw)
        gh = int(self.image_size // self.image_latent_downsample)
        gw = int(self.image_size // self.image_latent_downsample)
        return max(1, gh), max(1, gw)

    def _image_expected_feature_dim(self) -> int:
        if self.image_token_mode == "raw_rgb_patches":
            return int(3 * self.image_patch_size * self.image_patch_size)
        if self.image_token_mode == "rgb_unet":
            return int(self.image_rgb_unet_token_dim)
        return int(self.image_latent_channels)

    def _validate_image_feature_mode_io(self) -> None:
        model_mod = getattr(self.model, "module", None)
        mdl = model_mod if model_mod is not None else self.model
        out_proj = getattr(mdl, "output_projection", None)
        if out_proj is None:
            logger.warning("Image mode: model has no output_projection; skipping feature I/O validation")
            return

        out_dim = int(getattr(out_proj, "out_features", 0))
        expected_dim = int(self._image_expected_feature_dim())
        input_mode = str(getattr(mdl, "input_mode", "tokens")).lower()
        tie_weights = bool(getattr(mdl, "tie_weights", False))

        logger.info(
            "Image feature I/O: input_mode=%s tie_weights=%s output_projection_dim=%d expected_feature_dim=%d",
            input_mode,
            tie_weights,
            int(out_dim),
            int(expected_dim),
        )

        if input_mode == "features" and out_dim != expected_dim:
            raise RuntimeError(
                f"Image feature-mode output dimension mismatch: output_projection_dim={out_dim} expected={expected_dim}. "
                "Check image_token_mode and model vocab_size/effective feature dim wiring."
            )

        if input_mode == "features" and tie_weights:
            logger.warning(
                "input_mode=features with tie_weights=True can be incompatible with feature reconstruction; prefer tie_weights=False"
            )

    def _ensure_image_vae(self, device: torch.device):
        if self._image_vae is not None:
            return self._image_vae
        try:
            from diffusers import AutoencoderKL
        except Exception as exc:
            raise ImportError(
                "diffusers is required for --image_token_mode latent. Install diffusers and accelerate."
            ) from exc
        vae = AutoencoderKL.from_pretrained(self.image_latent_model_name)
        vae.requires_grad_(False)
        vae.eval()
        vae.to(device)
        self._image_vae = vae
        return self._image_vae

    def check_image_roundtrip(
        self,
        batch: Dict[str, Any],
        eval_tag: Optional[str] = None,
        num_samples: int = 4,
        fail_fast: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Verify the encode -> decode pipeline produces valid, finite outputs.
        Saves real vs reconstructed images to <examples_dir>/roundtrip_check/.

        Returns a dict of metrics on success, None on skip/catch.
        Raises RuntimeError if fail_fast=True and the check fails.
        """
        try:
            from torchvision.utils import make_grid, save_image
        except Exception as exc:
            logger.warning("check_image_roundtrip: torchvision.utils unavailable, skipping: %s", exc)
            return None

        is_discrete = (
            str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit"
            and str(getattr(self, "image_maskgit_variant", "continuous")).lower() == "discrete"
        )

        pixel_values = batch.get("pixel_values", None)
        if pixel_values is None:
            if batch.get("input_features", None) is not None or batch.get("token_ids", None) is not None or batch.get("input_ids", None) is not None:
                logger.info("check_image_roundtrip: cache-backed batch detected, skipping pixel roundtrip check")
            else:
                logger.warning("check_image_roundtrip: batch has no pixel_values, skipping")
            return None

        pixel_values = pixel_values.to(self.device)
        bsz = int(pixel_values.size(0))
        take = min(num_samples, bsz)
        real = pixel_values[:take].clamp(0.0, 1.0)

        out_dir = os.path.join(self.image_preview_examples_dir or os.path.join(self.checkpoint_dir or ".", "preview_examples"), "roundtrip_check", str(eval_tag or "init"))
        os.makedirs(out_dir, exist_ok=True)

        metrics = {}

        if is_discrete:
            vq_tokenizer = self._get_image_maskgit_vq_tokenizer(device=self.device)
            with torch.no_grad():
                token_ids, grid_shape = vq_tokenizer.encode(real)
                recon = vq_tokenizer.decode(token_ids, grid_shape)

            tokens_flat = token_ids[:take].reshape(-1)
            mask_token_id = int(getattr(self, "image_maskgit_mask_token_id", vq_tokenizer.mask_token_id))
            mask_count = int((tokens_flat == mask_token_id).sum().item())
            unique_tokens = int(torch.unique(tokens_flat).size(0))
            total_tokens = int(tokens_flat.numel())

            metrics["roundtrip_token_unique_ratio"] = float(unique_tokens) / float(max(1, total_tokens))
            metrics["roundtrip_mask_token_count"] = int(mask_count)
            metrics["roundtrip_mask_token_pct"] = float(mask_count) / float(max(1, total_tokens))
            metrics["roundtrip_unique_tokens"] = int(unique_tokens)
            metrics["roundtrip_total_tokens"] = int(total_tokens)

            logger.info(
                "Roundtrip check (discrete) mask=%.1f%% unique=%d/%d (%.4f) shape=%s",
                float(mask_count) / float(max(1, total_tokens)) * 100,
                unique_tokens,
                total_tokens,
                float(metrics["roundtrip_token_unique_ratio"]),
                tuple(token_ids.shape),
            )

            if unique_tokens < 2:
                msg = (
                    f"ROUNDTRIP CHECK FAILED: Only {unique_tokens} unique token(s) in encoded ids. "
                    f"This usually means the VQ encoder is mapping all inputs to the same codebook entry. "
                    f"Check the VQ model and encoding pipeline."
                )
                if fail_fast:
                    raise RuntimeError(msg)
                logger.warning(msg)

        else:
            if self.image_token_mode == "latent":
                vae = self._ensure_image_vae(self.device)
                with torch.no_grad():
                    vae_in = real * 2.0 - 1.0
                    latent_dist = vae.encode(vae_in).latent_dist
                    if getattr(latent_dist, "mode", None) is not None:
                        latents = latent_dist.mode()
                    else:
                        latents = latent_dist.sample()
                    latents_scaled = latents * float(self.image_latent_scaling_factor)
                    recon_dist = vae.decode(latents_scaled / float(self.image_latent_scaling_factor)).sample
                    recon = (recon_dist / 2.0 + 0.5).clamp(0.0, 1.0)
                metrics["roundtrip_latent_mean"] = float(latents.mean().item())
                metrics["roundtrip_latent_std"] = float(latents.std().item())
                logger.info(
                    "Roundtrip check (latent) latent mean=%.4f std=%.4f",
                    float(metrics["roundtrip_latent_mean"]),
                    float(metrics["roundtrip_latent_std"]),
                )
            else:
                logger.info("check_image_roundtrip: raw_rgb/rgb_unet skip, no roundtrip needed")
                return None

        if recon.isnan().any() or recon.isinf().any():
            msg = f"Roundtrip decode produced NaN/Inf values!"
            if fail_fast:
                raise RuntimeError(msg)
            logger.warning(msg)
        else:
            metrics["roundtrip_recon_finite"] = True

        mse = float((real - recon).pow(2).mean().item())
        mae = float((real - recon).abs().mean().item())
        metrics["roundtrip_mse"] = float(mse)
        metrics["roundtrip_mae"] = float(mae)
        logger.info("Roundtrip check MSE=%.6f MAE=%.6f", float(mse), float(mae))

        if mse > 0.1:
            msg = f"Roundtrip check MSE={mse:.4f} is very high (>0.1), indicating decode may be broken."
            if fail_fast:
                raise RuntimeError(msg)
            logger.warning(msg)

        real_cpu = real.detach().cpu().clamp(0.0, 1.0)
        recon_cpu = recon.detach().cpu().clamp(0.0, 1.0)

        for i in range(take):
            idx = i
            save_image(real_cpu[i], os.path.join(out_dir, f"real_{idx:03d}.png"))
            save_image(recon_cpu[i], os.path.join(out_dir, f"recon_{idx:03d}.png"))

        nrow = max(1, int(math.sqrt(take)))
        save_image(make_grid(real_cpu[:take], nrow=nrow), os.path.join(out_dir, "real_grid.png"))
        save_image(make_grid(recon_cpu[:take], nrow=nrow), os.path.join(out_dir, "recon_grid.png"))

        logger.info("Roundtrip check artifacts saved to %s", out_dir)
        return metrics

    def _image_tokens_from_pixels(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be [B,C,H,W], got {tuple(pixel_values.shape)}")
        if self.image_token_mode == "rgb_unet":
            tokens, context = self._encode_pixels_rgb_unet_tokens(pixel_values, model_ref=self.model)
            gg = context.get("graph_grid_shape", self._image_expected_grid_shape()) if isinstance(context, dict) else self._image_expected_grid_shape()
            return tokens, (int(gg[0]), int(gg[1]))
        if self.image_token_mode == "raw_rgb_patches":
            bsz, ch, h, w = pixel_values.shape
            p = int(self.image_patch_size)
            if h % p != 0 or w % p != 0:
                raise ValueError(f"Image size ({h},{w}) is not divisible by patch_size={p}")
            patches = F.unfold(pixel_values, kernel_size=p, stride=p).transpose(1, 2).contiguous()
            return patches, (int(h // p), int(w // p))

        vae = self._ensure_image_vae(pixel_values.device)
        with torch.no_grad():
            vae_in = pixel_values.clamp(0.0, 1.0) * 2.0 - 1.0
            post = vae.encode(vae_in).latent_dist
            latents = post.sample() * float(self.image_latent_scaling_factor)
        bsz, ch, gh, gw = latents.shape
        tokens = latents.permute(0, 2, 3, 1).reshape(bsz, gh * gw, ch).contiguous()
        return tokens, (int(gh), int(gw))

    def _image_tokens_to_pixels(self, tokens_btd: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        if tokens_btd.dim() != 3:
            raise ValueError(f"tokens must be [B,T,D], got {tuple(tokens_btd.shape)}")
        gh, gw = int(grid_shape[0]), int(grid_shape[1])
        bsz, tok, dim = tokens_btd.shape
        if gh * gw != tok:
            raise ValueError(f"grid_shape {grid_shape} does not match token count {tok}")

        if self.image_token_mode == "rgb_unet":
            raise ValueError("_image_tokens_to_pixels is not used for rgb_unet mode; use decode context path")

        if self.image_token_mode == "raw_rgb_patches":
            p = int(self.image_patch_size)
            x = tokens_btd.transpose(1, 2).contiguous()
            x = F.fold(x, output_size=(gh * p, gw * p), kernel_size=p, stride=p)
            return x.clamp(0.0, 1.0)

        vae = self._ensure_image_vae(tokens_btd.device)
        lat = tokens_btd.reshape(bsz, gh, gw, dim).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            dec = vae.decode(lat / float(self.image_latent_scaling_factor)).sample
        return (dec / 2.0 + 0.5).clamp(0.0, 1.0)

    def _get_rgb_token_unet_from_model(self, model_ref: Optional[nn.Module] = None):
        mdl = self.model if model_ref is None else model_ref
        mod = getattr(mdl, "module", None)
        bridge = getattr(mdl, "rgb_token_unet", None)
        if bridge is None and mod is not None:
            bridge = getattr(mod, "rgb_token_unet", None)
        return bridge

    def _encode_pixels_rgb_unet_tokens(
        self,
        pixel_values: torch.Tensor,
        model_ref: Optional[nn.Module] = None,
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mdl = self.model if model_ref is None else model_ref
        mod = getattr(mdl, "module", None)
        encode_fn = getattr(mdl, "encode_rgb_to_tokens", None)
        if encode_fn is None and mod is not None:
            encode_fn = getattr(mod, "encode_rgb_to_tokens", None)
        if encode_fn is not None:
            tokens, context = encode_fn(
                pixel_values,
                class_labels=class_labels,
                timesteps=timesteps,
                cond_vec=cond_vec,
            )
            if isinstance(context, dict):
                gg = context.get("graph_grid_shape", None)
                if gg is not None and len(gg) == 2:
                    gh = int(gg[0])
                    gw = int(gg[1])
                    if gh > 0 and gw > 0:
                        self._image_runtime_grid_shape = (gh, gw)
            self._image_runtime_feature_dim = int(tokens.size(-1))
            return tokens, context

        bridge = self._get_rgb_token_unet_from_model(model_ref=model_ref)
        if bridge is None:
            raise RuntimeError("RGB token U-Net bridge is not enabled on the model")
        tokens, context = bridge.encode(pixel_values, cond=cond_vec)
        if isinstance(context, dict):
            gg = context.get("graph_grid_shape", None)
            if gg is not None and len(gg) == 2:
                gh = int(gg[0])
                gw = int(gg[1])
                if gh > 0 and gw > 0:
                    self._image_runtime_grid_shape = (gh, gw)
        self._image_runtime_feature_dim = int(tokens.size(-1))
        return tokens, context

    def _decode_rgb_unet_tokens_to_pixels(
        self,
        token_features: torch.Tensor,
        context: Dict[str, Any],
        model_ref: Optional[nn.Module] = None,
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mdl = self.model if model_ref is None else model_ref
        mod = getattr(mdl, "module", None)
        decode_fn = getattr(mdl, "decode_tokens_to_rgb", None)
        if decode_fn is None and mod is not None:
            decode_fn = getattr(mod, "decode_tokens_to_rgb", None)
        if decode_fn is not None:
            return decode_fn(
                token_features,
                context,
                class_labels=class_labels,
                timesteps=timesteps,
                cond_vec=cond_vec,
            )

        bridge = self._get_rgb_token_unet_from_model(model_ref=model_ref)
        if bridge is None:
            raise RuntimeError("RGB token U-Net bridge is not enabled on the model")
        return bridge.decode(token_features, context, cond=cond_vec)

    def _get_image_diffusion_params(self, device: torch.device, diffusion_steps: Optional[int] = None):
        steps = int(self.image_diffusion_steps if diffusion_steps is None else diffusion_steps)
        steps = max(2, int(steps))
        schedule = str(getattr(self, "image_diffusion_schedule", "linear")).lower()
        key = f"{str(device)}::{steps}::{schedule}"
        cached = self._image_diffusion_cache.get(key, None)
        if cached is not None:
            return cached
        if schedule == "cosine":
            s = 0.008
            tt = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
            abar = torch.cos(((tt + s) / (1.0 + s)) * (math.pi * 0.5)) ** 2
            abar = abar / abar[0].clamp(min=1e-8)
            betas = (1.0 - (abar[1:] / abar[:-1]).clamp(min=1e-8, max=1.0)).clamp(min=1e-6, max=0.999)
        else:
            betas = torch.linspace(
                float(self.image_diffusion_beta_start),
                float(self.image_diffusion_beta_end),
                steps,
                device=device,
                dtype=torch.float32,
            ).clamp(min=1e-6, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        cached = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
        }
        self._image_diffusion_cache[key] = cached
        return cached

    def _build_inference_timestep_indices(self, inference_steps: int, device: torch.device) -> torch.Tensor:
        train_steps = max(2, int(self.image_diffusion_steps))
        infer_steps = max(2, int(inference_steps))

        if bool(getattr(self, "image_sampling_respace_timesteps", True)):
            if infer_steps > train_steps:
                logger.warning(
                    "Requested inference steps (%d) exceed training diffusion steps (%d); clamping to training steps.",
                    int(infer_steps),
                    int(train_steps),
                )
                infer_steps = train_steps
            idx = torch.linspace(
                float(train_steps - 1),
                0.0,
                int(infer_steps),
                device=device,
                dtype=torch.float32,
            ).round().to(dtype=torch.long)
            idx = idx.clamp(min=0, max=int(train_steps - 1))
            if idx.numel() > 1:
                keep = torch.ones_like(idx, dtype=torch.bool)
                keep[1:] = idx[1:] != idx[:-1]
                idx = idx[keep]
            if idx.numel() == 0:
                idx = torch.tensor([int(train_steps - 1), 0], device=device, dtype=torch.long)
            else:
                if int(idx[0].item()) != int(train_steps - 1):
                    idx = torch.cat(
                        [torch.tensor([int(train_steps - 1)], device=device, dtype=torch.long), idx],
                        dim=0,
                    )
                if int(idx[-1].item()) != 0:
                    idx = torch.cat(
                        [idx, torch.tensor([0], device=device, dtype=torch.long)],
                        dim=0,
                    )
                if idx.numel() > 1:
                    keep = torch.ones_like(idx, dtype=torch.bool)
                    keep[1:] = idx[1:] != idx[:-1]
                    idx = idx[keep]
        else:
            idx = torch.arange(
                int(infer_steps - 1),
                -1,
                -1,
                device=device,
                dtype=torch.long,
            )
            idx = idx.clamp(min=0, max=int(train_steps - 1))

        log_key = (
            int(train_steps),
            int(infer_steps),
            bool(getattr(self, "image_sampling_respace_timesteps", True)),
            int(idx.numel()),
            int(idx[0].item()) if idx.numel() > 0 else -1,
            int(idx[-1].item()) if idx.numel() > 0 else -1,
        )
        if log_key not in self._image_sampling_schedule_logged_keys:
            logger.info(
                "Image sampling timestep schedule: train_steps=%d infer_steps=%d respace=%s used_steps=%d t_first=%d t_last=%d",
                int(train_steps),
                int(infer_steps),
                bool(getattr(self, "image_sampling_respace_timesteps", True)),
                int(idx.numel()),
                int(idx[0].item()) if idx.numel() > 0 else -1,
                int(idx[-1].item()) if idx.numel() > 0 else -1,
            )
            self._image_sampling_schedule_logged_keys.add(log_key)

        return idx

    def _sample_prev_from_x0_xt(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        alpha_bar_prev: torch.Tensor,
    ) -> torch.Tensor:
        bsz = int(x_t.size(0))
        if alpha_bar_t.dim() == 0:
            ab_t = alpha_bar_t.to(dtype=x_t.dtype).reshape(1).expand(bsz)
        else:
            ab_t = alpha_bar_t.reshape(-1).to(dtype=x_t.dtype)
            if int(ab_t.numel()) != bsz:
                ab_t = ab_t[:1].expand(bsz)

        if alpha_bar_prev.dim() == 0:
            ab_prev = alpha_bar_prev.to(dtype=x_t.dtype).reshape(1).expand(bsz)
        else:
            ab_prev = alpha_bar_prev.reshape(-1).to(dtype=x_t.dtype)
            if int(ab_prev.numel()) != bsz:
                ab_prev = ab_prev[:1].expand(bsz)

        view_shape = [bsz] + [1] * max(0, x_t.dim() - 1)
        ab_t_v = ab_t.view(*view_shape)
        ab_prev_v = ab_prev.view(*view_shape)

        ratio = (ab_t_v / ab_prev_v.clamp(min=1e-8)).clamp(min=1e-8, max=1.0)
        one_minus_ab_t = (1.0 - ab_t_v).clamp(min=1e-8)

        coeff_x0 = torch.sqrt(ab_prev_v) * (1.0 - ratio) / one_minus_ab_t
        coeff_xt = torch.sqrt(ratio) * (1.0 - ab_prev_v) / one_minus_ab_t
        mean = coeff_x0 * x0_pred + coeff_xt * x_t

        var = ((1.0 - ab_prev_v) / one_minus_ab_t * (1.0 - ratio)).clamp(min=0.0)
        if bool(getattr(self, "image_sampling_legacy_update", False)):
            noise = torch.randn_like(x_t)
            return torch.sqrt(ab_prev_v) * x0_pred + torch.sqrt((1.0 - ab_prev_v).clamp(min=1e-8)) * noise

        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var.clamp(min=1e-12)) * noise

    def _rgb_to_centered(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

    def _centered_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5

    def _diffusion_target_from_x0_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        alpha_bar_t: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(getattr(self, "image_diffusion_prediction_runtime", "epsilon")).lower()
        if mode == "epsilon":
            return noise
        if mode == "x0":
            return x0
        # v-pred: v = alpha * eps - sigma * x0
        bsz = int(x0.size(0))
        view_shape = [bsz] + [1] * max(0, x0.dim() - 1)
        ab = alpha_bar_t.reshape(bsz).to(dtype=x0.dtype).view(*view_shape)
        alpha = torch.sqrt(ab)
        sigma = torch.sqrt((1.0 - ab).clamp(min=1e-8))
        return alpha * noise - sigma * x0

    def _prediction_to_x0_eps(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        alpha_bar_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = str(getattr(self, "image_diffusion_prediction_runtime", "epsilon")).lower()
        if alpha_bar_t.dim() == 0:
            ab = alpha_bar_t.to(dtype=x_t.dtype).reshape(1)
            bsz = int(x_t.size(0))
            ab = ab.expand(bsz)
        else:
            ab = alpha_bar_t.reshape(-1).to(dtype=x_t.dtype)
        bsz = int(x_t.size(0))
        view_shape = [bsz] + [1] * max(0, x_t.dim() - 1)
        abv = ab.view(*view_shape)
        alpha = torch.sqrt(abv)
        sigma = torch.sqrt((1.0 - abv).clamp(min=1e-8))

        if mode == "x0":
            x0_pred = pred
            eps_pred = (x_t - alpha * x0_pred) / sigma.clamp(min=1e-8)
            return x0_pred, eps_pred
        if mode == "v":
            x0_pred = alpha * x_t - sigma * pred
            eps_pred = sigma * x_t + alpha * pred
            return x0_pred, eps_pred

        # epsilon
        eps_pred = pred
        x0_pred = (x_t - sigma * eps_pred) / alpha.clamp(min=1e-8)
        return x0_pred, eps_pred

    def _record_image_debug_stats(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor],
    ) -> None:
        pred_std = float(pred.detach().float().std().item()) if pred.numel() > 0 else 0.0
        target_std = float(target.detach().float().std().item()) if target.numel() > 0 else 0.0
        pred_mean = float(pred.detach().float().mean().item()) if pred.numel() > 0 else 0.0
        target_mean = float(target.detach().float().mean().item()) if target.numel() > 0 else 0.0
        t_mean = None
        if timesteps is not None and torch.is_tensor(timesteps) and timesteps.numel() > 0:
            t_mean = float(timesteps.detach().float().mean().item())

        targets = [self.model]
        model_mod = getattr(self.model, "module", None)
        if model_mod is not None:
            targets.append(model_mod)
        for tgt in targets:
            tgt._last_image_pred_std = pred_std
            tgt._last_image_target_std = target_std
            tgt._last_image_pred_mean = pred_mean
            tgt._last_image_target_mean = target_mean
            tgt._last_image_t_mean = t_mean

    def _shared_image_cond_vec(
        self,
        model_ref: Optional[nn.Module],
        batch_size: int,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Compute the class+timestep conditioning vector ONCE per step so the RGB bridge
        encode, the core forward, and the bridge decode all see the SAME vector. Without
        this each of the three calls sampled its own CFG class-drop mask in training:
        ~27% of samples got inconsistent conditioning across the modules and the fully
        unconditional configuration CFG needs was trained with p=drop^3 instead of drop."""
        mdl = self.model if model_ref is None else model_ref
        fn = getattr(mdl, "_compute_conditioning_vector", None)
        if fn is None:
            mod = getattr(mdl, "module", None)
            if mod is not None:
                fn = getattr(mod, "_compute_conditioning_vector", None)
        if fn is None:
            return None
        return fn(
            batch_size=int(batch_size),
            device=device,
            class_labels=class_labels,
            timesteps=timesteps,
        )

    def _val_diffusion_draws(
        self,
        batch_index: int,
        batch_size: int,
        noise_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic (t, noise) for validation batches: seeded on a CPU generator per
        batch index, so the validation diffusion loss is comparable across epochs/runs.
        Unseeded draws made best-checkpoint selection a lottery over timestep luck.

        If image_diffusion_val_fixed_timesteps is set, t is PINNED to those values (cycled
        deterministically across batch positions and batches) instead of drawn over the full
        range — a low-variance progress signal at chosen noise levels. Noise stays seeded."""
        g = torch.Generator()
        g.manual_seed(int(getattr(self, "seed", 42) or 42) * 1000003 + int(batch_index))
        fixed_ts = getattr(self, "image_diffusion_val_fixed_timesteps", None) or []
        if fixed_ts:
            # Cycle the fixed timesteps across (batch_index, position) so each is applied to an
            # equal share of val images; fully deterministic (no RNG on t) => minimal variance.
            fixed = torch.tensor([int(v) for v in fixed_ts], dtype=torch.long)
            start = (int(batch_index) * int(batch_size)) % int(fixed.numel())
            sel = (torch.arange(int(batch_size), dtype=torch.long) + start) % int(fixed.numel())
            t = fixed.index_select(0, sel).to(device)
        else:
            t = torch.randint(
                0, int(self.image_diffusion_steps), (int(batch_size),), generator=g, dtype=torch.long
            ).to(device)
        noise = torch.randn(tuple(noise_shape), generator=g, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        return t, noise

    def _weighted_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha_bar_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sq = (pred - target) ** 2
        per_sample = sq.reshape(int(sq.size(0)), -1).mean(dim=1)
        gamma = float(getattr(self, "image_diffusion_min_snr_gamma", 0.0))
        if gamma > 0.0 and alpha_bar_t is not None:
            ab = alpha_bar_t.reshape(-1).to(per_sample.dtype)
            snr = ab / (1.0 - ab).clamp(min=1e-8)
            gamma_t = torch.full_like(snr, float(gamma))
            weights = torch.minimum(snr, gamma_t) / snr.clamp(min=1e-8)
            per_sample = per_sample * weights
        return per_sample.mean()

    def _fit_pred_to_target(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        context: str = "diffusion",
    ) -> torch.Tensor:
        if pred.shape == target.shape:
            return pred

        msg = (
            f"{context} shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
        if bool(getattr(self, "image_diffusion_strict_shapes", True)):
            raise RuntimeError(msg)

        if not bool(getattr(self, "_image_shape_mismatch_warned", False)):
            logger.warning("%s; cropping to shared minimum shape", msg)
            self._image_shape_mismatch_warned = True
        bsz = min(int(pred.size(0)), int(target.size(0)))
        tok = min(int(pred.size(1)), int(target.size(1)))
        dim = min(int(pred.size(2)), int(target.size(2)))
        return pred[:bsz, :tok, :dim]

    def _set_image_loss_stats(self, objective_loss_value: Optional[float]) -> None:
        targets = [self.model]
        model_mod = getattr(self.model, "module", None)
        if model_mod is not None:
            targets.append(model_mod)
        for tgt in targets:
            tgt._last_ce_loss = None
            tgt._last_copy_dst_ce_loss = None
            tgt._last_copy_dst_token_count = 0
            tgt._last_autoenc_ce_loss = None
            tgt._last_autoenc_next_ce_loss = None
            tgt._last_token_unet_lookahead_ce_loss = None
            tgt._last_objective_loss = None if objective_loss_value is None else float(objective_loss_value)

    def _prepare_image_batch_features(self, batch: Dict[str, Any], device: torch.device):
        if not isinstance(batch, dict):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                batch = {"pixel_values": batch[0], "class_labels": batch[1]}
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                batch = {"pixel_values": batch[0]}
            else:
                raise ValueError(f"Unsupported image batch format: {type(batch)}")

        class_labels = batch.get("class_labels", None)
        if class_labels is not None:
            class_labels = class_labels.to(device=device, dtype=torch.long)

        if batch.get("input_features", None) is not None:
            x0 = batch["input_features"].to(device)
            grid_shape = batch.get("grid_shape", None)
            if grid_shape is None:
                grid_shape = self._image_expected_grid_shape()
            self._image_runtime_grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
            self._image_runtime_feature_dim = int(x0.size(-1))
            return x0, self._image_runtime_grid_shape, class_labels

        pixel_values = batch.get("pixel_values", None)
        if pixel_values is None:
            raise ValueError("Image modality expects batch['pixel_values'] or batch['input_features']")
        pixel_values = pixel_values.to(device)
        x0, grid_shape = self._image_tokens_from_pixels(pixel_values)
        self._image_runtime_grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self._image_runtime_feature_dim = int(x0.size(-1))
        return x0, self._image_runtime_grid_shape, class_labels

    def _get_image_maskgit_vq_tokenizer(self, device: Optional[torch.device] = None):
        if str(getattr(self, "image_maskgit_variant", "continuous")).lower() != "discrete":
            return None

        tok = getattr(self, "_image_maskgit_vq_tokenizer", None)
        target_device = self.device if device is None else torch.device(device)
        if tok is None:
            vq_model_name = str(getattr(self, "image_maskgit_vq_model_name", "")).strip()
            if not vq_model_name:
                raise ValueError("image_maskgit_vq_model_name is required for image_maskgit_variant=discrete")
            from pinball.model.image_maskgit_vq import ImageMaskGITVQTokenizer

            tok = ImageMaskGITVQTokenizer.from_pretrained(vq_model_name, device=target_device)
            self._image_maskgit_vq_tokenizer = tok
            self.image_maskgit_codebook_size = int(tok.codebook_size)
            self.image_maskgit_mask_token_id = int(tok.mask_token_id)
        elif hasattr(tok, "to"):
            tok = tok.to(target_device)
            self._image_maskgit_vq_tokenizer = tok
        return tok

    def _prepare_image_batch_discrete_tokens(self, batch: Dict[str, Any], device: torch.device):
        if not isinstance(batch, dict):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                batch = {"pixel_values": batch[0], "class_labels": batch[1]}
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                batch = {"pixel_values": batch[0]}
            else:
                raise ValueError(f"Unsupported image batch format: {type(batch)}")

        class_labels = batch.get("class_labels", None)
        if class_labels is not None:
            class_labels = class_labels.to(device=device, dtype=torch.long)

        cached_token_ids = batch.get("token_ids", None)
        if cached_token_ids is not None:
            grid_shape = batch.get("grid_shape", None)
            if grid_shape is None:
                grid_shape = self._image_expected_grid_shape()
            self._image_runtime_grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
            if self.image_maskgit_codebook_size <= 0:
                vq_tokenizer = self._get_image_maskgit_vq_tokenizer(device=device)
                self._image_runtime_feature_dim = int(vq_tokenizer.vocab_size)
                self.image_maskgit_codebook_size = int(vq_tokenizer.codebook_size)
                self.image_maskgit_mask_token_id = int(vq_tokenizer.mask_token_id)
            else:
                self._image_runtime_feature_dim = int(self.image_maskgit_codebook_size) + 1
            return cached_token_ids.to(device=device, dtype=torch.long), self._image_runtime_grid_shape, class_labels

        pixel_values = batch.get("pixel_values", None)
        if pixel_values is None:
            raise ValueError("Discrete image MaskGIT expects batch['pixel_values'] or batch['token_ids']")

        vq_tokenizer = self._get_image_maskgit_vq_tokenizer(device=device)
        token_ids, grid_shape = vq_tokenizer.encode(pixel_values.to(device))
        self._image_runtime_grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self._image_runtime_feature_dim = int(vq_tokenizer.vocab_size)
        self.image_maskgit_codebook_size = int(vq_tokenizer.codebook_size)
        self.image_maskgit_mask_token_id = int(vq_tokenizer.mask_token_id)
        return token_ids, self._image_runtime_grid_shape, class_labels

    def _image_maskgit_decode_discrete_tokens(self, token_ids: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        vq_tokenizer = self._get_image_maskgit_vq_tokenizer(device=token_ids.device)
        return vq_tokenizer.decode(token_ids, grid_shape)

    def _fit_logits_to_target_ids(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        context: str = "image_maskgit_discrete",
    ) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(f"{context} logits must be [B,T,V], got {tuple(logits.shape)}")
        if target_ids.dim() != 2:
            raise ValueError(f"{context} target_ids must be [B,T], got {tuple(target_ids.shape)}")

        if int(logits.size(0)) == int(target_ids.size(0)) and int(logits.size(1)) == int(target_ids.size(1)):
            return logits

        msg = f"{context} shape mismatch: logits={tuple(logits.shape)} target_ids={tuple(target_ids.shape)}"
        if bool(getattr(self, "image_diffusion_strict_shapes", True)):
            raise RuntimeError(msg)

        if not bool(getattr(self, "_image_shape_mismatch_warned", False)):
            logger.warning("%s; cropping to shared minimum shape", msg)
            self._image_shape_mismatch_warned = True
        bsz = min(int(logits.size(0)), int(target_ids.size(0)))
        tok = min(int(logits.size(1)), int(target_ids.size(1)))
        return logits[:bsz, :tok, :]

    def _masked_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        mask: torch.Tensor,
        context: str = "image_maskgit_discrete",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        logits_fit = self._fit_logits_to_target_ids(logits, target_ids, context=context)
        bsz = min(int(logits_fit.size(0)), int(target_ids.size(0)), int(mask.size(0)))
        tok = min(int(logits_fit.size(1)), int(target_ids.size(1)), int(mask.size(1)))
        if bsz <= 0 or tok <= 0:
            return logits_fit.new_zeros(())

        logits_fit = logits_fit[:bsz, :tok, :]
        target_fit = target_ids[:bsz, :tok].to(device=logits_fit.device, dtype=torch.long)
        mask_fit = mask[:bsz, :tok].to(device=logits_fit.device, dtype=torch.bool)
        if not bool(mask_fit.any()):
            return logits_fit.new_zeros(())

        token_loss = F.cross_entropy(logits_fit.reshape(-1, logits_fit.size(-1)), target_fit.reshape(-1), reduction="none", label_smoothing=label_smoothing).view_as(target_fit)
        selected = token_loss.masked_select(mask_fit)
        if selected.numel() == 0:
            return logits_fit.new_zeros(())
        return selected.mean()

    def _sample_image_feature_mask(
        self,
        token_count: int,
        grid_shape: Optional[Tuple[int, int]],
        mask_ratio: float,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        mode = str(getattr(self, "diffusion_mask_mode", "random")).lower()
        if mode not in {"random", "block", "path"}:
            mode = "random"

        ratio = max(0.0, min(1.0, float(mask_ratio)))
        bsz = max(1, int(batch_size))
        tok = max(0, int(token_count))
        if tok <= 0:
            return torch.zeros((bsz, 0), dtype=torch.bool, device=device)
        if ratio <= 0.0:
            return torch.zeros((bsz, tok), dtype=torch.bool, device=device)

        inferred = _infer_grid_shape(tok, grid_shape=grid_shape)
        if inferred is None or mode == "random":
            mask = torch.rand((bsz, tok), device=device) < ratio
        else:
            gh, gw = inferred
            if mode == "block":
                mask = _sample_block_mask_2d(
                    bsz=bsz,
                    gh=int(gh),
                    gw=int(gw),
                    mask_prob=ratio,
                    block_size=int(getattr(self, "diffusion_mask_block_size", 4)),
                    device=device,
                )
            else:
                mask = _sample_path_mask_2d(
                    bsz=bsz,
                    gh=int(gh),
                    gw=int(gw),
                    mask_prob=ratio,
                    path_len=int(getattr(self, "diffusion_mask_path_length", 64)),
                    device=device,
                )

        if mask.dim() != 2:
            return torch.rand((bsz, tok), device=device) < ratio
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any()):
            row_idx = torch.nonzero(empty_rows, as_tuple=False).view(-1)
            col_idx = torch.randint(0, tok, (int(row_idx.numel()),), device=device)
            mask = mask.clone()
            mask[row_idx, col_idx] = True
        return mask

    def _maskgit_time_condition(self, mask_ratio: float, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not bool(getattr(self, "image_maskgit_use_timestep_cond", False)):
            return None
        steps = max(2, int(getattr(self, "image_diffusion_steps", 1000)))
        t = int(round(max(0.0, min(1.0, float(mask_ratio))) * float(steps - 1)))
        return torch.full((max(1, int(batch_size)),), int(t), device=device, dtype=torch.long)

    def _get_mask_token_vector(self, num: int, feat_dim: int, device: torch.device) -> torch.Tensor:
        """
        Returns a [num, feat_dim] tensor of the learned mask token vector.
        Priority:
        1. model.mask_vector (nn.Parameter, feature mode)
        2. model.token_embedding(mask_token_id) (token mode)
        3. zeros fallback
        """
        model_ref = self.model
        model_mod = getattr(model_ref, "module", None)
        mdl = model_mod if model_mod is not None else model_ref

        mask_vec = None

        if hasattr(mdl, "mask_vector"):
            mv = mdl.mask_vector
            if isinstance(mv, nn.Parameter):
                mask_vec = mv.data.view(1, -1)
            elif isinstance(mv, torch.Tensor):
                mask_vec = mv.view(1, -1)

        if mask_vec is None and hasattr(mdl, "token_embedding"):
            tok_emb = mdl.token_embedding
            mask_token_id = getattr(mdl, "mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = getattr(self, "tokenizer", None)
                if mask_token_id is not None:
                    mask_token_id = getattr(mask_token_id, "mask_token_id", None)
            if mask_token_id is not None:
                with torch.no_grad():
                    mask_vec = tok_emb(torch.as_tensor([[mask_token_id]], device=device)).view(1, -1)

        if mask_vec is None:
            logger.warning("_get_mask_token_vector: no learned mask vector found; using zeros")
            mask_vec = torch.zeros(1, feat_dim, device=device)

        if mask_vec.shape[-1] != feat_dim:
            logger.warning(
                "_get_mask_token_vector: mask_vec dim=%d != feat_dim=%d; using zeros",
                int(mask_vec.shape[-1]),
                int(feat_dim),
            )
            mask_vec = torch.zeros(1, feat_dim, device=device)

        return mask_vec.expand(num, feat_dim)

    def _masked_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        context: str = "image_maskgit",
    ) -> torch.Tensor:
        pred_fit = self._fit_pred_to_target(pred, target, context=context)
        bsz = min(int(pred_fit.size(0)), int(target.size(0)), int(mask.size(0)))
        tok = min(int(pred_fit.size(1)), int(target.size(1)), int(mask.size(1)))
        dim = min(int(pred_fit.size(2)), int(target.size(2)))
        if bsz <= 0 or tok <= 0 or dim <= 0:
            return pred_fit.new_zeros(())

        pred_fit = pred_fit[:bsz, :tok, :dim]
        target_fit = target[:bsz, :tok, :dim]
        mask_fit = mask[:bsz, :tok].to(device=pred_fit.device, dtype=torch.bool)
        if not bool(mask_fit.any()):
            return pred_fit.new_zeros(())

        mask_exp = mask_fit.unsqueeze(-1).expand_as(pred_fit)
        sq = (pred_fit - target_fit) ** 2
        selected = sq.masked_select(mask_exp)
        if selected.numel() == 0:
            return pred_fit.new_zeros(())
        return selected.mean()

    def _maskgit_gamma(self, progress: float, schedule: Optional[str] = None) -> float:
        sched = str(schedule or self.image_maskgit_schedule).lower()
        p = max(0.0, min(1.0, float(progress)))
        if sched == "linear":
            return max(0.0, 1.0 - p)
        return max(0.0, math.cos(0.5 * math.pi * p))

    def _maskgit_reveal_ratio(self, step_idx: int, total_steps: int) -> float:
        total = max(1, int(total_steps))
        frac = float(step_idx + 1) / float(total)
        return self._maskgit_gamma(frac)

    def _maskgit_temperature(self, step_idx: int, total_steps: int) -> float:
        if not bool(getattr(self, "image_maskgit_temperature_anneal", True)):
            return max(1e-6, float(getattr(self, "image_maskgit_temperature_start", 1.0)))

        total = max(1, int(total_steps) - 1)
        frac = max(0.0, min(1.0, float(step_idx) / float(total)))
        start = max(1e-6, float(getattr(self, "image_maskgit_temperature_start", 1.0)))
        end = max(1e-6, float(getattr(self, "image_maskgit_temperature_end", 0.1)))
        # Cosine anneal from start -> end.
        return max(1e-6, end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac)))

    def _process_batch_with_image_maskgit(self, batch, use_bf16=True):
        device = self.device
        amp_on = bool(self.mixed_precision and torch.cuda.is_available() and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if bool(use_bf16) else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else nullcontext()

        x0, grid_shape, class_labels = self._prepare_image_batch_features(batch, device=device)
        bsz, tok, _ = x0.shape

        mask_min = float(getattr(self, "image_maskgit_train_mask_min", 0.10))
        mask_max = float(getattr(self, "image_maskgit_train_mask_max", 1.00))
        if mask_max < mask_min:
            mask_max = mask_min
        mask_ratio = float(torch.empty(1, device=x0.device).uniform_(mask_min, mask_max).item())
        mask = self._sample_image_feature_mask(
            token_count=tok,
            grid_shape=grid_shape,
            mask_ratio=mask_ratio,
            batch_size=bsz,
            device=x0.device,
        )

        mask_fill_mode = str(getattr(self, "image_maskgit_mask_value", "zero")).lower()
        if mask_fill_mode == "noise":
            mask_fill = torch.randn_like(x0)
        elif mask_fill_mode == "zero":
            mask_fill = torch.zeros_like(x0)
        x_in = torch.where(mask.unsqueeze(-1), mask_fill, x0)

        timesteps = self._maskgit_time_condition(mask_ratio=mask_ratio, batch_size=bsz, device=x0.device)
        attention_mask = torch.ones((bsz, tok), device=x0.device, dtype=torch.long)

        with amp_ctx:
            pred = self.model(
                x_in,
                attention_mask=attention_mask,
                class_labels=class_labels,
                timesteps=timesteps,
            )

        pred_fit = self._fit_pred_to_target(pred, x0, context="train/image_maskgit")
        target_fit = x0[: pred_fit.size(0), : pred_fit.size(1), : pred_fit.size(2)]
        mask_fit = mask[: pred_fit.size(0), : pred_fit.size(1)]
        self._record_image_debug_stats(pred_fit, target_fit, timesteps if timesteps is not None else None)

        loss_masked = self._masked_mse_loss(pred, x0, mask, context="train/image_maskgit_masked")
        loss = loss_masked
        unmasked_weight = float(getattr(self, "image_maskgit_unmasked_weight", 0.0))
        if unmasked_weight > 0.0:
            loss_unmasked = self._masked_mse_loss(pred, x0, ~mask, context="train/image_maskgit_unmasked")
            loss = loss + (unmasked_weight * loss_unmasked)

        model_ref = self.model
        model_mod = getattr(model_ref, "module", None)
        aux = getattr(model_ref, "_last_hier_aux_loss", None)
        if aux is None and model_mod is not None:
            aux = getattr(model_mod, "_last_hier_aux_loss", None)
        lambda_hier_aux = getattr(model_ref, "lambda_hier_aux", None)
        if lambda_hier_aux is None and model_mod is not None:
            lambda_hier_aux = getattr(model_mod, "lambda_hier_aux", None)
        if aux is not None and lambda_hier_aux is not None:
            loss = loss + float(lambda_hier_aux) * aux.mean()

        self._set_image_loss_stats(float(loss.detach().item()))
        scaled_loss = loss / max(1, int(self.gradient_accumulation_steps))
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        return float(loss.detach().item())

    def _process_batch_with_image_maskgit_discrete(self, batch, use_bf16=True):
        device = self.device
        amp_on = bool(self.mixed_precision and torch.cuda.is_available() and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if bool(use_bf16) else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else nullcontext()

        token_ids, grid_shape, class_labels = self._prepare_image_batch_discrete_tokens(batch, device=device)
        bsz, tok = int(token_ids.size(0)), int(token_ids.size(1))

        mask_min = float(getattr(self, "image_maskgit_train_mask_min", 0.00))
        mask_max = float(getattr(self, "image_maskgit_train_mask_max", 1.00))
        if mask_max < mask_min:
            mask_max = mask_min

        progress = torch.rand((bsz,), device=token_ids.device)
        if str(self.image_maskgit_schedule).lower() == "linear":
            mask_ratios = (1.0 - progress).clamp(min=0.0)
        else:
            mask_ratios = torch.cos(0.5 * math.pi * progress).clamp(min=0.0)
        mask_ratios = mask_ratios.clamp(min=mask_min, max=mask_max)

        mode = str(getattr(self, "diffusion_mask_mode", "random")).lower()
        if mode in {"block", "path"}:
            # Structured masks take one scalar ratio per call -> per-sample loop.
            masks = []
            for ratio in mask_ratios.tolist():
                m = self._sample_image_feature_mask(
                    token_count=tok,
                    grid_shape=grid_shape,
                    mask_ratio=float(ratio),
                    batch_size=1,
                    device=token_ids.device,
                )[0]
                masks.append(m)
            mask = torch.stack(masks, dim=0)
        else:
            # Vectorized per-sample-ratio Bernoulli masks. The per-sample loop above
            # costs ~95ms/step in Python/sync overhead at B=2048 (0.1ms vectorized);
            # same distribution: Bernoulli(ratio_b) per token + >=1 masked per row.
            mask = torch.rand((bsz, tok), device=token_ids.device) < mask_ratios.unsqueeze(1)
            rows = torch.arange(bsz, device=token_ids.device)
            fill_cols = torch.randint(0, tok, (bsz,), device=token_ids.device)
            empty = ~mask.any(dim=1)
            mask[rows, fill_cols] = mask[rows, fill_cols] | empty

        mask_token_id = int(getattr(self, "image_maskgit_mask_token_id", -1))
        if mask_token_id < 0:
            vq_tokenizer = self._get_image_maskgit_vq_tokenizer(device=device)
            mask_token_id = int(vq_tokenizer.mask_token_id)

        input_ids = token_ids.clone()
        input_ids[mask] = int(mask_token_id)
        attention_mask = torch.ones((bsz, tok), device=token_ids.device, dtype=torch.long)

        with amp_ctx:
            logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                class_labels=class_labels,
            )

        logits_fit = self._fit_logits_to_target_ids(logits, token_ids, context="train/image_maskgit_discrete")
        token_ids_fit = token_ids[: logits_fit.size(0), : logits_fit.size(1)]
        mask_fit = mask[: logits_fit.size(0), : logits_fit.size(1)]

        with torch.no_grad():
            preds_diag = torch.argmax(logits_fit, dim=-1)
            masked_preds = preds_diag[mask_fit]
            unique_ratio = float(torch.unique(masked_preds).size(0)) / float(max(1, masked_preds.numel()))
            mask_ratio_val = float(mask_fit.sum().item()) / max(1, mask_fit.numel())
            mean_progress = float(progress.mean().item()) if progress.numel() > 0 else 0.0
            if unique_ratio < 0.01:
                if self._collapse_warning_cooldown <= 0:
                    bucket = int(round(float(mask_ratio_val) * 4))
                    logger.warning(
                        "TOKEN COLLAPSE WARNING: unique=%.6f (%d/%d masked). "
                        "gamma=%.3f progress=%.3f bucket=%d mask_ratio=%.2f batch=%d. "
                        "This may indicate mode collapse.",
                        float(unique_ratio), torch.unique(masked_preds).size(0), masked_preds.numel(),
                        float(mask_ratio_val), mean_progress, bucket,
                        float(mask_ratio_val), mask_fit.size(0)
                    )
                    self._collapse_warning_cooldown = max(1, int(getattr(self, "log_interval", 10)))
                else:
                    self._collapse_warning_cooldown -= 1

        loss_masked = self._masked_cross_entropy_loss(logits_fit, token_ids_fit, mask_fit, context="train/image_maskgit_discrete_masked", label_smoothing=self.ce_label_smoothing_train)
        loss = loss_masked
        unmasked_weight = float(getattr(self, "image_maskgit_unmasked_weight", 0.0))
        if unmasked_weight > 0.0:
            loss_unmasked = self._masked_cross_entropy_loss(logits_fit, token_ids_fit, ~mask_fit, context="train/image_maskgit_discrete_unmasked", label_smoothing=self.ce_label_smoothing_train)
            loss = loss + (unmasked_weight * loss_unmasked)

        model_ref = self.model
        model_mod = getattr(model_ref, "module", None)
        aux = getattr(model_ref, "_last_hier_aux_loss", None)
        if aux is None and model_mod is not None:
            aux = getattr(model_mod, "_last_hier_aux_loss", None)
        lambda_hier_aux = getattr(model_ref, "lambda_hier_aux", None)
        if lambda_hier_aux is None and model_mod is not None:
            lambda_hier_aux = getattr(model_mod, "lambda_hier_aux", None)
        if aux is not None and lambda_hier_aux is not None:
            loss = loss + float(lambda_hier_aux) * aux.mean()

        self._set_image_loss_stats(float(loss.detach().item()))
        scaled_loss = loss / max(1, int(self.gradient_accumulation_steps))
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        return float(loss.detach().item())

    def _process_batch_with_image_diffusion(self, batch, use_bf16=True):
        device = self.device
        amp_on = bool(self.mixed_precision and torch.cuda.is_available() and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if bool(use_bf16) else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else nullcontext()

        if self.image_token_mode == "rgb_unet":
            if not isinstance(batch, dict):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    batch = {"pixel_values": batch[0], "class_labels": batch[1]}
                elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    batch = {"pixel_values": batch[0]}
                else:
                    raise ValueError(f"Unsupported image batch format: {type(batch)}")
            pixel_values = batch.get("pixel_values", None)
            if pixel_values is None:
                raise ValueError("rgb_unet image mode expects batch['pixel_values']")
            pixel_values = pixel_values.to(device)
            class_labels = batch.get("class_labels", None)
            if class_labels is not None:
                class_labels = class_labels.to(device=device, dtype=torch.long)

            if self.image_diffusion_target_runtime == "rgb_epsilon":
                bsz = int(pixel_values.size(0))
                sched = self._get_image_diffusion_params(device=pixel_values.device)
                alpha_bars = sched["alpha_bars"]
                t = torch.randint(0, int(self.image_diffusion_steps), (bsz,), device=pixel_values.device, dtype=torch.long)
                ab = alpha_bars.index_select(0, t).view(bsz, 1, 1, 1).to(dtype=pixel_values.dtype)
                x0_rgb = self._rgb_to_centered(pixel_values) if bool(self.image_rgb_centered_diffusion) else pixel_values
                noise_rgb = torch.randn_like(x0_rgb)
                x_t_rgb = torch.sqrt(ab) * x0_rgb + torch.sqrt(1.0 - ab) * noise_rgb

                with amp_ctx:
                    # ONE conditioning vector for encode + core + decode (single CFG drop).
                    cond_vec = self._shared_image_cond_vec(
                        model_ref=self.model,
                        batch_size=bsz,
                        device=pixel_values.device,
                        class_labels=class_labels,
                        timesteps=t,
                    )
                    x_t_tokens, decode_context = self._encode_pixels_rgb_unet_tokens(
                        x_t_rgb,
                        model_ref=self.model,
                        class_labels=class_labels,
                        timesteps=t,
                        cond_vec=cond_vec,
                    )
                    tok = int(x_t_tokens.size(1))
                    attention_mask = torch.ones((bsz, tok), device=x_t_tokens.device, dtype=torch.long)
                    pred_tokens = self.model(
                        x_t_tokens,
                        attention_mask=attention_mask,
                        class_labels=class_labels,
                        timesteps=t,
                        cond_vec=cond_vec,
                    )
                    pred_tokens_fit = self._fit_pred_to_target(
                        pred_tokens,
                        x_t_tokens,
                        context="train/rgb_unet_rgb_epsilon_tokens",
                    )
                    pred_eps_rgb = self._decode_rgb_unet_tokens_to_pixels(
                        pred_tokens_fit,
                        decode_context,
                        model_ref=self.model,
                        class_labels=class_labels,
                        timesteps=t,
                        cond_vec=cond_vec,
                    )
                    pr = pred_eps_rgb
                    tg = self._diffusion_target_from_x0_noise(x0=x0_rgb, noise=noise_rgb, alpha_bar_t=ab.view(bsz))
                    if pr.shape != tg.shape:
                        msg = f"train/rgb_unet_rgb_epsilon_pixels shape mismatch: pred={tuple(pr.shape)} target={tuple(tg.shape)}"
                        if bool(getattr(self, "image_diffusion_strict_shapes", True)):
                            raise RuntimeError(msg)
                        if not bool(getattr(self, "_image_shape_mismatch_warned", False)):
                            logger.warning("%s; cropping to shared minimum spatial shape", msg)
                            self._image_shape_mismatch_warned = True
                        h = min(int(pr.size(-2)), int(tg.size(-2)))
                        w = min(int(pr.size(-1)), int(tg.size(-1)))
                        pr = pr[:, :, :h, :w]
                        tg = tg[:, :, :h, :w]
                    self._record_image_debug_stats(pr, tg, t)
                    loss = self._weighted_mse_loss(pr, tg, alpha_bar_t=ab.view(bsz))
            else:
                x0, _ = self._encode_pixels_rgb_unet_tokens(pixel_values, model_ref=self.model)
                bsz, tok, _ = x0.shape
                sched = self._get_image_diffusion_params(device=x0.device)
                alpha_bars = sched["alpha_bars"]
                t = torch.randint(0, int(self.image_diffusion_steps), (bsz,), device=x0.device, dtype=torch.long)
                ab = alpha_bars.index_select(0, t).view(bsz, 1, 1).to(dtype=x0.dtype)
                noise = torch.randn_like(x0)
                x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
                attention_mask = torch.ones((bsz, tok), device=x0.device, dtype=torch.long)

                with amp_ctx:
                    pred = self.model(
                        x_t,
                        attention_mask=attention_mask,
                        class_labels=class_labels,
                        timesteps=t,
                    )
                    target = self._diffusion_target_from_x0_noise(x0=x0, noise=noise, alpha_bar_t=ab.view(bsz))
                    pred_fit = self._fit_pred_to_target(
                        pred,
                        target,
                        context="train/rgb_unet_token_space",
                    )
                    target_fit = target[: pred_fit.size(0), : pred_fit.size(1), : pred_fit.size(2)]
                    self._record_image_debug_stats(pred_fit, target_fit, t)
                    loss = self._weighted_mse_loss(pred_fit, target_fit, alpha_bar_t=ab.view(bsz))
        else:
            x0, grid_shape, class_labels = self._prepare_image_batch_features(batch, device=device)
            bsz, tok, _ = x0.shape
            sched = self._get_image_diffusion_params(device=x0.device)
            alpha_bars = sched["alpha_bars"]
            t = torch.randint(0, int(self.image_diffusion_steps), (bsz,), device=x0.device, dtype=torch.long)
            ab = alpha_bars.index_select(0, t).view(bsz, 1, 1).to(dtype=x0.dtype)
            noise = torch.randn_like(x0)
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
            attention_mask = torch.ones((bsz, tok), device=x0.device, dtype=torch.long)

            with amp_ctx:
                pred = self.model(
                    x_t,
                    attention_mask=attention_mask,
                    class_labels=class_labels,
                    timesteps=t,
                )
                target = self._diffusion_target_from_x0_noise(x0=x0, noise=noise, alpha_bar_t=ab.view(bsz))
                pred_fit = self._fit_pred_to_target(
                    pred,
                    target,
                    context="train/image_latent_or_patch",
                )
                target_fit = target[: pred_fit.size(0), : pred_fit.size(1), : pred_fit.size(2)]
                self._record_image_debug_stats(pred_fit, target_fit, t)
                loss = self._weighted_mse_loss(pred_fit, target_fit, alpha_bar_t=ab.view(bsz))

        model_ref = self.model
        model_mod = getattr(model_ref, "module", None)
        aux = getattr(model_ref, "_last_hier_aux_loss", None)
        if aux is None and model_mod is not None:
            aux = getattr(model_mod, "_last_hier_aux_loss", None)
        lambda_hier_aux = getattr(model_ref, "lambda_hier_aux", None)
        if lambda_hier_aux is None and model_mod is not None:
            lambda_hier_aux = getattr(model_mod, "lambda_hier_aux", None)
        if aux is not None and lambda_hier_aux is not None:
            loss = loss + float(lambda_hier_aux) * aux.mean()

        self._set_image_loss_stats(float(loss.detach().item()))
        scaled_loss = loss / max(1, int(self.gradient_accumulation_steps))
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        return float(loss.detach().item())

    def _validate_image_maskgit_discrete(self, data_provider, use_ema=False):
        model_to_use = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model_to_use.eval()

        if isinstance(data_provider, dict) and "get_batch" in data_provider:
            get_batch = data_provider["get_batch"]
            steps = int(data_provider.get("steps", 50))
            data_iterator = range(steps)
            use_get_batch = True
            desc = "Validation (Image MaskGIT-Discrete Steps)"
        elif isinstance(data_provider, torch.utils.data.DataLoader):
            dataloader = data_provider
            steps = len(dataloader)
            data_iterator = dataloader
            use_get_batch = False
            desc = "Validation (Image MaskGIT-Discrete Batches)"
        else:
            raise ValueError("Invalid data_provider format.")

        pbar = tqdm(data_iterator, desc=desc, total=steps, dynamic_ncols=True)
        total_loss = 0.0
        total_masked_correct = 0
        total_masked_count = 0
        total_unmasked_correct = 0
        total_unmasked_count = 0
        denom = 0
        total_predicted_unique = 0
        total_predicted_count = 0
        val_ratio = float(getattr(self, "image_maskgit_val_mask_ratio", 0.5))
        amp_on = bool(self.mixed_precision and torch.cuda.is_available() and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if bool(self.use_bf16) else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else nullcontext()

        mask_token_id = int(getattr(self, "image_maskgit_mask_token_id", -1))
        if mask_token_id < 0:
            vq_tokenizer = self._get_image_maskgit_vq_tokenizer(device=self.device)
            mask_token_id = int(vq_tokenizer.mask_token_id)

        with torch.no_grad():
            for _, batch_data in enumerate(pbar):
                batch = get_batch(self.device) if use_get_batch else batch_data
                token_ids, grid_shape, class_labels = self._prepare_image_batch_discrete_tokens(batch, device=self.device)
                bsz, tok = int(token_ids.size(0)), int(token_ids.size(1))
                mask = self._sample_image_feature_mask(
                    token_count=tok,
                    grid_shape=grid_shape,
                    mask_ratio=val_ratio,
                    batch_size=bsz,
                    device=token_ids.device,
                )

                input_ids = token_ids.clone()
                input_ids[mask] = int(mask_token_id)
                attention_mask = torch.ones((bsz, tok), device=token_ids.device, dtype=torch.long)

                with amp_ctx:
                    logits = model_to_use(
                        input_ids,
                        attention_mask=attention_mask,
                        class_labels=class_labels,
                    )

                logits_fit = self._fit_logits_to_target_ids(logits, token_ids, context="val/image_maskgit_discrete")
                if int(mask_token_id) >= 0 and int(mask_token_id) < int(logits_fit.size(-1)):
                    logits_fit = logits_fit.clone()
                    logits_fit[..., int(mask_token_id)] = torch.finfo(logits_fit.dtype).min
                token_ids_fit = token_ids[: logits_fit.size(0), : logits_fit.size(1)]
                mask_fit = mask[: logits_fit.size(0), : logits_fit.size(1)]

                token_loss = F.cross_entropy(logits_fit.reshape(-1, logits_fit.size(-1)), token_ids_fit.reshape(-1), reduction="none").view_as(token_ids_fit)
                masked_loss = token_loss.masked_select(mask_fit)
                masked_loss_val = float(masked_loss.mean().item()) if masked_loss.numel() > 0 else 0.0
                loss = masked_loss.mean() if masked_loss.numel() > 0 else logits_fit.new_zeros(())

                unmasked_weight = float(getattr(self, "image_maskgit_unmasked_weight", 0.0))
                if unmasked_weight > 0.0:
                    unmasked_loss = token_loss.masked_select(~mask_fit)
                    loss_unmasked = unmasked_loss.mean() if unmasked_loss.numel() > 0 else logits_fit.new_zeros(())
                    loss = loss + (unmasked_weight * loss_unmasked)

                preds = torch.argmax(logits_fit, dim=-1)
                masked_correct = ((preds == token_ids_fit) & mask_fit).sum().item()
                masked_count = int(mask_fit.sum().item())
                unmasked_correct = ((preds == token_ids_fit) & (~mask_fit)).sum().item()
                unmasked_count = int((~mask_fit).sum().item())
                total_masked_correct += int(masked_correct)
                total_masked_count += int(masked_count)
                total_unmasked_correct += int(unmasked_correct)
                total_unmasked_count += int(unmasked_count)

                total_loss += float(loss.item())
                denom += 1
                # Count unique tokens ONLY at masked positions for validation metric
                masked_preds = preds[mask_fit]
                total_predicted_unique += int(torch.unique(masked_preds).size(0))
                total_predicted_count += int(masked_preds.numel())
                pbar.set_postfix({"val_img_maskgit_discrete": float(loss.item())})

        pbar.close()

        if denom <= 0:
            metrics = {
                "loss": float("inf"),
                "objective_loss": float("inf"),
                "masked_loss": float("inf"),
                "next_token_loss": float("inf"),
                "masked_acc": 0.0,
                "next_token_acc": 0.0,
                "perplexity": float("inf"),
                "image_maskgit_loss": float("inf"),
                "image_diffusion_loss": float("inf"),
                "selection_metric": "image_maskgit_loss",
                "selection_loss": float("inf"),
            }
            return float("inf"), metrics

        avg = total_loss / float(denom)
        masked_acc = float(total_masked_correct) / float(max(1, total_masked_count))
        perplexity = float(math.exp(min(20.0, float(avg)))) if math.isfinite(float(avg)) else float("inf")
        token_unique_ratio = float(total_predicted_unique) / float(max(1, total_predicted_count))
        metrics = {
            "loss": float(avg),
            "objective_loss": float(avg),
            "masked_loss": float(avg),
            "next_token_loss": float(avg),
            "masked_acc": float(masked_acc),
            "next_token_acc": float(float(total_unmasked_correct) / float(max(1, total_unmasked_count))),
            "perplexity": float(perplexity),
            "token_unique_ratio": float(token_unique_ratio),
            "image_maskgit_loss": float(avg),
            "image_diffusion_loss": float(avg),
            "selection_metric": "image_maskgit_loss",
            "selection_loss": float(avg),
        }
        return float(avg), metrics

    @torch.no_grad()
    def _image_sampling_amp_ctx(self):
        """Autocast context for image sampling forwards.

        Sampling must run under the SAME mixed-precision autocast as training:
        the packed local-attention path hard-requires fp16/bf16 inputs and
        raises on a bare fp32 forward. Fresh context per call (the autocast
        weight-cast cache is cleared on exit, so no_grad casts cannot leak
        into later grad-mode training calls).
        """
        if bool(self.mixed_precision) and torch.cuda.is_available() and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=amp_dtype("cuda"))
        return nullcontext()

    @torch.no_grad()
    def _sample_image_tokens_maskgit_discrete(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        use_ema: bool = False,
        diffusion_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        model_for_gen = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model_for_gen.eval()
        steps = int(self.image_maskgit_steps if diffusion_steps is None else diffusion_steps)
        steps = max(2, int(steps))

        vq_tokenizer = self._get_image_maskgit_vq_tokenizer(device=self.device)
        gh, gw = self._image_runtime_grid_shape or vq_tokenizer.infer_grid_shape(self.image_size)
        self._image_runtime_grid_shape = (int(gh), int(gw))
        mask_token_id = int(getattr(self, "image_maskgit_mask_token_id", vq_tokenizer.mask_token_id))
        codebook_size = int(getattr(self, "image_maskgit_codebook_size", vq_tokenizer.codebook_size))
        n_tok = int(gh * gw)

        x = torch.full((int(batch_size), n_tok), int(mask_token_id), device=self.device, dtype=torch.long)
        attn = torch.ones((int(batch_size), n_tok), device=self.device, dtype=torch.long)
        cls = None
        if class_labels is not None:
            cls = class_labels.to(device=self.device, dtype=torch.long)
        mask = torch.ones((int(batch_size), n_tok), device=self.device, dtype=torch.bool)

        for step in range(steps):
            with self._image_sampling_amp_ctx():
                logits_cond = model_for_gen(x, attention_mask=attn, class_labels=cls)
            logits_cond = self._fit_logits_to_target_ids(logits_cond, x, context="sample/image_maskgit_discrete_cond").float()
            if cls is not None and guidance_scale > 0.0:
                null_cls = torch.full_like(cls, int(getattr(model_for_gen, "class_null_index", self.image_num_classes)))
                with self._image_sampling_amp_ctx():
                    logits_null = model_for_gen(x, attention_mask=attn, class_labels=null_cls)
                logits_null = self._fit_logits_to_target_ids(logits_null, x, context="sample/image_maskgit_discrete_null").float()
                logits = logits_null + float(guidance_scale) * (logits_cond - logits_null)
            else:
                logits = logits_cond

            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            if int(mask_token_id) >= 0 and int(mask_token_id) < int(logits.size(-1)):
                logits = logits.clone()
                logits[..., int(mask_token_id)] = torch.finfo(logits.dtype).min

            temperature = self._maskgit_temperature(step, steps)
            logits = logits / max(1e-6, float(temperature))
            probs = F.softmax(logits, dim=-1)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(int(x.size(0)), int(x.size(1)))
            sampled_conf = probs.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
            sampled_conf = sampled_conf.masked_fill(~mask, float("-inf"))

            for b in range(int(x.size(0))):
                masked_idx = torch.nonzero(mask[b], as_tuple=False).view(-1)
                if masked_idx.numel() == 0:
                    continue
                if step >= steps - 1:
                    reveal_idx = masked_idx
                else:
                    remain_ratio = self._maskgit_reveal_ratio(step, steps)
                    target_remaining = int(round(float(remain_ratio) * float(n_tok)))
                    target_remaining = min(target_remaining, int(masked_idx.numel()))
                    reveal_count = int(masked_idx.numel()) - target_remaining
                    if reveal_count <= 0:
                        reveal_count = 1
                    reveal_count = min(reveal_count, int(masked_idx.numel()))
                    row_scores = sampled_conf[b, masked_idx]
                    top_pos = torch.topk(row_scores, k=reveal_count, largest=True).indices
                    reveal_idx = masked_idx.index_select(0, top_pos)

                x[b, reveal_idx] = sampled_ids[b, reveal_idx]
                mask[b, reveal_idx] = False

            if not bool(mask.any()):
                break

        if bool(mask.any()):
            x = torch.where(mask, sampled_ids, x)
        return x, (int(gh), int(gw))

    @torch.no_grad()
    def _sample_image_tokens(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        use_ema: bool = False,
        diffusion_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        model_for_gen = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model_for_gen.eval()
        steps = int(self.image_diffusion_steps if diffusion_steps is None else diffusion_steps)
        steps = max(2, int(steps))
        gh, gw = self._image_runtime_grid_shape or self._image_expected_grid_shape()
        feat_dim = int(self._image_runtime_feature_dim or self._image_expected_feature_dim())
        n_tok = int(gh * gw)
        x = torch.randn((int(batch_size), n_tok, feat_dim), device=self.device)
        sched = self._get_image_diffusion_params(device=self.device, diffusion_steps=self.image_diffusion_steps)
        alpha_bars = sched["alpha_bars"]
        t_indices = self._build_inference_timestep_indices(inference_steps=steps, device=self.device)

        attn = torch.ones((int(batch_size), n_tok), device=self.device, dtype=torch.long)
        cls = None
        if class_labels is not None:
            cls = class_labels.to(device=self.device, dtype=torch.long)
        for idx_pos in range(int(t_indices.numel())):
            t_idx = int(t_indices[idx_pos].item())
            t = torch.full((int(x.size(0)),), int(t_idx), device=self.device, dtype=torch.long)
            with self._image_sampling_amp_ctx():
                pred_cond = model_for_gen(x, attention_mask=attn, class_labels=cls, timesteps=t)
            pred_cond = self._fit_pred_to_target(
                pred_cond,
                x,
                context="sample/image_tokens_cond",
            ).float()
            if pred_cond.shape != x.shape:
                x = x[: pred_cond.size(0), : pred_cond.size(1), : pred_cond.size(2)]
                attn = attn[: pred_cond.size(0), : pred_cond.size(1)]
                if cls is not None and cls.size(0) != pred_cond.size(0):
                    cls = cls[: pred_cond.size(0)]
                if t.size(0) != pred_cond.size(0):
                    t = t[: pred_cond.size(0)]
            if cls is not None and guidance_scale > 0.0:
                null_cls = torch.full_like(cls, int(self.image_num_classes))
                with self._image_sampling_amp_ctx():
                    pred_null = model_for_gen(x, attention_mask=attn, class_labels=null_cls, timesteps=t)
                pred_null = self._fit_pred_to_target(
                    pred_null,
                    x,
                    context="sample/image_tokens_null",
                ).float()
                pred = pred_null + float(guidance_scale) * (pred_cond - pred_null)
            else:
                pred = pred_cond

            ab_t = alpha_bars[int(t_idx)].to(dtype=x.dtype)
            x0_pred, _eps_pred = self._prediction_to_x0_eps(pred=pred, x_t=x, alpha_bar_t=ab_t)

            if idx_pos + 1 < int(t_indices.numel()):
                t_prev = int(t_indices[idx_pos + 1].item())
                ab_prev = alpha_bars[int(t_prev)].to(dtype=x.dtype)
                x = self._sample_prev_from_x0_xt(
                    x_t=x,
                    x0_pred=x0_pred,
                    alpha_bar_t=ab_t,
                    alpha_bar_prev=ab_prev,
                )
            else:
                x = x0_pred
        return x, (int(gh), int(gw))

    @torch.no_grad()
    def _sample_image_tokens_maskgit(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        use_ema: bool = False,
        diffusion_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        model_for_gen = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model_for_gen.eval()
        steps = int(self.image_maskgit_steps if diffusion_steps is None else diffusion_steps)
        steps = max(2, int(steps))
        gh, gw = self._image_runtime_grid_shape or self._image_expected_grid_shape()
        feat_dim = int(self._image_runtime_feature_dim or self._image_expected_feature_dim())
        n_tok = int(gh * gw)

        fill_mode = str(getattr(self, "image_maskgit_mask_value", "zero")).lower()
        if fill_mode == "noise":
            x = torch.randn((int(batch_size), n_tok, feat_dim), device=self.device)
        elif fill_mode == "zero":
            x = torch.zeros((int(batch_size), n_tok, feat_dim), device=self.device)

        attn = torch.ones((int(batch_size), n_tok), device=self.device, dtype=torch.long)
        cls = None
        if class_labels is not None:
            cls = class_labels.to(device=self.device, dtype=torch.long)
        mask = torch.ones((int(batch_size), n_tok), device=self.device, dtype=torch.bool)
        prev_pred = None

        for step in range(steps):
            current_mask_ratio = float(mask.float().mean().item()) if mask.numel() > 0 else 0.0
            t = self._maskgit_time_condition(mask_ratio=current_mask_ratio, batch_size=int(batch_size), device=self.device)

            with self._image_sampling_amp_ctx():
                pred_cond = model_for_gen(x, attention_mask=attn, class_labels=cls, timesteps=t)
            pred_cond = self._fit_pred_to_target(pred_cond, x, context="sample/image_maskgit_cond").float()
            if pred_cond.shape != x.shape:
                x = x[: pred_cond.size(0), : pred_cond.size(1), : pred_cond.size(2)]
                attn = attn[: pred_cond.size(0), : pred_cond.size(1)]
                mask = mask[: pred_cond.size(0), : pred_cond.size(1)]
                if cls is not None and cls.size(0) != pred_cond.size(0):
                    cls = cls[: pred_cond.size(0)]
                if t is not None and t.size(0) != pred_cond.size(0):
                    t = t[: pred_cond.size(0)]
                batch_size = int(x.size(0))
                n_tok = int(x.size(1))

            if cls is not None and guidance_scale > 0.0:
                null_cls = torch.full_like(cls, int(getattr(model_for_gen, "class_null_index", self.image_num_classes)))
                with self._image_sampling_amp_ctx():
                    pred_null = model_for_gen(x, attention_mask=attn, class_labels=null_cls, timesteps=t)
                pred_null = self._fit_pred_to_target(pred_null, x, context="sample/image_maskgit_null").float()
                pred = pred_null + float(guidance_scale) * (pred_cond - pred_null)
            else:
                pred = pred_cond

            if not torch.isfinite(pred).all():
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

            if str(getattr(self, "image_maskgit_confidence", "stability")).lower() == "random":
                scores = torch.rand((int(batch_size), n_tok), device=self.device)
            else:
                if prev_pred is None:
                    scores = -((pred - x) ** 2).mean(dim=-1)
                else:
                    scores = -((pred - prev_pred) ** 2).mean(dim=-1)
            scores = scores.masked_fill(~mask, float("-inf"))

            for b in range(int(batch_size)):
                masked_idx = torch.nonzero(mask[b], as_tuple=False).view(-1)
                if masked_idx.numel() == 0:
                    continue
                if step >= steps - 1:
                    reveal_idx = masked_idx
                else:
                    remain_ratio = self._maskgit_reveal_ratio(step, steps)
                    target_remaining = int(round(float(remain_ratio) * float(n_tok)))
                    target_remaining = min(target_remaining, int(masked_idx.numel()))
                    reveal_count = int(masked_idx.numel()) - target_remaining
                    if reveal_count <= 0:
                        reveal_count = 1
                    reveal_count = min(reveal_count, int(masked_idx.numel()))
                    row_scores = scores[b, masked_idx]
                    top_pos = torch.topk(row_scores, k=reveal_count, largest=True).indices
                    reveal_idx = masked_idx.index_select(0, top_pos)

                x[b, reveal_idx] = pred[b, reveal_idx]
                mask[b, reveal_idx] = False

            prev_pred = pred.detach()
            if not bool(mask.any()):
                break

        if bool(mask.any()):
            x = torch.where(mask.unsqueeze(-1), pred, x)
        return x, (int(gh), int(gw))

    @torch.no_grad()
    def _sample_image_pixels_rgb_unet(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        use_ema: bool = False,
        diffusion_steps: Optional[int] = None,
    ) -> torch.Tensor:
        model_for_gen = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model_for_gen.eval()
        steps = int(self.image_diffusion_steps if diffusion_steps is None else diffusion_steps)
        steps = max(2, int(steps))
        x = torch.randn(
            (int(batch_size), 3, int(self.image_size), int(self.image_size)),
            device=self.device,
        )
        if bool(self.image_rgb_centered_diffusion):
            x = x.clamp(-3.0, 3.0)
        sched = self._get_image_diffusion_params(device=self.device, diffusion_steps=self.image_diffusion_steps)
        alpha_bars = sched["alpha_bars"]
        t_indices = self._build_inference_timestep_indices(inference_steps=steps, device=self.device)

        cls = None
        if class_labels is not None:
            cls = class_labels.to(device=self.device, dtype=torch.long)

        for idx_pos in range(int(t_indices.numel())):
            t_idx = int(t_indices[idx_pos].item())
            t = torch.full((int(x.size(0)),), int(t_idx), device=self.device, dtype=torch.long)
            with self._image_sampling_amp_ctx():
                cond_vec = self._shared_image_cond_vec(
                    model_ref=model_for_gen,
                    batch_size=int(x.size(0)),
                    device=x.device,
                    class_labels=cls,
                    timesteps=t,
                )
                tok_cond, dec_ctx = self._encode_pixels_rgb_unet_tokens(
                    x,
                    model_ref=model_for_gen,
                    class_labels=cls,
                    timesteps=t,
                    cond_vec=cond_vec,
                )
                attn = torch.ones((int(tok_cond.size(0)), int(tok_cond.size(1))), device=tok_cond.device, dtype=torch.long)
                pred_tok_cond = model_for_gen(tok_cond, attention_mask=attn, class_labels=cls, timesteps=t, cond_vec=cond_vec)
                pred_tok_cond = self._fit_pred_to_target(
                    pred_tok_cond,
                    tok_cond,
                    context="sample/rgb_unet_cond_tokens",
                )
                pred_rgb_cond = self._decode_rgb_unet_tokens_to_pixels(
                    pred_tok_cond,
                    dec_ctx,
                    model_ref=model_for_gen,
                    class_labels=cls,
                    timesteps=t,
                    cond_vec=cond_vec,
                )
            pred_rgb_cond = pred_rgb_cond.float()

            if cls is not None and guidance_scale > 0.0:
                null_cls = torch.full_like(cls, int(self.image_num_classes))
                with self._image_sampling_amp_ctx():
                    cond_vec_null = self._shared_image_cond_vec(
                        model_ref=model_for_gen,
                        batch_size=int(x.size(0)),
                        device=x.device,
                        class_labels=null_cls,
                        timesteps=t,
                    )
                    tok_null, dec_ctx_null = self._encode_pixels_rgb_unet_tokens(
                        x,
                        model_ref=model_for_gen,
                        class_labels=null_cls,
                        timesteps=t,
                        cond_vec=cond_vec_null,
                    )
                    attn_null = torch.ones((int(tok_null.size(0)), int(tok_null.size(1))), device=tok_null.device, dtype=torch.long)
                    pred_tok_null = model_for_gen(tok_null, attention_mask=attn_null, class_labels=null_cls, timesteps=t, cond_vec=cond_vec_null)
                    pred_tok_null = self._fit_pred_to_target(
                        pred_tok_null,
                        tok_null,
                        context="sample/rgb_unet_null_tokens",
                    )
                    pred_rgb_null = self._decode_rgb_unet_tokens_to_pixels(
                        pred_tok_null,
                        dec_ctx_null,
                        model_ref=model_for_gen,
                        class_labels=null_cls,
                        timesteps=t,
                        cond_vec=cond_vec_null,
                    )
                pred_rgb_null = pred_rgb_null.float()
                pred_rgb = pred_rgb_null + float(guidance_scale) * (pred_rgb_cond - pred_rgb_null)
            else:
                pred_rgb = pred_rgb_cond

            ab_t = alpha_bars[int(t_idx)].to(dtype=x.dtype)
            x0_pred, _eps_pred = self._prediction_to_x0_eps(pred=pred_rgb, x_t=x, alpha_bar_t=ab_t)
            if bool(getattr(self, "image_sampling_clamp_x0", True)):
                # Pixel data is bounded: clamping the per-step x0 estimate keeps the
                # trajectory on-manifold (standard DDPM practice; matters at few steps).
                if bool(self.image_rgb_centered_diffusion):
                    x0_pred = x0_pred.clamp(-1.0, 1.0)
                else:
                    x0_pred = x0_pred.clamp(0.0, 1.0)

            if idx_pos + 1 < int(t_indices.numel()):
                t_prev = int(t_indices[idx_pos + 1].item())
                ab_prev = alpha_bars[int(t_prev)].to(dtype=x.dtype)
                x = self._sample_prev_from_x0_xt(
                    x_t=x,
                    x0_pred=x0_pred,
                    alpha_bar_t=ab_t,
                    alpha_bar_prev=ab_prev,
                )
            else:
                x = x0_pred

        if bool(self.image_rgb_centered_diffusion):
            return self._centered_to_rgb(x).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)

    def _validate_image_diffusion(self, data_provider, use_ema=False):
        model_to_use = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model_to_use.eval()

        if isinstance(data_provider, dict) and "get_batch" in data_provider:
            get_batch = data_provider["get_batch"]
            steps = int(data_provider.get("steps", 50))
            data_iterator = range(steps)
            use_get_batch = True
            desc = "Validation (Image Steps)"
        elif isinstance(data_provider, torch.utils.data.DataLoader):
            dataloader = data_provider
            steps = len(dataloader)
            data_iterator = dataloader
            use_get_batch = False
            desc = "Validation (Image Batches)"
        else:
            raise ValueError("Invalid data_provider format.")

        pbar = tqdm(data_iterator, desc=desc, total=steps, dynamic_ncols=True)
        total_loss = 0.0
        denom = 0
        with torch.no_grad():
            for val_bi, batch_data in enumerate(pbar):
                batch = get_batch(self.device) if use_get_batch else batch_data
                if self.image_token_mode == "rgb_unet":
                    if not isinstance(batch, dict):
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            batch = {"pixel_values": batch[0], "class_labels": batch[1]}
                        elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                            batch = {"pixel_values": batch[0]}
                        else:
                            raise ValueError(f"Unsupported image batch format: {type(batch)}")
                    pixel_values = batch.get("pixel_values", None)
                    if pixel_values is None:
                        raise ValueError("rgb_unet image mode expects batch['pixel_values']")
                    pixel_values = pixel_values.to(self.device)
                    class_labels = batch.get("class_labels", None)
                    if class_labels is not None:
                        class_labels = class_labels.to(device=self.device, dtype=torch.long)

                    if self.image_diffusion_target_runtime == "rgb_epsilon":
                        bsz = int(pixel_values.size(0))
                        sched = self._get_image_diffusion_params(device=pixel_values.device)
                        alpha_bars = sched["alpha_bars"]
                        x0_rgb = self._rgb_to_centered(pixel_values) if bool(self.image_rgb_centered_diffusion) else pixel_values
                        t, noise_rgb = self._val_diffusion_draws(
                            batch_index=val_bi,
                            batch_size=bsz,
                            noise_shape=tuple(x0_rgb.shape),
                            device=x0_rgb.device,
                            dtype=x0_rgb.dtype,
                        )
                        ab = alpha_bars.index_select(0, t).view(bsz, 1, 1, 1).to(dtype=pixel_values.dtype)
                        x_t_rgb = torch.sqrt(ab) * x0_rgb + torch.sqrt(1.0 - ab) * noise_rgb
                        with self._image_sampling_amp_ctx():
                            cond_vec = self._shared_image_cond_vec(
                                model_ref=model_to_use,
                                batch_size=bsz,
                                device=pixel_values.device,
                                class_labels=class_labels,
                                timesteps=t,
                            )
                            x_t_tokens, decode_context = self._encode_pixels_rgb_unet_tokens(
                                x_t_rgb,
                                model_ref=model_to_use,
                                class_labels=class_labels,
                                timesteps=t,
                                cond_vec=cond_vec,
                            )
                            tok = int(x_t_tokens.size(1))
                            attn = torch.ones((bsz, tok), device=x_t_tokens.device, dtype=torch.long)
                            pred_tokens = model_to_use(x_t_tokens, attention_mask=attn, class_labels=class_labels, timesteps=t, cond_vec=cond_vec)
                            pred_tokens_fit = self._fit_pred_to_target(
                                pred_tokens,
                                x_t_tokens,
                                context="val/rgb_unet_rgb_epsilon_tokens",
                            )
                            pred_eps_rgb = self._decode_rgb_unet_tokens_to_pixels(
                                pred_tokens_fit,
                                decode_context,
                                model_ref=model_to_use,
                                class_labels=class_labels,
                                timesteps=t,
                                cond_vec=cond_vec,
                            )
                        pr = pred_eps_rgb.float()
                        tg = self._diffusion_target_from_x0_noise(x0=x0_rgb, noise=noise_rgb, alpha_bar_t=ab.view(bsz))
                        if pr.shape != tg.shape:
                            msg = f"val/rgb_unet_rgb_epsilon_pixels shape mismatch: pred={tuple(pr.shape)} target={tuple(tg.shape)}"
                            if bool(getattr(self, "image_diffusion_strict_shapes", True)):
                                raise RuntimeError(msg)
                            if not bool(getattr(self, "_image_shape_mismatch_warned", False)):
                                logger.warning("%s; cropping to shared minimum spatial shape", msg)
                                self._image_shape_mismatch_warned = True
                            h = min(int(pr.size(-2)), int(tg.size(-2)))
                            w = min(int(pr.size(-1)), int(tg.size(-1)))
                            pr = pr[:, :, :h, :w]
                            tg = tg[:, :, :h, :w]
                        self._record_image_debug_stats(pr, tg, t)
                        loss = self._weighted_mse_loss(pr, tg, alpha_bar_t=ab.view(bsz))
                    else:
                        with self._image_sampling_amp_ctx():
                            x0, _ = self._encode_pixels_rgb_unet_tokens(pixel_values, model_ref=model_to_use)
                        x0 = x0.float()
                        bsz, tok, _ = x0.shape
                        sched = self._get_image_diffusion_params(device=x0.device)
                        alpha_bars = sched["alpha_bars"]
                        t, noise = self._val_diffusion_draws(
                            batch_index=val_bi,
                            batch_size=bsz,
                            noise_shape=tuple(x0.shape),
                            device=x0.device,
                            dtype=x0.dtype,
                        )
                        ab = alpha_bars.index_select(0, t).view(bsz, 1, 1).to(dtype=x0.dtype)
                        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
                        attn = torch.ones((bsz, tok), device=x0.device, dtype=torch.long)
                        with self._image_sampling_amp_ctx():
                            pred = model_to_use(x_t, attention_mask=attn, class_labels=class_labels, timesteps=t)
                        pred = pred.float()
                        target = self._diffusion_target_from_x0_noise(x0=x0, noise=noise, alpha_bar_t=ab.view(bsz))
                        pred_fit = self._fit_pred_to_target(
                            pred,
                            target,
                            context="val/rgb_unet_token_space",
                        )
                        target_fit = target[: pred_fit.size(0), : pred_fit.size(1), : pred_fit.size(2)]
                        self._record_image_debug_stats(pred_fit, target_fit, t)
                        loss = self._weighted_mse_loss(pred_fit, target_fit, alpha_bar_t=ab.view(bsz))
                else:
                    x0, _, class_labels = self._prepare_image_batch_features(batch, device=self.device)
                    bsz, tok, _ = x0.shape
                    sched = self._get_image_diffusion_params(device=x0.device)
                    alpha_bars = sched["alpha_bars"]
                    t, noise = self._val_diffusion_draws(
                        batch_index=val_bi,
                        batch_size=bsz,
                        noise_shape=tuple(x0.shape),
                        device=x0.device,
                        dtype=x0.dtype,
                    )
                    ab = alpha_bars.index_select(0, t).view(bsz, 1, 1).to(dtype=x0.dtype)
                    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
                    attn = torch.ones((bsz, tok), device=x0.device, dtype=torch.long)
                    with self._image_sampling_amp_ctx():
                        pred = model_to_use(x_t, attention_mask=attn, class_labels=class_labels, timesteps=t)
                    pred = pred.float()
                    target = self._diffusion_target_from_x0_noise(x0=x0, noise=noise, alpha_bar_t=ab.view(bsz))
                    pred_fit = self._fit_pred_to_target(
                        pred,
                        target,
                        context="val/image_latent_or_patch",
                    )
                    target_fit = target[: pred_fit.size(0), : pred_fit.size(1), : pred_fit.size(2)]
                    self._record_image_debug_stats(pred_fit, target_fit, t)
                    loss = self._weighted_mse_loss(pred_fit, target_fit, alpha_bar_t=ab.view(bsz))
                total_loss += float(loss.item())
                denom += 1
                pbar.set_postfix({"val_img_diff": float(loss.item())})
        pbar.close()

        if denom <= 0:
            metrics = {
                "loss": float("inf"),
                "objective_loss": float("inf"),
                "masked_loss": float("inf"),
                "next_token_loss": float("inf"),
                "masked_acc": 0.0,
                "next_token_acc": 0.0,
                "perplexity": float("inf"),
                "image_diffusion_loss": float("inf"),
                "selection_metric": "image_diffusion_loss",
                "selection_loss": float("inf"),
            }
            return float("inf"), metrics

        avg = total_loss / float(denom)
        metrics = {
            "loss": float(avg),
            "objective_loss": float(avg),
            "masked_loss": float(avg),
            "next_token_loss": float(avg),
            "masked_acc": 0.0,
            "next_token_acc": 0.0,
            "perplexity": float("inf"),
            "image_diffusion_loss": float(avg),
            "selection_metric": "image_diffusion_loss",
            "selection_loss": float(avg),
        }
        return float(avg), metrics

    def _validate_image_maskgit(self, data_provider, use_ema=False):
        model_to_use = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        model_to_use.eval()

        if isinstance(data_provider, dict) and "get_batch" in data_provider:
            get_batch = data_provider["get_batch"]
            steps = int(data_provider.get("steps", 50))
            data_iterator = range(steps)
            use_get_batch = True
            desc = "Validation (Image MaskGIT Steps)"
        elif isinstance(data_provider, torch.utils.data.DataLoader):
            dataloader = data_provider
            steps = len(dataloader)
            data_iterator = dataloader
            use_get_batch = False
            desc = "Validation (Image MaskGIT Batches)"
        else:
            raise ValueError("Invalid data_provider format.")

        pbar = tqdm(data_iterator, desc=desc, total=steps, dynamic_ncols=True)
        total_loss = 0.0
        denom = 0
        val_ratio = float(getattr(self, "image_maskgit_val_mask_ratio", 0.5))
        amp_on = bool(self.mixed_precision and torch.cuda.is_available() and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if bool(self.use_bf16) else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_on else nullcontext()

        with torch.no_grad():
            for _, batch_data in enumerate(pbar):
                batch = get_batch(self.device) if use_get_batch else batch_data
                x0, grid_shape, class_labels = self._prepare_image_batch_features(batch, device=self.device)
                bsz, tok, _ = x0.shape
                mask = self._sample_image_feature_mask(
                    token_count=tok,
                    grid_shape=grid_shape,
                    mask_ratio=val_ratio,
                    batch_size=bsz,
                    device=x0.device,
                )
                if str(getattr(self, "image_maskgit_mask_value", "zero")).lower() == "noise":
                    mask_fill = torch.randn_like(x0)
                else:
                    mask_fill = torch.zeros_like(x0)
                x_in = torch.where(mask.unsqueeze(-1), mask_fill, x0)
                timesteps = self._maskgit_time_condition(mask_ratio=val_ratio, batch_size=bsz, device=x0.device)
                attn = torch.ones((bsz, tok), device=x0.device, dtype=torch.long)

                with amp_ctx:
                    pred = model_to_use(
                        x_in,
                        attention_mask=attn,
                        class_labels=class_labels,
                        timesteps=timesteps,
                    )

                loss = self._masked_mse_loss(pred, x0, mask, context="val/image_maskgit_masked")
                unmasked_weight = float(getattr(self, "image_maskgit_unmasked_weight", 0.0))
                if unmasked_weight > 0.0:
                    loss = loss + (unmasked_weight * self._masked_mse_loss(pred, x0, ~mask, context="val/image_maskgit_unmasked"))

                self._record_image_debug_stats(
                    self._fit_pred_to_target(pred, x0, context="val/image_maskgit"),
                    x0[: pred.size(0), : pred.size(1), : pred.size(2)],
                    timesteps,
                )
                total_loss += float(loss.item())
                denom += 1
                pbar.set_postfix({"val_img_maskgit": float(loss.item())})

        pbar.close()

        if denom <= 0:
            metrics = {
                "loss": float("inf"),
                "objective_loss": float("inf"),
                "masked_loss": float("inf"),
                "next_token_loss": float("inf"),
                "masked_acc": 0.0,
                "next_token_acc": 0.0,
                "perplexity": float("inf"),
                "image_maskgit_loss": float("inf"),
                "image_diffusion_loss": float("inf"),
                "selection_metric": "image_maskgit_loss",
                "selection_loss": float("inf"),
            }
            return float("inf"), metrics

        avg = total_loss / float(denom)
        metrics = {
            "loss": float(avg),
            "objective_loss": float(avg),
            "masked_loss": float(avg),
            "next_token_loss": float(avg),
            "masked_acc": 0.0,
            "next_token_acc": 0.0,
            "perplexity": float("inf"),
            "image_maskgit_loss": float(avg),
            "image_diffusion_loss": float(avg),
            "selection_metric": "image_maskgit_loss",
            "selection_loss": float(avg),
        }
        return float(avg), metrics

    def _resolve_image_examples_dir(self, root: str, default_subdir: str, eval_tag: Optional[str] = None) -> str:
        root = str(root).strip()
        if not root:
            root = os.path.join(str(self.checkpoint_dir), str(default_subdir))
        tag = str(eval_tag or f"epoch_{int(self.current_epoch) + 1:04d}")
        out_dir = os.path.join(root, tag)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _resolve_fid_examples_dir(self, eval_tag: Optional[str] = None) -> str:
        return self._resolve_image_examples_dir(self.image_fid_examples_dir, "fid_examples", eval_tag=eval_tag)

    def _resolve_image_preview_examples_dir(self, eval_tag: Optional[str] = None) -> str:
        return self._resolve_image_examples_dir(self.image_preview_examples_dir, "preview_examples", eval_tag=eval_tag)

    def _save_image_example_pairs(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor,
        out_dir: str,
        start_index: int,
        max_to_save: int,
    ) -> int:
        if int(max_to_save) <= 0:
            return 0
        try:
            from torchvision.utils import make_grid, save_image
        except Exception as exc:
            logger.warning("Image example saving unavailable (torchvision.utils): %s", str(exc))
            return 0

        take = min(int(max_to_save), int(fake_images.size(0)), int(real_images.size(0)))
        if take <= 0:
            return 0

        os.makedirs(out_dir, exist_ok=True)
        fake_cpu = fake_images[:take].detach().cpu().clamp(0.0, 1.0)
        real_cpu = real_images[:take].detach().cpu().clamp(0.0, 1.0)

        for i in range(take):
            idx = int(start_index) + i
            save_image(fake_cpu[i], os.path.join(out_dir, f"fake_{idx:05d}.png"))
            save_image(real_cpu[i], os.path.join(out_dir, f"real_{idx:05d}.png"))

        if int(start_index) == 0:
            nrow = max(1, int(round(math.sqrt(take))))
            fake_grid = make_grid(fake_cpu, nrow=nrow)
            real_grid = make_grid(real_cpu, nrow=nrow)
            save_image(fake_grid, os.path.join(out_dir, "fake_grid.png"))
            save_image(real_grid, os.path.join(out_dir, "real_grid.png"))

        return int(take)

    def _save_fid_examples(
        self,
        fake_images: torch.Tensor,
        real_images: torch.Tensor,
        eval_tag: Optional[str],
        start_index: int,
        max_to_save: int,
    ) -> int:
        out_dir = self._resolve_fid_examples_dir(eval_tag=eval_tag)
        return self._save_image_example_pairs(
            fake_images=fake_images,
            real_images=real_images,
            out_dir=out_dir,
            start_index=start_index,
            max_to_save=max_to_save,
        )

    def evaluate_image_fid(
        self,
        data_provider,
        use_ema=False,
        num_samples=None,
        guidance_scale=None,
        eval_tag: Optional[str] = None,
    ):
        if not bool(self.image_fid_enable):
            return None
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except Exception:
            logger.warning("FID requested but torchmetrics FID is unavailable. Install torchmetrics[image] and torch-fidelity.")
            return None

        target_samples = int(num_samples or self.image_fid_num_samples)
        cfg = float(self.image_fid_guidance_scale if guidance_scale is None else guidance_scale)
        fid_steps = int(self.image_fid_diffusion_steps if self.image_fid_diffusion_steps > 0 else self.image_diffusion_steps)
        fid_steps = max(2, int(fid_steps))
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

        get_batch = self._resolve_image_eval_get_batch(data_provider)
        if get_batch is None:
            logger.warning("FID skipped: invalid data provider")
            return None

        seen = 0
        saved_examples = 0
        pbar = tqdm(total=target_samples, desc="FID (Image Samples)", dynamic_ncols=True)
        while seen < target_samples:
            batch = get_batch(self.device)
            if not isinstance(batch, dict):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    batch = {"pixel_values": batch[0], "class_labels": batch[1]}
                elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    batch = {"pixel_values": batch[0]}
                else:
                    pbar.close()
                    logger.warning("FID skipped: unsupported batch format %s", str(type(batch)))
                    return None
            pixel_values = batch.get("pixel_values", None)
            if pixel_values is None:
                pbar.close()
                logger.warning("FID skipped: batch lacks pixel_values")
                return None
            pixel_values = pixel_values.to(self.device)
            bsz = int(pixel_values.size(0))
            cls = batch.get("class_labels", None)
            if cls is not None:
                cls = cls.to(device=self.device, dtype=torch.long)
            take = min(bsz, target_samples - seen)
            real = pixel_values[:take].clamp(0.0, 1.0)
            fid.update(real, real=True)
            if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit" and str(getattr(self, "image_maskgit_variant", "continuous")).lower() == "discrete":
                fake_tokens, grid_shape = self._sample_image_tokens_maskgit_discrete(
                    batch_size=take,
                    class_labels=(cls[:take] if cls is not None else None),
                    guidance_scale=cfg,
                    use_ema=use_ema,
                    diffusion_steps=int(self.image_maskgit_steps),
                )
                fake = self._image_maskgit_decode_discrete_tokens(fake_tokens, grid_shape).clamp(0.0, 1.0)
            elif self.image_token_mode == "rgb_unet":
                if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit" and not bool(getattr(self, "_image_maskgit_rgb_unet_warned", False)):
                    logger.warning(
                        "FID for image_objective=maskgit with rgb_unet falls back to diffusion sampling because the RGB U-Net decoder requires encoder context."
                    )
                    self._image_maskgit_rgb_unet_warned = True
                if self.image_diffusion_target_runtime != "rgb_epsilon":
                    pbar.close()
                    logger.warning(
                        "FID for image_token_mode=rgb_unet currently requires image_diffusion_target=rgb_epsilon; got %s",
                        str(self.image_diffusion_target_runtime),
                    )
                    return None
                fake = self._sample_image_pixels_rgb_unet(
                    batch_size=take,
                    class_labels=(cls[:take] if cls is not None else None),
                    guidance_scale=cfg,
                    use_ema=use_ema,
                    diffusion_steps=fid_steps,
                ).clamp(0.0, 1.0)
            else:
                if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit":
                    fake_tokens, grid_shape = self._sample_image_tokens_maskgit(
                        batch_size=take,
                        class_labels=(cls[:take] if cls is not None else None),
                        guidance_scale=cfg,
                        use_ema=use_ema,
                        diffusion_steps=int(self.image_maskgit_steps),
                    )
                else:
                    fake_tokens, grid_shape = self._sample_image_tokens(
                        batch_size=take,
                        class_labels=(cls[:take] if cls is not None else None),
                        guidance_scale=cfg,
                        use_ema=use_ema,
                        diffusion_steps=fid_steps,
                    )
                fake = self._image_tokens_to_pixels(fake_tokens, grid_shape).clamp(0.0, 1.0)
            fid.update(fake, real=False)

            if bool(self.image_fid_save_examples) and saved_examples < int(self.image_fid_examples_per_eval):
                remaining = int(self.image_fid_examples_per_eval) - int(saved_examples)
                just_saved = self._save_fid_examples(
                    fake_images=fake,
                    real_images=real,
                    eval_tag=eval_tag,
                    start_index=int(saved_examples),
                    max_to_save=remaining,
                )
                saved_examples += int(just_saved)

            seen += int(take)
            pbar.update(int(take))
            pbar.set_postfix({"cfg": f"{float(cfg):.2f}", "steps": int(fid_steps)})

        pbar.close()

        score = float(fid.compute().item())
        logger.info("Image FID: %.4f", score)
        if bool(self.image_fid_save_examples):
            out_dir = self._resolve_fid_examples_dir(eval_tag=eval_tag)
            logger.info("Saved %d FID example pairs to %s", int(saved_examples), out_dir)
        return score

    def _resolve_image_eval_get_batch(self, data_provider):
        if isinstance(data_provider, dict) and "get_batch" in data_provider:
            get_batch = data_provider["get_batch"]
            # When the val provider is token/latent-cache-backed it yields token_ids
            # (no pixel_values) — fine for CE validation, but the image preview / FID
            # paths need raw pixels (to render real|fake grids and size the batch). The
            # loader exposes a parallel raw-pixel eval provider (cache=None) for exactly
            # this case; prefer it so previews/FID keep working with caching enabled.
            if str(getattr(get_batch, "cache_mode", "off")).lower() != "off":
                raw = getattr(get_batch, "raw_eval_batch", None)
                if raw is not None:
                    return raw
            return get_batch
        if isinstance(data_provider, torch.utils.data.DataLoader):
            data_iter = iter(data_provider)

            def _next_batch(_dev):
                nonlocal data_iter
                try:
                    return next(data_iter)
                except StopIteration:
                    data_iter = iter(data_provider)
                    return next(data_iter)

            return _next_batch
        return None

    @torch.no_grad()
    def evaluate_image_preview(
        self,
        data_provider,
        num_samples=None,
        guidance_scale=None,
        eval_tag: Optional[str] = None,
    ):
        if not bool(self.image_preview_enable):
            return None

        target_samples = int(num_samples or self.image_preview_num_samples)
        cfg = float(self.image_preview_guidance_scale if guidance_scale is None else guidance_scale)
        preview_steps = int(self.image_preview_diffusion_steps if self.image_preview_diffusion_steps > 0 else self.image_diffusion_steps)
        preview_steps = max(2, int(preview_steps))

        get_batch = self._resolve_image_eval_get_batch(data_provider)
        if get_batch is None:
            logger.warning("Image preview skipped: invalid data provider")
            return None

        seen = 0
        saved_examples = 0
        out_dir = self._resolve_image_preview_examples_dir(eval_tag=eval_tag)
        pbar = tqdm(total=target_samples, desc="Preview (Image Samples)", dynamic_ncols=True)

        while seen < target_samples:
            batch = get_batch(self.device)
            if not isinstance(batch, dict):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    batch = {"pixel_values": batch[0], "class_labels": batch[1]}
                elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    batch = {"pixel_values": batch[0]}
                else:
                    pbar.close()
                    logger.warning("Image preview skipped: unsupported batch format %s", str(type(batch)))
                    return None

            pixel_values = batch.get("pixel_values", None)
            if pixel_values is None:
                pbar.close()
                logger.warning("Image preview skipped: batch lacks pixel_values")
                return None

            pixel_values = pixel_values.to(self.device)
            bsz = int(pixel_values.size(0))
            cls = batch.get("class_labels", None)
            if cls is not None:
                cls = cls.to(device=self.device, dtype=torch.long)

            take = min(bsz, target_samples - seen)
            real = pixel_values[:take].clamp(0.0, 1.0)

            if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit" and str(getattr(self, "image_maskgit_variant", "continuous")).lower() == "discrete":
                fake_tokens, grid_shape = self._sample_image_tokens_maskgit_discrete(
                    batch_size=take,
                    class_labels=(cls[:take] if cls is not None else None),
                    guidance_scale=cfg,
                    use_ema=True,
                    diffusion_steps=int(self.image_maskgit_steps),
                )
                fake = self._image_maskgit_decode_discrete_tokens(fake_tokens, grid_shape).clamp(0.0, 1.0)
            elif self.image_token_mode == "rgb_unet":
                if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit" and not bool(getattr(self, "_image_maskgit_rgb_unet_warned", False)):
                    logger.warning(
                        "Image preview for image_objective=maskgit with rgb_unet falls back to diffusion sampling because the RGB U-Net decoder requires encoder context."
                    )
                    self._image_maskgit_rgb_unet_warned = True
                if self.image_diffusion_target_runtime != "rgb_epsilon":
                    pbar.close()
                    logger.warning(
                        "Image preview for image_token_mode=rgb_unet currently requires image_diffusion_target=rgb_epsilon; got %s",
                        str(self.image_diffusion_target_runtime),
                    )
                    return None
                fake = self._sample_image_pixels_rgb_unet(
                    batch_size=take,
                    class_labels=(cls[:take] if cls is not None else None),
                    guidance_scale=cfg,
                    use_ema=True,
                    diffusion_steps=preview_steps,
                    ).clamp(0.0, 1.0)
            else:
                if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit":
                    fake_tokens, grid_shape = self._sample_image_tokens_maskgit(
                        batch_size=take,
                        class_labels=(cls[:take] if cls is not None else None),
                        guidance_scale=cfg,
                        use_ema=True,
                        diffusion_steps=int(self.image_maskgit_steps),
                    )
                else:
                    fake_tokens, grid_shape = self._sample_image_tokens(
                        batch_size=take,
                        class_labels=(cls[:take] if cls is not None else None),
                        guidance_scale=cfg,
                        use_ema=True,
                        diffusion_steps=preview_steps,
                    )
                fake = self._image_tokens_to_pixels(fake_tokens, grid_shape).clamp(0.0, 1.0)

            remaining = target_samples - saved_examples
            just_saved = self._save_image_example_pairs(
                fake_images=fake,
                real_images=real,
                out_dir=out_dir,
                start_index=int(saved_examples),
                max_to_save=remaining,
            )
            saved_examples += int(just_saved)

            seen += int(take)
            pbar.update(int(take))
            pbar.set_postfix({"cfg": f"{float(cfg):.2f}", "steps": int(preview_steps)})

        pbar.close()
        logger.info("Image preview generated: %d samples saved to %s", int(saved_examples), out_dir)
        return int(saved_examples)

    def _resolve_autoenc_runtime_mode(self, model_ref) -> Dict[str, Any]:
        model_mod = getattr(model_ref, "module", None)
        autoenc_graph_mode = str(
            getattr(
                model_ref,
                "autoenc_graph_mode",
                getattr(model_mod, "autoenc_graph_mode", "off"),
            )
        ).lower()
        policy = str(getattr(self, "autoenc_training_policy", "auxiliary")).lower()
        if policy not in {"auxiliary", "autoenc_only", "autoenc_only_diffusion"}:
            policy = "auxiliary"
        autoenc_only_mode = (
            autoenc_graph_mode == "twin_shared_l3"
            and policy == "autoenc_only"
        )
        autoenc_only_diffusion_mode = (
            autoenc_graph_mode == "twin_shared_l3"
            and policy == "autoenc_only_diffusion"
        )
        return {
            "autoenc_graph_mode": autoenc_graph_mode,
            "autoenc_training_policy": policy,
            "autoenc_only_mode": autoenc_only_mode,
            "autoenc_only_diffusion_mode": autoenc_only_diffusion_mode,
        }

    def _resolve_l0_local_runtime_mode(self, model_ref) -> Dict[str, Any]:
        model_mod = getattr(model_ref, "module", None)
        backend = str(
            getattr(
                model_ref,
                "l0_local_backend",
                getattr(model_mod, "l0_local_backend", "pyg"),
            )
        ).lower()
        try:
            window = int(
                getattr(
                    model_ref,
                    "l0_local_window",
                    getattr(model_mod, "l0_local_window", 0),
                )
            )
        except Exception:
            window = 0
        active = backend != "pyg" and window > 0
        # Multi-level local attention config
        local_attn_cfg = getattr(
            model_mod or model_ref, "local_attn_config", {}
        )
        active_levels = sorted(local_attn_cfg.keys()) if local_attn_cfg else []
        return {
            "backend": backend,
            "window": max(0, window),
            "active": bool(active) or bool(active_levels),
            "local_attn_levels": active_levels,
            "local_attn_config": dict(local_attn_cfg) if local_attn_cfg else {},
        }

    def _maybe_apply_copy_task(self, batch: Dict[str, torch.Tensor], is_validation: bool = False) -> Dict[str, torch.Tensor]:
        if (not self.copy_task_enable) or (self.copy_task_marker_ids is None):
            return batch
        if not isinstance(batch, dict) or "input_ids" not in batch:
            return batch

        apply_prob = self.copy_task_val_prob if is_validation else self.copy_task_train_prob
        if apply_prob <= 0.0:
            return batch

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        if input_ids is None:
            return batch

        try:
            out_ids, copy_meta = apply_copy_task_to_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                marker_ids=self.copy_task_marker_ids,
                apply_prob=apply_prob,
                src_len_min=self.copy_task_src_len_min,
                src_len_max=self.copy_task_src_len_max,
                min_gap=self.copy_task_min_gap,
                max_gap=self.copy_task_max_gap,
            )
        except Exception as e:
            logger.warning(f"Copy-task batch augmentation failed; keeping original batch: {e}")
            return batch

        if not bool(copy_meta["applied"].any()):
            return batch

        out_batch = dict(batch)
        out_batch["input_ids"] = out_ids
        out_batch["copy_task_meta"] = copy_meta
        out_batch["copy_dst_mask"] = copy_meta["dst_payload_mask"]
        out_batch["copy_force_mask_dst"] = True
        out_batch["copy_mask_dst_in_ar"] = bool(self.copy_task_mask_dst_in_ar)
        out_batch["copy_mask_token_id"] = int(self.copy_task_mask_token_id)
        return out_batch

    @staticmethod
    def _update_copy_bucket(stats_dict, key, tok_correct, tok_total, span_ok, first_ok, ce_numer, ce_denom):
        if key not in stats_dict:
            stats_dict[key] = {
                "tok_correct": 0,
                "tok_total": 0,
                "span_correct": 0,
                "span_total": 0,
                "first_correct": 0,
                "first_total": 0,
                "ce_numer": 0.0,
                "ce_denom": 0,
            }
        stats_dict[key]["tok_correct"] += int(tok_correct)
        stats_dict[key]["tok_total"] += int(tok_total)
        stats_dict[key]["span_correct"] += int(1 if span_ok else 0)
        stats_dict[key]["span_total"] += 1
        stats_dict[key]["first_correct"] += int(1 if first_ok else 0)
        stats_dict[key]["first_total"] += 1
        stats_dict[key]["ce_numer"] += float(ce_numer)
        stats_dict[key]["ce_denom"] += int(ce_denom)

    def _format_progress_postfix(self, full_postfix, step_index):
        if not full_postfix:
            return full_postfix, set()

        if self.progress_view == "full":
            return full_postfix, set(full_postfix.keys())

        core_keys = ["loss", "obj", "lr", "grad_norm", "tok/s", "tok/s_win", "cycles"]
        compact_keys = [
            "ce",
            "obj",
            "haux",
            "copy_ce",
            "ae_ce",
            "ae_next",
            "unet_lh",
            "c_hit",
            "cyc_p80",
            "cyc_gov_ema",
            "min_prob",
            "cycles_ema",
            "max_cycle_rate",
        ]
        rotate_groups = [
            ["haux", "cyc_p80", "cyc_gov_ema"],
            ["c_hit", "c_miss", "c_rst", "c_seed"],
            ["min_prob", "cycles_ema", "max_cycle_rate", "ticks", "gamma_mean", "gamma_max"],
        ]

        ordered = []
        for key in core_keys:
            if key in full_postfix:
                ordered.append(key)

        if self.progress_view == "rotate":
            group_idx = (step_index // self.progress_rotate_every) % len(rotate_groups)
            candidate = rotate_groups[group_idx]
        else:
            candidate = compact_keys

        for key in candidate:
            if key in full_postfix and key not in ordered:
                ordered.append(key)
            if len(ordered) >= self.progress_max_fields:
                break

        shown_keys = set(ordered)
        display = {k: full_postfix[k] for k in ordered}

        if self.progress_alias:
            alias = {
                "grad_norm": "g",
                "obj": "obj",
                "cycles": "cyc",
                "copy_ce": "cce",
                "ae_ce": "ae",
                "ae_next": "aen",
                "unet_lh": "ulh",
                "haux": "haux",
                "min_prob": "pmin",
                "cycles_ema": "cema",
                "max_cycle_rate": "mcr",
                "c_hit": "ch",
                "c_miss": "cm",
                "c_rst": "cr",
                "c_seed": "cs",
            }
            display = {alias.get(k, k): v for k, v in display.items()}

        return display, shown_keys
    

    def train_epoch(self, data_provider, epoch: int) -> float:
        """
        Train for one epoch with hybrid masking.
        Handles optimizer steps and gradient accumulation correctly.
        Includes gradient norm checking.

        Args:
            data_provider: Either a DataLoader or a dict with get_batch function
            epoch: Current epoch number

        Returns:
            epoch_loss: Average loss for the epoch (per step)
        """
        self.model.train()
        # Record the epoch we are actually in, so save_checkpoint() stamps the real number.
        # Without this it stays at its __init__ value (0) forever and every resume restarts
        # the epoch counter from scratch.
        self.current_epoch = int(epoch)
        total_loss = 0.0
        steps_processed_in_epoch = 0
        last_grad_norm = 0.0 # Variable to hold the last computed grad norm for display
        self.ema_model

        # Determine data iteration setup
        if isinstance(data_provider, dict) and "get_batch" in data_provider:
            get_batch = data_provider["get_batch"]
            steps_per_epoch = data_provider.get("steps_per_epoch", 100)
            data_iterator = range(steps_per_epoch)
            desc = f"Epoch {epoch} (Steps)"
            use_get_batch = True
        elif isinstance(data_provider, torch.utils.data.DataLoader):
            dataloader = data_provider
            steps_per_epoch = len(dataloader)
            data_iterator = dataloader
            desc = f"Epoch {epoch} (Batches)"
            use_get_batch = False
        else:
            raise ValueError("Invalid data_provider format.")

        pbar = tqdm(data_iterator, desc=desc, total=steps_per_epoch, dynamic_ncols=True)

        def _sync_if_cuda():
            if getattr(self.device, "type", str(self.device)) == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    pass

        # Image batches carry pixels, not token ids — a sample still costs the model one
        # full token grid (VQ/latent H*W), so count that for tok/s comparability with text.
        image_tokens_per_sample = 0
        if self.modality == "image":
            try:
                vq = getattr(self, "_image_maskgit_vq_tokenizer", None)
                if vq is not None and hasattr(vq, "infer_grid_shape"):
                    gh, gw = vq.infer_grid_shape(int(self.image_size))
                else:
                    gh, gw = self._image_expected_grid_shape()
                image_tokens_per_sample = max(1, int(gh) * int(gw))
            except Exception:
                image_tokens_per_sample = 0

        def _batch_token_sample_counts(batch_obj) -> Tuple[int, int]:
            input_tensor = None
            if isinstance(batch_obj, dict):
                input_tensor = batch_obj.get("input_ids", None)
                if input_tensor is None:
                    # cached VQ tokens [B,T]
                    t = batch_obj.get("token_ids", None)
                    if torch.is_tensor(t) and t.dim() >= 2:
                        return int(t.size(0) * t.size(1)), max(1, int(t.size(0)))
                    # cached latent/patch features [B,T,C] — count positions, not floats
                    t = batch_obj.get("input_features", None)
                    if torch.is_tensor(t) and t.dim() >= 2:
                        return int(t.size(0) * t.size(1)), max(1, int(t.size(0)))
                    # raw pixels [B,3,H,W] — one token grid per image
                    t = batch_obj.get("pixel_values", None)
                    if torch.is_tensor(t) and t.dim() > 0:
                        b = int(t.size(0))
                        return b * int(image_tokens_per_sample), max(1, b)
            elif isinstance(batch_obj, (list, tuple)) and len(batch_obj) > 0:
                input_tensor = batch_obj[0]
            if torch.is_tensor(input_tensor):
                samples = int(input_tensor.size(0)) if input_tensor.dim() > 0 else 1
                return int(input_tensor.numel()), max(1, samples)
            return 0, 0

        def _fmt_rate(value: float) -> str:
            value = float(value)
            if value >= 1_000_000:
                return f"{value / 1_000_000:.2f}M"
            if value >= 1_000:
                return f"{value / 1_000:.1f}k"
            return f"{value:.1f}"

        def _to_metric_float(value):
            if value is None:
                return None
            try:
                if torch.is_tensor(value):
                    value = value.detach().float().item()
                return float(value)
            except (TypeError, ValueError, RuntimeError):
                return None

        def _emit_train_metrics(
            *,
            step_loss,
            ce_loss_value,
            objective_loss_value,
            copy_ce_loss_value,
            autoenc_ce_loss_value,
            autoenc_next_ce_loss_value,
            unet_lh_ce_loss_value,
            grad_norm_value,
            tokens_per_sec_value,
            tokens_per_sec_window_value,
            samples_per_sec_value,
            samples_per_sec_window_value,
            epoch_value,
            step_value,
            steps_per_epoch_value,
            tokens_epoch_value,
            samples_epoch_value,
            cycles_value=None,
        ):
            callback = getattr(self, "train_metrics_callback", None)
            interval = int(getattr(self, "train_metrics_interval", 0) or 0)
            if callback is None or interval <= 0:
                return
            if int(self.global_step) % interval != 0:
                return

            payload = {
                "global_step": int(self.global_step),
                "epoch": int(epoch_value),
                "train/epoch_step": int(step_value),
                "train/steps_per_epoch": int(steps_per_epoch_value),
                "train/loss": float(step_loss),
                "train/grad_norm": float(grad_norm_value),
                "train/lr": float(self.lr_scheduler.get_last_lr()[0]) if self.lr_scheduler else None,
                "train/tokens_per_sec": float(tokens_per_sec_value),
                "train/tokens_per_sec_window": float(tokens_per_sec_window_value),
                "train/samples_per_sec": float(samples_per_sec_value),
                "train/samples_per_sec_window": float(samples_per_sec_window_value),
                "train/tokens_epoch": int(tokens_epoch_value),
                "train/samples_epoch": int(samples_epoch_value),
            }
            optional_values = {
                "train/ce_loss": ce_loss_value,
                "train/objective_loss": objective_loss_value,
                "train/copy_dst_ce_loss": copy_ce_loss_value,
                "train/autoenc_ce_loss": autoenc_ce_loss_value,
                "train/autoenc_next_ce_loss": autoenc_next_ce_loss_value,
                "train/token_unet_lookahead_ce_loss": unet_lh_ce_loss_value,
            }
            for key, value in optional_values.items():
                metric_value = _to_metric_float(value)
                if metric_value is not None:
                    payload[key] = metric_value
            if cycles_value is not None:
                try:
                    payload["train/cycles_used"] = int(cycles_value)
                except (TypeError, ValueError):
                    pass

            model_for_metrics = getattr(self.model, "module", self.model)
            for attr, key in (
                ("_last_sampled_edge_attr_batches", "sampled_edge_attr/batches"),
                ("_last_sampled_edge_attr_missing_eid", "sampled_edge_attr/missing_eid_batches"),
                ("_last_sampled_edge_attr_writebacks", "sampled_edge_attr/writebacks"),
            ):
                value = getattr(model_for_metrics, attr, None)
                if value is not None:
                    payload[key] = int(value)

            if getattr(self.device, "type", str(self.device)) == "cuda" and torch.cuda.is_available():
                try:
                    payload["train/peak_vram"] = int(torch.cuda.max_memory_allocated(self.device))
                except Exception:
                    pass
            try:
                callback(payload)
            except Exception as e:
                logger.warning("Train metrics callback failed; continuing without interrupting training: %s", e)

        _sync_if_cuda()
        throughput_start_time = time.perf_counter()
        throughput_last_log_time = throughput_start_time
        throughput_tokens = 0
        throughput_samples = 0
        throughput_window_tokens = 0
        throughput_window_samples = 0
        self._last_train_tokens_per_sec = 0.0
        self._last_train_samples_per_sec = 0.0
        self._last_train_tokens_per_sec_window = 0.0
        self._last_train_samples_per_sec_window = 0.0

        if self.gradient_accumulation_steps > 1:
            self.optimizer.zero_grad(set_to_none=True)

        def _ensure_muon_late_param_state() -> None:
            """Muon initializes group state only once; memory params can get grads later."""
            opt = getattr(self, "optimizer", None)
            if opt is None:
                return
            if not any("use_muon" in group for group in getattr(opt, "param_groups", [])):
                return
            for group in opt.param_groups:
                use_muon = bool(group.get("use_muon", False))
                for p in group.get("params", []):
                    if p.grad is None:
                        continue
                    state = opt.state[p]
                    if use_muon:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(p)
                        if "exp_avg_sq" not in state:
                            state["exp_avg_sq"] = torch.zeros_like(p)

        model_mod_epoch = getattr(self.model, "module", None)
        auto_prob_enabled_static = bool(getattr(self.model, "time_dilation_auto_prob_enable", False))
        if (not auto_prob_enabled_static) and model_mod_epoch is not None:
            auto_prob_enabled_static = bool(getattr(model_mod_epoch, "time_dilation_auto_prob_enable", False))
        cycle_governor_enabled_static = bool(getattr(self.model, "cycle_governor_enable", False))
        if (not cycle_governor_enabled_static) and model_mod_epoch is not None:
            cycle_governor_enabled_static = bool(getattr(model_mod_epoch, "cycle_governor_enable", False))

        for step_or_batch_idx, batch_data in enumerate(pbar):
            current_step_index = step_or_batch_idx

            if use_get_batch:
                batch = get_batch(self.device)
            else:
                batch = batch_data # Batch data is directly from iterator

            if isinstance(batch, dict):
                batch = self._maybe_apply_copy_task(batch, is_validation=False)

            loss = self._process_batch_with_hybrid_masking(batch, use_bf16=self.use_bf16)
            if loss is not None:
                total_loss += loss
                steps_processed_in_epoch += 1
                batch_tokens, batch_samples = _batch_token_sample_counts(batch)
                throughput_tokens += int(batch_tokens)
                throughput_samples += int(batch_samples)
                throughput_window_tokens += int(batch_tokens)
                throughput_window_samples += int(batch_samples)
            else: 
                continue

            # if (current_step_index + 1) % self.gradient_accumulation_steps == 0:
            #     if self.mixed_precision and self.scaler:
            #         self.scaler.unscale_(self.optimizer)

            #     # --- Start Inserted Gradient Check ---
            #     total_norm = 0
            #     for p in self.model.parameters():
            #         if p.grad is not None:
            #             param_norm = p.grad.data.norm(2)
            #             total_norm += param_norm.item() ** 2
            #     last_grad_norm = total_norm ** 0.5
            #     if self.global_step % (self.log_interval * self.gradient_accumulation_steps) == 0:
            #         # Using print directly for immediate feedback during debugging
            #         print(f"\nStep {self.global_step}, Grad Norm (Before Clip): {last_grad_norm:.4f}\n")
            #     # --- End Inserted Gradient Check ---

            #     if self.max_grad_norm > 0:
            #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            #     if self.mixed_precision and self.scaler and loss is not None:
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            #     else:
            #         self.optimizer.step()

            #     self.optimizer.zero_grad(set_to_none=True)

            #     # Update EMA parameters
            #     true_step = self.global_step / max(1, self.gradient_accumulation_steps)
            #     ema_decay = min(0.995, 0.95 + 0.0001 * true_step)  # Faster ramp, stops at 0.995
            #     with torch.no_grad():
            #         for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            #             ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)

            #     if self.lr_scheduler:
            #         self.lr_scheduler.step()
            if (current_step_index + 1) % self.gradient_accumulation_steps == 0:

                # 1) UN-SCALE before any grad ops (clip/norm/EMA sanity checks)
                if self.mixed_precision and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                # 2-3) Compute grad norm and optional clipping in one pass
                params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
                if params_with_grad:
                    clip_val = float(self.max_grad_norm) if (self.max_grad_norm and self.max_grad_norm > 0) else float("inf")
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(params_with_grad, clip_val)
                    last_grad_norm = float(grad_norm_tensor.detach().item()) if torch.is_tensor(grad_norm_tensor) else float(grad_norm_tensor)
                else:
                    last_grad_norm = 0.0

                # 4) optimizer step (AMP-aware)
                _ensure_muon_late_param_state()
                if self.mixed_precision and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # 5) zero grads
                self.optimizer.zero_grad(set_to_none=True)

                # 6) EMA update (device-safe)
                #    only move a tiny tensor view if devices differ
                if hasattr(self, "ema_model") and (self.ema_model is not None):
                    ema_dev = next(self.ema_model.parameters()).device
                    true_step = self.global_step / max(1, self.gradient_accumulation_steps)
                    ema_decay = min(0.995, 0.95 + 0.0001 * true_step)
                    with torch.no_grad():
                        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                            mp = model_param.data
                            if mp.device != ema_dev:
                                mp = mp.to(ema_dev, non_blocking=True)
                            ema_param.data.mul_(ema_decay).add_(mp, alpha=1.0 - ema_decay)

                # 7) LR schedule after step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()


            model_mod_quick = getattr(self.model, "module", None)
            ce_quick = getattr(self.model, "_last_ce_loss", None)
            if ce_quick is None and model_mod_quick is not None:
                ce_quick = getattr(model_mod_quick, "_last_ce_loss", None)
            copy_ce_quick = getattr(self.model, "_last_copy_dst_ce_loss", None)
            copy_tok_quick = getattr(self.model, "_last_copy_dst_token_count", None)
            if copy_ce_quick is None and model_mod_quick is not None:
                copy_ce_quick = getattr(model_mod_quick, "_last_copy_dst_ce_loss", None)
            if copy_tok_quick is None and model_mod_quick is not None:
                copy_tok_quick = getattr(model_mod_quick, "_last_copy_dst_token_count", None)
            ae_ce_quick = getattr(self.model, "_last_autoenc_ce_loss", None)
            if ae_ce_quick is None and model_mod_quick is not None:
                ae_ce_quick = getattr(model_mod_quick, "_last_autoenc_ce_loss", None)
            ae_next_quick = getattr(self.model, "_last_autoenc_next_ce_loss", None)
            if ae_next_quick is None and model_mod_quick is not None:
                ae_next_quick = getattr(model_mod_quick, "_last_autoenc_next_ce_loss", None)
            unet_lh_quick = getattr(self.model, "_last_token_unet_lookahead_ce_loss", None)
            if unet_lh_quick is None and model_mod_quick is not None:
                unet_lh_quick = getattr(model_mod_quick, "_last_token_unet_lookahead_ce_loss", None)
            obj_quick = getattr(self.model, "_last_objective_loss", None)
            if obj_quick is None and model_mod_quick is not None:
                obj_quick = getattr(model_mod_quick, "_last_objective_loss", None)
            haux_quick = getattr(self.model, "_last_hier_aux_loss", None)
            if haux_quick is None and model_mod_quick is not None:
                haux_quick = getattr(model_mod_quick, "_last_hier_aux_loss", None)
            cycles_quick = getattr(self.model, "_last_cycles_used", None)
            if cycles_quick is None and model_mod_quick is not None:
                cycles_quick = getattr(model_mod_quick, "_last_cycles_used", None)
            img_pred_std_quick = getattr(self.model, "_last_image_pred_std", None)
            img_target_std_quick = getattr(self.model, "_last_image_target_std", None)
            img_t_mean_quick = getattr(self.model, "_last_image_t_mean", None)
            if img_pred_std_quick is None and model_mod_quick is not None:
                img_pred_std_quick = getattr(model_mod_quick, "_last_image_pred_std", None)
            if img_target_std_quick is None and model_mod_quick is not None:
                img_target_std_quick = getattr(model_mod_quick, "_last_image_target_std", None)
            if img_t_mean_quick is None and model_mod_quick is not None:
                img_t_mean_quick = getattr(model_mod_quick, "_last_image_t_mean", None)

            if current_step_index % max(1, self.log_interval) == 0:
                _sync_if_cuda()
            throughput_now = time.perf_counter()
            throughput_elapsed = max(1e-9, throughput_now - throughput_start_time)
            throughput_window_elapsed = max(1e-9, throughput_now - throughput_last_log_time)
            tokens_per_sec_live = float(throughput_tokens) / throughput_elapsed
            samples_per_sec_live = float(throughput_samples) / throughput_elapsed
            tokens_per_sec_window = float(throughput_window_tokens) / throughput_window_elapsed
            samples_per_sec_window = float(throughput_window_samples) / throughput_window_elapsed
            self._last_train_tokens_per_sec = float(tokens_per_sec_live)
            self._last_train_samples_per_sec = float(samples_per_sec_live)
            self._last_train_tokens_per_sec_window = float(tokens_per_sec_window)
            self._last_train_samples_per_sec_window = float(samples_per_sec_window)

            _emit_train_metrics(
                step_loss=loss,
                ce_loss_value=ce_quick,
                objective_loss_value=obj_quick,
                copy_ce_loss_value=copy_ce_quick,
                autoenc_ce_loss_value=ae_ce_quick,
                autoenc_next_ce_loss_value=ae_next_quick,
                unet_lh_ce_loss_value=unet_lh_quick,
                grad_norm_value=last_grad_norm,
                tokens_per_sec_value=tokens_per_sec_live,
                tokens_per_sec_window_value=tokens_per_sec_window,
                samples_per_sec_value=samples_per_sec_live,
                samples_per_sec_window_value=samples_per_sec_window,
                epoch_value=epoch,
                step_value=current_step_index,
                steps_per_epoch_value=steps_per_epoch,
                tokens_epoch_value=throughput_tokens,
                samples_epoch_value=throughput_samples,
                cycles_value=cycles_quick,
            )

            quick_postfix = {
                'loss': loss,
                'lr': self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 'N/A',
                'grad_norm': f"{last_grad_norm:.2f}",
                'tok/s': _fmt_rate(tokens_per_sec_live),
            }
            if ce_quick is not None:
                quick_postfix['ce'] = f"{float(ce_quick):.3f}"
            if copy_ce_quick is not None and (copy_tok_quick is None or int(copy_tok_quick) > 0):
                quick_postfix['copy_ce'] = f"{float(copy_ce_quick):.3f}"
            if ae_ce_quick is not None:
                quick_postfix['ae_ce'] = f"{float(ae_ce_quick):.3f}"
            if ae_next_quick is not None:
                quick_postfix['ae_next'] = f"{float(ae_next_quick):.3f}"
            if unet_lh_quick is not None:
                quick_postfix['unet_lh'] = f"{float(unet_lh_quick):.3f}"
            if obj_quick is not None:
                quick_postfix['obj'] = f"{float(obj_quick):.3f}"
            if haux_quick is not None:
                if torch.is_tensor(haux_quick):
                    quick_postfix['haux'] = f"{float(haux_quick.detach().item()):.3f}"
                else:
                    quick_postfix['haux'] = f"{float(haux_quick):.3f}"
            if cycles_quick is not None:
                quick_postfix['cycles'] = int(cycles_quick)
            if img_pred_std_quick is not None:
                quick_postfix['p_std'] = f"{float(img_pred_std_quick):.3f}"
            if img_target_std_quick is not None:
                quick_postfix['t_std'] = f"{float(img_target_std_quick):.3f}"
            if img_t_mean_quick is not None:
                quick_postfix['t_mean'] = f"{float(img_t_mean_quick):.1f}"

            if current_step_index % self.progress_update_every == 0:
                quick_display_postfix, _ = self._format_progress_postfix(quick_postfix, current_step_index)
                pbar.set_postfix(quick_display_postfix)

            collect_heavy_metrics = (
                (current_step_index % self.log_interval == 0)
                or (self.progress_view != "full" and (current_step_index % self.progress_detail_interval == 0))
            )
            if not collect_heavy_metrics:
                self.global_step += 1
                continue

            zip_edges = getattr(self.model, "_last_zip_added_total", None)
            hqd_edges = getattr(self.model, "_last_hqd_added_total", None)
            hqd_avg_l0 = getattr(self.model, "_last_hqd_avg_l0", None)
            attn_saved_pct = getattr(self.model, "_last_attn_saved_pct", None)
            win_dense_pct = getattr(self.model, "_last_win_dense_pct", None)
            sparse_dense_pct = getattr(self.model, "_last_sparse_dense_pct", None)
            graph_l0_edges = getattr(self.model, "_last_graph_l0_edges", None)
            l0_attn_is_causal = getattr(self.model, "_last_l0_attn_is_causal", None)
            tick_mean = getattr(self.model, "_last_tick_count", None)
            gamma_mean = getattr(self.model, "_last_gamma_mean", None)
            gamma_max = getattr(self.model, "_last_gamma_max", None)
            cycles_used = getattr(self.model, "_last_cycles_used", None)
            min_prob = getattr(self.model, "_last_auto_prob_min_prob", None)
            cycles_ema = getattr(self.model, "_last_auto_prob_cycles_ema", None)
            max_cycle_rate = getattr(self.model, "_last_auto_prob_max_cycle_rate", None)
            cycle_metric_p80 = getattr(self.model, "_last_cycle_metric_p80", None)
            cycle_metric_ema = getattr(self.model, "_last_cycle_metric_ema", None)
            haux_mean = getattr(self.model, "_last_hier_aux_loss", None)
            cache_hits = getattr(self.model, "_uf_cache_fast_hits", None)
            cache_misses = getattr(self.model, "_uf_cache_fast_misses", None)
            cache_restarts = getattr(self.model, "_uf_cache_restart_count", None)
            cache_seeds = getattr(self.model, "_uf_cache_seed_count", None)
            cache_last_seq = getattr(self.model, "_uf_cache_last_seq_len", None)
            cache_build_ms = getattr(self.model, "_uf_cache_last_build_ms", None)
            cache_rehydrate_ms = getattr(self.model, "_uf_cache_last_rehydrate_ms", None)
            cache_last_reason = getattr(self.model, "_uf_cache_last_miss_reason", None)
            ce_loss = getattr(self.model, "_last_ce_loss", None)
            copy_ce_loss = getattr(self.model, "_last_copy_dst_ce_loss", None)
            copy_ce_tokens = getattr(self.model, "_last_copy_dst_token_count", None)
            autoenc_ce_loss = getattr(self.model, "_last_autoenc_ce_loss", None)
            autoenc_next_ce_loss = getattr(self.model, "_last_autoenc_next_ce_loss", None)
            unet_lh_ce_loss = getattr(self.model, "_last_token_unet_lookahead_ce_loss", None)
            objective_loss = getattr(self.model, "_last_objective_loss", None)
            model_mod = model_mod_epoch
            auto_prob_enabled = auto_prob_enabled_static
            cycle_governor_enabled = cycle_governor_enabled_static

            if (
                (zip_edges is None and hqd_edges is None)
                or tick_mean is None
                or (
                    auto_prob_enabled
                    and (min_prob is None or cycles_ema is None or max_cycle_rate is None)
                )
                or (
                    cycle_governor_enabled
                    and (cycle_metric_p80 is None or cycle_metric_ema is None)
                )
            ):
                if model_mod is not None:
                    if zip_edges is None:
                        zip_edges = getattr(model_mod, "_last_zip_added_total", None)
                    if hqd_edges is None:
                        hqd_edges = getattr(model_mod, "_last_hqd_added_total", None)
                    if hqd_avg_l0 is None:
                        hqd_avg_l0 = getattr(model_mod, "_last_hqd_avg_l0", None)
                    if attn_saved_pct is None:
                        attn_saved_pct = getattr(model_mod, "_last_attn_saved_pct", None)
                    if win_dense_pct is None:
                        win_dense_pct = getattr(model_mod, "_last_win_dense_pct", None)
                    if sparse_dense_pct is None:
                        sparse_dense_pct = getattr(model_mod, "_last_sparse_dense_pct", None)
                    if graph_l0_edges is None:
                        graph_l0_edges = getattr(model_mod, "_last_graph_l0_edges", None)
                    if l0_attn_is_causal is None:
                        l0_attn_is_causal = getattr(model_mod, "_last_l0_attn_is_causal", None)
                    if tick_mean is None:
                        tick_mean = getattr(model_mod, "_last_tick_count", None)
                    if gamma_mean is None:
                        gamma_mean = getattr(model_mod, "_last_gamma_mean", None)
                    if gamma_max is None:
                        gamma_max = getattr(model_mod, "_last_gamma_max", None)
                    if cycles_used is None:
                        cycles_used = getattr(model_mod, "_last_cycles_used", None)
                    if auto_prob_enabled and min_prob is None:
                        min_prob = getattr(model_mod, "_last_auto_prob_min_prob", None)
                    if auto_prob_enabled and cycles_ema is None:
                        cycles_ema = getattr(model_mod, "_last_auto_prob_cycles_ema", None)
                    if auto_prob_enabled and max_cycle_rate is None:
                        max_cycle_rate = getattr(model_mod, "_last_auto_prob_max_cycle_rate", None)
                    if cycle_governor_enabled and cycle_metric_p80 is None:
                        cycle_metric_p80 = getattr(model_mod, "_last_cycle_metric_p80", None)
                    if cycle_governor_enabled and cycle_metric_ema is None:
                        cycle_metric_ema = getattr(model_mod, "_last_cycle_metric_ema", None)
                    if haux_mean is None:
                        haux_mean = getattr(model_mod, "_last_hier_aux_loss", None)
                    if cache_hits is None:
                        cache_hits = getattr(model_mod, "_uf_cache_fast_hits", None)
                    if cache_misses is None:
                        cache_misses = getattr(model_mod, "_uf_cache_fast_misses", None)
                    if cache_restarts is None:
                        cache_restarts = getattr(model_mod, "_uf_cache_restart_count", None)
                    if cache_seeds is None:
                        cache_seeds = getattr(model_mod, "_uf_cache_seed_count", None)
                    if cache_last_seq is None:
                        cache_last_seq = getattr(model_mod, "_uf_cache_last_seq_len", None)
                    if cache_build_ms is None:
                        cache_build_ms = getattr(model_mod, "_uf_cache_last_build_ms", None)
                    if cache_rehydrate_ms is None:
                        cache_rehydrate_ms = getattr(model_mod, "_uf_cache_last_rehydrate_ms", None)
                    if cache_last_reason is None:
                        cache_last_reason = getattr(model_mod, "_uf_cache_last_miss_reason", None)
                    if ce_loss is None:
                        ce_loss = getattr(model_mod, "_last_ce_loss", None)
                    if copy_ce_loss is None:
                        copy_ce_loss = getattr(model_mod, "_last_copy_dst_ce_loss", None)
                    if copy_ce_tokens is None:
                        copy_ce_tokens = getattr(model_mod, "_last_copy_dst_token_count", None)
                    if autoenc_ce_loss is None:
                        autoenc_ce_loss = getattr(model_mod, "_last_autoenc_ce_loss", None)
                    if autoenc_next_ce_loss is None:
                        autoenc_next_ce_loss = getattr(model_mod, "_last_autoenc_next_ce_loss", None)
                    if unet_lh_ce_loss is None:
                        unet_lh_ce_loss = getattr(model_mod, "_last_token_unet_lookahead_ce_loss", None)
                    if objective_loss is None:
                        objective_loss = getattr(model_mod, "_last_objective_loss", None)
            postfix = {
                'loss': loss,
                'lr': self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 'N/A',
                'grad_norm': f"{last_grad_norm:.2f}",
                'tok/s': _fmt_rate(tokens_per_sec_live),
                'tok/s_win': _fmt_rate(tokens_per_sec_window),
            }
            if objective_loss is not None:
                postfix['obj'] = f"{float(objective_loss):.3f}"
            if ce_loss is not None:
                postfix['ce'] = f"{float(ce_loss):.3f}"
            if copy_ce_loss is not None and (copy_ce_tokens is None or int(copy_ce_tokens) > 0):
                postfix['copy_ce'] = f"{float(copy_ce_loss):.3f}"
            if autoenc_ce_loss is not None:
                postfix['ae_ce'] = f"{float(autoenc_ce_loss):.3f}"
            if autoenc_next_ce_loss is not None:
                postfix['ae_next'] = f"{float(autoenc_next_ce_loss):.3f}"
            if unet_lh_ce_loss is not None:
                postfix['unet_lh'] = f"{float(unet_lh_ce_loss):.3f}"
            if zip_edges is not None:
                postfix['zip_edges'] = int(zip_edges)
            if hqd_edges is not None:
                postfix['hqd_edges'] = int(hqd_edges)
            if hqd_avg_l0 is not None:
                postfix['hqd_l0_avg'] = f"{float(hqd_avg_l0):.1f}"
            if tick_mean is not None:
                try:
                    postfix['ticks'] = f"{float(tick_mean):.2f}"
                except (TypeError, ValueError):
                    postfix['ticks'] = tick_mean
            if gamma_mean is not None:
                postfix['gamma_mean'] = f"{float(gamma_mean):.2f}"
            if gamma_max is not None:
                postfix['gamma_max'] = f"{float(gamma_max):.2f}"
            if cycles_used is not None:
                postfix['cycles'] = int(cycles_used)
            if auto_prob_enabled and min_prob is not None:
                postfix['min_prob'] = f"{float(min_prob):.4f}"
            if auto_prob_enabled and cycles_ema is not None:
                postfix['cycles_ema'] = f"{float(cycles_ema):.2f}"
            if auto_prob_enabled and max_cycle_rate is not None:
                postfix['max_cycle_rate'] = f"{float(max_cycle_rate):.2f}"
            if cycle_governor_enabled and cycle_metric_p80 is not None:
                postfix['cyc_p80'] = f"{float(cycle_metric_p80):.3f}"
            if cycle_governor_enabled and cycle_metric_ema is not None:
                postfix['cyc_gov_ema'] = f"{float(cycle_metric_ema):.3f}"
            if haux_mean is not None:
                if torch.is_tensor(haux_mean):
                    postfix['haux'] = f"{float(haux_mean.detach().item()):.3f}"
                else:
                    postfix['haux'] = f"{float(haux_mean):.3f}"
            if cache_hits is not None and cache_misses is not None:
                cache_calls = float(cache_hits + cache_misses)
                if cache_calls > 0:
                    postfix['c_hit'] = f"{float(cache_hits) / cache_calls:.2f}"
                postfix['c_miss'] = int(cache_misses)
            if cache_restarts is not None:
                postfix['c_rst'] = int(cache_restarts)
            if cache_seeds is not None:
                postfix['c_seed'] = int(cache_seeds)
            display_postfix, shown_progress_keys = self._format_progress_postfix(postfix, current_step_index)
            if current_step_index % self.progress_update_every == 0:
                pbar.set_postfix(display_postfix)

            if current_step_index % self.log_interval == 0:
                ce_loss_str = f"{float(ce_loss):.4f}" if ce_loss is not None else "N/A"
                copy_ce_str = f"{float(copy_ce_loss):.4f}" if copy_ce_loss is not None else "N/A"
                autoenc_ce_str = f"{float(autoenc_ce_loss):.4f}" if autoenc_ce_loss is not None else "N/A"
                autoenc_next_ce_str = f"{float(autoenc_next_ce_loss):.4f}" if autoenc_next_ce_loss is not None else "N/A"
                logger.info(
                    f"Epoch: {epoch}, Step: {current_step_index}/{steps_per_epoch}, "
                    f"Micro-Batch Loss: {loss:.4f}, CE: {ce_loss_str}, Copy-CE: {copy_ce_str}, AE-CE: {autoenc_ce_str}, AE-Next-CE: {autoenc_next_ce_str}, "
                    f"LR: {self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 'N/A'}, "
                    f"Tok/s: {_fmt_rate(tokens_per_sec_live)}, Tok/s(win): {_fmt_rate(tokens_per_sec_window)}, "
                    f"Samples/s: {_fmt_rate(samples_per_sec_live)}"
                )
                if hqd_edges is not None or attn_saved_pct is not None:
                    causal_str = "Y" if l0_attn_is_causal else "N"
                    hqd_edges_str = str(int(hqd_edges)) if hqd_edges is not None else "N/A"
                    hqd_avg_str = f"{float(hqd_avg_l0):.1f}" if hqd_avg_l0 is not None else "N/A"
                    attn_saved_str = f"{float(attn_saved_pct):.1f}%" if attn_saved_pct is not None else "N/A"
                    win_dense_str = f"{float(win_dense_pct):.1f}%" if win_dense_pct is not None else "N/A"
                    sparse_dense_str = f"{float(sparse_dense_pct):.1f}%" if sparse_dense_pct is not None else "N/A"
                    graph_l0_str = str(int(graph_l0_edges)) if graph_l0_edges is not None else "N/A"
                    logger.info(
                        f"EFF: hqd_edges={hqd_edges_str} hqd_l0_avg={hqd_avg_str} "
                        f"attn_saved={attn_saved_str} win_dense={win_dense_str} "
                        f"sparse_dense={sparse_dense_str} graph_l0={graph_l0_str} causal={causal_str}"
                    )
                throughput_last_log_time = time.perf_counter()
                throughput_window_tokens = 0
                throughput_window_samples = 0
                if cache_hits is not None and cache_misses is not None:
                    ch = int(cache_hits)
                    cm = int(cache_misses)
                    cr = int(cache_restarts) if cache_restarts is not None else 0
                    cs = int(cache_seeds) if cache_seeds is not None else 0
                    dch = ch - int(self._cache_prev_hits)
                    dcm = cm - int(self._cache_prev_misses)
                    dcr = cr - int(self._cache_prev_restarts)
                    dcs = cs - int(self._cache_prev_seeds)
                    self._cache_prev_hits = ch
                    self._cache_prev_misses = cm
                    self._cache_prev_restarts = cr
                    self._cache_prev_seeds = cs
                    calls = ch + cm
                    hit_rate = (float(ch) / float(calls)) if calls > 0 else 0.0
                    logger.info(
                        "Cache: hit=%d miss=%d restart=%d seed=%d | d_hit=%d d_miss=%d d_restart=%d d_seed=%d | hit_rate=%.3f | seq=%s build_ms=%s rehydrate_ms=%s reason=%s",
                        ch,
                        cm,
                        cr,
                        cs,
                        dch,
                        dcm,
                        dcr,
                        dcs,
                        hit_rate,
                        str(cache_last_seq),
                        f"{float(cache_build_ms):.2f}" if cache_build_ms is not None else "n/a",
                        f"{float(cache_rehydrate_ms):.2f}" if cache_rehydrate_ms is not None else "n/a",
                        str(cache_last_reason) if cache_last_reason is not None else "n/a",
                    )
                zip_profile = getattr(self.model, "_last_zip_profile_stats", None)
                if zip_profile is None and model_mod is not None:
                    zip_profile = getattr(model_mod, "_last_zip_profile_stats", None)
                if zip_profile and isinstance(zip_profile, dict):
                    parts = [f"{k}={v:.1f}" for k, v in zip_profile.items()]
                    zip_ch = getattr(model_mod or self.model, "_zipper_children_cache_hits", None)
                    zip_cm = getattr(model_mod or self.model, "_zipper_children_cache_misses", None)
                    if zip_ch is not None and zip_cm is not None:
                        parts.append(f"ct_hit={zip_ch}")
                        parts.append(f"ct_miss={zip_cm}")
                    logger.info("[ZIP profile] %s", " ".join(parts))
                hqd_profile = getattr(self.model, "_last_hqd_profile_stats", None)
                if hqd_profile is None and model_mod is not None:
                    hqd_profile = getattr(model_mod, "_last_hqd_profile_stats", None)
                if hqd_profile and isinstance(hqd_profile, dict):
                    parts = [f"{k}={float(v):.1f}" for k, v in hqd_profile.items()]
                    logger.info("[HQD profile] %s", " ".join(parts))
                if self.progress_view != "full" and (current_step_index % self.progress_detail_interval == 0):
                    hidden_metrics = [
                        f"{k}={v}"
                        for k, v in postfix.items()
                        if k not in shown_progress_keys
                    ]
                    if hidden_metrics:
                        logger.info("Metrics: " + ", ".join(hidden_metrics))
            # clear cache every 10 steps to avoid memory issues
            # if current_step_index % 10 == 0:
            #     torch.cuda.empty_cache()
            self.global_step += 1

        _sync_if_cuda()
        throughput_elapsed_final = max(1e-9, time.perf_counter() - throughput_start_time)
        self._last_train_tokens_per_sec = float(throughput_tokens) / throughput_elapsed_final
        self._last_train_samples_per_sec = float(throughput_samples) / throughput_elapsed_final
        pbar.close()

        epoch_loss = total_loss / steps_processed_in_epoch if steps_processed_in_epoch > 0 else 0.0
        self.train_losses.append(epoch_loss)

        return epoch_loss

    def _recurrent_base_model(self, model=None):
        model = self.model if model is None else model
        return getattr(model, "module", model)

    def _set_recurrent_train_stats(self, ce_loss: Optional[float], objective_loss: Optional[float], token_count: int = 0) -> None:
        targets = [self.model]
        model_mod = getattr(self.model, "module", None)
        if model_mod is not None:
            targets.append(model_mod)
        for tgt in targets:
            tgt._last_ce_loss = None if ce_loss is None else float(ce_loss)
            tgt._last_objective_loss = None if objective_loss is None else float(objective_loss)
            tgt._last_recurrent_ce_loss = None if ce_loss is None else float(ce_loss)
            tgt._last_recurrent_token_count = int(token_count)
            tgt._last_copy_dst_ce_loss = None
            tgt._last_copy_dst_token_count = 0
            tgt._last_autoenc_ce_loss = None
            tgt._last_autoenc_next_ce_loss = None
            tgt._last_token_unet_lookahead_ce_loss = None

    def _process_batch_recurrent_ar(self, batch, use_bf16=True):
        """True recurrent teacher-forced AR training path (opt-in only)."""
        if not isinstance(batch, dict) and isinstance(batch, (list, tuple)) and len(batch) > 0:
            batch = {"input_ids": batch[0]}
        if not isinstance(batch, dict) or "input_ids" not in batch:
            return None
        input_ids = batch["input_ids"].to(self.device)
        if input_ids.dim() != 2 or input_ids.size(1) < 2:
            self._set_recurrent_train_stats(None, None, 0)
            return None

        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        B, T = input_ids.shape
        target_valid = torch.ones((B, T - 1), device=self.device, dtype=torch.bool)
        if attention_mask is not None:
            target_valid = attention_mask[:, 1:].to(device=self.device, dtype=torch.bool)
        positions = torch.arange(T - 1, device=self.device, dtype=torch.long)
        warmup = int(self.recurrent_warmup_tokens)
        stride = max(1, int(self.recurrent_loss_stride))
        supervise_pos = (positions >= warmup) & (((positions - warmup).clamp_min(0) % stride) == 0)
        target_valid = target_valid & supervise_pos.view(1, -1)
        total_valid = int(target_valid.sum().item())
        if total_valid <= 0:
            self._set_recurrent_train_stats(None, None, 0)
            return None

        model_for_recurrent = self._recurrent_base_model(self.model)
        if not hasattr(model_for_recurrent, "init_recurrent_state") or not hasattr(model_for_recurrent, "recurrent_chunk_forward"):
            raise AttributeError("recurrent_training_enable requires a model with recurrent state APIs")

        chunk_size = max(1, int(self.recurrent_unroll_len))
        if int(self.recurrent_l0_window or 0) > 0 and chunk_size > int(self.recurrent_l0_window):
            raise ValueError(
                f"recurrent_unroll_len/chunk size ({chunk_size}) cannot exceed recurrent_l0_window ({int(self.recurrent_l0_window)})"
            )
        detach_every_chunks = max(0, int(self.recurrent_detach_every))
        if detach_every_chunks <= 0:
            detach_every_chunks = 0

        state = model_for_recurrent.init_recurrent_state(
            batch_size=int(B),
            l0_window=(self.recurrent_l0_window or None),
            device=self.device,
        )
        total_numer_value = 0.0
        total_correct = 0
        ce_label_smoothing = max(0.0, float(self.ce_label_smoothing_train))
        device_type = getattr(self.device, "type", str(self.device))

        def _train_amp_ctx():
            return (
                torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16 if use_bf16 else torch.float16)
                if self.mixed_precision and (self.scaler is not None or use_bf16) and device_type == "cuda"
                else nullcontext()
            )

        chunk_counter = 0
        group_loss = None
        group_chunks = 0
        for chunk_start in range(0, T - 1, chunk_size):
            chunk_end = min(T - 1, chunk_start + chunk_size)
            if chunk_end <= chunk_start:
                continue
            input_chunk = input_ids[:, chunk_start:chunk_end]
            target_chunk = input_ids[:, chunk_start + 1:chunk_end + 1]
            valid = target_valid[:, chunk_start:chunk_end]
            with _train_amp_ctx():
                state, logits = model_for_recurrent.recurrent_chunk_forward(
                    state,
                    input_chunk,
                    num_cycles=self.unified_refinement_cycles,
                    detach_state=False,
                )

            if bool(valid.any()):
                ce = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    target_chunk.reshape(-1),
                    reduction="none",
                    label_smoothing=ce_label_smoothing,
                )
                chunk_numer = (ce * valid.reshape(-1).to(dtype=ce.dtype)).sum()
                count = int(valid.sum().item())
                total_numer_value += float(chunk_numer.detach().item())
                total_correct += int(((logits.argmax(dim=-1) == target_chunk) & valid).sum().item())
                loss = float(self.lambda_base_ce_loss) * (chunk_numer / float(total_valid))
                group_loss = loss if group_loss is None else (group_loss + loss)
                group_chunks += 1

            chunk_counter += 1
            boundary_reached = detach_every_chunks > 0 and (chunk_counter % detach_every_chunks) == 0
            last_chunk = chunk_end >= (T - 1)
            if boundary_reached or last_chunk:
                if group_loss is not None:
                    scaled_loss = group_loss / max(1, int(self.gradient_accumulation_steps))
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                group_loss = None
                group_chunks = 0
                if detach_every_chunks > 0:
                    state = model_for_recurrent.detach_recurrent_state(state)

        ce_loss = total_numer_value / float(total_valid)
        objective_loss = float(self.lambda_base_ce_loss) * ce_loss
        self._set_recurrent_train_stats(ce_loss, objective_loss, total_valid)
        for tgt in [self.model, model_for_recurrent]:
            setattr(tgt, "_last_recurrent_acc", float(total_correct) / float(max(1, total_valid)))
        return float(objective_loss)

    @torch.no_grad()
    def _recurrent_teacher_forced_eval(self, model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        model_for_recurrent = self._recurrent_base_model(model)
        if not hasattr(model_for_recurrent, "init_recurrent_state") or not hasattr(model_for_recurrent, "recurrent_chunk_forward"):
            return {"loss": float("inf"), "tokens": 0, "correct": 0, "acc": 0.0, "ppl": float("inf")}
        if input_ids.dim() != 2 or input_ids.size(1) < 2:
            return {"loss": float("inf"), "tokens": 0, "correct": 0, "acc": 0.0, "ppl": float("inf")}

        B, T = input_ids.shape
        chunk_size = max(1, int(self.recurrent_unroll_len))
        if int(self.recurrent_l0_window or 0) > 0 and chunk_size > int(self.recurrent_l0_window):
            chunk_size = int(self.recurrent_l0_window)
        state = model_for_recurrent.init_recurrent_state(
            batch_size=int(B),
            l0_window=(self.recurrent_l0_window or None),
            device=input_ids.device,
        )

        total_numer = 0.0
        total_tokens = 0
        total_correct = 0
        for chunk_start in range(0, T - 1, chunk_size):
            chunk_end = min(T - 1, chunk_start + chunk_size)
            if chunk_end <= chunk_start:
                continue
            input_chunk = input_ids[:, chunk_start:chunk_end]
            target_chunk = input_ids[:, chunk_start + 1:chunk_end + 1]
            valid = torch.ones((B, chunk_end - chunk_start), device=input_ids.device, dtype=torch.bool)
            if attention_mask is not None:
                valid = attention_mask[:, chunk_start + 1:chunk_end + 1].to(device=input_ids.device, dtype=torch.bool)

            state, logits = model_for_recurrent.recurrent_chunk_forward(
                state,
                input_chunk,
                num_cycles=self.unified_refinement_cycles,
                detach_state=True,
            )
            if bool(valid.any()):
                ce = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    target_chunk.reshape(-1),
                    reduction="none",
                )
                chunk_tokens = int(valid.sum().item())
                total_numer += float((ce * valid.reshape(-1).to(dtype=ce.dtype)).sum().item())
                total_tokens += chunk_tokens
                total_correct += int(((logits.argmax(dim=-1) == target_chunk) & valid).sum().item())

        if total_tokens <= 0:
            return {"loss": float("inf"), "tokens": 0, "correct": 0, "acc": 0.0, "ppl": float("inf")}
        loss_value = total_numer / float(total_tokens)
        ppl = math.exp(min(loss_value, 700.0)) if loss_value > 0 and math.isfinite(loss_value) else float("inf")
        return {
            "loss": loss_value,
            "tokens": total_tokens,
            "correct": total_correct,
            "acc": float(total_correct) / float(max(1, total_tokens)),
            "ppl": ppl,
        }
    
    def _process_batch_with_hybrid_masking(self, batch, use_bf16=True):
        """
        Process a batch with hybrid masking (random + last token).

        Now with memory-augmented training support! When memory is enabled,
        the training loop will automatically use episode() for retrieval-augmented
        forward passes.

        Args:
            batch: Training batch

        Returns:
            loss: Loss for this batch
        """
        if self.modality == "image" or (isinstance(batch, dict) and ("pixel_values" in batch or "input_features" in batch)):
            if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit":
                if str(getattr(self, "image_maskgit_variant", "continuous")).lower() == "discrete":
                    return self._process_batch_with_image_maskgit_discrete(batch, use_bf16=use_bf16)
                return self._process_batch_with_image_maskgit(batch, use_bf16=use_bf16)
            return self._process_batch_with_image_diffusion(batch, use_bf16=use_bf16)

        if self.recurrent_training_enable and self.train_objective_mode == "ar":
            return self._process_batch_recurrent_ar(batch, use_bf16=use_bf16)

        return train_with_hybrid_masking(
            model=self.model,
            batch=batch,
            criterion=self.criterion,
            optimizer=self.optimizer,
            tokenizer=self.tokenizer,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            scaler=self.scaler,
            device=self.device,  # Pass device explicitly
            use_bf16=use_bf16,
            objective_mode=self.train_objective_mode,
            lambda_masked=self.lambda_masked_loss,
            lambda_ar=self.lambda_ar_loss,
            lambda_base_ce=self.lambda_base_ce_loss,
            ce_label_smoothing=self.ce_label_smoothing_train,
            lambda_copy=self.lambda_copy_loss,
            lambda_unmasked=self.lambda_unmasked_loss,
            lambda_autoenc=self.lambda_autoenc_loss,
            lambda_autoenc_next=self.lambda_autoenc_next_loss,
            lambda_token_unet_lookahead_ce=self.lambda_token_unet_lookahead_ce,
            chunked_ce_enable=self.chunked_ce_enable,
            chunked_ce_seq_chunk=self.chunked_ce_seq_chunk,
            train_feature_chunked_ce_enable=self.train_feature_chunked_ce_enable,
            diffusion_mask_mode=self.diffusion_mask_mode,
            diffusion_mask_block_size=self.diffusion_mask_block_size,
            diffusion_mask_path_length=self.diffusion_mask_path_length,
            diffusion_grid_shape=(
                (int(getattr(self.model, "graph_grid_height", 0)), int(getattr(self.model, "graph_grid_width", 0)))
                if int(getattr(self.model, "graph_grid_height", 0)) > 0 and int(getattr(self.model, "graph_grid_width", 0)) > 0
                else None
            ),
            autoenc_training_policy=self.autoenc_training_policy,
            llada_loss_weighting=self.llada_loss_weighting,
        )
    
    
    def _eval_forward(self, model, input_ids, attention_mask, reveal_target_ids, reveal_mask):
        """Plain evaluation forward (memory subsystem removed from the public build)."""
        return model(input_ids, attention_mask=attention_mask,
                     reveal_target_ids=reveal_target_ids, reveal_mask=reveal_mask)

    # Long-context diagnostic: next-token PPL bucketed by in-context recurrence distance
    # (distance to the target token's previous occurrence within the same sequence). The
    # split that matters is <128 (inside the local window) vs >=128 (needs the hierarchy).
    _LCD_LABELS = ("never", "<128", "128-511", "512-2k", ">2k")

    @staticmethod
    def _lcd_bucket(dist):
        if dist is None:
            return 0
        if dist < 128:
            return 1
        if dist < 512:
            return 2
        if dist < 2048:
            return 3
        return 4

    def _accumulate_longctx_diag(self, input_ids, shift_logits, valid_2d, acc, seqs_done):
        """Accumulate per-token NLL into recurrence-distance buckets. `acc` is a list of
        [sum_nll, count] per bucket; returns the updated sequence count. Cheap + read-only:
        reuses the already-computed validation logits, capped at longctx_diag_max_seqs."""
        B, Tm1, V = shift_logits.shape
        with torch.no_grad():
            nll = F.cross_entropy(
                shift_logits.reshape(-1, V), input_ids[:, 1:].reshape(-1),
                reduction="none",
            ).view(B, Tm1).float().cpu()
        ids_cpu = input_ids.detach().cpu()
        valid_cpu = valid_2d.detach().cpu()
        for b in range(B):
            if seqs_done >= self.longctx_diag_max_seqs:
                break
            seq = ids_cpu[b].tolist()
            last = {}
            for j in range(len(seq)):
                tid = seq[j]
                # target token at index j is predicted from shifted position j-1
                if j >= 1 and bool(valid_cpu[b, j - 1]):
                    prev = last.get(tid)
                    bi = self._lcd_bucket(j - prev if prev is not None else None)
                    acc[bi][0] += float(nll[b, j - 1]); acc[bi][1] += 1
                last[tid] = j
            seqs_done += 1
        return seqs_done

    def validate(self, data_provider, use_ema = False):
        """
        Validate the model using an objective that mirrors the hybrid training loss,
        and report individual components.
        """
        if self.modality == "image":
            if str(getattr(self, "image_objective", "diffusion")).lower() == "maskgit":
                if str(getattr(self, "image_maskgit_variant", "continuous")).lower() == "discrete":
                    return self._validate_image_maskgit_discrete(data_provider, use_ema=use_ema)
                return self._validate_image_maskgit(data_provider, use_ema=use_ema)
            return self._validate_image_diffusion(data_provider, use_ema=use_ema)

        if use_ema:
            self.ema_model.eval()
        else:
            self.model.eval()
        # Long-context diagnostic gate: run every `longctx_diag_every` validations.
        _lcd_on = self.longctx_diag_every > 0 and (self._longctx_diag_calls % self.longctx_diag_every == 0)
        self._longctx_diag_calls += 1
        _lcd_acc = [[0.0, 0] for _ in self._LCD_LABELS]
        _lcd_seqs = 0
        total_combined_loss = 0.0
        total_masked_loss_component = 0.0 # Track masked loss separately
        total_masked_loss_numer = 0.0
        total_masked_loss_denom = 0
        total_next_token_loss_component = 0.0 # Mean next-token loss (token-weighted at end)
        total_next_token_loss_numer = 0.0
        total_next_token_loss_denom = 0
        total_next_token_loss_trunc_numer = 0.0
        total_next_token_loss_trunc_denom = 0

        total_masked_tokens = 0
        correct_masked_tokens = 0
        total_next_tokens = 0 # For next-token accuracy
        correct_next_tokens = 0 # For next-token accuracy
        total_next_tokens_trunc = 0
        correct_next_tokens_trunc = 0

        copy_samples = 0
        copy_token_total = 0
        copy_token_correct = 0
        copy_ce_numer = 0.0
        copy_ce_denom = 0
        copy_span_total = 0
        copy_span_correct = 0
        copy_first_total = 0
        copy_first_correct = 0
        copy_log_bucket_stats = {}
        copy_hier_bucket_stats = {}

        total_autoenc_loss_numer = 0.0
        total_autoenc_loss_denom = 0
        total_autoenc_tokens = 0
        correct_autoenc_tokens = 0
        total_autoenc_next_loss_numer = 0.0
        total_autoenc_next_loss_denom = 0
        total_autoenc_next_tokens = 0
        correct_autoenc_next_tokens = 0

        total_hier_aux_loss = 0.0
        total_hier_aux_denom = 0
        total_objective_loss = 0.0
        total_objective_denom = 0
        total_recurrent_loss_numer = 0.0
        total_recurrent_loss_denom = 0
        total_recurrent_correct = 0

        steps_processed = 0

        # Import hybrid_mask_tokens (ensure accessible)
        #from hybrid_masking import hybrid_mask_tokens

        # Determine data iteration setup
        if isinstance(data_provider, dict) and "get_batch" in data_provider:
            get_batch = data_provider["get_batch"]
            steps = data_provider.get("steps", 50)
            data_iterator = range(steps)
            desc = "Validation (Steps)"
            use_get_batch = True
        elif isinstance(data_provider, torch.utils.data.DataLoader):
            dataloader = data_provider
            steps = len(dataloader)
            data_iterator = dataloader
            desc = "Validation (Batches)"
            use_get_batch = False
        else:
            raise ValueError("Invalid data_provider format.")

        pbar = tqdm(data_iterator, desc=desc, total=steps, dynamic_ncols=True)
        objective_mode = self.train_objective_mode
        model_for_mode = self.ema_model if use_ema else self.model
        autoenc_mode = self._resolve_autoenc_runtime_mode(model_for_mode)
        autoenc_only_mode = bool(autoenc_mode["autoenc_only_mode"])
        autoenc_only_diffusion_mode = bool(autoenc_mode.get("autoenc_only_diffusion_mode", False))
        l0_local_mode = self._resolve_l0_local_runtime_mode(model_for_mode)
        model_for_mode_mod = getattr(model_for_mode, "module", None)
        lambda_ce_anchor_val = getattr(model_for_mode, "lambda_ce_anchor", None)
        if lambda_ce_anchor_val is None and model_for_mode_mod is not None:
            lambda_ce_anchor_val = getattr(model_for_mode_mod, "lambda_ce_anchor", None)
        # Validation must run under the SAME mixed-precision autocast as training. This was
        # previously gated on the L0 local backend being flash, so non-flash configs (e.g.
        # attn_backend: pyg) validated in fp32 while training ran in bf16 — a train/val
        # precision mismatch that also fed fp32 into the flash-only packed multirate refiners
        # and crashed them. Mirror the training amp gate (self.mixed_precision) instead.
        use_val_autocast = (
            bool(getattr(self, "mixed_precision", False))
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        if autoenc_only_diffusion_mode:
            objective_mode = "masked"

        with torch.no_grad():
            for step_or_batch_idx, batch_data in enumerate(pbar):
                # --- Get Batch ---
                if use_get_batch:
                    batch = get_batch(self.device)
                    if batch is None or "input_ids" not in batch or batch["input_ids"] is None:
                        print("Warning: Skipping empty validation batch.")
                        continue
                    if isinstance(batch, dict):
                        batch = self._maybe_apply_copy_task(batch, is_validation=True)
                    input_ids = batch["input_ids"] # Original IDs
                else:
                    batch = batch_data
                    if not isinstance(batch, dict) or "input_ids" not in batch:
                        if isinstance(batch, (list, tuple)) and len(batch) > 0:
                            input_ids = batch[0].to(self.device)
                        else:
                            print(f"Warning: Skipping unrecognized validation batch format: {type(batch)}")
                            continue
                    else:
                        batch = self._maybe_apply_copy_task(batch, is_validation=True)
                        input_ids = batch["input_ids"].to(self.device)

                if input_ids is None or input_ids.numel() == 0:
                    print("Warning: Skipping validation batch with empty input_ids.")
                    continue
                if isinstance(batch, dict) and batch.get("labels", None) is not None:
                    labels = batch["labels"].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) > 1 and batch[1] is not None:
                    labels = batch[1].to(self.device)
                else:
                    labels = torch.full_like(input_ids, -100)
                    if input_ids.size(1) > 1:
                        labels[:, :-1] = input_ids[:, 1:]
                # --- End Get Batch ---
                #if input_ids.dtype != torch.float32:
                #    input_ids = input_ids.float()
                # Apply the SAME masking as in training
                #masked_ids, target_tokens = hybrid_mask_tokens(input_ids, self.tokenizer)

                diffusion_mask_prob = 1.0
                if objective_mode == "ar":
                    masked_ids = input_ids
                    target_tokens = torch.full_like(input_ids, -100)
                    reveal_target_ids_for_model = input_ids.clone()
                    if input_ids.size(1) > 1:
                        reveal_target_ids_for_model[:, :-1] = input_ids[:, 1:]
                    reveal_mask_for_model = torch.zeros_like(input_ids, dtype=torch.bool)
                    if input_ids.size(1) > 1:
                        reveal_mask_for_model[:, :-1] = True
                else:
                    grid_shape = None
                    gh = int(getattr(self.model, "graph_grid_height", 0))
                    gw = int(getattr(self.model, "graph_grid_width", 0))
                    if gh > 0 and gw > 0:
                        grid_shape = (gh, gw)
                    if self.llada_loss_weighting:
                        masked_ids, target_tokens, diffusion_mask_prob = hybrid_diffusion_mask_tokens(
                            input_ids,
                            self.tokenizer,
                            val=True,
                            mode=self.diffusion_mask_mode,
                            grid_shape=grid_shape,
                            block_size=self.diffusion_mask_block_size,
                            path_length=self.diffusion_mask_path_length,
                            return_mask_prob=True,
                        )
                    else:
                        masked_ids, target_tokens = hybrid_diffusion_mask_tokens(
                            input_ids,
                            self.tokenizer,
                            val=True,
                            mode=self.diffusion_mask_mode,
                            grid_shape=grid_shape,
                            block_size=self.diffusion_mask_block_size,
                            path_length=self.diffusion_mask_path_length,
                        )
                        diffusion_mask_prob = 1.0
                    reveal_target_ids_for_model = input_ids
                    reveal_mask_for_model = (target_tokens != -100)
                diffusion_mask_loss_weight = 1.0
                if self.llada_loss_weighting and objective_mode in {"masked", "hybrid"}:
                    diffusion_mask_loss_weight = 1.0 / max(float(diffusion_mask_prob), 1e-3)

                copy_dst_mask = batch.get("copy_dst_mask", None) if isinstance(batch, dict) else None
                if copy_dst_mask is not None:
                    copy_dst_mask = copy_dst_mask.to(self.device, dtype=torch.bool)
                copy_force_mask_dst = bool(batch.get("copy_force_mask_dst", False)) if isinstance(batch, dict) else False
                copy_mask_token_id = batch.get("copy_mask_token_id", None) if isinstance(batch, dict) else None
                if (
                    copy_force_mask_dst
                    and copy_dst_mask is not None
                    and copy_mask_token_id is not None
                    and bool(copy_dst_mask.any())
                    and (objective_mode != "ar" or self.copy_task_mask_dst_in_ar)
                ):
                    masked_ids = masked_ids.clone()
                    masked_ids[copy_dst_mask] = int(copy_mask_token_id)
                    target_tokens = target_tokens.clone()
                    target_tokens[copy_dst_mask] = input_ids[copy_dst_mask]
                    if objective_mode != "ar":
                        reveal_mask_for_model = (target_tokens != -100)

                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    if objective_mode == "ar" and input_ids.size(1) > 1:
                        reveal_mask_for_model[:, :-1] = attention_mask[:, 1:].to(dtype=torch.bool)

                # Forward pass with masked input
                def _val_amp_ctx():
                    return (
                        torch.autocast(device_type="cuda", dtype=amp_dtype("cuda"))
                        if use_val_autocast
                        else nullcontext()
                    )

                with _val_amp_ctx():
                    # Eval must mirror the memory-augmented training forward (else a
                    # train/val regime mismatch makes val loss explode). Read-only.
                    eval_model = self.ema_model if use_ema else self.model
                    logits = self._eval_forward(
                        eval_model,
                        masked_ids,
                        attention_mask,
                        reveal_target_ids_for_model,
                        reveal_mask_for_model,
                    )

                current_autoenc_loss = 0.0
                current_autoenc_next_loss = 0.0
                if use_ema:
                    ae_logits = getattr(self.ema_model, "_last_autoenc_logits", None)
                    model_mod_local = getattr(self.ema_model, "module", None)
                else:
                    ae_logits = getattr(self.model, "_last_autoenc_logits", None)
                    model_mod_local = getattr(self.model, "module", None)
                if ae_logits is None and model_mod_local is not None:
                    ae_logits = getattr(model_mod_local, "_last_autoenc_logits", None)
                if ae_logits is not None and not autoenc_only_diffusion_mode:
                    ae_logits = ae_logits.to(device=self.device)
                    ae_token_loss = F.cross_entropy(
                        ae_logits.reshape(-1, ae_logits.size(-1)),
                        input_ids.reshape(-1),
                        reduction="none",
                    ).view_as(input_ids)
                    if attention_mask is not None:
                        ae_mask = attention_mask.to(self.device, dtype=torch.bool)
                    else:
                        ae_mask = torch.ones_like(input_ids, device=self.device, dtype=torch.bool)
                    if bool(ae_mask.any()):
                        ae_count = int(ae_mask.sum().item())
                        ae_loss = (ae_token_loss * ae_mask.to(ae_token_loss.dtype)).sum() / ae_mask.sum().clamp_min(1)
                        current_autoenc_loss = float(ae_loss.item())
                        total_autoenc_loss_numer += current_autoenc_loss * float(ae_count)
                        total_autoenc_loss_denom += ae_count

                        ae_pred = torch.argmax(ae_logits, dim=-1)
                        ae_correct = ((ae_pred == input_ids) & ae_mask).sum().item()
                        total_autoenc_tokens += ae_count
                        correct_autoenc_tokens += int(ae_correct)

                    ae_next_mask = (labels != -100)
                    if attention_mask is not None:
                        ae_next_mask = ae_next_mask & attention_mask.to(self.device, dtype=torch.bool)
                    if bool(ae_next_mask.any()):
                        ae_next_count = int(ae_next_mask.sum().item())
                        ae_next_token_loss = F.cross_entropy(
                            ae_logits.reshape(-1, ae_logits.size(-1)),
                            labels.reshape(-1),
                            reduction="none",
                            ignore_index=-100,
                        ).view_as(labels)
                        ae_next_loss = (
                            ae_next_token_loss * ae_next_mask.to(ae_next_token_loss.dtype)
                        ).sum() / ae_next_mask.sum().clamp_min(1)
                        current_autoenc_next_loss = float(ae_next_loss.item())
                        total_autoenc_next_loss_numer += current_autoenc_next_loss * float(ae_next_count)
                        total_autoenc_next_loss_denom += ae_next_count

                        ae_next_pred = torch.argmax(ae_logits, dim=-1)
                        ae_next_correct = ((ae_next_pred == labels) & ae_next_mask).sum().item()
                        total_autoenc_next_tokens += ae_next_count
                        correct_autoenc_next_tokens += int(ae_next_correct)

                metric_model = self.ema_model if use_ema else self.model
                metric_model_mod = getattr(metric_model, "module", None)
                current_hier_aux = getattr(metric_model, "_last_hier_aux_loss", None)
                if current_hier_aux is None and metric_model_mod is not None:
                    current_hier_aux = getattr(metric_model_mod, "_last_hier_aux_loss", None)
                if current_hier_aux is not None:
                    if torch.is_tensor(current_hier_aux):
                        current_hier_aux = float(current_hier_aux.detach().item())
                    else:
                        current_hier_aux = float(current_hier_aux)
                    total_hier_aux_loss += float(current_hier_aux)
                    total_hier_aux_denom += 1

                # --- Calculate BOTH loss components ---
                # ensure logits are of right float type for loss compared to target tokens
                if isinstance(logits, dict):
                    logits = logits["logits"]
                if (
                    autoenc_only_diffusion_mode
                    and ae_logits is not None
                    and tuple(ae_logits.shape[:2]) == tuple(logits.shape[:2])
                ):
                    logits = ae_logits.to(device=logits.device, dtype=logits.dtype)
                #if logits.dtype != torch.float32:
                #    logits = logits.float()
                # 1. Masked Prediction Loss (Imputation)
                masked_logits = logits.view(-1, logits.size(-1))
                
                target_tokens_flat = target_tokens.view(-1)
                mask_impute = (target_tokens_flat != -100)
                current_masked_loss = 0.0
                if mask_impute.any():
                    masked_count = int(mask_impute.sum().item())
                    current_masked_loss = self.criterion(masked_logits[mask_impute], target_tokens_flat[mask_impute]).item()
                    current_masked_loss = current_masked_loss * float(diffusion_mask_loss_weight)
                    total_masked_loss_component += current_masked_loss
                    total_masked_loss_numer += current_masked_loss * float(masked_count)
                    total_masked_loss_denom += masked_count

                    # Masked accuracy calculation
                    predictions_masked = torch.argmax(masked_logits[mask_impute], dim=-1)
                    correct_masked = (predictions_masked == target_tokens_flat[mask_impute])
                    total_masked_tokens += masked_count
                    correct_masked_tokens += correct_masked.sum().item()
                elif objective_mode == "masked":
                    print("Warning: Skipping validation batch with no masks applied at t 1.")
                    continue
                
                # Correct next-token (AR) loss over shifted sequence
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels_flat = shift_labels.view(-1)

                if attention_mask is not None:
                    valid_next = attention_mask[..., 1:].contiguous().view(-1) > 0
                else:
                    valid_next = torch.ones_like(shift_labels_flat, dtype=torch.bool)

                if _lcd_on and objective_mode == "ar" and _lcd_seqs < self.longctx_diag_max_seqs:
                    _lcd_seqs = self._accumulate_longctx_diag(
                        input_ids, shift_logits, valid_next.view(shift_labels.shape), _lcd_acc, _lcd_seqs)

                current_next_token_loss_trunc = 0.0
                trunc_ignore = int(getattr(self, "eval_ppl_ignore_prefix_tokens", 0))
                if trunc_ignore > 0:
                    seq_len_shift = int(shift_labels.size(1))
                    pos = torch.arange(seq_len_shift, device=shift_labels.device, dtype=torch.long)
                    pos = pos.view(1, seq_len_shift).expand(shift_labels.size(0), seq_len_shift).reshape(-1)
                    valid_next_trunc = valid_next & (pos >= trunc_ignore)
                else:
                    valid_next_trunc = valid_next

                if bool(valid_next.any()):
                    valid_count = int(valid_next.sum().item())
                    current_next_token_loss = self.criterion(
                        shift_logits_flat[valid_next],
                        shift_labels_flat[valid_next],
                    ).item()
                    total_next_token_loss_numer += current_next_token_loss * float(valid_count)
                    total_next_token_loss_denom += valid_count
                    total_next_token_loss_component += current_next_token_loss

                    predictions_next = torch.argmax(shift_logits_flat[valid_next], dim=-1)
                    correct_next = (predictions_next == shift_labels_flat[valid_next])
                    total_next_tokens += valid_count
                    correct_next_tokens += int(correct_next.sum().item())

                    if bool(valid_next_trunc.any()):
                        valid_trunc_count = int(valid_next_trunc.sum().item())
                        current_next_token_loss_trunc = self.criterion(
                            shift_logits_flat[valid_next_trunc],
                            shift_labels_flat[valid_next_trunc],
                        ).item()
                        total_next_token_loss_trunc_numer += current_next_token_loss_trunc * float(valid_trunc_count)
                        total_next_token_loss_trunc_denom += valid_trunc_count

                        predictions_next_trunc = torch.argmax(shift_logits_flat[valid_next_trunc], dim=-1)
                        correct_next_trunc = (predictions_next_trunc == shift_labels_flat[valid_next_trunc])
                        total_next_tokens_trunc += valid_trunc_count
                        correct_next_tokens_trunc += int(correct_next_trunc.sum().item())
                else:
                    current_next_token_loss = 0.0

                current_recurrent_loss = None
                current_recurrent_acc = None
                if self.recurrent_val_enable and objective_mode == "ar":
                    with _val_amp_ctx():
                        rec_metrics = self._recurrent_teacher_forced_eval(
                            self.ema_model if use_ema else self.model,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                    rec_tokens = int(rec_metrics.get("tokens", 0))
                    if rec_tokens > 0 and math.isfinite(float(rec_metrics.get("loss", float("inf")))):
                        current_recurrent_loss = float(rec_metrics["loss"])
                        current_recurrent_acc = float(rec_metrics.get("acc", 0.0))
                        total_recurrent_loss_numer += current_recurrent_loss * float(rec_tokens)
                        total_recurrent_loss_denom += rec_tokens
                        total_recurrent_correct += int(rec_metrics.get("correct", 0))

                copy_meta = batch.get("copy_task_meta", None) if isinstance(batch, dict) else None
                if copy_meta is not None:
                    applied = copy_meta.get("applied", None)
                    dst_payload_mask = copy_meta.get("dst_payload_mask", None)
                    distance = copy_meta.get("distance", None)
                    if applied is not None and dst_payload_mask is not None and distance is not None:
                        applied = applied.to(self.device, dtype=torch.bool)
                        dst_payload_mask = dst_payload_mask.to(self.device, dtype=torch.bool)
                        distance = distance.to(self.device, dtype=torch.long)
                        valid_next_2d = valid_next.view_as(shift_labels)
                        copy_shift_mask = dst_payload_mask[:, 1:] & valid_next_2d
                        pred_shift = torch.argmax(shift_logits, dim=-1)
                        for b in range(shift_labels.size(0)):
                            if not bool(applied[b]):
                                continue
                            sample_mask = copy_shift_mask[b]
                            if not bool(sample_mask.any()):
                                continue

                            sample_pred = pred_shift[b][sample_mask]
                            sample_true = shift_labels[b][sample_mask]
                            sample_correct = (sample_pred == sample_true)
                            tok_total = int(sample_mask.sum().item())
                            tok_correct = int(sample_correct.sum().item())
                            span_ok = bool(sample_correct.all())
                            sample_ce = float(
                                self.criterion(
                                    shift_logits[b][sample_mask],
                                    shift_labels[b][sample_mask],
                                ).item()
                            )

                            sample_positions = sample_mask.nonzero(as_tuple=False).view(-1)
                            first_idx = int(sample_positions[0].item())
                            first_ok = bool(pred_shift[b, first_idx] == shift_labels[b, first_idx])

                            copy_samples += 1
                            copy_token_total += tok_total
                            copy_token_correct += tok_correct
                            copy_ce_numer += sample_ce * float(tok_total)
                            copy_ce_denom += tok_total
                            copy_span_total += 1
                            copy_span_correct += int(span_ok)
                            copy_first_total += 1
                            copy_first_correct += int(first_ok)

                            dist = int(distance[b].item())
                            log_key = log_bucket_label(dist)
                            self._update_copy_bucket(
                                copy_log_bucket_stats,
                                log_key,
                                tok_correct,
                                tok_total,
                                span_ok,
                                first_ok,
                                sample_ce * float(tok_total),
                                tok_total,
                            )

                            hier_key = hierarchy_bucket_label(dist, self.copy_task_hierarchy_thresholds)
                            self._update_copy_bucket(
                                copy_hier_bucket_stats,
                                hier_key,
                                tok_correct,
                                tok_total,
                                span_ok,
                                first_ok,
                                sample_ce * float(tok_total),
                                tok_total,
                            )


                # 3. Combined Loss (mirrors training objective mode)
                if objective_mode == "ar":
                    current_combined_loss = self.lambda_base_ce_loss * float(current_next_token_loss)
                elif objective_mode == "hybrid":
                    current_combined_loss = (
                        self.lambda_masked_loss * float(current_masked_loss)
                        + self.lambda_ar_loss * float(current_next_token_loss)
                    )
                else:
                    current_combined_loss = float(current_masked_loss)

                current_objective_loss = float(current_combined_loss)

                total_objective_loss += float(current_objective_loss)
                total_objective_denom += 1

                total_combined_loss += current_combined_loss
                # --- End Loss Calculation ---


                steps_processed += 1
                postfix = {
                    'val_comb_loss': current_combined_loss,
                    'val_obj_loss': current_objective_loss,
                    'val_mask_loss': current_masked_loss,
                    'val_next_loss': current_next_token_loss
                    }
                if total_autoenc_loss_denom > 0:
                    postfix['val_ae_loss'] = current_autoenc_loss
                if total_autoenc_next_loss_denom > 0:
                    postfix['val_ae_next_loss'] = current_autoenc_next_loss
                if self.eval_report_truncated_ppl and self.eval_ppl_ignore_prefix_tokens > 0:
                    postfix['val_next_loss_trunc'] = current_next_token_loss_trunc
                if current_recurrent_loss is not None:
                    postfix['val_rec_loss'] = current_recurrent_loss
                    postfix['val_rec_acc'] = current_recurrent_acc
                pbar.set_postfix(postfix)


        pbar.close()

        if steps_processed == 0:
            logger.warning("No validation steps were processed.")
            # Return default bad values
            metrics = {'loss': float('inf'), 'masked_loss': float('inf'), 'next_token_loss': float('inf'),
                    'objective_loss': float('inf'),
                    'masked_acc': 0, 'next_token_acc': 0, 'perplexity': float('inf'),
                    'next_token_loss_trunc': float('inf'), 'next_token_acc_trunc': 0,
                    'next_token_perplexity_trunc': float('inf'),
                    'recurrent_next_loss': float('inf'), 'recurrent_next_acc': 0.0,
                    'recurrent_next_perplexity': float('inf'),
                    'autoenc_ce_loss': float('inf'), 'autoenc_acc': 0.0,
                    'autoenc_next_ce_loss': float('inf'), 'autoenc_next_acc': 0.0,
                    'selection_metric': 'autoenc_ce_loss' if autoenc_only_mode else 'combined_loss',
                    'selection_loss': float('inf'),
                    'copy_samples': 0, 'copy_token_acc': 0.0, 'copy_span_exact': 0.0,
                    'copy_first_token_acc': 0.0, 'copy_dst_ce_loss': float('inf'),
                    'copy_log_buckets': {}, 'copy_hierarchy_buckets': {}}
            return float('inf'), metrics

        # Calculate average losses
        avg_combined_loss = total_combined_loss / steps_processed
        if total_masked_loss_denom > 0:
            avg_masked_loss = total_masked_loss_numer / float(total_masked_loss_denom)
        else:
            avg_masked_loss = 0.0
        if total_next_token_loss_denom > 0:
            avg_next_token_loss = total_next_token_loss_numer / float(total_next_token_loss_denom)
        else:
            avg_next_token_loss = 0.0
        if total_next_token_loss_trunc_denom > 0:
            avg_next_token_loss_trunc = total_next_token_loss_trunc_numer / float(total_next_token_loss_trunc_denom)
        else:
            avg_next_token_loss_trunc = float('inf')

        if objective_mode == "ar":
            avg_combined_loss = self.lambda_base_ce_loss * avg_next_token_loss
        elif objective_mode == "hybrid":
            avg_combined_loss = (
                self.lambda_masked_loss * avg_masked_loss
                + self.lambda_ar_loss * avg_next_token_loss
            )
        else:
            avg_combined_loss = avg_masked_loss

        avg_hier_aux_loss = (
            total_hier_aux_loss / float(total_hier_aux_denom)
            if total_hier_aux_denom > 0
            else float('inf')
        )
        avg_objective_loss = (
            total_objective_loss / float(total_objective_denom)
            if total_objective_denom > 0
            else float(avg_combined_loss)
        )
        if total_recurrent_loss_denom > 0:
            avg_recurrent_next_loss = total_recurrent_loss_numer / float(total_recurrent_loss_denom)
            recurrent_next_acc = float(total_recurrent_correct) / float(total_recurrent_loss_denom)
            recurrent_next_ppl = math.exp(min(avg_recurrent_next_loss, 700.0)) if avg_recurrent_next_loss > 0 else float('inf')
        else:
            avg_recurrent_next_loss = float('inf')
            recurrent_next_acc = 0.0
            recurrent_next_ppl = float('inf')

        # Calculate accuracies
        masked_accuracy = correct_masked_tokens / total_masked_tokens if total_masked_tokens > 0 else 0
        next_token_accuracy = correct_next_tokens / total_next_tokens if total_next_tokens > 0 else 0
        next_token_accuracy_trunc = (
            correct_next_tokens_trunc / total_next_tokens_trunc if total_next_tokens_trunc > 0 else 0
        )

        # Primary PPL follows the active objective; next-token PPL remains diagnostic.
        combined_perplexity_loss = min(avg_combined_loss, 700)
        combined_perplexity = math.exp(combined_perplexity_loss) if avg_combined_loss > 0 else float('inf')
        next_perplexity_loss = min(avg_next_token_loss, 700)
        next_token_perplexity = math.exp(next_perplexity_loss) if avg_next_token_loss > 0 else float('inf')
        next_perplexity_trunc_loss = min(avg_next_token_loss_trunc, 700)
        next_token_perplexity_trunc = (
            math.exp(next_perplexity_trunc_loss) if math.isfinite(avg_next_token_loss_trunc) and avg_next_token_loss_trunc > 0 else float('inf')
        )
        if objective_mode in {"masked", "hybrid"}:
            perplexity = combined_perplexity
            perplexity_source = "objective CE"
        else:
            perplexity = next_token_perplexity
            perplexity_source = "next-token CE"

        self.val_losses.append(avg_combined_loss) # Store combined loss for best model tracking

        if _lcd_on and any(c for _, c in _lcd_acc):
            parts = []
            for lbl, (s, c) in zip(self._LCD_LABELS, _lcd_acc):
                if c:
                    parts.append(f"{lbl}={math.exp(min(20.0, s / c)):.1f}(n{c})")
            logger.info("[LONGCTX] next-token PPL by in-context recurrence distance (%d seqs): %s",
                        _lcd_seqs, "  ".join(parts))

        copy_token_acc = (copy_token_correct / copy_token_total) if copy_token_total > 0 else 0.0
        copy_span_exact = (copy_span_correct / copy_span_total) if copy_span_total > 0 else 0.0
        copy_first_token_acc = (copy_first_correct / copy_first_total) if copy_first_total > 0 else 0.0
        copy_dst_ce_loss = (copy_ce_numer / float(copy_ce_denom)) if copy_ce_denom > 0 else float('inf')
        autoenc_ce_loss = (
            total_autoenc_loss_numer / float(total_autoenc_loss_denom)
            if total_autoenc_loss_denom > 0
            else float('inf')
        )
        autoenc_acc = (
            float(correct_autoenc_tokens) / float(total_autoenc_tokens)
            if total_autoenc_tokens > 0
            else 0.0
        )
        autoenc_next_ce_loss = (
            total_autoenc_next_loss_numer / float(total_autoenc_next_loss_denom)
            if total_autoenc_next_loss_denom > 0
            else float('inf')
        )
        autoenc_next_acc = (
            float(correct_autoenc_next_tokens) / float(total_autoenc_next_tokens)
            if total_autoenc_next_tokens > 0
            else 0.0
        )
        selection_metric = "combined_loss"
        selection_loss = float(avg_combined_loss)
        if autoenc_only_mode and math.isfinite(autoenc_ce_loss):
            selection_metric = "autoenc_ce_loss"
            selection_loss = float(autoenc_ce_loss)
        elif objective_mode == "ar" and math.isfinite(avg_next_token_loss):
            selection_metric = "next_token_loss"
            selection_loss = float(avg_next_token_loss)
        if self.recurrent_training_enable and self.recurrent_val_enable and math.isfinite(avg_recurrent_next_loss):
            selection_metric = "recurrent_next_loss"
            selection_loss = float(avg_recurrent_next_loss)

        copy_log_buckets = {}
        for key, stat in sorted(copy_log_bucket_stats.items(), key=lambda kv: copy_log_bucket_sort_key(kv[0])):
            tok_total = max(1, int(stat["tok_total"]))
            span_total = max(1, int(stat["span_total"]))
            first_total = max(1, int(stat["first_total"]))
            copy_log_buckets[key] = {
                "token_acc": float(stat["tok_correct"]) / float(tok_total),
                "span_exact": float(stat["span_correct"]) / float(span_total),
                "first_token_acc": float(stat["first_correct"]) / float(first_total),
                "copy_ce_loss": float(stat["ce_numer"]) / float(max(1, int(stat["ce_denom"]))),
                "samples": int(stat["span_total"]),
            }

        copy_hierarchy_buckets = {}
        for key, stat in sorted(copy_hier_bucket_stats.items()):
            tok_total = max(1, int(stat["tok_total"]))
            span_total = max(1, int(stat["span_total"]))
            first_total = max(1, int(stat["first_total"]))
            copy_hierarchy_buckets[key] = {
                "token_acc": float(stat["tok_correct"]) / float(tok_total),
                "span_exact": float(stat["span_correct"]) / float(span_total),
                "first_token_acc": float(stat["first_correct"]) / float(first_total),
                "copy_ce_loss": float(stat["ce_numer"]) / float(max(1, int(stat["ce_denom"]))),
                "samples": int(stat["span_total"]),
            }

        metrics = {
            'loss': avg_combined_loss,        # Primary loss metric reported
            'objective_loss': float(avg_objective_loss),
            'masked_loss': avg_masked_loss,    # Component 1
            'next_token_loss': avg_next_token_loss, # Component 2
            'hier_aux_loss': float(avg_hier_aux_loss),
            'masked_acc': masked_accuracy,     # Accuracy for component 1
            'next_token_acc': next_token_accuracy, # Accuracy for component 2
            'perplexity': perplexity,
            'perplexity_source': perplexity_source,
            'next_token_perplexity': next_token_perplexity,
            'next_token_loss_trunc': avg_next_token_loss_trunc,
            'next_token_acc_trunc': next_token_accuracy_trunc,
            'next_token_perplexity_trunc': next_token_perplexity_trunc,
            'combined_perplexity': combined_perplexity,
            'recurrent_next_loss': float(avg_recurrent_next_loss),
            'recurrent_next_acc': float(recurrent_next_acc),
            'recurrent_next_perplexity': float(recurrent_next_ppl),
            'copy_samples': int(copy_samples),
            'copy_token_acc': float(copy_token_acc),
            'copy_span_exact': float(copy_span_exact),
            'copy_first_token_acc': float(copy_first_token_acc),
            'copy_dst_ce_loss': float(copy_dst_ce_loss),
            'autoenc_ce_loss': float(autoenc_ce_loss),
            'autoenc_acc': float(autoenc_acc),
            'autoenc_next_ce_loss': float(autoenc_next_ce_loss),
            'autoenc_next_acc': float(autoenc_next_acc),
            'selection_metric': selection_metric,
            'selection_loss': float(selection_loss),
            'copy_log_buckets': copy_log_buckets,
            'copy_hierarchy_buckets': copy_hierarchy_buckets,
        }

        # Log the detailed metrics
        logger.info(
            f"Validation Combined Loss: {avg_combined_loss:.4f}, "
            f"Objective Loss: {avg_objective_loss:.4f}, "
            f"Masked Loss: {avg_masked_loss:.4f}, Next Token Loss: {avg_next_token_loss:.4f}, "
            f"Masked Acc: {masked_accuracy:.4f}, Next Token Acc: {next_token_accuracy:.4f}, "
            f"Perplexity ({perplexity_source}): {perplexity:.4f} "
            f"(Next-token PPL diagnostic: {next_token_perplexity:.4f}, Combined PPL: {combined_perplexity:.4f})"
        )
        if math.isfinite(avg_hier_aux_loss):
            logger.info(
                "Validation Aux Objective Terms: hier_aux=%s",
                f"{avg_hier_aux_loss:.4f}" if math.isfinite(avg_hier_aux_loss) else "n/a",
            )
        if self.eval_report_truncated_ppl and self.eval_ppl_ignore_prefix_tokens > 0:
            logger.info(
                f"Truncated Next-Token (ignore first {self.eval_ppl_ignore_prefix_tokens} positions): "
                f"Loss={avg_next_token_loss_trunc:.4f}, Acc={next_token_accuracy_trunc:.4f}, "
                f"PPL={next_token_perplexity_trunc:.4f}"
            )
        if total_recurrent_loss_denom > 0:
            logger.info(
                "Recurrent Teacher-Forced Next-Token: Loss=%.4f, Acc=%.4f, PPL=%.4f, tokens=%d",
                avg_recurrent_next_loss,
                recurrent_next_acc,
                recurrent_next_ppl,
                int(total_recurrent_loss_denom),
            )
        if copy_samples > 0:
            logger.info(
                "Copy Eval: token_acc=%.4f span_exact=%.4f first_token_acc=%.4f copy_ce=%.4f samples=%d",
                copy_token_acc,
                copy_span_exact,
                copy_first_token_acc,
                copy_dst_ce_loss,
                int(copy_samples),
            )
            if copy_log_buckets:
                logger.info(
                    "Copy Eval Log Buckets: %s",
                    ", ".join(
                        f"{k}:tok={v['token_acc']:.3f}/span={v['span_exact']:.3f}/ce={v['copy_ce_loss']:.3f}"
                        for k, v in sorted(copy_log_buckets.items(), key=lambda kv: copy_log_bucket_sort_key(kv[0]))
                    ),
                )
            if copy_hierarchy_buckets:
                logger.info(
                    "Copy Eval Hier Buckets: %s",
                    ", ".join(
                        f"{k}:tok={v['token_acc']:.3f}/span={v['span_exact']:.3f}/ce={v['copy_ce_loss']:.3f}"
                        for k, v in copy_hierarchy_buckets.items()
                    ),
                )

        if total_autoenc_tokens > 0:
            logger.info(
                "Autoenc Eval: ce=%.4f acc=%.4f tokens=%d",
                autoenc_ce_loss,
                autoenc_acc,
                int(total_autoenc_tokens),
            )
        if total_autoenc_next_tokens > 0:
            logger.info(
                "Autoenc-Next Eval: ce=%.4f acc=%.4f tokens=%d",
                autoenc_next_ce_loss,
                autoenc_next_acc,
                int(total_autoenc_next_tokens),
            )
        if autoenc_only_mode:
            logger.info(
                "Validation selection metric: %s=%.4f",
                selection_metric,
                selection_loss,
            )

        # Return combined loss plus detailed metrics (selection metric is in metrics)
        return avg_combined_loss, metrics
    
    def generate_sample(
        self,
        prompt: str,
        max_length: int = 100,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        use_ema = False,
        generation_mode: str = "auto",
        recurrent_l0_window: Optional[int] = None,
        diffusion_chunk_size: int = 32,
        diffusion_steps: int = 16,
        diffusion_max_total_steps: int = 32,
        diffusion_initial_threshold: float = 0.2,
        diffusion_final_threshold: float = 0.0,
        diffusion_random_remask_prob: float = 0.0,
        diffusion_random_remask_cutoff: float = 0.0,
        return_parts: bool = False,  # if True, return (prompt_text, continuation_text) instead of one string
        # use_incremental argument is removed as we now default to imputation
    ) -> str:
        """
        Generate a sample output from the model using mask-based imputation generation.

        Args:
            prompt: Text prompt to begin generation
            max_length: Maximum total length (prompt + generated)
            max_new_tokens: Optional continuation budget; if set, overrides max_length semantics
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            repetition_penalty: Penalty for repeating tokens
            generation_mode: "auto", "ar", "diffusion", "autoenc_query", "autoenc_ar", or "recurrent"

        Returns:
            generated_text: Generated text from the model
        """
        try:
            self.model.eval()
            if self.ema_model is not None:
                self.ema_model.eval()

            # Tokenize prompt
            # Use a shorter context for the initial prompt to avoid exceeding max_seq_len quickly
            # max_prompt_len = self.model.max_seq_len // 2 if hasattr(self.model, 'max_seq_len') else 64
            # Let's just use a fixed reasonable prompt length for sampling
            max_prompt_len = 64
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                #max_length=max_prompt_len,
                #truncation=True
                # No padding needed here, sequence length will grow
            )

            input_ids = inputs.input_ids.to(self.device)
            
            # --- Check against model's max_seq_len ---
            # Generation cannot handle prompts longer than model's capacity
            model_max_len = getattr(self.model, 'max_seq_len', 512) # Get from model
            if input_ids.shape[1] >= model_max_len:
                 logger.warning(f"Prompt length ({input_ids.shape[1]}) exceeds model max sequence length ({model_max_len}). Truncating prompt.")
                 input_ids = input_ids[:, :model_max_len - 1] # Truncate to leave space for 1 generated token initially

            prompt_token_len = int(input_ids.shape[1])
            if max_new_tokens is not None:
                new_token_budget = max(1, int(max_new_tokens))
                target_total_len = min(int(model_max_len), prompt_token_len + new_token_budget)
            else:
                target_total_len = max(1, int(max_length))
                if target_total_len <= prompt_token_len:
                    target_total_len = min(int(model_max_len), prompt_token_len + target_total_len)
            if target_total_len <= prompt_token_len:
                logger.warning(
                    "No room to generate new tokens (prompt=%d, target_total=%d, model_max=%d).",
                    prompt_token_len,
                    target_total_len,
                    int(model_max_len),
                )
                return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            model_for_gen = self.ema_model if (use_ema and self.ema_model is not None) else self.model
            autoenc_mode = self._resolve_autoenc_runtime_mode(model_for_gen)
            autoenc_graph_mode = autoenc_mode["autoenc_graph_mode"]
            autoenc_only_mode = bool(autoenc_mode["autoenc_only_mode"])
            autoenc_only_diffusion_mode = bool(autoenc_mode.get("autoenc_only_diffusion_mode", False))
            l0_local_mode = self._resolve_l0_local_runtime_mode(model_for_gen)
            use_flash_gen_autocast = (
                bool(l0_local_mode.get("active", False))
                and str(l0_local_mode.get("backend", "pyg")) == "flash"
                and self.device.type == "cuda"
                and torch.cuda.is_available()
            )
            mode = str(generation_mode).lower()
            if mode not in {"auto", "ar", "diffusion", "autoenc_query", "autoenc_ar", "recurrent"}:
                mode = "auto"
            if mode == "auto":
                if autoenc_graph_mode == "twin_shared_l3" and autoenc_only_diffusion_mode:
                    mode = "diffusion"
                elif autoenc_graph_mode == "twin_shared_l3" and autoenc_only_mode:
                    if float(getattr(self, "lambda_autoenc_next_loss", 0.0)) > 0.0:
                        mode = "autoenc_ar"
                    else:
                        mode = "autoenc_query"
                elif autoenc_graph_mode == "twin_shared_l3":
                    mode = "ar"
                else:
                    objective_mode = str(getattr(self, "train_objective_mode", "masked")).lower()
                    if objective_mode in {"ar", "hybrid"}:
                        mode = "ar"
                    else:
                        mode = "diffusion"

            logger.info(
                f"Generating sample | mode={mode} source={'ema' if (use_ema and self.ema_model is not None) else 'model'} "
                f"prompt_tokens={prompt_token_len} target_total_tokens={target_total_len}"
            )
            if bool(l0_local_mode.get("active", False)):
                logger.info(
                    "L0 local runtime for generation: backend=%s window=%d autocast_bf16=%s",
                    l0_local_mode.get("backend", "pyg"),
                    int(l0_local_mode.get("window", 0)),
                    use_flash_gen_autocast,
                )

            with torch.no_grad():
                def _gen_amp_ctx():
                    return (
                        torch.autocast(device_type="cuda", dtype=amp_dtype("cuda"))
                        if use_flash_gen_autocast
                        else nullcontext()
                    )

                with _gen_amp_ctx():
                    if mode == "autoenc_query":
                        if not hasattr(model_for_gen, "generate"):
                            raise AttributeError("Selected generation mode 'autoenc_query' but model has no generate() method")

                        generate_kwargs = dict(
                            input_ids=input_ids,
                            max_length=target_total_len,
                            temperature=temperature,
                            do_sample=do_sample,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            rebuild_graph=True,
                        )

                        try:
                            generated_ids = model_for_gen.generate(
                                **generate_kwargs,
                                use_autoenc_query=True,
                                force_autoregressive=False,
                            )
                        except TypeError:
                            if hasattr(model_for_gen, "_generate_with_autoenc_query"):
                                generated_ids = model_for_gen._generate_with_autoenc_query(
                                    input_ids=input_ids,
                                    max_length=target_total_len,
                                    temperature=temperature,
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p,
                                    repetition_penalty=repetition_penalty,
                                )
                            else:
                                logger.warning("Autoenc-query generation unavailable; falling back to AR generate().")
                                mode = "ar"

                if mode == "autoenc_ar":
                    if not hasattr(model_for_gen, "generate"):
                        raise AttributeError("Selected generation mode 'autoenc_ar' but model has no generate() method")

                    generate_kwargs = dict(
                        input_ids=input_ids,
                        max_length=target_total_len,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        rebuild_graph=True,
                    )

                    with _gen_amp_ctx():
                        try:
                            generated_ids = model_for_gen.generate(
                                **generate_kwargs,
                                use_autoenc_ar=True,
                                force_autoregressive=False,
                            )
                        except TypeError:
                            try:
                                generated_ids = model_for_gen.generate(
                                    **generate_kwargs,
                                    use_autoenc_ar=True,
                                )
                            except TypeError:
                                logger.warning("Autoenc-AR kwargs unsupported; falling back to AR generate().")
                                mode = "ar"

                if mode == "diffusion":
                    if not hasattr(model_for_gen, "_generate_with_diffusion_from_prompt"):
                        logger.warning("Diffusion generation unavailable; falling back to AR generate().")
                        mode = "ar"
                    else:
                        with _gen_amp_ctx():
                            generated_ids = model_for_gen._generate_with_diffusion_from_prompt(
                                input_ids=input_ids,
                                generate_length=max(1, int(target_total_len - prompt_token_len)),
                                chunk_size=max(1, int(diffusion_chunk_size)),
                                temperature=temperature,
                                do_sample=do_sample,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                num_cycles=max(1, int(diffusion_steps)),
                                max_total_cycles=max(1, int(diffusion_max_total_steps)),
                                initial_threshold=float(diffusion_initial_threshold),
                                final_threshold=float(diffusion_final_threshold),
                                random_remask_prob=float(diffusion_random_remask_prob),
                                random_remask_cutoff=float(diffusion_random_remask_cutoff),
                                use_autoenc_head=(autoenc_only_mode or autoenc_only_diffusion_mode),
                            )

                if mode == "recurrent":
                    if not hasattr(model_for_gen, "generate"):
                        raise AttributeError("Selected generation mode 'recurrent' but model has no generate() method")

                    generate_kwargs = dict(
                        input_ids=input_ids,
                        max_length=target_total_len,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_recurrent=True,
                        recurrent_l0_window=recurrent_l0_window,
                    )

                    with _gen_amp_ctx():
                        generated_ids = model_for_gen.generate(**generate_kwargs)

                if mode == "ar":
                    if not hasattr(model_for_gen, "generate"):
                        raise AttributeError("Selected generation mode 'ar' but model has no generate() method")

                    generate_kwargs = dict(
                        input_ids=input_ids,
                        max_length=target_total_len,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        rebuild_graph=True,
                    )

                    with _gen_amp_ctx():
                        try:
                            generated_ids = model_for_gen.generate(
                                **generate_kwargs,
                                use_imputation=False,
                                force_autoregressive=True,
                            )
                        except TypeError:
                            # Older signatures may not support one or both kwargs.
                            try:
                                generated_ids = model_for_gen.generate(
                                    **generate_kwargs,
                                    use_imputation=False,
                                )
                            except TypeError:
                                generated_ids = model_for_gen.generate(**generate_kwargs)

            generated_ids = generated_ids.to(self.device)
            generated_total_tokens = int(generated_ids.shape[1]) if generated_ids.dim() == 2 else 0
            generated_new_tokens = max(0, generated_total_tokens - prompt_token_len)
            logger.info(
                "Sample generation lengths: prompt=%d total=%d new=%d",
                prompt_token_len,
                generated_total_tokens,
                generated_new_tokens,
            )
            if generated_new_tokens > 0:
                special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
                if special_ids:
                    tail_ids = generated_ids[0, prompt_token_len:generated_total_tokens].tolist()
                    special_count = sum(1 for tid in tail_ids if int(tid) in special_ids)
                    special_ratio = float(special_count) / float(max(1, len(tail_ids)))
                    logger.info("Generated tail special-token ratio: %.3f", special_ratio)

            # Decode prompt and continuation separately so callers can show the boundary.
            prompt_text = self.tokenizer.decode(
                generated_ids[0, :prompt_token_len], skip_special_tokens=True)
            continuation_text = self.tokenizer.decode(
                generated_ids[0, prompt_token_len:generated_total_tokens], skip_special_tokens=True)
            if return_parts:
                return prompt_text, continuation_text
            return prompt_text + continuation_text

        except Exception as e:
            logger.error(f"Error generating sample: {str(e)}", exc_info=True)
            return ("", f"Error generating sample: {str(e)}") if return_parts else f"Error generating sample: {str(e)}"
        
    
    #old version of generate_sample, does not use mask
    # def generate_sample(
    #     self, 
    #     prompt: str, 
    #     max_length: int = 100,
    #     temperature: float = 1.0,
    #     do_sample: bool = True,
    #     top_k: int = 50,
    #     top_p: float = 0.9,
    #     repetition_penalty: float = 1.2,
    #     use_incremental: bool = True,  # Use incremental generation by default
    # ) -> str:
    #     """
    #     Generate a sample output from the model with incremental graph updates.
        
    #     Args:
    #         prompt: Text prompt to begin generation
    #         max_length: Maximum generation length
    #         temperature: Sampling temperature
    #         do_sample: Whether to use sampling
    #         top_k: Top-k filtering parameter
    #         top_p: Top-p filtering parameter
    #         repetition_penalty: Penalty for repeating tokens
    #         use_incremental: Whether to use incremental graph generation
            
    #     Returns:
    #         generated_text: Generated text from the model
    #     """
    #     try:
    #         self.model.eval()
            
    #         # Tokenize prompt
    #         inputs = self.tokenizer(
    #             prompt, 
    #             return_tensors="pt", 
    #             padding="max_length",
    #             max_length=min(64, self.model.max_seq_len // 2),  # Use reasonable context size
    #             truncation=True
    #         )
            
    #         input_ids = inputs.input_ids.to(self.device)
            
    #         # Generate text using the appropriate method
    #         with torch.no_grad():
    #             if use_incremental and hasattr(self.model, 'generate_with_graph_updates'):
    #                 # Use incremental graph updates
    #                 logger.info("Using incremental graph generation")
    #                 generated_ids = self.model.generate_with_graph_updates(
    #                     input_ids,
    #                     max_length=max_length,
    #                     temperature=temperature,
    #                     top_k=top_k,
    #                     top_p=top_p,
    #                     repetition_penalty=repetition_penalty,
    #                 )
    #             else:
    #                 # Fall back to standard generation
    #                 logger.info("Using standard graph generation")
    #                 generated_ids = self.model.generate(
    #                     input_ids,
    #                     max_length=max_length,
    #                     temperature=temperature,
    #                     do_sample=do_sample,
    #                     top_k=top_k,
    #                     top_p=top_p,
    #                     repetition_penalty=repetition_penalty,
    #                     rebuild_graph=True,  # More reliable approach
    #                 )
            
    #         # Decode and return
    #         return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
    #     except Exception as e:
    #         logger.error(f"Error generating sample: {str(e)}", exc_info=True)
    #         return f"Error generating sample: {str(e)}"
    
    def save_checkpoint(self, path, extra_data=None, is_best=False,):
        """
        Save a checkpoint of the current state.
        
        Args:
            path: Path to save the checkpoint
            extra_data: Additional data to include in the checkpoint
            is_best: Whether this is a best model checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)


        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Add optimizer and scheduler if available
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Add extra data if provided
        if extra_data is not None:
            checkpoint.update(extra_data)
        
        # Save the checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        
        if is_best:
            logger.info("This is a best model checkpoint")
    
    def load_checkpoint(self, path):
        """
        Load a checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if it exists and optimizer is provided
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists and scheduler is provided
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Loaded checkpoint from {path}, current epoch: {self.current_epoch}")
        
        return checkpoint
    
    def save_ema_checkpoint(self, path, extra_data=None, is_best=False,):
        """
        Save a checkpoint of the current state.
        
        Args:
            path: Path to save the checkpoint
            extra_data: Additional data to include in the checkpoint
            is_best: Whether this is a best model checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.ema_model.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Add extra data if provided
        if extra_data is not None:
            checkpoint.update(extra_data)
        
        # Save the checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        
        if is_best:
            logger.info("This is a best ema model checkpoint")
    
    def load_ema_checkpoint(self, path):
        """
        Load a checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model weights
        self.ema_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Loaded checkpoint from {path}, current epoch: {self.current_epoch}")
        
        return checkpoint
