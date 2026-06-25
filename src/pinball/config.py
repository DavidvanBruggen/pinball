# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""PinballConfig — the single configuration object for building and training Pinball.

It is an ``argparse.Namespace`` that the model registry and trainer read via
``getattr(cfg, name, default)``. You can pass either:

  * **friendly toggles** (recommended): ``attn_backend``, ``qkv_sharing``, ``use_hqd``,
    ``train_mode`` — these expand to the underlying flags below, or
  * **raw underlying names** (e.g. ``l0_local_backend``, ``share_transformers``,
    ``train_objective_mode``) — passed straight through.

Unknown keys are preserved untouched; the registry supplies sensible defaults for anything
you do not set, so configs stay small.

Friendly toggle expansions
--------------------------
  attn_backend = pyg | flash | sdpa      -> l0_local_backend
  qkv_sharing  = shared | separate       -> share_transformers / per_level_local_qkv
  use_hqd      = bool                     -> hierarchical_query_descent_enable
  train_mode   = ar | masked_diffusion    -> train_objective_mode (+ use_hybrid_masking)
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any

_ATTN_BACKENDS = {"pyg", "flash", "sdpa"}


def _apply_friendly_aliases(d: dict) -> dict:
    """Expand friendly toggles into the underlying argument names (non-destructive:
    an explicitly-set underlying name always wins via ``setdefault``)."""
    out = dict(d)

    if "attn_backend" in out:
        backend = str(out.pop("attn_backend")).lower()
        if backend not in _ATTN_BACKENDS:
            raise ValueError(f"attn_backend must be one of {sorted(_ATTN_BACKENDS)}, got {backend!r}")
        out.setdefault("l0_local_backend", backend)

    if "qkv_sharing" in out:
        sharing = str(out.pop("qkv_sharing")).lower()
        if sharing not in {"shared", "separate"}:
            raise ValueError("qkv_sharing must be 'shared' or 'separate'")
        # qkv_sharing controls ONLY whether the local-attention QKV projections are
        # shared across hierarchy levels (shared) or per-level (separate). It must NOT
        # flip ``share_transformers``: that flag reuses the per-level ``level_transformers``
        # stack (built from ``num_layers``) as the refinement stack, which is empty in the
        # standard configs (depth lives in ``num_refinement_layers``). Setting it True there
        # silently yields zero refinement layers — an identity model that only trains the
        # embeddings. The dedicated ``refinement_transformers`` stack already honors
        # ``per_level_local_qkv``, so the toggle maps there and leaves share_transformers off.
        out.setdefault("per_level_local_qkv", sharing == "separate")

    if "use_hqd" in out:
        out.setdefault("hierarchical_query_descent_enable", bool(out.pop("use_hqd")))

    # AR graph connectivity: one umbrella switch that makes the graph causal for
    # autoregressive training (level-wise + L0 intra-level edges are time-forward only).
    # The individual underlying dials (hier_ar_enable, l0_ar_enable,
    # hier_ar_allow_same_time, enable_l0_parent_edges, l0_parent_edges_bidirectional,
    # ensure_l0_past_l1_edges, ensure_past_hier_edges_all_levels, long_range_distance)
    # can still be set directly for fine-grained control and override this default.
    if "ar_graph_causal" in out:
        causal = bool(out.pop("ar_graph_causal"))
        out.setdefault("hier_ar_enable", causal)
        out.setdefault("l0_ar_enable", causal)
        # Without downward edges the AR causal filter detaches every coarse level from L0
        # (they become inert). Enable the staggered uncut L0->past-parent context edges so
        # the hierarchy actually informs the prediction. They are strictly-past by
        # construction, so they survive the AR filter without leaking. Explicit override wins.
        if causal:
            out.setdefault("ensure_l0_past_parent_edges", True)

    # Friendly alias for the model's gradient-checkpointing flag.
    if "gradient_checkpointing" in out:
        out.setdefault("use_gradient_checkpointing", bool(out.pop("gradient_checkpointing")))

    # Friendly alias for the trainer's gradient-accumulation steps.
    if "grad_accum" in out:
        out.setdefault("gradient_accumulation_steps", int(out.pop("grad_accum")))

    if "train_mode" in out:
        mode = str(out.pop("train_mode")).lower()
        if mode in {"ar", "autoregressive"}:
            out.setdefault("train_objective_mode", "ar")
        elif mode in {"masked_diffusion", "masked", "diffusion"}:
            out.setdefault("train_objective_mode", "masked")
            out.setdefault("use_hybrid_masking", True)
        else:
            raise ValueError("train_mode must be 'ar' or 'masked_diffusion'")

    return out


class PinballConfig(argparse.Namespace):
    """Config namespace consumed by ``pinball.build_model`` and the trainer."""

    def __init__(self, **kwargs: Any):
        super().__init__(**_apply_friendly_aliases(kwargs))

    @classmethod
    def from_dict(cls, data: dict | None) -> "PinballConfig":
        return cls(**(data or {}))

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "PinballConfig":
        import yaml
        data = yaml.safe_load(pathlib.Path(path).read_text())
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return dict(vars(self))
