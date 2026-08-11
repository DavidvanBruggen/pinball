# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Build Pinball as a reusable backbone block for another model.

``build_pinball`` mirrors the interface the DNA/omni model already calls, so a host model
can embed Pinball the way it would embed a stack of transformer blocks::

    from pinball import build_pinball

    model, tokenizer, args, device = build_pinball(
        cfg_path="args.json", num_tracks=768, tie_weights=False,
    )
    h = model(x)          # [B, L, 768] -> [B, L, 768]

Notes worth knowing before using this:

* **No weights are loaded.** This returns a freshly initialised module; the host model's
  own checkpoint carries the Pinball parameters (they serialise as a normal submodule).
  To load trained Pinball weights instead, build here and then
  ``load_state_dict(torch.load(ckpt)["model_state_dict"], strict=False)``.
* ``num_tracks`` is the **feature width in and out**, not a track count. Pinball uses one
  ``vocab_size`` knob for both the input projection (``nn.Linear(vocab_size, hidden_dim)``)
  and the output projection (``nn.Linear(hidden_dim, vocab_size)``), so the block is
  width-preserving; ``hidden_dim`` from the config is the separate internal width.
* Configs are routed through :class:`~pinball.config.PinballConfig` so its friendly
  aliases expand. That matters: ``attn_backend``, ``qkv_sharing``, ``use_hqd``,
  ``ar_graph_causal``, ``gradient_checkpointing`` and ``train_mode`` are rewritten there
  and are invisible to the model registry otherwise — a raw namespace silently leaves
  ``attn_backend: flash`` as the ``pyg`` default.
"""
from __future__ import annotations

import json
import logging
import pathlib
import random
import re
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import PinballConfig
from .model import build_model, count_parameters, normalize_model_type
from .model_inputs import resolve_model_inputs

logger = logging.getLogger("pinball.instantiate")

# Config keys the trainer/data/generation layers own. They never reach build_model, so the
# unused-key audit must not flag them.
_NON_MODEL_KEYS = frozenset({
    "batch_size", "checkpoint_dir", "dataset", "data_path", "device", "eval_every",
    "eval_interval", "early_stopping_patience", "gen_max_length", "gen_max_new_tokens",
    "gen_prompt_tokens", "generate_every", "generation_method", "gradient_accumulation_steps",
    "grad_accum", "learning_rate", "log_interval", "longctx_diag_every", "mask_prob",
    "max_grad_norm", "min_lr", "mixed_precision", "modality", "muon_adjust_lr_fn",
    "keep_last_milestones", "muon_betas", "muon_lr_mult", "num_epochs", "optimizer",
    "resume_from_checkpoint", "save_last_every_epochs", "save_milestone_every_epochs",
    "resume_strict",
    "resume_ema_from_checkpoint", "samples_per_epoch", "save_every", "seed", "text_file",
    "tokenizer_name", "train_objective_mode", "use_ema", "use_hybrid_masking", "val_split",
    "val_samples", "warmup_steps", "weight_decay", "input_mode", "model_type", "vocab_size",
    "block_size", "chunked_ce_seq_chunk", "train_feature_chunked_ce_enable",
    "do_sample", "temperature", "top_k", "top_p", "repetition_penalty", "sample_all_methods",
    "use_incremental_generation", "cycle_warmup_epochs", "cycle_warmup_mode",
    "use_cycle_warmup", "variable_train_cycles",
})

# Friendly aliases PinballConfig expands (config.py::_apply_friendly_aliases). They are
# legitimate config keys even though the registry never reads them under these names.
_ALIAS_KEYS = frozenset({
    "attn_backend", "qkv_sharing", "use_hqd", "ar_graph_causal", "ar_hier_edge_mode",
    "gradient_checkpointing", "grad_accum", "train_mode",
})

# Flags whose resolved value is worth printing: silent fallback to a default here changes
# the model's behaviour substantially.
_RECIPE_FLAGS = (
    "l0_local_backend", "hier_ar_enable", "l0_ar_enable", "unified_refinement_cycles",
    "num_refinement_layers", "spatial_curve", "local_pack_cross_level",
    "hier_upward_refresh", "hier_downward_refresh", "refine_cond_mode",
    "hierarchical_query_descent_enable",
)


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config_dict(cfg_path: Union[str, pathlib.Path]) -> dict:
    cfg_path = pathlib.Path(cfg_path)
    suffix = cfg_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(cfg_path.read_text())
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML config files") from exc
        data = yaml.safe_load(cfg_path.read_text())
    else:
        raise ValueError(f"Unsupported config file extension: {cfg_path.suffix!r}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(data).__name__}")
    return data


def load_args(
    cfg_path: Optional[Union[str, pathlib.Path]] = None,
    extra_cli: Optional[Sequence[str]] = None,
) -> PinballConfig:
    """Load a ``.json`` / ``.yaml`` config into a :class:`PinballConfig`.

    Defaults come from ``model_registry``'s ``getattr(args, name, default)`` reads rather
    than an argparse parser, so a partial config is legal — any key the file omits falls
    back to the registry default.

    ``extra_cli`` exists for signature parity with the original argparse-based helper and
    must be empty; there is no parser to feed it to.
    """
    if extra_cli:
        raise ValueError(
            "extra_cli is not supported: this config path has no argparse parser. "
            "Pass overrides via build_pinball(override={...}) instead."
        )
    return PinballConfig.from_dict(_load_config_dict(cfg_path) if cfg_path is not None else {})


def _known_config_keys() -> frozenset:
    """Config keys the model registry reads, scraped from its source.

    Self-maintaining (no hand-kept list to drift), and only ever used for a warning.
    """
    try:
        from .model import model_registry
        src = pathlib.Path(model_registry.__file__).read_text()
        names = set(re.findall(r'getattr\(\s*args\s*,\s*["\']([A-Za-z0-9_]+)["\']', src))
    except Exception:  # pragma: no cover - the audit is best-effort by design
        return frozenset()
    # A refactor that breaks the regex would flag every key; stay quiet instead.
    return frozenset(names) if len(names) >= 100 else frozenset()


def _audit_config_keys(args) -> None:
    known = _known_config_keys()
    if not known:
        return
    unused = sorted(
        k for k in vars(args)
        if k not in known and k not in _ALIAS_KEYS and k not in _NON_MODEL_KEYS
        and not k.startswith("_") and not k.startswith("image_")
    )
    if unused:
        logger.warning(
            "%d config key(s) reach neither the model nor a known trainer setting "
            "(typo, or a knob from an older Pinball): %s",
            len(unused), ", ".join(unused),
        )


def _resolve_device(args, device: Optional[Union[str, torch.device]]) -> torch.device:
    # The explicit argument wins over the config: a host model embedding Pinball needs the
    # submodule on ITS device, and configs carry a stale hard-coded "device" more often
    # than not. (The original helper let the config win, which silently pinned the block
    # to whatever GPU the config was written on.)
    if device is not None:
        return torch.device(device)
    if getattr(args, "device", None):
        return torch.device(str(args.device))
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_tokenizer(args):
    """Tokenizer for the model's coarse-seed init.

    The model reads only ``mask_token_id`` / ``pad_token_id`` and never stores it, so the
    stub is sufficient whenever the config names no tokenizer or the real one fails to
    load (offline hosts). ``None`` is not an option — the constructor dereferences it.
    """
    name = str(getattr(args, "tokenizer_name", "") or "")
    if not name:
        return SimpleNamespace(mask_token_id=0, pad_token_id=0)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if bool(getattr(args, "use_hybrid_masking", False)) and tok.mask_token is None:
            tok.add_special_tokens({"mask_token": "<mask>"})
        return tok
    except Exception as exc:
        logger.warning("Could not load tokenizer %r (%s); using an id-only stub.", name, exc)
        return SimpleNamespace(mask_token_id=0, pad_token_id=0)


def build_pinball(
    cfg_path: Union[str, pathlib.Path],
    device: Optional[Union[str, torch.device]] = None,
    override: Optional[dict] = None,
    num_tracks: Optional[int] = None,
    tie_weights: bool = False,
    set_global_seed: bool = True,
    warn_unused_keys: bool = True,
) -> Tuple[torch.nn.Module, Any, PinballConfig, torch.device]:
    """Build a Pinball backbone from a config file.

    Args:
        cfg_path: ``.json`` / ``.yaml`` config.
        device: where to place the model. Takes precedence over ``cfg.device``; falls back
            to it, then to cuda/mps/cpu.
        override: config keys applied on top of the file.
        num_tracks: feature width **in and out**. Defaults to ``hidden_dim`` (and is
            ignored for ``model_type: transformer``, which uses the tokenizer vocabulary).
        tie_weights: only meaningful for token input; forced off for feature input, where
            tying is a shape error.
        set_global_seed: seed the process RNGs from ``cfg.seed``. Off by default is often
            what a host model wants — this mutates global RNG state mid-construction.
        warn_unused_keys: log config keys that reach nothing (catches typos).

    Returns:
        ``(model, tokenizer, args, device)``. The model is on ``device`` with freshly
        initialised weights.
    """
    args = load_args(cfg_path)
    for key, value in (override or {}).items():
        setattr(args, key, value)

    device = _resolve_device(args, device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    if set_global_seed:
        set_seed(int(getattr(args, "seed", 42) or 42))

    if warn_unused_keys:
        _audit_config_keys(args)

    model_type = normalize_model_type(getattr(args, "model_type", "pinball"))
    modality = str(getattr(args, "modality", "text")).lower()

    # Image configs derive the token grid, spatial_dims (curve mode) and feature width from
    # the image knobs; text/other configs just need a tokenizer. Feature input is the
    # default here because that is what an embedded backbone consumes — the argparse
    # front end this replaces defaulted the same way.
    inputs = None
    if modality == "image":
        inputs = resolve_model_inputs(args)
        tokenizer = inputs.tokenizer
        vocab_size = inputs.vocab_size
        input_mode = str(getattr(args, "input_mode", inputs.input_mode))
        block_size = inputs.block_size
        tie_weights = bool(tie_weights) and inputs.tie_weights
        if inputs.vq_tokenizer is not None:
            args.input_mode = input_mode = "tokens"
    else:
        tokenizer = _resolve_tokenizer(args)
        input_mode = str(getattr(args, "input_mode", "features"))
        block_size = int(getattr(args, "block_size", 1024))
        if input_mode == "tokens":
            vocab_size = len(tokenizer)
        else:
            vocab_size = int(num_tracks if num_tracks is not None else getattr(args, "hidden_dim", 384))

    # num_tracks is the explicit request for the block's in/out width and wins over any
    # width derived above -- but only for feature input, where the width is a free choice.
    # Under token input the width IS the vocabulary (a tokenizer's, or a VQ codebook's) and
    # overriding it would build a model that cannot embed its own token ids.
    if num_tracks is not None:
        if input_mode == "features":
            vocab_size = int(num_tracks)
        else:
            logger.warning(
                "Ignoring num_tracks=%s: input_mode=%r makes the width the vocabulary (%d).",
                num_tracks, input_mode, int(vocab_size),
            )

    if input_mode == "features" and tie_weights:
        # nn.Linear(vocab, hidden).weight is [hidden, vocab]; the output projection needs
        # [vocab, hidden]. Tying is accepted at construction and raises at first forward.
        logger.warning("tie_weights is not supported with feature input; ignoring it.")
        tie_weights = False
    if model_type == "transformer":
        tie_weights = True

    model = build_model(
        args,
        tokenizer=tokenizer,
        vocab_size=int(vocab_size),
        input_mode=input_mode,
        tie_weights=bool(tie_weights),
        max_seq_len=int(block_size),
        class_cond_enable=bool(modality == "image" and getattr(args, "class_cond_enable", True)),
    ).to(device)

    if inputs is not None and inputs.vq_tokenizer is not None:
        model.mask_token_id = int(inputs.vq_tokenizer.mask_token_id)
    # Kept for parity with the training entry point. The flag is inert — features come out
    # because vocab_size is the output width, not because of this.
    model.emit_features_only = True

    logger.info(
        "Built %s: %s params | input_mode=%s width=%d hidden=%d block_size=%d device=%s",
        model_type, f"{count_parameters(model):,}", input_mode, int(vocab_size),
        int(getattr(args, "hidden_dim", 0)), int(block_size), device,
    )
    logger.info(
        "Recipe: %s",
        " ".join(f"{f}={getattr(args, f, '<default>')}" for f in _RECIPE_FLAGS),
    )
    return model, tokenizer, args, device
