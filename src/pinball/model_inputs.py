# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Config -> ``build_model`` arguments.

Everything ``build_model`` needs that the config does not state literally lives here:
the tokenizer, the feature/vocab width, the input mode, and (for image runs) the token
grid + ``spatial_dims`` that curve mode reads. It used to be inline in ``cli.py``; it is
factored out so the training entry point and ``build_pinball`` (embedding Pinball as a
submodule of another model) derive identical arguments from the same config instead of
drifting apart as new token modes are added.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from transformers import AutoTokenizer

logger = logging.getLogger("pinball.model_inputs")


def build_text_tokenizer(cfg):
    """HF tokenizer for text runs, with a ``<mask>`` added for the masked objective."""
    tok = AutoTokenizer.from_pretrained(getattr(cfg, "tokenizer_name", "gpt2"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Masked-diffusion objective needs a [MASK] token in the vocab.
    objective = str(getattr(cfg, "train_objective_mode", "ar")).lower()
    if objective in {"masked", "hybrid"} and tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    return tok


@dataclass
class ModelInputs:
    """The resolved arguments ``build_model`` is called with."""

    tokenizer: Any
    vocab_size: int
    input_mode: str
    tie_weights: bool
    block_size: int
    vq_tokenizer: Optional[Any] = None


def resolve_model_inputs(cfg, block_size: Optional[int] = None) -> ModelInputs:
    """Derive the ``build_model`` arguments from ``cfg``.

    Mutates ``cfg`` for image runs (``graph_grid_height/width``, ``spatial_dims``,
    ``block_size``) exactly as the training path always has — the model reads the grid
    off the config, and curve mode reads ``spatial_dims``.
    """
    if block_size is None:
        block_size = int(getattr(cfg, "block_size", 1024))
    block_size = int(block_size)
    modality = str(getattr(cfg, "modality", "text")).lower()

    if modality != "image":
        tokenizer = build_text_tokenizer(cfg)
        return ModelInputs(tokenizer, len(tokenizer), "tokens", True, block_size)

    # Image runs: discrete MaskGIT uses the VQ codebook as the vocabulary (tokens mode),
    # continuous latent/rgb modes feed features. NOTE: unlike the legacy script we do NOT
    # force graph_geometry_mode=grid2d — curve mode (spatial_curve) keeps "sequence"
    # geometry so the pack/refresh recipe runs; the grid enters via spatial_dims.
    vq_tokenizer = None
    image_objective = str(getattr(cfg, "image_objective", "diffusion")).lower()
    maskgit_variant = str(getattr(cfg, "image_maskgit_variant", "continuous")).lower()
    if image_objective == "maskgit" and maskgit_variant == "discrete":
        vq_name = str(getattr(cfg, "image_maskgit_vq_model_name", "") or "")
        if not vq_name:
            raise ValueError("Discrete MaskGIT requires image_maskgit_vq_model_name in the config.")
        import torch

        from .model.image_maskgit_vq import ImageMaskGITVQTokenizer
        vq_tokenizer = ImageMaskGITVQTokenizer.from_pretrained(
            vq_name, device=torch.device("cpu"),
            subfolder=getattr(cfg, "image_maskgit_vq_subfolder", None),
        )
        vq_grid = vq_tokenizer.infer_grid_shape(int(getattr(cfg, "image_size", 256)))
        cfg.graph_grid_height, cfg.graph_grid_width = int(vq_grid[0]), int(vq_grid[1])
        if not getattr(cfg, "spatial_dims", None):
            cfg.spatial_dims = [int(vq_grid[0]), int(vq_grid[1])]
        tokenizer = vq_tokenizer  # exposes mask_token_id for the model's coarse-seed init
        vocab_size = int(vq_tokenizer.vocab_size)
        input_mode, tie_weights = "tokens", False
        logger.info("Discrete MaskGIT VQ: codebook=%d mask_id=%d grid=%dx%d vocab=%d",
                    int(vq_tokenizer.codebook_size), int(vq_tokenizer.mask_token_id),
                    int(vq_grid[0]), int(vq_grid[1]), vocab_size)
    else:
        tokenizer = SimpleNamespace(mask_token_id=0, pad_token_id=0)  # features mode: unused ids
        image_tok_mode = str(getattr(cfg, "image_token_mode", "latent")).lower()
        if image_tok_mode == "raw_rgb_patches":
            ps = int(getattr(cfg, "image_patch_size", 16))
            vocab_size = 3 * ps * ps
        elif image_tok_mode == "rgb_unet":
            vocab_size = int(getattr(cfg, "image_rgb_unet_token_dim", 64))
        else:  # latent
            vocab_size = int(getattr(cfg, "image_latent_channels", 4))
        input_mode, tie_weights = "features", False
        # Grid sync (the discrete branch gets this from the VQ tokenizer): the token grid
        # side comes from the mode's downsample factor, and feeds spatial_dims (curve mode)
        # + the block_size sync below.
        if image_tok_mode == "raw_rgb_patches":
            ds = int(getattr(cfg, "image_patch_size", 16))
        elif image_tok_mode == "rgb_unet":
            ds = int(getattr(cfg, "image_rgb_unet_downsample", 16))
        else:
            ds = int(getattr(cfg, "image_latent_downsample", 8))
        side = max(1, int(getattr(cfg, "image_size", 256)) // max(1, ds))
        cfg.graph_grid_height, cfg.graph_grid_width = side, side
        if not getattr(cfg, "spatial_dims", None):
            cfg.spatial_dims = [side, side]
        logger.info("Continuous image mode: token_mode=%s grid=%dx%d feature_dim=%d",
                    image_tok_mode, side, side, int(vocab_size))

    expected_tokens = int(cfg.graph_grid_height or 0) * int(cfg.graph_grid_width or 0)
    if expected_tokens > 0 and expected_tokens != block_size:
        logger.info("Image modality: block_size %d -> %d (grid %dx%d)",
                    block_size, expected_tokens, int(cfg.graph_grid_height), int(cfg.graph_grid_width))
        block_size = expected_tokens
        cfg.block_size = expected_tokens

    return ModelInputs(tokenizer, int(vocab_size), input_mode, tie_weights, block_size, vq_tokenizer)
