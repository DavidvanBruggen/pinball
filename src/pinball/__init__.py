# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Pinball — a clean, hierarchical graph transformer for language modeling.

Public API:
    from pinball import build_model, PinballConfig, count_parameters
    from pinball import build_pinball          # Pinball as a backbone block in another model
    from pinball.train import EnhancedHierarchicalTrainer, train_with_hybrid_masking
    from pinball.data import create_karpathy_dataloaders
"""
from .model import build_model, count_parameters, normalize_model_type
from .config import PinballConfig
from .instantiate_PINBALL_model import build_pinball, load_args, set_seed

__all__ = [
    "build_model",
    "build_pinball",
    "load_args",
    "set_seed",
    "count_parameters",
    "normalize_model_type",
    "PinballConfig",
]
