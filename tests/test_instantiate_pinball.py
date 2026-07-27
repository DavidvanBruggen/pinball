# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""``build_pinball``: Pinball as a backbone block inside another model.

The contract under test is the one the DNA/omni host model relies on::

    model, tokenizer, args, device = build_pinball(cfg_path=..., num_tracks=768, tie_weights=False)
    h = model(x)                     # [B, L, D] -> [B, L, D], one tensor, no kwargs

i.e. a drop-in replacement for an ``nn.Sequential`` of residual transformer blocks.

Run directly:   python tests/test_instantiate_pinball.py
Run via pytest: pytest tests/test_instantiate_pinball.py -s
"""
import json
import logging
import pathlib
import sys
import tempfile

import torch

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pinball import PinballConfig, build_pinball, load_args
from pinball.model_inputs import resolve_model_inputs

# Deliberately sparse: the point is that a partial config is legal and the registry
# defaults fill the rest. No tokenizer_name -> the id-only stub path (works offline).
TINY_CFG = dict(
    model_type="pinball",
    hidden_dim=64,
    num_heads=4,
    num_refinement_layers=2,
    num_layers=[0, 0, 0, 0],
    internal_cycles=[0, 0, 0, 0],
    refinement_style="unified",
    unified_refinement_cycles=1,
    compression_ratios=[8, 4, 2],
    overlap_ratios=[0.1, 0.2, 0.4],
    local_attn_windows=[16, 8, 8, 8],
    local_attn_levels=[0, 1, 2, 3],
    block_size=64,
    dropout=0.0,
    norm_type="layernorm",
    lap_pe_k=0,
    l0_cycles=0,
    iterative_refinement_cycles=0,
    local_connectivity_window_size=0,
    attn_backend="sdpa",
    l0_local_window=16,
    device="cpu",
)

WIDTH = 32
SEQ = 64


def _write_cfg(tmpdir, extra=None, suffix=".json"):
    cfg = dict(TINY_CFG)
    cfg.update(extra or {})
    path = pathlib.Path(tmpdir) / f"cfg{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(cfg))
    else:
        import yaml
        path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_consumer_contract():
    """The exact call the host model makes, and the exact shape contract it relies on."""
    with tempfile.TemporaryDirectory() as tmp:
        out = build_pinball(cfg_path=_write_cfg(tmp), num_tracks=WIDTH, tie_weights=False)

    assert isinstance(out, tuple) and len(out) == 4, "must return (model, tokenizer, args, device)"
    model, tokenizer, args, device = out
    assert isinstance(model, torch.nn.Module)
    assert isinstance(args, PinballConfig)
    assert isinstance(device, torch.device)

    x = torch.randn(2, SEQ, WIDTH, device=device)
    with torch.no_grad():
        h = model(x)  # single positional arg, no kwargs — the whole point

    assert torch.is_tensor(h), f"expected one tensor (nn.Sequential drop-in), got {type(h)}"
    assert h.shape == x.shape, f"must be width- and length-preserving: {tuple(h.shape)} != {tuple(x.shape)}"
    assert torch.isfinite(h).all()
    print(f"  contract OK: {tuple(x.shape)} -> {tuple(h.shape)}, single tensor")


def test_backward_flows():
    """Grads must reach the input projection and the output head."""
    with tempfile.TemporaryDirectory() as tmp:
        model, _, _, device = build_pinball(cfg_path=_write_cfg(tmp), num_tracks=WIDTH)

    model(torch.randn(2, SEQ, WIDTH, device=device)).square().mean().backward()
    for name in ("token_embedding", "output_projection"):
        g = getattr(model, name).weight.grad
        assert g is not None, f"{name} received no gradient"
        assert torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} gradient is zero/non-finite"
    print("  backward OK: token_embedding + output_projection both receive gradient")


def test_tie_weights_guard():
    """tie_weights=True with feature input is a shape error at forward; it must be refused."""
    with tempfile.TemporaryDirectory() as tmp:
        model, _, _, device = build_pinball(cfg_path=_write_cfg(tmp), num_tracks=WIDTH, tie_weights=True)

    emb, out = model.token_embedding.weight, model.output_projection.weight
    assert emb.data_ptr() != out.data_ptr(), "weights were tied despite feature input"
    with torch.no_grad():
        model(torch.randn(1, SEQ, WIDTH, device=device))  # would raise if tied
    print("  tie_weights guard OK: coerced to False, forward runs")


def test_num_tracks_sets_both_widths():
    """num_tracks is the in AND out width; without it the width falls back to hidden_dim."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_cfg(tmp)
        m_a, _, _, _ = build_pinball(cfg_path=path, num_tracks=WIDTH)
        m_b, _, args_b, _ = build_pinball(cfg_path=path)

    assert m_a.token_embedding.in_features == WIDTH
    assert m_a.output_projection.out_features == WIDTH
    assert m_b.token_embedding.in_features == int(args_b.hidden_dim)
    print(f"  width OK: num_tracks={WIDTH} in/out; default falls back to hidden_dim={args_b.hidden_dim}")


def test_yaml_and_override_and_seed():
    """YAML loads like JSON, override wins over the file, seeding is opt-out."""
    with tempfile.TemporaryDirectory() as tmp:
        _, _, args, _ = build_pinball(
            cfg_path=_write_cfg(tmp, suffix=".yaml"),
            num_tracks=WIDTH,
            override={"num_refinement_layers": 3},
        )
        assert int(args.num_refinement_layers) == 3, "override did not win over the config file"

        torch.manual_seed(1234)
        before = torch.initial_seed()
        build_pinball(cfg_path=_write_cfg(tmp, {"seed": 7}), num_tracks=WIDTH, set_global_seed=False)
        assert torch.initial_seed() == before, "set_global_seed=False still touched global RNG"

        build_pinball(cfg_path=_write_cfg(tmp, {"seed": 7}), num_tracks=WIDTH, set_global_seed=True)
        assert torch.initial_seed() == 7, "set_global_seed=True did not seed from cfg.seed"
    print("  yaml/override/seed OK")


def test_device_argument_wins_over_config():
    """A host embedding Pinball must be able to place it; the config's device is a fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        _, _, _, device = build_pinball(
            cfg_path=_write_cfg(tmp, {"device": "cuda:7"}), num_tracks=WIDTH, device="cpu",
        )
    assert device.type == "cpu", f"explicit device= was ignored (got {device})"
    print("  device OK: explicit argument overrides cfg.device")


def test_friendly_aliases_expand():
    """PinballConfig aliases must expand — a raw namespace would silently drop them."""
    with tempfile.TemporaryDirectory() as tmp:
        _, _, args, _ = build_pinball(
            cfg_path=_write_cfg(tmp, {
                "attn_backend": "sdpa",
                "use_hqd": False,
                "ar_graph_causal": True,
                "gradient_checkpointing": False,
            }),
            num_tracks=WIDTH,
        )
    # The aliases are consumed (popped) and rewritten to the names the registry reads.
    assert args.l0_local_backend == "sdpa", "attn_backend did not expand"
    assert args.hierarchical_query_descent_enable is False, "use_hqd did not expand"
    assert args.hier_ar_enable is True and args.l0_ar_enable is True, "ar_graph_causal did not expand"
    assert args.use_gradient_checkpointing is False, "gradient_checkpointing did not expand"
    assert not hasattr(args, "attn_backend"), "alias should be consumed, not left dangling"
    print("  aliases OK: attn_backend/use_hqd/ar_graph_causal/gradient_checkpointing all expanded")


def test_modern_knobs_reach_the_model():
    """Knobs added since the old config format must take effect when present."""
    knobs = {
        "spatial_curve": "hilbert",
        "spatial_dims": [8, 8],
        "local_pack_cross_level": True,
        "hier_upward_refresh": True,
        "hier_downward_refresh": True,
        "hier_downward_bidi_parent": True,
        "refine_cond_mode": "film",
        "upper_init": "pooled",
        "block_size": 64,
    }
    with tempfile.TemporaryDirectory() as tmp:
        model, _, args, device = build_pinball(cfg_path=_write_cfg(tmp, knobs), num_tracks=WIDTH)

    for key, want in knobs.items():
        assert getattr(args, key) == want, f"{key} did not survive into the config"
    # ...and the model actually picked them up rather than defaulting.
    assert str(getattr(model, "spatial_curve", "none")) == "hilbert", "spatial_curve did not reach the model"
    assert bool(getattr(model, "local_pack_cross_level", False)), "local_pack_cross_level did not reach the model"
    with torch.no_grad():
        h = model(torch.randn(2, SEQ, WIDTH, device=device))
    assert h.shape == (2, SEQ, WIDTH) and torch.isfinite(h).all()
    print("  modern knobs OK: curve/pack/refresh/film all reach the model and it still runs")


def test_matches_cli_derivation_for_image_configs():
    """build_pinball and the training entry point must derive identical build_model args."""
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "pinball_image_diffusion_latent.yaml"
    if not cfg_path.exists():
        print("  image config absent; skipping")
        return
    cfg_cli = PinballConfig.from_yaml(cfg_path)
    inputs = resolve_model_inputs(cfg_cli)
    args = load_args(cfg_path)

    assert inputs.input_mode == "features" and inputs.tie_weights is False
    assert inputs.vocab_size == int(args.image_latent_channels)
    side = int(args.image_size) // int(args.image_latent_downsample)
    assert list(cfg_cli.spatial_dims) == [side, side], "spatial_dims not derived (curve mode would be inert)"
    assert inputs.block_size == side * side
    print(f"  resolver OK: grid {side}x{side}, spatial_dims derived, feature_dim={inputs.vocab_size}")


def test_repo_configs_load():
    """Every config in configs/ must build through build_pinball and run a forward.

    These are training configs, so they exercise paths a hand-written backbone config
    would not: flash attention, the pack/curve recipe, discrete-VQ token mode, and the
    transformer baseline. Needs CUDA — several set attn_backend: flash, which has no CPU
    fallback.
    """
    if not torch.cuda.is_available():
        print("  no CUDA; skipping (flash-backed configs cannot run on CPU)")
        return
    cfg_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
    configs = sorted(cfg_dir.glob("*.yaml"))
    assert configs, "no configs found"

    for path in configs:
        model, _, args, device = build_pinball(
            cfg_path=str(path), num_tracks=WIDTH, device="cuda",
            set_global_seed=False, warn_unused_keys=False,
        )
        # TransformerLM projects features with feature_projection; Pinball reuses
        # token_embedding for both modes.
        proj = getattr(model, "feature_projection", None) or getattr(model, "token_embedding", None)
        seq = min(int(args.block_size), 256)
        if str(getattr(model, "input_mode", "tokens")) == "tokens" and hasattr(proj, "num_embeddings"):
            x = torch.randint(0, proj.num_embeddings, (1, seq), device=device)
            want_width = proj.num_embeddings
        else:
            x = torch.randn(1, seq, proj.in_features, device=device)
            want_width = WIDTH
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        assert out.shape[:2] == (1, seq), f"{path.name}: length not preserved ({tuple(out.shape)})"
        assert out.shape[-1] == want_width, f"{path.name}: width {out.shape[-1]} != {want_width}"
        assert torch.isfinite(out).all(), f"{path.name}: non-finite output"
        del model, out, x
        torch.cuda.empty_cache()
    print(f"  repo configs OK: {len(configs)} configs built and ran")


def test_unused_key_audit_warns_without_raising():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_cfg(tmp, {"hiden_dim": 128})  # typo
        logger = logging.getLogger("pinball.instantiate")

        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Cap()
        logger.addHandler(handler)
        try:
            build_pinball(cfg_path=path, num_tracks=WIDTH)  # must NOT raise
            warned = [m for m in records if "hiden_dim" in m]
            assert warned, "typo'd config key was not reported"
            records.clear()
            build_pinball(cfg_path=_write_cfg(tmp), num_tracks=WIDTH)
            assert not [m for m in records if "reach neither" in m], "clean config produced a false positive"
        finally:
            logger.removeHandler(handler)
    print("  audit OK: typo warned, clean config silent, neither raised")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    for fn in [
        test_consumer_contract,
        test_backward_flows,
        test_tie_weights_guard,
        test_num_tracks_sets_both_widths,
        test_yaml_and_override_and_seed,
        test_device_argument_wins_over_config,
        test_friendly_aliases_expand,
        test_modern_knobs_reach_the_model,
        test_matches_cli_derivation_for_image_configs,
        test_repo_configs_load,
        test_unused_key_audit_warns_without_raising,
    ]:
        print(f"\n== {fn.__name__}")
        fn()
    print("\nAll build_pinball tests passed.")
