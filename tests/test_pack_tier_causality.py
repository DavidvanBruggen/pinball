# SPDX-License-Identifier: GPL-3.0-or-later
"""Causality probe for the 3-kernel packed lane (coarse window + top-global tier).

Future-perturbation test: change one token, assert every output at an EARLIER position
is bit-identical. Any change is future information flowing backwards.

The tier is the part that needs proving. It is an all-to-all flash kernel over the top
level(s), and when the budget spans several levels its rows are gathered level-major --
which is NOT time order. Causality depends entirely on re-sorting those rows by ar_time
before the causal mask is applied, so this test drives a multi-level tier on purpose.

Runs on CPU: bit-deterministic, so the same-input baseline is exactly 0 and any non-zero
delta at a past position is a real, reproducible leak. On CUDA, reduction non-determinism
alone produces ~1e-2 and buries the signal.

Run:   python tests/test_pack_tier_causality.py
"""
import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from transformers import AutoTokenizer

from pinball import PinballConfig, build_model, load_args

CONFIG = _ROOT / "configs" / "pinball_wikitext_3kernel.yaml"

# Shrunk for CPU. Depth and width do not affect whether an attention mask admits a
# future key, so the lane, hierarchy and tier structure are kept exactly as shipped.
SMALL = dict(
    block_size=1024, hidden_dim=64, num_heads=4, num_refinement_layers=4,
    dropout=0.0, attn_backend="sdpa",
    # flash has no CPU kernel; sdpa computes the SAME windowed mask densely, which is
    # what makes the leak test possible at all -- an admitted future key shows up
    # identically either way.
    l0_local_backend="sdpa",
)


def _build(overrides):
    torch.manual_seed(0)
    args = load_args(str(CONFIG))
    for key, value in {**SMALL, **overrides}.items():
        setattr(args, key, value)
    cfg = PinballConfig(**{k: v for k, v in vars(args).items()
                           if k in PinballConfig.__dataclass_fields__}) \
        if hasattr(PinballConfig, "__dataclass_fields__") else args
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                        tie_weights=True, max_seq_len=int(getattr(cfg, "block_size")))
    model.emit_features_only = True
    model.eval()
    return model, int(getattr(cfg, "block_size")), tok


def _measure(overrides, probe_positions=(0.35, 0.6, 0.9)):
    """Return (determinism, worst past delta, position, probed js)."""
    model, seq_len, tok = _build(overrides)
    torch.manual_seed(1)
    ids = torch.randint(0, tok.vocab_size, (1, seq_len))

    def feats(x):
        with torch.no_grad():
            out = model(x)
        return (out[0] if isinstance(out, (tuple, list)) else out).float()

    base = feats(ids)
    determinism = (base - feats(ids)).abs().max().item()

    worst, worst_pos, probed = 0.0, -1, []
    for fraction in probe_positions:
        j = int(seq_len * fraction)
        probed.append(j)
        perturbed = ids.clone()
        perturbed[0, j] = (perturbed[0, j].item() + 12345) % tok.vocab_size
        delta = (base - feats(perturbed)).abs().flatten(start_dim=2).amax(dim=-1)[0]
        past = delta[:j]
        if past.numel() and past.max().item() > worst:
            worst = past.max().item()
            worst_pos = int(past.argmax().item())
    return determinism, worst, worst_pos, probed


def _report(label, overrides):
    determinism, worst, pos, probed = _measure(overrides)
    print(f"[{label:<34}] determinism={determinism:.1e}  "
          f"max|delta| at past positions={worst:.3e}  (probed j={probed})")
    assert determinism < 1e-6, f"{label}: not deterministic on CPU"
    assert worst < 1e-6, f"{label}: LEAK -- past position {pos} moved by {worst:.3e}"


def test_lane_without_tier_is_causal():
    """Control: the packed lane alone, so a failure below isolates to the tier."""
    _report("coarse lane only, tier OFF", dict(local_pack_top_global=False))


def test_single_level_tier_is_causal():
    _report("tier ON, top level only", dict(local_pack_top_global=True,
                                            local_pack_top_global_budget=2))


def test_multi_level_tier_is_causal():
    """The real arm: budget spans several levels, so rows arrive level-major."""
    _report("tier ON, multi-level (sqrt)", dict(local_pack_top_global=True,
                                                local_pack_top_global_budget="sqrt"))


def test_wide_multi_level_tier_is_causal():
    """A budget deep enough to pull in large, densely-interleaved coarse levels."""
    _report("tier ON, wide multi-level", dict(local_pack_top_global=True,
                                              local_pack_top_global_budget=200))


def test_global_block_is_causal_at_tier_sized_budgets():
    """The AR-safe alternative to the tier, at the budgets the config actually uses.

    The global block selects whole levels top-down exactly as the tier does, but its rows
    are read-only keys masked by (k_row <= q_row), with the result written to the QUERY
    rows. Since every block row is also a query row, the top level still attends to
    itself, so this keeps the tier's function without its write-back.
    """
    for budget in (24, 64, 256):
        _report(f"global block budget {budget}", dict(local_pack_top_global=False,
                                                      local_pack_global_block=budget))


def test_sqrt_budget_multiplier_parses():
    """``sqrt*k`` is accepted and normalized; junk raises instead of silently meaning 0."""
    from pinball.model.hierarchical_flow_gat_cached_batch import _parse_sqrt_budget

    assert _parse_sqrt_budget("sqrt") == 1.0
    assert _parse_sqrt_budget("sqrt*2") == 2.0
    assert _parse_sqrt_budget("  SQRT * 2.5 ") == 2.5
    for bad in ("sqrt*0", "sqrt*-2", "sqrt*", "sqrt2", "cbrt", "", "sqrt*nan"):
        try:
            _parse_sqrt_budget(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not parse as a sqrt budget")
    print("  [OK] sqrt*k parsing")


def test_sqrt_multiplier_widens_the_tier():
    """k=2 must take strictly more levels than k=1 where a level boundary falls between.

    Guards the property the multiplier exists for: at ``block_size`` 1024 with this
    config's ratios, isqrt(1024) = 32 buys the top level alone while 2x it reaches one
    level further down. A regression that ignored the multiplier would tie the two.
    """
    counts = {}
    for budget in ("sqrt", "sqrt*2"):
        model, _, _ = _build(dict(local_pack_top_global=True,
                                  local_pack_top_global_budget=budget))
        bb = model.pinball if hasattr(model, "pinball") else model
        sizes = list(bb._predict_level_sizes(int(bb.max_seq_len)))
        top = len(sizes) - 1
        rows = [torch.zeros(int(n), dtype=torch.long) for n in sizes]
        tier = bb._resolve_global_tier(rows, top)
        counts[budget] = sum(int(r.numel()) for r in tier)
    print(f"  level sizes drive tier rows: sqrt={counts['sqrt']}, sqrt*2={counts['sqrt*2']}")
    assert counts["sqrt*2"] >= counts["sqrt"], counts
    assert counts["sqrt*2"] > counts["sqrt"], (
        "sqrt*2 did not widen the tier; the multiplier is being ignored")
    print("  [OK] sqrt*2 widens the tier")


def test_multiplied_sqrt_tier_is_causal():
    """Widening the budget must not buy reach by admitting future rows."""
    _report("tier ON, sqrt*2", dict(local_pack_top_global=True,
                                    local_pack_top_global_budget="sqrt*2"))


def test_auto_lane_window_is_causal():
    """auto sizes the lane from tier spacing; a wrong radius must not become a leak."""
    _report("tier ON + coarse_window auto", dict(local_pack_top_global=True,
                                                 local_pack_top_global_budget="sqrt",
                                                 local_pack_coarse_window="auto"))


if __name__ == "__main__":
    failures = []
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        try:
            fn()
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"  FAIL: {exc}")
    print("\nall causality probes passed" if not failures
          else f"\n{len(failures)} causality probe(s) FAILED")
    sys.exit(1 if failures else 0)
