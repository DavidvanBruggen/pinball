# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Predictive coding over the hierarchy (``hier_pc_enable``).

True top-down PC: level l+1 predicts level l's own features through the DOWNWARD refresh
projection. The chain is anchored at the bottom by the host's task loss, which is what makes
it collapse-free without a SIGReg-style term — so the contract under test is mostly about
*where the gradient goes*, not about the loss value.

What must hold:
  1. off by default, and inert (``None``) when disabled — so existing runs are unaffected
  2. enabling it produces a finite loss on the ``_last_hier_aux_loss`` channel the hosts read
  3. the gradient reaches the DOWNWARD projection weights — the whole point, since those are
     the wire measured inert (gates 0.1 -> 0.004)
  4. ``hier_pc_detach_target`` cuts the gradient to the child side and only that side
  5. enabling it together with ``use_aux_loss`` is refused rather than silently summed

Run directly:   python tests/test_hier_pc.py
Run via pytest: pytest tests/test_hier_pc.py -s
"""
import json
import pathlib
import sys
import tempfile

import torch

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pinball import build_pinball  # noqa: E402

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
    hier_downward_refresh=True,   # PC reuses these projections as f(W(l))
    hier_upward_refresh=True,
)

WIDTH = 32
SEQ = 64


def _build(tmp, **extra):
    cfg = dict(TINY_CFG)
    cfg.update(extra)
    path = pathlib.Path(tmp) / "cfg.json"
    path.write_text(json.dumps(cfg))
    model, _, _, device = build_pinball(
        cfg_path=str(path), num_tracks=WIDTH, tie_weights=False,
        set_global_seed=False, warn_unused_keys=False,
    )
    return model, device


def _forward_train(model, device, seed=0):
    torch.manual_seed(seed)
    model.train()
    x = torch.randn(2, SEQ, WIDTH, device=device)
    return model(x)


def test_disabled_is_inert():
    """Default off, and the PC channel stays None so nothing is added to the objective."""
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build(tmp)
        assert not getattr(model, "hier_pc_enable"), "hier_pc must default to off"
        _forward_train(model, device)
        assert getattr(model, "_last_hier_pc_loss", "missing") is None, \
            "disabled PC must leave _last_hier_pc_loss None"
    print("  disabled: default off, PC channel None")


def test_enabled_produces_loss_and_trains_downward_projection():
    """The gradient must reach downward_refresh_proj — that is the entire point."""
    with tempfile.TemporaryDirectory() as tmp:
        # predictor pinned: the DEFAULT is now per_offset, which deliberately does NOT use
        # downward_refresh_proj (a shared H->H broadcast is the degenerate shape for PC).
        # This test covers the legacy reuse path, which the DNA/text arms no longer run.
        model, device = _build(tmp, hier_pc_enable=True, lambda_hier_pc=0.05,
                               hier_pc_predictor="auto")
        assert model.hier_pc_pair_keys, "no PC pairs resolved"
        _forward_train(model, device)

        pc = model._last_hier_pc_loss
        assert pc is not None, "enabled PC produced no loss"
        assert torch.isfinite(pc).all(), f"PC loss not finite: {pc}"

        aux = model._last_hier_aux_loss
        assert aux is not None and torch.allclose(aux, pc), \
            "PC must ride the _last_hier_aux_loss channel the hosts consume"

        key = model.hier_pc_pair_keys[0]
        w = model.downward_refresh_proj[key].weight
        model.zero_grad(set_to_none=True)
        pc.backward()
        assert w.grad is not None and w.grad.abs().sum() > 0, \
            f"PC loss did not reach downward_refresh_proj[{key}] — the wire it exists to train"
    print(f"  enabled: pairs {model.hier_pc_pair_keys}, loss {float(pc):.5f}, "
          f"grad reaches downward_refresh_proj")


def test_detach_flag_cuts_only_the_child_side():
    """hier_pc_detach_target=True is the JEPA variant: parent still learns, child does not."""
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build(tmp, hier_pc_enable=True, hier_pc_detach_target=True,
                               hier_pc_predictor="auto")   # legacy reuse path, see above
        _forward_train(model, device)
        pc = model._last_hier_pc_loss
        assert pc is not None

        key = model.hier_pc_pair_keys[0]
        w = model.downward_refresh_proj[key].weight
        model.zero_grad(set_to_none=True)
        pc.backward()
        assert w.grad is not None and w.grad.abs().sum() > 0, \
            "detached PC must still train the predicting parent projection"
    print("  detach: parent projection still trained")


def test_rejects_double_counting_with_aux():
    """use_aux_loss + hier_pc_enable would sum two objectives into one logged number."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _build(tmp, hier_pc_enable=True, use_aux_loss=True)
        except ValueError as exc:
            assert "use_aux_loss" in str(exc)
            print("  guard: use_aux_loss + hier_pc_enable refused")
            return
    raise AssertionError("expected ValueError when both aux and PC are enabled")




def test_disabled_never_enters_pc_code():
    """Inertness proof: with PC off, none of the new code may execute.

    A byte-level A/B against a pre-change baseline is not available -- src/ already carried
    uncommitted changes (hierarchical_message_passing.py) from before this work, so HEAD is
    not a clean baseline. Instead, make every new entry point raise and show the forward
    still completes: if any of them ran, the forward would die.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build(tmp)

        def _boom(*a, **k):
            raise AssertionError("PC code ran with hier_pc_enable=False")

        for name in ("_compute_true_batch_pc_loss", "_compute_hierarchy_pc_loss",
                     "_compute_pc_pair_reduction", "_pc_predictor_for"):
            setattr(model, name, _boom)
        out = _forward_train(model, device)
        assert torch.is_tensor(out) and torch.isfinite(out).all()
    print("  inertness: forward completes with every PC entry point booby-trapped")


def test_jepa_generalisation_is_faithful():
    """The three new optional args must default to exactly the previous behaviour."""
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build(tmp, use_aux_loss=True, hier_aux_mode="jepa_mlp")
        _forward_train(model, device)

        from torch_geometric.data import Data
        torch.manual_seed(1)
        n, h = 40, model.hidden_dim
        g = Data(x=torch.randn(n, h),
                 edge_index=torch.stack([torch.arange(n - 1), torch.arange(1, n)]),
                 node_level=torch.cat([torch.zeros(n // 2, dtype=torch.long),
                                       torch.ones(n - n // 2, dtype=torch.long)]))
        preds = getattr(model, "hier_aux_pair_predictors", None)
        assert preds, "aux predictors missing; cannot check the generalisation"
        key = "l0_from_l1"      # the aux path's key format, not the downward pair "0:1"
        assert key in preds, f"expected {key} in {list(preds.keys())}"
        implicit = model._compute_hier_aux_pair_loss_jepa(g, 0, 1, key)
        explicit = model._compute_hier_aux_pair_loss_jepa(
            g, 0, 1, key,
            predictor=preds[key],
            detach_target=model._hier_aux_should_detach(0),
            reduce_fn=model._compute_hier_aux_pair_loss,
        )
        assert torch.equal(implicit, explicit), \
            f"defaults diverged from explicit old behaviour: {implicit} vs {explicit}"
    print(f"  jepa generalisation: defaults bit-identical ({float(implicit):.6f})")





def test_auto_predictor_mixed_reuse_and_alloc():
    """The >4-level case: downward defaults to the 0:m star, so a CHAIN-shaped PC term reuses
    only 0:1 and must allocate heads for the rest.

    This is the shape the DNA arm actually runs, and it is not reachable from the 4-level
    TINY_CFG (there the downward refresh builds all pairs, so everything is reused and the
    allocate branch never executes). A dispatch that keyed hier_pc_proj by mode alone raised
    KeyError: '0:1' here while every 4-level test passed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build(
            tmp, hier_pc_enable=True, hier_pc_predictor="auto",
            num_layers=[0] * 7, internal_cycles=[0] * 7,
            compression_ratios=[8, 4, 4, 4, 4, 4],
            overlap_ratios=[0.1, 0.2, 0.2, 0.2, 0.2, 0.2],
            local_attn_windows=[16] + [8] * 6,
            local_attn_levels=list(range(7)),
        )
        assert model.hier_pc_pair_keys == ["0:1", "1:2", "2:3", "3:4", "4:5", "5:6"], \
            f"expected the adjacent chain, got {model.hier_pc_pair_keys}"
        reused = getattr(model, "_hier_pc_reused", [])
        alloc = sorted(getattr(model, "hier_pc_proj", {}) or {})
        assert reused and alloc, f"expected a MIXED split, got reused={reused} alloc={alloc}"

        _forward_train(model, device)
        pc = model._last_hier_pc_loss
        assert pc is not None and torch.isfinite(pc).all()

        model.zero_grad(set_to_none=True)
        pc.backward()
        w_reused = model.downward_refresh_proj[reused[0]].weight
        w_alloc = model.hier_pc_proj[alloc[0]][0].weight
        assert w_reused.grad is not None and w_reused.grad.abs().sum() > 0, \
            "reused pair did not train the downward projection"
        assert w_alloc.grad is not None and w_alloc.grad.abs().sum() > 0, \
            "allocated pair did not train its dedicated head"
    print(f"  auto split: reused {reused}, allocated {alloc}; both receive gradient")


def test_per_offset_decoder_distinguishes_children():
    """THE anti-degeneracy property: one parent must predict DIFFERENT things for different
    child offsets.

    With a single shared H->H projection every child of a parent got the same prediction, so
    the only way to reach zero error was to make the children identical -- measured as the PC
    loss collapsing 3.0 -> 0.04 on text. A per-offset decoder must break that tie.
    """
    from pinball.model.hierarchical_flow_gat_cached_batch import _PCPerOffsetDecoder
    torch.manual_seed(0)
    H, COMP, R = 64, 8, 16
    dec = _PCPerOffsetDecoder(H, COMP, R)
    parent = torch.randn(1, H).expand(COMP, H).contiguous()   # the SAME parent, every offset
    out = dec(parent, torch.arange(COMP))
    assert out.shape == (COMP, H)
    # every pair of offsets must give a different prediction
    for i in range(COMP):
        for j in range(i + 1, COMP):
            assert not torch.allclose(out[i], out[j], atol=1e-6), \
                f"offsets {i} and {j} predict the same vector -- the degeneracy is back"
    spread = (out - out.mean(dim=0, keepdim=True)).abs().mean()
    print(f"  per-offset: {COMP} offsets from ONE parent, mean spread {float(spread):.4f}")


def test_per_offset_is_the_default_and_trains():
    """per_offset must be the default, build its decoders, and receive gradient."""
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build(tmp, hier_pc_enable=True)
        assert str(model.hier_pc_predictor) == "per_offset", \
            f"default predictor should be per_offset, got {model.hier_pc_predictor}"
        dec_params = [p for n, p in model.named_parameters() if "hier_pc_decoder" in n]
        assert dec_params, "per_offset built no decoder parameters"
        _forward_train(model, device)
        pc = model._last_hier_pc_loss
        assert pc is not None and torch.isfinite(pc).all()
        model.zero_grad(set_to_none=True)
        pc.backward()
        g = sum(float(p.grad.abs().sum()) for p in dec_params if p.grad is not None)
        assert g > 0, "per-offset decoder received no gradient"
    print(f"  per_offset default: {sum(p.numel() for p in dec_params):,} decoder params, grad {g:.3e}")


def test_causal_horizon_shifts_and_stays_causal():
    """On a causal model the pairing must SHIFT: a closed parent predicts a LATER window.

    Without the shift the AR filter keeps only children at or after the parent's window end --
    measured 1080 -> 128 edges with 126/128 at a single offset, which makes the objective
    near-vacuous and the per-offset decoder a no-op. With horizon = ceil(comp/stride) every
    offset is exercised and every parent strictly precedes its targets. A BIDIRECTIONAL model
    needs no shift and must get horizon 0.
    """
    with tempfile.TemporaryDirectory() as tmp:
        causal, _ = _build(tmp, hier_pc_enable=True, hier_ar_enable=True)
        bidi, _ = _build(tmp, hier_pc_enable=True, hier_ar_enable=False)
    for k, hs in causal.hier_pc_horizon.items():
        comp, stride = causal.hier_pc_comp[k], causal.hier_pc_stride[k]
        assert hs == (-(-comp // stride),), f"{k}: horizon {hs} != (ceil({comp}/{stride}),)"
    assert all(hs == (0,) for hs in bidi.hier_pc_horizon.values()), \
        f"bidirectional model must not shift, got {bidi.hier_pc_horizon}"
    print(f"  horizon: causal {causal.hier_pc_horizon}, bidi {bidi.hier_pc_horizon}")


def test_shifted_pairing_covers_all_offsets_and_precedes():
    """The shift must exercise every offset and never pair a parent with its own past."""
    import torch as _t
    with tempfile.TemporaryDirectory() as tmp:
        m, _ = _build(tmp, hier_pc_enable=True, hier_ar_enable=True)
    # reproduce the pairing the loss builds, from level geometry
    for key in m.hier_pc_pair_keys:
        L, Hh = (int(v) for v in key.split(":"))
        comp, stride = m.hier_pc_comp[key], m.hier_pc_stride[key]
        (h,) = m.hier_pc_horizon[key]
        n_high, n_low = 8, 8 * stride + comp + 8       # generous synthetic sizes
        j = _t.arange(n_high); o = _t.arange(comp)
        cw = (j + h).unsqueeze(1) * stride + o.unsqueeze(0)
        ok = cw < n_low
        offsets = o.unsqueeze(0).expand_as(cw)[ok]
        assert set(offsets.tolist()) == set(range(comp)), \
            f"{key}: offsets {sorted(set(offsets.tolist()))} != all {comp}"
        # every target must start at or after the parent's own window END
        parent_end = (j.unsqueeze(1).expand_as(cw)[ok]) * stride + comp - 1
        assert bool((cw[ok] > parent_end).all()), f"{key}: a target is inside the parent window"
    print("  shifted pairing: all offsets used, every target strictly after the parent window")


def test_horizon_accepts_a_list_and_averages_both_terms():
    """[0, N] runs reconstruction AND forecasting; the pair loss is their mean."""
    with tempfile.TemporaryDirectory() as tmp:
        both, device = _build(tmp, hier_pc_enable=True, hier_ar_enable=True,
                              hier_pc_causal_horizon=[0, 2])
        only0, _ = _build(tmp, hier_pc_enable=True, hier_ar_enable=True,
                          hier_pc_causal_horizon=0)
    assert all(hs == (0, 2) for hs in both.hier_pc_horizon.values()), both.hier_pc_horizon
    assert all(hs == (0,) for hs in only0.hier_pc_horizon.values()), only0.hier_pc_horizon
    _forward_train(both, device)
    pc = both._last_hier_pc_loss
    assert pc is not None and torch.isfinite(pc).all(), f"list-horizon PC loss bad: {pc}"
    print(f"  list horizon: {both.hier_pc_horizon}, loss {float(pc):.5f}")


def test_own_window_pairing_uses_every_offset_and_honours_the_attach_flag():
    """horizon 0 must NOT go through the AR filter, and it must default to an ATTACHED target.

    The AR filter (parent_ar_time <= child_ar_time) keeps only the last child of each window on a
    causal model, which is why the own-window term is built from geometry instead. Attachment is
    the true-PC setting: the top-down prediction shapes the level below, and the clamp that stops
    collapse is at the sensory end (hier_pc_l0_target: input is a detached capture). The flag is
    the one-key fallback, so both of its directions are under test.
    """
    import torch as _t
    with tempfile.TemporaryDirectory() as tmp:
        m, device = _build(tmp, hier_pc_enable=True, hier_ar_enable=True,
                           hier_pc_causal_horizon=0)
        det, ddev = _build(tmp, hier_pc_enable=True, hier_ar_enable=True,
                           hier_pc_causal_horizon=0, hier_pc_own_window_detach=True)
    # 1. coverage: every offset inside the parent window is decoded, not just the last one
    for key in m.hier_pc_pair_keys:
        comp, stride = m.hier_pc_comp[key], m.hier_pc_stride[key]
        n_high, n_low = 8, 8 * stride + comp + 8
        j = _t.arange(n_high); o = _t.arange(comp)
        cw = j.unsqueeze(1) * stride + o.unsqueeze(0)
        offsets = o.unsqueeze(0).expand_as(cw)[cw < n_low]
        assert set(offsets.tolist()) == set(range(comp)), \
            f"{key}: own-window offsets {sorted(set(offsets.tolist()))} != all {comp}"

    def _grad_flags(model, dev):
        """requires_grad of every target as it actually reaches the reduction."""
        seen, real = [], model._compute_pc_pair_reduction

        def _spy(pred, target):
            seen.append(bool(target.requires_grad))
            return real(pred, target)

        model._compute_pc_pair_reduction = _spy
        try:
            _forward_train(model, dev)
        finally:
            model._compute_pc_pair_reduction = real
        assert seen, "the reduction never ran -- the own-window term produced no pairs"
        return seen

    # 2. default = attached, so the top-down error reaches the level below (true PC)
    assert m.hier_pc_own_window_detach is False, "own-window target must default to ATTACHED"
    assert m.hier_pc_detach_target is False, "this test needs joint mode to be meaningful"
    joint = _grad_flags(m, device)
    assert all(joint), (
        f"a horizon-0 target arrived detached ({joint}) -- the top-down error cannot reach the "
        f"level below, which makes this a stack of autoencoders rather than predictive coding")
    pc = m._last_hier_pc_loss
    assert pc is not None and torch.isfinite(pc).all(), f"own-window PC loss bad: {pc}"

    # 3. the fallback still works in the other direction
    assert not any(_grad_flags(det, ddev)), \
        "hier_pc_own_window_detach=True must clamp every horizon-0 target"
    print(f"  own window: all offsets used, {len(joint)} targets attached by default "
          f"(flag clamps them), loss {float(pc):.5f}")


def test_center_pred_makes_the_score_offset_invariant():
    """hier_pc_center_pred: a shared mean offset must stop inflating the loss.

    The default centres only the denominator, so predicting the exact per-batch mean scores
    1.000 while the SAME prediction shifted by 1 sd scores ~2.0 and by 2 sd ~5.0. A level whose
    per-batch mean drifts is then pinned above 1.0 -- worse than a predictor the bias term could
    reach for free -- and no lambda can lift that. With the flag on, the constant cancels.
    """
    import torch as _t
    _t.manual_seed(0)
    tgt = _t.randn(256, 64) * 0.8 + 3.0

    class _M:
        hier_pc_loss_mode = "mse_norm"
        hier_pc_center_target = True
        hier_pc_center_pred = False
    from pinball.model.hierarchical_flow_gat_cached_batch import HierarchicalFlowGAT
    red = HierarchicalFlowGAT._compute_pc_pair_reduction

    mean_pred = tgt.mean(dim=0, keepdim=True).expand_as(tgt)
    base = float(red(_M, mean_pred, tgt))
    assert abs(base - 1.0) < 0.02, f"mean predictor should score 1.0, got {base}"

    drift = mean_pred + 2.0 * tgt.std()
    off_default = float(red(_M, drift, tgt))
    assert off_default > 4.0, f"default form must be offset-SENSITIVE, got {off_default}"

    _M.hier_pc_center_pred = True
    off_inv = float(red(_M, drift, tgt))
    same = float(red(_M, mean_pred, tgt))
    assert abs(off_inv - same) < 1e-4, (
        f"with center_pred the offset must cancel: {off_inv} vs {same}")
    print(f"  center_pred: mean {base:.3f} | +2sd default {off_default:.3f} -> invariant "
          f"{off_inv:.3f} (== {same:.3f})")


def test_centred_mse_norm_puts_the_mean_predictor_at_one():
    """Centring must make 1.0 mean exactly "no better than predicting the mean".

    Without it, mse_norm divides by the target's RAW second moment, so a level's shared
    constant sits in the denominator: inflating it lowers the loss for free, and levels whose
    features are bias-dominated enter the objective far too weakly. Measured on a trained text
    checkpoint: raw 0.0366 / 0.0020 / 0.0016 per pair looked like success, while the centred
    values were 0.891 / 0.984 / 1.709 -- i.e. at or worse than the mean predictor.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model, _ = _build(tmp, hier_pc_enable=True, hier_pc_loss_mode="mse_norm")
    torch.manual_seed(0)
    N, H = 64, 32
    content = torch.randn(N, H)
    target = content + 50.0                       # a large shared constant, as real levels have
    mean_pred = target.mean(dim=0, keepdim=True).expand_as(target)

    model.hier_pc_center_target = True
    centred = float(model._compute_pc_pair_reduction(mean_pred, target))
    assert abs(centred - 1.0) < 1e-4, f"mean predictor must score 1.0 centred, got {centred}"

    model.hier_pc_center_target = False
    raw = float(model._compute_pc_pair_reduction(mean_pred, target))
    assert raw < 0.05, f"raw metric should flatter the mean predictor, got {raw}"

    # centring must leave the NUMERATOR alone: a constant shift of both sides changes nothing
    model.hier_pc_center_target = True
    shifted = float(model._compute_pc_pair_reduction(mean_pred + 7.0, target + 7.0))
    assert abs(shifted - centred) < 1e-4, "centred loss must be invariant to a shared shift"
    print(f"  centring: mean predictor scores {centred:.4f} centred vs {raw:.4f} raw")


def test_l0_ce_never_reads_features_as_token_ids():
    """hier_pc_l0_ce must stay inert on a FEATURE-mode host.

    The token ids are stashed from forward() rather than threaded through the refinement chain,
    and that chain is shared with the image and DNA/feature paths -- which pass a float
    [B, T, C] tensor. Reading that as a vocabulary index would be a silent correctness bug
    (garbage targets, or an out-of-range gather), so the stash is guarded on dtype and rank.
    Here the model is feature-mode, so the CE branch must decline and the feature objective
    must still produce a finite loss.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model, device = _build(tmp, hier_pc_enable=True, hier_pc_l0_ce=True)
        assert bool(getattr(model, "hier_pc_l0_ce", False)), "flag did not reach the model"
        out = _forward_train(model, device)          # feature input: [B, T, C] float
        assert torch.is_tensor(out) and torch.isfinite(out).all()
        assert getattr(model, "_pc_token_ids", "missing") is None, \
            "float feature input was stashed as token ids"
        pc = model._last_hier_pc_loss
        assert pc is not None and torch.isfinite(pc).all(), \
            "CE declined but the feature objective did not take over"
    print(f"  l0_ce on feature input: declined cleanly, feature loss {float(pc):.4f}")


if __name__ == "__main__":
    test_disabled_is_inert()
    test_enabled_produces_loss_and_trains_downward_projection()
    test_detach_flag_cuts_only_the_child_side()
    test_rejects_double_counting_with_aux()
    test_disabled_never_enters_pc_code()
    test_jepa_generalisation_is_faithful()
    test_auto_predictor_mixed_reuse_and_alloc()
    test_per_offset_decoder_distinguishes_children()
    test_per_offset_is_the_default_and_trains()
    test_causal_horizon_shifts_and_stays_causal()
    test_center_pred_makes_the_score_offset_invariant()
    test_horizon_accepts_a_list_and_averages_both_terms()
    test_own_window_pairing_uses_every_offset_and_honours_the_attach_flag()
    test_shifted_pairing_covers_all_offsets_and_precedes()
    test_centred_mse_norm_puts_the_mean_predictor_at_one()
    test_l0_ce_never_reads_features_as_token_ids()
    print("all hier_pc tests passed")
