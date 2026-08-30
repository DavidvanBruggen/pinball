# SPDX-License-Identifier: GPL-3.0-or-later
"""Correctness of the segmented backward used by the per-layer downward refresh.

The refresh gathers, for every fine row, one row of a coarser level. Written as
`index_select`, its backward lowers to `aten::index_add`: 4096 L0 rows scattering into an
8-node top level is a 512-way atomic collision, measured at 5 ms per call and 54 of the
157 ms train step of the 8-level text arm.

`chosen` is monotone non-decreasing on that path (each coarse node owns a contiguous run
of fine rows -- true for both the AR "most recent closed" gather and the bidi
containing-parent gather, and for curve mode, where node order is the curve order), so the
same reduction is a segment sum. This test pins the two things that make the swap safe:

  1. `_SegmentedBroadcast` is EXACTLY index_select/index_add in exact arithmetic (fp64),
     including ragged runs, empty segments and trailing destination rows nothing selects.
  2. `_segment_offsets` returns None whenever monotonicity does not hold, so a layout that
     breaks the assumption silently falls back to the scatter instead of computing garbage.

fp64 on CPU on purpose: the equality is exact there. Under bf16/CUDA the scatter is an
atomic accumulation that is not even reproducible against itself.

Run:   python tests/test_segmented_broadcast.py
"""
import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pinball.model.hierarchical_flow_gat_cached_batch import (
    _HAS_SEGMENT_REDUCE,
    _SegmentedBroadcast,
    _segment_offsets,
)


def _assert_matches_index_select(chosen, n_src, batch=1, hidden=5):
    """Forward and grad must equal the index_select/index_add pair exactly in fp64."""
    chosen = chosen.long()
    offsets = _segment_offsets(chosen, n_src)
    assert offsets is not None, "monotone index should take the segmented path"
    assert int(offsets.numel()) == n_src + 1
    assert int(offsets[0]) == 0 and int(offsets[-1]) == chosen.numel()

    src = torch.randn(batch, n_src, hidden, dtype=torch.float64, requires_grad=True)
    seg_src = src.detach().clone().requires_grad_(True)
    grad_out = torch.randn(batch, chosen.numel(), hidden, dtype=torch.float64)

    ref = src.index_select(1, chosen)
    ref.backward(grad_out)
    out = _SegmentedBroadcast.apply(seg_src, chosen, offsets)
    out.backward(grad_out)

    assert torch.equal(out, ref.detach()), "forward diverged from index_select"
    assert torch.allclose(seg_src.grad, src.grad, rtol=0.0, atol=1e-12), (
        f"grad diverged: max |d| = {(seg_src.grad - src.grad).abs().max().item():.3e}"
    )


def test_uniform_runs():
    """The regular case: every coarse node owns an equal run of fine rows."""
    _assert_matches_index_select(torch.arange(4096) // 512, 8)


def test_ragged_runs():
    """Real 0:7 plan shape at 4096: run lengths 1, 8, 512 and a 2039-row tail."""
    lengths = torch.tensor([1, 8, 512, 2039])
    _assert_matches_index_select(torch.repeat_interleave(torch.arange(4), lengths), 4)


def test_long_tail_run():
    """Once the last coarse node closes, every remaining fine row reads it -- the worst
    collision the scatter path sees, and the reason this change exists."""
    _assert_matches_index_select(torch.clamp(torch.arange(4096) // 256 - 1, 0, 7), 8)


def test_all_rows_select_one_destination():
    _assert_matches_index_select(torch.zeros(4096), 8)


def test_trailing_destination_rows_unselected():
    """Coarse rows past the last selected one get no gradient, but must still be shaped
    into the output -- segment_reduce only emits rows for segments it was given."""
    _assert_matches_index_select(torch.arange(100) // 25, 8)


def test_empty_leading_segment():
    """Destination row 0 selected by nothing (zero-length leading segment)."""
    _assert_matches_index_select(torch.clamp(torch.arange(64) // 8, 1, 7), 8)


def test_single_row():
    _assert_matches_index_select(torch.zeros(1), 4)


def test_batched():
    """B > 1 folds the batch into the feature axis; check it unfolds to the right rows."""
    _assert_matches_index_select(torch.arange(4096) // 512, 8, batch=4)
    lengths = torch.tensor([3, 1, 100, 20])
    _assert_matches_index_select(torch.repeat_interleave(torch.arange(4), lengths), 6, batch=3)


def test_non_monotone_falls_back():
    """A non-contiguous assignment must return None so the caller keeps the scatter."""
    assert _segment_offsets(torch.tensor([3, 1, 2, 0, 5]), 6) is None
    assert _segment_offsets(torch.tensor([0, 1, 2, 1]), 4) is None
    assert _segment_offsets(torch.empty(0, dtype=torch.long), 4) is None


def test_index_beyond_declared_destination_falls_back():
    """A stale index wider than the destination would silently drop rows -- stay on the
    scatter (which raises) rather than returning a wrong-shaped gradient."""
    assert _segment_offsets(torch.tensor([0, 1, 2, 9]), 4) is None


def test_double_backward_fails_by_name():
    """Single-differentiable by design. Nothing in pinball differentiates a gradient here,
    but if that changes it must raise, not silently return a detached grad."""
    chosen = (torch.arange(64) // 8).long()
    offsets = _segment_offsets(chosen, 8)
    src = torch.randn(1, 8, 3, dtype=torch.float64, requires_grad=True)
    out = _SegmentedBroadcast.apply(src, chosen, offsets)
    grad, = torch.autograd.grad(out.sum(), src, create_graph=True)
    # once_differentiable detaches the backward's output, so the second order raises
    # instead of quietly producing a zero or a wrong gradient.
    assert grad.grad_fn is None, "backward should not be graph-connected"
    try:
        torch.autograd.grad(grad.pow(2).sum(), src)
    except RuntimeError:
        return
    raise AssertionError("expected grad-of-grad to raise")


if __name__ == "__main__":
    if not _HAS_SEGMENT_REDUCE:
        print("torch.segment_reduce unavailable; the refresh stays on the scatter path")
        raise SystemExit(0)
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
            passed += 1
    print(f"\n{passed} passed")
