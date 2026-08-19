"""Hierarchy sizing and attention-cost planner.

Answers, for a given (sequence length, compression_ratios, overlap_ratios, window) tuple:
how big is every level, what does the packed attention cost, and is that cost linear or
quadratic in N.

WHY THIS EXISTS. The coarse bank is a fixed *fraction* of N, not a fixed size -- with
compression_ratios [16,4,4] at overlap 0.5 the strides are [8,2,2], so the bank is
N/8 + N/16 + N/32 = 0.219*N. An all-to-all attention over that bank therefore costs
0.0479*N^2: 31x cheaper than full attention over the pack, but the SAME asymptotics. The
hierarchy buys a constant, not a scaling law. What buys the scaling law is:

    give every level a fixed window (linear), and make the TOP level all-to-all. The top
    term costs n_top^2, which stays inside the O(N) budget iff n_top <= sqrt(N), i.e. iff
    the total compression is at least sqrt(N).

That threshold depends entirely on the compression/overlap settings and must never be
hardcoded. Measured spread: [16,4,4]@0.5 needs 4 coarse levels at N=1M, comp 16 with no
overlap needs 3, comp 8 at overlap 0.75 needs 10.

The level-size rule below is copied from the model so the two cannot drift:
  _create_next_level      stride = max(1, int(comp * (1 - overlap)))
                          n_higher = max(1, (n_lower - 1) // stride + 1)
  _pooled_seed_indices    same stride rule
Note the max(1, ...): a level never disappears, it floors at a single node. A hierarchy
sized for a long sequence therefore degrades into 1-node global registers on short inputs
rather than breaking, which is what makes a fixed deep hierarchy safe to configure once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

__all__ = ["HierarchyPlan", "level_sizes", "plan_hierarchy", "min_coarse_levels", "format_plan"]


def _strides(compression_ratios: Sequence[float], overlap_ratios: Sequence[float]) -> List[int]:
    """Exactly _create_next_level's rule. Kept in one place so it cannot drift."""
    out: List[int] = []
    for i, comp in enumerate(compression_ratios):
        ov = float(overlap_ratios[i]) if i < len(overlap_ratios) else 0.0
        out.append(max(1, int(int(comp) * (1.0 - ov))))
    return out


def level_sizes(n_tokens: int, compression_ratios: Sequence[float],
                overlap_ratios: Sequence[float]) -> List[int]:
    """Node count per level, L0 first. Mirrors _create_next_level, including its max(1, ...)
    floor -- levels never vanish, they collapse to a single node."""
    sizes = [int(n_tokens)]
    n = int(n_tokens)
    for st in _strides(compression_ratios, overlap_ratios):
        n = max(1, (n - 1) // st + 1)
        sizes.append(n)
    return sizes


def min_coarse_levels(n_tokens: int, compression_ratios: Sequence[float],
                      overlap_ratios: Sequence[float], extra_stride: int = 2) -> int:
    """Smallest number of coarse levels whose cumulative compression reaches sqrt(N), i.e.
    the depth at which an all-to-all top level costs no more than the linear terms.

    Uses the configured strides as far as they go, then assumes `extra_stride` for any
    levels beyond them -- so the answer respects the user's actual compression/overlap
    rather than assuming a uniform ratio.
    """
    target = math.sqrt(max(1, int(n_tokens)))
    st = _strides(compression_ratios, overlap_ratios)
    cum, levels = 1, 0
    while cum < target:
        step = st[levels] if levels < len(st) else max(2, int(extra_stride))
        if step <= 1:                     # stride 1 never compresses; refuse to spin
            return -1
        cum *= step
        levels += 1
        if levels > 512:                  # pathological config guard
            return -1
    return levels


@dataclass
class HierarchyPlan:
    n_tokens: int
    strides: List[int]
    sizes: List[int]                      # L0 first
    window: int
    coarse_window: Optional[int]          # None/0 -> lane disabled
    coarse_bank: int
    pack: int
    n_top: int
    sqrt_n: float
    lane_all_to_all: bool
    mixed_pairs: int
    lane_pairs: int
    full_pairs: int                       # pack^2, for reference
    min_levels_needed: int
    levels_configured: int
    warnings: List[str] = field(default_factory=list)

    @property
    def total_pairs(self) -> int:
        return self.mixed_pairs + self.lane_pairs

    @property
    def pairs_per_token(self) -> float:
        return self.total_pairs / max(1, self.n_tokens)

    @property
    def global_top_affordable(self) -> bool:
        """True when an ALL-TO-ALL top level would cost no more than the linear terms, i.e.
        n_top <= sqrt(N).

        This is NOT the same as "this config is cheap". A config can have flat cost simply
        because its lane window truncates the coarse bank -- that is linear precisely
        BECAUSE it gave up global reach. This property asks the different, load-bearing
        question: can this hierarchy afford global attention at all?
        """
        return self.n_top <= self.sqrt_n

    @property
    def degenerate_levels(self) -> int:
        """Levels collapsed to a single node -- harmless (they act as global registers) but
        they cannot teach the model level-specific behaviour."""
        return sum(1 for s in self.sizes[1:] if s == 1)


def plan_hierarchy(
    n_tokens: int,
    compression_ratios: Sequence[float],
    overlap_ratios: Sequence[float],
    local_pack_window: int = 0,
    local_pack_coarse_window: Optional[Union[int, Sequence[int]]] = None,
    extra_stride: int = 2,
) -> HierarchyPlan:
    """Size the hierarchy and cost its packed attention.

    local_pack_coarse_window accepts the legacy scalar (one shared radius over the whole
    interleaved coarse bank) or a per-level list (radius in that level's OWN rank space,
    0/None meaning unrestricted). Costs are query-key pair counts, which is what actually
    scales -- wallclock adds constants this cannot know.
    """
    n_tokens = int(n_tokens)
    st = _strides(compression_ratios, overlap_ratios)
    sizes = level_sizes(n_tokens, compression_ratios, overlap_ratios)
    coarse = sizes[1:]
    bank = sum(coarse)
    pack = sum(sizes)
    w = int(local_pack_window or 0)
    span = min(2 * w + 1, pack) if w > 0 else pack

    mixed_pairs = pack * span

    # lane_pairs: total lane cost. quad_pairs: the part that grows as N^2, i.e. the pairs
    # contributed by levels whose reach covers their whole population. Under a per-level
    # window only the TOP level is unrestricted, and n_top^2 is exactly the term the sqrt(N)
    # rule is designed to keep affordable -- so it must NOT be charged as bank^2, which is
    # what a shared scalar window costs.
    lane_pairs = 0
    quad_pairs = 0
    lane_all = False
    if local_pack_coarse_window is None:
        cw_repr: Optional[int] = None
    elif isinstance(local_pack_coarse_window, (list, tuple)):
        cw_repr = None
        for i, n_l in enumerate(coarse):
            r = int(local_pack_coarse_window[i]) if i < len(local_pack_coarse_window) else 0
            reach_l = n_l if r <= 0 else min(2 * r + 1, n_l)
            lane_pairs += n_l * reach_l
            if reach_l >= n_l:
                quad_pairs += n_l * n_l
        lane_all = quad_pairs > 0
    else:
        cw_repr = int(local_pack_coarse_window)
        reach = bank if cw_repr <= 0 else min(2 * cw_repr + 1, bank)
        lane_pairs = bank * reach
        lane_all = reach >= bank
        quad_pairs = bank * bank if lane_all else 0

    n_top = coarse[-1] if coarse else n_tokens
    sqrt_n = math.sqrt(max(1, n_tokens))
    need = min_coarse_levels(n_tokens, compression_ratios, overlap_ratios, extra_stride)

    warns: List[str] = []
    if n_top > sqrt_n:
        short = max(0, need - len(coarse)) if need > 0 else 0
        warns.append(
            f"top level has {n_top} nodes > sqrt(N)={sqrt_n:.0f}: an all-to-all top level is "
            f"SUPERLINEAR at this depth. Need >= {need} coarse levels "
            f"({short} more than the {len(coarse)} configured)."
        )
    if quad_pairs > 4 * mixed_pairs:
        warns.append(
            f"unrestricted coarse attention costs {quad_pairs / 1e6:.2f}M pairs vs "
            f"{mixed_pairs / 1e6:.2f}M for the linear mixed window -- the N^2 term now "
            f"dominates. Give the coarse levels per-level windows and keep only the top global."
        )
    flat = [i + 1 for i, s in enumerate(st) if s <= 1]
    if flat:
        warns.append(
            f"level(s) {flat} have stride 1 and do not compress at all: "
            f"stride = max(1, int(comp*(1-overlap))), so e.g. comp=2 at overlap=0.5 gives "
            f"int(1.0)=1, not 2. Use comp=4 at overlap=0.5 for a stride-2 level."
        )
    deg = sum(1 for s in coarse if s == 1)
    if deg:
        warns.append(
            f"{deg} coarse level(s) collapsed to a single node at N={n_tokens}. Harmless "
            f"(they act as global registers) but they cannot train level-specific behaviour."
        )

    return HierarchyPlan(
        n_tokens=n_tokens, strides=st, sizes=sizes, window=w, coarse_window=cw_repr,
        coarse_bank=bank, pack=pack, n_top=n_top, sqrt_n=sqrt_n, lane_all_to_all=lane_all,
        mixed_pairs=mixed_pairs, lane_pairs=lane_pairs, full_pairs=pack * pack,
        min_levels_needed=need, levels_configured=len(coarse), warnings=warns,
    )


def format_plan(p: HierarchyPlan) -> str:
    """One-line summary for the build log."""
    return (
        f"hierarchy N={p.n_tokens} strides={p.strides} sizes={p.sizes} "
        f"coarse_bank={p.coarse_bank} ({100.0 * p.coarse_bank / max(1, p.n_tokens):.1f}% of N) "
        f"n_top={p.n_top} sqrt(N)={p.sqrt_n:.0f} "
        f"pairs={p.total_pairs / 1e6:.2f}M ({p.pairs_per_token:.0f}/token, "
        f"mixed {p.mixed_pairs / 1e6:.2f}M + lane {p.lane_pairs / 1e6:.2f}M) "
        f"vs full {p.full_pairs / 1e6:.2f}M -- "
        f"{'global-top affordable' if p.global_top_affordable else 'GLOBAL-TOP TOO COSTLY'}"
    )
