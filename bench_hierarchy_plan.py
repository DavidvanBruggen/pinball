#!/usr/bin/env python
"""Size a Pinball hierarchy and cost its packed attention, without building a model.

  # audit a config as written
  python bench_hierarchy_plan.py --config configs/pinball_dna_bidi_full.yaml

  # what happens if I scale this config out?
  python bench_hierarchy_plan.py --config configs/pinball_dna_bidi_windowed.yaml \
      --scan 4096 16384 65536 262144 1048576

  # design a deep hierarchy: auto-extend levels until the top level fits sqrt(N)
  python bench_hierarchy_plan.py --config configs/pinball_dna_bidi_windowed.yaml \
      --scan 4096 65536 1048576 --balance --coarse-window 64

Reports query-key PAIR counts, which is what scales. Wallclock adds constants this cannot
know (kernel choice, head count, dropout path), so use bench_coarse_window.py for timings.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pinball.model.hierarchy_plan import (  # noqa: E402
    level_sizes, min_coarse_levels, plan_hierarchy,
)


def load_cfg(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def balance(n, comps, overs, extra_comp, extra_overlap):
    """Append levels until the top level fits inside sqrt(N)."""
    comps, overs = list(comps), list(overs)
    need = min_coarse_levels(n, comps, overs)
    guard = 0
    while need > 0 and len(comps) < need and guard < 64:
        comps.append(extra_comp)
        overs.append(extra_overlap)
        need = min_coarse_levels(n, comps, overs)
        guard += 1
    return comps, overs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scan", type=int, nargs="*", default=None,
                    help="sequence lengths to scan (default: the config's block_size)")
    ap.add_argument("--balance", action="store_true",
                    help="append levels until the top level fits sqrt(N)")
    ap.add_argument("--extra-comp", type=int, default=4,
                    help="compression for appended levels (4 @ overlap 0.5 = stride 2)")
    ap.add_argument("--extra-overlap", type=float, default=0.5)
    ap.add_argument("--coarse-window", type=str, default=None,
                    help="override local_pack_coarse_window: int, or comma list per coarse "
                         "level where 0 = global (e.g. '64,64,64,0')")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    comps = cfg.get("compression_ratios", [128, 16, 8])
    overs = cfg.get("overlap_ratios", [0.5] * len(comps))
    win = int(cfg.get("local_pack_window", 0) or 0)
    lane_on = bool(cfg.get("local_pack_coarse_lane", False))
    cw = cfg.get("local_pack_coarse_window", None) if lane_on else None
    if args.coarse_window is not None:
        cw = ([int(v) for v in args.coarse_window.split(",")]
              if "," in args.coarse_window else int(args.coarse_window))
    lengths = args.scan or [int(cfg.get("block_size", 1024))]

    print(f"config           : {args.config}")
    print(f"compression      : {comps}")
    print(f"overlap          : {overs}")
    print(f"local_pack_window: {win}   coarse_window: {cw}"
          f"{'' if lane_on else '   (coarse lane OFF)'}")
    if args.balance:
        print(f"balance          : append comp={args.extra_comp} @ overlap="
              f"{args.extra_overlap} until n_top <= sqrt(N)")
    print()
    hdr = (f"{'N':>9} {'lvls':>5} {'n_top':>7} {'sqrt(N)':>8} {'bank%':>6} "
           f"{'pairs':>11} {'/token':>7} {'full':>12} {'x cheaper':>10}  global top")
    print(hdr)
    print("-" * len(hdr))

    seen = []
    for n in lengths:
        c, o = (balance(n, comps, overs, args.extra_comp, args.extra_overlap)
                if args.balance else (list(comps), list(overs)))
        w = cw
        if args.balance and isinstance(cw, list):
            # keep the last entry (global top) pinned to the actual top level
            body = [v for v in cw[:-1]] or [64]
            w = [body[min(i, len(body) - 1)] for i in range(len(c) - 1)] + [cw[-1]]
        p = plan_hierarchy(n, c, o, win, w)
        print(f"{n:>9} {len(c):>5} {p.n_top:>7} {p.sqrt_n:>8.0f} "
              f"{100.0 * p.coarse_bank / n:>5.1f}% {p.total_pairs / 1e6:>10.2f}M "
              f"{p.pairs_per_token:>7.0f} {p.full_pairs / 1e6:>11.2f}M "
              f"{p.full_pairs / max(1, p.total_pairs):>9.0f}x  "
              f"{'ok' if p.global_top_affordable else 'TOO COSTLY'}")
        seen.append((n, c, o, p))

    print()
    for n, c, o, p in seen:
        if p.warnings:
            print(f"N={n}:")
            for w in p.warnings:
                print(f"  WARN {w}")
            if not p.global_top_affordable:
                need = p.min_levels_needed
                cc = list(c) + [args.extra_comp] * max(0, need - len(c))
                oo = list(o) + [args.extra_overlap] * max(0, need - len(o))
                print(f"  FIX  compression_ratios: {cc}")
                print(f"       overlap_ratios: {oo}")
                print(f"       num_layers: {[0] * (len(cc) + 1)}   "
                      f"# length = levels; sizes -> {level_sizes(n, cc, oo)[1:]}")

    # pairs/token drift is the linear-scaling test: flat = linear.
    if len(seen) > 1:
        first, last = seen[0][3], seen[-1][3]
        growth = last.pairs_per_token / max(1e-9, first.pairs_per_token)
        span = last.n_tokens / max(1, first.n_tokens)
        print(f"\ncost of the config AS WRITTEN: pairs/token "
              f"{first.pairs_per_token:.0f} -> {last.pairs_per_token:.0f} ({growth:.2f}x) "
              f"across a {span:.0f}x range in N")
        if not last.global_top_affordable:
            print("  NOTE flat cost here does NOT mean global reach: the lane window "
                  "truncates the\n       coarse bank, which is cheap precisely because it "
                  "gave up all-to-all coarse.\n       Re-run with --balance to see a "
                  "hierarchy that affords a global top level.")


if __name__ == "__main__":
    main()
