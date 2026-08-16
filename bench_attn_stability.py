#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
"""Attention/residual stability probe for the packed cross-level path.

Answers, for one forward pass, the questions you cannot get from a loss curve:

  * where does softmax mass go?   attention mass per (query level -> source level)
  * are logits growing?           max |q.k/sqrt(d)| and RMS(Q), RMS(K) per layer
  * is attention collapsing?      entropy relative to a uniform draw over visible keys
  * is the residual stream drifting?  RMS of x per level, per layer

Run it on a checkpoint from before the instability and one from after; the pair of
reports is the diagnostic. A level whose mass goes from a healthy share to ~1.0, or a
residual RMS that climbs with depth run-over-run, is the answer.

    python bench_attn_stability.py --config configs/pinball_dna_bidi_full.yaml
    python bench_attn_stability.py --config ... --ckpt path/to/step_N.pt --device cuda:1

Attention is recomputed densely from the captured post-RoPE Q/K, because the flash and
flex kernels never materialize the probability matrix. That costs [B,H,N,N] floats for
one layer at a time (~100 MB at B=1, N=1248, H=16), so keep --batch small.

The flex-union arm routes through _flex_union_attn and is NOT captured here; the probe
reports what it saw so a silent empty result cannot be mistaken for a healthy one.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys

import torch

sys.path.insert(0, str(__file__.rsplit("/", 1)[0] + "/src"))

from pinball.instantiate_PINBALL_model import build_pinball  # noqa: E402
from pinball.model.layers.hierarchical_message_passing import (  # noqa: E402
    HierarchicalMessagePassing,
    HierarchicalTransformerLayer,
)

LEVEL_NAMES = ["L0", "L1", "L2", "L3"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/pinball_dna_bidi_full.yaml")
    p.add_argument("--ckpt", default=None,
                   help="optional checkpoint; loaded with strict=False so a host-model "
                        "state dict with a 'pinball.' prefix still lands")
    p.add_argument("--ckpt-prefix", default="pinball.",
                   help="strip this prefix off checkpoint keys (host models nest pinball)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=768, help="num_tracks / feature width")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=None, help="defaults to the config block_size")
    p.add_argument("--input", default=None, help=".pt file with a real [B,L,width] batch")
    return p.parse_args()


def load_checkpoint(model, path, prefix):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
            break
    if prefix:
        stripped = {k[len(prefix):]: v for k, v in obj.items() if k.startswith(prefix)}
        if stripped:
            obj = stripped
    missing, unexpected = model.load_state_dict(obj, strict=False)
    loaded = len(obj) - len(unexpected)
    print(f"checkpoint: {loaded}/{len(model.state_dict())} tensors matched "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")
    if loaded == 0:
        print("  WARNING: nothing matched -- wrong --ckpt-prefix? Report below is at INIT.")


def install_probes(model, store):
    """Capture post-RoPE Q/K for the packed calls, and the residual stream per layer."""
    raw_attn = HierarchicalMessagePassing._compute_local_attn_from_qkv
    raw_layer = HierarchicalTransformerLayer.forward

    def attn_probe(self, q_lvl, k_lvl, v_lvl, window, causal, backend, level,
                   attn_bias=None, force_sdpa=False, skip_out_proj=False):
        if level in (-1, -2):
            store["attn"].append({
                "layer": store["layer_idx"], "level": level,
                # detach + clone: the caller reuses these buffers
                "q": q_lvl.detach().float().cpu(), "k": k_lvl.detach().float().cpu(),
                "window": int(window), "causal": bool(causal),
                "levels": (self._local_pack_spec or {}).get(
                    "levels" if level == -1 else "lane_levels", None),
            })
        return raw_attn(self, q_lvl, k_lvl, v_lvl, window, causal, backend, level,
                        attn_bias=attn_bias, force_sdpa=force_sdpa,
                        skip_out_proj=skip_out_proj)

    def layer_probe(self, *a, **kw):
        # The refinement loop calls this positionally with up to 8 args; take x/node_level
        # from whichever side they arrive on.
        x = a[0] if a else kw["x"]
        node_level = a[2] if len(a) > 2 else kw["node_level"]
        idx = store["layer_idx"]
        out = raw_layer(self, *a, **kw)
        x_out = out[0] if isinstance(out, tuple) else out
        nl = node_level.to(x_out.device)
        store["resid"].append({
            "layer": idx,
            "in": [float(x[..., nl == l, :].float().pow(2).mean().sqrt())
                   if int((nl == l).sum()) else float("nan") for l in range(4)],
            "out": [float(x_out[..., nl == l, :].float().pow(2).mean().sqrt())
                    if int((nl == l).sum()) else float("nan") for l in range(4)],
        })
        store["layer_idx"] = idx + 1
        return out

    HierarchicalMessagePassing._compute_local_attn_from_qkv = attn_probe
    HierarchicalTransformerLayer.forward = layer_probe
    return lambda: (setattr(HierarchicalMessagePassing,
                            "_compute_local_attn_from_qkv", raw_attn),
                    setattr(HierarchicalTransformerLayer, "forward", raw_layer))


def analyse(rec):
    """Dense recompute of one captured attention call -> per-level mass, entropy, logits."""
    q, k = rec["q"], rec["k"]                      # [B, N, H, D]
    B, N, H, D = q.shape
    lv = rec["levels"]
    lv = lv.cpu() if lv is not None else torch.zeros(N, dtype=torch.long)
    logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(D)

    pos = torch.arange(N)
    if rec["causal"]:
        vis = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] <= rec["window"])
    else:
        vis = (pos[:, None] - pos[None, :]).abs() <= rec["window"]
    logits = logits.masked_fill(~vis[None, None], float("-inf"))

    attn = torch.softmax(logits, dim=-1)
    n_vis = vis.sum(-1).clamp_min(1)                                     # [N]
    ent = -(attn.clamp_min(1e-9).log() * attn).sum(-1)                   # [B,H,N]
    ent_rel = (ent / n_vis.float().log().clamp_min(1e-9)).mean(dim=(0, 1))  # [N], 1.0=uniform

    mass = torch.zeros(4, 4)     # [query level, source level]
    for ql in range(4):
        qsel = lv == ql
        if not bool(qsel.any()):
            mass[ql] = float("nan")
            continue
        a = attn[:, :, qsel, :].mean(dim=(0, 1, 2))                      # [N] over sources
        for sl in range(4):
            mass[ql, sl] = float(a[lv == sl].sum())

    ent_by_q = torch.tensor([float(ent_rel[lv == l].mean()) if bool((lv == l).any())
                             else float("nan") for l in range(4)])
    finite = logits[torch.isfinite(logits)]
    return {
        "mass": mass, "ent": ent_by_q,
        "max_logit": float(finite.abs().max()),
        "q_rms": float(q.pow(2).mean().sqrt()), "k_rms": float(k.pow(2).mean().sqrt()),
    }


def main():
    args = parse_args()
    logging.getLogger("pinball").setLevel(logging.ERROR)

    model, _, cfg, _ = build_pinball(
        cfg_path=args.config, num_tracks=args.width, tie_weights=False,
        device=args.device, set_global_seed=False,
        override={"hier_layer_compile": False, "hier_refresh_compile": False,
                  "use_gradient_checkpointing": False})
    if args.ckpt:
        load_checkpoint(model, args.ckpt, args.ckpt_prefix)
    model.eval()

    seq = args.seq or int(getattr(cfg, "block_size", 1024))
    if args.input:
        x = torch.load(args.input, map_location=args.device).float()
    else:
        torch.manual_seed(0)
        x = torch.randn(args.batch, seq, args.width, device=args.device)
        print("NOTE: random input. Attention MASS and residual RMS are still meaningful "
              "(they are properties of the weights), but feed --input a real batch to "
              "read entropy and max-logit as they occur in training.")

    store = {"attn": [], "resid": [], "layer_idx": 0}
    restore = install_probes(model, store)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
            model(x)
    finally:
        restore()

    mixed = [r for r in store["attn"] if r["level"] == -1]
    lane = [r for r in store["attn"] if r["level"] == -2]
    print(f"\ncaptured: {len(mixed)} mixed-window calls, {len(lane)} coarse-lane calls, "
          f"{len(store['resid'])} layers")
    if not mixed:
        print("NO packed attention captured. The flex-union arm bypasses this hook, and a "
              "config with local_pack_cross_level off has no packed path at all. Nothing "
              "below would be meaningful; stopping.")
        return

    print("\n=== ATTENTION MASS  (row = query level, col = source level, per layer) ===")
    print("    a row that goes to ~1.0 in one column is the collapse to look for\n")
    hdr = "  ".join(f"{n:>6s}" for n in LEVEL_NAMES)
    for rec in mixed:
        st = analyse(rec)
        print(f"-- layer {rec['layer']:2d}   max|logit|={st['max_logit']:8.2f}   "
              f"RMS(Q)={st['q_rms']:.3f} RMS(K)={st['k_rms']:.3f}")
        print(f"     {'':4s}{hdr}      entropy(rel)")
        for ql in range(4):
            row = "  ".join(f"{float(st['mass'][ql, sl]):6.3f}" for sl in range(4))
            print(f"     {LEVEL_NAMES[ql]:4s}{row}      {float(st['ent'][ql]):.3f}")

    print("\n=== RESIDUAL STREAM RMS  (per level, layer input -> layer output) ===")
    print("    growth across depth that compounds run-over-run is suspect #3\n")
    print(f"  {'layer':>5s}  " + "  ".join(f"{n:>14s}" for n in LEVEL_NAMES))
    for r in store["resid"]:
        cells = "  ".join(f"{r['in'][l]:6.2f}->{r['out'][l]:6.2f}" for l in range(4))
        print(f"  {r['layer']:>5d}  {cells}")

    first, last = store["resid"][0], store["resid"][-1]
    print("\n  depth growth (layer 0 input -> last layer output):")
    for l in range(4):
        a, b = first["in"][l], last["out"][l]
        if a == a and b == b and a > 0:
            print(f"    {LEVEL_NAMES[l]}: {a:6.2f} -> {b:7.2f}   ({b / a:5.2f}x)")


if __name__ == "__main__":
    main()
