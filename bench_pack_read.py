# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
"""How much of an L0 query's attention actually lands on COARSE rows?

THE QUESTION THIS SETTLES
  Every proposal for "give L0 more hierarchy" (HQD, a wider cross-level window, a
  dense coarse read) assumes L0 queries are STARVED of coarse access. They may not
  be. An L0 query already sees ~56 coarse rows inside its own +/-156 mixed window,
  and the coarse lane has already mixed those rows near-globally for ~11% of the
  mixed window's cost. If the model puts almost no weight on them, more coarse
  connectivity is 30% more compute for nothing -- the HQD result, repeated.

  So: measure the softmax mass an L0 query spends on L1/L2/L3 keys, and compare

    windowed control (w=156)   -- coarse access is scarce (56 of 224 rows)
    full-window run  (w=1248)  -- coarse access is unlimited (all 224 rows)

  READING IT
    control mass ~= full mass, both small   -> L0 does not want coarse. More
                                               coarse access is not the bottleneck;
                                               stop building it.
    control mass < full mass                -> the window is binding. Widening the
                                               cross-level reach should pay.
    control SATURATED (mass/row high, few
    rows available)                         -> strongest case: it wants more than
                                               the window can give it.

METHOD
  Wraps the packed-window attention, recomputes its scores densely under the same
  window mask (diagnostic only -- the real path stays flash), softmaxes, and bins
  the mass by SOURCE LEVEL for L0 queries only. Real cached SD-VAE val latents at
  the run's val timesteps, same reasoning as bench_hqd_read.py.

USAGE
  python bench_pack_read.py --config configs/pinball_image_diffusion_latent.yaml \
      --ckpt <ckpt> --window 156
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

sys.path.insert(0, "src")
from pinball import build_pinball  # noqa: E402
from pinball.model.layers.hierarchical_message_passing import (  # noqa: E402
    HierarchicalMessagePassing,
)

STATS: list = []
_ORIG = HierarchicalMessagePassing._compute_local_attn_from_qkv


def _probe(self, q_lvl, k_lvl, v_lvl, window, causal, backend, level, **kw):
    spec = getattr(self, "_local_pack_spec", None)
    lv = spec.get("levels") if isinstance(spec, dict) else None
    # level == -1 marks the packed mixed-level call; the shape guard rejects the
    # per-level fallbacks and any subset call (multirate cycling etc.).
    if int(level) == -1 and lv is not None and int(q_lvl.size(1)) == int(lv.numel()):
        with torch.no_grad():
            N = int(lv.numel())
            B, H, D = int(q_lvl.size(0)), int(q_lvl.size(2)), int(q_lvl.size(3))
            rows = torch.arange(N, device=q_lvl.device)
            l0 = (lv == 0).nonzero(as_tuple=False).view(-1)
            qf = q_lvl.index_select(1, l0).permute(0, 2, 1, 3).float()   # [B,H,n0,D]
            kf = k_lvl.permute(0, 2, 1, 3).float()                       # [B,H,N,D]
            mass = torch.zeros(4, device=q_lvl.device, dtype=torch.float64)
            CH = 256
            for s in range(0, int(l0.numel()), CH):
                e = min(int(l0.numel()), s + CH)
                sc = torch.matmul(qf[:, :, s:e], kf.transpose(-1, -2)) / math.sqrt(D)
                dr = rows.view(1, -1) - l0[s:e].view(-1, 1)
                ok = (dr >= -int(window)) & (dr <= 0) if causal else (dr.abs() <= int(window))
                sc = sc.masked_fill(~ok.view(1, 1, e - s, N), float("-inf"))
                w = torch.nan_to_num(sc.softmax(dim=-1), nan=0.0)         # [B,H,c,N]
                per_key = w.sum(dim=(0, 1, 2)).double()                   # [N]
                for L in range(4):
                    mass[L] += per_key[lv == L].sum()
            tot = float(mass.sum())
            avail = {L: int((lv == L).sum()) for L in range(4)}
            # how many rows of each level a query can actually reach in-window
            span = (2 * int(window) + 1) if not causal else (int(window) + 1)
            reach = {L: min(avail[L], int(round(span * avail[L] / N))) for L in range(4)}
            STATS.append({
                "frac": [float(mass[L]) / tot for L in range(4)],
                "reach": reach, "avail": avail, "window": int(window),
            })
    return _ORIG(self, q_lvl, k_lvl, v_lvl, window, causal, backend, level, **kw)


def _cosine_abar(t, steps, s=0.008):
    f = lambda u: math.cos((u + s) / (1.0 + s) * math.pi / 2.0) ** 2
    return f(t / steps) / f(0.0)


def _latents(root, n, device):
    meta = glob.glob(os.path.join(root, ".image_cache", "val_*_metadata.json"))
    fp = json.load(open(meta[0]))["fingerprint"]
    sh = os.path.join(root, ".image_cache", f"val_{fp}_shard0000")
    tk = torch.load(os.path.join(sh, "tokens.pt"), map_location="cpu", weights_only=False)
    lb = torch.load(os.path.join(sh, "labels.pt"), map_location="cpu", weights_only=False)
    sel = torch.arange(n) * max(1, tk.size(0) // max(1, n))
    return tk[sel].float().to(device), lb[sel].long().to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--window", type=int, default=0, help="override local_pack_window")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--timesteps", default="250,500,750")
    p.add_argument("--label", default="")
    a = p.parse_args()

    ov = {"hier_layer_compile": False, "hqd_static_compile": False,
          "hier_refresh_compile": False}
    if a.window > 0:
        ov["local_pack_window"] = a.window
        ov["local_pack_coarse_window"] = max(a.window, 896)
    model, _, args, dev = build_pinball(cfg_path=a.config, device=a.device,
                                        set_global_seed=True, warn_unused_keys=False,
                                        override=ov)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    model.eval()

    x0, labels = _latents(str(getattr(args, "image_dataset_root")), a.batch, dev)
    HierarchicalMessagePassing._compute_local_attn_from_qkv = _probe
    steps = int(getattr(args, "image_diffusion_steps", 1000))

    rows = []
    for t in (int(v) for v in a.timesteps.split(",")):
        ab = _cosine_abar(t, steps)
        torch.manual_seed(20260812)
        xt = math.sqrt(ab) * x0 + math.sqrt(1.0 - ab) * torch.randn_like(x0)
        STATS.clear()
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            model(xt,
                  attention_mask=torch.ones(x0.shape[:2], device=dev, dtype=torch.long),
                  class_labels=labels,
                  timesteps=torch.full((x0.size(0),), t, device=dev, dtype=torch.long))
        if not STATS:
            raise RuntimeError("packed window never ran -- is local_pack_cross_level on?")
        f = [sum(s["frac"][L] for s in STATS) / len(STATS) for L in range(4)]
        rows.append(f)
        print(f"  t={t:4d}  L0={f[0]:6.3f}  L1={f[1]:6.3f}  L2={f[2]:6.3f}  L3={f[3]:6.3f}"
              f"   coarse={sum(f[1:]):6.3f}   ({len(STATS)} layers)")

    m = [sum(r[L] for r in rows) / len(rows) for L in range(4)]
    s0 = STATS[0]
    coarse_reach = sum(s0["reach"][L] for L in (1, 2, 3))
    coarse_avail = sum(s0["avail"][L] for L in (1, 2, 3))
    tag = a.label or os.path.basename(a.ckpt)
    print(f"\n{tag}  window={s0['window']}")
    print(f"  L0 query mass:  on L0 = {m[0]:.3f}   on COARSE = {sum(m[1:]):.3f}"
          f"   (L1 {m[1]:.3f} / L2 {m[2]:.3f} / L3 {m[3]:.3f})")
    print(f"  coarse rows reachable in-window: {coarse_reach} of {coarse_avail}"
          f"   -> mass per reachable coarse row = {sum(m[1:]) / max(1, coarse_reach):.5f}")


if __name__ == "__main__":
    main()
