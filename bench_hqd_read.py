# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
"""Is the HQD read SELECTIVE, or just flat? Paired prerope-vs-RoPE'd probe.

WHY THIS EXISTS
  bench_erf.py reports far-field gradient magnitude. That metric cannot tell
  "precise retrieval from a distant token" apart from "uniform attention over
  distant junk" -- both push gradient out to far inputs. A mechanism that is
  merely NOISIER therefore scores HIGHER on ERF. The HQD graft measured 3.069
  far/near at the moment it was spliced in untrained, the highest number ever
  recorded on this model, which is exactly that failure mode.

  So when a change raises ERF, the follow-up question is whether the read got
  sharper or flatter. This measures that directly:

    eff_k   exp(entropy) of the read's softmax = how many nodes a query
            effectively attends to. Lower = more selective.
    norm_H  entropy / log(n_valid). 1.0 = uniform over everything available,
            0.0 = one-hot. Scale-free, so it survives n_valid differing.
    w_dist  softmax-WEIGHTED mean |src - query| index distance: where the
            attention mass actually lands.
    s_dist  UNWEIGHTED mean over the same candidates = where selection put
            them. The descent is untouched by the read flag, so this is a
            control: it must match across arms or the probe is wrong.

  READING IT
    higher ERF + higher eff_k        -> flatter read; the ERF gain is noise
    higher ERF + equal/lower eff_k   -> sharper read reaching further; real

REAL LATENTS, NOT NOISE (this is load-bearing)
  RoPE contributes an input-INDEPENDENT term to q.k, so on random input the
  RoPE'd arm looks peaked purely from the rotation while the content-only arm
  has nothing to key on. That biases the comparison toward RoPE. This reads
  actual cached SD-VAE val latents and applies the run's cosine forward noising
  at a mid timestep, so both arms see the structure the model trains on.

PAIRED BY CONSTRUCTION
  ONE model, ONE set of weights, ONE input tensor. The arms differ only by
  mp.hqd_read_prerope, flipped between forward passes. (Safe to flip at
  runtime: q_prerope is captured whenever a pack spec is present, which it
  always is under local_pack_cross_level.) Compile is forced off so the
  instrumentation is not traced away.

USAGE
  python bench_hqd_read.py --config configs/pinball_image_diffusion_latent_hqd.yaml \
      --ckpt pinball/.../pinball_last.pt --device cuda:0
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
_ORIG_TILED = HierarchicalMessagePassing._compute_hqd_tiled_attn


def _probe_tiled_attn(self, q, k, v, b_idx, src_idx, dst_idx, num_nodes, B):
    """Record read-selectivity stats, then delegate to the real implementation.

    The scoring prefix below mirrors _compute_hqd_tiled_attn. It is recomputed
    rather than hooked so the recorded weights are exactly the ones the real
    path uses; the duplicated work is irrelevant for a few probe forwards. If
    that function's scoring changes, this must change with it.
    """
    with torch.no_grad():
        device = q.device
        tile = max(1, int(getattr(self, "hqd_tile_size", 64)))
        num_heads, qk_dim = int(q.size(-2)), int(q.size(-1))
        n_tiles = (int(num_nodes) + tile - 1) // tile
        topk = max(1, min(int(getattr(self, "hqd_tile_topk", 64)), int(num_nodes)))

        tile_id = torch.div(dst_idx, tile, rounding_mode="floor").clamp_(0, n_tiles - 1)
        lin = (b_idx * n_tiles + tile_id) * int(num_nodes) + src_idx
        hist = torch.zeros(B * n_tiles * int(num_nodes), device=device, dtype=torch.float32)
        hist.index_add_(0, lin, torch.ones_like(lin, dtype=torch.float32))
        hist = hist.view(B, n_tiles, int(num_nodes))
        tile_nodes = hist.topk(topk, dim=-1).indices
        tile_valid = hist.gather(-1, tile_nodes) > 0

        batch_off = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1) * int(num_nodes)
        flat_nodes = (tile_nodes + batch_off).reshape(-1)
        k_t = k.reshape(-1, num_heads, qk_dim).index_select(0, flat_nodes)
        k_t = k_t.view(B, n_tiles, topk, num_heads, qk_dim)

        pad = n_tiles * tile - int(num_nodes)
        q_p = q if pad == 0 else torch.cat([q, q.new_zeros(B, pad, num_heads, qk_dim)], dim=1)
        q_t = q_p.view(B, n_tiles, tile, num_heads, qk_dim)

        qb = q_t.permute(0, 1, 3, 2, 4).reshape(B * n_tiles * num_heads, tile, qk_dim)
        kb = k_t.permute(0, 1, 3, 2, 4).reshape(B * n_tiles * num_heads, topk, qk_dim)
        scores = torch.bmm(qb, kb.transpose(1, 2)) / math.sqrt(float(qk_dim))
        keep = tile_valid.view(B, n_tiles, 1, topk).expand(B, n_tiles, num_heads, topk)
        scores = scores.masked_fill(~keep.reshape(-1, 1, topk), float("-inf"))
        w = torch.nan_to_num(scores.softmax(dim=-1), nan=0.0).float()
        w = w.view(B, n_tiles, num_heads, tile, topk)

        # Index distance from each query row to each candidate, in packed-node
        # coordinates -- the same measure the hqd.mean_dist gate monitor uses.
        qpos = (torch.arange(n_tiles, device=device).view(1, n_tiles, 1) * tile
                + torch.arange(tile, device=device).view(1, 1, tile))       # [1,n_tiles,tile]
        dist = (tile_nodes.view(B, n_tiles, 1, topk).float()
                - qpos.view(1, n_tiles, tile, 1).float()).abs()             # [B,n_tiles,tile,topk]

        # Live rows only: drop query-axis padding and tiles that produced no edges.
        alive = (w.sum(-1) > 1e-6)                                          # [B,n_tiles,H,tile]
        real_q = (qpos < int(num_nodes)).view(1, n_tiles, 1, tile)
        alive = alive & real_q
        n_alive = int(alive.sum().item())
        if n_alive == 0:
            return _ORIG_TILED(self, q, k, v, b_idx, src_idx, dst_idx, num_nodes, B)

        H = -(w.clamp_min(1e-12).log() * w).sum(-1)                         # [B,n_tiles,H,tile]
        n_valid = tile_valid.sum(-1).clamp_min(1).float()                   # [B,n_tiles]
        n_val_b = n_valid.view(B, n_tiles, 1, 1).expand_as(H)
        wd = (w * dist.unsqueeze(2)).sum(-1)                                # weighted distance
        sd_t = ((dist * tile_valid.view(B, n_tiles, 1, topk).float()).sum(-1)
                / n_valid.view(B, n_tiles, 1))                              # [B,n_tiles,tile]
        sd = sd_t.unsqueeze(2).expand_as(H)

        m = alive
        STATS.append({
            "eff_k": float(H[m].exp().mean()),
            "norm_H": float((H[m] / n_val_b[m].log().clamp_min(1e-6)).mean()),
            "w_dist": float(wd[m].mean()),
            "s_dist": float(sd[m].mean()),
            "n_valid": float(n_val_b[m].mean()),
            "top1": float(w.max(-1).values[m].mean()),
            "rows": n_alive,
        })
    return _ORIG_TILED(self, q, k, v, b_idx, src_idx, dst_idx, num_nodes, B)


def _cosine_abar(t, steps, s=0.008):
    """alpha_bar for the run's cosine schedule (image_diffusion_schedule: cosine)."""
    f = lambda u: math.cos((u + s) / (1.0 + s) * math.pi / 2.0) ** 2
    return f(t / steps) / f(0.0)


def _load_latents(root, n, device):
    """Real cached SD-VAE val latents: [n, 1024, 4] float + int labels."""
    meta = glob.glob(os.path.join(root, ".image_cache", "val_*_metadata.json"))
    if not meta:
        raise FileNotFoundError(f"no val latent cache under {root}/.image_cache")
    fp = json.load(open(meta[0]))["fingerprint"]
    shard = os.path.join(root, ".image_cache", f"val_{fp}_shard0000")
    toks = torch.load(os.path.join(shard, "tokens.pt"), map_location="cpu", weights_only=False)
    labs = torch.load(os.path.join(shard, "labels.pt"), map_location="cpu", weights_only=False)
    # Stride the shard: consecutive entries are the same class, and one class of
    # conditioning is not a sample of the model's behaviour.
    sel = torch.arange(n) * max(1, toks.size(0) // max(1, n))
    return toks[sel].float().to(device), labs[sel].long().to(device)


def _agg(rows):
    """Mean over layers, weighted by how many live query rows each contributed."""
    tot = sum(r["rows"] for r in rows)
    keys = ("eff_k", "norm_H", "w_dist", "s_dist", "n_valid", "top1")
    return {k: sum(r[k] * r["rows"] for r in rows) / tot for k in keys}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--timesteps", default="250,500,750",
                   help="forward-noising timesteps to probe (the run's val timesteps)")
    p.add_argument("--input", choices=("latent", "noise"), default="latent",
                   help="latent = real cached SD-VAE val latents (default, and the "
                        "only fair setting -- see the module docstring). noise = "
                        "random, ONLY to reproduce bench_erf.py's own input regime.")
    a = p.parse_args()

    # Compile off: the instrumentation lives inside a layer torch.compile would trace.
    model, _, args, dev = build_pinball(
        cfg_path=a.config, device=a.device, set_global_seed=True, warn_unused_keys=False,
        override={"hier_layer_compile": False, "hqd_static_compile": False,
                  "hier_refresh_compile": False},
    )
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    if a.input == "latent":
        x0, labels = _load_latents(str(getattr(args, "image_dataset_root")), a.batch, dev)
    else:
        torch.manual_seed(1234)   # bench_erf.py's seed and distribution
        gh = gw = int(round(int(getattr(args, "block_size", 1024)) ** 0.5))
        x0 = torch.randn(a.batch, gh * gw, int(getattr(args, "image_latent_channels", 4)),
                         device=dev)
        labels = torch.zeros(a.batch, device=dev, dtype=torch.long)
    print(f"input={a.input} {tuple(x0.shape)}  std={x0.std().item():.3f}  "
          f"labels={labels.tolist()}")

    mps = [m for m in model.modules() if isinstance(m, HierarchicalMessagePassing)]
    HierarchicalMessagePassing._compute_hqd_tiled_attn = _probe_tiled_attn
    steps = int(getattr(args, "image_diffusion_steps", 1000))
    tlist = [int(t) for t in a.timesteps.split(",")]

    print(f"\n{'arm':<22} {'t':>5} {'eff_k':>7} {'norm_H':>7} {'top1':>7} "
          f"{'w_dist':>8} {'s_dist':>8} {'n_valid':>8}")
    out = {}
    for prerope in (True, False):
        name = "read PRE-RoPE" if prerope else "read RoPE'd"
        for m in mps:
            m.hqd_read_prerope = bool(prerope)
        per_t = []
        for t in tlist:
            ab = _cosine_abar(t, steps)
            torch.manual_seed(20260812)          # identical noise across arms
            xt = math.sqrt(ab) * x0 + math.sqrt(1.0 - ab) * torch.randn_like(x0)
            ts = torch.full((x0.size(0),), t, device=dev, dtype=torch.long)
            am = torch.ones((x0.size(0), x0.size(1)), device=dev, dtype=torch.long)
            STATS.clear()
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                model(xt, attention_mask=am, class_labels=labels, timesteps=ts)
            if not STATS:
                raise RuntimeError("HQD tiled apply never ran -- is use_hqd/hqd_tiled_apply on?")
            g = _agg(STATS)
            per_t.append(g)
            print(f"{name:<22} {t:>5} {g['eff_k']:>7.2f} {g['norm_H']:>7.3f} "
                  f"{g['top1']:>7.3f} {g['w_dist']:>8.1f} {g['s_dist']:>8.1f} "
                  f"{g['n_valid']:>8.1f}   ({len(STATS)} layers)")
        out[name] = {k: sum(d[k] for d in per_t) / len(per_t) for k in per_t[0]}

    A, Bm = out["read PRE-RoPE"], out["read RoPE'd"]
    print(f"\n{'':<22} {'':>5} {'eff_k':>7} {'norm_H':>7} {'top1':>7} {'w_dist':>8} {'s_dist':>8}")
    for n, d in out.items():
        print(f"{n+' (mean)':<22} {'':>5} {d['eff_k']:>7.2f} {d['norm_H']:>7.3f} "
              f"{d['top1']:>7.3f} {d['w_dist']:>8.1f} {d['s_dist']:>8.1f}")
    print(f"{'ratio RoPEd/prerope':<22} {'':>5} {Bm['eff_k']/A['eff_k']:>7.2f} "
          f"{Bm['norm_H']/A['norm_H']:>7.2f} {Bm['top1']/A['top1']:>7.2f} "
          f"{Bm['w_dist']/A['w_dist']:>8.2f} {Bm['s_dist']/A['s_dist']:>8.2f}")
    print("\ns_dist ratio must be ~1.00 (the descent is identical in both arms).")
    print("eff_k ratio > 1 => RoPE'd read is FLATTER: the ERF gain is dilution, not reach.")
    print("eff_k ratio <= 1 => RoPE'd read is at least as selective while reaching further.")


if __name__ == "__main__":
    main()
