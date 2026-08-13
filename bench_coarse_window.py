# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
"""Per-level coarse windows on FLASH, unified with the L0 window in ONE softmax.

THE PROBLEM
  We want an L0 query to see a WIDER window of coarse rows than of L0 rows --
  coarse rows are sparse in position, so a fixed count of them spans much further
  (level L has stride s in L0 tokens, so n_L = N/s nodes cover the whole sequence).
  Expressed over L0 coordinates that constraint is

      |s*m - p| <= W          query p, level-L node m

  a band of slope 1/s. Flash's sliding window is |i - j| <= w -- slope 1 -- and
  cross-attention with unequal seqlens only shifts the diagonal, it doesn't tilt
  it. Dense SDPA with an explicit mask can express it, but materialises
  [n_q, n_k]; since n_L = N/s grows with N, at 131k tokens that is 2.1e9 entries.

THE CONSTRUCTION
  Fold the stride into the batch dimension so the slope becomes 1:

      q: [B, N, H, D]   -> [B, N/s, s, H, D] -> [B*s, N/s, H, D]
      k: [B, n_L, H, D] -> broadcast          -> [B*s, n_L, H, D]

  N/s == n_L by construction, so the reshaped view is square and the band is
  |block - node| <= W/s, which flash takes natively as window_size. Nothing
  [n_q, n_k]-shaped is ever materialised.

  APPROXIMATION: all s queries in a block share one key window, so a query at the
  start of a block sees the same coarse rows as one at its end. That is a half-
  stride of slop -- the same coarsening the hierarchy already applies by pooling
  those s tokens into one node, and the same trick hqd_tiled_apply uses.

THE SHARED SOFTMAX
  Two flash calls (L0 window, coarse band) are two independently-normalised
  softmaxes. Summing them is NOT the same as one softmax over the union. With
  Z = exp(lse) from each call, though,

      out = (Z_local * out_local + Z_coarse * out_coarse) / (Z_local + Z_coarse)

  IS exactly the softmax over the concatenated key list -- provided the key sets
  are DISJOINT, which they are here (L0 keys and level-L keys are different rows).
  That disjointness is why this L0-only + coarse-only split is used rather than
  the mixed window, which would double-count the coarse rows inside +-W.

WHAT THIS SCRIPT CHECKS
  1. band     : flash block-reshape == dense reference, same mask
  2. union    : LSE merge == dense softmax over the concatenated keys
  3. multi    : several levels, each with its own radius, all merged
  4. memory   : dense reference vs flash, as N grows
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func


def dense_band_reference(q, k, v, stride, w_blocks, scale):
    """[B,N,H,D] x [B,n_L,H,D] under |p//stride - m| <= w_blocks. fp32, explicit mask."""
    B, N, H, D = q.shape
    n_L = k.size(1)
    qs, ks, vs = q.float(), k.float(), v.float()
    scores = torch.einsum("bqhd,bkhd->bhqk", qs, ks) * scale
    p = torch.arange(N, device=q.device).view(-1, 1) // stride
    m = torch.arange(n_L, device=q.device).view(1, -1)
    keep = (p - m).abs() <= int(w_blocks)
    scores = scores.masked_fill(~keep.view(1, 1, N, n_L), float("-inf"))
    w = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", w, vs), torch.logsumexp(scores, dim=-1)


def flash_band(q, k, v, stride, w_blocks):
    """Same thing on flash: fold the stride into batch so the band is slope 1."""
    B, N, H, D = q.shape
    n_L = k.size(1)
    assert N % stride == 0 and N // stride == n_L, (
        f"block count N/s={N // stride} must equal the level size {n_L}")
    nb = N // stride
    # [B,N,H,D] -> [B,nb,s,H,D] -> [B,s,nb,H,D] -> [B*s,nb,H,D]
    qb = q.view(B, nb, stride, H, D).permute(0, 2, 1, 3, 4).reshape(B * stride, nb, H, D)
    kb = k.unsqueeze(1).expand(B, stride, n_L, H, D).reshape(B * stride, n_L, H, D)
    vb = v.unsqueeze(1).expand(B, stride, n_L, H, D).reshape(B * stride, n_L, H, D)
    out, lse, _ = flash_attn_func(
        qb.contiguous(), kb.contiguous(), vb.contiguous(),
        causal=False, window_size=(int(w_blocks), int(w_blocks)), return_attn_probs=True)
    out = out.view(B, stride, nb, H, D).permute(0, 2, 1, 3, 4).reshape(B, N, H, D)
    lse = lse.view(B, stride, H, nb).permute(0, 2, 3, 1).reshape(B, H, N)
    return out, lse


def flash_local(q, k, v, w):
    out, lse, _ = flash_attn_func(q, k, v, causal=False, window_size=(int(w), int(w)),
                                  return_attn_probs=True)
    return out, lse


def lse_merge(parts):
    """One softmax over the union of DISJOINT key sets, from per-part (out, lse)."""
    lses = torch.stack([p[1] for p in parts], 0)                 # [P,B,H,N]
    mx = lses.max(dim=0, keepdim=True).values
    wts = (lses - mx).exp()
    wts = wts / wts.sum(dim=0, keepdim=True).clamp_min(1e-20)    # [P,B,H,N]
    out = 0
    for i, (o, _) in enumerate(parts):
        out = out + o * wts[i].permute(0, 2, 1).unsqueeze(-1)    # [B,H,N]->[B,N,H,1]
    return out


def dense_union_reference(q, keys, vals, masks, scale):
    """One fp32 softmax over concatenated keys with per-part masks."""
    B, N, H, D = q.shape
    sc = [torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale for k in keys]
    sc = [s.masked_fill(~m.view(1, 1, *m.shape), float("-inf")) for s, m in zip(sc, masks)]
    allsc = torch.cat(sc, dim=-1)
    allv = torch.cat([v.float() for v in vals], dim=1)
    return torch.einsum("bhqk,bkhd->bqhd", allsc.softmax(dim=-1), allv)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-9)).item()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=256, help="L0 tokens (small, for the dense ref)")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    dev = torch.device(a.device)
    torch.manual_seed(0)
    B, N, H, D = a.batch, a.n, a.heads, a.dim
    scale = 1.0 / math.sqrt(D)
    dt = torch.float16
    mk = lambda n: torch.randn(B, n, H, D, device=dev, dtype=dt)

    # Level strides mirror compression_ratios [16,4,4] @ overlap 0.5 -> 8 / 16 / 32
    LEVELS = [("L1", 8), ("L2", 16), ("L3", 32)]
    print(f"N={N} B={B} H={H} D={D}   levels: "
          + ", ".join(f"{n}(stride {s}, {N // s} nodes)" for n, s in LEVELS))

    print("\n1. BAND: flash block-reshape vs dense reference")
    for name, s in LEVELS:
        n_L = N // s
        k, v = mk(n_L), mk(n_L)
        q = mk(N)
        wb = max(1, n_L // 4)
        ref, _ = dense_band_reference(q, k, v, s, wb, scale)
        got, _ = flash_band(q, k, v, s, wb)
        print(f"   {name}: nodes={n_L:4d} w_blocks={wb:3d} "
              f"(spans {(2 * wb + 1) * s:5d} L0 tokens)   rel err = {rel(got.float(), ref):.2e}")

    print("\n2. UNION: LSE merge of (L0 window + L1 band) vs one dense softmax")
    q = mk(N)
    k0, v0 = mk(N), mk(N)
    s1 = 8
    k1, v1 = mk(N // s1), mk(N // s1)
    w0, wb1 = N // 8, (N // s1) // 4
    o0 = flash_local(q, k0, v0, w0)
    o1 = flash_band(q, k1, v1, s1, wb1)
    merged = lse_merge([o0, o1])
    idx = torch.arange(N, device=dev)
    m0 = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= w0
    m1 = ((idx.view(-1, 1) // s1) - torch.arange(N // s1, device=dev).view(1, -1)).abs() <= wb1
    ref = dense_union_reference(q, [k0, k1], [v0, v1], [m0, m1], scale)
    print(f"   L0 window +-{w0} ({2 * w0 + 1} keys) UNION L1 band +-{wb1} "
          f"({2 * wb1 + 1} keys, spans {(2 * wb1 + 1) * s1} L0)")
    print(f"   rel err = {rel(merged.float(), ref):.2e}")
    add = (o0[0].float() + o1[0].float())
    print(f"   [additive sum, for contrast]  rel err = {rel(add, ref):.2e}")

    print("\n3. MULTI-LEVEL: L0 window + L1 + L2 + L3, each its own radius, one softmax")
    parts, keys, vals, masks = [flash_local(q, k0, v0, w0)], [k0], [v0], [m0]
    for name, s in LEVELS:
        n_L = N // s
        kk, vv = mk(n_L), mk(n_L)
        wb = max(1, n_L // 4)
        parts.append(flash_band(q, kk, vv, s, wb))
        keys.append(kk); vals.append(vv)
        masks.append(((idx.view(-1, 1) // s) - torch.arange(n_L, device=dev).view(1, -1)).abs() <= wb)
    merged = lse_merge(parts)
    ref = dense_union_reference(q, keys, vals, masks, scale)
    print(f"   parts={len(parts)}  rel err = {rel(merged.float(), ref):.2e}")

    print("\n4. MEMORY: dense band mask vs flash, per level, as N grows")
    print(f"   {'N':>8} {'level':>6} {'n_L':>8} {'dense [n_q,n_k]':>18} {'flash k/v replica':>19}")
    for n in (1024, 16384, 131072):
        for name, s in LEVELS[:1]:
            n_L = n // s
            dense = n * n_L * 2 / 2**30
            flash = s * n_L * H * D * 2 / 2**30
            print(f"   {n:>8} {name:>6} {n_L:>8} {dense:>15.2f} GB {flash:>16.3f} GB")


if __name__ == "__main__":
    main()
