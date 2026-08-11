# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
"""Effective receptive field + step cost per architecture variant.

Answers "can information from a distant pixel reach this prediction, and what does
that cost?" in ~1 minute per variant, instead of a training run per variant.

METHOD (erf)
  x : [1, T, D] random input latents, requires_grad
  probe = || model(x)[0, t_probe, :] ||^2          (a scalar at ONE token)
  g = d(probe)/dx                                   (one backward)
  influence(j) = || g[0, j, :] ||_2                 (per input token)
  Bin influence by Chebyshev distance from the probe token on the H x W grid,
  normalise each arm by its own near-field (distance 0-2) mean, average over probes.

  The reported far/near ratio is the headline: 1.0 = influence independent of
  distance (fully global), 0.1 = strongly local.

WHAT IT DOES AND DOES NOT TELL YOU
  It measures whether a route EXISTS and carries signal through the architecture.
  It does NOT tell you the trained model will USE that route -- gates can and do
  close routes during training. Treat a flat ERF as necessary, not sufficient.
  Absolute values are not meaningful (LayerNorm/residual diffuse gradients); only
  compare arms measured in the same run.

USAGE
  python bench_erf.py --config configs/pinball_image_diffusion_latent.yaml
  python bench_erf.py --config <cfg> --arms '{"hqd": {"hierarchical_query_descent_enable": true}}'
  python bench_erf.py --config <cfg> --ckpt path/to/pinball_best.pt   # trained weights

NOTE ON OVERRIDES
  override= is applied AFTER PinballConfig, so friendly aliases do NOT expand.
  Pass expanded keys (hierarchical_query_descent_enable), never aliases (use_hqd).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "src")
from pinball import build_pinball  # noqa: E402

BANDS = [(0, 2), (3, 5), (6, 8), (9, 12), (13, 16), (17, 24)]

DEFAULT_ARMS = {
    "baseline": {},
    "hqd every layer": {"hierarchical_query_descent_enable": True},
    "hqd n=4": {"hierarchical_query_descent_enable": True, "hqd_every_n": 4},
}


def _grid(args) -> tuple:
    h = int(getattr(args, "graph_grid_height", 0) or 0)
    w = int(getattr(args, "graph_grid_width", 0) or 0)
    if h and w:
        return h, w
    side = int(round(int(getattr(args, "block_size", 1024)) ** 0.5))
    return side, side


def _input_layer(model):
    """Pinball uses token_embedding; TransformerLM in features mode uses feature_projection."""
    for name in ("token_embedding", "feature_projection"):
        layer = getattr(model, name, None)
        if layer is not None and hasattr(layer, "in_features"):
            return layer
    raise AttributeError("no feature input layer found on the model")


def measure(cfg_path, override, ckpt, probes, iters, device):
    override = dict(override)
    # An arm may point at a DIFFERENT config file (e.g. the transformer baseline)
    # via the reserved "__config" key, so unlike architectures land in one table.
    cfg_path = override.pop("__config", cfg_path)
    model, _, args, dev = build_pinball(
        cfg_path=cfg_path, device=device, override=override,
        set_global_seed=True, warn_unused_keys=False,
    )
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        model.load_state_dict(sd, strict=False)
    gh, gw = _grid(args)
    n_tok, feat = gh * gw, _input_layer(model).in_features

    # --- ERF ---
    # Bin PER PROBE against that probe's own distance map, then average the ratios.
    # (Averaging influence maps across probes first and binning with one probe's
    # distances mispairs the data -- each probe has a different distance field.)
    model.eval()
    idx = torch.arange(n_tok)
    per_probe_bands, per_probe_far = [], []
    for (pr, pc) in probes:
        torch.manual_seed(1234)
        x = torch.randn(1, n_tok, feat, device=dev, requires_grad=True)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out = model(x)
            out = out[0] if isinstance(out, (tuple, list)) else out
            probe = out[0, pr * gw + pc, :].float().pow(2).sum()
        g, = torch.autograd.grad(probe, x)
        infl = g[0].float().norm(dim=-1).cpu()
        d = torch.maximum((idx // gw - pr).abs().float(), (idx % gw - pc).abs().float())
        base = infl[(d >= BANDS[0][0]) & (d <= BANDS[0][1])].mean().item()
        if not base:
            continue
        row = []
        for lo, hi in BANDS:
            m = (d >= lo) & (d <= hi)
            row.append(infl[m].mean().item() / base if bool(m.any()) else float("nan"))
        per_probe_bands.append(row)
        fm = d > 8
        per_probe_far.append(infl[fm].mean().item() / base if bool(fm.any()) else float("nan"))

    def _nanmean(vals):
        ok = [v for v in vals if v == v]
        return sum(ok) / len(ok) if ok else float("nan")

    bands = [_nanmean([r[i] for r in per_probe_bands]) for i in range(len(BANDS))]
    far = _nanmean(per_probe_far)

    # --- step cost (fwd + bwd, same shape) ---
    model.train()
    x = torch.randn(1, n_tok, feat, device=dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    def step():
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out = model(x)
            out = out[0] if isinstance(out, (tuple, list)) else out
            loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    n_par = sum(p.numel() for p in model.parameters())
    del model, x, opt
    torch.cuda.empty_cache()
    return bands, far, ms, n_par


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--arms", default=None, help='JSON {"name": {override...}}; default = HQD sweep')
    p.add_argument("--ckpt", default=None, help="optional checkpoint to load (strict=False)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--probes", default="16,16;8,20;24,10", help='"r,c;r,c" probe token positions')
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)  # internal: one arm, JSON out
    a = p.parse_args()

    probes = [tuple(int(v) for v in pr.split(",")) for pr in a.probes.split(";")]

    # Each arm runs in its OWN process. Building several models in one process blows
    # through torch._dynamo's recompile limit, after which frames silently fall back
    # to eager and every later arm's timing is wrong.
    if a.single is not None:
        bands, far, ms, n_par = measure(
            a.config, json.loads(a.single), a.ckpt, probes, a.iters, torch.device(a.device))
        print("__RESULT__" + json.dumps({"bands": bands, "far": far, "ms": ms, "params": n_par}))
        return

    import subprocess
    arms = json.loads(a.arms) if a.arms else DEFAULT_ARMS
    hdr = " ".join(f"{lo}-{hi:<3}" for lo, hi in BANDS)
    print(f"\nconfig={a.config}  probes={probes}  ckpt={a.ckpt or 'init weights'}")
    print(f"{'arm':26} {hdr}  far/near   ms/step   params")
    base_ms = None
    for name, ov in arms.items():
        cmd = [sys.executable, __file__, "--config", a.config, "--device", a.device,
               "--iters", str(a.iters), "--probes", a.probes, "--single", json.dumps(ov)]
        if a.ckpt:
            cmd += ["--ckpt", a.ckpt]
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("__RESULT__")), None)
        if line is None:
            print(f"{name:26} FAILED: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else '?'}")
            continue
        r = json.loads(line[len("__RESULT__"):])
        base_ms = r["ms"] if base_ms is None else base_ms
        print(f"{name:26} " + " ".join(f"{b:5.3f}" for b in r["bands"])
              + f"  {r['far']:8.3f}  {r['ms']:7.1f} ({r['ms'] / base_ms - 1:+6.1%})  {r['params'] / 1e6:7.2f}M")


if __name__ == "__main__":
    main()
