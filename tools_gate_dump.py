# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
"""Per-layer attention source gates from any checkpoint. No GPU, no model build.

    python tools_gate_dump.py <ckpt.pt> [<ckpt.pt> ...]

For a gate that starts uniform, the per-layer SPREAD is the signal: a falling mean
with a widening spread is the source specialising by depth; a falling mean with a
tight spread is rejection. The mean alone cannot tell them apart.
"""
import sys
import torch

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    print(f"\n{path}   step={ck.get('global_step')} epoch={ck.get('current_epoch')}")
    for name in ("hqd", "local", "graph"):
        rows = sorted(
            (int(k.split(".")[1]), torch.sigmoid(v.float()).item())
            for k, v in sd.items() if f"{name}_source_gate_logit" in k
        )
        if not rows:
            continue
        vals = [v for _, v in rows]
        print(f"  {name:5s} mean={sum(vals)/len(vals):.5f} min={min(vals):.5f} "
              f"max={max(vals):.5f} spread={max(vals)-min(vals):.5f}")
        print("        " + " ".join(f"{v:.4f}" for v in vals))
