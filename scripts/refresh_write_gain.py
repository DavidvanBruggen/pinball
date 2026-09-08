#!/usr/bin/env python
"""Effective refresh write gain (gate x projection gain) from a trained checkpoint.

CPU-only, no model build. Usage:
    python scripts/refresh_write_gain.py <checkpoint.pt>

Handles both gate layouts: one gate shared across depth (pre-2026-09-07 checkpoints) and
per-layer gates (hier_refresh_per_layer_gates, default on). For per-layer gates the depth
MEAN is what compares against a shared-gate run; early/late columns show the depth profile.

Watch for: projection RMS gain near 1.0, which is the xavier value -- the native init is
std 0.02 -> 0.64, so ~1.0 means the projection never moved off a clobbered init.
"""
import re
import sys

import torch


def _rows(gate, n_slots):
    """Return the gate as [layers, slots] (1 layer when shared across depth)."""
    if gate.dim() == 0:
        return gate.reshape(1, 1)
    if gate.numel() == n_slots:
        return gate.reshape(1, n_slots)
    return gate.reshape(-1, n_slots)


def main(path: str) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj.get("model_state_dict", obj)

    def gain(w):
        return (w.float().norm() / (w.shape[1] ** 0.5)).item()

    for direction in ("downward", "upward"):
        projs = {k: v for k, v in sd.items()
                 if re.search(rf"{direction}_refresh_proj", k) and v.dim() == 2}
        gates = {k: v.float() for k, v in sd.items()
                 if re.search(rf"{direction}_refresh_gates", k)}
        if not projs or not gates:
            continue
        names = sorted(projs)
        shared = len(gates) == 1 and direction == "upward"
        table = _rows(next(iter(gates.values())), len(names)) if shared else None

        print(f"\n{direction} refresh")
        print(f"  {'pair/level':<12}{'gate':>9}{'proj gain':>11}{'write':>9}"
              f"{'early':>9}{'late':>8}{'late/early':>12}")
        for i, key in enumerate(names):
            w = projs[key].float()
            tag = key.split("_proj.")[-1].replace(".weight", "")
            if shared:
                col = table[:, i]
            else:
                match = [v for k, v in gates.items() if k.endswith(tag)]
                if not match:
                    continue
                col = match[0].reshape(-1)
            g, gg = col.mean().item(), gain(w)
            n = col.numel()
            if n > 1:
                e = col[: max(1, n // 3)].mean().item()
                l = col[-max(1, n // 3):].mean().item()
                prof = f"{e:9.4f}{l:8.4f}{l / e:11.2f}x"
            else:
                prof = f"{'':9}{'':8}{'shared':>12}"
            print(f"  {tag:<12}{g:>9.4f}{gg:>11.3f}{g * gg:>9.4f}{prof}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
