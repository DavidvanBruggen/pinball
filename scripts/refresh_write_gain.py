#!/usr/bin/env python
"""Effective refresh write gain (gate x projection gain) from a trained checkpoint.

CPU-only, no model build. Usage:
    python scripts/refresh_write_gain.py <checkpoint.pt>

Watch for: upward write growing monotonically with level (compounding, since the
refresh chain runs after the shared postnorm with no normalisation between levels),
and projection RMS gain near 1.0, which is the xavier value -- the native init is
std 0.02 -> gain 0.64, so ~1.0 means the projection never moved off a clobbered init.
"""
import re
import sys

import torch


def main(path: str) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj.get("model_state_dict", obj)

    def gain(w):
        return (w.float().norm() / (w.shape[1] ** 0.5)).item()

    for direction in ("downward", "upward"):
        gates = {k: v for k, v in sd.items()
                 if re.search(rf"{direction}_refresh_gates", k)}
        projs = {k: v for k, v in sd.items()
                 if re.search(rf"{direction}_refresh_proj", k) and v.dim() == 2}
        if not projs:
            continue
        print(f"\n{direction} refresh")
        print(f"  {'pair/level':<12}{'gate':>9}{'proj gain':>11}{'write':>9}{'spectral':>10}")
        flat = torch.cat([v.reshape(-1).float() for v in gates.values()]) if gates else None
        for i, key in enumerate(sorted(projs)):
            w = projs[key].float()
            tag = key.split("_proj.")[-1].replace(".weight", "")
            named = [v for k, v in gates.items() if k.endswith(tag)]
            g = float(named[0]) if named else (float(flat[i]) if flat is not None and i < flat.numel() else float("nan"))
            print(f"  {tag:<12}{g:>9.4f}{gain(w):>11.3f}{g * gain(w):>9.4f}"
                  f"{torch.linalg.matrix_norm(w, 2).item():>10.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
