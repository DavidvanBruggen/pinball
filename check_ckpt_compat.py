#!/usr/bin/env python
"""Read-only check: does this checkpoint still load into the current model code?

    python check_ckpt_compat.py --ckpt PATH --config configs/pinball_dna_bidi_full.yaml

Never writes anything. (Note: do NOT use `python -m pinball.cli --config <real>.yaml` to
test a config -- that ALWAYS saves best/final into its checkpoint_dir on exit.)
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from pinball.instantiate_PINBALL_model import build_pinball

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--num-tracks", type=int, default=768)
ap.add_argument("--block-size", type=int, default=None)
ap.add_argument("--prefix", default=None, help="key prefix to strip, e.g. 'pinball.' (auto-detected if omitted)")
a = ap.parse_args()

obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = obj
for k in ("model_state_dict", "state_dict", "model"):
    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
        sd = sd[k]; print(f"unwrapped '{k}'"); break
sd = {k: v for k, v in sd.items() if hasattr(v, "shape")}

ov = {} if a.block_size is None else dict(block_size=a.block_size)
m = build_pinball(cfg_path=a.config, num_tracks=a.num_tracks, tie_weights=False,
                  device="cpu", set_global_seed=False, override=ov)[0]
want = m.state_dict()

pre = a.prefix
if pre is None:                      # auto-detect a host wrapper prefix (e.g. ChromScape's "pinball.")
    for cand in ("", "pinball.", "module.", "module.pinball."):
        if sum(1 for k in want if cand + k in sd) > 0.5 * len(want):
            pre = cand; break
    pre = pre or ""
if pre:
    print(f"stripping prefix {pre!r}")
    sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}

missing   = [k for k in want if k not in sd]
unexpected= [k for k in sd if k not in want]
mismatch  = [(k, tuple(want[k].shape), tuple(sd[k].shape)) for k in want
             if k in sd and tuple(want[k].shape) != tuple(sd[k].shape)]

print(f"\ncheckpoint tensors {len(sd)}   model tensors {len(want)}")
print(f"model params {sum(p.numel() for p in m.parameters()):,}   ckpt params {sum(v.numel() for v in sd.values()):,}")
print(f"missing (model needs, ckpt lacks) : {len(missing)}")
for k in missing[:8]: print(f"    - {k}")
print(f"unexpected (ckpt has, model lacks): {len(unexpected)}")
for k in unexpected[:8]: print(f"    + {k}")
print(f"shape mismatches                  : {len(mismatch)}")
for k, w, g in mismatch[:8]: print(f"    ! {k}  model{w} vs ckpt{g}")

dead = [k for k in unexpected if "hier_aux_pair_predictors" in k]
print()
if not missing and not unexpected and not mismatch:
    print("VERDICT: exact match -> load_state_dict(sd, strict=True)")
elif not missing and not mismatch and unexpected and set(unexpected) == set(dead):
    print(f"VERDICT: only {len(dead)} dead aux predictors are extra (trained pre hier_aux_mode threading).")
    print("         Safe for EVAL -- they feed the aux loss only, never the forward output:")
    print("           missing, unexpected = model.load_state_dict(sd, strict=False)")
    print("           assert not missing and all('hier_aux_pair_predictors' in k for k in unexpected)")
else:
    print("VERDICT: REAL mismatch -- do not load. The config used here does not match the")
    print("         one this checkpoint was trained with. Diff the yaml against the training commit.")
