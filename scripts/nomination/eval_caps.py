"""Enforce region-capped top-k at eval on a trained checkpoint: loss per bucket + pick distance profile."""
import os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_nom.py")).read().split("res = {}")[0]
exec(compile(src, "diag_nom_head", "exec"))
SETTINGS = [("none", 0, 2), ("L2 cap4", 4, 2), ("L3 cap8", 8, 3), ("L3 cap4", 4, 3), ("L3 cap2", 2, 3)]
edges = [0, 256, 1024, 2048, 10**9]
def profile():
    out = []
    for li in (3, 6, 11):
        rows, gg, meta, lvlp = CAP[li]["nom"]
        mp = mps[li]; spec = mp._local_pack_spec
        C, K, nch = meta["chunk"], meta["G"], meta["nch"]
        pos = spec["pos"]; okv = meta["ok"].view(nch, K); rv = rows.view(rows.size(0), nch, K)
        d, co = [], []
        for b in range(rows.size(0)):
            for ch in range(nch):
                live = rv[b, ch][okv[ch]]
                if live.numel() == 0: continue
                t0 = int(pos[min(ch * C, int(pos.numel()) - 1)])
                d.append((t0 - pos[live]).float()); co.append((lvlp[live] > 0).float())
        d = torch.cat(d); co = torch.cat(co)
        out.append(f"L{li}: med {d.median():.0f} far>2k {float((d >= 2048).float().mean()):.2f} summ {float(co.mean()):.2f}")
    return " | ".join(out)
base = None
for name, cap, lvl in SETTINGS:
    for mp in mps:
        mp.local_pack_global_region_cap = cap; mp.local_pack_global_region_level = lvl
    ppl, bk, n = run("normal")
    prof = profile()
    if base is None: base = (ppl, bk)
    rel = " ".join(f"{l} {a/b:.3f}" for l, a, b in zip(LBL, bk, base[1]))
    print(f"{name:8s} ppl {ppl:7.3f} (x{ppl/base[0]:.4f}) | {rel} || {prof}", flush=True)
