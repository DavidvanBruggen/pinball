"""Nomination diagnostics on a trained checkpoint: gates, slot attention mass, and
val CE per recurrence bucket under normal / random-picks / no-slots.
usage: python diag_nom.py CONFIG CKPT"""
import sys, math, logging, torch, torch.nn.functional as F
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
cfgp, ck = sys.argv[1], sys.argv[2]
cfg = PinballConfig.from_yaml(cfgp); L = int(cfg.block_size)
import os
if os.environ.get("REGCAP") is not None:
    cfg.local_pack_global_region_cap = int(os.environ["REGCAP"])
if os.environ.get("REGLEVEL") is not None:
    cfg.local_pack_global_region_level = int(os.environ["REGLEVEL"])
inp = resolve_model_inputs(cfg, block_size=L)
torch.manual_seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=inp.block_size).cuda().eval()
sd = torch.load(ck, map_location="cuda", weights_only=False)
print("ckpt epoch", sd.get("epoch"), "|", cfgp.split("/")[-1])
r = m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd["model_state_dict"].items()}, strict=False)
assert not r.missing_keys and not r.unexpected_keys, r
del sd
from pathlib import Path
toks = torch.load(Path(cfg.text_file).with_suffix(".pt"), mmap=True)
val = toks[int(len(toks) * (1 - float(getattr(cfg, "val_split", 0.01)))):]
g = torch.Generator().manual_seed(1234)
NSEQ, B = 32, 4
starts = torch.randint(0, len(val) - L - 1, (NSEQ,), generator=g)
X = torch.stack([val[s:s + L] for s in starts.tolist()]).long()
MODE = {"m": "normal"}
CAP = {}
mps = [t.message_passing for t in m.refinement_transformers]
for li, mp in enumerate(mps):
    def mk(mp, li):
        orig_nom, orig_attn = mp._nominate_by_head, mp._flex_union_attn
        def nom(spec, lvlp, xn):
            if MODE["m"] == "noisy":
                mp.training = True
                try:
                    rows, gg, meta = orig_nom(spec, lvlp, xn)
                finally:
                    mp.training = False
            else:
                rows, gg, meta = orig_nom(spec, lvlp, xn)
            if MODE["m"] == "noslots":
                # remove the slots from the softmax via the MASK: all slots dead, gate/boost 0.
                # The cached BlockMask key does not include ok, so drop it before and after.
                meta = dict(meta, ok=torch.zeros_like(meta["ok"]))
                gg = torch.full_like(gg, torch.finfo(gg.dtype).min)
                if mp.local_pack_global_boost == "key":
                    gg = torch.where(meta["ok"].view(1, -1), gg, mp.nom_gate_bias.detach().expand_as(gg).to(gg.dtype) - 30.0)
                spec.pop("flex_block_mask", None); spec.pop("flex_block_mask_key", None)
            elif MODE.get("was_noslots"):
                spec.pop("flex_block_mask", None); spec.pop("flex_block_mask_key", None)
                MODE["was_noslots"] = False
            elif MODE["m"] == "random":
                C, K, nch = meta["chunk"], meta["G"], meta["nch"]; W = int(spec["window"])
                allowed = next(v for k_, v in spec.items() if isinstance(k_, tuple) and k_[:1] == ("nom_geom",))[0]
                Bn, n = xn.size(0), int(spec["num_nodes"])
                rnd = torch.rand(Bn, nch, n, device=xn.device).masked_fill(~allowed.unsqueeze(0), -1.0)
                rows = torch.topk(rnd, K, dim=-1).indices.reshape(Bn, nch * K)
                w = mp._nomination_weights(spec, lvlp, xn)
                gg = w.gather(1, rows)
                if mp.local_pack_global_nom_gate == "zscore":
                    am = allowed.to(w.dtype); cnt = am.sum(1).clamp(min=1.0)
                    mu = (w.unsqueeze(1) * am).sum(-1) / cnt
                    sdv = ((((w.unsqueeze(1) - mu.unsqueeze(-1)) ** 2) * am).sum(-1) / cnt + 1e-6).sqrt()
                    gg = ((gg.view(Bn, nch, K) - mu.unsqueeze(-1)) / sdv.unsqueeze(-1)).reshape(Bn, -1) + mp.nom_gate_bias
                gg = torch.where(meta["ok"].view(1, -1), gg, torch.full_like(gg, torch.finfo(gg.dtype).min))
            CAP.setdefault(li, {})["nom"] = (rows, gg, meta, lvlp)
            return rows, gg, meta
        def attn(qp, kp, vp, spec, causal=True, gsel=None):
            out = orig_attn(qp, kp, vp, spec, causal, gsel=gsel)
            if MODE.get("cap"):
                CAP.setdefault(li, {})["attn"] = (qp[:2].float(), kp[:2].float(), vp[:2].float(), spec, gsel)
            return out
        return nom, attn
    mp._nominate_by_head, mp._flex_union_attn = mk(mp, li)
EDGES = (128, 512, 2048)
LBL = ("never", "<128", "128-511", "512-2k", ">2k")
def bucket(d):
    if d is None: return 0
    for i, e in enumerate(EDGES):
        if d < e: return i + 1
    return len(EDGES) + 1
def run(mode):
    MODE["m"] = mode
    acc = [[0.0, 0] for _ in LBL]; tot = [0.0, 0]
    for i in range(0, NSEQ, B):
        x = X[i:i + B].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
            out = m(x)
        lg = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
        nll = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none").view(x.size(0), -1).cpu()
        tot[0] += float(nll.sum()); tot[1] += nll.numel()
        for b in range(x.size(0)):
            seq = x[b].tolist(); last = {}
            for j, tid in enumerate(seq):
                if j >= 1:
                    p = last.get(tid); bi = bucket(j - p if p is not None else None)
                    acc[bi][0] += float(nll[b, j - 1]); acc[bi][1] += 1
                last[tid] = j
    return math.exp(tot[0] / tot[1]), [math.exp(s / max(c, 1)) for s, c in acc], [c for _, c in acc]
res = {}
import os
MODES = os.environ.get("MODES", "normal,random,noslots").split(",")
for mode in MODES:
    if mode != "noslots" and MODE.get("m") == "noslots":
        MODE["was_noslots"] = True
    if mode == "nologit":
        for mp in mps: mp.local_pack_global_logit = False
        res[mode] = run("normal")
        for mp in mps: mp.local_pack_global_logit = True
        ppl, bk, n = res[mode]
        print(f"{mode:8s} ppl {ppl:7.3f} | " + " ".join(f"{l} {v:7.2f}" for l, v in zip(LBL, bk)))
        continue
    res[mode] = run(mode)
    ppl, bk, n = res[mode]
    print(f"{mode:8s} ppl {ppl:7.3f} | " + " ".join(f"{l} {v:7.2f}" for l, v in zip(LBL, bk)))
print("counts", dict(zip(LBL, res['normal'][2])))
for mode in [x for x in MODES if x != "normal"]:
    print(f"{mode}/normal: overall {res[mode][0]/res['normal'][0]:.4f} | " + " ".join(f"{l} {a/b:.4f}" for l, a, b in zip(LBL, res[mode][1], res['normal'][1])))
if os.environ.get("NOATTN"): sys.exit(0)
# gates + slot attention mass (normal picks, first batch, 2 seqs, sampled queries)
MODE["m"] = "normal"; MODE["cap"] = True
with torch.no_grad():
    m(X[:B].cuda())          # fp32 for the mass readout
MODE["cap"] = False
print("layer | gate mean±std  bias | summary share | attn mass on: static  slots(raw)  slots(gated)  band")
gq = torch.Generator().manual_seed(7)
for li, mp in enumerate(mps):
    rows, gg, meta, lvlp = CAP[li]["nom"]
    qp, kp, vp, spec, gsel = CAP[li]["attn"]
    ok = meta["ok"]; live = ok.view(1, -1).expand(rows.size(0), -1)
    sg = torch.sigmoid(gg.clamp(min=-60).float())
    gm, gs = float(sg[live].mean()), float(sg[live].std())
    cof = float((lvlp.index_select(0, rows.reshape(-1)).view(rows.shape)[live] > 0).float().mean())
    C, K, nch = meta["chunk"], meta["G"], meta["nch"]; W = int(spec["window"]); n = int(spec["num_nodes"])
    st = meta["static_rows"]; sset = torch.zeros(n, dtype=torch.bool, device=qp.device); sset[st] = True
    D = qp.size(-1); mass = torch.zeros(4)
    rv = rows.view(rows.size(0), nch, K); okv = ok.view(nch, K); gv = gg.view(gg.size(0), nch, K)
    for b in range(2):
        for qi in torch.randint(W + C, n, (24,), generator=gq).tolist():
            ch = qi // C
            sr = st[st <= qi]; sl = rv[b, ch][okv[ch]]; gl = gv[b, ch][okv[ch]].float()
            band = torch.arange(max(0, qi - W), qi + 1, device=qp.device); band = band[~sset[band]]
            kr = torch.cat([sr, sl, band])
            kk = kp[b, kr].clone()
            if mp.local_pack_global_boost == "key":
                bias = torch.zeros(len(kr), device=qp.device)
                if len(sl):
                    zz = gl - float(mp.nom_gate_bias)
                    H_, D_ = mp.nom_boost_u.shape
                    ur = mp.rotary_pos_enc.apply_rotary_pos_emb(mp.nom_boost_u.float().view(1, H_, D_).expand(len(sl), -1, -1).contiguous(), spec["pos"][sl])
                    kk[len(sr):len(sr) + len(sl)] += zz.view(-1, 1, 1) * ur
            else:
                bias = torch.cat([torch.zeros(len(sr), device=qp.device), gl, torch.zeros(len(band), device=qp.device)])
            p = (torch.einsum("hd,khd->hk", qp[b, qi], kk) / math.sqrt(D) + bias.view(1, -1)).softmax(-1).mean(0)
            a, s_ = len(sr), len(sl)
            mass += torch.tensor([float(p[:a].sum()), float(p[a:a + s_].sum()), float((p[a:a + s_] * torch.sigmoid(gl)).sum()), float(p[a + s_:].sum())])
    mass /= 48
    bias_v = float(mp.nom_gate_bias) if hasattr(mp, "nom_gate_bias") else float("nan")
    print(f"L{li:2d} | {gm:.3f}±{gs:.3f} {bias_v:+.2f} | {cof:.2f} | {mass[0]:.3f} {mass[1]:.3f} {mass[2]:.3f} {mass[3]:.3f}")

if os.environ.get("QU"):
    print("q.u boost (extra score per slot, head-mean): layer | |u| | mean boost on live slots by TARGET bucket of the query token")
    x0 = X[:2]
    for li, mp in enumerate(mps):
        if not hasattr(mp, "nom_boost_u"):
            continue
        rows, gg, meta, lvlp = CAP[li]["nom"]
        qp, kp, vp, spec, gsel = CAP[li]["attn"]
        C, K, nch = meta["chunk"], meta["G"], meta["nch"]; n = int(spec["num_nodes"])
        pos = spec["pos"]; okv = meta["ok"].view(nch, K)
        rv = rows.view(rows.size(0), nch, K); gv = gg.view(gg.size(0), nch, K)
        H, D = mp.nom_boost_u.shape
        acc = [[0.0, 0] for _ in LBL]
        for b in range(2):
            seq = x0[b].tolist(); last = {}; tb = {}
            for j, tid in enumerate(seq):
                if j >= 1:
                    pv = last.get(tid); tb[j - 1] = bucket(j - pv if pv is not None else None)
                last[tid] = j
            l0rows = torch.nonzero(lvlp == 0).view(-1)
            for qi in l0rows[torch.randperm(l0rows.numel(), generator=gq)[:400]].tolist():
                t = int(pos[qi]); ch = qi // C
                if t not in tb or not okv[ch].any(): continue
                sl = rv[b, ch][okv[ch]]; z = gv[b, ch][okv[ch]].float() - float(mp.nom_gate_bias)
                ur = mp.rotary_pos_enc.apply_rotary_pos_emb(mp.nom_boost_u.float().view(1, H, D).expand(len(sl), -1, -1).contiguous(), pos[sl])
                boost = (z.view(-1, 1) * torch.einsum("hd,shd->sh", qp[b, qi], ur) / math.sqrt(D)).mean()
                acc[tb[t]][0] += float(boost); acc[tb[t]][1] += 1
        print(f"L{li:2d} | {float(mp.nom_boost_u.norm()):.3f} | " + " ".join(f"{l} {s_/max(c,1):+.3f}(n{c})" for l, (s_, c) in zip(LBL, acc)))
if os.environ.get("DIST"):
    MODE["m"] = "normal"
    with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
        m(X[:B].cuda())
    print("pick distance from chunk start (tokens), nominated vs random allowed | share by distance band")
    edges = [0, 256, 1024, 2048, 10**9]
    gd = torch.Generator().manual_seed(3)
    for li in (0, 3, 6, 9, 11):
        rows, gg, meta, lvlp = CAP[li]["nom"]
        spec = CAP[li]["attn"][3] if "attn" in CAP[li] else None
        C, K, nch = meta["chunk"], meta["G"], meta["nch"]
        allowed = next(v for k_, v in m.refinement_transformers[li].message_passing._local_pack_spec.items() if isinstance(k_, tuple) and k_[:1] == ("nom_geom",))[0]
        pos = m.refinement_transformers[li].message_passing._local_pack_spec["pos"]
        okv = meta["ok"].view(nch, K); rv = rows.view(rows.size(0), nch, K)
        dn, dr = [], []
        for b in range(rows.size(0)):
            for ch in range(nch):
                live = rv[b, ch][okv[ch]]
                if live.numel() == 0: continue
                t0 = int(pos[min(ch * C, int(pos.numel()) - 1)])
                al = torch.nonzero(allowed[ch]).view(-1)
                rnd = al[torch.randperm(al.numel(), generator=gd)[:live.numel()].to(al.device)]
                dn.append(t0 - pos[live]); dr.append(t0 - pos[rnd])
        dn, dr = torch.cat(dn).float(), torch.cat(dr).float()
        band = lambda d: [round(float(((d >= a) & (d < b_)).float().mean()), 2) for a, b_ in zip(edges[:-1], edges[1:])]
        print(f"L{li:2d}: median nominated {dn.median():.0f} vs random {dr.median():.0f} | nominated {band(dn)} random {band(dr)}  (bands <256, 256-1k, 1k-2k, >2k)")
