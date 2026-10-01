"""Hierarchy-nominated picks as copy sources. usage: python pick_quality_head.py CONFIG [CKPT|init]"""
import sys, os, logging, torch
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
cfgp, ck = sys.argv[1], sys.argv[2]
cfg = PinballConfig.from_yaml(cfgp); L = int(cfg.block_size)
inp = resolve_model_inputs(cfg, block_size=L)
torch.manual_seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=inp.block_size).cuda().eval()
if ck != "init":
    sd = torch.load(ck, map_location="cuda", weights_only=False)["model_state_dict"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    r = m.load_state_dict(sd, strict=False); print("missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys))
f = open(cfg.text_file, "rb"); f.seek(0, 2); sz = f.tell(); f.seek(sz - 3_000_000); txt = f.read().decode("utf8", "ignore")
ids = inp.tokenizer(txt, return_tensors="pt")["input_ids"][0]
B = 8
starts = torch.linspace(0, len(ids) - L - 1, B).long()
x = torch.stack([ids[s:s + L] for s in starts]).cuda()
caps = {}
for li, t in enumerate(m.refinement_transformers):
    mp = t.message_passing
    def mk(mp, li):
        orig = mp._nominate_by_head
        def spy(spec, lvl_packed, x_nodes):
            r = orig(spec, lvl_packed, x_nodes); caps[li] = (spec, lvl_packed, r, {k: float(v) for k, v in mp._nom_stats.items()}); return r
        return spy
    mp._nominate_by_head = mk(mp, li)
with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
    m(x)
g = torch.Generator(device="cpu").manual_seed(0)
for li, (spec, lvlp, (rows, sc, meta), st) in sorted(caps.items()):
    n = int(spec["num_nodes"]); perm = spec["perm"]; pos = spec["pos"]      # packed token-scale close time
    C, K, nch = meta["chunk"], meta["G"], meta["nch"]; ok = meta["ok"].view(nch, K); W = int(spec["window"])
    allowed = spec[("nom_geom", C, W, K, m.refinement_transformers[li].message_passing.local_pack_global_candidates)][0]
    rows = rows.view(B, nch, K)
    span1 = m._cumulative_window(1)[0]
    tot = dict(t=0, l0=0, l0r=0, co=0, cor=0, nl0=0, nco=0)
    for b in range(B):
        xb = x[b]
        for ch in range(nch):
            live = rows[b, ch][ok[ch]]
            if live.numel() == 0: continue
            q = torch.arange(ch * C, min(n, (ch + 1) * C), device=x.device)
            qt = pos[q][lvlp[q] == 0]; qt = qt[qt < L - 1]
            if qt.numel() == 0: continue
            far = torch.tensor([bool((xb[max(0, int(t) - 127):int(t) + 1] != xb[int(t) + 1]).all()) for t in qt], device=x.device)
            tgt = xb[qt[far] + 1]
            if tgt.numel() == 0: continue
            tot["t"] += int(tgt.numel())
            al = torch.nonzero(allowed[ch]).view(-1)
            al0, alc = al[lvlp[al] == 0], al[lvlp[al] > 0]
            p0, pc = live[lvlp[live] == 0], live[lvlp[live] > 0]
            tot["nl0"] += int(p0.numel()); tot["nco"] += int(pc.numel())
            r0 = al0[torch.randperm(al0.numel(), generator=g)[:p0.numel()].to(x.device)]
            rc = alc[torch.randperm(alc.numel(), generator=g)[:pc.numel()].to(x.device)]
            def toks_l0(r): return xb[pos[r]]
            def toks_co(r):   # tokens inside each picked L1-or-higher window (by its span)
                if r.numel() == 0: return xb[:0]
                sp = torch.tensor([m._cumulative_window(int(l))[0] for l in lvlp[r].tolist()], device=x.device)
                idx = torch.cat([torch.arange(int(e) - int(s_) + 1, int(e) + 1, device=x.device) for e, s_ in zip(pos[r].tolist(), sp.tolist())]).clamp(0, L - 1)
                return xb[idx]
            tot["l0"] += int(torch.isin(tgt, toks_l0(p0)).sum()); tot["l0r"] += int(torch.isin(tgt, toks_l0(r0)).sum())
            tot["co"] += int(torch.isin(tgt, toks_co(pc)).sum()); tot["cor"] += int(torch.isin(tgt, toks_co(rc)).sum())
    T = max(1, tot["t"])
    print(f"layer {li:2d}: slots L0 {tot['nl0']} / summaries {tot['nco']} | far-target coverage: L0 picks {tot['l0']/T:.3f} vs random-L0 {tot['l0r']/T:.3f} | summary windows {tot['co']/T:.3f} vs random {tot['cor']/T:.3f} | gate {st['gate_mean']:.3f}±{st['gate_std']:.3f}")
