"""local_pack_global_nominator: tdesc probe (tiled descent).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_tdesc.py CONFIG [CKPT|-] [L] [B]
Eager layers. CKPT (optional) loads trained weights (strict=False; the indexer stays at init).
Checks:
  1. picks == an independent per-chunk reference (python loops: block representatives and
     effective close rows from the definition, previous-chunk queries, per-query top-v
     votes, per-level beams, recency exclusion), eval mode (no Gumbel);
  2. slot invariants: live slots are L0 rows below the chunk limit, no duplicates per chunk;
  3. gradient isolation: the KL reaches ONLY tdesc_q / tdesc_k, the task loss never does;
  4. KL finite, slot mass per query in (0, 1);
  5. causal: perturbing tokens >= p leaves logits < p bit-unchanged (eval and train);
  6. picks follow content."""
import sys, random, logging, torch
import torch.nn.functional as F
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
import pinball.model.layers.hierarchical_message_passing as H

CFG = sys.argv[1]
CK = sys.argv[2] if len(sys.argv) > 2 else "-"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
B = int(sys.argv[4]) if len(sys.argv) > 4 else 1
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)
def seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); random.seed(s); np.random.seed(s)

cfg = PinballConfig.from_yaml(CFG)
cfg.block_size = L
for k in ("hier_layer_compile", "hier_refresh_compile", "hier_layer_cudagraphs"):
    setattr(cfg, k, False)
inp = resolve_model_inputs(cfg, block_size=L)
seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda()
if CK != "-":
    sd = torch.load(CK, map_location="cpu", weights_only=False)
    mi, un = m.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    print(f"[load] missing {len(mi)} unexpected {len(un)}", mi[:2], un[:2]); del sd
mps = [t.message_passing for t in m.refinement_transformers]
assert all(mp.local_pack_global_nominator == "tdesc" for mp in mps)

REC = []
CAP = {"on": False}
_orig = H.HierarchicalMessagePassing._nominate_by_tdesc
def _rec(self, spec, x_nodes, n):
    out = _orig(self, spec, x_nodes, n)
    if CAP["on"]:
        REC.append((self, spec, x_nodes.detach().clone(), n, out[0].detach().clone(), out[2],
                    [k.detach().clone() for k in out[2].get("tdesc", {}).get("kept", [])]))
    return out
H.HierarchicalMessagePassing._nominate_by_tdesc = _rec

g = torch.Generator().manual_seed(1)
tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
st0 = int(len(tok) * 0.995)
x = torch.stack([tok[st0 + i * L: st0 + (i + 1) * L].clone() for i in range(B)]).cuda()

AC = {"on": True}
def fwd(xx, train=False, s=7, grad=False):
    m.train(train); seed(s)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", torch.bfloat16, enabled=AC["on"]):
        o = m(xx); o = o[0] if isinstance(o, tuple) else o
    return o.float()

fwd(x)
AC["on"] = False                                    # check 1 in fp32: exact, no near-tie rounding
CAP["on"] = True; REC.clear(); fwd(x); CAP["on"] = False
AC["on"] = True
print(f"recorded {len(REC)} selector calls (fp32)")

print("1. picks == reference")
def reference(mp, spec, xn, n, chunks):
    C = int(mp.local_pack_global_chunk); W = int(spec.get("window", 0) or 0)
    lr, pos, perm = spec["level_rows"], spec["pos"], spec["perm"]
    lr0 = lr[0].tolist(); n_tok = len(lr0)
    tl, fl = mp.local_pack_global_tile_level, mp.local_pack_global_tdesc_final_level
    beams, v, R = mp.local_pack_global_tdesc_beams, mp.local_pack_global_tdesc_votes, mp.local_pack_global_tdesc_recent
    lv = {}
    for l in range(fl, tl + 1):
        rows = lr[l].tolist(); t = [int(pos[r]) for r in rows]
        cov, S = t[0] + 1, 2 * (t[1] - t[0])
        reps = []
        for b in range(n_tok // S):
            s0 = b * S
            j = next((i for i, tt in enumerate(t) if tt >= s0 + S - 1), None)
            if j is None or t[j] - cov + 1 > s0:
                reps.append((None, 10 ** 18)); continue
            close = max(rows[j], lr0[min(t[j], n_tok - 1)])
            if l == fl:
                close = max(close, lr0[s0 + S - 1])
            reps.append((int(perm[rows[j]]), close))
        lv[l] = {"S": S, "reps": reps}
    for l in range(fl + 1, tl + 1):                          # effective close: max over subtree
        r = lv[l]["S"] // lv[l - 1]["S"]; lv[l]["r"] = r
        lo = lv[l - 1]["reps"]
        lv[l]["reps"] = [(nd, max([c] + [lo[b * r + i][1] if b * r + i < len(lo) else 10 ** 18 for i in range(r)]))
                         for b, (nd, c) in enumerate(lv[l]["reps"])]
    xd = F.layer_norm(xn.float(), (xn.size(-1),))[0]
    d = mp.local_pack_global_tdesc_dim
    out = {}
    for c in chunks:
        lim = c * C - W
        rows_q = list(range((c - 1) * C, min(c * C, n)))
        with torch.no_grad():
            q = mp.tdesc_q(xd[perm[rows_q]].unsqueeze(0))[0].float()
        nf = len(lv[fl]["reps"])
        a_c = sum(1 for b in range(nf) if lr0[b * lv[fl]["S"] + lv[fl]["S"] - 1] < lim)
        kept_lv = []
        prev = None
        for i, l in enumerate(range(tl, fl - 1, -1)):
            reps = lv[l]["reps"]
            if prev is None:
                cand = [b for b in range(len(reps)) if reps[b][1] < lim]
            else:
                r = lv[l + 1]["r"]
                cand = [b * r + j for b in prev for j in range(r) if b * r + j < len(reps)]
            if l == fl:
                cand = [b for b in cand if b < a_c - min(R, a_c)]
            if not cand:
                kept_lv.append([]); prev = []; continue
            with torch.no_grad():
                kk = mp.tdesc_k[l](xd[[reps[b][0] for b in cand]].unsqueeze(0))[0].float()
            s = (q @ kk.t()) * d ** -0.5                          # [Cq, M], fp32
            votes = torch.zeros(len(cand), device=s.device)
            for qi in range(s.size(0)):
                for j in torch.topk(s[qi], min(v, len(cand))).indices.tolist():
                    votes[j] += 1
            score = votes + 0.5 * torch.softmax(s, -1).sum(0) / C
            order = torch.argsort(score, descending=True).tolist()[:beams[i]]
            prev = [cand[j] for j in order]
            kept_lv.append(prev)
        S_f = lv[fl]["S"]
        rec = [a_c - 1 - i for i in range(min(R, a_c))]
        out[c] = (kept_lv, {lr0[b * S_f + i] for b in prev + rec for i in range(S_f)})
    return out

nchk = exact = lvl_exact = 0
REC1 = list(REC[:3])
CAP["on"] = True; REC.clear(); fwd(x); CAP["on"] = False       # bf16 record for checks 2+
for (mp, spec, xn, n, rows, meta, kept) in REC1:
    C, G, nch = int(meta["chunk"]), int(meta["G"]), int(meta["nch"])
    ok = meta["ok"].view(nch, G)
    chunks = [c for c in np.linspace(2, nch - 1, 5).astype(int).tolist() if bool(ok[c].any())]
    ref = reference(mp, spec, xn, n, chunks)
    for c in chunks:
        got = set(rows[0].view(nch, G)[c][ok[c]].tolist())
        kl_ref, want = ref[c]
        nchk += 1; exact += int(got == want)
        live = meta["tdesc"]["live"]
        lvl_exact += int(all(set(kept[i][0, c, :int(live[i][c])].tolist()) == set(kl_ref[i])
                             for i in range(len(kept))))
rep("final slots == reference", exact == nchk, f"{exact}/{nchk} chunks")
rep("per-level beams == reference", lvl_exact == nchk, f"{lvl_exact}/{nchk} chunks")

print("2. slot invariants")
bad = dup = nl0 = 0
for (mp, spec, xn, n, rows, meta, kept) in REC:
    C, G, nch = int(meta["chunk"]), int(meta["G"]), int(meta["nch"]); W = int(spec.get("window", 0) or 0)
    ok = meta["ok"].view(nch, G); r = rows.view(-1, nch, G)
    lim = (torch.arange(nch, device=r.device) * C - W).view(1, -1, 1)
    bad += int(((r >= lim) & ok.unsqueeze(0)).sum())
    nl0 += int(((spec["levels"][r] != 0) & ok.unsqueeze(0)).sum())
    for b in range(r.size(0)):
        for c in range(0, nch, 5):
            vv = r[b, c][ok[c]]; dup += int(vv.numel() - vv.unique().numel())
rep("live slots below the chunk limit", bad == 0, f"{bad} violations")
rep("live slots are L0 rows", nl0 == 0, f"{nl0} non-L0")
rep("no duplicate slots per chunk", dup == 0, f"{dup} duplicates")
print(f"  live-slot fraction {float(np.mean([float(r_[5]['ok'].float().mean()) for r_ in REC])):.3f}")

print("3. gradient isolation / 4. KL")
TD = lambda nme: ".tdesc_" in nme
def grads():
    return {nme: (p.grad is not None and float(p.grad.abs().max()) > 0) for nme, p in m.named_parameters()
            if p.requires_grad}
m.zero_grad(set_to_none=True)
o = fwd(x, train=True, grad=True)
kl = m._last_nom_kl_loss
st = m._nom_kl_acc[1] / m._nom_kl_acc[2]
rep("KL finite and positive", kl is not None and bool(torch.isfinite(kl)) and float(kl) > 0,
    f"KL {float(kl):.4f} (mean over {m._nom_kl_acc[2]} layers)")
rep("slot mass per query in (0, 1)", 0.0 < float(st[1]) < 1.0, f"{float(st[1]):.3f}")
kl.backward(); gk = grads()
hit = [nme for nme, vv in gk.items() if vv]
rep("KL reaches ONLY the indexer", len(hit) > 0 and all(TD(nme) for nme in hit),
    f"{len(hit)} tensors, non-indexer: {[nme for nme in hit if not TD(nme)][:3]}")
rep("KL reaches every layer's indexer", all(any(f"refinement_transformers.{i}." in nme for nme in hit)
                                            for i in range(len(mps))), "")
m.zero_grad(set_to_none=True)
o = fwd(x, train=True, grad=True)
ce = F.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)), x[:, 1:].reshape(-1))
ce.backward(); gc = grads()
rep("task loss never reaches the indexer", not any(vv for nme, vv in gc.items() if TD(nme)),
    f"{[nme for nme, vv in gc.items() if vv and TD(nme)][:3]}")
m.zero_grad(set_to_none=True)

print("5. causal")
p = L // 2 + 37
x2 = x.clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (B, L - p), generator=g).cuda()
for train in (False, True):
    a = fwd(x, train); a2 = fwd(x, train); b = fwd(x2, train)
    rep(f"{'train' if train else 'eval'} reproducible", (a - a2).abs().max().item() == 0.0,
        f"{(a - a2).abs().max().item():.3e}")
    past = (a[:, :p] - b[:, :p]).abs().max().item(); fut = (a[:, p:] - b[:, p:]).abs().max().item()
    rep(f"causal ({'train' if train else 'eval'})", past == 0.0 and fut > 0.0,
        f"past max|diff| {past:.3e}, future {fut:.3e}")

print("6. picks follow content")
CAP["on"] = True; REC.clear()
xs = x.clone(); xs[:, : L // 4] = x[:, L // 4: L // 2]
fwd(xs); r_new = REC[0][4]; REC.clear(); fwd(x); r_old = REC[0][4]; meta = REC[0][5]; CAP["on"] = False
nch, G = int(meta["nch"]), int(meta["G"]); okm = meta["ok"].view(nch, G)
diff = float(((r_new.view(-1, nch, G) != r_old.view(-1, nch, G)) & okm).float().sum() / okm.sum().clamp(min=1))
rep("picks follow content", diff > 0.05, f"{diff:.2f} of live slots changed")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
