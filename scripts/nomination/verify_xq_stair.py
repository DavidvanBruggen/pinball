"""xq_descent_read: staircase probe (per-query descent read as tile unions through flex).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_xq_stair.py CONFIG [CKPT|-] [L] [B]
Eager layers (flex itself compiled). CKPT (optional) loads trained weights strict=False; the
salience heads / alpha / boost u are new and get set to nonzero test values so every path is
live. Checks:
  1. union, first voter, staircase boundary n_i, overflow drop and per-pick slot == an
     independent python loop over the descent's candidates (definition, not the sort code);
  2. the read == a dense fp32 reference built from the definition: per query, softmax over
     its tile's slots whose first voter is <= it, the valid recency tokens and the sink
     (boosted keys, slow pairs at node close times), then the out projection;
  3. causal: perturbing tokens >= p leaves logits < p bit-unchanged (eval, and train with
     Gumbel noise and the KL query sample);
  4. gradients: the task loss reaches the salience heads and u (boost) but not the indexer
     or alpha; the indexer KL reaches the indexer and alpha but not the salience heads or u;
  5. union / overflow stats."""
import sys, random, logging, torch
import numpy as np
logging.disable(logging.WARNING)
import torch._functorch.config as _fc; _fc.donated_buffer = False   # check 4 takes two grads of one graph
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
import pinball.model.layers.hierarchical_message_passing as H
import pinball.model.hierarchical_flow_gat_cached_batch as M

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
    print(f"[load] missing {len(mi)} {mi[:4]} unexpected {len(un)}"); del sd
assert m.xq_descent_read == "staircase"
with torch.no_grad():                       # make the new paths live (they init at 0)
    if m.xq_salience_prior: m.xq_sal_alpha.fill_(0.5)
    if m.xq_salience_boost: m.xq_stair_boost_u.normal_(0, 0.2)

REC_B, REC_R = [], []
_ob = M.HierarchicalFlowGAT._xq_stair_build
def rec_build(self, cand_abs, o, t, sal, round_idx):
    st = _ob(self, cand_abs, o, t, sal, round_idx)
    if CAP["on"]:
        ent = self._xq_stair_cache[("xq_stair_bm", int(cand_abs.size(0)), st["nT"], st["Gt"], int(round_idx))]
        REC_B.append((cand_abs.clone(), list(o), t.clone(), st, ent[0].clone(), ent[1].clone()))
    return st
M.HierarchicalFlowGAT._xq_stair_build = rec_build
_orr = H.HierarchicalMessagePassing._xq_staircase_read
def rec_read(self, q, k, v, num_nodes, B_, st):
    out = _orr(self, q, k, v, num_nodes, B_, st)
    if CAP["on"]:
        REC_R.append((self, q.detach().float(), k.detach().float(), v.detach().float(), st, out.detach().float()))
    return out
H.HierarchicalMessagePassing._xq_staircase_read = rec_read
CAP = {"on": False}

tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
st0 = int(len(tok) * 0.995)
x = torch.stack([tok[st0 + i * L: st0 + (i + 1) * L].clone() for i in range(B)]).cuda()
def fwd(xx, train=False, s=7):
    m.train(train); seed(s)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        o = m(xx); o = o[0] if isinstance(o, tuple) else o
    return o.float()

fwd(x)
CAP["on"] = True; REC_B.clear(); REC_R.clear(); fwd(x); CAP["on"] = False
print(f"recorded {len(REC_B)} rounds, {len(REC_R)} reads")

print("1. union / staircase == loop reference")
T, G, R = m.xq_descent_tile, m.xq_descent_union_cap, m.xq_descent_recent
bad = {"rows": 0, "nlim": 0, "slot": 0, "rec": 0}; ntile = 0; nover = 0
for cand, o, t, st, nlim, rec in REC_B[:2]:
    o0 = o[0]; Gt = st["Gt"]; nT = st["nT"]
    rows = st["rows"].view(cand.size(0), nT, Gt)
    for b in range(cand.size(0)):
        for c in sorted({1, 2, nT // 3, nT // 2, nT - 1}):
            ntile += 1
            s = c * T
            slots, seen = [], set()
            for i in range(T):
                qi = s + i
                if qi >= cand.size(1): break
                for node in sorted(int(v) for v in cand[b, qi].tolist() if v >= 0):
                    if o0 <= node < o[1] and node - o0 >= s - R:
                        continue                                  # held by the recency slots
                    if node not in seen:
                        seen.add(node); slots.append((i, node))
            nover += int(len(slots) > G)
            kept = slots[:G]
            got = rows[b, c, :len(kept)].tolist()
            bad["rows"] += int(got != [nd for _, nd in kept])
            want_n = [sum(1 for fvv, _ in kept if fvv <= i) for i in range(T)]
            bad["nlim"] += int(nlim[b, s:s + T].tolist() != want_n)
            pos = {nd: j for j, (_, nd) in enumerate(kept)}
            for i in range(min(T, cand.size(1) - s)):
                for kk, node in enumerate(cand[b, s + i].tolist()):
                    if node < 0: want = -1
                    elif o0 <= node < o[1] and node - o0 >= s - R: want = G + (node - o0) - (s - R)
                    else: want = pos.get(node, -1)
                    bad["slot"] += int(int(st["pick_slot"][b, s + i, kk]) != want)
            want_rec = [o0 + max(0, s - R + j) for j in range(R)]
            bad["rec"] += int(rows[b, c, G:G + R].tolist() != want_rec) + int(int(rec[c]) != min(R, max(0, R - s)))
rep("union rows / boundaries / pick slots / recency == loop", sum(bad.values()) == 0,
    f"{ntile} tiles ({nover} overflowing), mismatches {bad}")

print("2. read == dense reference")
maxd = 0.0; scale_ = 0.0; nq = 0
for (mp, q, k, v, st, out) in REC_R[:2]:
    cand_rec = next(r for r in REC_B if r[3] is st or r[3]["rows"].data_ptr() == st["rows"].data_ptr())
    cand, o, t, _, nlim, rec = cand_rec
    o0, Gt, nT = st["o0"], st["Gt"], st["nT"]
    Hh, D, Dv = q.size(2), q.size(3), v.size(3)
    sp = int(mp.local_pack_far_slow_pairs); d0 = D - int(mp.local_pack_far_nope_dims); sl = int(mp.local_pack_far_slow_len)
    rows = st["rows"].view(-1, nT, Gt)
    bnode, bok, u = st.get("boost_node"), st.get("boost_ok"), st.get("u")
    sink = mp.hqd_read_sink_k.detach().float()
    W_out = mp.sparse_out_proj
    for b in range(q.size(0)):
        for c in (2, nT // 2, nT - 1):
            s = c * T
            for i in (0, 5, 77, T - 1):
                qi = s + i
                if qi >= st["n0"]: continue
                # visible slot indices from the DEFINITION: union slots with first voter <= i
                vis = list(range(int(nlim[b, qi]))) + [G + j for j in range(R) if s - R + j >= 0]
                kv = []
                for j in vis:
                    node = int(rows[b, c, j])
                    kj = k[b, node].clone()
                    if bnode is not None and u is not None and bool(bok.view(-1, nT, Gt)[b, c, j]):
                        kj = kj + float(bnode[b, node]) * u.detach().float()   # boosted iff available before the tile
                    kv.append((kj, v[b, node].clone(), float(t[node])))
                qv = q[b, o0 + qi].clone()
                if sp:
                    qv = H._slow_rope(qv.unsqueeze(0), torch.tensor([float(t[o0 + qi])], device=qv.device).view(1, 1), d0, sp, sl)[0]
                    kv = [(H._slow_rope(kj.unsqueeze(0), torch.tensor([tp], device=kj.device).view(1, 1), d0, sp, sl)[0], vj, tp)
                          for kj, vj, tp in kv]
                Kt = torch.stack([kj for kj, _, _ in kv] + [sink])           # [n+1, H, D]
                Vt = torch.stack([vj for _, vj, _ in kv] + [torch.zeros_like(v[b, 0])])
                sc = torch.einsum("hd,nhd->hn", qv, Kt) / D ** 0.5
                msg = torch.einsum("hn,nhd->hd", torch.softmax(sc, -1), Vt).reshape(1, Hh * Dv)
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    ref = W_out(msg).float()[0]            # the read's own output projection
                got = out[b, o0 + qi]
                maxd = max(maxd, (ref - got).abs().max().item()); scale_ = max(scale_, ref.abs().max().item()); nq += 1
rep("staircase read == dense fp32 reference", maxd <= 0.02 * max(scale_, 1e-3) + 2e-3,
    f"{nq} queries, max|diff| {maxd:.2e} (|ref| max {scale_:.2e})")

print("3. causal")
g = torch.Generator().manual_seed(1)
p = L // 2 + 37
x2 = x.clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (B, L - p), generator=g).cuda()
for train in (False, True):
    a = fwd(x, train); a2 = fwd(x, train); bb = fwd(x2, train)
    rep(f"{'train' if train else 'eval'} reproducible", (a - a2).abs().max().item() == 0.0, f"{(a - a2).abs().max().item():.3e}")
    past = (a[:, :p] - bb[:, :p]).abs().max().item(); fut = (a[:, p:] - bb[:, p:]).abs().max().item()
    rep(f"causal ({'train' if train else 'eval'})", past == 0.0 and fut > 0.0, f"past max|diff| {past:.3e}, future {fut:.3e}")

print("4. gradient routing")
m.train(); seed(3)
with torch.autocast("cuda", torch.bfloat16):
    o = m(x[:, :L]); o = o[0] if isinstance(o, tuple) else o
    task = torch.nn.functional.cross_entropy(o[0, :-1].float(), x[0, 1:L])
kl = m._last_xq_index_loss
groups = {"indexer": [p_ for n, p_ in m.named_parameters() if n.startswith("xq_index_")],
          "alpha": [m.xq_sal_alpha] if m.xq_salience_prior else [],
          "salience": [p_ for n, p_ in m.named_parameters() if n.startswith("xq_sal_q") or n.startswith("xq_sal_k")],
          "boost_u": [m.xq_stair_boost_u] if m.xq_salience_boost else []}
def gnorm(loss, ps):
    gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    return sum(float(gg.float().norm()) for gg in gs if gg is not None)
gt = {k_: gnorm(task, v_) for k_, v_ in groups.items() if v_}
gk = {k_: gnorm(kl, v_) for k_, v_ in groups.items() if v_} if kl is not None else {}
print(f"  task-loss grad norms {gt}\n  indexer-KL grad norms {gk}")
rep("task loss -> salience + u, not indexer/alpha",
    gt.get("salience", 1) > 0 and gt.get("boost_u", 1) > 0 and gt.get("indexer", 0) == 0 and gt.get("alpha", 0) == 0)
rep("KL -> indexer + alpha, not salience/u", kl is not None and gk.get("indexer", 0) > 0 and gk.get("alpha", 1) > 0
    and gk.get("salience", 0) == 0 and gk.get("boost_u", 0) == 0, f"KL {float(kl) if kl is not None else None}")
print("5. stats:", getattr(m, "_last_xq_stair_stats", None))
print("PASS" if not FAILS else f"FAIL: {FAILS}")
