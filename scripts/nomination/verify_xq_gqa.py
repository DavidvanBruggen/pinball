"""xq descent with L1 stop + salience L0 + grouped (MQA/GQA) far read: probe.

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_xq_gqa.py CONFIG [CKPT|-] [L] [B]
Eager layers. CKPT (optional) loads trained weights strict=False; the new salience heads,
alpha, boost u and shared far K/V get nonzero test values so every path is live. Checks:
  1. L0 picks == the children of the query's own L1 picks (deduped), causal and >= 128 back,
     ranked by available salience (eval: no noise) -- an independent loop;
  2. boosted-copy encoding: a pick reads the boosted row (id >= N) iff its salience is
     available before the query;
  3. the grouped read == a dense fp32 reference (per query and head: its picks' shared keys
     and values from the read's own source rows, + the sink; then the out projection);
  4. causal: perturbing tokens >= p leaves logits < p bit-unchanged (eval and train);
  5. gradients: the task loss reaches the shared far K/V, the salience heads and u, not the
     indexer or alpha; the indexer KL reaches the indexer and alpha, not the salience heads,
     u or the far K/V."""
import sys, random, logging, torch
import numpy as np
logging.disable(logging.WARNING)
import torch._functorch.config as _fc; _fc.donated_buffer = False   # check 5 takes two grads of one graph
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
    print(f"[load] missing {len(mi)} {mi[:3]} unexpected {len(un)}"); del sd
assert m.xq_far_kv_groups > 0 and m.xq_descent_final_level >= 1
with torch.no_grad():
    if m.xq_salience_prior: m.xq_sal_alpha.fill_(0.5)
    if m.xq_salience_boost: m.xq_far_boost_u.normal_(0, 0.2)
    for t_ in m.refinement_transformers:
        mp = t_.message_passing
        mp.xq_far_level_k.normal_(0, 0.05); mp.xq_far_level_v.normal_(0, 0.05)

REC_N, REC_R = [], []
_on = M.HierarchicalFlowGAT._xq_descent_nominate
def rec_nom(self, x, lo, t):
    out = _on(self, x, lo, t)
    if CAP["on"] and out is not None:
        r = self._xq_desc_record
        REC_N.append((out[1].clone(), [(l, k.clone(), None if p is None else p.clone()) for l, k, p in r["levels"]],
                      list(r["o"]), t.clone(), {l: w.clone() for l, w in r["sal"][0].items()},
                      {l: a.clone() for l, a in r["sal"][1].items()}, r["allow_same"]))
    return out
M.HierarchicalFlowGAT._xq_descent_nominate = rec_nom
_or = H.HierarchicalMessagePassing._compute_hqd_packed_l0_attn
def rec_read(self, q, k, v, dst_nodes, candidate_nodes, num_nodes, B):
    out = _or(self, q, k, v, dst_nodes, candidate_nodes, num_nodes, B)
    if CAP["on"] and out is not None:
        REC_R.append((self, q.detach().float(), k.detach().float(), v.detach().float(),
                      candidate_nodes.clone(), int(num_nodes), out.detach().float()))
    return out
H.HierarchicalMessagePassing._compute_hqd_packed_l0_attn = rec_read
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
CAP["on"] = True; REC_N.clear(); REC_R.clear(); fwd(x); CAP["on"] = False
print(f"recorded {len(REC_N)} rounds, {len(REC_R)} reads")

print("1. L0 picks == children of the own L1 picks, ranked by available salience")
tabs = m._xq_tree_tables([REC_N[0][2][i + 1] - REC_N[0][2][i] for i in range(len(REC_N[0][2]) - 1)], x.device)
bad = n_ = 0
for cand, levels, o, t, w, av, allow_same in REC_N[:2]:
    kept = {l: k for l, k, _ in levels}
    K0 = int(kept[0].size(-1))
    for b in range(cand.size(0)):
        for qi in (300, 2049, 7777, 12000, o[1] - o[0] - 1):
            l1 = [int(v) for v in kept[1][b, qi].tolist() if v >= 0]
            ch = sorted({int(c) for p in l1 for c in tabs[1][p].tolist() if c >= 0})
            qt = int(t[o[0] + qi])
            ok = [c for c in ch if (int(t[o[0] + c]) <= qt if allow_same else int(t[o[0] + c]) < qt)
                  and c <= qi - int(m.xq_descent_local_exclude)]
            sc = {c: (float(w[0][b, c]) if (int(av[0][c]) <= qt if allow_same else int(av[0][c]) < qt) else 0.0) for c in ok}
            want = sorted(sorted(ok, key=lambda c: -sc[c])[:K0])
            got = sorted(int(v) for v in kept[0][b, qi].tolist() if v >= 0)
            if len(ok) > K0:                     # exact ties at the cut can swap: compare scores
                same = sorted(sc[c] for c in got) == sorted(sc[c] for c in want)
            else:
                same = got == want
            bad += int(not same); n_ += 1
rep("L0 picks == reference", bad == 0, f"{n_ - bad}/{n_} queries")

print("2. boosted-copy encoding")
bad = tot = 0
for cand, levels, o, t, w, av, allow_same in REC_N[:2]:
    Nn = o[-1]
    BIGT = torch.iinfo(torch.long).max // 4
    af = torch.cat([av[l] if l in av else torch.full((o[l + 1] - o[l],), BIGT, device=t.device) for l in range(len(o) - 1)])
    qt = t[o[0]:o[1]].view(1, -1, 1)
    node = torch.where(cand >= Nn, cand - Nn, cand)
    valid = cand >= 0
    want_b = valid & ((af[node.clamp(min=0)] <= qt) if allow_same else (af[node.clamp(min=0)] < qt))
    got_b = cand >= Nn
    bad += int(((want_b != got_b) & valid).sum()); tot += int(valid.sum())
rep("pick reads the boosted row iff salience available", bad == 0, f"{bad} of {tot} picks wrong")

print("3. grouped read == dense reference")
maxd = sc_ = 0.0; nq = 0
for (mp, q, k, v, cand, N_, out) in REC_R[:2]:
    G, D = k.size(2), k.size(3); Hh = q.size(2); r = Hh // G
    sink = mp.hqd_read_sink_k.detach().float()
    o0 = int(mp._xq_dst_range[0])
    for b in range(q.size(0)):
        for qi in (300, 5000, 12000, int(cand.size(1)) - 1):
            ids = [int(c) for c in cand[b, qi].tolist() if c >= 0]
            Kt = k[b, ids]; Vt = v[b, ids]                                     # [n, G, D]
            qv = q[b, o0 + qi]                                                 # [H, D]
            msg = []
            for h in range(Hh):
                g = h // r
                s = torch.cat([Kt[:, g] @ qv[h], (sink[h] @ qv[h]).view(1)]) / D ** 0.5
                p = torch.softmax(s, 0)[:-1]
                msg.append(p @ Vt[:, g])
            msg = torch.stack(msg).reshape(1, Hh * D)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                ref = mp.sparse_out_proj(msg).float()[0]
            got = out[b, o0 + qi]
            maxd = max(maxd, (ref - got).abs().max().item()); sc_ = max(sc_, ref.abs().max().item()); nq += 1
rep("grouped read == dense fp32 reference", maxd <= 0.02 * max(sc_, 1e-3) + 2e-3, f"{nq} queries, max|diff| {maxd:.2e} (|ref| {sc_:.2e})")

print("4. causal")
g = torch.Generator().manual_seed(1)
p = L // 2 + 37
x2 = x.clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (B, L - p), generator=g).cuda()
for train in (False, True):
    a = fwd(x, train); a2 = fwd(x, train); bb = fwd(x2, train)
    rep(f"{'train' if train else 'eval'} reproducible", (a - a2).abs().max().item() == 0.0, f"{(a - a2).abs().max().item():.3e}")
    past = (a[:, :p] - bb[:, :p]).abs().max().item(); fut = (a[:, p:] - bb[:, p:]).abs().max().item()
    rep(f"causal ({'train' if train else 'eval'})", past == 0.0 and fut > 0.0, f"past {past:.3e}, future {fut:.3e}")

print("5. gradient routing")
m.train(); seed(3)
with torch.autocast("cuda", torch.bfloat16):
    o = m(x[:, :L]); o = o[0] if isinstance(o, tuple) else o
    task = torch.nn.functional.cross_entropy(o[0, :-1].float(), x[0, 1:L])
kl = m._last_xq_index_loss
groups = {"indexer": [p_ for n, p_ in m.named_parameters() if n.startswith("xq_index_")],
          "alpha": [m.xq_sal_alpha] if m.xq_salience_prior else [],
          "salience": [p_ for n, p_ in m.named_parameters() if n.startswith("xq_sal_q") or n.startswith("xq_sal_k")],
          "boost_u": [m.xq_far_boost_u] if m.xq_salience_boost else [],
          "far_kv": [p_ for n, p_ in m.named_parameters() if ".xq_far_k." in n or ".xq_far_v." in n]}
def gnorm(loss, ps):
    gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    return sum(float(gg.float().norm()) for gg in gs if gg is not None)
gt = {k_: gnorm(task, v_) for k_, v_ in groups.items() if v_}
gk = {k_: gnorm(kl, v_) for k_, v_ in groups.items() if v_} if kl is not None else {}
print(f"  task-loss grad norms {gt}\n  indexer-KL grad norms {gk}")
rep("task loss -> far K/V + salience + u, not indexer/alpha",
    gt.get("far_kv", 0) > 0 and gt.get("salience", 1) > 0 and gt.get("boost_u", 1) > 0
    and gt.get("indexer", 0) == 0 and gt.get("alpha", 0) == 0)
rep("KL -> indexer + alpha only", kl is not None and gk.get("indexer", 0) > 0 and gk.get("alpha", 1) > 0
    and gk.get("salience", 0) == 0 and gk.get("boost_u", 0) == 0 and gk.get("far_kv", 0) == 0,
    f"KL {float(kl) if kl is not None else None}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
