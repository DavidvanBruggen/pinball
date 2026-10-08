"""local_pack_global_nominator: attn probe (attention-guided tile selection).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_tile_attn.py CONFIG [CKPT|-] [L] [B]
CONFIG is a tile arm. Eager layers. CKPT (optional) loads trained q/k (strict=False) so the
attention mass is peaked and the reference comparison is not decided by near-ties. Checks:
  1. picks == an independent per-chunk reference built from the definition (python loops over
     queries of the PREVIOUS chunk, allowed nodes by token rows, overlap-weighted block score,
     top-k, second hop); compared by reference score so exact ties cannot fail it;
  2. slot invariants: live slots are L0 rows, below the chunk limit, no duplicates per chunk;
  3. causal: perturbing tokens >= p leaves logits < p bit-unchanged (eval and train);
  4. the selector is query-conditioned: a different chunk's queries give different picks."""
import sys, random, logging, torch
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
assert all(mp.local_pack_global_nominator == "attn" for mp in mps)

REC = []
_orig = H.HierarchicalMessagePassing._nominate_by_tile_attn
def _rec(self, spec, qp, kp):
    out = _orig(self, spec, qp, kp)
    if CAP["on"]:
        REC.append((self, spec, qp.detach().float(), kp.detach().clone(), out[0].detach().clone(), out[2]))
    return out
H.HierarchicalMessagePassing._nominate_by_tile_attn = _rec
CAP = {"on": False}

g = torch.Generator().manual_seed(1)
tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
st0 = int(len(tok) * 0.995)
x = torch.stack([tok[st0 + i * L: st0 + (i + 1) * L].clone() for i in range(B)]).cuda()

def fwd(xx, train=False, s=7):
    m.train(train); seed(s)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        o = m(xx); o = o[0] if isinstance(o, tuple) else o
    return o.float()

fwd(x)
CAP["on"] = True; REC.clear(); fwd(x); CAP["on"] = False
print(f"recorded {len(REC)} selector calls")

print("1. picks == reference")
def reference(mp, spec, qp, kp, chunks):
    """the selection from its definition: python loops, no shared code but _far_keys"""
    C = int(mp.local_pack_global_chunk); W = int(spec.get("window", 0) or 0)
    lr, pos, lvl = spec["level_rows"], spec["pos"], spec["levels"]
    lr0 = lr[0].tolist(); n_tok = len(lr0)
    l1, k1 = mp.local_pack_global_tile_level, mp.local_pack_global_tile_k
    l2, k2 = mp.local_pack_global_tile_fine_level, mp.local_pack_global_tile_fine_k
    D = qp.size(-1); n = qp.size(1)
    def level(l):
        rows = lr[l].tolist(); t = [int(pos[r]) for r in rows]
        cov = t[0] + 1; S = 2 * (t[1] - t[0])
        return [(r, max(0, tt - cov + 1), tt + 1, max(r, lr0[tt])) for r, tt in zip(rows, t)], S
    def mass(q, rows):
        kf = mp._far_keys(kp[0, rows].unsqueeze(0), lvl[rows]).float()[0]
        return torch.softmax(torch.einsum("qhd,jhd->qhj", q, kf) * D ** -0.5, -1).sum((0, 1))
    ovf = lambda a0, a1, b0, b1: max(0, min(a1, b1) - max(a0, b0)) / max(1, b1 - b0)
    N1, S1 = level(l1)
    if l2 >= 1:
        N2, S2 = level(l2)
    out = {}
    for c in chunks:
        lim = c * C - W
        q = qp[0, (c - 1) * C:min(c * C, n)]
        okn = [nd for nd in N1 if nd[3] < lim]
        ms = mass(q, [nd[0] for nd in okn]).tolist()
        blocks = []
        for bi in range(n_tok // S1):
            s0 = bi * S1
            if lr0[s0 + S1 - 1] >= lim:
                continue
            blocks.append((sum(mj * ovf(s0, s0 + S1, nd[1], nd[2]) for mj, nd in zip(ms, okn)), s0))
        blocks.sort(key=lambda z: -z[0])
        top = blocks[:k1]
        if l2 < 1:
            out[c] = {lr0[s0 + i] for (_, s0) in top for i in range(S1)}; continue
        ent = [(s0, nd) for (_, s0) in top for nd in N2
               if nd[3] < lim and ovf(s0, s0 + S1, nd[1], nd[2]) > 0]
        ms2 = mass(q, [nd[0] for _, nd in ent]).tolist()
        subs = []
        for (_, s0) in top:
            for i in range(S1 // S2):
                a0 = s0 + i * S2
                subs.append((sum(mj * ovf(a0, a0 + S2, nd[1], nd[2])
                                 for mj, (b0, nd) in zip(ms2, ent) if b0 == s0), a0))
        subs.sort(key=lambda z: -z[0])
        out[c] = {lr0[a0 + i] for (_, a0) in subs[:k2] for i in range(S2)}
    return out

nchk = exact = same = 0
for (mp, spec, qp, kp, rows, meta) in REC[:3]:
    C, G, nch = int(meta["chunk"]), int(meta["G"]), int(meta["nch"])
    ok = meta["ok"].view(nch, G)
    chunks = [c for c in np.linspace(1, nch - 1, 6).astype(int).tolist() if bool(ok[c].any())]
    ref = reference(mp, spec, qp, kp, chunks)
    rr, _, _ = _orig(mp, spec, qp.to(torch.bfloat16), kp)    # same inputs, contiguous layout
    for c in chunks:
        got = set(rr[0].view(nch, G)[c][ok[c]].tolist())
        rec = set(rows[0].view(nch, G)[c][ok[c]].tolist())
        nchk += 1; exact += int(got == ref[c]); same += int(rec == got)
rep("picks == reference (same inputs)", exact == nchk, f"{exact}/{nchk} chunks exact")
rep("in-forward picks == recomputed (autocast-independent)", same == nchk, f"{same}/{nchk} chunks")

print("2. slot invariants")
bad = 0; dup = 0; nl0 = 0
for (mp, spec, qp, kp, rows, meta) in REC:
    C, G, nch = int(meta["chunk"]), int(meta["G"]), int(meta["nch"]); W = int(spec.get("window", 0) or 0)
    ok = meta["ok"].view(nch, G); r = rows.view(-1, nch, G)
    lim = (torch.arange(nch, device=r.device) * C - W).view(1, -1, 1)
    bad += int(((r >= lim) & ok.unsqueeze(0)).sum())
    nl0 += int(((spec["levels"][r] != 0) & ok.unsqueeze(0)).sum())
    for b in range(r.size(0)):
        for c in range(0, nch, 7):
            v = r[b, c][ok[c]]
            dup += int(v.numel() - v.unique().numel())
rep("live slots below the chunk limit", bad == 0, f"{bad} violations")
rep("live slots are L0 rows", nl0 == 0, f"{nl0} non-L0")
rep("no duplicate slots per chunk", dup == 0, f"{dup} duplicates")
lf = float(np.mean([float(meta["ok"].float().mean()) for *_, meta in REC]))
print(f"  live-slot fraction {lf:.3f}")

print("3. causal")
p = L // 2 + 37
x2 = x.clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (B, L - p), generator=g).cuda()
for train in (False, True):
    a = fwd(x, train); a2 = fwd(x, train); b = fwd(x2, train)
    rep(f"{'train' if train else 'eval'} reproducible", (a - a2).abs().max().item() == 0.0,
        f"{(a - a2).abs().max().item():.3e}")
    past = (a[:, :p] - b[:, :p]).abs().max().item(); fut = (a[:, p:] - b[:, p:]).abs().max().item()
    rep(f"causal ({'train' if train else 'eval'})", past == 0.0 and fut > 0.0,
        f"past max|diff| {past:.3e}, future {fut:.3e}")

print("4. query-conditioned")
CAP["on"] = True; REC.clear()
xs = x.clone(); xs[:, : L // 4] = x[:, L // 4: L // 2]                  # change early context only
fwd(xs); CAP["on"] = False
r_new = REC[0][4]
CAP["on"] = True; REC.clear(); fwd(x); CAP["on"] = False
r_old = REC[0][4]
meta = REC[0][5]; nch, G = int(meta["nch"]), int(meta["G"])
okm = meta["ok"].view(nch, G)
diff = float(((r_new.view(-1, nch, G) != r_old.view(-1, nch, G)) & okm).float().sum() / okm.sum().clamp(min=1))
rep("picks follow content", diff > 0.05, f"{diff:.2f} of live slots changed")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
