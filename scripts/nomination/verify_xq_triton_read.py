"""xq_read_kernel: triton vs gather on a trained descent checkpoint (compiled layers, as trained).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_xq_triton_read.py CONFIG CKPT [NSEQ]
  1. eval: logits and per-sequence NLL, triton vs gather (bf16 noise level expected);
  2. train step (fixed seed, B=2): task loss, indexer KL, gradient cosine per parameter group;
  3. causal (triton): perturbing tokens >= p leaves logits < p bit-unchanged (eval and train)."""
import sys, logging, random, torch, torch.nn.functional as F
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
CFG, CK = sys.argv[1:3]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)
def seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); random.seed(s); np.random.seed(s)
cfg = PinballConfig.from_yaml(CFG); L = int(cfg.block_size); cfg.hier_layer_cudagraphs = False
inp = resolve_model_inputs(cfg, block_size=L)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda()
sd = torch.load(CK, map_location="cpu", weights_only=False)
mi, un = m.load_state_dict(sd.get("model_state_dict", sd), strict=False); del sd
print(f"[load] missing {len(mi)} unexpected {len(un)}")
tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
val = tok[int(len(tok) * (1 - float(cfg.val_split))):]
st = np.linspace(0, len(val) - L - 2, NS).astype(np.int64)
X = torch.stack([val[s:s + L + 1].clone() for s in st.tolist()]).cuda()
def logits(ids, train, s=7):
    m.train(train); seed(s)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        o = m(ids); o = o[0] if isinstance(o, tuple) else o
    return o.float()
def mode(kname):
    m.xq_read_kernel = kname

print("1. eval: triton vs gather")
nll = {}
for kn in ("gather", "triton"):
    mode(kn); logits(X[:1, :L], False)
    out = []
    for i in range(NS):
        lg = logits(X[i:i + 1, :L], False)
        out.append(F.cross_entropy(lg[0], X[i, 1:L + 1], reduction="mean").item())
        if i == 0: nll[kn + "_lg"] = lg
    nll[kn] = np.array(out)
dpos = (nll["gather_lg"] - nll["triton_lg"]).abs().amax(-1).flatten()      # per position
d, d999 = dpos.max().item(), dpos.quantile(0.999).item()
frac = (dpos > 0.25).float().mean().item()
dn = np.abs(nll["gather"] - nll["triton"]).max()
# NOT an exactness check: the discrete selection amplifies any rounding. Baseline (descent
# ckpt, 16k): gather + 1e-3 relative noise on the read output already gives max 3.2,
# 99.9th pct 1.26, 6.0% of positions > 0.25, |dNLL| 2e-4 -- triton sits at the same level.
rep("eval logits (vs the 1e-3 noise baseline)", d999 < 2.0 and frac < 0.1,
    f"max|dlogit| {d:.3e}, 99.9th pct {d999:.3e}, positions > 0.25: {frac:.2e}")
rep("eval NLL", dn < 2e-3, f"per-seq NLL gather {nll['gather'].round(4).tolist()} triton {nll['triton'].round(4).tolist()}")

print("2. train step: loss, KL, gradients")
G = {}
for kn in ("gather", "triton"):
    mode(kn); m.train(); m.zero_grad(set_to_none=True)
    for _ in range(2):          # warm (compile) then measure
        m.zero_grad(set_to_none=True); seed(3)
        with torch.autocast("cuda", torch.bfloat16):
            o = m(X[:2, :L]); o = o[0] if isinstance(o, tuple) else o
            task = F.cross_entropy(o.flatten(0, 1).float(), X[:2, 1:L + 1].flatten())
            kl = m._last_xq_index_loss; m._last_xq_index_loss = None
            loss = task + (kl if kl is not None else 0.0)
        loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.grad is not None}
    G[kn] = (task.item(), None if kl is None else kl.item(), grads)
(tg, kg, gg), (tt, kt, gt) = G["gather"], G["triton"]
rep("task loss", abs(tg - tt) / tg < 2e-3, f"gather {tg:.5f} triton {tt:.5f}")
rep("indexer KL", kg is not None and kt is not None and abs(kg - kt) / max(kg, 1e-6) < 0.05, f"gather {kg} triton {kt}")
groups = {"indexer": lambda n: n.startswith("xq_index_"), "read sink": lambda n: "hqd_read_sink_k" in n,
          "attn qkv": lambda n: ".message_passing." in n and ("q_proj" in n or "k_proj" in n or "v_proj" in n or "qkv" in n),
          "sparse_out_proj": lambda n: "sparse_out_proj" in n, "all": lambda n: True}
for gname, f in groups.items():
    ns = [n for n in gg if f(n) and n in gt]
    if not ns:
        print(f"  (no params for {gname})"); continue
    a = torch.cat([gg[n].flatten() for n in ns]); b = torch.cat([gt[n].flatten() for n in ns])
    cos = F.cosine_similarity(a, b, dim=0).item(); r = (b.norm() / a.norm().clamp(min=1e-12)).item()
    rep(f"grad {gname}", cos > 0.99 and abs(r - 1) < 0.05, f"cos {cos:.5f} norm ratio {r:.4f} ({len(ns)} tensors)")

print("3. causal (triton)")
mode("triton")
p = L // 2 + 37
g = torch.Generator().manual_seed(1)
x2 = X[:1, :L].clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (1, L - p), generator=g).cuda()
for train in (False, True):
    a = logits(X[:1, :L], train); a2 = logits(X[:1, :L], train); b = logits(x2, train)
    rep(f"{'train' if train else 'eval'} reproducible", (a - a2).abs().max().item() == 0.0, f"{(a - a2).abs().max().item():.3e}")
    past = (a[:, :p] - b[:, :p]).abs().max().item(); fut = (a[:, p:] - b[:, p:]).abs().max().item()
    rep(f"causal ({'train' if train else 'eval'})", past == 0.0 and fut > 0.0, f"past {past:.3e}, future {fut:.3e}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
