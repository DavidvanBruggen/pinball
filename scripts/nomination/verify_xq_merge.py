"""xq_read_merge (descent v2: picks LSE-merged into the packed flex softmax), on a checkpoint.

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_xq_merge.py CONFIG CKPT
The config is run with xq_read_kernel / xq_select_kernel triton and xq_read_merge on (eager
layers, so the merge's inputs can be captured). Checks:
  1. disjoint: every pick that is READ is invisible to the main call (the cached BlockMask's
     own mask_mod at (query row, pick's band key) is False) and every pick dropped as "in
     band" IS visible there -- the merge is one softmax over a disjoint union;
  2. merged output == a dense fp32 recompute for sampled queries: (o_A e^{L_A} + sum_j e^{s_j}
     v_j) / (e^{L_A} + sum_j e^{s_j}) with s_j = q . far_key_j / sqrt(D) built independently;
  3. a training step: finite loss and indexer KL, gradients reach q/k/v projections and the
     indexer, none reach the (unused) read sink or sparse_out_proj;
  4. causal: perturbing tokens >= p leaves logits < p bit-unchanged (eval and train)."""
import sys, math, logging, random, torch, torch.nn.functional as F
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
import pinball.model.layers.hierarchical_message_passing as H
CFG, CK = sys.argv[1:3]
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)
def seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); random.seed(s); np.random.seed(s)
cfg = PinballConfig.from_yaml(CFG); L = int(cfg.block_size)
for k_, v_ in (("hier_layer_cudagraphs", False), ("hier_layer_compile", False), ("hier_refresh_compile", False),
               ("xq_read_kernel", "triton"), ("xq_select_kernel", "triton"), ("xq_read_merge", True),
               ("xq_select_cuda_graph", True)):
    setattr(cfg, k_, v_)
inp = resolve_model_inputs(cfg, block_size=L)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda()
sd = torch.load(CK, map_location="cpu", weights_only=False)
mi, un = m.load_state_dict(sd.get("model_state_dict", sd), strict=False); del sd
print(f"[load] missing {len(mi)} unexpected {len(un)}")
tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
val = tok[int(len(tok) * (1 - float(cfg.val_split))):]
X = torch.stack([val[i * (L + 1):(i + 1) * (L + 1)].clone() for i in range(2)]).cuda()
CAP = []
_om = H.HierarchicalMessagePassing._xq_pick_merge
def cap(self, qp, kp, v_s, spec, gsel, out, lse_main, causal, cand):
    res = _om(self, qp, kp, v_s, spec, gsel, out, lse_main, causal, cand)
    if len(CAP) < 2:
        CAP.append(dict(mp=self, qp=qp.detach().float(), kp=kp.detach(), v_s=v_s.detach().float(), spec=spec,
                        out=out.detach().float(), lse=lse_main.detach().float(), causal=causal,
                        cand=cand.clone(), res=res.detach().float()))
    return res
H.HierarchicalMessagePassing._xq_pick_merge = cap
m.eval()
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    m(X[:1, :L])
print(f"captured {len(CAP)} merged reads")

print("1. disjoint from the main call (its own mask_mod)")
for ci, c in enumerate(CAP):
    mp, spec = c["mp"], c["spec"]
    bm = spec["flex_block_mask"]
    N = int(c["qp"].size(1)); n_pre = int(c["v_s"].size(2)) - N
    inv = spec["xq_inv_perm"]
    o0, n0 = (int(x) for x in mp._xq_dst_range)
    qrow = inv[o0:o0 + n0]
    cand = c["cand"][0]                                     # batch 0
    ok = cand >= 0
    prow = inv[cand.clamp(min=0)]
    W = int(spec["window"])
    dr = qrow.view(-1, 1) - prow
    band = (dr >= 0) & (dr <= W)
    qi = qrow.view(-1, 1).expand_as(prow)[ok]
    ki = (n_pre + prow)[ok]
    z = torch.zeros_like(qi)
    vis = bm.mask_mod(z, z, qi, ki)
    read = ~band[ok]
    rep(f"read {ci}: read picks invisible to the main call", not bool(vis[read].any()),
        f"{int(read.sum())} read, {int(vis[read].sum())} visible")
    rep(f"read {ci}: band-dropped picks visible to the main call", bool(vis[~read].all()) if bool((~read).any()) else True,
        f"{int((~read).sum())} dropped ({float((~read).float().mean()):.3f} of picks)")

print("2. merged output == dense fp32 recompute")
for ci, c in enumerate(CAP):
    mp, spec = c["mp"], c["spec"]
    N, Hh, D = int(c["qp"].size(1)), int(c["qp"].size(2)), int(c["qp"].size(3))
    n_pre = int(c["v_s"].size(2)) - N
    inv = spec["xq_inv_perm"]; o0, n0 = (int(x) for x in mp._xq_dst_range)
    kf = mp._far_keys(c["kp"], spec["levels"]).float()
    W = int(spec["window"])
    maxd = 0.0; scl = 0.0
    for i in (300, 2047, 5000, 9999, n0 - 1):
        r = int(inv[o0 + i])
        cd = c["cand"][0, i]; cd = cd[cd >= 0]
        pr = inv[cd]
        pr = pr[~((r - pr >= 0) & (r - pr <= W))]
        q = c["qp"][0, r]                                                  # [H, D]
        s = torch.einsum("hd,khd->kh", q, kf[0, pr]) / math.sqrt(D)          # [k, H]
        v = c["v_s"][0, :, n_pre + pr].transpose(0, 1)                      # [k, H, D]
        la = c["lse"][0, :, r]; oa = c["out"][0, :, r]                      # [H], [H, D]
        mm = torch.maximum(la, s.max(0).values if s.numel() else la)
        wa = (la - mm).exp(); ws = (s - mm).exp()
        ref = (oa * wa[:, None] + (ws[:, :, None] * v).sum(0)) / (wa + ws.sum(0))[:, None]
        got = c["res"][0, :, r]
        maxd = max(maxd, (ref - got).abs().max().item()); scl = max(scl, ref.abs().max().item())
    rep(f"read {ci}: merged == dense recompute", maxd <= 2e-2 * max(scl, 1e-3) + 1e-3, f"max|diff| {maxd:.2e} (|ref| {scl:.2e})")

print("3. training step")
H.HierarchicalMessagePassing._xq_pick_merge = _om
m.train(); m.zero_grad(set_to_none=True); seed(3)
with torch.autocast("cuda", torch.bfloat16):
    o = m(X[:, :L]); o = o[0] if isinstance(o, tuple) else o
    task = F.cross_entropy(o.flatten(0, 1).float(), X[:, 1:L + 1].flatten())
    kl = m._last_xq_index_loss; m._last_xq_index_loss = None
(task + kl).backward()
g = {n: (p.grad.float().norm().item() if p.grad is not None else 0.0) for n, p in m.named_parameters()}
gi = sum(v for n, v in g.items() if n.startswith("xq_index_"))
gs = sum(v for n, v in g.items() if "hqd_read_sink_k" in n or "sparse_out_proj" in n)
gq = sum(v for n, v in g.items() if ".message_passing." in n and ("q_proj" in n or "k_proj" in n or "v_proj" in n or "qkv" in n))
st = [getattr(t_.message_passing, "_xq_merge_stats", None) for t_ in m.refinement_transformers]
st = [s_ for s_ in st if s_ is not None]
rep("train step", bool(torch.isfinite(task)) and kl is not None and bool(torch.isfinite(kl)) and gi > 0 and gq > 0 and gs == 0,
    f"task {task.item():.4f} KL {kl.item():.4f} | grad indexer {gi:.3e} qkv {gq:.3e} sink/sparse_out {gs:.1e}")
if st:
    print("  merge stats per fed layer: " + "; ".join(f"read frac {float(s_['read_frac'].detach()):.3f}, pick mass {float(s_['pick_mass'].detach()):.3f}" for s_ in st))

print("4. causal")
def logits(ids, train, s=7):
    m.train(train); seed(s)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        oo = m(ids); oo = oo[0] if isinstance(oo, tuple) else oo
    return oo.float()
p = L // 2 + 37
gg = torch.Generator().manual_seed(1)
x2 = X[:1, :L].clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (1, L - p), generator=gg).cuda()
for train in (False, True):
    logits(X[:1, :L], train)       # warm: the first call captures the selection CUDA graphs (its warm-up consumes RNG)
    a = logits(X[:1, :L], train); a2 = logits(X[:1, :L], train); b = logits(x2, train)
    rep(f"{'train' if train else 'eval'} reproducible", (a - a2).abs().max().item() == 0.0, f"{(a - a2).abs().max().item():.3e}")
    past = (a[:, :p] - b[:, :p]).abs().max().item(); fut = (a[:, p:] - b[:, p:]).abs().max().item()
    rep(f"causal ({'train' if train else 'eval'})", past == 0.0 and fut > 0.0, f"past {past:.3e}, future {fut:.3e}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
