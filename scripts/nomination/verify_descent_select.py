"""xq_select_kernel: triton vs torch on a trained descent checkpoint.

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_descent_select.py CONFIG CKPT
  1. eval, round 0 (identical inputs): per level, the fraction of queries whose kept SET
     matches exactly, and the mean pick overlap; parent slots point at a parent containing
     the pick; every pick is causal and (L0) >= local_exclude back; no duplicate picks;
  2. eval NLL on 4 sequences (both kernels on vs both off);
  3. a training step runs with noise (finite loss and KL), and causality holds (triton)."""
import sys, logging, random, torch, torch.nn.functional as F
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
import pinball.model.hierarchical_flow_gat_cached_batch as M
CFG, CK = sys.argv[1:3]
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
m.load_state_dict(torch.load(CK, map_location="cpu", weights_only=False)["model_state_dict"], strict=False)
tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
val = tok[int(len(tok) * (1 - float(cfg.val_split))):]
st = np.linspace(0, len(val) - L - 2, 4).astype(np.int64)
X = torch.stack([val[s:s + L + 1].clone() for s in st.tolist()]).cuda()
REC = []
_on = M.HierarchicalFlowGAT._xq_descent_nominate
@torch._dynamo.disable
def nom(self, x, lo, t):
    out = _on(self, x, lo, t)
    r = self._xq_desc_record
    REC.append(([(l, k.clone(), None if p is None else p.clone()) for l, k, p in r["levels"]], list(r["o"]), t.clone(),
                r["q"].detach().clone(), {l: v.detach().clone() for l, v in r["k"].items()}, r["scale"]))
    return out
M.HierarchicalFlowGAT._xq_descent_nominate = nom
def run(sel, x, train=False, s=7):
    m.xq_select_kernel = sel; m.train(train); seed(s); REC.clear()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        o = m(x); o = o[0] if isinstance(o, tuple) else o
    return o.float(), list(REC)

print("1. round-0 picks (eval): the kernel vs an fp32 reference on ITS OWN parents")
# NOT vs the torch path: its bf16 GEMM output rounds the indexer scores so coarsely that at
# L3 ~90% of queries tie exactly at the top-8 cut (L2 ~53%), so torch picks are partly
# index-order tie breaks; the kernel scores in fp32.
for BAND in (False, True):
  m.xq_descent_exclude_band = BAND
  print(f" band exclusion {'ON' if BAND else 'off'}")
  run("triton", X[:1, :L]); _, rb = run("triton", X[:1, :L])
  levels_b, o, t, qr, kr, sc = rb[0]
  tabs = m._xq_tree_tables([o[i + 1] - o[i] for i in range(len(o) - 1)], X.device)
  excl = int(m.xq_descent_local_exclude); n0 = o[1] - o[0]
  q_t = t[o[0]:o[1]]
  if BAND:
      _sp = m._local_pack_spec_cache[1]; inv = _sp["xq_inv_perm"]; W = int(_sp["window"])
  for l, kb, pb in levels_b:
      if pb is None:
          continue
      par = next(kk for ll, kk, _ in levels_b if ll == l + 1)
      ch = tabs[l + 1][par.clamp(min=0)]
      ch = torch.where((par >= 0).unsqueeze(-1), ch, torch.full_like(ch, -1)).flatten(-2)
      ok = ch >= 0; safe = ch.clamp(min=0)
      tl_ = t[o[l]:o[l + 1]]
      ok &= tl_[safe] <= q_t.view(1, -1, 1)
      if l == 0:
          ok &= safe <= torch.arange(n0, device=X.device).view(1, -1, 1) - excl
      if BAND:
          _dr = inv[o[0]:o[1]].view(1, -1, 1) - inv[o[l]:o[l + 1]][safe]
          ok &= ~((_dr >= 0) & (_dr <= W))
      s_ = (qr.float().unsqueeze(2) * kr[l].float()[0][safe[0]].unsqueeze(0)).sum(-1) * sc
      s_ = s_.masked_fill(~ok, float("-inf"))
      srt, order = safe.masked_fill(~ok, -1).sort(-1)
      dup = torch.zeros_like(ok); dup[..., 1:] = (srt[..., 1:] == srt[..., :-1]) & (srt[..., 1:] >= 0)
      s_ = s_.masked_fill(torch.zeros_like(ok).scatter(-1, order, dup), float("-inf"))
      top = s_.topk(kb.size(-1), -1)
      ref = torch.where(torch.isfinite(top.values), ch.gather(-1, top.indices), torch.full_like(top.indices, -1))
      vr, vb = ref >= 0, kb >= 0
      has = vr.any(-1)
      inter = ((ref.unsqueeze(-1) == kb.unsqueeze(-2)) & vr.unsqueeze(-1) & vb.unsqueeze(-2)).sum((-1, -2)).float()
      ov = (inter / vr.sum(-1).clamp(min=1).float())[has].mean().item()
      cnt = (vr.sum(-1) == vb.sum(-1)).float().mean().item()
      sb = torch.where(vb, kb, -(torch.arange(kb.size(-1), device=X.device) + 2)).sort(-1).values
      nodup = not bool((sb[..., 1:] == sb[..., :-1]).any())
      par_id = par.gather(-1, pb.clamp(min=0))
      slot_ok = torch.where(vb, (tabs[l + 1][par_id.clamp(min=0)] == kb.unsqueeze(-1)).any(-1), torch.ones_like(vb)).all().item()
      inband = 0
      if BAND:
          _db = inv[o[0]:o[1]].view(1, -1, 1) - inv[o[l]:o[l + 1]][kb.clamp(min=0)]
          inband = int((vb & (_db >= 0) & (_db <= W)).sum())
      rep(f"L{l}{' band' if BAND else ''}", ov > 0.995 and cnt > 0.995 and nodup and slot_ok and inband == 0,
          f"overlap with fp32 ref {ov:.5f} (queries with candidates), same pick count {cnt:.5f}, "
          f"no-dup {nodup}, slot->parent {slot_ok}, in-band picks {inband}")
m.xq_descent_exclude_band = False

print("2. eval NLL")
def nll(kern):
    m.xq_read_kernel = "triton" if kern else "gather"
    out = []
    for i in range(4):
        lg, _ = run("triton" if kern else "torch", X[i:i + 1, :L])
        out.append(F.cross_entropy(lg[0], X[i, 1:L + 1]).item())
    return np.array(out)
a, b = nll(False), nll(True)
rep("eval NLL (both kernels vs both torch)", np.abs(a - b).max() < 3e-3, f"torch {a.round(4).tolist()} triton {b.round(4).tolist()}")

print("3. training step + causal (both kernels)")
m.xq_read_kernel = "triton"; m.xq_select_kernel = "triton"; m.train(); seed(3)
with torch.autocast("cuda", torch.bfloat16):
    o = m(X[:2, :L]); o = o[0] if isinstance(o, tuple) else o
    task = F.cross_entropy(o.flatten(0, 1).float(), X[:2, 1:L + 1].flatten())
    kl = m._last_xq_index_loss; m._last_xq_index_loss = None
(task + kl).backward()
rep("train step", bool(torch.isfinite(task)) and kl is not None and bool(torch.isfinite(kl)), f"task {task.item():.4f} KL {kl.item():.4f}")
p = L // 2 + 37
g = torch.Generator().manual_seed(1)
x2 = X[:1, :L].clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (1, L - p), generator=g).cuda()
for train in (False, True):
    run("triton", X[:1, :L], train)       # warm: first call captures the selection CUDA graphs (consumes RNG)
    A, _ = run("triton", X[:1, :L], train); A2, _ = run("triton", X[:1, :L], train); Bb, _ = run("triton", x2, train)
    rep(f"{'train' if train else 'eval'} reproducible", (A - A2).abs().max().item() == 0.0)
    past = (A[:, :p] - Bb[:, :p]).abs().max().item(); fut = (A[:, p:] - Bb[:, p:]).abs().max().item()
    rep(f"causal ({'train' if train else 'eval'})", past == 0.0 and fut > 0.0, f"past {past:.3e}, future {fut:.3e}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
