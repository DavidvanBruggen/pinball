"""local_pack_global_nom_query probe (per-chunk query-conditioned slot ranking, dense selector).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_nom_query.py CONFIG [L] [B]
CONFIG is a head-nominator slot arm; the probe forces select_impl dense, nom_compile off and
eager layers (exact comparisons), and checks:
  1. at init (q = 0) the knob is BIT-identical to the query-free dense selector: logits, picks
     and every shared grad, train mode with Gumbel noise, fixed seed;
  2. the query projection gets gradient at init (it can leave zero);
  3. with q randomised the picks change (the term is live);
  4. causal with q randomised: perturbing tokens >= p leaves logits < p unchanged."""
import sys, random, logging, torch
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
import pinball.model.layers.hierarchical_message_passing as H

CFG = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
B = int(sys.argv[3]) if len(sys.argv) > 3 else 2
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

def seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); random.seed(s); np.random.seed(s)

def mk(query):
    cfg = PinballConfig.from_yaml(CFG)
    cfg.block_size = L
    cfg.local_pack_global_select_impl = "dense"
    cfg.local_pack_global_nom_compile = False
    cfg.local_pack_global_nom_inline = False
    cfg.local_pack_global_nom_query = bool(query)
    cfg.hier_layer_compile = False
    cfg.hier_refresh_compile = False
    inp = resolve_model_inputs(cfg, block_size=L)
    seed(0)
    m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                    tie_weights=inp.tie_weights, max_seq_len=L).cuda().train()
    return m, inp

_orig = H.HierarchicalMessagePassing._nominate_by_head
def _spy(self, spec, lvl_packed, x_nodes):
    r = _orig(self, spec, lvl_packed, x_nodes)
    self._probe_rows = r[0].detach().clone()
    return r
H.HierarchicalMessagePassing._nominate_by_head = _spy

m0, inp = mk(False)
m1, _ = mk(True)
mi, un = m1.load_state_dict(m0.state_dict(), strict=False)
assert not un and all("nom_query" in k for k in mi), (mi, un)
mps1 = [t.message_passing for t in m1.refinement_transformers]
assert all(float(mp.nom_query_q.abs().max()) == 0.0 for mp in mps1)

g = torch.Generator().manual_seed(1)
x = torch.randint(0, inp.vocab_size, (B, L), generator=g).cuda()

def fwd(m, xx, s=7, grad=True):
    seed(s)
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        o = m(xx); o = o[0] if isinstance(o, tuple) else o
        o = o.float()
    if grad:
        torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)), xx[:, 1:].reshape(-1)).backward()
    return o.detach()

def picks(m):
    return [t.message_passing._probe_rows for t in m.refinement_transformers]

print("1. init: nom_query (q = 0) vs query-free dense selector")
fwd(m0, x); fwd(m1, x)          # warm-up: the first call builds caches and is not reproducible
o0 = fwd(m0, x); p0 = picks(m0); g0 = {n: p.grad for n, p in m0.named_parameters() if p.grad is not None}
o1 = fwd(m1, x); p1 = picks(m1); g1 = {n: p.grad for n, p in m1.named_parameters() if p.grad is not None}
rep("logits bit-identical", (o0 - o1).abs().max().item() == 0.0, f"max|diff| {(o0 - o1).abs().max().item():.3e}")
rep("picks identical", all(torch.equal(a, b) for a, b in zip(p0, p1)))
shared = [n for n in g0 if n in g1]
gd = max((g0[n] - g1[n]).abs().max().item() for n in shared)
# the backward has atomics: compare against the same model twice
fwd(m0, x); g0b = {n: p.grad for n, p in m0.named_parameters() if p.grad is not None}
floor = max((g0[n] - g0b[n]).abs().max().item() for n in shared)
rep("grads equal (vs run-to-run floor)", gd <= max(floor * 1.5, 0.0) or gd == 0.0,
    f"max|dg| {gd:.3e}, floor {floor:.3e}, {len(shared)} tensors")

print("2. query projection trains from zero")
fwd(m1, x)
gq = [mp.nom_query_q.grad for mp in mps1]
rep("nom_query_q grad nonzero in every layer",
    all(t is not None and float(t.abs().max()) > 0 for t in gq),
    " ".join(f"{float(t.abs().max()):.1e}" if t is not None else "None" for t in gq))

print("3. randomised q: picks change")
with torch.no_grad():
    for mp in mps1:
        mp.nom_query_q.normal_(0, 0.5)
fwd(m1, x, grad=False); p2 = picks(m1)
frac = [1.0 - (a == b).float().mean().item() for a, b in zip(p1, p2)]
rep("picks changed", min(frac) > 0.0, "fraction of slots changed per layer " + " ".join(f"{f:.2f}" for f in frac))

print("4. randomised q: causal")
p = L // 2
x2 = x.clone()
x2[:, p:] = torch.randint(0, inp.vocab_size, (B, L - p), generator=g).cuda()
with torch.no_grad():
    a = fwd(m1, x, grad=False); a2 = fwd(m1, x, grad=False); b = fwd(m1, x2, grad=False)
rep("reseeded forward reproducible", (a - a2).abs().max().item() == 0.0, f"{(a - a2).abs().max().item():.3e}")
past = (a[:, :p] - b[:, :p]).abs().max().item()
fut = (a[:, p:] - b[:, p:]).abs().max().item()
rep("causal", past == 0.0 and fut > 0.0, f"past max|diff| {past:.3e}, future {fut:.3e}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
