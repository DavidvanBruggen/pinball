"""local_pack_global_nom_inline probe: the selector traced INTO the compiled layer.

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_nom_inline.py CONFIG [CKPT] [L]
  1. eval (no Gumbel noise), compiled layers: inline vs the opaque nom_compile path -- logits
     close, CE equal, picks agree as per-chunk sets;
  2. train mode, inline: causal (perturbing tokens >= p leaves logits < p unchanged; the
     in-graph noise is reproduced by reseeding);
  3. train step, inline: zero graph breaks, one compiled region per layer."""
import sys, random, logging, torch
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
import pinball.model.layers.hierarchical_message_passing as H

CFG = sys.argv[1]
CK = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

cfg = PinballConfig.from_yaml(CFG)
L = int(sys.argv[3]) if len(sys.argv) > 3 else int(cfg.block_size)
cfg.block_size = L
assert cfg.local_pack_global_nom_compile and getattr(cfg, "hier_layer_compile", False)
inp = resolve_model_inputs(cfg, block_size=L)
torch.manual_seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda()
if CK:
    sd = torch.load(CK, map_location="cuda", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    mi, un = m.load_state_dict(sd, strict=False)
    print(f"loaded {CK}: missing {len(mi)} unexpected {len(un)}")
mps = [t.message_passing for t in m.refinement_transformers]

# record each layer's picks: a module-attribute store, so it also works inside the compiled
# layer (dynamo replays it as a side effect after the graph)
_orig_finish = H.HierarchicalMessagePassing._nom_stream_finish
def _finish(self, spec, rows, g, stats, geo, C, nch, K):
    self._probe_rows = rows
    return _orig_finish(self, spec, rows, g, stats, geo, C, nch, K)
H.HierarchicalMessagePassing._nom_stream_finish = _finish

def set_inline(on):
    for mp in mps:
        mp.local_pack_global_nom_inline = bool(on)

g = torch.Generator().manual_seed(1)
if CK and getattr(cfg, "text_file", None):
    txt = open(cfg.text_file, errors="ignore").read()
    ids = inp.tokenizer(txt[-600000:], return_tensors="pt").input_ids[0]
    x = torch.stack([ids[i * L:(i + 1) * L] for i in range(2)]).cuda()
else:
    x = torch.randint(0, inp.vocab_size, (2, L), generator=g).cuda()

def fwd(xx):
    with torch.autocast("cuda", torch.bfloat16):
        o = m(xx)
    return (o[0] if isinstance(o, tuple) else o).float()

def ce(o, xx):
    return torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)),
                                             xx[:, 1:].reshape(-1)).item()

def picks():
    return [mp._probe_rows.detach().clone() for mp in mps]

def set_overlap(ra, rb, K):
    a = ra.view(ra.size(0), -1, K).sort(-1).values
    b = rb.view(rb.size(0), -1, K).sort(-1).values
    hit = 0
    for i in range(K):   # membership of each of a's picks in b's chunk set
        hit += (a[..., i:i + 1] == b).any(-1).sum().item()
    return hit / a.numel()

K = int(mps[0].local_pack_global_l0_budget)

# 1. eval, inline vs opaque
print("1. eval: inline vs opaque nom_compile (compiled layers)")
m.eval()
out = {}
with torch.no_grad():
    for on in (False, True):
        set_inline(on)
        fwd(x)                       # warm (compiles)
        o = fwd(x)
        out[on] = (o, ce(o, x), picks())
d = (out[True][0] - out[False][0]).abs().max().item()
ref = out[False][0].abs().max().item()
rep("logits close", d <= 2e-2 * max(1.0, ref), f"max|diff| {d:.3e} (|ref|max {ref:.1f})")
dce = abs(out[True][1] - out[False][1])
rep("CE equal", dce <= 2e-3 * out[False][1], f"opaque {out[False][1]:.5f} inline {out[True][1]:.5f}")
ov = [set_overlap(a, b, K) for a, b in zip(out[True][2], out[False][2])]
rep("picks agree", min(ov) >= 0.95, "per-layer set overlap " + " ".join(f"{v:.3f}" for v in ov))

# 2. train-mode causality, inline
print("2. train mode, inline: causality")
m.train(); set_inline(True)
p = L // 2
x2 = x.clone()
x2[:, p:] = torch.randint(0, inp.vocab_size, (x.size(0), L - p), generator=g).cuda()
def seeded(xx):
    torch.manual_seed(7); torch.cuda.manual_seed(7); random.seed(7); np.random.seed(7)
    with torch.no_grad():
        return fwd(xx)
seeded(x)
a, a2, b = seeded(x), seeded(x), seeded(x2)
rep("reseeded forward reproducible", (a - a2).abs().max().item() == 0.0,
    f"max|diff| {(a - a2).abs().max().item():.3e}")
past = (a[:, :p] - b[:, :p]).abs().max().item()
fut = (a[:, p:] - b[:, p:]).abs().max().item()
rep("causal", past == 0.0 and fut > 0.0, f"past max|diff| {past:.3e}, future {fut:.3e}")

# 3. graph count on a train step
print("3. train step, inline: graph breaks / compiled regions")
from torch._dynamo.utils import counters
from torch.profiler import profile, ProfilerActivity
counters["graph_break"].clear()
def step():
    with torch.autocast("cuda", torch.bfloat16):
        o = m(x); o = o[0] if isinstance(o, tuple) else o
        loss = o[:, :256].float().logsumexp(-1).mean()
    loss.backward(); m.zero_grad(set_to_none=True)
for _ in range(3): step()
nb = sum(counters["graph_break"].values())
with profile(activities=[ProfilerActivity.CPU]) as prof:
    step(); torch.cuda.synchronize()
regions = sum(1 for e in prof.events() if e.name.startswith("Torch-Compiled Region"))
rep("no graph breaks", nb == 0, f"{nb} (" + "; ".join(k.splitlines()[0][:80] for k in counters["graph_break"]) + ")")
n_lay = len(mps)
rep("one region per layer (+2 refresh cores)", regions <= 3 * n_lay, f"{regions} regions for {n_lay} layers")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
