"""Upper stage on a descent config at full length: does it run, and is it causal?

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_upper_stage_16k.py CONFIG [CKPT]
The stage's zero-init outputs make it an identity at init, so a causality test on a fresh or
parent checkpoint would pass vacuously: every stage output projection / FFN down projection is
re-drawn at the scale of the main layers' first. Checks (compiled layers + graphs, as trained):
  1. stage runs 11x per forward over the configured levels; its write is non-trivial;
  2. a training step: finite loss and KL, gradient reaches every stage block;
  3. causal: perturbing tokens >= p leaves logits < p bit-unchanged (eval and train)."""
import sys, logging, random, torch, torch.nn.functional as F
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
CFG = sys.argv[1]; CK = sys.argv[2] if len(sys.argv) > 2 else None
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)
def seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); random.seed(s); np.random.seed(s)
cfg = PinballConfig.from_yaml(CFG); L = int(cfg.block_size)
inp = resolve_model_inputs(cfg, block_size=L)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda()
if CK:
    sd = torch.load(CK, map_location="cpu", weights_only=False)
    mi, un = m.load_state_dict(sd.get("model_state_dict", sd), strict=False); del sd
    print(f"[load] missing {len(mi)} (stage: {sum('upper_stage_blocks' in k for k in mi)}) unexpected {len(un)}")
ref = next(p for n, p in m.named_parameters() if "refinement_transformers.0." in n and "out_proj" in n and p.dim() == 2)
rms = ref.float().pow(2).mean().sqrt().item()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "upper_stage_blocks" in n and ("out_proj" in n or "down_proj" in n):
            p.normal_(0.0, rms)
print(f"stage outputs re-drawn at rms {rms:.4f}")
tok = torch.load("data/pg19_train.pt", mmap=True, weights_only=False)
val = tok[int(len(tok) * (1 - float(cfg.val_split))):]
X = torch.stack([val[i * (L + 1):(i + 1) * (L + 1)].clone() for i in range(2)]).cuda()

print("1. stage runs")
m.eval(); m._upper_stage_runs = 0
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    m(X[:1, :L])
info = m._upper_stage_info
rep("runs per forward", m._upper_stage_runs == len(m.upper_stage_blocks), f"{m._upper_stage_runs} of {len(m.upper_stage_blocks)} stages")
rep("levels", info["levels"] == [2, 3, 4, 5] and info["core_levels"] == [4, 5] and info["causal"],
    f"levels {info['levels']} core {info['core_levels']} rows {info['rows']} causal {info['causal']}")

print("2. training step")
m.train(); m.zero_grad(set_to_none=True); seed(3)
for _ in range(2):              # warm (compile, graph capture) then measure
    m.zero_grad(set_to_none=True); seed(3)
    with torch.autocast("cuda", torch.bfloat16):
        o = m(X[:, :L]); o = o[0] if isinstance(o, tuple) else o
        task = F.cross_entropy(o.flatten(0, 1).float(), X[:, 1:L + 1].flatten())
        kl = m._last_xq_index_loss; m._last_xq_index_loss = None
    (task + kl).backward()
gb = [sum(p.grad.float().norm().item() for p in blk.parameters() if p.grad is not None)
      for st in m.upper_stage_blocks for blk in st]
rep("train step", bool(torch.isfinite(task)) and kl is not None and bool(torch.isfinite(kl)) and min(gb) > 0,
    f"task {task.item():.4f} KL {kl.item():.4f} | stage blocks with grad {sum(g > 0 for g in gb)}/{len(gb)}")

print("3. causal")
def logits(ids, train, s=7):
    m.train(train); seed(s)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        oo = m(ids); oo = oo[0] if isinstance(oo, tuple) else oo
    return oo.float()
gg = torch.Generator().manual_seed(1)
for p in (L // 2 + 37, L - 300, 3000):
    x2 = X[:1, :L].clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (1, L - p), generator=gg).cuda()
    for train in (False, True):
        logits(X[:1, :L], train)           # warm: graph capture consumes RNG
        a = logits(X[:1, :L], train); a2 = logits(X[:1, :L], train); b = logits(x2, train)
        repro = (a - a2).abs().max().item()
        past = (a[:, :p] - b[:, :p]).abs().max().item(); fut = (a[:, p:] - b[:, p:]).abs().max().item()
        rep(f"causal p={p} ({'train' if train else 'eval'})", repro == 0.0 and past == 0.0 and fut > 0.0,
            f"repro {repro:.1e}, past {past:.3e}, future {fut:.3e}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
