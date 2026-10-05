"""BatchedMuon foreach updates == the per-tensor loops, bit for bit (params and optimizer state).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/verify_muon_foreach.py CONFIG [steps]
Builds CONFIG's model and its real optimizer param groups (cli._build_optimizer), feeds both
copies the same random grads for a few steps, and asserts max|diff| == 0 everywhere."""
import copy, sys, logging, torch
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
from pinball.cli import _build_optimizer
from pinball.train.muon_batched import BatchedMuon

cfg = PinballConfig.from_yaml(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cfg.muon_batched = True
inp = resolve_model_inputs(cfg, block_size=cfg.block_size)
torch.manual_seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=inp.block_size).cuda()
models = {True: m, False: copy.deepcopy(m)}
opts = {k: _build_optimizer(mm, cfg) for k, mm in models.items()}
assert all(isinstance(o, BatchedMuon) for o in opts.values())
for o in opts.values():
    for g in o.param_groups:
        g["lr"] = 1e-3          # a visible update (the schedulers are not built here)
params = {k: list(mm.parameters()) for k, mm in models.items()}
init = [p.detach().clone() for p in params[True]]
print(f"{sys.argv[1].split('/')[-1]}: {len(params[True])} tensors, groups "
      + ", ".join(f"{'muon' if g.get('use_muon') else 'adamw'}:{len(g['params'])}" for g in opts[True].param_groups))
gen = torch.Generator(device="cuda").manual_seed(1)
worst = 0.0
for s in range(steps):
    grads = [torch.randn(p.shape, generator=gen, device="cuda", dtype=p.dtype) * 1e-2 for p in params[True]]
    for k in (True, False):
        for p, g_ in zip(params[k], grads):
            p.grad = g_.clone()
        BatchedMuon._FOREACH = k
        opts[k].step()
    dp = max((a - b).abs().max().item() for a, b in zip(params[True], params[False]))
    ds = 0.0
    for a, b in zip(params[True], params[False]):
        sa, sb = opts[True].state[a], opts[False].state[b]
        assert sa.keys() == sb.keys()
        for key in sa:
            if torch.is_tensor(sa[key]):
                ds = max(ds, (sa[key] - sb[key]).abs().max().item())
    print(f"  step {s + 1}: max|param diff| {dp:.3e}  max|state diff| {ds:.3e}")
    worst = max(worst, dp, ds)
BatchedMuon._FOREACH = True
moved = max((a - b).abs().max().item() for a, b in zip(params[True], init))
print(f"  params moved by up to {moved:.3e} (sanity: must be > 0)")
worst = worst if moved > 0 else float("nan")
print("PASS (bit-identical)" if worst == 0.0 else f"FAIL: max diff {worst:.3e}")
