"""hier_layer_cudagraphs probe: CUDA-graph replay of the compiled layers in training.

usage: CUDA_VISIBLE_DEVICES=1 python scripts/verify_layer_cudagraphs.py CONFIG [L] [B] [steps]
Trains three fresh copies (same seed, same batches, real optimizer) for `steps` steps, with an
eval forward in the middle (train -> eval -> train must not touch stale graph outputs):
  A, A' = plain compiled layers (run-to-run noise floor: the backward has atomics)
  C     = hier_layer_cudagraphs
and asserts |loss_C - loss_A| stays within a few x the A-vs-A' floor, that C replays graphs
(launches per step well under A's) and that nothing raised."""
import sys, random, logging, collections, torch
import numpy as np
logging.disable(logging.WARNING)
from torch.profiler import profile, ProfilerActivity
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
from pinball.cli import _build_optimizer

CFG = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
B = int(sys.argv[3]) if len(sys.argv) > 3 else 4
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
LA = ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC")

def seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); random.seed(s); np.random.seed(s)

def run(cg):
    torch._dynamo.reset()   # fresh compile per copy (dynamo caches are per code object)
    cfg = PinballConfig.from_yaml(CFG)
    cfg.block_size = L
    cfg.hier_layer_cudagraphs = bool(cg)
    # the intended pairing: with the nominator traced inline each layer is ONE graph
    cfg.local_pack_global_nom_inline = bool(
        getattr(cfg, "local_pack_global_nom_compile", False)
        and str(getattr(cfg, "local_pack_global_nominator", "")) == "head")
    inp = resolve_model_inputs(cfg, block_size=L)
    seed(0)
    m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                    tie_weights=inp.tie_weights, max_seq_len=L).cuda().train()
    opt = _build_optimizer(m, cfg)
    for g in opt.param_groups:
        g["lr"] = g["lr"] * 0.1 if g.get("lr") else 1e-4
    gen = torch.Generator().manual_seed(1)
    data = [torch.randint(0, inp.vocab_size, (B, L), generator=gen).cuda() for _ in range(4)]
    xe = torch.randint(0, inp.vocab_size, (1, L), generator=gen).cuda()
    losses, launches = [], None
    seed(3)

    def step(x):
        with torch.autocast("cuda", torch.bfloat16):
            o = m(x); o = o[0] if isinstance(o, tuple) else o
            loss = torch.nn.functional.cross_entropy(o[:, :-1].float().reshape(-1, o.size(-1)),
                                                     x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        return loss.detach()

    for i in range(STEPS):
        if i == STEPS // 2:
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                m(xe); m(xe)
            m.train()
        if i in (STEPS // 2 - 1, STEPS - 2):
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                losses.append(step(data[i % len(data)])); torch.cuda.synchronize()
            n_ = sum(1 for e in prof.events() if e.name in LA)
            print(f"    [{'cudagraphs' if cg else 'plain'}] step {i}: {n_} launches", flush=True)
            launches = n_
        else:
            losses.append(step(data[i % len(data)]))
    out = torch.stack(losses).float().cpu()
    del m, opt
    torch.cuda.empty_cache()
    return out, launches

FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

la, nla = run(False)
la2, _ = run(False)
lc, nlc = run(True)
floor = (la - la2).abs().max().item()
d = (lc - la).abs().max().item()
print("plain   :", " ".join(f"{v:.4f}" for v in la.tolist()))
print("cudagr. :", " ".join(f"{v:.4f}" for v in lc.tolist()))
rep("loss trajectory", d <= max(4 * floor, 2e-3 * la.abs().max().item()),
    f"max|C - A| {d:.3e} vs run-to-run floor max|A - A'| {floor:.3e}")
rep("graphs replay", nlc is not None and nla is not None and nlc < 0.5 * nla,
    f"launches/step plain {nla}, cudagraphs {nlc}")
rep("training moved", (la[0] - la[-1]).item() > 0, f"loss {la[0]:.4f} -> {la[-1]:.4f}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
