# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-0 step profiler: where does a Pinball training step actually spend time/memory?

Builds the model from configs/pinball_wikitext.yaml (live config, use_hqd=False), then:
  (1) wraps the major stages (hierarchy build, per-level processing, unified-graph build,
      refinement, output projection) with CUDA-synchronized wall timers,
  (2) reports fwd / bwd / total ms, tokens/s, and peak memory,
  (3) dumps a torch.profiler top-op table (CUDA self-time) for the steady-state step,
  (4) measures the cross-level edge degree distribution per level (uniform vs ragged ->
      decides whether the Tier-3 packed/dense reformulation can pay off).

Run on the FREE card only:  CUDA_VISIBLE_DEVICES=0 python bench_step_profile.py --batch 8
(nvidia-smi index is inverted vs torch; torch cuda:0 == the idle 4090.)
"""
import sys, time, pathlib, argparse, types, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

TOK = AutoTokenizer.from_pretrained("gpt2")
DEV = "cuda:0"


def build(overrides):
    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    tok = TOK
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok),
                        input_mode="tokens", tie_weights=True,
                        max_seq_len=cfg.block_size).to(DEV)
    return model


# ---- stage timing via method wrapping -------------------------------------------------
STAGE_METHODS = [
    "_get_embeddings",
    "_process_level",
    "_build_unified_graph",
    "_apply_unified_refinement_true_batch_nozip",
    "output_projection",
    "_create_next_level",
]


def install_timers(model):
    acc = collections.defaultdict(lambda: [0.0, 0])  # name -> [ms, calls]

    def wrap(name, fn):
        def timed(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            acc[name][0] += (time.perf_counter() - t0) * 1000.0
            acc[name][1] += 1
            return r
        return timed

    import inspect
    for name in STAGE_METHODS:
        fn = getattr(model, name, None)
        if fn is None or not inspect.ismethod(fn):
            # skip nn.Module submodules / non-bound-method attributes
            if fn is not None and not inspect.ismethod(fn):
                print(f"  (skip timer for {name}: not a bound method, type={type(fn).__name__})")
            continue
        setattr(model, name, types.MethodType(lambda self, *a, _f=fn, _n=name, **k: wrap(_n, _f)(*a, **k), model))
    return acc


def capture_unified_edges(model):
    """Stash the unified edge_index/node_level from _build_unified_graph for degree stats."""
    fn = getattr(model, "_build_unified_graph", None)
    if fn is None:
        return
    def wrapped(self, *a, _f=fn, **k):
        out = _f(*a, **k)
        try:
            g = out[0] if isinstance(out, (tuple, list)) else out
            ei = getattr(g, "edge_index", None)
            nl = getattr(g, "node_level", None)
            if ei is not None:
                self._prof_edge_index = ei.detach()
            if nl is not None:
                self._prof_node_level = nl.detach()
        except Exception:
            pass
        return out
    setattr(model, "_build_unified_graph", types.MethodType(wrapped, model))


def fwd_bwd(model, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        loss = feats.float().pow(2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def time_fwd_bwd(model, ids, iters, warmup):
    f_ms, b_ms = [], []
    for i in range(warmup + iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids)
            feats = out[0] if isinstance(out, (tuple, list)) else out
            loss = feats.float().pow(2).mean()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        if i >= warmup:
            f_ms.append((t1 - t0) * 1000.0); b_ms.append((t2 - t1) * 1000.0)
    return sum(f_ms) / len(f_ms), sum(b_ms) / len(b_ms)


def degree_stats(model):
    ei = getattr(model, "_prof_edge_index", None)
    nl = getattr(model, "_prof_node_level", None)
    if ei is None:
        print("  (no edge_index captured)")
        return
    dst = ei[1]
    N = int(nl.numel()) if nl is not None else int(ei.max().item()) + 1
    deg = torch.zeros(N, dtype=torch.long, device=ei.device)
    deg.scatter_add_(0, dst, torch.ones_like(dst))
    print(f"  total edges={ei.size(1):,}  nodes={N:,}")
    if nl is not None:
        for lvl in sorted(set(nl.tolist())):
            m = nl == lvl
            d = deg[m].float()
            if d.numel() == 0:
                continue
            print(f"  L{lvl}: nodes={int(m.sum()):>6d}  in-degree min/mean/max/std = "
                  f"{int(d.min())}/{d.mean():.1f}/{int(d.max())}/{d.std(unbiased=False):.1f}"
                  f"   ragged={'YES' if d.std(unbiased=False) > 0.25*max(1.0,d.mean()) else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info(0)[0]/1e9:.1f}GB")
    torch.manual_seed(0)
    ids = torch.randint(0, TOK.vocab_size, (args.batch, args.block), device=DEV)

    model = build(dict(batch_size=args.batch, block_size=args.block))
    model.train()
    capture_unified_edges(model)

    # warmup (also fills Tier-1-relevant caches if any)
    for _ in range(args.warmup):
        fwd_bwd(model, ids)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ---- end-to-end fwd/bwd timing ----
    f_ms, b_ms = time_fwd_bwd(model, ids, args.iters, warmup=0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    tok_s = (args.batch * args.block) / ((f_ms + b_ms) / 1000.0)
    print(f"\n=== step timing (batch={args.batch} block={args.block}) ===")
    print(f"  fwd={f_ms:.1f}ms  bwd={b_ms:.1f}ms  total={f_ms+b_ms:.1f}ms  "
          f"tokens/s={tok_s:,.0f}  peak_mem={peak:.1f}GB")

    torch.cuda.empty_cache()

    # ---- per-stage breakdown (forward only, NO grad so it stays light) ----
    acc = install_timers(model)
    with torch.no_grad():
        for _ in range(args.iters):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids)
                _ = out[0] if isinstance(out, (tuple, list)) else out
    torch.cuda.synchronize()
    print(f"\n=== forward stage breakdown (per step, ms; note: nested stages double-count parents) ===")
    for name, (ms, calls) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:46s} {ms/args.iters:7.2f} ms/step   ({calls//max(1,args.iters)} calls/step)")

    # ---- degree distribution (edges captured during warmup) ----
    print(f"\n=== cross-level edge in-degree distribution ===")
    degree_stats(model)

    # ---- torch.profiler top ops ----
    print(f"\n=== torch.profiler top CUDA ops (1 step fwd+bwd) ===")
    torch.cuda.empty_cache()
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fwd_bwd(model, ids)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22))
    except Exception as e:
        print(f"  (profiler step skipped: {type(e).__name__}: {str(e)[:120]})")


if __name__ == "__main__":
    main()
