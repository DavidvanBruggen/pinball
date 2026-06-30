# SPDX-License-Identifier: GPL-3.0-or-later
"""Drill into the refinement loop: split per-step time between local-window attention,
cross-level scatter message passing, and the SwiGLU FFN. Patches CLASS methods so all 12
layers are covered. Forward-only (no_grad) attribution.

Run:  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python bench_refine_drill.py --batch 2 --block 1024
"""
import sys, time, pathlib, argparse, collections, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig
from pinball.model.layers.hierarchical_message_passing import HierarchicalMessagePassing

TOK = AutoTokenizer.from_pretrained("gpt2")
DEV = "cuda:0"
ACC = collections.defaultdict(lambda: [0.0, 0])


def patch_class(cls, name, label):
    orig = getattr(cls, name, None)
    if orig is None:
        print(f"  (no {cls.__name__}.{name})"); return
    def timed(self, *a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig(self, *a, **k)
        torch.cuda.synchronize()
        ACC[label][0] += (time.perf_counter() - t0) * 1000.0
        ACC[label][1] += 1
        return r
    setattr(cls, name, timed)


def patch_ffn(model):
    for m in model.modules():
        if type(m).__name__ == "SwiGLUFFN":
            orig = m.forward
            def timed(*a, _f=orig, **k):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                r = _f(*a, **k)
                torch.cuda.synchronize()
                ACC["ffn"][0] += (time.perf_counter() - t0) * 1000.0
                ACC["ffn"][1] += 1
                return r
            m.forward = timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    cfg.batch_size = args.batch; cfg.block_size = args.block
    tok = TOK
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                        tie_weights=True, max_seq_len=cfg.block_size).to(DEV)
    model.train()
    patch_ffn(model)

    torch.manual_seed(0)
    ids = torch.randint(0, TOK.vocab_size, (args.batch, args.block), device=DEV)

    with torch.no_grad():
        for _ in range(args.warmup):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(ids)
    torch.cuda.synchronize()

    # patch AFTER warmup so cache-build isn't counted
    patch_class(HierarchicalMessagePassing, "_forward_batched", "mp_total")
    patch_class(HierarchicalMessagePassing, "_compute_level_local_out_batched", "local_attn")
    patch_class(HierarchicalMessagePassing, "_sparse_graph_attention_chunked_batched", "crosslevel_scatter")
    ACC.clear()

    with torch.no_grad():
        for _ in range(args.iters):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(ids)
    torch.cuda.synchronize()

    print(f"\n=== refinement sub-stage breakdown (per step, ms; batch={args.batch} block={args.block}) ===")
    for label in ("mp_total", "crosslevel_scatter", "local_attn", "ffn"):
        ms, calls = ACC.get(label, [0.0, 0])
        print(f"  {label:22s} {ms/args.iters:7.2f} ms/step   ({calls//max(1,args.iters)} calls/step)")
    print("  note: mp_total ⊇ crosslevel_scatter + local_attn; ffn is separate (post-MP).")


if __name__ == "__main__":
    main()
