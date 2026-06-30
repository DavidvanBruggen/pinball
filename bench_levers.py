# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the two real remaining efficiency levers on the live config:
  (A) gradient_checkpointing  -> peak memory (the 'less memory' ask)
  (B) torch.compile of the SwiGLU FFN blocks -> step speed (GEMM/elementwise fusion)

Run: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python bench_levers.py --batch 2 --block 1024
"""
import sys, time, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

TOK = AutoTokenizer.from_pretrained("gpt2")
DEV = "cuda:0"


def build(batch, block, grad_ckpt=False, compile_ffn=False):
    torch.manual_seed(0)
    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    cfg.batch_size = batch; cfg.block_size = block
    cfg.gradient_checkpointing = bool(grad_ckpt)
    tok = TOK
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                    tie_weights=True, max_seq_len=cfg.block_size).to(DEV)
    m.train()
    try: m.emit_features_only = True
    except Exception: pass
    n_ffn = 0
    if compile_ffn:
        for mod in m.modules():
            if type(mod).__name__ == "SwiGLUFFN":
                mod.forward = torch.compile(mod.forward, dynamic=False)
                n_ffn += 1
    return m, n_ffn


def time_fb(m, ids, iters=8, warmup=5):
    dt = []
    for i in range(warmup + iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(ids); f = out[0] if isinstance(out, (tuple, list)) else out
            loss = f.float().pow(2).mean()
        loss.backward(); torch.cuda.synchronize()
        m.zero_grad(set_to_none=True)
        if i >= warmup:
            dt.append((time.perf_counter() - t0) * 1000.0)
    return sum(dt) / len(dt)


def run(name, ids, **kw):
    m, n_ffn = build(kw.pop("batch"), kw.pop("block"), **kw)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    try:
        ms = time_fb(m, ids)
        mem = torch.cuda.max_memory_allocated() / 1e9
        extra = f"  (compiled {n_ffn} FFN)" if n_ffn else ""
        print(f"  {name:32s} {ms:7.1f} ms   peak {mem:5.1f} GB{extra}")
    except Exception as e:
        print(f"  {name:32s} FAILED {type(e).__name__}: {str(e)[:90]}")
    finally:
        del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()
    torch.manual_seed(1)
    ids = torch.randint(0, TOK.vocab_size, (args.batch, args.block), device=DEV)
    b, bl = args.batch, args.block
    print(f"\n=== efficiency levers (batch={b} block={bl}, {torch.cuda.get_device_name(0)}) ===")
    run("baseline",                 ids, batch=b, block=bl)
    run("gradient_checkpointing",   ids, batch=b, block=bl, grad_ckpt=True)
    run("compile FFN",              ids, batch=b, block=bl, compile_ffn=True)
    run("grad_ckpt + compile FFN",  ids, batch=b, block=bl, grad_ckpt=True, compile_ffn=True)


if __name__ == "__main__":
    main()
