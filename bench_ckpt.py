# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate refinement-layer gradient checkpointing: (1) peak memory off vs on across
batch sizes (where activations dominate), (2) correctness — loss matches within bf16 tol.

Run: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python bench_ckpt.py --block 1024
"""
import sys, time, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

TOK = AutoTokenizer.from_pretrained("gpt2")
DEV = "cuda:0"


def build(batch, block, grad_ckpt):
    torch.manual_seed(0)
    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    cfg.batch_size = batch; cfg.block_size = block
    # NOTE: `gradient_checkpointing` is only aliased to `use_gradient_checkpointing`
    # at from_dict load time; set the real attribute the model reads directly.
    cfg.use_gradient_checkpointing = bool(grad_ckpt)
    tok = TOK
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                    tie_weights=True, max_seq_len=cfg.block_size).to(DEV)
    m.train()
    try: m.emit_features_only = True
    except Exception: pass
    return m


def step(m, ids, want_loss=False):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(ids); f = out[0] if isinstance(out, (tuple, list)) else out
        loss = f.float().pow(2).mean()
    loss.backward()
    lv = float(loss.detach())
    m.zero_grad(set_to_none=True)
    return lv


def run(batch, block):
    torch.manual_seed(1)
    ids = torch.randint(0, TOK.vocab_size, (batch, block), device=DEV)
    res = {}
    for ck in (False, True):
        m = build(batch, block, ck)
        for _ in range(3): step(m, ids)          # warmup
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = step(m, ids, want_loss=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        mem = torch.cuda.max_memory_allocated() / 1e9
        res[ck] = (ms, mem, loss)
        del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    (ms0, mem0, l0), (ms1, mem1, l1) = res[False], res[True]
    dl = abs(l0 - l1) / max(1e-9, abs(l0))
    print(f"  batch={batch:2d}: OFF {ms0:6.1f}ms {mem0:5.2f}GB  |  ON(ckpt) {ms1:6.1f}ms {mem1:5.2f}GB  "
          f"|  mem -{(1-mem1/mem0)*100:4.1f}%  time +{(ms1/ms0-1)*100:4.1f}%  loss_rel_diff={dl:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--batches", type=str, default="2,8,16")
    args = ap.parse_args()
    print(f"\n=== refinement-layer gradient checkpointing ({torch.cuda.get_device_name(0)}, block={args.block}) ===")
    for b in [int(x) for x in args.batches.split(",")]:
        try:
            run(b, args.block)
        except Exception as e:
            print(f"  batch={b:2d}: FAILED {type(e).__name__}: {str(e)[:80]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
