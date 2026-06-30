# SPDX-License-Identifier: GPL-3.0-or-later
"""Profile the current AR generation path: it calls full forward(current_ids) per token
(rebuild_graph=True), i.e. O(N^2) decoding. Measure ms/token and how it scales with prefix
length -> quantifies the prize for KV-cached incremental generation.

Run: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python bench_generation.py --prompt 256 --new 12
"""
import sys, time, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

TOK = AutoTokenizer.from_pretrained("gpt2")
DEV = "cuda:0"


def build(block):
    torch.manual_seed(0)
    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    cfg.batch_size = 1; cfg.block_size = block
    tok = TOK
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                    tie_weights=True, max_seq_len=block).to(DEV)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--new", type=int, default=12)
    args = ap.parse_args()
    block = args.prompt + args.new + 8
    m = build(block)
    torch.manual_seed(1)
    ids = torch.randint(0, TOK.vocab_size, (1, args.prompt), device=DEV)

    # Per-token wall time by calling forward over a growing prefix (mirrors generate()).
    print(f"\n=== current AR generation cost (prompt={args.prompt}, +{args.new} tokens) ===")
    cur = ids.clone()
    # warmup (builds caches for the prompt length)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        m(cur, logits_last_only=True)
    torch.cuda.synchronize()
    per_tok = []
    total0 = time.perf_counter()
    for step in range(args.new):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = m(cur, logits_last_only=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
        per_tok.append((cur.size(1) - 1, dt))
    total = (time.perf_counter() - total0) * 1000.0
    for plen, dt in per_tok:
        print(f"  prefix_len={plen:5d} -> {dt:7.1f} ms/token")
    avg = sum(d for _, d in per_tok) / len(per_tok)
    print(f"  avg = {avg:.1f} ms/token over {args.new} tokens; total {total:.0f} ms")
    print(f"  => generating 512 tokens at this avg ~= {avg*512/1000:.1f} s (and grows with prefix)")
    print(f"  peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
