# SPDX-License-Identifier: GPL-3.0-or-later
"""Decisive test for KV-cached generation feasibility: when we APPEND a token (which
rebuilds the hierarchy — coarse node count/windows change), do the PAST positions' outputs
stay the same?

  exact cache possible  <=>  forward(ids[:P+k])[:, :P] == forward(ids[:P])   (within bf16)

If past positions are stable, incremental decode is EXACT (cache settled K/V, recompute only
the new token + frontier coarse nodes). If they drift, quantify the error per appended token
and per distance-from-frontier (tells us how many recent positions need recompute).

Run: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python bench_gen_causal_stability.py --prompt 256 --new 4
"""
import sys, pathlib, argparse
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
    return build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                       tie_weights=True, max_seq_len=block).to(DEV).eval()


def logits(m, ids):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m(ids, logits_last_only=False).float()  # [1, T, V]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--new", type=int, default=4)
    args = ap.parse_args()
    P = args.prompt
    block = P + args.new + 8
    m = build(block)
    torch.manual_seed(1)
    full = torch.randint(0, TOK.vocab_size, (1, P + args.new), device=DEV)

    base = logits(m, full[:, :P])               # [1, P, V]
    scale = base.abs().max().item()
    print(f"\n=== causal stability under APPEND (prompt={P}) ===")
    print(f"  (compare first P positions of forward(prefix P+k) vs forward(prefix P))")
    for k in range(1, args.new + 1):
        ext = logits(m, full[:, :P + k])         # [1, P+k, V]
        d = (ext[:, :P, :] - base).abs()         # change at the original P positions
        # change as a function of distance from the (old) frontier position P-1
        per_pos = d.flatten(0, 1).amax(-1)       # [P] max-abs logit change per position
        far = per_pos[:P - 64].max().item() if P > 64 else float("nan")
        near = per_pos[P - 64:].max().item()
        print(f"  +{k} tok: max|dlogit|={d.max().item():.3e} (rel {d.max().item()/scale:.2e})  "
              f"near-frontier(<64)={near:.3e}  far-past(>64 back)={far:.3e}")
    print("\n  interpretation: far-past≈bf16 noise => settled positions are STABLE (exact KV "
          "cache); only the last ~window positions drift => recompute just the frontier.")


if __name__ == "__main__":
    main()
