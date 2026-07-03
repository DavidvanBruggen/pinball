# SPDX-License-Identifier: GPL-3.0-or-later
"""Frontier-consistent AR generation matches the training forward.

A raw prefix builds TRUNCATED coarse windows, so the frontier token reads its own OPEN parent;
in the full training forward that parent's window extends past it and is causal-cut. Padding the
prefix forward by the top-level coarse window span closes those windows with causally-invisible
pad tokens, so the frontier's projected logits equal the full-forward logits at that position.

This asserts the padded-frontier logits match the full teacher-forced logits bit-exactly on CPU,
and that the raw (unpadded) last-token logits do NOT — i.e. the fix is doing real work.

Run:  python tests/test_gen_frontier_consistency.py
"""
import sys, pathlib
import torch

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

CFG = dict(
    model_type="pinball", modality="text", tokenizer_name="gpt2",
    block_size=128, hidden_dim=64, num_heads=4, num_refinement_layers=4,
    num_layers=[0, 0, 0, 0], internal_cycles=[0, 0, 0, 0],
    refinement_style="unified", unified_refinement_cycles=1,
    compression_ratios=[16, 4, 4], overlap_ratios=[0.5, 0.5, 0.5],
    local_attn_levels=[1, 2, 3], local_attn_windows=[0, 8, 16, 32],
    local_attn_causal_levels=[1, 2, 3],
    dropout=0.0, norm_type="layernorm", l0_cycles=0,
    iterative_refinement_cycles=0, local_connectivity_window_size=0,
    l0_local_window=1, train_mode="ar", ar_graph_causal=True,
    use_hqd=False, upper_init="pooled",
)


def test_frontier_consistency():
    torch.manual_seed(0)
    cfg = PinballConfig(**CFG)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                        tie_weights=True, max_seq_len=cfg.block_size).eval()

    look = model._gen_frontier_lookahead()
    assert look > 0, look
    T = cfg.block_size
    torch.manual_seed(1)
    ids = torch.randint(0, tok.vocab_size, (1, T))
    pad = tok.pad_token_id or 0

    def L(x):
        with torch.no_grad():
            o = model(x, logits_last_only=False) if x.size(1) == T else None
        return o

    with torch.no_grad():
        full = model(ids, logits_last_only=False)
        full = (full[0] if isinstance(full, (tuple, list)) else full)

    worst_fc, best_raw = 0.0, 0.0
    for k in (32, 64, 96, 112):
        with torch.no_grad():
            raw = model(ids[:, :k], logits_last_only=True)
            raw = (raw[0] if isinstance(raw, (tuple, list)) else raw)[:, -1, :]
            Lp = min(T, k + look)
            padded = torch.cat([ids[:, :k], torch.full((1, Lp - k), pad, dtype=ids.dtype)], 1)
            fc = model(padded, logits_last_index=k - 1)
            fc = (fc[0] if isinstance(fc, (tuple, list)) else fc)[:, -1, :]
        d_fc = (fc - full[:, k - 1]).abs().max().item()
        d_raw = (raw - full[:, k - 1]).abs().max().item()
        print(f"k={k:>4}  frontier-consistent Δ={d_fc:.2e}   raw-last Δ={d_raw:.2e}")
        worst_fc = max(worst_fc, d_fc)
        best_raw = max(best_raw, d_raw)

    assert worst_fc < 1e-5, f"frontier-consistent logits drifted from training: {worst_fc:.2e}"
    assert best_raw > 1e-2, f"raw path unexpectedly already consistent ({best_raw:.2e}) — test not exercising the fix"
    print("PASS")


if __name__ == "__main__":
    test_frontier_consistency()
