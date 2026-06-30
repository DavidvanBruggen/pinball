# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate Tier-2 flash-ALiBi local attention against the dense SDPA role-bias path.

Checks:
  (1) flag ON activates the flash-alibi path (counts non-None slope returns).
  (2) flag ON output ≈ flag OFF output (same weights) within flash-vs-sdpa bf16 tolerance
      -> confirms the ALiBi sign/scale reproduces base_bias + (-gate*dist/window).
  (3) AR causality: perturbing a FUTURE token leaves earlier positions ~unchanged.
  (4) speed: fwd+bwd ms and peak mem, ON vs OFF.

Run:  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python bench_tier2_validate.py --batch 2 --block 1024
"""
import sys, time, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig
import pinball.model.layers.hierarchical_message_passing as hmp

TOK = AutoTokenizer.from_pretrained("gpt2")
DEV = "cuda:0"


def build(alibi, batch, block):
    torch.manual_seed(0)
    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    cfg.batch_size = batch; cfg.block_size = block
    cfg.local_attn_flash_alibi = bool(alibi)
    tok = TOK
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                    tie_weights=True, max_seq_len=cfg.block_size).to(DEV)
    m.eval()
    try:
        m.emit_features_only = True
    except Exception:
        pass
    return m


def feats(m, ids):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(ids)
    return (out[0] if isinstance(out, (tuple, list)) else out).float()


def count_alibi_activations(m, ids):
    n = {"hit": 0, "miss": 0}
    orig = hmp.HierarchicalMessagePassing._maybe_local_alibi_slopes
    def wrapped(self, *a, **k):
        r = orig(self, *a, **k)
        n["hit" if r is not None else "miss"] += 1
        return r
    hmp.HierarchicalMessagePassing._maybe_local_alibi_slopes = wrapped
    try:
        feats(m, ids)
    finally:
        hmp.HierarchicalMessagePassing._maybe_local_alibi_slopes = orig
    return n


def time_fb(m, ids, iters=8, warmup=4):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()
    torch.manual_seed(1)
    ids = torch.randint(0, TOK.vocab_size, (args.batch, args.block), device=DEV)
    ids2 = ids.clone()
    j = args.block - 8
    ids2[:, j] = (ids2[:, j] + 12345) % TOK.vocab_size

    print("building OFF model..."); m_off = build(False, args.batch, args.block)
    print("building ON model...");  m_on = build(True, args.batch, args.block)

    # (1) activation
    n = count_alibi_activations(m_on, ids)
    print(f"\n[1] flash-alibi activations: hit={n['hit']} miss={n['miss']} "
          f"(OFF model should be all-miss)")
    n_off = count_alibi_activations(m_off, ids)
    print(f"    OFF model activations: hit={n_off['hit']} miss={n_off['miss']}")

    # (2) numerical equivalence ON vs OFF (same weights)
    f_off = feats(m_off, ids)
    f_on = feats(m_on, ids)
    diff = (f_off - f_on).abs()
    rel = diff.max().item() / max(1e-6, f_off.abs().max().item())
    print(f"\n[2] |OFF - ON| max={diff.max().item():.3e} mean={diff.mean().item():.3e} "
          f"rel_max={rel:.3e}  (expect small: flash vs sdpa bf16)")

    # (3) causality on the ON model
    base = feats(m_on, ids)
    past = (base - feats(m_on, ids2)).abs().flatten(start_dim=2).amax(-1)[0][:j].max().item()
    print(f"\n[3] ON causality: max|delta| at positions < {j} = {past:.3e} (expect ~0)")

    # (4) speed
    del f_off, f_on, base; torch.cuda.empty_cache()
    for m in (m_off, m_on):
        m.train()
        try: m.emit_features_only = True
        except Exception: pass
    torch.cuda.reset_peak_memory_stats()
    t_off = time_fb(m_off, ids); mem_off = torch.cuda.max_memory_allocated()/1e9
    torch.cuda.reset_peak_memory_stats()
    t_on = time_fb(m_on, ids); mem_on = torch.cuda.max_memory_allocated()/1e9
    print(f"\n[4] fwd+bwd:  OFF={t_off:.1f}ms peak={mem_off:.1f}GB   "
          f"ON={t_on:.1f}ms peak={mem_on:.1f}GB   speedup={t_off/max(1e-6,t_on):.2f}x")


if __name__ == "__main__":
    main()
