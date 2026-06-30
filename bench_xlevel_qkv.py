# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate per-level cross-level QKV (`cross_level_qkv` modes).

(1) build + fwd/bwd for each mode (shared / per_level_qk / per_level_qkv / reuse_local).
(2) parity: with copy-init from shared, per-level routing == shared path. Tested WITHIN one
    model (toggle each message_passing's cross_level_qkv) to avoid RNG drift between builds.
(3) AR causality (CPU): appending future tokens leaves settled past logits stable.

    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python bench_xlevel_qkv.py
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

TOK = AutoTokenizer.from_pretrained("gpt2")
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token


def build(mode, block, device, backend="flash"):
    torch.manual_seed(0)
    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    cfg.block_size = block; cfg.batch_size = 2
    cfg.cross_level_qkv = mode
    if str(device).startswith("cpu") and backend == "flash":
        backend = "sdpa"   # flash needs CUDA+bf16
    if backend != "flash":
        cfg.attn_backend = backend; cfg.l0_local_backend = backend
    m = build_model(cfg, tokenizer=TOK, vocab_size=len(TOK), input_mode="tokens",
                    tie_weights=True, max_seq_len=block).to(device)
    return m


def mps(m):
    for mod in m.modules():
        if hasattr(mod, "cross_level_qkv") and hasattr(mod, "_resolve_xlevel_lists"):
            yield mod


@torch.no_grad()
def logits(m, ids, device):
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else __import__("contextlib").nullcontext()
    rt = ids.clone(); rm = torch.zeros_like(ids, dtype=torch.bool)
    if ids.size(1) > 1:
        rt[:, :-1] = ids[:, 1:]; rm[:, :-1] = True
    with ctx:
        out = m(ids, attention_mask=None, reveal_target_ids=rt, reveal_mask=rm)
    if isinstance(out, dict):
        out = out.get("logits", out)
    f = out[0] if isinstance(out, (tuple, list)) else out
    return f.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", type=int, default=256)
    args = ap.parse_args()
    dev = args.device

    print("\n=== (1) build + fwd/bwd per mode ===")
    for mode in ("shared", "per_level_qk", "per_level_qkv", "reuse_local"):
        try:
            m = build(mode, args.block, dev); m.train()
            ids = torch.randint(0, TOK.vocab_size, (2, args.block), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(ids); f = out[0] if isinstance(out, (tuple, list)) else out
                loss = f.float().pow(2).mean()
            loss.backward()
            nmod = sum(1 for _ in mps(m))
            print(f"  {mode:14s}: fwd+bwd OK  loss={float(loss):.4f}  mp_modules={nmod}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {mode:14s}: FAILED {type(e).__name__}: {str(e)[:90]}")
            torch.cuda.empty_cache()

    print("\n=== (2) parity: per-level routing (copy-init) == shared, within one model (CPU fp32) ===")
    pdev = "cpu"   # fp32/deterministic so parity isn't masked by bf16 reduction-order noise
    for mode in ("per_level_qk", "per_level_qkv"):
        m = build(mode, 192, pdev); m.eval()
        # ensure xlevel == shared (fresh build already copy-inits; re-sync to be explicit)
        for mod in mps(m):
            mod._sync_cross_level_qkv_from_shared()
        ids = torch.randint(0, TOK.vocab_size, (2, 192), device=pdev)
        a = logits(m, ids, pdev)
        for mod in mps(m):
            mod.cross_level_qkv = "shared"          # toggle to shared path
        b = logits(m, ids, pdev)
        d = (a - b).abs().max().item()
        print(f"  {mode:14s}: max|per_level - shared| = {d:.3e}   {'PASS' if d < 1e-3 else 'FAIL'}")
        del m; torch.cuda.empty_cache()

    print("\n=== (3) AR causality (CPU, per_level_qk) ===")
    m = build("per_level_qk", 512, "cpu", backend="sdpa"); m.eval()
    for mod in mps(m):
        mod._sync_cross_level_qkv_from_shared()
    ids = torch.randint(0, TOK.vocab_size, (2, 512))
    P = 512 - 64; win = 128; settled = max(1, P - win)
    lb = logits(m, ids[:, :P], "cpu")[:, :settled]
    le = logits(m, ids, "cpu")[:, :settled]
    d = (lb - le).abs().max().item()
    print(f"  settled-past drift under append = {d:.3e}   {'PASS' if d < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
