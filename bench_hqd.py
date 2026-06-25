# SPDX-License-Identifier: GPL-3.0-or-later
"""Find the HQD bottleneck: edge count + descent stages vs attention apply.

Runs on a free GPU. (1) confirms the skeleton cache hits across steps, (2) shows
how edge count / descent time / apply time move with the knobs (l0_topk_enable,
bag cap, read_levels), (3) compares scatter|dense|flash end-to-end on a bounded
config where the attention kernel can actually matter.
"""
import sys, time, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

TOK = AutoTokenizer.from_pretrained("gpt2")


def build(overrides):
    cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    tok = TOK
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok),
                        input_mode="tokens", tie_weights=True,
                        max_seq_len=cfg.block_size).to("cuda:0")
    return model


def fwd_bwd(model, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        loss = feats.float().pow(2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def timed(model, ids, iters=8, warmup=3):
    dt = []
    for i in range(warmup + iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fwd_bwd(model, ids)
        torch.cuda.synchronize()
        if i >= warmup:
            dt.append(time.perf_counter() - t0)
    return sum(dt) / len(dt) * 1000.0


def probe(name, overrides, ids):
    """Two forwards (so skeleton is cached on the 2nd); report edges + stage ms."""
    model = build({**overrides, "hqd_attn_impl": "scatter", "hqd_debug": True})
    model.train()
    skel = []
    for _ in range(2):
        fwd_bwd(model, ids); torch.cuda.synchronize()
        ps = getattr(model, "_last_hqd_profile_stats", {}) or {}
        skel.append(ps.get("skeleton_ms", 0.0))
    added = getattr(model, "_last_hqd_added_total", 0)
    ss = getattr(model, "_last_hqd_stage_stats", {}) or {}
    sel = {k: ss.get(k, 0) for k in ("l0_selected_total", "final_selected_total")}
    desc = sum(ps.get(k, 0.0) for k in ("l3_stage_ms", "l2_stage_ms", "l1_stage_ms", "l0_stage_ms"))
    print(f"  {name:28s} edges={added:>9d}  desc(score)={desc:6.1f}ms  apply={ps.get('apply_ms',0):5.1f}ms  skel[1st,2nd]={skel[0]:.0f},{skel[1]:.0f}ms")
    del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()
    torch.manual_seed(0)
    ids = torch.randint(0, TOK.vocab_size, (args.batch, args.block), device="cuda:0")
    full = dict(batch_size=args.batch, block_size=args.block)

    print(f"\n=== edge count + stage breakdown (batch={args.batch} block={args.block}) ===")
    probe("current (l0_topk=False)", full, ids)
    probe("l0_topk_enable=True", {**full, "hqd_l0_topk_enable": True}, ids)
    probe("+ bag_topk=8", {**full, "hqd_l0_topk_enable": True, "hqd_window_bag_topk": 8}, ids)
    probe("+ read_levels=[0]", {**full, "hqd_l0_topk_enable": True, "hqd_read_levels": [0]}, ids)
    probe("bounded (topk+bag8+read0)", {**full, "hqd_l0_topk_enable": True, "hqd_window_bag_topk": 8, "hqd_read_levels": [0]}, ids)
    probe("descent_stop_at=2", {**full, "hqd_l0_topk_enable": True, "hqd_descent_stop_at": 2}, ids)

    bounded = {**full, "hqd_l0_topk_enable": True, "hqd_window_bag_topk": 8}
    print(f"\n=== fwd+bwd ms, impls on bounded config ===")
    for impl, backend in [("scatter", "-"), ("dense", "sdpa"), ("dense", "flash")]:
        try:
            model = build({**bounded, "hqd_attn_impl": impl, "hqd_dense_backend": backend})
            ms = timed(model, ids)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {impl:7s}/{backend:5s}: {ms:7.1f} ms   peak {mem:5.1f} GB")
        except Exception as e:
            print(f"  {impl:7s}/{backend:5s}: FAILED {type(e).__name__}: {str(e)[:100]}")
        finally:
            try: del model
            except Exception: pass
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    print(f"\n=== fwd+bwd ms, current (unbounded) config ===")
    for impl, backend in [("scatter", "-"), ("dense", "sdpa"), ("dense", "flash")]:
        try:
            model = build({**full, "hqd_attn_impl": impl, "hqd_dense_backend": backend})
            ms = timed(model, ids)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {impl:7s}/{backend:5s}: {ms:7.1f} ms   peak {mem:5.1f} GB")
        except Exception as e:
            print(f"  {impl:7s}/{backend:5s}: FAILED {type(e).__name__}: {str(e)[:100]}")
        finally:
            try: del model
            except Exception: pass
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
