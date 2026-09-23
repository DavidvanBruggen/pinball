"""Verify FlashAttention-4 on this machine before trusting any run that uses it.

FA4 (flash_attn.cute, CuTe DSL) only runs on sm_90 / sm_100, so none of this can be
checked on the workstation cards. Run it once per new environment (container image,
torch or flash_attn upgrade) on a GPU node:

    python scripts/check_fa4.py                 # all checks
    python scripts/check_fa4.py --skip-model    # kernels only, no model build

Four checks, each printed PASS / FAIL / SKIP, exit code 1 on any FAIL:

1. L0 window, `flash_impl: fa4` -- does the picker resolve to fa4, and does the
   sliding-window kernel match a dense SDPA reference (forward output and all three
   input gradients, causal and bidirectional)?
2. The same through the FA2-dialect adapter's LSE path under no_grad, which the
   lse-merge code needs at eval time.
3. flex_attention with kernel_options BACKEND="FLASH" against BACKEND="TRITON" on a
   pinball-shaped block mask (band + global prefix), forward and backward, plus timing.
4. The real glob400 DNA config with local_pack_flex_backend: flash and flash_impl: fa4:
   forward + backward; flex_union_status() must report flash_live_modules > 0 and
   flash_failed_modules == 0 (a fallback to Triton would otherwise look fine), and step
   time against the triton/fa2 baseline. The baseline needs FA2 installed; without it the
   picker resolves to SDPA and the speed ratio is not a flash-vs-flash comparison.

Tolerances are bf16-sized (relative max error < 2e-2 on outputs, < 5e-2 on grads). A
kernel that is merely faster but outside them is a FAIL -- speed is only worth having
on the same function.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import pinball.model.layers.hierarchical_message_passing as hmp  # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print(f"[{tag}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if ok is False:
        FAILS.append(name)


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def band_mask(n, left, right, device):
    i = torch.arange(n, device=device)
    d = i[None, :] - i[:, None]           # key - query
    return (d >= -left) & (d <= right)


def check_window_kernel(dev):
    hmp.set_flash_preference("fa4")
    name, fn = hmp.pick_attention_backend(dev)
    report("picker resolves flash_impl=fa4", name == "fa4", f"got {name!r}")
    if name != "fa4":
        return
    torch.manual_seed(0)
    B, S, H, D = 2, 2048, 8, 64
    for causal, win in ((True, (256, 0)), (False, (256, 256))):
        q, k, v = (torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
                   for _ in range(3))
        out = fn(q, k, v, causal=causal, window_size=win, dropout_p=0.0)
        g = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        m = band_mask(S, win[0], win[1], dev)
        ref = F.scaled_dot_product_attention(
            *(t.transpose(1, 2).float() for t in (q, k, v)), attn_mask=m).transpose(1, 2)
        rq, rk, rv = torch.autograd.grad(ref, (q, k, v), g.float())
        e = [rel(out, ref), rel(dq, rq), rel(dk, rk), rel(dv, rv)]
        ok = e[0] < 2e-2 and max(e[1:]) < 5e-2
        report(f"fa4 window {'causal' if causal else 'bidi'} {win} vs dense SDPA", ok,
               "rel err out/dq/dk/dv = " + " ".join(f"{x:.1e}" for x in e))
    with torch.no_grad():
        q = torch.randn(1, 1024, 4, 64, device=dev, dtype=torch.bfloat16)
        try:
            o, lse, _ = fn(q, q, q, causal=False, window_size=(64, 64), return_attn_probs=True)
            report("fa4 LSE under no_grad (eval-time lse merge)", lse is not None,
                   f"lse shape {tuple(lse.shape) if lse is not None else None}")
        except Exception as exc:
            report("fa4 LSE under no_grad (eval-time lse merge)", False, repr(exc)[:200])


def check_flex_flash(dev):
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except Exception as exc:
        report("flex FLASH backend", None, f"no flex_attention: {exc!r}")
        return
    B, H, N, D, W, G = 1, 8, 8192, 64, 512, 256

    def mask_mod(b, h, qi, ki):          # pinball-shaped: band + global prefix rows
        return ((qi - ki).abs() <= W) | (ki < G) | (qi < G)

    bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=str(dev),
                           BLOCK_SIZE=(128, 128))
    fx = torch.compile(flex_attention, dynamic=False)
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    g = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16)
    res = {}
    for be in ("TRITON", "FLASH"):
        try:
            o = fx(q, k, v, block_mask=bm, kernel_options={"BACKEND": be})
            grads = torch.autograd.grad(o, (q, k, v), g)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                o2 = fx(q, k, v, block_mask=bm, kernel_options={"BACKEND": be})
                torch.autograd.grad(o2, (q, k, v), g)
            torch.cuda.synchronize()
            res[be] = (o, grads, (time.perf_counter() - t0) / 10 * 1e3)
        except Exception as exc:
            report(f"flex BACKEND={be} runs", False, f"{type(exc).__name__}: {str(exc)[:200]}")
    if "TRITON" in res and "FLASH" in res:
        (ot, gt, tt), (of, gf, tf) = res["TRITON"], res["FLASH"]
        e = [rel(of, ot)] + [rel(a, b) for a, b in zip(gf, gt)]
        ok = e[0] < 2e-2 and max(e[1:]) < 5e-2
        report("flex FLASH vs TRITON (band + prefix mask, fwd+bwd)", ok,
               "rel err out/dq/dk/dv = " + " ".join(f"{x:.1e}" for x in e)
               + f" | fwd+bwd {tt:.2f} ms triton, {tf:.2f} ms flash ({tt / tf:.2f}x)")


def check_model(dev, block):
    from pinball import build_pinball
    cfg = os.path.join(REPO, "configs", "pinball_dna_bidi_linear_flexhier_glob400.yaml")

    def run(over, label):
        torch.manual_seed(0)
        m, *_ = build_pinball(cfg, device=str(dev), override={"block_size": block, **over},
                              num_tracks=64, set_global_seed=True, warn_unused_keys=False)
        m.train()
        x = torch.randn(1, block, 64, device=dev)
        times = []
        for i in range(6):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(x)
            y = y[0] if isinstance(y, (tuple, list)) else y
            y.float().square().mean().backward()
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        st = m.flex_union_status()
        name, _ = hmp.pick_attention_backend(dev)
        ms = sorted(times[2:])[len(times[2:]) // 2] * 1e3
        print(f"    {label}: l0={name} flex={st['backend']} failed={st['failed_modules']} "
              f"flash live/failed={st['flash_live_modules']}/{st['flash_failed_modules']} step {ms:.0f} ms", flush=True)
        del m; torch.cuda.empty_cache()
        return st, name, ms

    base_st, base_l0, base_ms = run({"flash_impl": "fa2", "local_pack_flex_backend": "triton"},
                                    "baseline fa2 + flex triton")
    st, l0, ms = run({"flash_impl": "fa4", "flash_nodropout_mode": "token_v",
                      "local_pack_flex_backend": "flash"}, "fa4 + flex FLASH")
    report("model: flex union live", st["failed_modules"] == 0, str(st["reasons"])[:200])
    report("model: flex FLASH live, no fallback",
           st["flash_failed_modules"] == 0 and st["flash_live_modules"] > 0,
           f"live {st['flash_live_modules']} / failed {st['flash_failed_modules']} / "
           f"requested {st['flash_modules']} {str(st['reasons'])[:160]}")
    report("model: L0 windows on fa4", l0 == "fa4", f"got {l0}")
    print(f"    step time {base_ms:.0f} -> {ms:.0f} ms ({base_ms / ms:.2f}x); NOTE the fa4 arm "
          "uses token_v dropout on the L0 windows, a different regulariser than fa2", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--block", type=int, default=8192)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device"); sys.exit(2)
    dev = torch.device("cuda", 0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | torch {torch.__version__}")
    for mod in ("flash_attn", "flash_attn.cute", "flash_attn_interface"):
        try:
            __import__(mod); print(f"  {mod}: importable")
        except Exception as exc:
            print(f"  {mod}: NOT importable ({type(exc).__name__}: {str(exc)[:120]})")
    if cap[0] not in (9, 10):
        report("FA4-capable GPU", False, "FA4 needs sm_90 or sm_100"); sys.exit(1)
    check_window_kernel(dev)
    check_flex_flash(dev)
    if not a.skip_model:
        check_model(dev, a.block)
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAIL: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
