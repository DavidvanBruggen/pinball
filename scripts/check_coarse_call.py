#!/usr/bin/env python
"""Verify the flex coarse call (local_pack_level_windows / local_pack_down_highway) on this box.

The coarse call is a second flex_attention over coarse rows in (level, position) order,
merged into the union call by log-sum-exp. The merge is exact only if the two key sets are
disjoint for every query, so this checks that directly, then the numbers, then causality:

  1. disjoint     every packed key row reaches a query through at most one of the calls
  2. exact        merged flex output == dense fp32 softmax over the combined key set
  3. semantics    the coarse call's keys == the definition (same-level window, direct
                  children by the sequence builder's own rule, minus band and global block)
  4. causal       text config: perturbing inputs after p leaves outputs before p unchanged
                  (exactly 0), while a deliberately two-sided coarse mask leaks (control)
  5. live         flex_union_status(): failed_modules 0, coarse_call_live == coarse_call_layers
  6. gradients    out/dq/dk/dv of the merged call vs dense fp32 on the text 1024 highway arm
                  (the smallest real coarse call, 224 x 192)
  7. host pattern  no_grad warm-up forward, THEN a compiled training step on a permuted,
                  grad-requiring input -- what the ChromScape notebook does. The first merge
                  compiled here once and then failed AOT autograd's functional-graph check on
                  recompile (index_copy on flex's permuted output view), which checks 1-6 all
                  missed because none of them compiles the layer after a no_grad trace.

Checks 1-3 run on the DNA config with a 50-row global budget, so L3 sits outside the prefix
and the L4 -> L3 highway is exercised (at glob400's 400 budget every L4 child is prefix).

    python scripts/check_coarse_call.py                 # default cuda:0
    python scripts/check_coarse_call.py --device cuda:1

Prints PASS/FAIL per check and exits non-zero on any failure. Needs a GPU with flex.
"""
import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
logging.disable(logging.WARNING)

from pinball.instantiate_PINBALL_model import build_pinball  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DNA = os.path.join(REPO, "configs", "pinball_dna_bidi_linear_flexhier_glob400.yaml")
TEXT = os.path.join(REPO, "configs", "pinball_wikitext_pack_glob400.yaml")
TEXT_1024 = os.path.join(REPO, "configs", "pinball_wikitext_pack_pc_highway_glob32.yaml")
RADIUS = 32
FAILS = []


def _report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def _build(cfg, device, **ov):
    return build_pinball(cfg_path=cfg, num_tracks=768, tie_weights=False, device=device,
                         set_global_seed=False, override=dict(hier_layer_compile=False, **ov))[0]


def check_dense(device, mode, causal):
    m = _build(DNA, device, local_pack_global_block=50, local_pack_top_global_budget=50,
               local_pack_level_windows=RADIUS, local_pack_down_highway=mode).eval()
    with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
        m(torch.randn(1, 4096, 768, device=device))
    spec = m._local_pack_spec_cache[1]
    cc = spec["flex_cc"]
    layer = next(x for x in m.modules() if hasattr(x, "_flex_coarse_call_merge"))
    n, pre, w = int(spec["num_nodes"]), spec["flex_kv_prefix"], int(spec["window"])
    g_rows = int(pre.numel())
    h, d = layer.num_heads, layer.head_dim
    gen = torch.Generator(device=device).manual_seed(0)
    qp, kp, vp = (torch.randn(1, n, h, d, device=device, generator=gen).to(torch.bfloat16)
                  for _ in range(3))
    with torch.no_grad():
        out = layer._flex_union_attn(qp, kp, vp, spec, causal).float()
    tag = f"{mode}/{'causal' if causal else 'bidi'}"

    qi = torch.arange(n, device=device).view(-1, 1)
    a = spec["flex_block_mask"].mask_mod(0, 0, qi, torch.arange(g_rows + n, device=device).view(1, -1))
    qr, kr = cc["q_rows"], cc["k_rows"]
    b = spec["flex_cc_bm"].mask_mod(0, 0, torch.arange(qr.numel(), device=device).view(-1, 1),
                                    torch.arange(kr.numel(), device=device).view(1, -1))
    reach = torch.zeros(n, n, device=device, dtype=torch.int16)
    key_a = torch.cat([pre, torch.arange(n, device=device)])
    reach.index_put_((qi.expand(-1, g_rows + n)[a], key_a.view(1, -1).expand(n, -1)[a]),
                     torch.ones(int(a.sum()), device=device, dtype=torch.int16), accumulate=True)
    reach.index_put_((qr.view(-1, 1).expand(-1, kr.numel())[b], kr.view(1, -1).expand(qr.numel(), -1)[b]),
                     torch.ones(int(b.sum()), device=device, dtype=torch.int16), accumulate=True)
    _report(f"disjoint {tag}", int(reach.max()) == 1,
            f"max reach {int(reach.max())} over {int(b.sum())} coarse-call pairs")

    keys = reach > 0
    qf, kf, vf = (t[0].float().transpose(0, 1) for t in (qp, kp, vp))
    ref = torch.softmax(((qf @ kf.transpose(-1, -2)) / d ** 0.5).masked_fill(~keys, float("-inf")), -1) @ vf
    err = ((out[0].transpose(0, 1) - ref).abs().max() / ref.abs().max()).item()
    _report(f"exact {tag}", err < 1e-2, f"max rel err vs dense fp32 {err:.2e} (bf16 inputs)")
    if causal:
        ar = torch.arange(n, device=device)
        fut = int((keys & (ar.view(1, -1) > ar.view(-1, 1))).sum())
        _report(f"no future keys {tag}", fut == 0, f"{fut} keys rank after their query")

    lv, perm, glob = spec["levels"], spec["perm"], spec["global_block_mask"]
    comp = [0] + list(m.compression_ratios)
    ratio = [0] + [max(1, int(c * (1 - o))) for c, o in zip(m.compression_ratios, m.overlap_ratios)]
    off = torch.tensor([0] + torch.cumsum(torch.bincount(lv), 0).tolist(), device=device)
    lidx = perm - off[lv]
    lv_l, lidx_l, glob_l = lv.tolist(), lidx.tolist(), glob.tolist()
    bad = 0
    samples = torch.linspace(0, qr.numel() - 1, 40).long().tolist()
    for t in samples:
        row = int(qr[t]); lq, j = lv_l[row], lidx_l[row]
        want = set()
        for k in range(n):
            kl = lv_l[k]
            if glob_l[k] or (kl == 0 and mode != "l0"):
                continue
            ok = kl == lq and abs(lidx_l[k] - j) <= RADIUS
            if mode in ("children", "l0") and kl == lq - 1:
                n_lower = int(off[lq] - off[lq - 1])
                st = min(j * ratio[lq], n_lower - 1)
                ok = ok or (st <= lidx_l[k] < min(st + comp[lq], n_lower))
            dr = row - k
            band = (0 <= dr <= w) if causal else abs(dr) <= w
            if ok and not band and (dr >= 0 or not causal):
                want.add(k)
        got = set(kr[b[t]].tolist())
        if mode == "l0":
            got = {k for k in got if lv_l[k] > 0}
        bad += int(want != got)
    _report(f"semantics {tag}", bad == 0, f"{len(samples) - bad}/{len(samples)} sampled coarse queries")


def check_causal(device):
    import pinball.model.layers.hierarchical_message_passing as hmp
    cls = hmp.HierarchicalMessagePassing
    orig = cls._flex_coarse_call_merge

    def prefix_diff(leak):
        cls._flex_coarse_call_merge = (
            (lambda self, qp, kp, vp, spec, cc, causal, out, lse:
             orig(self, qp, kp, vp, spec, cc, False, out, lse)) if leak else orig)
        torch.manual_seed(0)
        m = _build(TEXT, device, hier_refresh_compile=False, local_pack_global_block=32,
                   local_pack_top_global_budget=32, local_pack_level_windows=RADIUS,
                   local_pack_down_highway="children").eval()
        gen = torch.Generator(device=device).manual_seed(3)
        x = torch.randn(1, 1024, 768, device=device, generator=gen)
        worst = 0.0
        for p in (200, 511, 700):
            x2 = x.clone()
            x2[:, p:] = torch.randn(1, 1024 - p, 768, device=device, generator=gen)
            with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
                worst = max(worst, (m(x).float()[:, :p] - m(x2).float()[:, :p]).abs().max().item())
        st = m.flex_union_status()
        return worst, st

    try:
        clean, st = prefix_diff(False)
        leak, _ = prefix_diff(True)
    finally:
        cls._flex_coarse_call_merge = orig
    _report("causal (text)", clean == 0.0 and leak > 0.0,
            f"prefix diff {clean:.3e} (must be 0) vs leak control {leak:.3e} (must be > 0)")
    _report("live (text)", st["failed_modules"] == 0
            and st["coarse_call_live_modules"] == st["coarse_call_layers"] > 0,
            f"flex failed {st['failed_modules']}, coarse call "
            f"{st['coarse_call_live_modules']}/{st['coarse_call_layers']}")


def check_grad(device):
    m = _build(TEXT_1024, device, hier_refresh_compile=False).eval()
    with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
        m(torch.randn(1, 1024, 768, device=device))
    spec = m._local_pack_spec_cache[1]
    cc = spec["flex_cc"]
    layer = next(x for x in m.modules() if hasattr(x, "_flex_coarse_call_merge"))
    n, pre = int(spec["num_nodes"]), spec["flex_kv_prefix"]
    g_rows, h, d = int(pre.numel()), layer.num_heads, layer.head_dim
    gen = torch.Generator(device=device).manual_seed(0)
    base = [torch.randn(1, n, h, d, device=device, generator=gen) for _ in range(3)]
    wgt = torch.randn(1, n, h, d, device=device, generator=gen)
    qp, kp, vp = (t.clone().to(torch.bfloat16).requires_grad_(True) for t in base)
    out = layer._flex_union_attn(qp, kp, vp, spec, True).float()
    (out * wgt).sum().backward()
    qi = torch.arange(n, device=device).view(-1, 1)
    a = spec["flex_block_mask"].mask_mod(0, 0, qi, torch.arange(g_rows + n, device=device).view(1, -1))
    qr, kr = cc["q_rows"], cc["k_rows"]
    b = spec["flex_cc_bm"].mask_mod(0, 0, torch.arange(qr.numel(), device=device).view(-1, 1),
                                    torch.arange(kr.numel(), device=device).view(1, -1))
    keys = torch.zeros(n, n, device=device, dtype=torch.bool)
    key_a = torch.cat([pre, torch.arange(n, device=device)])
    keys[qi.expand(-1, g_rows + n)[a], key_a.view(1, -1).expand(n, -1)[a]] = True
    keys[qr.view(-1, 1).expand(-1, kr.numel())[b], kr.view(1, -1).expand(qr.numel(), -1)[b]] = True
    qd, kd, vd = (t.clone().to(torch.bfloat16).float().requires_grad_(True) for t in base)
    qh, kh, vh = (t[0].transpose(0, 1) for t in (qd, kd, vd))
    ref = torch.softmax(((qh @ kh.transpose(-1, -2)) / d ** 0.5).masked_fill(~keys, float("-inf")), -1) @ vh
    ref = ref.transpose(0, 1).unsqueeze(0)
    (ref * wgt).sum().backward()
    rel = lambda x, y: ((x - y).abs().max() / y.abs().max()).item()
    errs = [rel(out, ref), rel(qp.grad.float(), qd.grad), rel(kp.grad.float(), kd.grad), rel(vp.grad.float(), vd.grad)]
    _report("gradients (text 1024, small coarse call)", max(errs) < 2e-2,
            f"{qr.numel()}q x {kr.numel()}k | out/dq/dk/dv rel err " + " ".join(f"{e:.1e}" for e in errs))


def check_host_pattern(device):
    ok, why = True, ""
    for mode in ("children", "l0"):
        torch._dynamo.reset()
        m = build_pinball(cfg_path=DNA, num_tracks=1024, tie_weights=False, device=device, set_global_seed=False,
                          override=dict(local_pack_level_windows=RADIUS, local_pack_down_highway=mode,
                                        hier_upward_refresh=False, hier_downward_refresh=False))[0]
        with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
            m(torch.randn(1, 1024, 4096, device=device).permute(0, 2, 1))
        m.train()
        x = torch.randn(2, 1024, 4096, device=device).permute(0, 2, 1).requires_grad_(True)
        try:
            with torch.amp.autocast("cuda", torch.bfloat16):
                out = m(x)
            out.float().mean().backward()
            live = m.flex_union_status()
            if live["coarse_call_live_modules"] != live["coarse_call_layers"]:
                ok, why = False, f"{mode}: coarse call {live['coarse_call_live_modules']}/{live['coarse_call_layers']}"
        except Exception as exc:  # noqa: BLE001 -- report, do not crash the other checks
            ok, why = False, f"{mode}: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
        del m
    _report("host pattern (no_grad warm-up, then compiled train step)", ok,
            why or "children and l0 both compile and train")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(torch.device(args.device))}")
    for mode, causal in (("children", False), ("children", True), ("l0", False)):
        check_dense(args.device, mode, causal)
    check_causal(args.device)
    check_grad(args.device)
    check_host_pattern(args.device)
    print("ALL PASS" if not FAILS else f"FAILED: {', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
