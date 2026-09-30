#!/usr/bin/env python
"""Verify level-specific weights (hier_level_qkv / hier_level_ffn) on text glob400, both modes:
only level keys are new, identity at init (copy-init; fp32 ~3e-6, bf16 ~1 ulp from sliced
GEMM accumulation order), a fresh build is copy-initialized, groups change the output and get
gradient, and the causal prefix is untouched. Off bit-identity is checked separately against
the committed code (see docs/hierarchy_experiments.md).

    python scripts/check_level_weights.py
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_here, "check_upper_stage.py")).read().split("# 1. identity at init")[0])
G400 = REPO + "pinball_wikitext_pack_glob400.yaml"
x = dna_in(1, 1024)
def fwd(m, xx=x):
    with torch.no_grad(), ac():
        o = m(xx)
    return (o[0] if isinstance(o, tuple) else o).float()
off = build(G400, hier_layer_compile=False).eval()
yo = fwd(off)
for mode in ("l0_coarse", "per_level"):
    print(f"== {mode}")
    on = build(G400, hier_layer_compile=False, hier_level_qkv=mode, hier_level_ffn=mode).eval()
    extra = sum(p.numel() for n, p in on.named_parameters() if "_lg." in n)
    miss, unexp = on.load_state_dict(off.state_dict(), strict=False)
    rep("only level keys missing", all("_lg." in k for k in miss) and not unexp, f"{len(miss)} missing, +{extra/1e6:.1f}M params")
    # copy-init: groups == shared -> after loading shared, re-sync
    for m_ in on.modules():
        if hasattr(m_, "sync_level_qkv_from_shared") and getattr(m_, "hier_level_qkv", "shared") != "shared": m_.sync_level_qkv_from_shared()
        if hasattr(m_, "sync_level_ffn_from_shared") and getattr(m_, "hier_level_ffn", "shared") != "shared": m_.sync_level_ffn_from_shared()
    d = (fwd(on) - yo).abs().max().item()
    rep("identity at init (bf16)", d < 5e-2, f"max|diff| {d:.3e}")
    with torch.no_grad():
        d32 = (on(x).float() - off(x).float()).abs().max().item()
    rep("identity at init (fp32)", d32 < 1e-4, f"max|diff| {d32:.3e}")
    # fresh-built model: groups are copies of shared (the init re-sync)
    fresh = build(G400, hier_layer_compile=False, hier_level_qkv=mode, hier_level_ffn=mode)
    mp = fresh.refinement_transformers[0].message_passing; ly = fresh.refinement_transformers[0]
    same = all(torch.equal(mp.q_proj.weight, g.weight) for g in mp.q_proj_lg) and all(torch.equal(ly.ffn.down_proj.weight, g.down_proj.weight) for g in ly.ffn_lg)
    rep("fresh build copy-init", same, f"{len(mp.q_proj_lg)} qkv groups, {len(ly.ffn_lg)} ffn groups per layer")
    # grads on every group after perturbing groups
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in on.named_parameters():
            if "_lg." in n and p.dim() >= 2: p.add_((torch.randn(p.shape, generator=g) * 0.01).to(p))
    d2 = (fwd(on) - yo).abs().max().item()
    rep("groups change output", d2 > 1e-3, f"max|diff| {d2:.3e}")
    on.train()
    with ac(): loss = on(x.clone().requires_grad_(True)).float().pow(2).mean()
    loss.backward()
    zg = [n for n, p in on.named_parameters() if "_lg." in n and p.dim() >= 2 and (p.grad is None or p.grad.abs().max() == 0)]
    rep("grad on every group matrix", not zg, f"zero-grad {zg[:3]}")
    on.eval()
    # causality
    p0 = 700; x2 = x.clone(); x2[:, p0:] = torch.randn(1, 1024 - p0, 768).to(dev)
    leak = (fwd(on)[:, :p0] - fwd(on, x2)[:, :p0]).abs().max().item()
    rep("causal", leak == 0.0, f"max|diff| before p {leak:.3e}")
    del on, fresh
print("FAILS:", FAILS or "none")
