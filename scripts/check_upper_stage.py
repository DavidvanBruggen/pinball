#!/usr/bin/env python
"""Verify the upper stage (local_pack_upper_*): extra compute cycles on the upper hierarchy.

  1. identity    a stage-on model loaded with a stage-off model's weights gives the SAME output
                 (zero-init residual outputs; guards the model-wide _init_weights re-randomizing them)
  2. live/grad   nonzero stage weights change the output; every stage matrix gets gradient
  3. variants    per_level qkv/ffn, fat ffn, extend, from_level, depth all build and run
  4. mask        the stage mask == a brute-force definition (DNA extend 3; text causal extend 2)
  5. causal      text: perturbing inputs >= p leaves outputs < p unchanged (exactly 0), while a
                 deliberately two-sided mask leaks (control)

    python scripts/check_upper_stage.py            # cuda
Prints PASS/FAIL per check and exits non-zero on any failure.
"""
import logging, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
logging.disable(logging.WARNING)
from pinball.instantiate_PINBALL_model import build_pinball

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs") + os.sep
DNA = REPO + "pinball_dna_bidi_linear_flexhier_glob400_dropkey_highway.yaml"
TEXT = REPO + "pinball_wikitext_pack_pc_highway_glob32.yaml"
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
FAILS = []


def rep(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def build(cfg, seed=0, **ov):
    torch.manual_seed(seed)
    m, tok, args, _ = build_pinball(cfg_path=cfg, num_tracks=768, tie_weights=False, device=dev,
                                    set_global_seed=False, override=dict(ov))
    return m


def randomize_stage(m, scale=0.02):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "upper_stage_blocks" in n and p.dim() >= 2:
                p.copy_((torch.randn(p.shape, generator=g) * scale).to(p))


def dna_in(B=1, L=4096):
    g = torch.Generator().manual_seed(2)
    return torch.randn(B, L, 768, generator=g).to(dev)


ac = (lambda: torch.amp.autocast("cuda", torch.bfloat16)) if dev == "cuda" else (lambda: torch.autocast("cpu", enabled=False))

# 1. identity at init: on-model loaded with off-model weights == off-model exactly
print("1. identity at init (DNA)")
off = build(DNA, hier_layer_compile=False).eval()
on = build(DNA, hier_layer_compile=False, local_pack_upper_every=3).eval()
missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
rep("only stage keys missing", all("upper_stage_blocks" in k for k in missing) and not unexpected,
    f"{len(missing)} missing, {len(unexpected)} unexpected")
x = dna_in()
with torch.no_grad(), ac():
    yo = off(x); yn = on(x)
d = (yo.float() - yn.float()).abs().max().item()
st = on.flex_union_status()
rep("identity", d == 0.0, f"max|diff| {d:.3e}")
rep("live", st["upper_stage_runs"] > 0, f"stages {st['upper_stage_stages']} runs {st['upper_stage_runs']} info {st['upper_stage_info']}")
nst = sum(p.numel() for n, p in on.named_parameters() if "upper_stage_blocks" in n)
print(f"     stage params {nst/1e6:.2f}M of {sum(p.numel() for p in on.parameters())/1e6:.2f}M")

# 2. non-identity once weights are nonzero; gradients reach every stage matrix
print("2. live + gradients (DNA, train mode, dropout on)")
randomize_stage(on)
with torch.no_grad(), ac():
    yn2 = on(x)
d2 = (yo.float() - yn2.float()).abs().max().item()
rep("stage changes output", d2 > 1e-4, f"max|diff| {d2:.3e}")
on.train()
with ac():
    loss = on(x).float().pow(2).mean()
loss.backward()
gz = [n for n, p in on.named_parameters() if "upper_stage_blocks" in n and p.dim() >= 2 and (p.grad is None or p.grad.abs().max() == 0)]
rep("grad on every stage matrix", not gz, f"zero-grad: {gz[:4]}")
on.zero_grad(set_to_none=True)

# 3. variants build and run
print("3. variants (DNA)")
for ov in (dict(local_pack_upper_qkv="per_level", local_pack_upper_from_level=3), dict(local_pack_upper_ffn="per_level", local_pack_upper_from_level=3),
           dict(local_pack_upper_ffn_dim=4096), dict(local_pack_upper_extend=2),
           dict(local_pack_upper_from_level=1, local_pack_upper_qkv="per_level", local_pack_upper_ffn="per_level"),
           dict(local_pack_upper_depth=2)):
    m = build(DNA, hier_layer_compile=False, local_pack_upper_every=3, **ov).eval()
    randomize_stage(m)
    with torch.no_grad(), ac():
        y = m(x)
    s = m.flex_union_status()
    nst = sum(p.numel() for n, p in m.named_parameters() if "upper_stage_blocks" in n)
    rep(f"{ov}", bool(torch.isfinite(y.float()).all()) and s["upper_stage_runs"] > 0,
        f"{s['upper_stage_info']} params {nst/1e6:.1f}M")
    del m

# 4. mask semantics vs brute force (DNA, extend 3 so lane/children/core all appear)
print("4. mask semantics (DNA extend 3, and text causal extend 2)")
def brute(m, spec, info, causal):
    rl = spec["row_level"].tolist(); pos = spec["pos"].tolist(); R = len(rl)
    counts = {}
    for l in rl: counts[l] = counts.get(l, 0) + 1
    first = {}
    for i, l in enumerate(rl): first.setdefault(l, i)
    li = [i - first[l] for i, l in enumerate(rl)]
    core = set(info["core_levels"]); W = m.local_pack_upper_window
    lvl_all = None
    ok = torch.zeros(R, R, dtype=torch.bool)
    comp = [0] + list(m.compression_ratios)
    ratio = [0] + [max(1, int(int(c) * (1 - float(o)))) for c, o in zip(m.compression_ratios, m.overlap_ratios)]
    nl = m._upper_nlower
    for q in range(R):
        for k in range(R):
            a = rl[q] in core or rl[k] in core
            a = a or (rl[q] == rl[k] and abs(li[q] - li[k]) <= W)
            for p, c in ((q, k), (k, q)):
                if rl[c] == rl[p] - 1:
                    lo = min(li[p] * ratio[rl[p]], nl[rl[p]] - 1)
                    hi = min(lo + comp[rl[p]] - 1, nl[rl[p]] - 1)
                    a = a or (lo <= li[c] <= hi)
            if causal:
                a = a and (pos[k], rl[k]) <= (pos[q], rl[q])
            ok[q, k] = a
    return ok

for cfg, ov, causal in ((DNA, dict(local_pack_upper_extend=3), False),
                        (TEXT, dict(local_pack_upper_extend=2), True)):
    m = build(cfg, hier_layer_compile=False, local_pack_upper_every=3, **ov).eval()
    xin = dna_in() if cfg == DNA else dna_in(1, 1024)
    with torch.no_grad(), ac():
        m(xin)
    spec = m._upper_stage_spec_cache[1]; info = m._upper_stage_info
    cnt = info["level_counts"]
    m._upper_nlower = [0] + cnt[:-1] + [0] * 8
    b = brute(m, spec, info, causal)
    got = spec["mask"].cpu() if spec["mask"] is not None else torch.ones_like(b)
    rep(f"{os.path.basename(cfg)} mask == brute", bool((got == b).all()),
        f"rows {info['rows']} core {info['core_levels']} levels {info['levels']} density {b.float().mean():.3f} mismatches {(got != b).sum().item()}")
    del m

# 5. causality (text): perturb tokens >= p, outputs < p unchanged; control with two-sided mask leaks
print("5. causality (text, extend 2, stage weights random)")
m = build(TEXT, hier_layer_compile=False, local_pack_upper_every=3, local_pack_upper_extend=2).eval()
randomize_stage(m, 0.05)
t = dna_in(1, 1024)
p = 700
t2 = t.clone(); t2[0, p:] = torch.randn(1024 - p, 768).to(dev)
def run(mm, tt):
    with torch.no_grad(), ac():
        o = mm(tt)
    return (o[0] if isinstance(o, tuple) else o).float()
o1, o2 = run(m, t), run(m, t2)
leak = (o1[:, :p] - o2[:, :p]).abs().max().item()
rep("no leak", leak == 0.0, f"max|diff| before p {leak:.3e} (after p {(o1[:, p:] - o2[:, p:]).abs().max().item():.3e})")
orig = m._upper_stage_spec
def two_sided(*a, **k):
    s = orig(*a, **k)
    if s is not None and s["mask"] is not None:
        s = dict(s); s["mask"] = s["mask"] | s["mask"].transpose(0, 1)
    return s
m._upper_stage_spec = two_sided
o1c, o2c = run(m, t), run(m, t2)
leakc = (o1c[:, :p] - o2c[:, :p]).abs().max().item()
rep("control leaks", leakc > 0.0, f"max|diff| before p {leakc:.3e}")
del m

print("FAILS:", FAILS or "none")
sys.exit(1 if FAILS else 0)
