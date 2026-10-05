"""LapPE plumbing check on the text glob400 q config (token mode, Blackwell, fp32 eager)."""
import sys, math, time, logging, torch
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs

CFG = "/home/david/Projects/pinball/configs/pinball_wikitext_pack_glob400_l0coarse_l0sel_q_d384_4k.yaml"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

def mk(k, L=4096):
    cfg = PinballConfig.from_yaml(CFG)
    cfg.hier_layer_compile = False
    cfg.lap_pe_k = k
    inp = resolve_model_inputs(cfg, block_size=L)
    torch.manual_seed(0)
    m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                    tie_weights=inp.tie_weights, max_seq_len=inp.block_size).cuda().eval()
    return m, inp

m0, inp = mk(0)
m1, _ = mk(K)
rep("knob reaches model", m1.lap_pe_k == K and m1.lap_pe_proj is not None and m0.lap_pe_proj is None,
    f"k={m1.lap_pe_k}")
rep("proj zero after model init", float(m1.lap_pe_proj.weight.abs().max()) == 0.0, "")
r = m1.load_state_dict(m0.state_dict(), strict=False)
rep("only lap_pe_proj missing", sorted(r.missing_keys) == ["lap_pe_proj.bias", "lap_pe_proj.weight"] and not r.unexpected_keys, str(r.missing_keys))

L, B = 4096, 2
g = torch.Generator().manual_seed(3)
x = torch.randint(0, inp.vocab_size, (B, L), generator=g).cuda()
def fwd(m, xx):
    with torch.no_grad():
        o = m(xx)
    return (o[0] if isinstance(o, tuple) else o).float()

# capture the PE that reaches the add site
seen = {}
orig = m1._lap_pe_raw_on_device
def spy(pe, device, dtype):
    seen["pe"] = pe
    return orig(pe, device, dtype)
m1._lap_pe_raw_on_device = spy
y0 = fwd(m0, x); y1 = fwd(m1, x)
rep("PE reaches the add site", "pe" in seen, str(tuple(seen["pe"].shape)) if "pe" in seen else "never called")
rep("zero proj == off model (bit-identical)", (y0 - y1).abs().max().item() == 0.0, f"max|diff| {(y0 - y1).abs().max().item():.3e}")
rep("flex ran", m1.flex_union_status()["failed_modules"] == 0, "")

pe = seen["pe"]
g_ = m1._cached_unified_graph
nl, ar = g_.node_level.cpu(), g_.node_ar_time.cpu(); pe = pe.cpu()
n = nl.numel()
rep("PE shape [N, k]", tuple(pe.shape) == (n, K), f"N={n}")
rms = pe.pow(2).mean(0).sqrt()
rep("unit RMS columns", bool(((rms - 1).abs() < 1e-4).all()), f"{rms.min():.4f}-{rms.max():.4f}")
# eigvec 1 on L0 ~ cos(pi (t+.5)/L) up to sign
l0 = torch.nonzero(nl == 0).view(-1)
t0 = ar[l0].double()
ref = torch.cos(math.pi * (t0 + 0.5) / L)
v1 = pe[l0, 0].double()
v1 = pe[l0, 0].double()
c = torch.corrcoef(torch.stack([v1, ref]))[0, 1].abs().item()
rep("1st eigvec on L0 ~ cos(pi t/L)", c > 0.98, f"|corr| {c:.4f}; eigvals {m1._hier_lap_pe_eigvals[:4].round(6)}")
# coarse node ~ mean of its TRUE L0 coverage [j * cum_stride, ar] (smooth columns only, first 4)
tl0 = ar[l0]
for lv in (1, 2, 3):
    idx = torch.nonzero(nl == lv).view(-1)
    cs = m1._cumulative_window(lv)[1]
    errs = []
    for jj, i in list(enumerate(idx.tolist()))[:: max(1, idx.numel() // 64)]:
        ch = l0[(tl0 >= jj * cs) & (tl0 <= int(ar[i]))]
        errs.append((pe[i, :4] - pe[ch, :4].mean(0)).abs().max().item())
    rep(f"L{lv} value ~ its region's mean", max(errs) < 0.15, f"max err (4 low cols) {max(errs):.3f}")

# nonzero proj: output changes, causal, grads reach proj
with torch.no_grad():
    m1.lap_pe_proj.weight.normal_(0, 0.5)
y2 = fwd(m1, x)
rep("nonzero proj changes output", (y2 - y1).abs().max().item() > 1e-3, f"{(y2 - y1).abs().max().item():.3e}")
p = 2000
x2 = x.clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (B, L - p), generator=g).cuda()
y3 = fwd(m1, x2)
d = (y3[:, :p] - y2[:, :p]).abs().max().item()
rep("causal (perturb >= p, outputs < p)", d == 0.0, f"max|diff| before p {d:.3e}")
m1.train()
o = m1(x[:1]); o = o[0] if isinstance(o, tuple) else o
o.float().logsumexp(-1).mean().backward()
gw = m1.lap_pe_proj.weight.grad
rep("grad reaches lap_pe_proj", gw is not None and gw.abs().max().item() > 0, f"{gw.abs().max().item():.3e}" if gw is not None else "None")

# optimizer routing (same marker list as cli._build_optimizer)
from pinball import cli
src = open(cli.__file__).read()
rep("lap_pe_proj routed to AdamW", '"lap_pe_proj"' in src.split("head_name_markers")[1].split(")")[0], "")

# geometry rule == the built graph: recursive window end vs real node_ar_time, ALL nodes
ends = [torch.arange((nl == 0).sum().item())]
for lv in range(1, 4):
    comp = int(m1.compression_ratios[lv - 1]); st = max(1, int(comp * (1 - m1.overlap_ratios[lv - 1])))
    nlo = ends[-1].numel(); nhi = int((nl == lv).sum())
    last = torch.clamp(torch.arange(nhi) * st + comp, max=nlo) - 1
    ends.append(ends[-1][last])
ok_all = all(bool((ar[torch.nonzero(nl == lv).view(-1)] == ends[lv]).all()) for lv in range(4))
rep("child-index rule == built hierarchy (every node's window end)", ok_all, "")
pred = tuple(m1._predict_level_sizes(L)); real = tuple(int((nl == l).sum()) for l in range(4))
rep("_predict_level_sizes == built sizes", pred[:4] == real, f"{pred} vs {real}")

def pe_of(m, xx, **kw):
    box = {}
    o_ = m._lap_pe_raw_on_device
    def sp_(pe_, device, dtype):
        box["pe"] = pe_.detach().cpu(); return o_(pe_, device, dtype)
    m._lap_pe_raw_on_device = sp_
    with torch.no_grad():
        out = m(xx, **kw)
    m._lap_pe_raw_on_device = o_
    sizes = tuple(m._predict_level_sizes(int(xx.size(1))))
    nl_ = torch.cat([torch.full((s_,), l, dtype=torch.long) for l, s_ in enumerate(sizes)])
    return (out[0] if isinstance(out, tuple) else out).float(), box["pe"], nl_, sizes
m1.eval()
look = int(m1._gen_frontier_lookahead())
pad_id = int(getattr(m1, "pad_token_id", None) or getattr(m1, "mask_token_id", 0) or 0)
yf, pe_f, nl_f, _ = pe_of(m1, x)
for Lp in (1500, 777):
    xp = torch.cat([x[:, :Lp], torch.full((B, look), pad_id, dtype=x.dtype, device=x.device)], 1)
    yp, pe_p, nl_p, sz_p = pe_of(m1, xp, logits_last_index=Lp - 1)
    d_pe, d_ctl = 0.0, 0.0
    ctl = m1._hier_lap_pe_geometry(sz_p)          # what a per-length basis would give
    for lv in range(4):
        ip = torch.nonzero(nl_p == lv).view(-1); iF = torch.nonzero(nl_f == lv).view(-1)[: ip.numel()]
        d_pe = max(d_pe, (pe_p[ip] - pe_f[iF]).abs().max().item())
        d_ctl = max(d_ctl, (ctl[ip] - pe_f[iF]).abs().max().item())
    rep(f"prefix {Lp}+{look}: PE rows == training rows", d_pe == 0.0 and d_ctl > 0.1,
        f"max|diff| {d_pe:.3e} (a per-length basis would be off by {d_ctl:.2f})")
# end to end, the configured gen path (compile -> fixed shape: pad to max_seq_len)
Lp = 1500
xfix = x.clone(); xfix[:, Lp:] = pad_id
with torch.no_grad():
    yfix = m1(xfix, logits_last_index=Lp - 1)
yfix = (yfix[0] if isinstance(yfix, tuple) else yfix).float()
d = (yfix[:, -1] - yf[:, Lp - 1]).abs().max().item()
rep("fixed-shape gen frontier logits == training (LapPE on)", d < 1e-4, f"max|diff| {d:.3e}")
# end to end, uncompiled gen (lookahead pad only, the graph size changes every step)
for Lp in (1500, 777):
    xp = torch.cat([x[:, :Lp], torch.full((B, look), pad_id, dtype=x.dtype, device=x.device)], 1)
    yp, _, _, _ = pe_of(m1, xp, logits_last_index=Lp - 1)
    d = (yp[:, -1] - yf[:, Lp - 1]).abs().max().item()
    rep(f"variable-length gen frontier logits == training, prefix {Lp} (LapPE on)", d < 1e-4, f"max|diff| {d:.3e}")
# control at a length never built before: a per-length basis breaks it
orig_cpu = m1._hier_lap_pe_cpu
def per_len(node_level):
    lv_ = node_level.cpu(); sz = tuple(int((lv_ == l).sum()) for l in range(int(lv_.max()) + 1))
    return m1._hier_lap_pe_geometry(sz)
m1._hier_lap_pe_cpu = per_len
Lc = 1111
xp = torch.cat([x[:, :Lc], torch.full((B, look), pad_id, dtype=x.dtype, device=x.device)], 1)
yc, _, _, _ = pe_of(m1, xp, logits_last_index=Lc - 1)
m1._hier_lap_pe_cpu = orig_cpu
d = (yc[:, -1] - yf[:, Lc - 1]).abs().max().item()
rep("control: per-length basis breaks gen (prefix 1111)", d > 1e-2, f"max|diff| {d:.3e}")
rep("kv-cache guard names lap_pe", any("lap_pe" in r_ for r_ in m1._gen_kv_guard_reasons(1)), "")
gen = m1.generate(x[:1, :300], max_length=4, do_sample=False)
rep("generate() runs", gen.shape[1] == 304, str(tuple(gen.shape)))

# eigensolve cost per length (CPU, once per process at max_seq_len)
for LL in (4096, 16384, 65536):
    sizes = tuple(m1._predict_level_sizes(LL))
    t1 = time.time(); m1._hier_lap_pe_cache = {}
    m1._hier_lap_pe_geometry(sizes)
    print(f"   eigsh L={LL} N={sum(sizes)}: {time.time() - t1:.2f}s")
print("FAILS:", FAILS or "none")
