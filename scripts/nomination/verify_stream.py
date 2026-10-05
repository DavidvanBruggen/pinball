"""Stream selector == dense selector (picks, gates, grads); model-level equivalence; causality.
usage: python verify_stream.py CONFIG [L]"""
import sys, math, logging, torch
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
CFG = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

def mk(impl, norm):
    cfg = PinballConfig.from_yaml(CFG); cfg.hier_layer_compile = False; cfg.block_size = L
    cfg.local_pack_global_select_impl = impl; cfg.local_pack_global_gumbel_norm = norm
    inp = resolve_model_inputs(cfg, block_size=L); torch.manual_seed(0)
    m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                    tie_weights=inp.tie_weights, max_seq_len=L).cuda().eval()
    return m, inp

m, inp = mk("dense", "prefix")
g = torch.Generator().manual_seed(5)
with torch.no_grad():
    for n_, p in m.named_parameters():
        if "nom_" in n_:
            p.copy_((torch.randn(p.shape, generator=g) * (0.3 if p.dim() <= 1 else 0.05)).to(p))
B = 2
x = torch.randint(0, inp.vocab_size, (B, L), generator=torch.Generator().manual_seed(3)).cuda()
mps = [t.message_passing for t in m.refinement_transformers]
cap = {}
for li, mp in enumerate(mps):
    def mk_spy(mp, li):
        o_ = mp._nominate_by_head
        def spy(spec, lvl, xn):
            cap[li] = (spec, lvl, xn.detach())
            return o_(spec, lvl, xn)
        return spy
    mp._nominate_by_head = mk_spy(mp, li)
def fwd(mm, xx):
    with torch.no_grad():
        o = mm(xx)
    return (o[0] if isinstance(o, tuple) else o).float()
y_dense = fwd(m, x)
for li in (0, 6, 11):
    mp = mps[li]; spec, lvl, xn = cap[li]
    n = int(lvl.numel()); C = int(mp.local_pack_global_chunk); W = int(spec["window"]); nch = (n + C - 1) // C
    K = max(1, min(int(mp.local_pack_global_l0_budget), n))
    # geometry equivalence
    geo = mp._nom_stream_geom(spec, lvl, C, W, K, nch, n)
    allowed = next(v for k_, v in spec.items() if isinstance(k_, tuple) and k_[:1] == ("nom_geom",))[0]
    al_s = geo["a"].view(1, -1) <= torch.arange(nch, device=lvl.device).view(-1, 1)
    rep(f"L{li} activation geometry == dense allowed mask", torch.equal(allowed, al_s),
        f"{nch} chunks x {n} rows, {int(allowed.sum())} allowed pairs")
    for mode in ("eval", "train raw", "train prefix"):
        res = {}
        for impl in ("dense", "stream"):
            mp.local_pack_global_select_impl = impl
            mp.local_pack_global_gumbel_norm = "prefix" if mode == "train prefix" else "raw"
            mp.train(mode != "eval")
            torch.manual_seed(77); torch.cuda.manual_seed(77)
            xg = xn.clone().requires_grad_(True)
            rows, gate, meta = mp._nom_original(spec, lvl, xg) if hasattr(mp, "_nom_original") else type(mp)._nominate_by_head(mp, spec, lvl, xg)
            ok = meta["ok"]
            (gate.masked_fill(~ok.view(1, -1), 0.0).float().sin().sum()).backward()
            res[impl] = (rows, gate.detach(), ok, xg.grad.detach())
        mp.train(False); mp.local_pack_global_select_impl = "dense"
        (rd, gd, okd, xgd), (rs, gs, oks, xgs) = res["dense"], res["stream"]
        nchK = rd.view(B, nch, K)
        same_rows = torch.equal(rd, rs)
        # set equality per chunk (ties may reorder)
        live = okd.view(nch, K)
        set_eq = all(set(rd.view(B, nch, K)[b, c][live[c]].tolist()) == set(rs.view(B, nch, K)[b, c][live[c]].tolist())
                     for b in range(B) for c in range(nch))
        dg = (gd - gs).masked_fill(~okd.view(1, -1), 0).abs().max().item()
        dx = (xgd - xgs).abs().max().item() / max(1e-12, xgd.abs().max().item())
        rep(f"L{li} {mode}: picks identical (sets per chunk), gates, grads", torch.equal(okd, oks) and set_eq and dg < 1e-4 and dx < 1e-3,
            f"rows bit-equal {same_rows}; max|gate diff| {dg:.1e}; rel grad diff {dx:.1e}")
# model level: stream model vs dense model (same weights), eval
for mp in mps:
    mp.local_pack_global_select_impl = "stream"
y_stream = fwd(m, x)
d = (y_stream - y_dense).abs().max().item()
rep("model eval output: stream == dense", d < 1e-3, f"max|diff| {d:.2e} (logit scale {y_dense.abs().max().item():.1f})")
rep("flex ran", m.flex_union_status()["failed_modules"] == 0, "")
# causality with stream (eval + train/prefix noise)
for p0 in (L - 100, L // 2):
    x2 = x.clone(); x2[:, p0:] = torch.randint(0, inp.vocab_size, (B, L - p0), generator=torch.Generator().manual_seed(p0)).cuda()
    lk = (y_stream[:, :p0] - fwd(m, x2)[:, :p0]).abs().max().item()
    rep(f"stream causal p={p0}", lk == 0.0, f"{lk:.3e}")
m.train()
def fwd_t(xx):
    torch.manual_seed(123); torch.cuda.manual_seed(123)
    return fwd(m, xx)
yt = fwd_t(x); x2 = x.clone(); x2[:, L // 2:] = torch.randint(0, inp.vocab_size, (B, L - L // 2), generator=torch.Generator().manual_seed(1)).cuda()
lk = (yt[:, :L // 2] - fwd_t(x2)[:, :L // 2]).abs().max().item()
rep("stream causal in train mode (prefix noise)", lk == 0.0, f"{lk:.3e}")
print("FAILS:", FAILS or "none")
