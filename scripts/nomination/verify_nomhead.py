"""Independent verification of the hierarchy nomination head (fp32, Blackwell)."""
import os, sys, math
sys.argv = [sys.argv[0], "cuda"]
exec(open("/home/david/Projects/pinball/scripts/check_upper_stage.py").read().split("# 1. identity at init")[0])
CFG = "/home/david/Projects/pinball/configs/pinball_wikitext_pack_glob400_l0coarse_l0sel_d384_4k.yaml"
L, B = 4096, 2
OV = dict(hier_layer_compile=False, local_pack_global_nominator="head",
          local_pack_global_candidates=os.environ.get("CAND", "all"), local_pack_global_gumbel=1.0, local_pack_global_nom_gate=os.environ.get("GATE", "raw"), local_pack_global_boost=os.environ.get("BOOST", "logit"), local_pack_global_gumbel_norm=os.environ.get("GNORM", "raw"), local_pack_global_region_cap=int(os.environ.get("CAP", 0)), local_pack_global_region_level=2)
m = build(CFG, seed=0, **OV).eval()
# nonzero heads so weights are informative
g = torch.Generator().manual_seed(5)
with torch.no_grad():
    for n_, p in m.named_parameters():
        if "nom_" in n_:
            p.copy_((torch.randn(p.shape, generator=g) * (0.3 if p.dim() == 1 else (0.5 if "boost_u" in n_ else 0.05))).to(p))
x = dna_in(B, L)
caps = {}
for li, t in enumerate(m.refinement_transformers):
    mp = t.message_passing
    def mk(mp, li):
        o1, o2 = mp._flex_union_attn, mp._nominate_by_head
        def spy_attn(qp, kp, vp, spec, causal=True, gsel=None):
            out = o1(qp, kp, vp, spec, causal, gsel=gsel)
            caps.setdefault(li, {}).update(qp=qp.detach(), kp=kp.detach(), vp=vp.detach(), out=out.detach(), gsel=gsel, spec=spec, mp=mp)
            return out
        def spy_nom(spec, lvl_packed, x_nodes):
            r = o2(spec, lvl_packed, x_nodes)
            caps.setdefault(li, {}).update(x_nodes=x_nodes.detach(), lvl=lvl_packed)
            return r
        return spy_attn, spy_nom
    mp._flex_union_attn, mp._nominate_by_head = mk(mp, li)
def fwd(xx):
    with torch.no_grad():
        o = m(xx)
    return (o[0] if isinstance(o, tuple) else o).float()
y = fwd(x)
rep("flex ran", m.flex_union_status()["failed_modules"] == 0, "")
c0 = caps[0]; spec = c0["spec"]
n = int(spec["num_nodes"]); perm = spec["perm"]; lvlp = c0["lvl"]
node_lvl = torch.empty_like(lvlp); node_lvl[perm] = lvlp
# ar_time per node: packed pos is token-scale close time
tnode = torch.empty_like(spec["pos"]); tnode[perm] = spec["pos"]
inv = torch.empty_like(perm); inv[perm] = torch.arange(n, device=perm.device)
# 1. parent map brute force (earliest-closing container)
def span(l): return 1 if l == 0 else m._cumulative_window(l)[0]
par_b = torch.full((n,), -1, dtype=torch.long)
tn, ln = tnode.cpu(), node_lvl.cpu()
for l in range(int(ln.max())):
    P = torch.nonzero(ln == l + 1).view(-1); C = torch.nonzero(ln == l).view(-1)
    ps, pe = tn[P] - span(l + 1) + 1, tn[P]
    for c in C.tolist():
        cs, ce = tn[c] - span(l) + 1, tn[c]
        okp = (ps <= cs) & (pe >= ce)
        if okp.any():
            cand = P[okp]; par_b[c] = cand[torch.argmin(tn[cand])]   # earliest end; ties -> first
par_m = spec["nom_parent_node"].cpu()
tie_ok = (par_m == par_b) | ((par_m >= 0) & (par_b >= 0) & (tn[par_m.clamp(min=0)] == tn[par_b.clamp(min=0)]))
rep("parent map == brute force", bool(tie_ok.all()), f"mismatch {(~tie_ok).sum().item()}; with parent per level {[int(((ln == l) & (par_m >= 0)).sum()) for l in range(int(ln.max()) + 1)]} of {[int((ln == l).sum()) for l in range(int(ln.max()) + 1)]}")
print("   ar_time sample L0/L1/L2/L3:", [tn[torch.nonzero(ln == l).view(-1)[:3]].tolist() for l in range(4)])
CAPV = int(os.environ.get("CAP", 0))
reg_node = torch.arange(n)
for c_ in range(n):
    a = c_
    while int(ln[a]) < 2 and int(par_b[a]) >= 0:
        a = int(par_b[a])
    reg_node[c_] = a if int(ln[a]) >= 2 else n + c_
reg_packed = reg_node[perm.cpu()].to(perm.device)
for li in (0, 5, 11):
    c = caps[li]; mp = c["mp"]; gsel = c["gsel"]; meta = gsel[2]
    xn = c["x_nodes"]
    # 2. weights recomputed independently
    d = mp.local_pack_global_nom_dim
    e = torch.zeros(B, n, device=xn.device)
    pm = spec["nom_parent_node"]
    for cn in torch.nonzero(pm >= 0).view(-1).tolist()[:0]: pass
    for l in range(len(mp.nom_child_k)):
        cs = torch.nonzero((node_lvl == l) & (pm >= 0)).view(-1)
        if cs.numel() == 0: continue
        ps = pm[cs]
        e[:, cs] = (mp.nom_parent_q[l](xn[:, ps]) * mp.nom_child_k[l](xn[:, cs])).sum(-1) / math.sqrt(d)
    w_node = e + torch.where(pm >= 0, e[:, pm.clamp(min=0)], torch.zeros_like(e)) + mp.nom_level_offset[node_lvl].view(1, -1)
    w = w_node[:, perm]
    # 3. allowed + picks brute force
    C_, W_ = meta["chunk"], int(spec["window"]); K = meta["G"]; nch = meta["nch"]
    gp = torch.where(pm >= 0, pm[pm.clamp(min=0)], pm)
    avail_node = torch.where(pm >= 0, torch.maximum(inv[pm.clamp(min=0)], torch.where(gp >= 0, inv[gp.clamp(min=0)], torch.full_like(gp, -1))), torch.full_like(pm, n))
    avail = avail_node[perm]
    static = torch.zeros(n, dtype=torch.bool, device=xn.device); static[meta["static_rows"]] = True
    cand = (avail < n) & ~static & ((lvlp == 0) if mp.local_pack_global_candidates == "l0" else torch.ones_like(static))
    rows = gsel[0].view(B, nch, K); ok = meta["ok"].view(nch, K)
    gexp = w.gather(1, gsel[0]).view(B, nch, K).clone()
    bad = 0; badc = 0; chk = 0
    for b in range(B):
        for ch in range(nch):
            lim = max(0, ch * C_ - W_)
            al = torch.nonzero(cand & (torch.arange(n, device=xn.device) < lim) & (avail < lim)).view(-1)
            if CAPV > 0:
                o_ = al[torch.argsort(w[b, al], descending=True)]
                seen = {}; keep_ = []
                for r_ in o_.tolist():
                    g_ = int(reg_packed[r_]); seen[g_] = seen.get(g_, 0) + 1
                    if seen[g_] <= CAPV: keep_.append(r_)
                cnt_ = len(keep_)
                exp = torch.tensor(keep_[:K], dtype=torch.long)
            else:
                cnt_ = al.numel()
                exp = al[torch.topk(w[b, al], min(K, al.numel())).indices] if al.numel() else al
            if int(ok[ch].sum()) != min(K, cnt_): badc += 1
            if al.numel() == 0: continue
            if mp.local_pack_global_nom_gate == "zscore":
                mu_, sd_ = w[b, al].mean(), (w[b, al].var(unbiased=False) + 1e-6).sqrt()
                gexp[b, ch] = (w[b, rows[b, ch]] - mu_) / sd_ + mp.nom_gate_bias
            if set(exp.tolist()) != set(rows[b, ch][ok[ch]].tolist()): bad += 1
            chk += 1
    rep(f"L{li} picks == brute top-K", bad == 0 and badc == 0, f"{chk} checked, set mismatch {bad}, count mismatch {badc}")
    gexp = gexp.view(B, -1)
    if CAPV > 0:
        viol = 0
        for b in range(B):
            for ch in range(nch):
                rr = rows[b, ch][ok[ch]]
                if rr.numel():
                    viol += int((torch.bincount(reg_packed[rr]) > CAPV).sum())
        rep(f"L{li} region cap respected (<= {CAPV}/region)", viol == 0, f"violations {viol}")
    dw = (gsel[1][:, meta["ok"]] - gexp[:, meta["ok"]]).abs().max().item()
    rep(f"L{li} gate values == recompute ({mp.local_pack_global_nom_gate})", dw < 1e-3, f"max|diff| {dw:.2e}")
    st = mp._nom_stats
    print(f"   L{li}: coarse_frac {st['coarse_frac']:.3f} gate {st['gate_mean']:.3f}±{st['gate_std']:.3f} illegal {int(st['illegal_picks'])}")
    # 4. flex output == dense reference
    qp, kp, vp = c["qp"], c["kp"], c["vp"]; D = qp.size(-1)
    srows = meta["static_rows"]
    gg = torch.Generator().manual_seed(li)
    qs = torch.cat([torch.randint(0, n, (30,), generator=gg), torch.tensor([n - 1, 2000, 700])]).tolist()
    err = 0.0
    for b in range(B):
        for qi in qs:
            ch = qi // C_
            sr = srows[srows <= qi]; sl = rows[b, ch][ok[ch]]
            band = torch.arange(max(0, qi - W_), qi + 1, device=xn.device); band = band[~static[band]]
            kr = torch.cat([sr, sl, band])
            gsl = gexp.view(B, nch, K)[b, ch][ok[ch]]
            _key = mp.local_pack_global_boost == "key"
            bias = torch.cat([torch.zeros(len(sr), device=xn.device), gsl if (mp.local_pack_global_logit and not _key) else torch.zeros(len(sl), device=xn.device), torch.zeros(len(band), device=xn.device)])
            kk = kp[b, kr].clone()
            if _key and len(sl) > 0:
                zsl = gsl - mp.nom_gate_bias
                ur = mp.rotary_pos_enc.apply_rotary_pos_emb(mp.nom_boost_u.float().view(1, *mp.nom_boost_u.shape).expand(len(sl), -1, -1).contiguous(), spec["pos"][sl])
                kk[len(sr):len(sr) + len(sl)] += zsl.view(-1, 1, 1) * ur
            gate = torch.cat([torch.ones(len(sr), device=xn.device), torch.sigmoid(gsl), torch.ones(len(band), device=xn.device)])
            lg = torch.einsum("hd,khd->hk", qp[b, qi], kk) / math.sqrt(D) + bias.view(1, -1)
            o = torch.einsum("hk,khd->hd", lg.softmax(-1), vp[b, kr] * gate.view(-1, 1, 1))
            err = max(err, (o - c["out"][b, qi]).abs().max().item())
    rep(f"L{li} flex out == dense reference", err < 2e-3, f"max|diff| {err:.2e}")
if os.environ.get("GATE") == "zscore":
    _saved = [t_.message_passing.nom_level_offset.detach().clone() for t_ in m.refinement_transformers]
    with torch.no_grad():
        for t_ in m.refinement_transformers: t_.message_passing.nom_level_offset.add_(5.0)
    ysh = fwd(x)
    with torch.no_grad():
        for t_, sv in zip(m.refinement_transformers, _saved): t_.message_passing.nom_level_offset.copy_(sv)
    rep("offsets restored exactly", (fwd(x) - y).abs().max().item() == 0.0, "")
    dsh = (ysh - y).abs().max().item()
    rep("uniform weight shift leaves output unchanged (zscore)", dsh < 1e-3, f"max|diff| {dsh:.2e}")
# 5. causality (eval) + cross-sample
for p0 in (4095, 3000, 1500, 700):
    x2 = x.clone(); x2[:, p0:] = torch.randn(B, L - p0, 768).to(dev)
    lk = (y[:, :p0] - fwd(x2)[:, :p0]).abs().max().item(); rep(f"causal p={p0}", lk == 0.0, f"{lk:.3e}")
x3 = x.clone(); x3[1] = torch.randn(L, 768).to(dev)
rep("no cross-sample leak", (y[0] - fwd(x3)[0]).abs().max().item() == 0.0, "")
# 6. causality in TRAIN mode with Gumbel on (same RNG for both forwards; dropout too)
m.train()
def fwd_t(xx):
    torch.manual_seed(123); torch.cuda.manual_seed(123)
    with torch.no_grad():
        o = m(xx)
    return (o[0] if isinstance(o, tuple) else o).float()
yt = fwd_t(x)
x2 = x.clone(); x2[:, 1500:] = torch.randn(B, L - 1500, 768).to(dev)
lk = (yt[:, :1500] - fwd_t(x2)[:, :1500]).abs().max().item()
rep("causal in train mode (gumbel+dropout, same seed)", lk == 0.0, f"{lk:.3e}")
# 7. grads to heads
for p in m.parameters(): p.grad = None
loss = m(x)
loss = (loss[0] if isinstance(loss, tuple) else loss).float().pow(2).mean(); loss.backward()
zg = [n_ for n_, p in m.named_parameters() if "nom_" in n_ and (p.grad is None or p.grad.abs().max() == 0)]
nh = sum(1 for n_, p in m.named_parameters() if "nom_" in n_)
print("   boost_u grads:", [float(t_.message_passing.nom_boost_u.grad.abs().max()) for t_ in m.refinement_transformers][:3] if hasattr(m.refinement_transformers[0].message_passing, "nom_boost_u") else "n/a")
rep("grads reach every nomination param", not zg, f"{nh} params; zero-grad {zg[:4]}")
print("FAILS:", FAILS or "none")
