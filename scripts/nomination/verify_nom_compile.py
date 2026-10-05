"""local_pack_global_nom_compile probe (stream selector).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_nom_compile.py CONFIG [L]
  1. refactored EAGER stream path == the pre-refactor methods (verbatim copies below), bit for
     bit: logits and every grad, train mode with Gumbel noise, fixed seed;
  2. COMPILED selector vs eager, train mode, same seed: logits / grads close, picks agree;
  3. compiled path causal in train mode; flex ran."""
import sys, types, random, logging, torch
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs

CFG = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

# ---- pre-refactor reference (verbatim) --------------------------------------------------
def old_prefix_stats(v, geo):
    nch = int(geo["nch"]); B = int(v.size(0))
    vd = v.double()
    ix = geo["a"].view(1, -1).expand(B, -1)
    s1 = vd.new_zeros(B, nch + 1).scatter_add(1, ix, vd)[:, :nch].cumsum(1)
    s2 = vd.new_zeros(B, nch + 1).scatter_add(1, ix, vd * vd)[:, :nch].cumsum(1)
    cnt = geo["cnt"].to(vd.dtype).clamp(min=1.0).view(1, -1)
    mu = s1 / cnt
    var = (s2 / cnt - mu * mu).clamp(min=0.0)
    return mu.to(v.dtype), var.to(v.dtype)

def old_nomination_weights(self, spec, lvl_packed, x_nodes):
    B, N = int(x_nodes.size(0)), int(x_nodes.size(1))
    par = spec["nom_parent_node"]
    cache = spec.get("nom_level_nodes", None)
    if cache is None:
        node_lvl = torch.empty_like(lvl_packed); node_lvl[spec["perm"]] = lvl_packed
        cache = []
        for l in range(len(self.nom_child_k)):
            c = torch.nonzero((node_lvl == l) & (par >= 0), as_tuple=False).view(-1)
            P = torch.nonzero(node_lvl == l + 1, as_tuple=False).view(-1)
            slot = torch.full((N,), -1, dtype=torch.long, device=par.device)
            slot[P] = torch.arange(int(P.numel()), device=par.device)
            cache.append((l, c, P, slot.index_select(0, par.index_select(0, c))))
        spec["nom_level_nodes"] = cache
        spec["nom_node_level"] = node_lvl
    scale = float(self.local_pack_global_nom_dim) ** -0.5
    e = x_nodes.new_zeros(B, N, dtype=torch.float32)
    assert self.local_pack_global_select_impl == "stream"
    segs = [t_ for l, c, P, pslot in cache if int(c.numel()) > 0 for t_ in (P, c)]
    if segs:
        xg = x_nodes.index_select(1, torch.cat(segs))
        parts = torch.split(xg, [int(t_.numel()) for t_ in segs], dim=1)
        j = 0
        for l, c, P, pslot in cache:
            if int(c.numel()) == 0:
                continue
            qp_ = self.nom_parent_q[l](parts[j]).index_select(1, pslot)
            kc_ = self.nom_child_k[l](parts[j + 1])
            e = e.index_copy(1, c, ((qp_ * kc_).sum(-1).float() * scale))
            j += 2
    up = torch.where((par >= 0).view(1, -1), e.index_select(1, par.clamp(min=0)),
                     torch.zeros_like(e))
    w = e + up + self.nom_level_offset.float().index_select(0, spec["nom_node_level"]).view(1, -1)
    return w.index_select(1, spec["perm"])

def old_nominate_by_head_stream(self, spec, lvl_packed, w, C, W, nch, K):
    B, n = int(w.size(0)), int(w.size(1))
    dev = w.device
    geo = self._nom_stream_geom(spec, lvl_packed, C, W, K, nch, n)
    rank = w.detach()
    neg = torch.finfo(rank.dtype).min
    tau = float(getattr(self, "local_pack_global_gumbel", 0.0))
    if self.training and tau > 0.0:
        u = torch.rand_like(rank).clamp_(1e-6, 1.0 - 1e-6)
        gn = -torch.log(-torch.log(u))
        if self.local_pack_global_gumbel_norm == "prefix":
            _, var = old_prefix_stats(rank, geo)
            sd = (var + 1e-6).sqrt()
            ia = geo["a"].clamp(max=nch - 1).view(1, -1).expand(int(rank.size(0)), -1)
            key = rank + tau * sd.gather(1, ia) * gn
        else:
            key = rank + tau * gn
    else:
        key = rank
    order, a_sorted = geo["order"], geo["a_sorted"]
    S_rows = torch.zeros(B, K, dtype=torch.long, device=dev)
    S_key = torch.full((B, K), neg, dtype=key.dtype, device=dev)
    tops, vals = [], []
    for c0, c1, i0, i1 in geo["groups"]:
        Gc = c1 - c0
        nr = order[i0:i1]
        pk = torch.cat([S_key, key.index_select(1, nr)], 1)
        pr = torch.cat([S_rows, nr.view(1, -1).expand(B, -1)], 1)
        vn = a_sorted[i0:i1].view(1, -1) <= torch.arange(c0, c1, device=dev).view(-1, 1)
        valid = torch.cat([torch.ones(Gc, K, dtype=torch.bool, device=dev), vn], 1)
        sc = pk.unsqueeze(1).masked_fill(~valid.unsqueeze(0), neg)
        tv, ti = torch.topk(sc, K, dim=-1)
        tr = pr.unsqueeze(1).expand(-1, Gc, -1).gather(2, ti)
        tops.append(tr); vals.append(tv)
        S_rows, S_key = tr[:, -1], tv[:, -1]
    top = torch.cat(tops, 1)
    legal = torch.cat(vals, 1) > neg
    top = torch.where(legal, top, torch.zeros_like(top))
    rows = top.reshape(B, nch * K).contiguous()
    ok = geo["ok"]
    okf = ok.reshape(-1).contiguous()
    g = w.gather(1, rows)
    if self.local_pack_global_nom_gate == "zscore":
        mu, var = old_prefix_stats(w, geo)
        sd = (var + 1e-6).sqrt()
        g = ((g.view(B, nch, K) - mu.unsqueeze(-1)) / sd.unsqueeze(-1)).reshape(B, nch * K)
        g = g + self.nom_gate_bias.to(g.dtype)
    g = torch.where(okf.view(1, -1), g, torch.full_like(g, torch.finfo(g.dtype).min))
    st = spec["global_block"]["rows"]
    self._nom_stats = {}
    return rows, g, {"chunk": C, "G": K, "nch": int(nch), "ok": okf,
                     "n_static": int(st.numel()), "static_rows": st, "per_batch": True}

# ---- model ------------------------------------------------------------------------------
cfg = PinballConfig.from_yaml(CFG); cfg.block_size = L
cfg.local_pack_global_select_impl = "stream"
inp = resolve_model_inputs(cfg, block_size=L)
torch.manual_seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda()
g0 = torch.Generator().manual_seed(5)
with torch.no_grad():   # non-trivial nominator weights (trained-like spread)
    for n_, p in m.named_parameters():
        if "nom_" in n_:
            p.copy_((torch.randn(p.shape, generator=g0) * (0.3 if p.dim() <= 1 else 0.05)).to(p))
mps = [t.message_passing for t in m.refinement_transformers]
print(f"{CFG.split('/')[-1]} @ {L}: {len(mps)} layers, hier_layer_compile={cfg.hier_layer_compile}, "
      f"gumbel {mps[0].local_pack_global_gumbel} norm {mps[0].local_pack_global_gumbel_norm}")
B = 2
x = torch.randint(0, inp.vocab_size, (B, L), generator=torch.Generator().manual_seed(3)).cuda()
picks = {}
for li, mp in enumerate(mps):
    orig = mp._nom_stream_finish
    def spy(spec, rows, g, stats, geo, C, nch, K, _o=orig, _li=li):
        picks[_li] = (rows.detach().clone(), geo["ok"].reshape(-1).clone())
        return _o(spec, rows, g, stats, geo, C, nch, K)
    mp._nom_stream_finish = spy

def run(xx, seed=123):
    m.zero_grad(set_to_none=True)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    with torch.autocast("cuda", torch.bfloat16):
        o = m(xx); o = (o[0] if isinstance(o, tuple) else o)
        loss = torch.nn.functional.cross_entropy(o[:, :-1].float().reshape(-1, o.size(-1)), xx[:, 1:].reshape(-1))
    loss.backward()
    grads = {n_: p.grad.detach().clone() for n_, p in m.named_parameters() if p.grad is not None}
    return o.detach().float(), grads

def gdiff(ga, gb):
    """global relative grad difference ||ga - gb|| / ||gb|| over all tensors"""
    num = sum(((ga[k] - gb[k]).float() ** 2).sum().item() for k in gb)
    den = sum((gb[k].float() ** 2).sum().item() for k in gb)
    return (num / max(den, 1e-30)) ** 0.5

def set_overlap(ra, rb, ok, K):
    """mean over (batch, chunk) of |picks_a & picks_b| / live slots"""
    B = int(ra.size(0)); nch = int(ok.numel()) // K
    ra, rb, okv = ra.view(B, nch, K).tolist(), rb.view(B, nch, K).tolist(), ok.view(nch, K).tolist()
    tot = hit = 0
    for b in range(B):
        for c in range(nch):
            live = [i for i in range(K) if okv[c][i]]
            if not live: continue
            sa = {ra[b][c][i] for i in live}; sb = {rb[b][c][i] for i in live}
            hit += len(sa & sb); tot += len(live)
    return hit / max(1, tot)

m.train()
# 1. refactored eager == pre-refactor
for mp in mps:
    mp.local_pack_global_nom_compile = False
run(x)                                  # warm-up (compile, lazy caches)
y_new, g_new = run(x)
y_rep, g_rep = run(x)
dy = (y_new - y_rep).abs().max().item(); dg = max((g_new[k] - g_rep[k]).abs().max().item() for k in g_new)
_pr = {k: v[0].clone() for k, v in picks.items()}
g_noise = gdiff(g_rep, g_new)
rep("CONTROL: same path twice, same seed (fwd exact; bwd has atomics)", dy == 0.0,
    f"max|logit diff| {dy:.3e}; grad noise floor ||dg||/||g|| = {g_noise:.2e}")
for mp in mps:
    mp._nomination_weights = types.MethodType(old_nomination_weights, mp)
    mp._nominate_by_head_stream = types.MethodType(old_nominate_by_head_stream, mp)
y_old, g_old = run(x)
for mp in mps:
    del mp._nomination_weights, mp._nominate_by_head_stream
dy = (y_new - y_old).abs().max().item()
dg = max((g_new[k] - g_old[k]).abs().max().item() for k in g_old)
_gr = gdiff(g_old, g_new)
rep("eager refactor == pre-refactor (train, Gumbel)", dy == 0.0 and g_new.keys() == g_old.keys() and _gr < 3 * g_noise + 1e-6,
    f"max|logit diff| {dy:.3e}; grads ||dg||/||g|| {_gr:.2e} (noise floor {g_noise:.2e}), {len(g_old)} grads")
p_eager = dict(picks)

# 2. compiled vs eager
for mp in mps:
    mp.local_pack_global_nom_compile = True
run(x)                                  # warm-up / compile
y_c, g_c = run(x)
dy = (y_c - y_new).abs().max().item(); sy = y_new.abs().max().item()
_gc = gdiff(g_c, g_new)
nom = [k for k in g_new if "nom_" in k]
_gn = gdiff({k: g_c[k] for k in nom}, {k: g_new[k] for k in nom})
_gn0 = gdiff({k: g_rep[k] for k in nom}, {k: g_new[k] for k in nom})
K_ = int(mps[0].local_pack_global_l0_budget)
ov = [set_overlap(p_eager[li][0], picks[li][0], p_eager[li][1], K_) for li in sorted(p_eager)]
rep("compiled vs eager: logits", dy < 0.1 * sy, f"max|diff| {dy:.3e} (logit scale {sy:.1f})")
rep("compiled vs eager: grads", _gc < 0.05, f"all ||dg||/||g|| {_gc:.2e} (noise floor {g_noise:.2e}); nominator {_gn:.2e} (floor {_gn0:.2e})")
rep("compiled vs eager: pick sets per chunk", min(ov) > 0.95,
    "overlap per layer " + " ".join(f"{v:.3f}" for v in ov))
rep("compiled path used", all(mp.local_pack_global_nom_compile for mp in mps), "")
rep("flex ran", m.flex_union_status()["failed_modules"] == 0, "")

# 3. causality, compiled, train mode (noise drawn eagerly -> same noise for both inputs)
p0 = L // 2
x2 = x.clone(); x2[:, p0:] = torch.randint(0, inp.vocab_size, (B, L - p0), generator=torch.Generator().manual_seed(9)).cuda()
with torch.no_grad():
    def fw(xx):
        torch.manual_seed(77); torch.cuda.manual_seed(77)
        with torch.autocast("cuda", torch.bfloat16):
            o = m(xx)
        return (o[0] if isinstance(o, tuple) else o).float()
    lk = (fw(x)[:, :p0] - fw(x2)[:, :p0]).abs().max().item()
rep(f"compiled causal p={p0} (train mode, no_grad)", lk == 0.0, f"{lk:.3e}")
print("FAILS:", FAILS or "none")
