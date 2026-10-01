import os, sys, math
sys.argv = [sys.argv[0], "cuda"]
exec(open("/home/david/Projects/pinball/scripts/check_upper_stage.py").read().split("# 1. identity at init")[0])
CFG = "/home/david/Projects/pinball/configs/pinball_wikitext_pack_glob400_l0coarse_l0sel_z_d384_4k.yaml"
for norm in ("raw", "chunk"):
    m = build(CFG, seed=0, hier_layer_compile=False, local_pack_global_gumbel_norm=norm, local_pack_global_boost="key").train()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for n_, p in m.named_parameters():
            if "nom_" in n_: p.copy_((torch.randn(p.shape, generator=g) * (0.3 if p.dim() == 1 else 0.05)).to(p))
    mp = m.refinement_transformers[5].message_passing
    cap = {}
    o = mp._nominate_by_head
    mp._nominate_by_head = lambda spec, lv, xn: (cap.update(spec=spec, lv=lv, xn=xn.detach()), o(spec, lv, xn))[1]
    with torch.no_grad(): m(dna_in(2, 4096))
    spec, lv, xn = cap["spec"], cap["lv"], cap["xn"]
    torch.manual_seed(77); rows, gg, meta = o(spec, lv, xn)
    w = mp._nomination_weights(spec, lv, xn).detach()
    C, K, nch = meta["chunk"], meta["G"], meta["nch"]; W = int(spec["window"])
    allowed = spec[("nom_geom", C, W, K, mp.local_pack_global_candidates)][0]
    torch.manual_seed(77); u = torch.rand_like(w).clamp_(1e-6, 1 - 1e-6)
    tau = mp.local_pack_global_gumbel
    if norm == "raw":
        rank = (w - tau * torch.log(-torch.log(u))).unsqueeze(1).expand(-1, nch, -1)   # the OLD formula
    else:
        B, n = w.shape; rank = torch.empty(B, nch, n, device=w.device)
        for c in range(nch):
            al = allowed[c]
            if al.sum() == 0: rank[:, c] = 0; continue
            mu, sd = w[:, al].mean(1, keepdim=True), (w[:, al].var(1, unbiased=False, keepdim=True) + 1e-6).sqrt()
            rank[:, c] = (w - mu) / sd + tau * (-torch.log(-torch.log(u)))            # z + tau * Gumbel
    rank = rank.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    exp = torch.topk(rank, K, dim=-1).indices
    okv = meta["ok"].view(nch, K); got = rows.view(-1, nch, K)
    bad = sum(set(exp[b, c][okv[c]].tolist()) != set(got[b, c][okv[c]].tolist()) for b in range(got.size(0)) for c in range(nch) if okv[c].any())
    rep(f"gumbel_norm={norm}: picks == hand-derived ({'old raw formula' if norm == 'raw' else 'z + tau*Gumbel'})", bad == 0, f"mismatched chunks {bad}")
    # scale invariance of exploration (chunk only): scaling all weights x10 leaves the noisy picks unchanged
    if norm == "chunk":
        _nw = mp._nomination_weights
        mp._nomination_weights = lambda *a: 10.0 * _nw(*a)
        torch.manual_seed(77); rows2, _, _ = o(spec, lv, xn)
        same = all(set(rows2.view(-1, nch, K)[b, c][okv[c]].tolist()) == set(got[b, c][okv[c]].tolist()) for b in range(got.size(0)) for c in range(nch) if okv[c].any())
        rep("chunk: 10x weight scale leaves noisy picks unchanged", same, "")
        # and the raw mode would NOT be invariant (control): exploration shrinks with scale

    del m
print("FAILS:", FAILS or "none")
