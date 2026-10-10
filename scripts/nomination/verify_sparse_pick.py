"""sparse_pick_attention (Triton) vs the dense fp32 reference, then speed vs the gather read.

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_sparse_pick.py [bench]
Checks (fp32 and bf16 inputs; per-head G == H and grouped G in {1, 2}): o, lse, dq, dk, dv
against sparse_pick_attention_ref with a random dO AND a random dlse (the merge path), on
picks with padding (-1), duplicates, rows with no valid pick, and popular keys (many queries
on one row: the dK/dV segment split). 'bench' times forward+backward at the descent's 16k x4
shape against the current gather read (_hqd_packed_chunk math: fp32 gather + softmax)."""
import sys, time, torch
from pinball.model.layers.sparse_pick_attn import sparse_pick_attention, sparse_pick_attention_ref

FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

def make(B, Nq, Nk, H, G, D, K, dtype, popular=0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, Nq, H, D, device="cuda", generator=g).to(dtype)
    k = torch.randn(B, Nk, G, D, device="cuda", generator=g).to(dtype)
    v = torch.randn(B, Nk, G, D, device="cuda", generator=g).to(dtype)
    idx = torch.randint(0, Nk, (B, Nq, K), device="cuda", generator=g)
    idx[torch.rand(B, Nq, K, device="cuda", generator=g) < 0.2] = -1        # padding
    if popular:
        idx[:, :, :popular] = torch.arange(popular, device="cuda")           # every query on rows 0..p-1
    idx[:, 3] = -1                                                           # a row with no pick
    idx[:, 5, 1] = idx[:, 5, 0]                                              # a duplicate pick
    return q, k, v, idx

def check(name, B, Nq, Nk, H, G, D, K, dtype, popular=0):
    q, k, v, idx = make(B, Nq, Nk, H, G, D, K, dtype, popular)
    sc = D ** -0.5
    ins = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    o, lse = sparse_pick_attention(*ins, idx, sc)
    ref_in = [t.detach().clone().float().requires_grad_(True) for t in (q, k, v)]
    o_r, lse_r = sparse_pick_attention_ref(*ref_in, idx, sc)
    g = torch.Generator(device="cuda").manual_seed(9)
    do = torch.randn(o.shape, device="cuda", generator=g)
    dl = torch.randn(lse.shape, device="cuda", generator=g)
    fin = torch.isfinite(lse_r)
    lse_c, lse_rc = torch.where(fin, lse, 0.0), torch.where(fin, lse_r, 0.0)
    ((o.float() * do).sum() + (lse_c * dl).sum()).backward()
    ((o_r * do).sum() + (lse_rc * dl).sum()).backward()
    tol = 2e-4 if dtype == torch.float32 else 2e-2
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()
    res = {"o": rel(o, o_r), "lse": rel(lse_c, lse_rc), "dq": rel(ins[0].grad, ref_in[0].grad),
           "dk": rel(ins[1].grad, ref_in[1].grad), "dv": rel(ins[2].grad, ref_in[2].grad)}
    with torch.no_grad():
        _, _, pk = sparse_pick_attention(q, k, v, idx, sc, return_p=True)
        _, _, pr = sparse_pick_attention_ref(q, k, v, idx, sc, return_p=True)
    res["p"] = rel(pk, pr)
    with torch.no_grad():
        rows = torch.arange(0, Nq, 7, device="cuda")
        _, _, ps = sparse_pick_attention(q, k, v, idx, sc, return_p=True, p_rows=rows)
    res["p_rows"] = rel(ps, pr[:, rows])
    empty_ok = bool((lse[:, 3] == float("-inf")).all()) and float(o[:, 3].detach().abs().max()) == 0.0
    rep(name, all(x <= tol for x in res.values()) and empty_ok,
        " ".join(f"{a} {b:.1e}" for a, b in res.items()) + f" | empty row ok {empty_ok}")

print("1. correctness vs dense fp32 reference (max rel err)")
for dt in (torch.float32, torch.bfloat16):
    tag = "fp32" if dt == torch.float32 else "bf16"
    check(f"{tag} per-head G=H=6", 2, 300, 500, 6, 6, 64, 56, dt)
    check(f"{tag} grouped G=2", 2, 300, 500, 6, 2, 64, 56, dt)
    check(f"{tag} MQA G=1", 2, 300, 500, 6, 1, 64, 56, dt)
    check(f"{tag} popular rows (segment split)", 2, 1000, 500, 6, 6, 64, 24, dt, popular=8)
    check(f"{tag} H=8 D=32 K=7", 1, 257, 64, 8, 8, 32, 7, dt)
print("PASS" if not FAILS else f"FAIL: {FAILS}")

if len(sys.argv) > 1 and sys.argv[1] == "bench":
    print("\n2. speed at the descent shape (fwd + bwd, bf16 inputs)")
    import math
    B, Nq, Nk, H, D, K = 4, 16384, 20352, 6, 64, 56
    q, k, v, idx = make(B, Nq, Nk, H, H, D, K, torch.bfloat16)
    # realistic sharing: 24 coarse picks from a few hundred rows (many queries each), 32 tokens
    g = torch.Generator(device="cuda").manual_seed(1)
    idx[..., :24] = 16384 + torch.randint(0, 3968, (B, Nq, 24), device="cuda", generator=g)
    pos = torch.arange(Nq, device="cuda").view(1, -1, 1)
    idx[..., 24:] = (pos - 128 - torch.randint(0, 4096, (B, Nq, 32), device="cuda", generator=g)).clamp(min=-1)
    sc = D ** -0.5
    def gather_read(q, k, v, idx, chunk=2048):
        # the current read's math (fp32 gather, sink-free softmax), per 2048-query chunk
        ok = idx >= 0; safe = idx.clamp(min=0)
        bo = torch.arange(B, device="cuda").view(B, 1, 1) * Nk
        fk, fv = k.float().reshape(-1, H, D), v.float().reshape(-1, H, D)
        outs = []
        for s0 in range(0, Nq, chunk):
            gi = (safe[:, s0:s0 + chunk] + bo).reshape(-1)
            kc = fk.index_select(0, gi).view(B, -1, K, H, D); vc = fv.index_select(0, gi).view(B, -1, K, H, D)
            s = (q[:, s0:s0 + chunk].float().unsqueeze(2) * kc).sum(-1) * sc
            s = s.masked_fill(~ok[:, s0:s0 + chunk].unsqueeze(-1), -1e30)
            w = torch.softmax(s, 2)
            outs.append((w.unsqueeze(-1) * vc).sum(2).to(q.dtype))
        return torch.cat(outs, 1)
    def timeit(fn, n=10):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
    qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    do = torch.randn(B, Nq, H, D, device="cuda", dtype=torch.bfloat16)
    def kern_fb():
        o, lse = sparse_pick_attention(qq, kk, vv, idx, sc); o.backward(do)
    def kern_f():
        with torch.no_grad(): sparse_pick_attention(q, k, v, idx, sc)
    def gath_fb():
        o = gather_read(qq, kk, vv, idx); o.backward(do)
    def gath_f():
        with torch.no_grad(): gather_read(q, k, v, idx)
    tk_f, tk_fb = timeit(kern_f), timeit(kern_fb)
    tg_f, tg_fb = timeit(gath_f), timeit(gath_fb)
    print(f"  kernel: fwd {tk_f:.2f} ms, fwd+bwd {tk_fb:.2f} ms")
    print(f"  gather: fwd {tg_f:.2f} ms, fwd+bwd {tg_fb:.2f} ms   (per read; 4 reads per step)")
    print(f"  speedup fwd {tg_f / tk_f:.1f}x, fwd+bwd {tg_fb / tk_fb:.1f}x")
