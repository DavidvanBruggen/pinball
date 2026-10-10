"""Fused per-query descent level selection (xq descent, xq_select_kernel: triton).

One program per (batch, query) does what _xq_select_core's torch loop does for a level below
the start level, in registers:
  children of the query's kept parents (window-rule table) -> causal / local-exclusion mask
  -> indexer scores q . k_l[child] * scale over ONLY those candidates (+ Gumbel noise in
  training, drawn per NODE) -> top-k by repeated argmax (descending, like torch.topk), each
  pick retiring every slot of its node (dedup: a node reachable from two overlapping
  parents counts once, its first slot wins).
The torch loop instead scored every node of the level with a [queries x n_l] GEMM and then
gathered the candidate columns -- at L0 that is 16k x 16k per round for ~72 real candidates
per query. Outputs match the torch path's (kept ids, parent slot) up to ties and the fp32
(vs bf16-GEMM) score rounding; the noise uses Triton's Philox stream (same distribution)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _desc_level_kernel(Q, KL, PAR, CH, TL, QT, PROW, QROW, OUT_ID, OUT_SLOT, SEED,
                       n0, n_l, kp, Kc, scale, tau, excl, band_w,
                       D: tl.constexpr, DC: tl.constexpr, M: tl.constexpr, KK: tl.constexpr,
                       ALLOW_SAME: tl.constexpr, IS_L0: tl.constexpr, NOISE: tl.constexpr,
                       BAND: tl.constexpr):
    # Validity is carried as an int32 vector, never an i1 tensor reused as a 2-D mask: Triton
    # 3.6's RemoveLayoutConversions miscompiles i1 values that cross the 1-D / 2-D layouts.
    pid = tl.program_id(0)                        # = b * n0 + query
    b = pid // n0
    i = pid % n0
    m = tl.arange(0, M)
    pj = m // Kc
    cj = m % Kc
    par = tl.load(PAR + pid * kp + pj, mask=m < kp * Kc, other=-1)
    pok = (par >= 0).to(tl.int64)
    cid = tl.load(CH + (par * pok) * Kc + cj, mask=(m < kp * Kc) & (par >= 0), other=-1)
    okv = (cid >= 0).to(tl.int32)
    safe = cid * okv.to(tl.int64)                 # cid where valid, else 0
    tc = tl.load(TL + safe)
    qt = tl.load(QT + i)
    if ALLOW_SAME:
        okv = okv * (tc <= qt).to(tl.int32)
    else:
        okv = okv * (tc < qt).to(tl.int32)
    if IS_L0:
        okv = okv * (cid <= i - excl).to(tl.int32)
    if BAND > 0:
        # selection-time band exclusion: a node the packed window already gives this query
        # (0 <= q_row - row <= band_w causal, |.| <= band_w bidi) is not a candidate, so the
        # whole top-k budget lands outside the band
        dr = tl.load(QROW + i) - tl.load(PROW + safe)
        if BAND == 1:
            inb = (dr >= 0) & (dr <= band_w)
        else:
            inb = (dr <= band_w) & (dr >= -band_w)
        okv = okv * (1 - inb.to(tl.int32))
    row = b.to(tl.int64) * n_l + safe
    s = tl.zeros([M], tl.float32)
    for d0 in range(0, D, DC):                    # D in chunks: an [M, D] tile spills at M >= 256
        d = d0 + tl.arange(0, DC)
        q = tl.load(Q + pid.to(tl.int64) * D + d).to(tl.float32)
        kt = tl.load(KL + row[:, None] * D + d[None, :]).to(tl.float32)   # row 0 for invalid: masked below
        s += tl.sum(kt * q[None, :], axis=1)
    s = s * scale
    if NOISE:
        seed = tl.load(SEED)
        # noise keyed by NODE (not slot): a node reachable from two overlapping parents gets
        # the same score in both slots, so dedup-at-pick below is exact (one draw per node)
        u = tl.rand(seed, pid.to(tl.int64) * n_l + safe)
        u = tl.minimum(tl.maximum(u, 1e-6), 1.0 - 1e-6)
        s = s - tau * tl.log(-tl.log(u))
    s = s + (okv.to(tl.float32) - 1.0) * 3.0e38          # invalid -> ~-3e38 (sentinel)
    for r in range(KK):
        mx = tl.max(s, axis=0)
        am = tl.argmax(s, axis=0)
        nid = tl.sum(cid * (m == am).to(tl.int64), axis=0)
        live = (mx > -1.0e38).to(tl.int64)
        tl.store(OUT_ID + pid * KK + r, nid * live - (1 - live))
        tl.store(OUT_SLOT + pid * KK + r, am // Kc)
        # dedup at pick: retire EVERY slot holding this node (argmax took the first one)
        s = s - (cid == nid).to(tl.float32) * 3.0e38


def descent_level_select(q_all: torch.Tensor, k_l: torch.Tensor, par: torch.Tensor, ch: torch.Tensor,
                         t_l: torch.Tensor, q_t: torch.Tensor, kk: int, scale: float, tau: float,
                         allow_same: bool, is_l0: bool, local_exclude: int,
                         prow_l=None, qrow=None, band_w: int = 0, band_mode: int = 0):
    """-> (kept [B, n0, kk] local node ids or -1, parent slot [B, n0, kk]).
    q_all [B, n0, d] indexer queries; k_l [B, n_l, d] indexer keys of the level; par [B, n0, kp]
    kept parents (local ids at level + 1, -1 padding); ch [n_{l+1}, Kc] children table (-1
    padding); t_l [n_l] close times; q_t [n0] query times. band_mode 1 (causal) / 2 (bidi):
    drop candidates with 0 <= qrow - prow_l <= band_w (|.| bidi) -- prow_l [n_l] the level's
    packed rows, qrow [n0] the queries'. CUDA-graph safe (the noise seed is drawn on device)."""
    B, n0, d = (int(x) for x in q_all.shape)
    n_l, kp, Kc = int(k_l.size(1)), int(par.size(-1)), int(ch.size(-1))
    M = max(16, triton.next_power_of_2(kp * Kc))
    assert d == triton.next_power_of_2(d) and d >= 16 and M <= 1024, (d, kp, Kc)
    out_id = torch.empty(B, n0, kk, device=q_all.device, dtype=torch.int64)
    out_slot = torch.empty(B, n0, kk, device=q_all.device, dtype=torch.int64)
    noise = tau > 0.0
    seed = (torch.randint(0, 2 ** 31 - 1, (1,), device=q_all.device, dtype=torch.int64) if noise
            else torch.zeros(1, device=q_all.device, dtype=torch.int64))
    if band_mode:
        prow_l, qrow = prow_l.contiguous(), qrow.contiguous()
    else:
        prow_l = qrow = t_l                           # unused (BAND 0)
    _desc_level_kernel[(B * n0,)](
        q_all.contiguous(), k_l.contiguous(), par.contiguous(), ch.contiguous(), t_l.contiguous(),
        q_t.contiguous(), prow_l, qrow, out_id, out_slot, seed, n0, n_l, kp, Kc, float(scale), float(tau),
        int(local_exclude), int(band_w), D=d, DC=16, M=M, KK=int(kk), ALLOW_SAME=bool(allow_same),
        IS_L0=bool(is_l0), NOISE=noise, BAND=int(band_mode), num_warps=1)    # 1 warp: 0.07 ms (M 32) / 0.47 ms (L0, M 128, k 32) vs 0.33 / 1.69 at 4
    return out_id, out_slot
