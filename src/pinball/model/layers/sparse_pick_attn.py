"""Per-query sparse attention over picked rows (xq descent v2 read), fused in Triton.

Each query attends over its OWN K picked rows of a node table -- the descent's per-query
beam picks -- with no materialised [B, Q, K, H, D] gather:

    s_j = scale * q . k[idx_j]      (j < K, idx_j >= 0; idx_j = -1 is padding)
    o   = sum_j softmax(s)_j v[idx_j],   lse = log sum_j exp(s_j)

q [B, Nq, H, D]; k, v [B, Nk, G, D] with G | H (G == H per-head, G < H grouped: head h reads
group h // (H // G)); idx [B, Nq, K]. Returns o [B, Nq, H, D] (q's dtype) and lse [B, Nq, H]
(fp32, -inf for a query with no valid pick, whose o is 0). lse is differentiable, so the read
can be merged EXACTLY into another softmax over a disjoint key set:
    o = (o_A e^{L_A} + o_B e^{L_B}) / (e^{L_A} + e^{L_B}).

Kernels (all fp32 math on bf16/fp32 inputs):
  forward  one program per query, all heads: the picks' K/V rows are loaded straight from the
           table (H*D contiguous per pick) and folded in an online softmax.
  dQ       the same loop, recomputing p from the saved lse.
  dK/dV    an inverted index -- picks sorted by key, split into segments of <= SEG entries --
           so each program accumulates ONE key's gradient over its queries in registers and
           adds it once. Popular coarse rows (picked by ~1000 queries) cost ceil(n/SEG)
           atomic adds instead of n; that contention is what made the gather read's
           index_add backward slow.
The lse gradient enters as delta_eff = sum(dO * o) - dlse (ds = p * (dp - delta_eff)).
Not deterministic (fp32 atomics across segments of one key)."""
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _pick_fwd(Q, K, V, IDX, O, LSE, P, PSLOT, scale,
              Nq, Nk, KP, NP,
              H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
              HP: tl.constexpr, BK: tl.constexpr, STORE_P: tl.constexpr):
    pid = tl.program_id(0)                       # = b * Nq + query
    b = pid // Nq
    h = tl.arange(0, HP)
    d = tl.arange(0, D)
    hm = h < H
    kvh = h // (H // G)
    q = tl.load(Q + pid * H * D + h[:, None] * D + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    m_i = tl.full([HP], -1.0e30, tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for k0 in range(0, KP, BK):
        j = k0 + tl.arange(0, BK)
        ids = tl.load(IDX + pid * KP + j, mask=j < KP, other=-1)
        ok = ids >= 0
        row = (b * Nk + tl.where(ok, ids, 0)).to(tl.int64)
        off = row[:, None, None] * (G * D) + kvh[None, :, None] * D + d[None, None, :]
        msk = ok[:, None, None] & hm[None, :, None]
        kt = tl.load(K + off, mask=msk, other=0.0).to(tl.float32)            # [BK, HP, D]
        s = tl.sum(q[None, :, :] * kt, axis=2) * scale                       # [BK, HP]
        s = tl.where(ok[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[None, :])
        l_i = l_i * alpha + tl.sum(p, axis=0)
        vt = tl.load(V + off, mask=msk, other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * vt, axis=0)
        m_i = m_new
    live = l_i > 0.0
    o = acc / tl.where(live, l_i, 1.0)[:, None]
    tl.store(O + pid * H * D + h[:, None] * D + d[None, :], o.to(O.dtype.element_ty), mask=hm[:, None])
    lse = tl.where(live, m_i + tl.log(tl.where(live, l_i, 1.0)), float("-inf"))
    tl.store(LSE + pid * H + h, lse, mask=hm)
    if STORE_P:
        # head-mean attention per pick (the indexer-KL target): a second pass over the keys,
        # only for the KL's sampled queries (PSLOT[i] = row in P, or -1)
        slot = tl.load(PSLOT + pid % Nq)
        for k0 in range(0, KP * (slot >= 0).to(tl.int32), BK):
            j = k0 + tl.arange(0, BK)
            ids = tl.load(IDX + pid * KP + j, mask=j < KP, other=-1)
            ok = ids >= 0
            row = (b * Nk + tl.where(ok, ids, 0)).to(tl.int64)
            off = row[:, None, None] * (G * D) + kvh[None, :, None] * D + d[None, None, :]
            kt = tl.load(K + off, mask=ok[:, None, None] & hm[None, :, None], other=0.0).to(tl.float32)
            s = tl.sum(q[None, :, :] * kt, axis=2) * scale
            pr = tl.where(ok[:, None] & live[None, :] & hm[None, :], tl.exp(s - lse[None, :]), 0.0)
            tl.store(P + (b * NP + slot) * KP + j, tl.sum(pr, axis=1) / H, mask=j < KP)


@triton.jit
def _pick_bwd_dq(Q, K, V, IDX, DO, LSE, DELTA, DQ, scale,
                 Nq, Nk, KP,
                 H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                 HP: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // Nq
    h = tl.arange(0, HP)
    d = tl.arange(0, D)
    hm = h < H
    kvh = h // (H // G)
    qo = pid * H * D + h[:, None] * D + d[None, :]
    q = tl.load(Q + qo, mask=hm[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + qo, mask=hm[:, None], other=0.0).to(tl.float32)
    lse = tl.load(LSE + pid * H + h, mask=hm, other=float("-inf"))
    live = lse > float("-inf")
    lse = tl.where(live, lse, 0.0)
    delta = tl.load(DELTA + pid * H + h, mask=hm, other=0.0)
    dq = tl.zeros([HP, D], tl.float32)
    for k0 in range(0, KP, BK):
        j = k0 + tl.arange(0, BK)
        ids = tl.load(IDX + pid * KP + j, mask=j < KP, other=-1)
        ok = ids >= 0
        row = (b * Nk + tl.where(ok, ids, 0)).to(tl.int64)
        off = row[:, None, None] * (G * D) + kvh[None, :, None] * D + d[None, None, :]
        msk = ok[:, None, None] & hm[None, :, None]
        kt = tl.load(K + off, mask=msk, other=0.0).to(tl.float32)
        vt = tl.load(V + off, mask=msk, other=0.0).to(tl.float32)
        s = tl.sum(q[None, :, :] * kt, axis=2) * scale
        p = tl.where(ok[:, None] & live[None, :], tl.exp(s - lse[None, :]), 0.0)
        dp = tl.sum(do[None, :, :] * vt, axis=2)
        ds = p * (dp - delta[None, :])
        dq += tl.sum(ds[:, :, None] * kt, axis=0)
    tl.store(DQ + qo, (dq * scale).to(DQ.dtype.element_ty), mask=hm[:, None])


@triton.jit
def _pick_bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, SEG_START, SEG_END, SEG_KEY, ENTRY_Q, scale,
                   H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                   HP: tl.constexpr, BE: tl.constexpr):
    sid = tl.program_id(0)
    e0 = tl.load(SEG_START + sid)
    e1 = tl.load(SEG_END + sid)
    key = tl.load(SEG_KEY + sid).to(tl.int64)               # b * Nk + node
    h = tl.arange(0, HP)
    d = tl.arange(0, D)
    hm = h < H
    kvh = h // (H // G)
    ko = key * (G * D) + kvh[:, None] * D + d[None, :]
    kj = tl.load(K + ko, mask=hm[:, None], other=0.0).to(tl.float32)     # [HP, D]
    vj = tl.load(V + ko, mask=hm[:, None], other=0.0).to(tl.float32)
    dk = tl.zeros([HP, D], tl.float32)
    dv = tl.zeros([HP, D], tl.float32)
    for c0 in range(e0, e1, BE):
        e = c0 + tl.arange(0, BE)
        em = e < e1
        qi = tl.load(ENTRY_Q + e, mask=em, other=0).to(tl.int64)          # b * Nq + query
        qo = qi[:, None, None] * (H * D) + h[None, :, None] * D + d[None, None, :]
        msk = em[:, None, None] & hm[None, :, None]
        qt = tl.load(Q + qo, mask=msk, other=0.0).to(tl.float32)          # [BE, HP, D]
        dot = tl.load(DO + qo, mask=msk, other=0.0).to(tl.float32)
        lo = qi[:, None] * H + h[None, :]
        lse = tl.load(LSE + lo, mask=em[:, None] & hm[None, :], other=float("-inf"))
        live = lse > float("-inf")
        delta = tl.load(DELTA + lo, mask=em[:, None] & hm[None, :], other=0.0)
        s = tl.sum(qt * kj[None, :, :], axis=2) * scale                   # [BE, HP]
        p = tl.where(live, tl.exp(s - tl.where(live, lse, 0.0)), 0.0)
        dv += tl.sum(p[:, :, None] * dot, axis=0)
        dp = tl.sum(dot * vj[None, :, :], axis=2)
        ds = p * (dp - delta)
        dk += tl.sum(ds[:, :, None] * qt, axis=0)
    live_seg = e1 > e0                       # padded segments (sync-free grid) are empty
    tl.atomic_add(DK + ko, dk * scale, mask=hm[:, None] & live_seg)
    tl.atomic_add(DV + ko, dv, mask=hm[:, None] & live_seg)


def _pick_inverse(idx: torch.Tensor, Nk: int, seg: int):
    """Inverted index of the picks, SYNC-FREE (no boolean indexing / nonzero, so the backward
    never stalls the host): every (query, slot) entry is sorted by key (b * Nk + node; padding
    sorts last under a sentinel), runs of one key are cut into segments of <= seg entries,
    and the segment table is a fixed-size buffer of UB = min(E, B * Nk + E // seg + 1) rows
    whose unused tail is empty (start == E > end). -> (seg_start, seg_end, seg_key, entry_q
    = b * Nq + query per sorted entry, UB)."""
    B, Nq, KP = (int(x) for x in idx.shape)
    dev = idx.device
    E = B * Nq * KP
    sent = B * Nk
    # int32 keys halve the radix passes (-3 ms at 176 picks, 16k x4); b * Nk + node only
    # outgrows them past ~2.1e9 nodes per batch (1M tokens ~ 1.24M nodes, so B ~ 1700).
    # Positions into the sorted entry list (E = B * Nq * KP) go int64 past 2^31.
    kdt = torch.int32 if sent < 2 ** 31 - 1 else torch.int64
    pdt = torch.int32 if E < 2 ** 31 - 1 else torch.int64
    flat = idx.reshape(B, Nq * KP).to(kdt)
    key = torch.where(flat >= 0, flat + torch.arange(B, device=dev, dtype=kdt).view(B, 1) * Nk,
                      torch.full_like(flat, sent)).reshape(-1)
    skey, order = torch.sort(key)
    entry_q = (order // KP).to(pdt)                       # flat entry (b, q, slot) -> b * Nq + q
    valid = skey < sent
    pos = torch.arange(E, device=dev)
    first = valid.clone()
    first[1:] &= skey[1:] != skey[:-1]
    run0 = torch.searchsorted(skey, skey)                  # run start (keys are sorted; cummax: 5.3 ms)
    start = first | (valid & ((pos - run0) % seg == 0))
    slot = torch.cumsum(start.to(pdt), 0) - 1
    UB = max(1, min(E, sent + E // seg + 1))
    seg_start = torch.full((UB + 1,), E, dtype=pdt, device=dev)
    seg_start.scatter_(0, torch.where(start, slot.long(), torch.full_like(pos, UB)), pos.to(pdt))
    seg_start = seg_start[:UB].contiguous()
    n_valid = valid.sum().to(pdt)
    seg_end = torch.minimum(torch.cat([seg_start[1:], seg_start.new_full((1,), E)]), n_valid).contiguous()
    seg_key = skey.gather(0, seg_start.clamp(max=E - 1).long()).contiguous()
    return seg_start, seg_end, seg_key, entry_q, UB


def _cfg(H: int, D: int) -> Tuple[int, int, int]:
    """(HP, BK, num_warps) for the per-query kernels. Small tiles win: the loops are gather-
    and latency-bound, and bigger [BK, HP, D] tiles spill (16k x4, H 6, D 64, Blackwell:
    fwd 0.64 ms at BK 1 / 1 warp vs 3.6 ms at BK 16 / 4 warps)."""
    HP = max(1, triton.next_power_of_2(H))
    BK = max(1, 512 // (HP * D))
    return HP, BK, max(1, min(4, (HP * D) // 512))


SEG = 128            # entries per dK/dV segment
BE = 1               # entries per dK/dV inner step (1 entry / 2 warps: 0.90 ms vs 3.8 at 8 / 4)
DKDV_WARPS = 2


# Registered as custom ops (not an autograd.Function under dynamo.disable) so a compiled layer
# traces through the read without a graph break; the ops are opaque to inductor.
@torch.library.custom_op("pinball::sparse_pick_fwd", mutates_args=())
def _sparse_pick_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor,
                     scale: float, return_p: bool, p_slot: torch.Tensor,
                     n_p: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Nq, H, D = (int(x) for x in q.shape)
    Nk, G = int(k.size(1)), int(k.size(2))
    assert H % G == 0 and D == triton.next_power_of_2(D), (H, G, D)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    idx = idx.to(torch.int32).contiguous()
    KP = int(idx.size(-1))
    o = torch.empty_like(q)
    lse = torch.empty(B, Nq, H, device=q.device, dtype=torch.float32)
    p = torch.zeros((B, n_p, KP) if return_p else (0,), device=q.device, dtype=torch.float32)
    HP, BK, nw = _cfg(H, D)
    _pick_fwd[(B * Nq,)](q, k, v, idx, o, lse, p if return_p else lse, p_slot.to(torch.int32).contiguous(),
                         float(scale), Nq, Nk, KP, int(n_p),
                         H=H, G=G, D=D, HP=HP, BK=BK, STORE_P=bool(return_p), num_warps=nw)
    return o, lse, p


@_sparse_pick_fwd.register_fake
def _(q, k, v, idx, scale, return_p, p_slot, n_p):
    B, Nq, H, D = q.shape
    KP = idx.shape[-1]
    return (q.new_empty(q.shape), q.new_empty((B, Nq, H), dtype=torch.float32),
            q.new_empty((B, n_p, KP) if return_p else (0,), dtype=torch.float32))


@torch.library.custom_op("pinball::sparse_pick_bwd", mutates_args=())
def _sparse_pick_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor,
                     o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, dlse: torch.Tensor,
                     scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Nq, H, D = (int(x) for x in q.shape)
    Nk, G = int(k.size(1)), int(k.size(2))
    q, k, v, o, do = q.contiguous(), k.contiguous(), v.contiguous(), o.contiguous(), do.contiguous()
    idx = idx.to(torch.int32).contiguous()
    KP = int(idx.size(-1))
    delta = (do.float() * o.float()).sum(-1)
    delta = (delta - torch.where(torch.isfinite(lse), dlse.float(), torch.zeros_like(lse))).contiguous()
    HP, BK, nw = _cfg(H, D)
    dq = torch.empty_like(q)
    _pick_bwd_dq[(B * Nq,)](q, k, v, idx, do, lse, delta, dq, float(scale), Nq, Nk, KP,
                            H=H, G=G, D=D, HP=HP, BK=BK, num_warps=nw)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    st, en, sk, eq, UB = _pick_inverse(idx, Nk, SEG)
    _pick_bwd_dkdv[(UB,)](q, k, v, do, lse, delta, dk, dv, st, en, sk, eq, float(scale),
                          H=H, G=G, D=D, HP=HP, BE=BE, num_warps=DKDV_WARPS)
    return dq, dk.to(k.dtype), dv.to(v.dtype)


@_sparse_pick_bwd.register_fake
def _(q, k, v, idx, o, lse, do, dlse, scale):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def _setup_ctx(ctx, inputs, output):
    q, k, v, idx, scale = inputs[:5]
    o, lse, _ = output
    ctx.save_for_backward(q, k, v, idx, o, lse)
    ctx.scale = float(scale)


def _backward(ctx, do, dlse, dp):
    q, k, v, idx, o, lse = ctx.saved_tensors
    do = torch.zeros_like(o) if do is None else do
    dlse = torch.zeros_like(lse) if dlse is None else dlse
    dq, dk, dv = _sparse_pick_bwd(q, k, v, idx, o, lse, do, dlse, ctx.scale)
    return dq, dk, dv, None, None, None, None, None


_sparse_pick_fwd.register_autograd(_backward, setup_context=_setup_ctx)


def sparse_pick_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor,
                          scale: float, return_p: bool = False, p_rows=None):
    """-> (o [B, Nq, H, D], lse [B, Nq, H] fp32[, p fp32 head-mean attention per pick,
    non-differentiable, when return_p]). p is [B, Nq, K], or [B, len(p_rows), K] for the
    queries p_rows (sorted indices, e.g. the indexer KL's sample: the second pass then runs
    only for them). See the module docstring."""
    Nq = int(q.size(1))
    if return_p and p_rows is not None:
        n_p = int(p_rows.numel())
        p_slot = torch.full((Nq,), -1, dtype=torch.int32, device=q.device)
        p_slot[p_rows] = torch.arange(n_p, dtype=torch.int32, device=q.device)
    else:
        n_p = Nq if return_p else 0
        p_slot = torch.arange(Nq, dtype=torch.int32, device=q.device)
    o, lse, p = _sparse_pick_fwd(q, k, v, idx, float(scale), bool(return_p), p_slot, n_p)
    return (o, lse, p.detach()) if return_p else (o, lse)


def sparse_pick_attention_ref(q, k, v, idx, scale, return_p=False):
    """Dense fp32 reference (materialises the gather): same outputs as sparse_pick_attention."""
    B, Nq, H, D = q.shape
    G = int(k.size(2))
    ok = idx >= 0
    safe = idx.clamp(min=0).long()
    bo = torch.arange(B, device=q.device).view(B, 1, 1) * k.size(1)
    kf = k.float().reshape(-1, G, D).index_select(0, (safe + bo).reshape(-1)).view(B, Nq, -1, G, D)
    vf = v.float().reshape(-1, G, D).index_select(0, (safe + bo).reshape(-1)).view(B, Nq, -1, G, D)
    kf = kf.repeat_interleave(H // G, dim=3)
    vf = vf.repeat_interleave(H // G, dim=3)
    s = (q.float().unsqueeze(2) * kf).sum(-1) * scale                    # [B, Nq, K, H]
    s = s.masked_fill(~ok.unsqueeze(-1), float("-inf"))
    lse = torch.logsumexp(s, dim=2)                                       # [B, Nq, H]
    p = torch.where(torch.isfinite(lse).unsqueeze(2), (s - lse.unsqueeze(2)).exp(), torch.zeros_like(s))
    o = (p.unsqueeze(-1) * vf).sum(2)
    return (o, lse, p.mean(-1)) if return_p else (o, lse)
