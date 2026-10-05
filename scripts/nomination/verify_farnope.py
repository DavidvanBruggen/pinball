"""Far-path partial RoPE (+ far bias) checks on the l0coarse q config, token mode, fp32 eager.
usage: python verify_farnope.py ND BIAS   (e.g. 32 off | 32 level)"""
import sys, math, logging, torch
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs

CFG = "/home/david/Projects/pinball/configs/pinball_wikitext_pack_glob400_l0coarse_l0sel_q_d384_4k.yaml"
ND = int(sys.argv[1]) if len(sys.argv) > 1 else 32
BIAS = sys.argv[2] if len(sys.argv) > 2 else "off"
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

def mk(nd, bias, L=4096):
    cfg = PinballConfig.from_yaml(CFG)
    cfg.hier_layer_compile = False
    cfg.local_pack_far_nope_dims = nd
    cfg.local_pack_far_bias = bias
    inp = resolve_model_inputs(cfg, block_size=L)
    torch.manual_seed(0)
    m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                    tie_weights=inp.tie_weights, max_seq_len=inp.block_size).cuda().eval()
    return m, inp

m, inp = mk(ND, BIAS)
mps = [t.message_passing for t in m.refinement_transformers]
rep("knobs reach every layer", all(mp.local_pack_far_nope_dims == ND and mp.local_pack_far_bias == BIAS for mp in mps), "")
# informative weights: nonzero u, bias, gates
g = torch.Generator().manual_seed(5)
with torch.no_grad():
    for n_, p in m.named_parameters():
        if "nom_" in n_ or "far_level_bias" in n_:
            sc = 0.3 if p.dim() <= 1 else (0.5 if ("boost_u" in n_ or "far_level" in n_) else 0.05)
            p.copy_((torch.randn(p.shape, generator=g) * sc).to(p))
L, B = 4096, 2
x = torch.randint(0, inp.vocab_size, (B, L), generator=torch.Generator().manual_seed(3)).cuda()

caps = {}
for li, mp in enumerate(mps):
    def mk_spy(mp, li):
        o_attn = mp._flex_union_attn
        o_rot = mp.rotary_pos_enc.apply_rotary_pos_emb
        rot_log = []
        def spy_rot(t, pos):
            out = o_rot(t, pos); rot_log.append((t.detach(), out.detach())); return out
        def spy_attn(qp, kp, vp, spec, causal=True, gsel=None):
            out = o_attn(qp, kp, vp, spec, causal, gsel=gsel)
            caps[li] = dict(qp=qp.detach(), kp=kp.detach(), vp=vp.detach(), out=out.detach(), spec=spec,
                            gsel=gsel, rot=list(rot_log[-2:]), mp=mp)
            rot_log.clear()
            return out
        return spy_attn, spy_rot
    mp._flex_union_attn, mp.rotary_pos_enc.apply_rotary_pos_emb = mk_spy(mp, li)
def fwd(mm, xx, **kw):
    with torch.no_grad():
        o = mm(xx, **kw)
    return (o[0] if isinstance(o, tuple) else o).float()
y = fwd(m, x)
rep("flex ran", m.flex_union_status()["failed_modules"] == 0, "")

for li in (0, 6, 11):
    c = caps[li]; mp = c["mp"]; spec = c["spec"]; gsel = c["gsel"]; meta = gsel[2]
    qp, kp, vp = c["qp"], c["kp"], c["vp"]
    n = int(spec["num_nodes"]); H, D = qp.size(2), qp.size(3)
    # 1. partial rotation: head dims == rotated, tail == unrotated (q and k), bias dim
    (qi_, qo_), (ki_, ko_) = c["rot"]
    qv, kv = qp.reshape(B * n, H, D), kp.reshape(B * n, H, D)
    nb = 1 if BIAS != "off" else 0
    e_head = max((qv[..., :D - ND] - qo_[..., :D - ND]).abs().max().item(), (kv[..., :D - ND] - ko_[..., :D - ND]).abs().max().item())
    e_tail = max((qv[..., D - ND:D - nb] - qi_[..., D - ND:D - nb]).abs().max().item(), (kv[..., D - ND:D - nb] - ki_[..., D - ND:D - nb]).abs().max().item())
    tail_rot = (qo_[..., D - ND:] - qi_[..., D - ND:]).abs().max().item()
    rep(f"L{li} q/k: rotated head, UNROTATED tail", e_head == 0 and e_tail == 0,
        f"head err {e_head:.1e}, tail err {e_tail:.1e} (full RoPE would move the tail by {tail_rot:.2e})")
    if nb:
        rep(f"L{li} reserved dim: q == 1, k == 0", bool((qv[..., -1] == 1).all() and (kv[..., -1] == 0).all()), "")
    # 2. dense reference with exactly-once keys
    assert spec.get("flex_ring_wtok", None) is None, "ring windows not covered by this reference"
    C_, G_ = int(meta["chunk"]), int(meta["G"]); W_ = int(spec["window"])
    srows = meta["static_rows"]; rows = gsel[0]; gl = gsel[1]; ok = meta["ok"]
    lvl = spec["levels"]
    def far(k, lv):
        k = torch.cat([torch.zeros_like(k[..., :D - ND]), k[..., D - ND:]], -1)
        if nb:
            b = mp.far_level_bias.float()[:, lv.clamp(max=mp.far_level_bias.size(1) - 1)].t()   # [R, H]
            k = torch.cat([k[..., :-1], (b * math.sqrt(D)).unsqueeze(-1)], -1)
        return k
    um = torch.zeros(D, device=qp.device); um[D - ND:] = 1
    if nb: um[-1] = 0
    gq = torch.Generator().manual_seed(li)
    qs = torch.cat([torch.randint(0, n, (40,), generator=gq), torch.tensor([n - 1, 3000, 1500, 700, 200])]).tolist()
    err, nst_band = 0.0, 0
    for b in range(B):
        for qi in qs:
            ch = qi // C_
            sr = srows[(srows <= qi) & ~((qi - srows) <= W_)]            # closed, outside band
            nst_band += int(((srows <= qi) & ((qi - srows) <= W_)).sum())
            sl_idx = torch.arange(ch * G_, (ch + 1) * G_, device=qp.device)
            sl_idx = sl_idx[ok[sl_idx]]
            sl = rows[b, sl_idx]
            band = torch.arange(max(0, qi - W_), qi + 1, device=qp.device)  # ALL rows, static included
            ks_st = far(kp[b, sr], lvl[sr])
            z = gl[b, sl_idx] - mp.nom_gate_bias
            ks_sl = far(kp[b, sl] + z.view(-1, 1, 1) * (mp.nom_boost_u.float() * um).view(1, H, D), lvl[sl])
            kk = torch.cat([ks_st, ks_sl, kp[b, band]])
            vv = torch.cat([vp[b, sr], vp[b, sl] * torch.sigmoid(gl[b, sl_idx]).view(-1, 1, 1), vp[b, band]])
            lg = torch.einsum("hd,khd->hk", qp[b, qi], kk) / math.sqrt(D)
            o = torch.einsum("hk,khd->hd", lg.softmax(-1), vv)
            err = max(err, (o - c["out"][b, qi]).abs().max().item())
    rep(f"L{li} flex out == dense reference (exactly-once keys)", err < 2e-3,
        f"max|diff| {err:.2e}; band-near static rows routed via the band: {nst_band}")

# 3. far scores are position-free: shifting the slot positions cannot matter -> a far key's
#    rotated dims are zero, so q.k_far == q_tail . k_tail exactly (checked in the reference).
# 4. causality (eval) and no cross-sample leak
for p0 in (4000, 2500, 900):
    x2 = x.clone(); x2[:, p0:] = torch.randint(0, inp.vocab_size, (B, L - p0), generator=torch.Generator().manual_seed(p0)).cuda()
    lk = (y[:, :p0] - fwd(m, x2)[:, :p0]).abs().max().item()
    rep(f"causal p={p0}", lk == 0.0, f"{lk:.3e}")
x3 = x.clone(); x3[1] = torch.randint(0, inp.vocab_size, (L,), generator=torch.Generator().manual_seed(9)).cuda()
rep("no cross-sample leak", (y[0] - fwd(m, x3)[0]).abs().max().item() == 0.0, "")
# 5. causality in train mode (gumbel + dropout, same seed)
m.train()
def fwd_t(xx):
    torch.manual_seed(123); torch.cuda.manual_seed(123)
    return fwd(m, xx)
yt = fwd_t(x)
x2 = x.clone(); x2[:, 1500:] = torch.randint(0, inp.vocab_size, (B, L - 1500), generator=torch.Generator().manual_seed(1)).cuda()
lk = (yt[:, :1500] - fwd_t(x2)[:, :1500]).abs().max().item()
rep("causal in train mode", lk == 0.0, f"{lk:.3e}")
# 6. grads
for p in m.parameters(): p.grad = None
o = m(x[:1]); o = o[0] if isinstance(o, tuple) else o
o.float().logsumexp(-1).mean().backward()
names = [n_ for n_, p in m.named_parameters() if ("far_level_bias" in n_ or "nom_boost_u" in n_)]
zg = [n_ for n_, p in m.named_parameters() if n_ in names and (p.grad is None or p.grad.abs().max() == 0)]
rep("grads reach u (and far bias)", not zg and names, f"{len(names)} params; zero-grad {zg[:3]}")
m.eval()
# 7. generation consistency: variable-length prefix frontier == training forward
look = int(m._gen_frontier_lookahead())
pad_id = int(getattr(m, "pad_token_id", None) or getattr(m, "mask_token_id", 0) or 0)
for Lp in (1500, 777):
    xp = torch.cat([x[:, :Lp], torch.full((B, look), pad_id, dtype=x.dtype, device=x.device)], 1)
    d = (fwd(m, xp, logits_last_index=Lp - 1)[:, -1] - y[:, Lp - 1]).abs().max().item()
    rep(f"gen prefix {Lp} frontier == training", d < 1e-4, f"{d:.3e}")
print("FAILS:", FAILS or "none")
