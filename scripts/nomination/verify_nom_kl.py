"""local_pack_global_nom_kl probe (lightning-style KL on the head nominator's slot weights).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_nom_kl.py CONFIG [L] [B]
CONFIG is a head-nominator stream slot arm (chunk slots in the flex prefix). Eager layers, so
comparisons are exact. Checks:
  1. the forward is untouched: train-mode logits with the KL on are BIT-identical to it off
     (the KL's chunk sampling runs under a forked RNG so dropout draws line up);
  2. the KL target is the real flex softmax: for sampled queries the returned LSE matches a
     dense logsumexp over the flex mask_mod's keys, and the slot mass the KL uses matches the
     dense softmax's mass on those slots;
  3. the KL is finite, and its gradient reaches the nomination head (and only adds to it)."""
import sys, random, logging, torch
import numpy as np
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
import pinball.model.layers.hierarchical_message_passing as H

CFG = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
B = int(sys.argv[3]) if len(sys.argv) > 3 else 2
FAILS = []
def rep(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    if not ok: FAILS.append(n)

def seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s); random.seed(s); np.random.seed(s)

def mk(kl):
    cfg = PinballConfig.from_yaml(CFG)
    cfg.block_size = L
    cfg.local_pack_global_nom_kl = bool(kl)
    cfg.hier_layer_compile = False
    cfg.hier_refresh_compile = False
    cfg.hier_layer_cudagraphs = False
    cfg.local_pack_global_nom_inline = False
    inp = resolve_model_inputs(cfg, block_size=L)
    seed(0)
    m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                    tie_weights=inp.tie_weights, max_seq_len=L).cuda().train()
    return m, inp

# record the flex mask_mod (kv length -> mask_mod) for the dense reference
MASKS = {}
_orig_sweep = H._block_mask_sweep
def _sweep(create_block_mask, mask_mod, q_len, kv_len, device, kw):
    MASKS[int(kv_len)] = (mask_mod, int(q_len))
    return _orig_sweep(create_block_mask, mask_mod, q_len, kv_len, device, kw)
H._block_mask_sweep = _sweep

REF = []
_orig_kl = H.HierarchicalMessagePassing._nom_kl_from_flex
def _kl(self, q_s, k_s, lse, gsel):
    dev = q_s.device
    with torch.random.fork_rng(devices=[dev]):
        _orig_kl(self, q_s, k_s, lse, gsel)
        if CHECK["on"] and len(REF) < 3:
            # dense reference for 64 query rows of a random live chunk
            meta = gsel[2]
            G, C, nch, S = int(meta["G"]), int(meta["chunk"]), int(meta["nch"]), int(meta["n_static"])
            ok = meta["ok"].view(nch, G)
            live_ch = torch.nonzero(ok.any(1)).view(-1)
            c = int(live_ch[len(live_ch) // 2])
            N, KV, D = int(q_s.size(2)), int(k_s.size(2)), int(q_s.size(3))
            qi = torch.arange(c * C, min(N, c * C + 64), device=dev)
            mm = MASKS[KV][0]
            ki = torch.arange(KV, device=dev)
            z = torch.zeros((), dtype=torch.long, device=dev)
            msk = mm(z, z, qi.view(-1, 1), ki.view(1, -1))                      # [Q, KV]
            s = torch.einsum("bhqd,bhkd->bhqk", q_s[:, :, qi].float(), k_s.float()) * D ** -0.5
            s = s.masked_fill(~msk.view(1, 1, *msk.shape), float("-inf"))
            lse_ref = torch.logsumexp(s, -1)
            dl = (lse_ref - lse[:, :, qi].float()).abs().max().item()
            p_ref = torch.softmax(s, -1)[..., S + c * G:S + (c + 1) * G]          # [B,H,Q,G]
            p_lse = ((torch.einsum("bhqd,bhgd->bhqg", q_s[:, :, qi].float(),
                                   k_s[:, :, S + c * G:S + (c + 1) * G].float()) * D ** -0.5
                      - lse[:, :, qi].float().unsqueeze(-1)).exp()
                     * ok[c].float().view(1, 1, 1, -1))
            dp = (p_ref - p_lse).abs().max().item()
            REF.append((dl, dp, float(p_ref.sum(-1).mean()), float(lse_ref.abs().max())))
CHECK = {"on": False}
H.HierarchicalMessagePassing._nom_kl_from_flex = _kl

m0, inp = mk(False)
m1, _ = mk(True)
mi, un = m1.load_state_dict(m0.state_dict(), strict=False)
assert not mi and not un, (mi, un)
g = torch.Generator().manual_seed(1)
x = torch.randint(0, inp.vocab_size, (B, L), generator=g).cuda()

def fwd(m, s=7, extra=True):
    seed(s)
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        o = m(x); o = o[0] if isinstance(o, tuple) else o
        o = o.float()
        loss = torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)), x[:, 1:].reshape(-1))
    kl = getattr(m, "_last_nom_kl_loss", None)
    if extra and kl is not None:
        loss = loss + m.lambda_nom_kl * kl
    loss.backward()
    return o.detach(), kl

fwd(m0); fwd(m1)                       # warm-up (first call builds caches, not reproducible)
print("1. forward untouched")
o0, _ = fwd(m0); o1, kl = fwd(m1)
rep("train logits bit-identical (KL on vs off)", (o0 - o1).abs().max().item() == 0.0,
    f"max|diff| {(o0 - o1).abs().max().item():.3e}")

print("2. KL target = the flex softmax")
CHECK["on"] = True
fwd(m1)
CHECK["on"] = False
dl = max(r[0] for r in REF); dp = max(r[1] for r in REF)
rep("flex LSE == dense logsumexp over mask_mod keys", dl < 5e-2,
    f"max|dLSE| {dl:.2e} (bf16 q/k; |lse| up to {max(r[3] for r in REF):.1f})")
rep("slot mass from LSE == dense softmax mass", dp < 5e-3,
    f"max|dp| {dp:.2e}; dense slot mass per query {np.mean([r[2] for r in REF]):.3f}")

print("3. KL value and gradient path")
_, kl = fwd(m1)
rep("KL finite and positive", kl is not None and bool(torch.isfinite(kl)) and float(kl) > 0,
    f"KL {float(kl):.4f} over {int(m1._nom_kl_acc[2])} layers, slot mass/query "
    f"{float(m1._nom_kl_acc[1][1]) / int(m1._nom_kl_acc[2]):.3f}")
fwd(m1, extra=False); gA = {n: p.grad.clone() for n, p in m1.named_parameters() if p.grad is not None}
fwd(m1, extra=False); gA2 = {n: p.grad.clone() for n, p in m1.named_parameters() if p.grad is not None}
fwd(m1); gB = {n: p.grad.clone() for n, p in m1.named_parameters() if p.grad is not None}
# the backward has atomics: per-tensor run-to-run floor from the same model twice
fl = {n: (gA[n] - gA2[n]).abs().max().item() for n in gA}
dk = {n: (gB[n] - gA[n]).abs().max().item() for n in gA}
nom = sorted(n for n in gA if "nom_" in n)
mv = [n for n in nom if dk[n] > 4 * fl[n] + 1e-12]
rep("KL gradient reaches the nomination head", len(mv) > 0,
    f"{len(mv)} of {len(nom)} nom_* tensors moved beyond the floor, e.g. "
    + ", ".join(f"{n.split('.', 2)[-1]} {dk[n]:.1e} vs floor {fl[n]:.1e}" for n in mv[:3]))
print("PASS" if not FAILS else f"FAIL: {FAILS}")
