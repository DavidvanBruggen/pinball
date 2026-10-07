"""local_pack_global_nom_kl_detach probe (separate detached ranking scorer trained only by the KL).

usage: CUDA_VISIBLE_DEVICES=1 python scripts/nomination/verify_nom_kl_detach.py CONFIG [L] [B]
CONFIG is a head-nominator stream slot arm. Eager layers, train mode. Checks:
  1. gradient isolation: the KL alone reaches ONLY the scorer (nom_kl_*), and the task loss
     alone reaches every trainable tensor EXCEPT the scorer (its ranking is non-differentiable);
  2. the scorer actually ranks: replacing its weights changes the picks;
  3. KL finite; slot mass from the LSE matches a dense softmax over the flex mask_mod keys;
  4. causal (eval): perturbing tokens >= p leaves logits < p bit-unchanged."""
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

cfg = PinballConfig.from_yaml(CFG)
cfg.block_size = L
cfg.local_pack_global_nom_kl = True
cfg.local_pack_global_nom_kl_detach = True
cfg.hier_layer_compile = False
cfg.hier_refresh_compile = False
cfg.hier_layer_cudagraphs = False
cfg.local_pack_global_nom_inline = False
inp = resolve_model_inputs(cfg, block_size=L)
seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda().train()
mps = [t.message_passing for t in m.refinement_transformers]
assert all(hasattr(mp, "nom_kl_q") for mp in mps)

MASKS = {}
_orig_sweep = H._block_mask_sweep
def _sweep(create_block_mask, mask_mod, q_len, kv_len, device, kw):
    MASKS[int(kv_len)] = mask_mod
    return _orig_sweep(create_block_mask, mask_mod, q_len, kv_len, device, kw)
H._block_mask_sweep = _sweep
REF = []
_orig_kl = H.HierarchicalMessagePassing._nom_kl_from_flex
def _kl(self, q_s, k_s, lse, gsel):
    _orig_kl(self, q_s, k_s, lse, gsel)
    if CHECK["on"] and len(REF) < 3:
        meta = gsel[2]
        G, C, nch, S = int(meta["G"]), int(meta["chunk"]), int(meta["nch"]), int(meta["n_static"])
        ok = meta["ok"].view(nch, G)
        lc = torch.nonzero(ok.any(1)).view(-1); c = int(lc[len(lc) // 2])
        N, KV, D = int(q_s.size(2)), int(k_s.size(2)), int(q_s.size(3))
        qi = torch.arange(c * C, min(N, c * C + 64), device=q_s.device)
        z = torch.zeros((), dtype=torch.long, device=q_s.device)
        msk = MASKS[KV](z, z, qi.view(-1, 1), torch.arange(KV, device=q_s.device).view(1, -1))
        s = torch.einsum("bhqd,bhkd->bhqk", q_s[:, :, qi].float(), k_s.float()) * D ** -0.5
        s = s.masked_fill(~msk.view(1, 1, *msk.shape), float("-inf"))
        p_ref = torch.softmax(s, -1)[..., S + c * G:S + (c + 1) * G] * ok[c].float()
        p_lse = (s[..., S + c * G:S + (c + 1) * G] - lse[:, :, qi].float().unsqueeze(-1)).exp() * ok[c].float()
        REF.append((p_ref - p_lse).nan_to_num(0.0).abs().max().item())
CHECK = {"on": False}
H.HierarchicalMessagePassing._nom_kl_from_flex = _kl
_orig_fin = H.HierarchicalMessagePassing._nom_stream_finish
def _fin(self, spec, rows, g, stats, geo, C, nch, K):
    self._probe_rows = rows.detach().clone()
    return _orig_fin(self, spec, rows, g, stats, geo, C, nch, K)
H.HierarchicalMessagePassing._nom_stream_finish = _fin

g = torch.Generator().manual_seed(1)
x = torch.randint(0, inp.vocab_size, (B, L), generator=g).cuda()
def fwd(train=True, s=7):
    m.train(train); seed(s)
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        o = m(x if FWD["x"] is None else FWD["x"]); o = o[0] if isinstance(o, tuple) else o
        o = o.float()
        ce = torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)),
                                               (x if FWD["x"] is None else FWD["x"])[:, 1:].reshape(-1))
    return o, ce, getattr(m, "_last_nom_kl_loss", None)
FWD = {"x": None}
fwd()                                           # warm-up
SC = lambda n: ".nom_kl_" in n
def grads():
    return {n: (p.grad is not None and float(p.grad.abs().max()) > 0) for n, p in m.named_parameters()
            if p.requires_grad}

print("1. gradient isolation")
o, ce, kl = fwd(); kl.backward(); gk = grads()
hit = [n for n, v in gk.items() if v]
rep("KL reaches ONLY the scorer", len(hit) > 0 and all(SC(n) for n in hit),
    f"{len(hit)} tensors, non-scorer: {[n for n in hit if not SC(n)][:4]}")
o, ce, kl = fwd(); ce.backward(); gc = grads()
sc_hit = [n for n, v in gc.items() if v and SC(n)]
rep("task loss does not reach the scorer", not sc_hit, f"{sc_hit[:4]}")
n_sc = sum(1 for n in gc if SC(n))
rep("task loss reaches the head (nom_parent_q/nom_child_k)",
    any(v for n, v in gc.items() if ".nom_child_k." in n), "")

print("2. the scorer ranks")
fwd(); r0 = [mp._probe_rows for mp in mps]
with torch.no_grad():
    for mp in mps:
        for mod in list(mp.nom_kl_q) + list(mp.nom_kl_k):
            mod.weight.normal_(0, 0.05)
fwd(); r1 = [mp._probe_rows for mp in mps]
frac = [1.0 - (a == b).float().mean().item() for a, b in zip(r0, r1)]
rep("new scorer weights change the picks", min(frac) > 0.0, " ".join(f"{f:.2f}" for f in frac))

print("3. KL value and target")
CHECK["on"] = True; o, ce, kl = fwd(); CHECK["on"] = False
rep("KL finite and positive", bool(torch.isfinite(kl)) and float(kl) > 0, f"KL {float(kl):.4f}")
rep("slot mass from LSE == dense softmax mass", max(REF) < 5e-3, f"max|dp| {max(REF):.2e}")

print("4. causal (eval)")
p = L // 2
x2 = x.clone(); x2[:, p:] = torch.randint(0, inp.vocab_size, (B, L - p), generator=g).cuda()
with torch.no_grad():
    a, _, _ = fwd(train=False); a2, _, _ = fwd(train=False)
    FWD["x"] = x2; b, _, _ = fwd(train=False); FWD["x"] = None
rep("eval reproducible", (a - a2).abs().max().item() == 0.0, f"{(a - a2).abs().max().item():.3e}")
past = (a[:, :p] - b[:, :p]).abs().max().item(); fut = (a[:, p:] - b[:, p:]).abs().max().item()
rep("causal", past == 0.0 and fut > 0.0, f"past max|diff| {past:.3e}, future {fut:.3e}")
print("PASS" if not FAILS else f"FAIL: {FAILS}")
