import math, logging, torch
logging.disable(logging.WARNING)
from pinball.config import PinballConfig
from pinball.model import build_model
from pinball.model_inputs import resolve_model_inputs
cfg = PinballConfig.from_yaml("configs/pinball_pg19_16k_l0coarse_nope_noslots.yaml")
L = 4096; cfg.block_size = L; cfg.hier_layer_compile = False
inp = resolve_model_inputs(cfg, block_size=L); torch.manual_seed(0)
m = build_model(cfg, tokenizer=inp.tokenizer, vocab_size=inp.vocab_size, input_mode=inp.input_mode,
                tie_weights=inp.tie_weights, max_seq_len=L).cuda().eval()
g = torch.Generator().manual_seed(5)
with torch.no_grad():
    for n_, p in m.named_parameters():
        if "far_level_bias" in n_ or "level_k_emb" in n_:
            p.copy_((torch.randn(p.shape, generator=g) * 0.5).to(p))
caps = {}
for li, t in enumerate(m.refinement_transformers):
    mp = t.message_passing
    def mk(mp, li):
        o_ = mp._flex_union_attn
        def spy(qp, kp, vp, spec, causal=True, gsel=None):
            out = o_(qp, kp, vp, spec, causal, gsel=gsel)
            caps[li] = (qp.detach(), kp.detach(), vp.detach(), spec, out.detach(), mp, gsel)
            return out
        return spy
    mp._flex_union_attn = mk(mp, li)
x = torch.randint(0, inp.vocab_size, (2, L), generator=torch.Generator().manual_seed(3)).cuda()
with torch.no_grad():
    y = m(x)
y = (y[0] if isinstance(y, tuple) else y).float()
print("flex failed:", m.flex_union_status()["failed_modules"])
ok_all = True
for li in (0, 6, 11):
    qp, kp, vp, spec, out, mp, gsel = caps[li]
    assert gsel is None and spec.get("flex_kv_prefix") is not None and spec.get("flex_ring_wtok") is None
    pre = spec["flex_kv_prefix"]; W = int(spec["window"]); lvl = spec["levels"]; D = qp.size(-1); ND = mp.local_pack_far_nope_dims
    def far(k, lv):
        k = torch.cat([torch.zeros_like(k[..., :D - ND]), k[..., D - ND:]], -1)
        b = mp.far_level_bias.float()[:, lv].t()
        return torch.cat([k[..., :-1], (b * math.sqrt(D)).unsqueeze(-1)], -1)
    l0 = torch.nonzero(lvl == 0).view(-1)
    qs = l0[torch.randperm(l0.numel(), generator=torch.Generator().manual_seed(li))[:40]].tolist() + [int(l0[-1])]
    err, nb = 0.0, 0
    for b in range(2):
        for qi in qs:
            fr = pre[(pre <= qi) & ((qi - pre) > W)]
            nb += int(((pre <= qi) & ((qi - pre) <= W)).sum())
            band = torch.arange(max(0, qi - W), qi + 1, device=qp.device)
            kk = torch.cat([far(kp[b, fr], lvl[fr]), kp[b, band]]); vv = torch.cat([vp[b, fr], vp[b, band]])
            o = torch.einsum("hk,khd->hd", (torch.einsum("hd,khd->hk", qp[b, qi], kk) / math.sqrt(D)).softmax(-1), vv)
            err = max(err, (o - out[b, qi]).abs().max().item())
    ok_all &= err < 2e-3
    print(f"L{li}: static {pre.numel()} rows; flex out == dense reference (L0 queries): max|diff| {err:.2e}; band-near static rows via band {nb}")
x2 = x.clone(); x2[:, 3000:] = torch.randint(0, inp.vocab_size, (2, L - 3000), generator=torch.Generator().manual_seed(4)).cuda()
with torch.no_grad():
    y2 = m(x2)
y2 = (y2[0] if isinstance(y2, tuple) else y2).float()
print("causal p=3000:", (y[:, :3000] - y2[:, :3000]).abs().max().item(), "| ALL OK" if ok_all else "| FAIL")
