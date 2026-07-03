# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate CausalDynamicTokenUNet: (1) shape (T -> T/scale -> T), (2) channel growth,
(3) CAUSALITY (append future -> settled past decode output unchanged), for cnn + hyena.
Run on CPU (deterministic): python bench_dyn_unet.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from pinball.model.hierarchical_flow_gat_cached_batch import CausalDynamicTokenUNet, CausalTokenUNet


@torch.no_grad()
def run_unet(u, x):
    tok, ctx = u.encode(x)
    return tok, u.decode(tok, ctx, mode="strict")


def check(block, H=64, T=256, scale=8, growth=64):
    u = CausalDynamicTokenUNet(hidden_dim=H, scale=scale, block=block, channel_growth=growth,
                               concat_skips=True).eval()
    torch.manual_seed(0)
    x = torch.randn(2, T, H)
    tok, out = run_unet(u, x)
    print(f"\n[{block}] widths={u.widths}  input=[2,{T},{H}]  tokens={tuple(tok.shape)}  out={tuple(out.shape)}")
    assert tuple(tok.shape) == (2, T // scale, H), "token shape (T/scale, H) wrong"
    assert tuple(out.shape) == (2, T, H), "decode shape wrong"
    params = sum(p.numel() for p in u.parameters())
    const = CausalTokenUNet(hidden_dim=H, scale=scale, block=block)
    print(f"       params dynamic={params/1e3:.0f}k  vs constant-width={sum(p.numel() for p in const.parameters())/1e3:.0f}k")

    # Training causality (FIXED length): perturb the FUTURE half; the past output must be
    # unchanged (no future leak). (Append-length changes the Hyena's length-normalized filter,
    # so the append test only applies to length-stable ops -- that's a separate generation
    # concern, checked below.)
    p = T // 2
    x2 = x.clone(); x2[:, p:] = torch.randn(2, T - p, H)
    _, out2 = run_unet(u, x2)
    d = (out[:, :p] - out2[:, :p]).abs().max().item()
    print(f"       CAUSALITY (fixed-len, perturb future>={p}): past drift = {d:.3e}  {'PASS' if d < 1e-4 else 'FAIL'}")
    # Length-stability (generation): append future, settled past unchanged. CNN yes; Hyena no
    # (length-normalized positional filter) -> Hyena is train-causal but not KV-cache-stable.
    x3 = torch.cat([x, torch.randn(2, 32, H)], dim=1)
    _, out3 = run_unet(u, x3)
    dl = (out[:, : T - scale * 2] - out3[:, : T - scale * 2]).abs().max().item()
    print(f"       LENGTH-STABLE (append): settled drift = {dl:.3e}  {'stable' if dl < 1e-4 else 'length-dependent'}")
    # fwd+bwd
    u.train()
    xo = torch.randn(2, T, H, requires_grad=True)
    t, o = u.encode(xo)[0], None
    to, c = u.encode(xo)
    o = u.decode(to, c, mode="strict")
    o.pow(2).mean().backward()
    print(f"       fwd+bwd OK, grad on input = {xo.grad is not None}")


if __name__ == "__main__":
    for b in ("cnn", "hyena"):
        check(b)
