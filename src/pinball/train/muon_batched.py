"""Batched Newton-Schulz for pytorch_optimizer's Muon (config: ``muon_batched: true``).

pytorch_optimizer.Muon orthogonalizes every weight matrix in its own Python-loop iteration:
five Newton-Schulz rounds of three matmuls each, per matrix. Its own
``zero_power_via_newton_schulz_5`` already supports a batch dimension (baddbmm, with the
Frobenius normalization taken per matrix over the last two dims), so stacking every
same-shape update and orthogonalizing the stack once is the same math per matrix. Everything
else -- maximize, decoupled weight decay, momentum, Nesterov, the shape-adjusted LR, the state
layout ('momentum_buffer') -- follows Muon.step in the same order, so checkpoints move freely
between the two classes. The AdamW groups (use_muon False) run the parent's step untouched.

Not bit-identical: batched and per-matrix GEMMs accumulate in a different order. Measured
on the text glob400 model: max |param diff| 5.8e-05 after 6 steps (about 1% of one update,
bf16 Newton-Schulz). Per-step cost on the 4090 at text b8: 43.2 -> 33.7 ms (glob400, 160
matrices), 132.5 -> 103.8 ms (levelsep, 412 matrices). The remainder is Newton-Schulz FLOPs.
"""
import math
from typing import Dict, List, Tuple

import torch
import pytorch_optimizer as po
from pytorch_optimizer.optimizer.muon import get_adjusted_lr, zero_power_via_newton_schulz_5

# Elements per stacked Newton-Schulz call: bounds the transient fp32 stack (+ its bf16 copies)
# to ~256 MB whatever the matrix shape.
_MAX_STACK_ELEMS = 64 * 1024 * 1024


class BatchedMuon(po.Muon):
    """pytorch_optimizer.Muon with Newton-Schulz run once per (shape, dtype) bucket."""

    # Multi-tensor (torch._foreach_*) updates instead of one Python iteration per tensor.
    # Same element-wise ops in the same order, so the result is unchanged (see
    # scripts/verify_muon_foreach.py); False restores the per-tensor loops for that check.
    # The per-tensor loops were launch-bound: ~2.4k AdamW + ~0.7k Muon kernel launches per
    # step on the 496-tensor text pinball (~20-25 ms of GPU idle per step at 4k x4).
    _FOREACH = True

    @torch.no_grad()
    def step(self, closure=None):
        muon_groups = [g for g in self.param_groups if g.get("use_muon")]
        all_groups = self.param_groups
        if self._FOREACH:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in all_groups:
                if not group.get("use_muon"):
                    self._adamw_step_foreach(group)
        else:
            self.param_groups = [g for g in all_groups if not g.get("use_muon")]
            try:
                loss = super().step(closure)
            finally:
                self.param_groups = all_groups
        for group in muon_groups:
            if group.get("cautious"):
                raise NotImplementedError("BatchedMuon: cautious updates are not supported")
            self.init_group(group)
            group["step"] += 1
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            if self._FOREACH:
                self._weight_decay_foreach(ps, [p.grad for p in ps], group)
            else:
                for p in ps:
                    self.maximize_gradient(p.grad, maximize=self.maximize)
                    self.apply_weight_decay(
                        p, grad=p.grad, lr=group["lr"], weight_decay=group["weight_decay"],
                        weight_decouple=group["weight_decouple"], fixed_decay=False,
                    )
            bufs = [self.state[p]["momentum_buffer"] for p in ps]
            grads = [p.grad for p in ps]
            torch._foreach_lerp_(bufs, grads, 1.0 - group["momentum"])
            if group["nesterov"]:
                torch._foreach_lerp_(grads, bufs, group["momentum"])
                updates = grads
            else:
                updates = bufs
            buckets: Dict[Tuple, List[int]] = {}
            for i, p in enumerate(ps):
                buckets.setdefault((tuple(p.shape), p.dtype), []).append(i)
            for (shape, _), idx in buckets.items():
                per = max(1, _MAX_STACK_ELEMS // max(1, ps[idx[0]].numel()))
                lr = get_adjusted_lr(group["lr"], ps[idx[0]].size(),
                                     use_adjusted_lr=group["use_adjusted_lr"])
                for c in range(0, len(idx), per):
                    chunk = idx[c:c + per]
                    stack = torch.stack([updates[i].reshape(shape[0], -1) for i in chunk])
                    ortho = zero_power_via_newton_schulz_5(
                        stack, num_steps=group["ns_steps"], weights=group["ns_coeffs"])
                    if self._FOREACH:
                        torch._foreach_add_([ps[i] for i in chunk],
                                            [ortho[j].reshape(ps[i].shape) for j, i in enumerate(chunk)],
                                            alpha=-lr)
                    else:
                        for j, i in enumerate(chunk):
                            ps[i].add_(ortho[j].reshape(ps[i].shape), alpha=-lr)
        return loss

    def _weight_decay_foreach(self, ps: List[torch.Tensor], grads: List[torch.Tensor], group) -> None:
        """maximize_gradient + apply_weight_decay(fixed_decay=False, ratio=None) per tensor."""
        if self.maximize:
            torch._foreach_neg_(grads)
        wd = group["weight_decay"]
        if group["weight_decouple"]:
            # the exact scalar apply_weight_decay builds (ratio None -> * 1.0)
            torch._foreach_mul_(ps, 1.0 - wd * group["lr"] * 1.0)
        elif wd > 0.0:
            torch._foreach_add_(grads, ps, alpha=wd)

    def _adamw_step_foreach(self, group) -> None:
        """pytorch_optimizer.Muon.step's use_muon=False branch over the whole group at once."""
        self.init_group(group)
        group["step"] += 1
        ps = [p for p in group["params"] if p.grad is not None]
        if not ps:
            return
        grads = [p.grad for p in ps]
        self._weight_decay_foreach(ps, grads, group)
        exp_avg = [self.state[p]["exp_avg"] for p in ps]
        exp_avg_sq = [self.state[p]["exp_avg_sq"] for p in ps]
        beta1, beta2 = group["betas"]
        bias_correction1 = self.debias(beta1, group["step"])
        bias_correction2_sq = math.sqrt(self.debias(beta2, group["step"]))
        torch._foreach_lerp_(exp_avg, grads, 1.0 - beta1)
        # _foreach_mul(g, g), not _foreach_pow(g, 2): pow rounds differently from square()
        torch._foreach_lerp_(exp_avg_sq, torch._foreach_mul(grads, grads), 1.0 - beta2)
        de_nom = torch._foreach_sqrt(exp_avg_sq)
        torch._foreach_add_(de_nom, group["eps"])
        torch._foreach_div_(de_nom, bias_correction2_sq)
        torch._foreach_addcdiv_(ps, torch._foreach_div(exp_avg, bias_correction1), de_nom,
                                value=-group["lr"])
