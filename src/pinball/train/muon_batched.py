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
from typing import Dict, List, Tuple

import torch
import pytorch_optimizer as po
from pytorch_optimizer.optimizer.muon import get_adjusted_lr, zero_power_via_newton_schulz_5

# Elements per stacked Newton-Schulz call: bounds the transient fp32 stack (+ its bf16 copies)
# to ~256 MB whatever the matrix shape.
_MAX_STACK_ELEMS = 64 * 1024 * 1024


class BatchedMuon(po.Muon):
    """pytorch_optimizer.Muon with Newton-Schulz run once per (shape, dtype) bucket."""

    @torch.no_grad()
    def step(self, closure=None):
        muon_groups = [g for g in self.param_groups if g.get("use_muon")]
        all_groups = self.param_groups
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
                    for j, i in enumerate(chunk):
                        ps[i].add_(ortho[j].reshape(ps[i].shape), alpha=-lr)
        return loss
