# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Hierarchy-level connectivity reporter.

Inspects a built unified graph and reports, per hierarchy level: node count, average
out/in degree, how those edges split into down/up/lateral, and **L0 coverage** — the
fraction of L0 (token) nodes that the level can reach via directed edges, i.e. the breadth
of the prediction it can influence. A coarse level with L0 coverage 0 is **inert** (cannot
affect the readout at all); very low coverage (e.g. only the final token) is flagged too.
Example: the top level under strict causal AR is inert — its summary spans the final token,
so it has no legal outgoing edges to lower levels.

Message-passing convention: an edge ``(src, dst)`` means ``dst`` aggregates from ``src``,
so information flows ``src -> dst``. A level influences an L0 node iff a directed path leads
from one of its nodes to that L0 node.

NOTE: this is a *structural* report. A level can be reachable yet contribute little
numerically; for the actual effect on outputs use the ``ablate_levels`` dial.
"""
from typing import Dict, Any, Optional, List
import torch


def _forward_reach_to_l0(seed: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                         is_l0: torch.Tensor, max_iters: int) -> torch.Tensor:
    """Boolean mask of nodes reachable from ``seed`` following src->dst edges."""
    reach = seed.clone()
    N = int(reach.numel())
    for _ in range(max_iters):
        contrib = reach[src].to(torch.float32)  # if src reachable, dst becomes reachable
        agg = torch.zeros(N, device=reach.device, dtype=torch.float32).scatter_add_(0, dst, contrib)
        new = reach | (agg > 0)
        if bool((new == reach).all()):
            break
        reach = new
    return reach & is_l0


def compute_level_connectivity(
    edge_index: torch.Tensor,
    node_level: torch.Tensor,
    num_levels: Optional[int] = None,
    near_inert_coverage: float = 0.02,
) -> Dict[str, Any]:
    """Return per-level connectivity stats, L0 coverage, and inert-level detection."""
    device = node_level.device
    nl = node_level.to(torch.long)
    if num_levels is None:
        num_levels = int(nl.max().item()) + 1 if nl.numel() > 0 else 0

    E = int(edge_index.size(1)) if edge_index is not None and edge_index.numel() > 0 else 0
    if E > 0:
        src, dst = edge_index[0].to(torch.long), edge_index[1].to(torch.long)
        src_lvl, dst_lvl = nl[src], nl[dst]
    else:
        src = dst = src_lvl = dst_lvl = torch.empty(0, dtype=torch.long, device=device)

    is_l0 = nl == 0
    n_l0 = int(is_l0.sum().item())
    max_iters = num_levels + 2

    levels: List[Dict[str, Any]] = []
    inert: List[int] = []
    near_inert: List[int] = []
    for L in range(num_levels):
        at = nl == L
        n = int(at.sum().item())
        if n == 0:
            levels.append(dict(level=L, n_nodes=0, avg_out=0.0, avg_in=0.0,
                               down=0, up=0, lateral=0, l0_coverage=0.0, inert=False, near_inert=False))
            continue
        out_mask = src_lvl == L if E > 0 else torch.empty(0, dtype=torch.bool, device=device)
        in_mask = dst_lvl == L if E > 0 else torch.empty(0, dtype=torch.bool, device=device)
        out_e = int(out_mask.sum().item()) if E > 0 else 0
        in_e = int(in_mask.sum().item()) if E > 0 else 0
        down = up = lateral = 0
        if E > 0 and out_e > 0:
            d = dst_lvl[out_mask]
            down = int((d < L).sum().item())
            up = int((d > L).sum().item())
            lateral = int((d == L).sum().item())

        # L0 coverage: fraction of token nodes reachable from this level (L0 covers itself).
        if L == 0:
            coverage = 1.0
        elif E > 0 and n_l0 > 0:
            reached_l0 = _forward_reach_to_l0(at, src, dst, is_l0, max_iters)
            coverage = int(reached_l0.sum().item()) / n_l0
        else:
            coverage = 0.0

        is_inert = (L != 0) and coverage == 0.0
        is_near = (L != 0) and (0.0 < coverage <= near_inert_coverage)
        if is_inert:
            inert.append(L)
        if is_near:
            near_inert.append(L)
        levels.append(dict(
            level=L, n_nodes=n,
            avg_out=out_e / n, avg_in=in_e / n,
            down=down, up=up, lateral=lateral,
            l0_coverage=coverage, inert=is_inert, near_inert=is_near,
        ))

    return {"num_levels": num_levels, "num_edges": E, "levels": levels,
            "inert_levels": inert, "near_inert_levels": near_inert}


def format_level_connectivity(report: Dict[str, Any]) -> str:
    """Render the report dict as a compact text table."""
    lines = [
        f"Level connectivity ({report['num_edges']} edges, {report['num_levels']} levels):",
        "  level  nodes  avg_out  avg_in   down    up  lateral  L0_cov  status",
    ]
    for lv in report["levels"]:
        if lv["inert"]:
            status = "INERT"
        elif lv.get("near_inert"):
            status = "near-inert"
        else:
            status = "token" if lv["level"] == 0 else "ok"
        lines.append(
            f"  L{lv['level']:<4d} {lv['n_nodes']:>6d} {lv['avg_out']:>8.2f} {lv['avg_in']:>7.2f}"
            f" {lv['down']:>6d} {lv['up']:>5d} {lv['lateral']:>8d} {lv['l0_coverage']:>6.2f}  {status}"
        )
    if report["inert_levels"]:
        lines.append(f"  -> INERT levels (reach 0% of L0 tokens, cannot influence the prediction): {report['inert_levels']}")
    if report.get("near_inert_levels"):
        lines.append(f"  -> near-inert levels (reach <=2% of L0 tokens): {report['near_inert_levels']}")
    if not report["inert_levels"] and not report.get("near_inert_levels"):
        lines.append("  -> no inert levels; every level reaches a meaningful share of L0 tokens.")
    return "\n".join(lines)
