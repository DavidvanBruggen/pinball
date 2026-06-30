# SPDX-License-Identifier: GPL-3.0-or-later
"""Tier-3 packed vs scatter cross-level attention: micro-bench the crossover degree.

Directly times one HierarchicalMessagePassing cross-level aggregation (packed path vs the
scatter default) on synthetic hierarchy edge sets of increasing per-destination degree, to
locate where the dense/batched packed path overtakes memory-bound scatter. Also reports the
live-config degree (~compression_ratio) for context.

Run on the free 4090 (torch ordering):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python bench_tier3_crossover.py
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import math
import torch
from torch_geometric.utils import softmax
try:
    from torch_scatter import scatter_add
except Exception:
    from torch_geometric.utils import scatter as _pyg_scatter
    def scatter_add(src, index, dim=0, dim_size=None):
        return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
from pinball.model.layers.hierarchical_message_passing import HierarchicalMessagePassing

DEV = "cuda:0"
H, D = 12, 64
HID = H * D
B = 8
NUM_LEVELS = 4


def make_mp(packed):
    torch.manual_seed(0)
    mp = HierarchicalMessagePassing(
        hidden_dim=HID, num_heads=H,
        cross_level_packed=packed,
    ).to(DEV).eval()
    return mp


def synth_edges(num_dst, degree, device):
    """Two-level graph: `num_dst` destination (parent) nodes each with `degree` child srcs.
    Children are level-0, parents level-1. Returns src,dst,node_level,num_nodes."""
    num_src = num_dst * degree
    num_nodes = num_src + num_dst
    node_level = torch.cat([
        torch.zeros(num_src, dtype=torch.long),
        torch.ones(num_dst, dtype=torch.long),
    ]).to(device)
    src = torch.arange(num_src, device=device)
    dst = (num_src + torch.arange(num_dst, device=device)).repeat_interleave(degree)
    return src, dst, node_level, num_nodes


@torch.no_grad()
def time_path(mp, x, src, dst, node_level, num_nodes, iters=30):
    q = mp.q_proj(x).view(B, num_nodes, H, D)
    k = mp.k_proj(x).view(B, num_nodes, H, D)
    v = mp.v_proj(x).view(B, num_nodes, H, D)
    use_packed = bool(getattr(mp, "cross_level_packed", False))

    index_flat = mp._batched_dst_index_flat(dst, B, num_nodes)
    num_edges = int(src.numel())

    def scatter_call():
        # Faithful copy of the inline cross-level scatter aggregation in _forward_batched.
        q_i = q[:, dst]; k_j = k[:, src]; v_j = v[:, src]
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(D)
        level_bias = mp._edge_level_attention_bias(node_level, src, dst, q.device)
        attn = attn + level_bias.unsqueeze(0)
        attn_flat = softmax(attn.reshape(B * num_edges, H), index_flat)
        attn = attn_flat.view(B, num_edges, H)
        messages = (v_j * attn.unsqueeze(-1)).reshape(B * num_edges, H, D)
        out_flat = scatter_add(messages, index_flat, dim=0, dim_size=B * num_nodes)
        return out_flat.view(B, num_nodes, H, D)

    def call():
        if use_packed:
            return mp._cross_level_packed_attention(q, k, v, src, dst, node_level, num_nodes, B, D, D)
        return scatter_call()

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    print(f"\n=== Tier-3 packed vs scatter cross-level crossover ({torch.cuda.get_device_name(0)}) ===")
    print(f"  B={B} H={H} D={D}; two-level synthetic graph, varying per-parent degree")
    print(f"  (live wikitext config cross-level degree ~= compression_ratio: 16/4/4)\n")
    mp_pk = make_mp(True)
    mp_sc = make_mp(False)
    num_dst = 256
    print(f"  {'degree':>7} {'edges':>9} {'scatter ms':>12} {'packed ms':>11} {'speedup':>9}")
    for degree in (4, 8, 16, 32, 64, 128):
        src, dst, node_level, num_nodes = synth_edges(num_dst, degree, DEV)
        torch.manual_seed(1)
        x = torch.randn(B, num_nodes, HID, device=DEV)
        try:
            t_sc = time_path(mp_sc, x, src, dst, node_level, num_nodes)
            t_pk = time_path(mp_pk, x, src, dst, node_level, num_nodes)
            sp = t_sc / t_pk if t_pk > 0 else float("nan")
            print(f"  {degree:>7} {int(src.numel()):>9} {t_sc:>12.3f} {t_pk:>11.3f} {sp:>8.2f}x")
        except Exception as e:
            print(f"  {degree:>7} FAILED {type(e).__name__}: {str(e)[:60]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
