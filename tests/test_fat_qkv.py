# SPDX-License-Identifier: GPL-3.0-or-later
"""Fat-QKV (`attn_sparse_qk_mult` / `attn_sparse_v_mult`) — decoupled wide Q/K and V
head dims on the SPARSE-SCATTER attention paths (cross-level message passing + HQD read).

Flash/SDPA local-window attention requires qk_head_dim == v_head_dim, so it is untouched
and keeps the shared head_dim projections; only the scatter paths consult the wide ones.

Probes:
  1. Off (mult == 1.0): bit-identical to the unflagged baseline — the wide projections
     are never created and every scatter path reuses q/k/v/out_proj.
  2. Wide (mult == 2.0): forward + backward run, and AR causality holds exactly — a
     perturbed *future* token leaves all *earlier* positions unchanged (CPU is
     bit-deterministic, so the bar is 0). Widening only changes the per-head dims, not
     the causal edge set, so past logits must be untouched.

Run:   python tests/test_fat_qkv.py
"""
import sys, pathlib
import torch

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

BASE_CFG = dict(
    model_type="pinball", modality="text", tokenizer_name="gpt2",
    block_size=256, hidden_dim=64, num_heads=4, num_refinement_layers=4,
    num_layers=[0, 0, 0, 0], internal_cycles=[0, 0, 0, 0],
    refinement_style="unified", unified_refinement_cycles=1,
    compression_ratios=[8, 4, 2], overlap_ratios=[0.1, 0.2, 0.4],
    local_attn_windows=[8, 8, 8, 8], local_attn_levels=[0, 1, 2, 3],
    local_attn_causal_levels=[0, 1, 2, 3],
    dropout=0.0, norm_type="layernorm", lap_pe_k=0, l0_cycles=0,
    iterative_refinement_cycles=0, local_connectivity_window_size=0,
    attn_backend="sdpa", l0_local_window=8,
    train_mode="ar", ar_graph_causal=True,
    use_hqd=True, hqd_every_n=1, hqd_query_chunk_size=4096,
    hqd_include_local_window=False, hqd_reuse_previous=False,
    hqd_topk_l3=4, hqd_topk_l2=4, hqd_topk_l1=4, hqd_topk_l0=16,
    hqd_l0_topk_enable=True, hqd_global_topk=0,
    hqd_use_existing_zipper_projections=True,
    hqd_query_level=0, hqd_stop_level=0,
    hqd_read_levels=[0, 1, 2, 3],
    use_witness_packets=True, use_summary_witnesses=True, use_rare_witnesses=True,
    witness_k_summary=4, witness_k_rare=4, witness_score_bias=5.0,
    hqd_window_bag_levels=[1, 2, 3], hqd_window_bag_routing="per_position",
    hqd_window_bag_topk=0,
    use_aux_loss=False,
)


def _build(overrides):
    torch.manual_seed(0)
    cfg = PinballConfig(**{**BASE_CFG, **overrides})
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok),
                        input_mode="tokens", tie_weights=True,
                        max_seq_len=cfg.block_size).to("cpu")
    model.emit_features_only = True
    model.eval()
    return model, tok


def _feats(model, x):
    with torch.no_grad():
        out = model(x)
    return (out[0] if isinstance(out, (tuple, list)) else out).float()


def _ids():
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    T = BASE_CFG["block_size"]
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size
    return ids, ids2, j


def test_fat_qkv_off_is_identical():
    # mult == 1.0 must not create wide projections nor alter any scatter path.
    ids, _, _ = _ids()
    base, _ = _build(dict())
    flag, _ = _build(dict(attn_sparse_qk_mult=1.0, attn_sparse_v_mult=1.0))
    diff = (_feats(base, ids) - _feats(flag, ids)).abs().max().item()
    print(f"[fat-qkv off] max|diff vs baseline|={diff:.3e}")
    assert diff == 0.0, f"mult=1.0 changed outputs ({diff:.3e}) — should be a pure no-op"


def test_fat_v_causality_and_backward():
    ids, ids2, j = _ids()
    model, _ = _build(dict(attn_sparse_v_mult=2.0))
    base = _feats(model, ids)
    det = (base - _feats(model, ids)).abs().max().item()
    past = (base - _feats(model, ids2)).abs().flatten(start_dim=2).amax(dim=-1)[0][:j].max().item()
    print(f"[fat-V x2] determinism={det:.1e}  max|delta| past={past:.3e}")
    assert det < 1e-6, "fat-V non-deterministic on CPU?!"
    assert past < 1e-6, f"fat-V leaked future info ({past:.3e})"
    # backward runs
    model.train()
    out = model(ids)
    feat = out[0] if isinstance(out, (tuple, list)) else out
    feat.float().pow(2).mean().backward()
    print("[fat-V x2] backward OK")


def test_fat_qk_causality_and_backward():
    ids, ids2, j = _ids()
    model, _ = _build(dict(attn_sparse_qk_mult=2.0))
    base = _feats(model, ids)
    det = (base - _feats(model, ids)).abs().max().item()
    past = (base - _feats(model, ids2)).abs().flatten(start_dim=2).amax(dim=-1)[0][:j].max().item()
    print(f"[fat-QK x2] determinism={det:.1e}  max|delta| past={past:.3e}")
    assert det < 1e-6, "fat-QK non-deterministic on CPU?!"
    assert past < 1e-6, f"fat-QK leaked future info ({past:.3e})"
    model.train()
    out = model(ids)
    feat = out[0] if isinstance(out, (tuple, list)) else out
    feat.float().pow(2).mean().backward()
    print("[fat-QK x2] backward OK")


def test_fat_both_causality():
    ids, ids2, j = _ids()
    model, _ = _build(dict(attn_sparse_qk_mult=2.0, attn_sparse_v_mult=3.0))
    base = _feats(model, ids)
    det = (base - _feats(model, ids)).abs().max().item()
    past = (base - _feats(model, ids2)).abs().flatten(start_dim=2).amax(dim=-1)[0][:j].max().item()
    print(f"[fat-QK x2 + fat-V x3] determinism={det:.1e}  max|delta| past={past:.3e}")
    assert det < 1e-6 and past < 1e-6, f"fat-QKV leaked ({past:.3e})"


if __name__ == "__main__":
    test_fat_qkv_off_is_identical()
    test_fat_v_causality_and_backward()
    test_fat_qk_causality_and_backward()
    test_fat_both_causality()
    print("\nAll fat-QKV tests passed.")
