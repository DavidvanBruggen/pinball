# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense HQD read (`hqd_attn_impl='dense'`) — causality + scatter-equivalence.

The dense read groups each destination's HQD edges into a padded block and runs a
fused SDPA instead of the edge gather + segment-softmax + scatter_add. Same edge
set => same candidate set per destination => same causality. This probe checks:

  1. Causality: perturbing a *future* token leaves all *earlier* positions exactly
     unchanged (CPU is bit-deterministic, so the bar is 0).
  2. Equivalence: dense vs scatter produce the same features for the same input
     (a softmax reformulation, so equal up to float summation order, not bitwise).

Exercises the full HQD edge set: L0 read + coarse read (read_levels) + witnesses +
per_position window bag, so every destination kind (L0 query and coarse node) is
hit by the grouping.

Run:   python tests/test_hqd_dense_read.py
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
    # full edge set: coarse read + witnesses + per_position bag
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


def test_dense_causality_and_equivalence():
    T = BASE_CFG["block_size"]
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size

    # --- dense path: causality ---
    model_d, _ = _build(dict(hqd_attn_impl="dense", hqd_dense_backend="sdpa"))
    base_d = _feats(model_d, ids)
    det = (base_d - _feats(model_d, ids)).abs().max().item()
    pert_d = _feats(model_d, ids2)
    past = (base_d - pert_d).abs().flatten(start_dim=2).amax(dim=-1)[0][:j]
    past_delta = past.max().item()
    print(f"[dense] determinism={det:.1e}  max|delta| past={past_delta:.3e}")
    assert det < 1e-6, "dense path non-deterministic on CPU?!"
    assert past_delta < 1e-6, f"dense read leaked future info ({past_delta:.3e} at pos {int(past.argmax())})"

    # --- equivalence: scatter vs dense ---
    # Run HQD exactly once (every_n=4 over 4 layers => layer 0 only) so there is no
    # downstream descent re-selection. With re-selection on every layer, a 1e-7
    # kernel-rounding difference flips a discrete top-k tie and the selections
    # diverge (the same discreteness that makes CUDA HQD non-deterministic) — that
    # is expected chaos, not a dense bug, so it must be isolated out to test the
    # kernel itself.
    md_once, _ = _build(dict(hqd_attn_impl="dense", hqd_every_n=4))
    ms_once, _ = _build(dict(hqd_attn_impl="scatter", hqd_every_n=4))
    bd = _feats(md_once, ids)
    bs = _feats(ms_once, ids)
    diff = (bd - bs).abs().max().item()
    rel = diff / (bs.abs().max().item() + 1e-9)
    print(f"[scatter vs dense, HQD once] max|diff|={diff:.3e}  rel={rel:.3e}")
    assert rel < 1e-5, f"dense read disagrees with scatter beyond fp tolerance (rel={rel:.3e})"


def test_descent_shortcut_causality():
    # hqd_descent_stop_at=2 skips the L1 scoring stage (structural L1 expansion).
    # It changes selection but must stay strictly causal at query_level 0.
    T = BASE_CFG["block_size"]
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size

    for impl in ("scatter", "dense"):
        model, _ = _build(dict(hqd_descent_stop_at=2, hqd_attn_impl=impl))
        base = _feats(model, ids)
        det = (base - _feats(model, ids)).abs().max().item()
        pert = _feats(model, ids2)
        past = (base - pert).abs().flatten(start_dim=2).amax(dim=-1)[0][:j]
        pd = past.max().item()
        print(f"[shortcut stop_at=2, {impl}] determinism={det:.1e}  max|delta| past={pd:.3e}")
        assert det < 1e-6, f"shortcut/{impl} non-deterministic on CPU?!"
        assert pd < 1e-6, f"shortcut/{impl} leaked future info ({pd:.3e} at pos {int(past.argmax())})"


def test_cross_level_witness_causality():
    # witness_levels=[1,2,3] builds cross-level L2->L0 / L3->L0 bags so a coarse
    # selection reaches its important L0 descendants. Must stay strictly causal (AR).
    T = BASE_CFG["block_size"]
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size
    for wl in ([1], [1, 2, 3]):
        model, _ = _build(dict(witness_levels=wl))
        base = _feats(model, ids)
        det = (base - _feats(model, ids)).abs().max().item()
        past = (base - _feats(model, ids2)).abs().flatten(start_dim=2).amax(dim=-1)[0][:j].max().item()
        print(f"[witness_levels={wl}] determinism={det:.1e}  max|delta| past={past:.3e}")
        assert det < 1e-6 and past < 1e-6, f"cross-level bag (levels={wl}) leaked ({past:.3e})"


def test_shallow_read_causality():
    # hqd_shallow_read_level=3 ("L0 queries L3, bags down"): stop scoring at L3, reach L0
    # only via the selected L3 nodes' cross-level bags. Skips L2/L1/L0 scoring. AR-causal.
    T = BASE_CFG["block_size"]
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size
    for lvl in (3, 2):
        model, _ = _build(dict(hqd_shallow_read_level=lvl, witness_levels=[1, 2, 3]))
        base = _feats(model, ids)
        det = (base - _feats(model, ids)).abs().max().item()
        past = (base - _feats(model, ids2)).abs().flatten(start_dim=2).amax(dim=-1)[0][:j].max().item()
        edges = getattr(model, "_last_hqd_added_total", 0)
        print(f"[shallow_read={lvl}] determinism={det:.1e}  max|delta| past={past:.3e}  edges={edges}")
        assert det < 1e-6 and past < 1e-6, f"shallow read (level={lvl}) leaked ({past:.3e})"
        assert edges and edges > 0, f"shallow read (level={lvl}) produced no edges — bag missing?"


def test_packed_graph_witness_read_causality():
    # Experimental efficient path: the current graph attention pass captures direct
    # child witnesses, HQD scores only L3, then composes L3->L2->L1->L0 packed witness
    # ids consumed inside message passing without flattening an L0 sparse edge list.
    T = BASE_CFG["block_size"]
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size

    model, _ = _build(dict(
        hqd_shallow_read_level=3,
        hqd_select_inside_message_passing=True,
        hqd_graph_witness_enable=True,
        hqd_graph_witness_topk=4,
        hqd_packed_witness_read=True,
        hqd_topk_l0=8,
    ))
    base = _feats(model, ids)
    det = (base - _feats(model, ids)).abs().max().item()
    pert = _feats(model, ids2)
    past = (base - pert).abs().flatten(start_dim=2).amax(dim=-1)[0][:j]
    pd = past.max().item()
    stats = getattr(model, "_last_hqd_stage_stats", {}) or {}
    packed = int(stats.get("packed_witness_l0", 0))
    print(f"[packed graph witnesses] determinism={det:.1e}  max|delta| past={pd:.3e}  packed={packed}")
    assert packed > 0, "packed graph-witness path produced no L0 witness reads"
    assert det < 1e-6 and pd < 1e-6, f"packed graph witnesses leaked ({pd:.3e})"


def test_packed_recomputed_witness_read_causality():
    # Use the original recomputed witness table as the source, but consume selected
    # rows through packed fixed-K L0 attention instead of materialized sparse edges.
    T = BASE_CFG["block_size"]
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size

    model, _ = _build(dict(
        hqd_shallow_read_level=3,
        hqd_select_inside_message_passing=True,
        hqd_graph_witness_enable=False,
        hqd_packed_witness_read=True,
        hqd_packed_witness_source="recompute",
        use_witness_packets=True,
        use_summary_witnesses=True,
        use_rare_witnesses=True,
        witness_levels=[3],
        hqd_topk_l0=8,
    ))
    base = _feats(model, ids)
    det = (base - _feats(model, ids)).abs().max().item()
    pert = _feats(model, ids2)
    past = (base - pert).abs().flatten(start_dim=2).amax(dim=-1)[0][:j]
    pd = past.max().item()
    stats = getattr(model, "_last_hqd_stage_stats", {}) or {}
    packed = int(stats.get("packed_witness_l0", 0))
    print(f"[packed recomputed witnesses] determinism={det:.1e}  max|delta| past={pd:.3e}  packed={packed}")
    assert packed > 0, "recomputed packed witness path did not run"
    assert det < 1e-6 and pd < 1e-6, f"packed recomputed witnesses leaked ({pd:.3e})"


def test_coarse_route_gating():
    # Option 2: bidirectional coarse routing (coarse-as-query) is NOT AR-causal, so it must
    # auto-disable under AR (no leak), and run + produce routing edges under bidirectional.
    T = BASE_CFG["block_size"]
    torch.manual_seed(1)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    j = T - 8
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j].item() + 12345) % tok0.vocab_size

    # AR: must be auto-disabled -> still exactly causal
    m_ar, _ = _build(dict(hqd_coarse_route_levels=[2, 3], witness_levels=[1, 2, 3]))
    base = _feats(m_ar, ids)
    past = (base - _feats(m_ar, ids2)).abs().flatten(start_dim=2).amax(dim=-1)[0][:j].max().item()
    print(f"[coarse_route under AR (auto-disabled)] max|delta| past={past:.3e}")
    assert past < 1e-6, "coarse routing leaked under AR — gating failed"

    # bidirectional baseline (no routing) for an edge-count reference
    m_off, _ = _build(dict(ar_graph_causal=False, witness_levels=[1, 2, 3]))
    _feats(m_off, ids)
    e_off = int(getattr(m_off, "_last_hqd_added_total", 0) or 0)

    # bidirectional: routing enabled, emits routing edges, trains
    m_bi, _ = _build(dict(ar_graph_causal=False, hqd_coarse_route_levels=[2, 3], witness_levels=[1, 2, 3]))
    m_bi.train()
    out = m_bi(ids)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    loss = feats.float().pow(2).mean()
    loss.backward()
    e_on = int(getattr(m_bi, "_last_hqd_added_total", 0) or 0)
    g = sum(p.grad.abs().sum().item() for p in m_bi.parameters() if p.grad is not None)
    print(f"[coarse_route bidi] edges {e_off}->{e_on}  loss={loss.item():.4f}  grad_sum={g:.2e}")
    assert torch.isfinite(loss) and g > 0 and e_on > e_off, "bidi coarse route produced no routing edges/grads"


def test_dense_backward_runs():
    # The causality/equivalence checks run under no_grad; the CUDA illegal-access bug
    # was in *backward* (non-contiguous q/k/v into SDPA). Exercise the grad path here.
    T = BASE_CFG["block_size"]
    torch.manual_seed(2)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    for ov in (dict(hqd_attn_impl="dense"), dict(hqd_attn_impl="dense", hqd_descent_stop_at=2)):
        model, _ = _build(ov)
        model.train()
        out = model(ids)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        loss = feats.float().pow(2).mean()
        loss.backward()
        gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        print(f"[dense backward {ov}] loss={loss.item():.4f}  grad_sum={gnorm:.3e}")
        assert torch.isfinite(loss) and gnorm > 0, "dense backward produced no/NaN grads"


def test_packed_graph_witness_backward_runs():
    T = BASE_CFG["block_size"]
    torch.manual_seed(3)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    model, _ = _build(dict(
        hqd_shallow_read_level=3,
        hqd_select_inside_message_passing=True,
        hqd_graph_witness_enable=True,
        hqd_graph_witness_topk=4,
        hqd_packed_witness_read=True,
        hqd_topk_l0=8,
    ))
    model.train()
    out = model(ids)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    loss = feats.float().pow(2).mean()
    loss.backward()
    stats = getattr(model, "_last_hqd_stage_stats", {}) or {}
    packed = int(stats.get("packed_witness_l0", 0))
    gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"[packed graph witnesses backward] packed={packed} loss={loss.item():.4f} grad_sum={gnorm:.3e}")
    assert packed > 0 and torch.isfinite(loss) and gnorm > 0, "packed witness backward produced no reads/grads"


def test_packed_recomputed_witness_backward_runs():
    T = BASE_CFG["block_size"]
    torch.manual_seed(4)
    tok0 = AutoTokenizer.from_pretrained(BASE_CFG["tokenizer_name"])
    ids = torch.randint(0, tok0.vocab_size, (1, T))
    model, _ = _build(dict(
        hqd_shallow_read_level=3,
        hqd_select_inside_message_passing=True,
        hqd_graph_witness_enable=False,
        hqd_packed_witness_read=True,
        hqd_packed_witness_source="recompute",
        use_witness_packets=True,
        use_summary_witnesses=True,
        use_rare_witnesses=True,
        witness_levels=[3],
        hqd_topk_l0=8,
    ))
    model.train()
    out = model(ids)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    loss = feats.float().pow(2).mean()
    loss.backward()
    stats = getattr(model, "_last_hqd_stage_stats", {}) or {}
    packed = int(stats.get("packed_witness_l0", 0))
    gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"[packed recomputed witnesses backward] packed={packed} loss={loss.item():.4f} grad_sum={gnorm:.3e}")
    assert packed > 0 and torch.isfinite(loss) and gnorm > 0, "recomputed packed witness backward failed"


if __name__ == "__main__":
    test_dense_causality_and_equivalence()
    test_descent_shortcut_causality()
    test_cross_level_witness_causality()
    test_shallow_read_causality()
    test_packed_graph_witness_read_causality()
    test_packed_recomputed_witness_read_causality()
    test_coarse_route_gating()
    test_dense_backward_runs()
    test_packed_graph_witness_backward_runs()
    test_packed_recomputed_witness_backward_runs()
    print("done")
