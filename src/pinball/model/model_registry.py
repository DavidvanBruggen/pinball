# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import logging
from typing import Any, Optional

from .enhanced_hierarchical_flow_gat import EnhancedHierarchicalFlowGAT
from .transformer_baseline import TransformerConfig, TransformerLM


logger = logging.getLogger(__name__)


def normalize_model_type(model_type: Any) -> str:
    value = str(model_type or "pinball").lower().strip()
    aliases = {
        "hgt": "pinball",
        "hierarchical": "pinball",
        "hierarchical_graph_transformer": "pinball",
        "pinball-base": "pinball",
        "pinball_hqd": "pinball_hqd",
        "pinball-hqd": "pinball_hqd",
        "gpt": "transformer",
        "gpt2": "transformer",
        "gpt_baseline": "transformer",
    }
    value = aliases.get(value, value)
    if value not in {"pinball", "pinball_hqd", "transformer"}:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    return value


def count_parameters(model, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


def build_transformer_model(args, tokenizer, vocab_size: int, max_seq_len: int) -> TransformerLM:
    n_embd = int(getattr(args, "transformer_n_embd", 0) or getattr(args, "hidden_dim", 768))
    n_head = int(getattr(args, "transformer_n_head", 0) or getattr(args, "num_heads", 12))
    n_layer = int(getattr(args, "transformer_n_layer", 0) or 12)
    config = TransformerConfig(
        vocab_size=int(vocab_size),
        block_size=int(max_seq_len),
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=float(getattr(args, "transformer_dropout", getattr(args, "dropout", 0.0))),
        bias=bool(getattr(args, "transformer_bias", True)),
        norm_type=str(getattr(args, "transformer_norm_type", getattr(args, "norm_type", "layernorm"))),
        norm_eps=float(getattr(args, "transformer_norm_eps", getattr(args, "norm_eps", 1e-5))),
        use_rope=bool(getattr(args, "transformer_use_rope", False)),
        use_abs_pos_emb=bool(getattr(args, "transformer_use_abs_pos_emb", True)),
        attn_backend=str(getattr(args, "transformer_attn_backend", "auto")),
        gradient_checkpointing=bool(getattr(args, "transformer_gradient_checkpointing", getattr(args, "use_gradient_checkpointing", False))),
        tie_weights=bool(getattr(args, "transformer_tie_weights", True)),
        ffn_type=str(getattr(args, "transformer_ffn_type", "swiglu")),
    )
    model = TransformerLM(config, tokenizer=tokenizer)
    logger.info(
        "Built Transformer baseline: layers=%d heads=%d embd=%d block=%d backend=%s ffn=%s rope=%s abs_pos=%s tied=%s",
        int(config.n_layer),
        int(config.n_head),
        int(config.n_embd),
        int(config.block_size),
        str(config.attn_backend),
        str(config.ffn_type),
        bool(config.use_rope),
        bool(config.use_abs_pos_emb),
        bool(config.tie_weights),
    )
    return model



def build_pinball_model(
    args,
    tokenizer,
    vocab_size: int,
    input_mode: str,
    tie_weights: bool,
    max_seq_len: int,
    class_cond_enable: bool = False,
):
    model = EnhancedHierarchicalFlowGAT(
        tokenizer=tokenizer,
        vocab_size=int(vocab_size),
        input_mode=input_mode,
        tie_weights=bool(tie_weights),
        hidden_dim=int(getattr(args, "hidden_dim", 384)),
        num_heads=int(getattr(args, "num_heads", 6)),
        num_layers=getattr(args, "num_layers", [4, 4, 4, 4]),
        dropout=float(getattr(args, "dropout", 0.1)),
        compression_ratios=getattr(args, "compression_ratios", [128, 16, 8]),
        overlap_ratios=getattr(args, "overlap_ratios", [0.5, 0.5, 0.5]),
        max_seq_len=int(max_seq_len),
        use_final_layer_for_prediction=bool(getattr(args, "use_final_layer", False)),
        refinement_cycles=int(getattr(args, "refinement_cycles", 3)),
        use_edge_attr=bool(getattr(args, "use_edge_embedding", False)),
        learn_edge_from_attn=bool(getattr(args, "learn_edge_from_attention", True)),
        sparse_attn_mode=str(getattr(args, "sparse_attn_mode", "off")),
        sparse_attn_chunk_size=int(getattr(args, "sparse_attn_chunk_size", 0)),
        share_transformers=bool(getattr(args, "share_transformers", False)),
        per_level_local_qkv=bool(getattr(args, "per_level_local_qkv", False)),
        norm_type=str(getattr(args, "norm_type", "rmsnorm")),
        norm_eps=float(getattr(args, "norm_eps", 1e-6)),
        add_self_loops=bool(getattr(args, "add_self_loops", True)),
        add_long_range_edges=bool(getattr(args, "add_long_range_edges", False)),
        long_range_distance=int(getattr(args, "long_range_distance", 3)),
        iterative_refinement_cycles=int(getattr(args, "iterative_refinement_cycles", 3)),
        unified_refinement_cycles=int(getattr(args, "unified_refinement_cycles", 0)),
        autoenc_graph_mode=str(getattr(args, "autoenc_graph_mode", "off")),
        autoenc_coupled_feedback=bool(getattr(args, "autoenc_coupled_feedback", False)),
        ensure_l0_past_parent_edges=bool(getattr(args, "ensure_l0_past_parent_edges", False)),
        l0_past_parent_min_level=int(getattr(args, "l0_past_parent_min_level", 1)),
        l0_past_parent_max_level=(
            None if getattr(args, "l0_past_parent_max_level", None) in (None, -1)
            else int(getattr(args, "l0_past_parent_max_level", None))
        ),
        l0_cycles=int(getattr(args, "l0_cycles", 8)),
        l0_local_backend=str(getattr(args, "l0_local_backend", "pyg")),
        l0_local_window=int(getattr(args, "l0_local_window", 0)),
        neighbor_sampling_backend=str(getattr(args, "neighbor_sampling_backend", "auto")),
        rope_mode=str(getattr(args, "rope_mode", "auto")),
        ablate_levels=getattr(args, "ablate_levels", None),
        report_level_connectivity=bool(getattr(args, "report_level_connectivity", False)),
        local_attn_levels=getattr(args, "local_attn_levels", None),
        local_attn_windows=getattr(args, "local_attn_windows", None) if getattr(args, "local_attn_windows", None) else None,
        local_attn_causal_levels=getattr(args, "local_attn_causal_levels", None),
        local_attn_flash_dtype_cast=bool(getattr(args, "local_attn_flash_dtype_cast", False)),
        local_attn_sampled_mode=str(getattr(args, "local_attn_sampled_mode", "safe_sdpa")),
        attention_source_gating_enable=bool(getattr(args, "attention_source_gating_enable", False)),
        attention_source_gate_init_graph=float(getattr(args, "attention_source_gate_init_graph", 1.0)),
        attention_source_gate_init_local=float(getattr(args, "attention_source_gate_init_local", 0.5)),
        attention_source_gate_init_hqd=float(getattr(args, "attention_source_gate_init_hqd", 0.1)),
        attention_source_gate_debug=bool(getattr(args, "attention_source_gate_debug", False)),
        lateral_edge_trace_enable=bool(getattr(args, "lateral_edge_trace_enable", False)),
        lateral_edge_trace_mode=str(getattr(args, "lateral_edge_trace_mode", "windowed_approx")),
        lateral_edge_trace_decay=float(getattr(args, "lateral_edge_trace_decay", 0.95)),
        lateral_edge_trace_eta=float(getattr(args, "lateral_edge_trace_eta", 0.02)),
        lateral_edge_trace_alpha=float(getattr(args, "lateral_edge_trace_alpha", 0.25)),
        lateral_edge_trace_max=float(getattr(args, "lateral_edge_trace_max", 2.0)),
        lateral_edge_trace_per_head=bool(getattr(args, "lateral_edge_trace_per_head", True)),
        lateral_edge_trace_credit=str(getattr(args, "lateral_edge_trace_credit", "attn")),
        lateral_edge_trace_center_per_dst=bool(getattr(args, "lateral_edge_trace_center_per_dst", True)),
        lateral_edge_trace_update_during_eval=bool(getattr(args, "lateral_edge_trace_update_during_eval", False)),
        lateral_edge_trace_detach=bool(getattr(args, "lateral_edge_trace_detach", True)),
        lateral_edge_trace_debug=bool(getattr(args, "lateral_edge_trace_debug", False)),
        edge_conditioning_enable=bool(getattr(args, "edge_conditioning_enable", False)),
        edge_type_generator_enable=bool(getattr(args, "edge_type_generator_enable", False)),
        edge_type_embedding_dim=int(getattr(args, "edge_type_embedding_dim", 32)),
        edge_condition_hidden_dim=int(getattr(args, "edge_condition_hidden_dim", 64)),
        edge_condition_num_types=int(getattr(args, "edge_condition_num_types", 16)),
        edge_logit_bias_enable=bool(getattr(args, "edge_logit_bias_enable", True)),
        edge_value_gate_enable=bool(getattr(args, "edge_value_gate_enable", True)),
        edge_logit_bias_per_head=bool(getattr(args, "edge_logit_bias_per_head", True)),
        edge_value_gate_per_head=bool(getattr(args, "edge_value_gate_per_head", False)),
        edge_value_gate_per_channel=bool(getattr(args, "edge_value_gate_per_channel", True)),
        edge_gate_init_identity=bool(getattr(args, "edge_gate_init_identity", True)),
        edge_logit_bias_init_zero=bool(getattr(args, "edge_logit_bias_init_zero", True)),
        edge_condition_dropout=float(getattr(args, "edge_condition_dropout", 0.0)),
        edge_condition_debug=bool(getattr(args, "edge_condition_debug", False)),
        edge_node_condition_enable=bool(getattr(args, "edge_node_condition_enable", False)),
        edge_node_condition_detach=bool(getattr(args, "edge_node_condition_detach", True)),
        edge_node_condition_dim=int(getattr(args, "edge_node_condition_dim", 32)),
        edge_node_condition_mode=str(getattr(args, "edge_node_condition_mode", "src_dst_prod")),
        edge_node_condition_zero_init=bool(getattr(args, "edge_node_condition_zero_init", True)),
        edge_gate_scale=float(getattr(args, "edge_gate_scale", 0.1)),
        token_unet_enable=bool(getattr(args, "token_unet_enable", False)),
        token_unet_mode=str(getattr(args, "token_unet_mode", "stem")),
        token_unet_dim=str(getattr(args, "token_unet_dim", "auto")),
        token_unet_2d_causal=bool(getattr(args, "token_unet_2d_causal", False)),
        token_unet_scale=int(getattr(args, "token_unet_scale", 1)),
        token_unet_kernel_size=int(getattr(args, "token_unet_kernel_size", 5)),
        token_unet_dropout=float(getattr(args, "token_unet_dropout", 0.0)),
        token_unet_right_edge_targets=bool(getattr(args, "token_unet_right_edge_targets", True)),
        token_unet_lookahead_decode_enable=bool(getattr(args, "token_unet_lookahead_decode_enable", False)),
        token_unet_lookahead_kernel_size=int(getattr(args, "token_unet_lookahead_kernel_size", 5)),
        token_unet_lookahead_blocks=int(getattr(args, "token_unet_lookahead_blocks", 2)),
        token_unet_block=str(getattr(args, "token_unet_block", "cnn")),
        rgb_token_unet_enable=bool(
            str(getattr(args, "modality", "text")).lower() == "image"
            and str(getattr(args, "image_token_mode", "latent")).lower() == "rgb_unet"
            and not (
                str(getattr(args, "image_objective", "diffusion")).lower() == "maskgit"
                and str(getattr(args, "image_maskgit_variant", "continuous")).lower() == "discrete"
            )
        ),
        rgb_token_unet_downsample=int(getattr(args, "image_rgb_unet_downsample", 16)),
        rgb_token_unet_base_channels=int(getattr(args, "image_rgb_unet_base_channels", 64)),
        rgb_token_unet_kernel_size=int(getattr(args, "image_rgb_unet_kernel_size", 5)),
        rgb_token_unet_decode_kernel_size=int(getattr(args, "image_rgb_unet_decode_kernel_size", 3)),
        rgb_token_unet_decode_separable=bool(getattr(args, "image_rgb_unet_decode_separable", True)),
        rgb_token_unet_max_channels=int(getattr(args, "image_rgb_unet_max_channels", 512)),
        graph_geometry_mode=str(getattr(args, "graph_geometry_mode", "sequence")),
        graph_grid_height=int(getattr(args, "graph_grid_height", 0)),
        graph_grid_width=int(getattr(args, "graph_grid_width", 0)),
        graph_spatial_metric=str(getattr(args, "graph_spatial_metric", "chebyshev")),
        graph_downsample_factor=int(getattr(args, "graph_downsample_factor", 2)),
        rope_level_axis_enable=bool(getattr(args, "rope_level_axis_enable", False)),
        rope_level_axis_scale=float(getattr(args, "rope_level_axis_scale", 32.0)),
        refine_cond_mode=str(getattr(args, "refine_cond_mode", "none")),
        refine_cond_strength=float(getattr(args, "refine_cond_strength", 1.0)),
        refine_cond_concat_gate_init=float(getattr(args, "refine_cond_concat_gate_init", -2.0)),
        local_connectivity_window_size=int(getattr(args, "local_connectivity_window_size", 4)),
        # --- Hierarchy auxiliary (reconstruction) loss ---
        use_aux_loss=bool(getattr(args, "use_aux_loss", False)),
        # --- AR graph connectivity (causal edge construction for autoregressive training) ---
        hier_ar_enable=bool(getattr(args, "hier_ar_enable", False)),
        hier_ar_allow_same_time=bool(getattr(args, "hier_ar_allow_same_time", True)),
        l0_ar_enable=bool(getattr(args, "l0_ar_enable", False)),
        enable_l0_parent_edges=bool(getattr(args, "enable_l0_parent_edges", False)),
        l0_parent_edges_bidirectional=bool(getattr(args, "l0_parent_edges_bidirectional", False)),
        ensure_l0_past_l1_edges=bool(getattr(args, "ensure_l0_past_l1_edges", False)),
        ensure_past_hier_edges_all_levels=bool(getattr(args, "ensure_past_hier_edges_all_levels", False)),
        l0_past_l1_edge_type_id=getattr(args, "l0_past_l1_edge_type_id", None),
        hierarchical_query_descent_enable=bool(getattr(args, "hierarchical_query_descent_enable", False)),
        hqd_topk_l3=int(getattr(args, "hqd_topk_l3", 4)),
        hqd_topk_l2=int(getattr(args, "hqd_topk_l2", 4)),
        hqd_topk_l1=int(getattr(args, "hqd_topk_l1", 4)),
        hqd_topk_l0=int(getattr(args, "hqd_topk_l0", 64)),
        hqd_l0_topk_enable=bool(getattr(args, "hqd_l0_topk_enable", True)),
        hqd_include_local_window=bool(getattr(args, "hqd_include_local_window", False)),
        hqd_local_window_size=int(getattr(args, "hqd_local_window_size", 0)),
        hqd_causal=getattr(args, "hqd_causal", "auto"),
        hqd_use_existing_zipper_projections=bool(getattr(args, "hqd_use_existing_zipper_projections", True)),
        hqd_debug=bool(getattr(args, "hqd_debug", False)),
        hqd_granularity=str(getattr(args, "hqd_granularity", "per_layer")),
        hqd_every_n=int(getattr(args, "hqd_every_n", -1)),
        hqd_reuse_previous=bool(getattr(args, "hqd_reuse_previous", False)),
        hqd_reuse_max_age=int(getattr(args, "hqd_reuse_max_age", 0)),
        hqd_query_chunk_size=int(getattr(args, "hqd_query_chunk_size", 512)),
        hqd_query_level=int(getattr(args, "hqd_query_level", 0)),
        hqd_stop_level=int(getattr(args, "hqd_stop_level", 0)),
        hqd_handoff_to_l0=bool(getattr(args, "hqd_handoff_to_l0", False)),
        hqd_global_topk=int(getattr(args, "hqd_global_topk", 0)),
        hqd_assume_disjoint_children=bool(getattr(args, "hqd_assume_disjoint_children", False)),
        hqd_validate_disjoint_children=bool(getattr(args, "hqd_validate_disjoint_children", False)),
        hqd_sparse_project_active_only=bool(getattr(args, "hqd_sparse_project_active_only", False)),
        hqd_select_inside_message_passing=bool(getattr(args, "hqd_select_inside_message_passing", False)),
        verbose=bool(getattr(args, "verbose", False)),
        class_cond_enable=bool(class_cond_enable),
        num_classes=int(getattr(args, "image_num_classes", 1000)),
        class_cond_drop_prob=float(getattr(args, "image_class_cond_drop_prob", 0.1)),
        internal_cycles_per_level=getattr(args, "internal_cycles", [4, 4, 4]),
        train_with_imputation=bool(getattr(args, "train_with_imputation", False)),
        lap_pe_k=int(getattr(args, "lap_pe_k", 0)),
        refinement_style=str(getattr(args, "refinement_style", "unified")),
        num_refinement_layers=int(getattr(args, "num_refinement_layers", 2)),
        use_gradient_checkpointing=bool(getattr(args, "use_gradient_checkpointing", False)),
        refinement_batch_mode=str(getattr(args, "refinement_batch_mode", "true_batch_nozip")),
        use_neighbor_sampling=bool(getattr(args, "use_neighbor_sampling", False)),
        num_neighbors=getattr(args, "num_neighbors", [8]),
        sampling_batch_size=int(getattr(args, "sampling_batch_size", 8192)),
        sampling_seed_budget=int(getattr(args, "sampling_seed_budget", 4096)),
        pinball_level_dims=getattr(args, "pinball_level_dims", None),
        pinball_work_dim=getattr(args, "pinball_work_dim", None),
        pinball_work_num_heads=getattr(args, "pinball_work_num_heads", None),
        pinball_adapter_type=str(getattr(args, "pinball_adapter_type", "low_rank")),
        pinball_adapter_rank=int(getattr(args, "pinball_adapter_rank", 128)),
        pinball_level_cycle_enable=bool(getattr(args, "pinball_level_cycle_enable", False)),
        pinball_level_cycles=getattr(args, "pinball_level_cycles", None),
        pinball_cycle_schedule=str(getattr(args, "pinball_cycle_schedule", "staggered_flush")),
        pinball_message_fn=str(getattr(args, "pinball_message_fn", "original")),
        pinball_graph_active_compute=str(getattr(args, "pinball_graph_active_compute", "all")),
        pinball_level_cycle_mode=str(getattr(args, "pinball_level_cycle_mode", "extra_cycles")),
        pinball_multirate_enable=bool(getattr(args, "pinball_multirate_enable", False)),
        pinball_multirate_schedule=str(getattr(args, "pinball_multirate_schedule", "after_full_stack")),
        pinball_multirate_midpoint_layer=getattr(args, "pinball_multirate_midpoint_layer", "auto"),
        pinball_multirate_midpoint_repeats=int(getattr(args, "pinball_multirate_midpoint_repeats", 1)),
        pinball_multirate_skip_after_last_cycle=bool(getattr(args, "pinball_multirate_skip_after_last_cycle", True)),
        pinball_multirate_debug=bool(getattr(args, "pinball_multirate_debug", False)),
        pinball_upper_refine_steps=int(getattr(args, "pinball_upper_refine_steps", 0)),
        pinball_top_refine_steps=int(getattr(args, "pinball_top_refine_steps", 0)),
        pinball_l3_workspace_tokens=int(getattr(args, "pinball_l3_workspace_tokens", 0)),
        pinball_upper_refine_shared_weights=bool(getattr(args, "pinball_upper_refine_shared_weights", True)),
        pinball_top_refine_shared_weights=bool(getattr(args, "pinball_top_refine_shared_weights", True)),
        pinball_consolidate_after_upper=bool(getattr(args, "pinball_consolidate_after_upper", False)),
        pinball_upper_refine_every=int(getattr(args, "pinball_upper_refine_every", 1)),
        pinball_top_refine_every=int(getattr(args, "pinball_top_refine_every", 1)),
        pinball_multirate_attn_backend=str(getattr(args, "pinball_multirate_attn_backend", "auto")),
        pinball_upper_refine_window=int(getattr(args, "pinball_upper_refine_window", 0)),
        pinball_top_refine_window=int(getattr(args, "pinball_top_refine_window", 0)),
        pinball_upper_refine_causal=bool(getattr(args, "pinball_upper_refine_causal", True)),
        pinball_top_refine_causal=bool(getattr(args, "pinball_top_refine_causal", True)),
        pinball_upper_cross_attn_steps=int(getattr(args, "pinball_upper_cross_attn_steps", 0)),
        pinball_upper_query_topk_l2=int(getattr(args, "pinball_upper_query_topk_l2", 0)),
        pinball_upper_cross_attn_backend=str(getattr(args, "pinball_upper_cross_attn_backend", "auto")),
        pinball_upper_cross_attn_causal=bool(getattr(args, "pinball_upper_cross_attn_causal", True)),
        pinball_upper_cross_attn_shared_weights=bool(getattr(args, "pinball_upper_cross_attn_shared_weights", True)),
        pinball_upper_update_l2_enable=bool(getattr(args, "pinball_upper_update_l2_enable", False)),
        pinball_upper_update_l2_scale_init=float(getattr(args, "pinball_upper_update_l2_scale_init", 1.0e-2)),
        pinball_upper_cross_write_scale_init=float(getattr(args, "pinball_upper_cross_write_scale_init", 1.0e-2)),
        pinball_cross_query_pairs=getattr(args, "pinball_cross_query_pairs", None),
        pinball_cross_query_steps=int(getattr(args, "pinball_cross_query_steps", 0)),
        pinball_cross_query_topk=int(getattr(args, "pinball_cross_query_topk", 0)),
        pinball_cross_query_l0_window=int(getattr(args, "pinball_cross_query_l0_window", 0)),
        pinball_cross_query_backend=str(getattr(args, "pinball_cross_query_backend", "auto")),
        pinball_cross_query_causal=bool(getattr(args, "pinball_cross_query_causal", True)),
        pinball_cross_query_shared_weights=bool(getattr(args, "pinball_cross_query_shared_weights", True)),
        pinball_cross_query_update_memory_enable=bool(getattr(args, "pinball_cross_query_update_memory_enable", False)),
        pinball_cross_query_selection=str(getattr(args, "pinball_cross_query_selection", "global_mean")),
        pinball_cross_query_write_scale_init=float(getattr(args, "pinball_cross_query_write_scale_init", 1.0e-2)),
        # Cycle governor (early-stop of refinement cycles by a convergence metric).
    )
    model.model_type = normalize_model_type(getattr(args, "model_type", "pinball"))
    logger.info(
        "Built Pinball model: model_type=%s hqd=%s hidden=%d heads=%d max_seq_len=%d",
        model.model_type,
        bool(getattr(args, "hierarchical_query_descent_enable", False)),
        int(getattr(args, "hidden_dim", 384)),
        int(getattr(args, "num_heads", 6)),
        int(max_seq_len),
    )
    model.unified_skeleton_device_cache_enable = bool(getattr(args, "unified_skeleton_device_cache_enable", True))
    if not model.unified_skeleton_device_cache_enable and hasattr(model, "_unified_skeleton_device_cache"):
        model._unified_skeleton_device_cache.clear()
    return model


def _list_arg(args, name: str, default):
    value = getattr(args, name, None)
    if value is None:
        return list(default)
    return list(value)


def _fit_list(values, length: int):
    values = list(values)
    if not values:
        raise ValueError("Expected a non-empty list")
    while len(values) < int(length):
        values.append(values[-1])
    return values[: int(length)]


def build_model(
    args,
    tokenizer,
    vocab_size: int,
    input_mode: str = "tokens",
    tie_weights: bool = True,
    max_seq_len: Optional[int] = None,
    class_cond_enable: bool = False,
):
    model_type = normalize_model_type(getattr(args, "model_type", "pinball"))
    max_seq_len = int(max_seq_len if max_seq_len is not None else getattr(args, "block_size", 1024))
    if model_type == "transformer":
        if str(getattr(args, "modality", "text")).lower() != "text":
            raise ValueError("model_type=transformer is currently supported for text/token inputs only")
        return build_transformer_model(args, tokenizer=tokenizer, vocab_size=int(vocab_size), max_seq_len=max_seq_len)
    return build_pinball_model(
        args,
        tokenizer=tokenizer,
        vocab_size=int(vocab_size),
        input_mode=input_mode,
        tie_weights=bool(tie_weights),
        max_seq_len=max_seq_len,
        class_cond_enable=bool(class_cond_enable),
    )
