# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
"""Config-driven training entry point for Pinball.

Usage:
    pinball-train --config configs/pinball_wikitext.yaml
    pinball-train --config configs/pinball_wikitext.yaml --text-file my.txt --max-steps 200

This drives the proven ``EnhancedHierarchicalTrainer`` epoch loop, which renders the
live progress bar with throughput (``tok/s`` + a short rolling ``tok/s_win`` — the
long-range model's selling point is that throughput stays roughly constant as context
grows), loss, objective/CE, grad-norm and learning rate. Training is organised into
epochs of ``samples_per_epoch`` samples; after each epoch it validates (reporting
perplexity) and prints a generated sample, so you get periodic generations throughout
the run. It supports the autoregressive (``ar``) and masked-diffusion (``masked``)
objectives selected by the config's ``train_mode`` / ``train_objective_mode``.
"""
from __future__ import annotations

import argparse
import logging
import math
import os

# Reduce CUDA allocator fragmentation (long sequences + variable-size FFT/graph buffers
# leave large reserved-but-unallocated gaps that OOM at epoch boundaries). Must be set
# before torch initialises the CUDA caching allocator. setdefault: a shell override wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

from .config import PinballConfig
from .model import build_model, count_parameters
from .data import create_karpathy_dataloaders
from .train import EnhancedHierarchicalTrainer

logger = logging.getLogger("pinball.train")


def _resolve_device(cfg) -> torch.device:
    dev = getattr(cfg, "device", None)
    if dev:
        return torch.device(dev)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(getattr(cfg, "tokenizer_name", "gpt2"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Masked-diffusion objective needs a [MASK] token in the vocab.
    objective = str(getattr(cfg, "train_objective_mode", "ar")).lower()
    if objective in {"masked", "hybrid"} and tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    return tok


def _build_optimizer(model, cfg):
    """Build the optimizer selected by ``cfg.optimizer`` (``adamw`` | ``muon_hybrid``).

    ``muon_hybrid`` mirrors the reference setup: 2D+ weight matrices are trained with
    Muon at ``learning_rate * muon_lr_mult`` (orthogonalized updates, RMS-matched to
    AdamW via ``muon_adjust_lr_fn``), while 1D params (embeddings, norms, biases) use
    AdamW at the base LR. Each group records ``lr_scale`` so the scheduler keeps the
    per-group LR ratio through warmup/decay.
    """
    lr = float(getattr(cfg, "learning_rate", 3e-4))
    weight_decay = float(getattr(cfg, "weight_decay", 0.1))
    kind = str(getattr(cfg, "optimizer", "adamw")).lower()

    if kind in {"adamw", "adam"}:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if kind != "muon_hybrid":
        raise SystemExit(f"Unknown optimizer '{kind}'. Use 'adamw' or 'muon_hybrid'.")

    try:
        import pytorch_optimizer as po
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "optimizer: muon_hybrid requires the 'pytorch_optimizer' package "
            "(pip install pytorch_optimizer, or pip install -e \".[muon]\")."
        ) from exc

    muon_lr_mult = float(getattr(cfg, "muon_lr_mult", 6.0))
    betas = getattr(cfg, "muon_betas", (0.9, 0.95))
    adjust_lr_fn = str(getattr(cfg, "muon_adjust_lr_fn", "match_rms_adamw"))

    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    other_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    param_groups = []
    if muon_params:
        param_groups.append(dict(
            params=muon_params, lr=lr * muon_lr_mult, lr_scale=muon_lr_mult,
            weight_decay=weight_decay, use_muon=True, adjust_lr_fn=adjust_lr_fn,
        ))
    if other_params:
        param_groups.append(dict(
            params=other_params, lr=lr, lr_scale=1.0, weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])), use_muon=False,
        ))
    if not param_groups:
        raise SystemExit("No trainable parameters found for the optimizer.")

    logger.info("Using Muon hybrid optimizer: muon_params=%d, other_params=%d, lr_mult=%.3f",
                len(muon_params), len(other_params), muon_lr_mult)
    return po.Muon(param_groups)


def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 10% of peak LR (keeps ``lr`` live in the bar)."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train a Pinball / transformer model from a YAML config.")
    p.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    p.add_argument("--text-file", default=None, help="Override the training text file.")
    p.add_argument("--device", default=None, help="Override device (cuda, cuda:0, cpu, ...).")
    p.add_argument("--attn-backend", "--attn_backend", dest="attn_backend", default=None,
                   choices=["pyg", "flash", "sdpa"],
                   help="Local-attention backend (overrides config). Use 'sdpa' or 'pyg' when "
                        "flash-attn is unavailable (e.g. Colab).")
    p.add_argument("--ablate-levels", "--ablate_levels", dest="ablate_levels", default=None,
                   help="Comma-separated hierarchy levels to disable for ablation, e.g. '3' or "
                        "'2,3' (cuts all their edges; L0 cannot be ablated). Overrides config.")
    p.add_argument("--report-levels", "--report_level_connectivity", dest="report_levels",
                   action="store_true", default=False,
                   help="Log a per-level connectivity / inert-level report on the first forward.")
    p.add_argument("--block-size", "--block_size", type=int, default=None, dest="block_size",
                   help="Sequence length (overrides config block_size).")
    p.add_argument("--batch-size", "--batch_size", type=int, default=None, dest="batch_size",
                   help="Batch size (overrides config batch_size).")
    p.add_argument("--gradient-accumulation-steps", "--gradient_accumulation_steps", "--grad-accum",
                   type=int, default=None, dest="gradient_accumulation_steps",
                   help="Accumulate grads over N batches per optimizer step (overrides config).")
    p.add_argument("--max-steps", type=int, default=None, help="Total optimizer-step budget (overrides config).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs to train (overrides config num_epochs; otherwise derived from max-steps).")
    p.add_argument("--samples-per-epoch", type=int, default=None,
                   help="Samples that constitute one epoch (validation + a generation follow each epoch).")
    p.add_argument("--log-every", type=int, default=None,
                   help="Emit the detailed log lines every N steps (overrides config; default 10000).")
    p.add_argument("--eval-batches", type=int, default=50, help="Validation batches per epoch.")
    p.add_argument("--generate-every", type=int, default=None,
                   help="Generate a sample every N training batches (overrides config; 0 = per-epoch only).")
    p.add_argument("--save-every", type=int, default=None,
                   help="Save a checkpoint every N training batches (overrides config; 0 = end + per-epoch only).")
    p.add_argument("--checkpoint-dir", default=None, help="Where to write checkpoints (default: ./checkpoints).")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint .pt (overrides config resume_from); restores weights, "
                        "optimizer, LR schedule and step/epoch counters and continues the run.")
    p.add_argument("--prompt", default=None, help="Prompt for the generation samples.")
    p.add_argument("--no-generate", action="store_true", help="Skip all generation samples.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = PinballConfig.from_yaml(args.config)
    if args.device:
        cfg.device = args.device
    if args.text_file:
        cfg.text_file = args.text_file
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.attn_backend is not None:
        cfg.l0_local_backend = args.attn_backend
    if args.ablate_levels is not None:
        cfg.ablate_levels = [int(x) for x in str(args.ablate_levels).split(",") if x.strip() != ""]
    if args.report_levels:
        cfg.report_level_connectivity = True
    device = _resolve_device(cfg)

    tokenizer = _build_tokenizer(cfg)
    block_size = int(getattr(cfg, "block_size", 1024))
    batch_size = int(getattr(cfg, "batch_size", 8))
    model = build_model(
        cfg, tokenizer=tokenizer, vocab_size=len(tokenizer),
        input_mode="tokens", tie_weights=True, max_seq_len=block_size,
    ).to(device)
    model.emit_features_only = True
    logger.info("Built %s: %s params, block_size=%d, device=%s",
                getattr(cfg, "model_type", "pinball"), f"{count_parameters(model):,}", block_size, device)

    text_file = getattr(cfg, "text_file", None)
    if not text_file:
        raise SystemExit("Config must set `text_file:` (or pass --text-file). See configs/ + README.")
    train_loader, val_loader = create_karpathy_dataloaders(
        text_path=text_file, tokenizer=tokenizer, block_size=block_size,
        batch_size=batch_size, val_split=float(getattr(cfg, "val_split", 0.01)),
        stream_name=getattr(cfg, "stream_name", None),
    )

    objective = str(getattr(cfg, "train_objective_mode", "ar")).lower()
    use_amp = device.type == "cuda" and bool(getattr(cfg, "mixed_precision", True))

    max_steps = args.max_steps if args.max_steps is not None else int(getattr(cfg, "max_steps", 10_000))
    samples_per_epoch = args.samples_per_epoch or int(getattr(cfg, "samples_per_epoch", 25_000))
    steps_per_epoch = max(1, samples_per_epoch // batch_size)
    # Epoch behaviour: an explicit epoch count (CLI or config) drives the run and sets the
    # step budget; otherwise the number of epochs is derived from the step budget.
    explicit_epochs = args.epochs if args.epochs is not None else getattr(cfg, "num_epochs", None)
    if explicit_epochs:
        num_epochs = max(1, int(explicit_epochs))
        budget = num_epochs * steps_per_epoch
        max_steps = min(max_steps, budget) if args.max_steps is not None else budget
    else:
        num_epochs = max(1, math.ceil(max_steps / steps_per_epoch))
    warmup_steps = int(getattr(cfg, "warmup_steps", min(200, max_steps // 20)))

    grad_accum = max(1, int(getattr(cfg, "gradient_accumulation_steps", 1)))
    checkpoint_dir = args.checkpoint_dir or str(getattr(cfg, "checkpoint_dir", "./checkpoints"))
    generate_every = int(args.generate_every if args.generate_every is not None else getattr(cfg, "generate_every", 0) or 0)
    save_every = int(args.save_every if args.save_every is not None else getattr(cfg, "save_every", 0) or 0)
    if save_every > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = _build_optimizer(model, cfg)
    lr_scheduler = _build_scheduler(optimizer, warmup_steps, max_steps)

    trainer = EnhancedHierarchicalTrainer(
        model, None,
        optimizer=optimizer, lr_scheduler=lr_scheduler, tokenizer=tokenizer, device=device,
        train_objective_mode=objective, mixed_precision=use_amp,
        log_interval=int(args.log_every if args.log_every is not None else getattr(cfg, "log_every", 10000)),
        eval_interval=int(args.eval_batches),
        gradient_accumulation_steps=grad_accum,
        checkpoint_dir=checkpoint_dir,
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 1.0)),
        unified_refinement_cycles=int(getattr(cfg, "unified_refinement_cycles", 1)),
        lambda_ar_loss=1.0 if objective == "ar" else float(getattr(cfg, "lambda_ar", 0.1)),
        lambda_masked_loss=1.0 if objective != "ar" else 0.0,
        lambda_base_ce_loss=1.0,
        lambda_copy_loss=float(getattr(cfg, "lambda_copy_loss", 1.0)),
        copy_task_enable=bool(getattr(cfg, "copy_task_enable", False)),
        copy_task_train_prob=float(getattr(cfg, "copy_task_train_prob", 0.1)),
        copy_task_val_prob=float(getattr(cfg, "copy_task_val_prob", 1.0)),
        copy_task_src_len_min=int(getattr(cfg, "copy_task_src_len_min", 8)),
        copy_task_src_len_max=int(getattr(cfg, "copy_task_src_len_max", 64)),
        copy_task_min_gap=int(getattr(cfg, "copy_task_min_gap", 64)),
        copy_task_max_gap=int(getattr(cfg, "copy_task_max_gap", 0)),
        modality="text",
    )

    # Resume: restore weights + optimizer + LR schedule + step/epoch counters and continue.
    resume_path = args.resume if args.resume is not None else getattr(cfg, "resume_from", None)
    start_epoch = 0
    resumed_steps = 0
    if resume_path:
        if not os.path.isfile(resume_path):
            raise SystemExit(f"--resume: checkpoint not found: {resume_path}")
        trainer.load_checkpoint(resume_path)
        resumed_steps = int(getattr(trainer, "global_step", 0))
        start_epoch = int(getattr(trainer, "current_epoch", 0))
        logger.info("Resumed from %s  (global_step=%d, epoch=%d)",
                    resume_path, resumed_steps, start_epoch)

    train_data = {"get_batch": train_loader, "steps_per_epoch": steps_per_epoch}
    val_data = {"get_batch": val_loader, "steps_per_epoch": int(args.eval_batches)}
    # Prompt source: an explicit --prompt / config sample_prompt is used verbatim; otherwise
    # we sample a fresh chunk of validation text each time (so the continuation is judged
    # against real held-out context, like the original trainer did).
    fixed_prompt = args.prompt if args.prompt is not None else getattr(cfg, "sample_prompt", None)
    gen_tokens = int(getattr(cfg, "gen_max_new_tokens", 512))
    prompt_tokens = int(getattr(cfg, "gen_prompt_tokens", min(64, block_size // 2)))
    default_prompt = "The"

    def _make_prompt() -> str:
        if fixed_prompt is not None:
            return str(fixed_prompt)
        if val_loader is None:
            return default_prompt
        try:
            batch = val_loader(device)
            ids = batch.get("input_ids") if isinstance(batch, dict) else None
            if ids is None or ids.numel() == 0:
                return default_prompt
            n = max(1, min(prompt_tokens, int(ids.shape[1])))
            text = tokenizer.decode(ids[0, :n], skip_special_tokens=True)
            return text if text.strip() else default_prompt
        except Exception as exc:
            logger.warning("val-text prompt fetch failed (%s); using default prompt", exc)
            return default_prompt

    def _generate(tag: str) -> None:
        if args.no_generate:
            return
        was_training = model.training
        # Generation needs no gradients; release the resident grad buffers (param-sized) and
        # reserved cache first so the generation forward (full model at the prompt length) has
        # headroom on top of the training footprint (params + optimizer states + EMA).
        try:
            opt = getattr(trainer, "optimizer", None)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            prompt_text, continuation = trainer.generate_sample(
                _make_prompt(), max_new_tokens=gen_tokens, generation_mode="auto",
                do_sample=True, temperature=0.9, top_k=50, top_p=0.95,
                return_parts=True,
            )
            # Long val-text prompts: show only the tail so the continuation boundary is visible.
            shown = prompt_text if len(prompt_text) <= 300 else "…" + prompt_text[-300:]
            logger.info("%s  sample:\n%s\n[prompt …] %s\n[continues >>>] %s\n%s",
                        tag, "-" * 60, shown, continuation, "-" * 60)
        except Exception as exc:  # generation is best-effort; never abort training on it
            logger.warning("%s  generation skipped (%s)", tag, exc)
        finally:
            if was_training:
                model.train()

    def _save(tag: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"pinball_{tag}.pt")
        try:
            trainer.save_checkpoint(path)
            logger.info("saved checkpoint -> %s", path)
        except Exception as exc:
            logger.warning("checkpoint save failed (%s)", exc)

    # Step-based hooks: the trainer fires train_metrics_callback every
    # train_metrics_interval optimizer steps with a payload carrying global_step.
    # We reuse it to drive periodic generation and checkpointing mid-epoch.
    if generate_every > 0 or save_every > 0:
        nonzero = [n for n in (generate_every, save_every) if n > 0]
        interval = nonzero[0] if len(nonzero) == 1 else math.gcd(*nonzero)

        def _step_hook(payload):
            gstep = int(payload.get("global_step", 0))
            if gstep <= 0:
                return
            if generate_every > 0 and gstep % generate_every == 0:
                _generate(f"step {gstep}")
            if save_every > 0 and gstep % save_every == 0:
                _save(f"step{gstep}")

        trainer.train_metrics_callback = _step_hook
        trainer.train_metrics_interval = max(1, interval)

    logger.info("Training: objective=%s max_steps=%d epochs=%d steps/epoch=%d grad_accum=%d "
                "generate_every=%d save_every=%d mixed_precision=%s",
                objective, max_steps, num_epochs, steps_per_epoch, grad_accum,
                generate_every, save_every, use_amp)

    # Best-checkpoint + early-stopping (monitors val_loss; lower is better).
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    patience_limit = int(getattr(cfg, "early_stop_patience", 0))   # 0 = disabled
    min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))

    completed_steps = resumed_steps
    epoch = start_epoch - 1
    for epoch in range(start_epoch, num_epochs):
        steps_this_epoch = min(steps_per_epoch, max_steps - completed_steps)
        if steps_this_epoch <= 0:
            break
        train_data["steps_per_epoch"] = steps_this_epoch
        trainer.train_epoch(train_data, epoch)
        completed_steps += steps_this_epoch

        if val_loader is not None:
            sel_loss, metrics = trainer.validate(val_data)
            ppl = metrics.get("next_token_perplexity") or metrics.get("perplexity")
            logger.info("epoch %d  (%d/%d steps)  val_loss=%.4f  val_ppl=%s",
                        epoch, completed_steps, max_steps, float(sel_loss),
                        f"{ppl:.2f}" if isinstance(ppl, (int, float)) and math.isfinite(ppl) else str(ppl))
            if metrics.get("copy_samples"):
                logger.info("  copy: token_acc=%.3f span_exact=%.3f first_token_acc=%.3f",
                            metrics.get("copy_token_acc", 0.0), metrics.get("copy_span_exact", 0.0),
                            metrics.get("copy_first_token_acc", 0.0))

            # Best-checkpoint tracking + patience counter.
            if float(sel_loss) < best_val - min_delta:
                best_val = float(sel_loss)
                best_epoch = epoch
                patience = 0
                _save("best")
                logger.info("  ** new best epoch (val_loss=%.4f) -> saved pinball_best.pt **", best_val)
            else:
                patience += 1
                limit_str = str(patience_limit) if patience_limit > 0 else "off"
                logger.info("  no improvement (best val_loss=%.4f @ epoch %d; patience %d/%s)",
                            best_val, best_epoch, patience, limit_str)
                if patience_limit > 0 and patience >= patience_limit:
                    logger.info("Early stopping: no val_loss improvement for %d epochs (best=%.4f @ epoch %d).",
                                patience_limit, best_val, best_epoch)
                    if generate_every <= 0:
                        _generate(f"epoch {epoch}")
                    break

        # Per-epoch generation, unless step-based generation is already running.
        if generate_every <= 0:
            _generate(f"epoch {epoch}")

        # Reclaim the reserved-but-unallocated segments left by validation/generation so the
        # next epoch's training starts from a clean pool (avoids epoch-boundary fragmentation
        # OOMs on long-sequence runs). Cheap: once per epoch, not per step.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save("final")
    if best_epoch >= 0:
        logger.info("Best epoch: %d  (val_loss=%.4f, saved as pinball_best.pt).", best_epoch, best_val)
    logger.info("Done after %d steps (%d epochs).", completed_steps, epoch + 1)


if __name__ == "__main__":
    main()
