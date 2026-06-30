# SPDX-License-Identifier: GPL-3.0-or-later
"""Ground-truth: drive the REAL trainer.validate() on a checkpoint, mirroring cli.py, to get
the reference next-token PPL. Used to diff against the standalone long-context harness.

    CUDA_VISIBLE_DEVICES=0 python bench_validate_ref.py --checkpoint <path> --eval-batches 5
"""
import sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig
from pinball.data import create_karpathy_dataloaders
from pinball.train.trainer import EnhancedHierarchicalTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="pinball/sepqkv_gpt2_128bin/checkpoints/pinball_best.pt")
    ap.add_argument("--config", default="configs/pinball_wikitext.yaml")
    ap.add_argument("--eval-batches", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = PinballConfig.from_yaml(args.config)
    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(getattr(cfg, "tokenizer_name", "gpt2"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    block_size = int(getattr(cfg, "block_size", 1024))
    batch_size = int(getattr(cfg, "batch_size", 8))
    model = build_model(cfg, tokenizer=tok, vocab_size=len(tok), input_mode="tokens",
                        tie_weights=True, max_seq_len=block_size).to(dev)
    model.emit_features_only = True

    _, val_loader = create_karpathy_dataloaders(
        text_path=getattr(cfg, "text_file"), tokenizer=tok, block_size=block_size,
        batch_size=batch_size, val_split=float(getattr(cfg, "val_split", 0.01)),
        stream_name=getattr(cfg, "stream_name", None),
    )

    trainer = EnhancedHierarchicalTrainer(
        model, None, optimizer=None, lr_scheduler=None, tokenizer=tok, device=dev,
        train_objective_mode=str(getattr(cfg, "train_objective_mode", "ar")).lower(),
        mixed_precision=True, eval_interval=args.eval_batches,
        unified_refinement_cycles=int(getattr(cfg, "unified_refinement_cycles", 1)),
        lambda_ar_loss=1.0, lambda_masked_loss=0.0, lambda_base_ce_loss=1.0, modality="text",
        longctx_diag_every=1, longctx_diag_max_seqs=64,
    )
    trainer.load_checkpoint(args.checkpoint)

    val_data = {"get_batch": val_loader, "steps_per_epoch": args.eval_batches}
    sel_loss, metrics = trainer.validate(val_data)
    print("\n===== trainer.validate() reference =====")
    print(f"  selected_loss = {sel_loss:.4f}")
    for k in ("perplexity", "perplexity_source", "next_token_perplexity",
              "next_token_perplexity_trunc", "next_token_acc"):
        if k in metrics:
            print(f"  {k} = {metrics[k]}")


if __name__ == "__main__":
    main()
