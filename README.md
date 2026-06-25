# Pinball, early preview (alpha release), things might break, work in progress.

**Status:** active research prototype / work in progress.  
The core hierarchy, training loop, configs, and baseline comparisons are being built out in public. Benchmarks, scaling plots, and ablations will be added as they become stable.

**Pinball** is a hierarchical graph transformer for language and general sequence modeling. Instead of a flat
sequence of tokens, it builds a multi-level graph — L0 tokens are compressed into coarser
parent nodes (L1, L2, L3, …) — and refines representations with message passing across levels.
This gives long-range reach at lower cost than dense attention, while a local windowed
attention keeps fine-grained detail at L0. We demonstrate transformer-like performance 
in small scale language modeling and DNA modeling tasks. At constant throughput and linear memory scaling behaviors.
In internal preliminary experiments on WikiText103, Pinball reaches approximately the same validation perplexity as a matched GPT2-small Transformer baseline under the same training setup (~19 PPL for both models). Full benchmark scripts, configs, and scaling plots will be added once finalized.

By decoupling local computation from global information flow, the proposed architecture scales 
approximately linearly with sequence length, enabling efficient modeling of substantially longer sequences.
These results suggest that hierarchical interaction structures provide a practical and generalizable alternative to full
attention for long-context sequence modeling.

This repository is a clean, minimal, stable implementation: a small set of clear
toggles, a config-driven trainer, and a transformer baseline for comparison.

---

# Unified Hierarchical Graph Transformer for Efficient Long-Context Sequence Modeling

A PyTorch Geometric optimized implementation of a Hierarchical Graph Transformer that efficiently processes long sequences by creating a unified hierarchical graph representation.

## Key Features

- **Unified Graph Structure**: A single connected graph for all hierarchy levels enables direct message passing between levels
- **PyTorch Geometric Optimization**: Efficient graph operations with sparse tensor operations for better performance
- **Graph Aware Positional Encoding**: Both rotary (sequence-aware) and Lagrangian (structure-aware) positional encodings
- **Level-Aware Attention**: Specially designed message passing that considers hierarchical structure
- **Efficient Information Flow**: Cross-level connections that enable direct information exchange between different levels

## Architecture

The architecture consists of four hierarchical levels connected in a unified graph:

1. **L0 (Token Level)**: Raw token representations with sequential connections
2. **L1 (Summary Level)**: Compresses token information with overlap between adjacent summaries
3. **L2 (Section Level)**: Further compresses summary level for broader context
4. **L3 (Document Level)**: Highest level that captures global document structure

Each level is processed by hierarchical transformer layers that share information through message passing across the unified graph structure.

## Install

```bash
git clone <this-repo> pinball && cd pinball
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Optional, faster local attention (attn_backend: flash) on CUDA:
pip install -e ".[flash]"
```

Requires Python ≥ 3.9, PyTorch ≥ 2.1, and PyTorch Geometric. SDPA and PyG attention backends
work out of the box; `flash` is optional.

---

## Quickstart

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidvanBruggen/pinball/blob/main/pinball_colab_run.ipynb)

1. Click the badge to open the runnable notebook in your own Google Colab account.

2. Select a GPU runtime ( e.g. a free T4) > then run all.

**Local run:**

1. Point a config at your text file (edit `text_file:` in `configs/pinball_wikitext.yaml`, or
   pass `--text-file`). Any plain UTF-8 text file works; it is tokenized and cached next to the
   file as `.pt`.

2. Train:

```bash
pinball-train --config configs/pinball_wikitext.yaml --text-file /path/to/wikitext.txt
# quick sanity run:
pinball-train --config configs/pinball_wikitext.yaml --text-file /path/to/wikitext.txt \
              --max-steps 200 --eval-every 100
```

3. Or drive it from Python:

```python
import torch
from transformers import AutoTokenizer
from pinball import build_model, PinballConfig

cfg = PinballConfig.from_yaml("configs/pinball_wikitext.yaml")
tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name); tok.pad_token = tok.eos_token
model = build_model(cfg, tokenizer=tok, vocab_size=len(tok),
                    input_mode="tokens", tie_weights=True, max_seq_len=cfg.block_size)
```

---

## Toggles

Configs accept toggles (which expand to the underlying flags) and/or the raw
underlying names. Anything you don't set falls back to default.

| Toggle | Values | Expands to | What it does |
|---|---|---|---|
| `model_type` | `pinball` \| `transformer` | — | Pinball graph model or GPT-style baseline |
| `attn_backend` | `pyg` \| `flash` \| `sdpa` | `l0_local_backend` | Local-attention kernel. `flash` needs CUDA + bf16/fp16; `sdpa` and `pyg` run anywhere |
| `qkv_sharing` | `shared` \| `separate` | `per_level_local_qkv` | Share one QKV across hierarchy levels, or give each level its own |
| `use_hqd` | `true` \| `false` | `hierarchical_query_descent_enable` | Hierarchical Query Descent: coarse→fine sparse top-k attention down the hierarchy |
| `train_mode` | `ar` \| `masked_diffusion` | `train_objective_mode` (+ `use_hybrid_masking`) | Autoregressive next-token, or BERT-style masked denoising |
| `ar_graph_causal` | `true` \| `false` | `hier_ar_enable`, `l0_ar_enable` | Make the hierarchy graph causal for AR training (no future leakage through parent edges). Recommended `true` for `ar`, `false` for masked diffusion |
| `gradient_checkpointing` | `true` \| `false` | `use_gradient_checkpointing` | Trade compute for memory in the refinement stack |
| `grad_accum` | int | `gradient_accumulation_steps` | Accumulate grads over N batches per optimizer step |
| `optimizer` | `adamw` \| `muon_hybrid` | optimizer construction | `muon_hybrid` trains 2D+ weight matrices with Muon at `learning_rate * muon_lr_mult` (RMS-matched to AdamW), 1D params (embeddings/norms/biases) with AdamW — usually learns notably faster. Tune with `muon_lr_mult`, `muon_betas`, `muon_adjust_lr_fn` |

Underlying knobs: `hidden_dim`, `num_heads`, `num_refinement_layers`,
`compression_ratios` / `overlap_ratios` (hierarchy shape), `local_attn_windows` /
`local_attn_levels` (windowed attention), `unified_refinement_cycles` (refinement passes),
`hqd_topk_l0..l3` (HQD fan-out), `use_aux_loss` (hierarchy reconstruction auxiliary loss),
`ablate_levels` (list of hierarchy levels to disable — cuts all their edges so they can't
inform the L0 prediction; for ablation studies. L0 can't be ablated. CLI: `--ablate-levels 2,3`),
`use_neighbor_sampling` + `num_neighbors` (PyG NeighborLoader sampling). Fine-grained AR
connectivity dials: `hier_ar_allow_same_time`, `enable_l0_parent_edges`,
`l0_parent_edges_bidirectional`, `ensure_l0_past_l1_edges`,
`ensure_past_hier_edges_all_levels`, `long_range_distance`.

**Training loop knobs** (config or CLI flags): `num_epochs` / `samples_per_epoch`
(`--epochs` / `--samples-per-epoch`) define epochs — validation and a generation follow each;
`generate_every` / `save_every` (`--generate-every` / `--save-every`, in batches) drive
periodic generation and checkpointing mid-run; `checkpoint_dir` (`--checkpoint-dir`) sets the
output directory.

**Memory / OOM on long sequences.** `pinball-train` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically (via `setdefault`, so a shell
value still wins) to curb allocator fragmentation — long `block_size` plus variable-size
FFT/graph buffers otherwise leave large reserved-but-unallocated gaps that OOM at epoch
boundaries; the trainer also calls `torch.cuda.empty_cache()` once per epoch after
validation/generation. If you still OOM, lower `block_size`, enable `gradient_checkpointing`,
reduce `gen_prompt_tokens`, or raise `grad_accum` while lowering `batch_size`.

### Hierarchical Query Descent (HQD)

With `use_hqd: true`, attention is computed **coarse-to-fine** instead of densely. A query
starts from conditioned l0 nodes, and crossattends with the compressed l3 representations at the top of the hierarchy and *descends*: at level L3 it keeps the `hqd_topk_l3` most
relevant parent nodes, expands only their children into L2, keeps the `hqd_topk_l2` best of
those, and so on down to the token level (`hqd_topk_l0`). Each token therefore attends to a
small, adaptively-routed set of globally-relevant nodes rather than the whole sequence — keeping
attention sparse and sub-quadratic while still reaching long-range context. The per-level
fan-out (`hqd_topk_l3..l0`) trades compute for recall, and `hqd_use_existing_zipper_projections:
true` reuses the model's existing q/k projections so HQD adds no extra parameters.

---

## Configs (`configs/`)

| Config | Purpose |
|---|---|
| `pinball_wikitext.yaml` | Canonical AR Pinball model on WikiText-103 |
| `transformer_wikitext.yaml` | GPT-style transformer baseline, same data/tokenizer |
| `pinball_masked_diffusion.yaml` | Pinball with the masked-diffusion (denoising) objective |
| `pinball_hqd.yaml` | Pinball with Hierarchical Query Descent enabled |
| `pinball_copy.yaml` | Copy-task — test that the hierarchy moves information end-to-end |

---

## Tutorials

**1. Train Pinball on WikiText-103 (AR).**
`pinball-train --config configs/pinball_wikitext.yaml --text-file /path/to/wikitext.txt`
Watch `loss` fall and periodic `val_ppl`.

**2. Transformer baseline.**
`pinball-train --config configs/transformer_wikitext.yaml --text-file /path/to/wikitext.txt`
Same data — compare perplexity vs the hierarchical model at equal parameter budgets.

**3. Masked-diffusion text.**
`pinball-train --config configs/pinball_masked_diffusion.yaml --text-file /path/to/wikitext.txt`
Trains the denoising objective; a `<mask>` token is added to the tokenizer automatically.

**4. Verify the hierarchy works (copy task).**
`pinball-train --config configs/pinball_copy.yaml --text-file /path/to/wikitext.txt --max-steps 5000`
The model must copy a marked source span across a long gap; near-perfect copy accuracy
demonstrates cross-level message passing.

**5. Switch backends / sharing / HQD.**
Edit `attn_backend` (`pyg`/`flash`/`sdpa`), `qkv_sharing` (`shared`/`separate`), or
`use_hqd` (`true`/`false`) in any Pinball config — no code changes needed.

---

## Tests

```bash
python tests/test_smoke.py          # tiny end-to-end train + eval, asserts loss decreases
# or: pytest tests/test_smoke.py -s
```

---

## Repository layout

```
src/pinball/
  config.py           # PinballConfig (friendly toggles -> underlying flags)
  cli.py              # `pinball-train` entry point (config-driven training loop)
  model/              # hierarchical graph transformer + transformer baseline + registry
  data/               # text dataloaders (WikiText / FineWeb / any text file)
  train/              # training step (AR + masked-diffusion + hierarchy aux loss)
  utils/              # PyG compatibility, schedulers
configs/              # curated YAML configs
tests/                # smoke test
```

---

## License & Citation

Pinball is released under the **GNU General Public License v3.0** (see [`LICENSE`](LICENSE)).

If you use Pinball in your research, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub
shows a "Cite this repository" button):

> van Bruggen, D. *Pinball: A Hierarchical Graph Transformer for Efficient Long-Context
> Sequence Modeling* (2026).
