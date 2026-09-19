# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

Pinball is a hierarchical graph transformer for long-context sequence modelling. Tokens form
level 0; each higher level is a coarser set of nodes pooled from overlapping windows of the level
below, with attention running *within* levels and message passing *between* them. The research
claim being defended is **linear scaling in context length** with quality competitive against a
dense transformer.

It lives a double life, and which one you are in changes almost everything:

1. **Standalone LM** — text (WikiText/PG19) and image (MaskGIT) training via `pinball-train`
   (`src/pinball/cli.py`). Self-contained in this repo.
2. **Backbone inside ChromScape** — DNA coverage-track prediction. `from pinball import
   build_pinball` returns a feature-in/feature-out block that a CNN host model wraps. The host
   lives in `/home/david/Projects/ChromScape/ChromScape/bin/`, training is driven from Jupyter
   notebooks there, and **this repo's `cli.py` and `trainer.py` are not used at all**.

Most active research is on path 2. If a question is about specificity, coverage tracks, epochs,
or BatchNorm, it is a ChromScape-hosted DNA run.

## Repository layout

```
src/pinball/
  config.py            # PinballConfig: friendly YAML keys -> underlying flags
  cli.py               # `pinball-train` entry point (text/image only)
  model/
    hierarchical_flow_gat_cached_batch.py   # THE model. ~17k lines.
    model_registry.py                        # YAML key -> constructor kwarg threading
    transformer_baseline.py                  # matched GPT-style baseline
    hierarchy_plan.py, param_estimator.py, batched_layer_executor.py
  train/trainer.py     # training step (text/image only, ~7.9k lines)
  data/                # text + image loaders
configs/               # 52 curated YAMLs; one file per experimental arm
tests/                 # 11 test modules, CPU-runnable except the GPU smoke
scripts/               # refresh_write_gain.py (gate diagnostics), download_pg19.py
bench_*.py             # standalone benchmark harnesses at repo root
```

**Navigating the 17k-line model file.** Do not read it whole. It is organised as: module-level
helpers and constants, predictor/decoder classes, the main `HierarchicalFlowGATCachedBatch` class
(constructor ~2700-4700 threading config into attributes), geometry helpers (`_predict_level_sizes`,
`_resolve_global_tier`, `_pooled_child_window_means`), loss builders (`_compute_hier_aux_*`,
`_compute_hier_pc_loss`, `_compute_predictive_aux_loss`), then `forward`. Grep for the flag name
first — config keys are threaded by exact string and appear in `model_registry.py`, the
constructor, and the forward branch.

## Environments and GPUs

Three conda envs under `/home/david/miniforge-pypy3/envs/`:

| Env | torch | pytest | Use for |
|---|---|---|---|
| `ChromScape-bw` | 2.7.0+cu128 | **yes** | DNA runs, scoring, and **all pytest invocations** |
| `pinball-cs` | 2.11.0+cu130 | no | text/image pinball runs |
| `pinball-bw` | 2.11.0+cu130 | no | as above; its flash wheel is **sm_120 only** (dies on the 4090) |

`pinball-cs`/`pinball-bw` have **no pytest** — run tests with
`/home/david/miniforge-pypy3/envs/ChromScape-bw/bin/python -m pytest`.

**GPU index inversion — verified, easy to get wrong.** `nvidia-smi` order is the reverse of
PyTorch/`CUDA_VISIBLE_DEVICES` order:

| | smi index | CUDA_VISIBLE_DEVICES |
|---|---|---|
| RTX PRO 6000 Blackwell | 0 | **1** |
| RTX 4090 | 1 | **0** |

Always pin with `CUDA_VISIBLE_DEVICES` (torch order) and confirm with
`torch.cuda.get_device_name(0)` before a long run.

## Commands

```bash
# install (editable)
pip install -e .            # or pip install -e ".[flash]" for the flash backend

# text training
pinball-train --config configs/pinball_wikitext.yaml --text-file /path/to/wikitext.txt
pinball-train --config ... --max-steps 200 --eval-every 100      # sanity run -- SEE WARNING BELOW

# tests (always via ChromScape-bw)
/home/david/miniforge-pypy3/envs/ChromScape-bw/bin/python -m pytest tests/ -q
python tests/test_smoke.py                       # tiny end-to-end, asserts loss decreases
```

## Traps — read before running anything

These are hard-won and each has cost real time or real data.

**1. Smoke runs clobber checkpoints.** `python -m pinball.cli --config <real>.yaml` *always*
writes `best_model.pt`/`final.pt` into that config's `checkpoint_dir` on exit, even after two
steps. This destroyed a trained 1.7 GB checkpoint once. Always redirect `checkpoint_dir` to the
scratchpad for a sanity run, or use `tests/test_smoke.py` instead.

**2. `flex_union` can be a silent no-op.** Inductor's divisibility guard checks the default
config, not our `kernel_options`, so a `flexhier` arm can silently run as a duplicate of the
windowed arm with no error. `flex_union_failed_modules == 0` on an eval row is the **only** proof
the arm is real. Check it on every flex run. (Separately, `torch.compile(flex_attention)` fails
on sm_120 — a Blackwell-specific bug, not present on sm_90.)

**3. Score DNA arms at epoch 15+, as envelopes, never as single points.** Norm-stack arms swap
rank between ep5 and ep15, and adjacent epochs swing up to 0.05 on specificity. Compare
best-so-far envelopes.

**4. Take every DNA number twice — running stats and batch stats.** Roughly 60% of the
apparent epoch-to-epoch cycling is stale BatchNorm buffers, not optimisation, and arms are *not*
equally affected (pre-norm arms move ~1e-4; the noprenorm arm moved +0.0297). Reporting one
number without saying which BN mode produced it is how two arms get compared unfairly.

**5. Zero-ablation is confounded.** Zeroing a pathway conflates "carries information" with "acts
as a learned bias/gain" — refresh looked like +0.886 nats under zero-ablation and +0.017 under a
mean control. Always pair a zero-ablation with a mean or stale control.

**6. `node_spec` is a ratio.** Decompose with `rms` into content and bias before calling it
enrichment. And never read `gate_monitor()` after a control forward — it reports the last call.

**7. The Muon weight-decay fix (2026-09-14) splits the record in two.** Before it, the
`adamw_fallback` group used `muon_weight_decay` (0.1) instead of `adamw_weight_decay` (1e-4) —
1000x too strong — hitting LayerNorm scales, every bias, 1-D refresh gates, embeddings and the
output head. **Arms trained before the fix are not comparable to arms trained after it.** Always
date an arm before quoting it in a comparison.

**8. Eager bf16 is 24% off fp32 on text while compiled bf16 is 1.1% off** — compile is the
*accurate* path there. This does not generalise; on the DNA path the same check reads 0.0014.

## Conventions

**Adding a config knob.** Thread it in four places, in this order: `model_registry.py` (YAML key
-> kwarg), the model constructor (store on `self`), the forward/loss branch, and the class
docstring. Then add the config to `tests/test_instantiate_pinball.py`'s sweep if it is a new
YAML. New flags must default to **off**, and off must be **bit-identical** to the previous
behaviour — verify with a fixed seed and input, asserting `max|diff| == 0.000e+00`, before
trusting any measurement made with it on.

**Config files are experimental arms.** One YAML per arm, with a header comment recording the
hypothesis, the diff against its parent config, and the score table so far. Follow the existing
`noprenorm`/`dropout03`/`pc` headers. Do not edit a config that has a run in flight.

**Loss naming.** `mse_norm` reductions are means over nodes/offsets then over pairs, so they are
context-length invariant and lambdas transfer across block sizes. Plain `mse` on unnormalised
features is unbounded and has blown up an aux loss 0.1 -> 20; prefer `mse_norm` or `cosine`.

**Scratchpad.** Temporary scripts, probes and scoring outputs go in the session scratchpad
directory, never in the repo root. The `bench_*.py` files at root are the curated, durable ones.

## Related repositories

- **ChromScape** (`/home/david/Projects/ChromScape`, private) — the DNA host model, data pipeline
  and training notebooks. Has its own `CLAUDE.md` and `AGENTS.md`.
- `hgt_geomet` — a legacy checkout. Some ChromScape files still import `build_pinball` from it;
  those paths are stale and lack flexhier/global-block/DropNode support. Current code should
  import from this repo.
