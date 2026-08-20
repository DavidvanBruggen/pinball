# DNA crossover benchmark — protocol

Data: `bench_crossover_dna.tsv` (long format, one row per arm x length).

Measured 2026-08-20 on the RTX PRO 6000 Blackwell (torch `cuda:1`), batch 1,
bf16 autocast, forward + backward, no optimizer step.

## Columns
| column | meaning |
|---|---|
| `n_tokens` | L0 sequence length. Pinball's packed sequence is ~1.22x this (tokens + coarse rows). |
| `ms_per_step` | median of 8 timed steps after 4 warm-up steps |
| `peak_gib` | `torch.cuda.max_memory_allocated` over the timed steps |
| `params_m` | total parameters, millions |
| `tokens_per_s` | `n_tokens / (ms_per_step/1000)` |
| `*_ratio_vs_transformer` | that arm divided by the transformer at the same length; <1 = pinball better |

## Protocol notes that affect interpretation

* **One model per process.** Four pinball variants in one process share dynamo's
  per-code-object compile cache; the later-built ones exceed the recompile limit and
  silently fall back to eager. An earlier version of this sweep did that and understated
  pinball by ~1.8x at 16k-32k. The transformer is a different code object and was not
  affected — which is how the discrepancy was caught.
* **Gradient checkpointing ON for every arm** (all four configs specify it). Matched, and
  it is how these would actually train.
* **`pinball_full`'s `local_pack_window` is rescaled per length** to the packed length
  (`1.22N`). Left at its config value (written for 4096) it silently degrades into a
  windowed model at longer N.
* **Depth is NOT matched: transformer 15 layers, pinball 12.** This is deliberate -- a
  pinball refinement layer carries more parameters than a plain block, so 12 vs 15 is what
  makes the arms param-matched (198.6M vs 191.9M at 4096). The consequence runs in opposite
  directions for the two claims. On QUALITY the transformer gets 25% more sequential depth
  at equal params, so pinball's Pearson parity is if anything understated. On SPEED the
  extra layers make the TRANSFORMER slower, so the per-step ratios flatter pinball.
  `ms_per_layer` and `ms_per_layer_ratio_vs_transformer` give the depth-normalised view:
  the windowed arm's advantage at 32768 goes from 2.82x to 2.26x, and the crossover moves
  from ~5.9k tokens to ~8.7k. Neither framing is "the fair one" -- you can match parameters
  or depth, not both. Per-step answers "which should I train at matched params and matched
  quality"; per-layer answers "is the attention mechanism itself cheaper".
* **The transformer baseline is only param-matched at 4096.** It allocates a position
  embedding table sized by `block_size` on top of `transformer_use_rope: true`, so it grows
  198.6M -> 227.9M across this sweep while the pinball arms stay fixed. That flatters
  pinball slightly at the long end.
* Peak memory is single-model-resident. An earlier sweep held all four arms in memory at
  once and reported peaks ~2x higher; those numbers are not comparable to these.
* All pinball rows include the checkpoint-fold change (per-layer `pinball_refinement_norm`
  moved inside the checkpointed region), which cut peak memory ~20% at no time cost.

## Headline

Crossover is at ~6-7k tokens: pinball is slower below it, faster above. At 32768 the
windowed arm is 2.8x faster than the full-attention transformer per step (2.26x per layer),
and the gap is still widening. Memory runs ~15% *higher* than the transformer at 16k-32k — the win is time.
