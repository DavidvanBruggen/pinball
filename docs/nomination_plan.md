# Hierarchy nomination: state and implementation plan

Written 2026-09-30 as a hand-off (context compaction). Companion files:
`docs/related_work.md` (papers + what to borrow), `docs/hierarchy_experiments.md` (status rows),
`scripts/nomination/` (probes, see the end).

## 1. What exists (all default-off, verified)

Content-selected global slots on the flex-union path, on top of the static top-down block
(`local_pack_global_select: content`, `local_pack_global_coarse: levels`,
`local_pack_global_chunk: 128`, `local_pack_global_l0_budget: 256`).

| knob | values | what it does |
|---|---|---|
| `local_pack_global_nominator` | `key` / **`head`** | `head`: every row weighted by its earliest-closing parent (bilinear on the layer input, per level) + that parent's own weight + a per-level offset; ONE global ranking; top-K per 128-row chunk over rows closed before `c*128 - window` |
| `local_pack_global_candidates` | `l0` / **`all`** | `all`: L1 summaries compete with L0 tokens for the slots |
| `local_pack_global_nom_gate` | `raw` / **`zscore`** | gate = sigmoid(z + b_layer), z standardised over the chunk's allowed set (causal); stops the uniform-shift saturation |
| `local_pack_global_boost` | `logit` / **`key`** | `key`: no detached logit; slot key += z_j * RoPE_pos(j)(u), u per layer/head (init 0, AdamW); z differentiable through the keys |
| `local_pack_global_gumbel` | float (1.0) | Gumbel top-K exploration, training only, on the ranking only |
| `local_pack_global_gumbel_norm` | `raw` / `chunk` | `chunk`: noise in units of each chunk's weight spread (ranking = z + tau*G) |
| `local_pack_global_region_cap` / `_region_level` | int / int | at most N picks per region (ancestor at the given level); live-slot count stays geometry-only |

Invariants every variant keeps: per-sequence picks, re-picked every call; causality enforced in
the selector (live rows clamped under the chunk limit), so the flex BlockMask is pure geometry,
built once per skeleton and shared by all layers.

Other changes this cycle: `global_nominate_vec` and `nom_boost_u` routed to AdamW
(`cli._build_optimizer`); `torch.load(..., mmap=True)` for cached token files; opt-in
`longctx_diag_long_edges` (e.g. `[8192]` -> `2k-8k`, `>8k`).

## 2. What was measured (WikiText-103 @4096, d384)

- key nominator: picks below random; scorer never learned. raw-gate head: learned, then every
  gate saturated (0 or ~1) by ep27, heads frozen.
- z arm (head + zscore + logit), l0_coarse: best 24.18 by ep103, below the transformer's
  full-run best 24.53; vs transformer at matched epochs 0.963-0.969 overall; 512-2k 0.95,
  >2k 0.93, 128-511 1.03 (transformer's exact copy still wins there).
- z arm, shared weights (param parity, ~25.8M live non-emb vs 21.3M): ahead early, parity
  ~ep95, transformer ahead by ep110-119 (1.012), incl. all long buckets. The level-specific
  weights are what keep pinball ahead at 4096.
- Diagnostics (`diag_nom.py`), z arms: slots carry the mid-range (no-slots 128-2k +27-40%);
  selection beats random at 128-2k (+8-12%) but random is better overall; the detached logit
  was the distraction (removing it at eval: never -4%, 128-2k +13-17%).
- q arms (head + zscore + key boost + chunk noise), ep~50: vs z 128-511 -1..-2%, 512-2k ~0,
  >2k +2.5-5%, never +2%. Distraction gone (no-slots: never 1.00, 128-511 x1.57-1.66, overall
  x1.11-1.12); random picks +24-30% worse at 128-2k. u became a mild, near-uniform suppressor,
  NOT a per-query switch. Remaining gap = pick diversity: random/noisy picks ~3% better on
  never; nominated picks skew recent (beyond 2k: 1-6% of slots vs 16-17% random).
- Region cap enforced at eval on all four checkpoints (`eval_caps.py`): distribution moves as
  intended (L3 cap 2: 31-32% beyond 2k), never -2.3..-3.4% and >2k -2.4..-4%, but 128-2k
  +7-13%; overall never better. A per-token cap trades copy context for coverage.

## 3. Plan (ordered)

### Step 1 — partial RoPE on the far path (borrowed: HSA NoPE)  [highest priority]
Why: HSA's ablation — NoPE on the long-range path is what lets training-short / running-long
work. Pinball's static block and slots are RoPE-rotated at token positions, so far matches
depend on distances unseen in training.
Design: split each head's dims into `rope_dims` (rotated) and `nope_dims`. Local band keys
use all dims as now. For static-block and slot keys, zero the rotated part (or project it
out), so their score is q_nope . k_nope — position-free. u moves into the NoPE dims (no
rotation needed; drop the RoPE-of-u code in that mode). Knob e.g.
`local_pack_far_nope_dims: 0` (0 = off, bit-identical). Applies only on the flex prefix layout.
Check: bit-identity when off; dense reference for mixed rotated/unrotated keys; causality;
then a length-transfer eval (train 4096, evaluate 8192/16384 PPL per bucket) vs the q arm.

### Step 2 — block selection (borrowed: NSA/SSA/HSA contiguous chunks)
Why: copying needs neighbours; the cap experiment showed isolated picks force a trade between
copy context and coverage. A block carries both.
Design: nominate L1 windows (the L1 row weight already exists), unfold each chosen window into
its contiguous child tokens in the slot prefix (budget e.g. 16 blocks x 16 tokens = 256);
summaries may still compete as single slots (`candidates: all`). Block rows are contiguous in
packed order -> flex-tile friendly. Gate/boost per block (z of the window) or per token (its
own z); start per block. Region cap then applies at block level (reuse
`_region_cap` / `_nom_region_ids`, which work on any row set). Knob e.g.
`local_pack_global_unit: token | l1_block`.
Check: picks == brute force at block level; unfolded rows all < chunk limit; dense reference;
causality; pick-quality at eval (copy coverage should now beat random — single tokens could
not); then the LONGCTX A/B vs the q arm.

### Step 3 — split budget: copy slots + coverage slots (uses the region cap)
Why: the cap's never/>2k gain (~3%) is real, but it must not take copy slots.
Design: K = K_copy + K_cov. Copy part = current top-K (or blocks from step 2). Coverage part =
same weights, region-capped at L3 with cap 1-2, excluding rows already chosen. Concatenate;
geometry-only live count = K_copy_live + min(K_cov, capped count). Knobs e.g.
`local_pack_global_cov_budget: 0`, reuse `region_cap/region_level` for the coverage part.
Check: eval preview on the four existing checkpoints first (like `eval_caps.py`: e.g.
192+64); only then a training arm.

### Step 4 — summary mass calibration (borrowed: HiLS)
Why: an L1 row competes with single tokens in one softmax with an uncalibrated key; with
summaries also competing for slots this matters more.
Design: per coarse row a content-dependent bias ~ log(#children) + learned entropy term from
the pooling attention (`hier_pool_mode: attn` already computes child weights). Added to the
row's key logit via a key-space term (like u) to avoid a score_mod. Knob off by default.
Check: bit-identity off; ablate at eval; LONGCTX.

### Step 5 — PG19 long-context set (the paper result)
- 16k context, token-matched (16,384 tokens/step, batch 1), `longctx_diag_long_edges: [8192]`.
- Arms: transformer d384; shared + nomination (best variant from steps 1-4); its plain parent
  (the missing control); optionally l0_coarse + nomination.
- Give the transformer the SAME epoch budget (the WikiText comparison ended at its ep119).
- Keep K fixed across lengths (2510.17196: enforce sparsity; train/test gap) and the window
  small (~128 tokens; HSA: larger windows weaken long-range learning).
- Val = last 1% of the train file; report `data/pg19_test.txt` at the end.

### Scaling fix needed before 32k+
The per-chunk selection tensors are [B, nch, n] (quadratic: ~50 MB/layer at 32k, worse beyond).
Replace with a tiled/streaming top-K (process chunks in groups; the allowed set is a prefix
per chunk, so a running top-K over row blocks works) before any >32k run.

## 4. Probes (`scripts/nomination/`)

Run with the pinball-cs env on the Blackwell (`CUDA_VISIBLE_DEVICES=1`).
- `verify_nomhead.py` — mechanics: parent map, weights, picks vs brute force, gate values,
  flex out vs dense reference (incl. key boost), causality eval+train, grads. Env:
  `GATE=zscore BOOST=key GNORM=chunk CAP=8 CAND=all`.
- `verify_gumbel.py` — raw / chunk noise vs hand-derived ranking; scale invariance.
- `diag_nom.py CONFIG CKPT` — CE per bucket under normal / noisy / random / noslots /
  nologit (`MODES=...`), gates + slot attention mass, `QU=1` q.u readout, `DIST=1` pick
  distances, `REGCAP=`/`REGLEVEL=` overrides.
- `eval_caps.py CONFIG CKPT` — region cap enforced at eval: loss + distribution.
- `pick_quality_head.py` — copy-target coverage (a weak proxy; trust the buckets).
- `lc.py` — log parser for val ppl + LONGCTX buckets (`parse(logfile)`).

Checkpoints used for the diagnostics: `pinball/text_glob400_{l0sel,l0coarse_l0sel}_{z,q}_d384_4k/checkpoints/pinball_last.pt`.
