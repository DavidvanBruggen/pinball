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

Order revised 2026-10-04: step 0 (LapPE) first, then block selection (step 2), then decide on
step 3 from step 2's never/>2k buckets; partial RoPE builds on what LapPE shows.

### Step 0 — graph LapPE (running first)
`lap_pe_k: 32`, config `pinball_wikitext_pack_glob400_l0coarse_l0sel_q_lappe_d384_4k.yaml`,
control = the l0coarse q run (to ~ep51). `_hier_lap_pe_cpu` / `_hier_lap_pe_geometry`
(cached_batch model): eigenvectors of the hierarchy membership graph (L0 chain +
`_create_next_level`'s parent-child edges), random-walk form, unit RMS, sign-fixed, computed
ONCE at max_seq_len and sliced by (level, local index) for any shorter graph, so generation
prefixes see their training codes. Absolute, level-consistent position in the residual stream,
so far keys keep a "where" under a NoPE far path. With a cos-only basis, q.k on those dims
contains a cos w(a-b) relative term plus an absolute one.
It replaced the old PyG path in `enhanced_hierarchical_flow_gat.py`, which WAS live (the
Enhanced forward) when lap_pe_k > 0. That path had AR-filtered directed edges, unit-norm
columns, unseeded signs and a per-length basis; no config set it. KV-cache decode is guarded
off with lap_pe.
Found on the way, and fixed for all arms: the global block chose its levels from the current
sizes, so a prefix under ~1830 tokens pulled L1 into the static block. Uncompiled generation
was off by 0.50; compiled generation pads to max_seq_len and was never affected. Membership is
now decided at max_seq_len, as `_resolve_global_tier` already did; it is bit-identical at the
training length. Verified: `scripts/nomination/verify_lappe.py`.

### Step 1 — partial RoPE on the far path + far-key bias (= steps 1 and 4 together)  [BUILT 2026-10-04]
Result of step 0 first: LapPE alone was neutral. Best 26.43 vs 26.42; ep43-51 ratio ~1.000.
At eval, removing its position term improved PPL 0.6%, and the projection shrank
0.068 -> 0.057: redundant while every key is rotated. Stopped at ep56.
Knobs (layer, all default off, bit-identical vs HEAD when off):
- `local_pack_far_nope_dims: 32`: the last 32 head dims (the 16 slowest pairs, <= 0.01
  rad/token) stay unrotated for every packed row. Far keys (static block + slots) get their
  rotated dims zeroed, so a far score is q_nope . k_nope. Static rows inside the band are read
  via the band (rotated); the static clause drops them, giving exactly-once keys. u lives in
  the unrotated dims, unrotated. Supported on the static-block prefix (levels) and on static +
  per-sequence chunk slots; any other branch raises. There is no additive fallback (it raises).
- `local_pack_far_bias: level`: reserved last dim (q=1, band k=0, far k=sqrt(d)*b[h, level]).
  This is a true per-level logit bias on far keys, init 0, AdamW (`far_level_bias`). It is a
  one-hot matmul: the index lookup's backward (atomics onto a 6x4 table) cost 3x step time.
  The entropy term (HiLS) is not built: the pooling here gives no attention weights.
- `lap_pe_bias: false` + RNG-isolated lap_pe_proj: a clean LapPE A/B (0 params differ at init).
Arms: `..._q_nope_d384_4k.yaml` (nope + bias) vs the q control; `..._q_nope_lappe_d384_4k.yaml`
(+ clean LapPE) vs the nope arm. Probe: `scripts/nomination/verify_farnope.py ND BIAS`.
Compiled step time: 7.29 vs 7.06 it/s.
Open: nope dims vs bias are not separated (one arm has both, as asked). A nope-only arm
answers which part matters, if needed. Full far NoPE (all 64 dims) needs q in two copies,
i.e. head dim 128 for the call (about 2x attention FLOPs); not built.

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
Revised design (2026-10-04): the level-tag site (`local_pack_level_k_emb`, added pre-RoPE) is
the place to write it, but today's tag is NOT a bias. Its score term is q.R(d)t_L, which
depends on q and on the distance, and it is the same for every row of a level. To make it one:
take one UNROTATED dim (needs step 1's partial RoPE), set q[d*] = const and
k[d*] = b_L + beta*H_r, with b_L init log(span_L in tokens) and H_r the pooling entropy.
That gives a true per-key bias at zero kernel cost. Diagnose first: the L1-row mass vs the
summed mass of its children on existing checkpoints.
Check: bit-identity off; ablate at eval; LONGCTX.

### Step 5 — PG19 long-context set (the paper result)
CONFIGS READY 2026-10-04 (cluster, 4 GPUs), generated by `scripts/nomination/make_pg19_configs.py`
from the WikiText arms, and checked at 16k by `scripts/nomination/smoke_pg19.py`:
`pinball_pg19_16k_l0coarse_{nope, q, nope_noslots}.yaml` + `transformer_pg19_16k_d384.yaml`.
The 2x2 design isolates NoPE (nope vs q) and nomination (nope vs nope_noslots). The hierarchy
deepens to 5 coarse levels ([16,4,4,4,4]), so the 400 budget again holds 384 static rows
(L4+L5); 3 levels would need 1536 rows per query. batch_size must be set per GPU (pinball:
~20 GiB per 16k sequence, eager). The lc.py parser needs the 2k-8k / >8k buckets added before
reading these logs.
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
