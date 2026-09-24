# Getting more out of the fixed hierarchy — experiment plan

Opened 2026-09-24. Status per item in the table at the bottom.

## Why these experiments

What the arms to date have established (details in memory):

- **ATAC does not need the hierarchy.** Window-only matches it on specificity (0.438 vs
  0.440 @ep5) while the hierarchy moves ~100x more distal signal at 240-256 kb.
- **Specificity is by construction what the training loss does not score.**
  `PoissonMultinomialLoss` = per-track profile along positions + per-track total.
  `TrackSpecificityPearson` quantile-normalises across tracks and subtracts the across-track
  mean at each position, i.e. removes exactly both. Only distal context can separate tracks.
- **The descent path only moves with its own objective.** Downward gates 0.1 -> 0.004,
  refresh projection gains init-pinned (0.88-1.17 over 35 ep), top levels become bias
  channels. PC is the only mechanism that woke them, and PC is still the best arm (0.4613).
- **Every arm peaks and declines, with the same ceiling.** Uncropped glob400 0.4686 @ep25;
  hard-cropped 0.4679 @ep35 then 0.4541 @ep49. Overfitting memorises through the LOCAL path:
  hierarchy-carried influence on train windows fell 2.4x ep10->25 while val held.
- **Wiring changes have not moved the ceiling**: tier, lane, flex union, node/window dropout,
  loss crop — inert, or timing only.

So the bet is on *incentive* and *optimisation* for the coarse levels, not more routing.

## Experiments, in order

### A. Checkpoint weight averaging (no training)
Epoch-to-epoch swings of up to 0.05 and a peak-then-decline shape suggest the model orbits
a good basin. Uniform-average the weights of checkpoints around the peak (SWA-style).
Score in both BN modes — averaged running stats are only approximate.
- Run on the hard-crop arm: avg(30,35,40) and avg(25..45), central 6144 bins.
- **Pass:** > 0.4686 (best single checkpoint of any arm). Then add a weight EMA to the notebook.

### B. Grid-phase sensitivity (no training)
Coarse windows sit at fixed offsets, so an element is pooled whole or split depending on
where the input window starts. This is the problem Swin's shifted windows address: Swin
alternates layers between a regular window grid and one shifted by half a window, so
tokens on a border in one layer share a window in the next. Here the grid is fixed for
the whole network, so the test is at the input: roll DNA by s bp, roll the output back
by s/32 bins (targets and scoring untouched).
- Geometry: 128 bp/token; level strides L1..L6 = 1, 2, 4, 16, 64, 256 kb.
- Shifts 0, 512 bp, 2 kb, 8 kb individually, plus the average over all four (test-time
  augmentation over phases). ep35 of the hard-crop arm.
- **Read:** spread across shifts = how much the grid phase costs. TTA > shift 0 = free gain,
  and random grid-phase shifts during training become the fix (the Swin analogue).

### 1. Cross-track loss (already written, never wired)
`CrossTrackMultinomialLoss` (`ChromScape/bin/losses.py:34`): per 32 kb region, a
multinomial across TRACKS plus a Poisson on the region total. The one term that only
distal context can lower. `region_bins: 1024` is calibrated against a shuffle null
(locus x track signal 1.41x null at 32 kb, zero at single bins) — do not lower it.
- **Scale:** raw, the term is ~470x the main loss (219.7 vs 0.47 on ep35) because it sums
  counts over 1024 bins while the main loss averages per bin. The notebook divides it by
  `region_bins`, so `cross_track_weight` is a per-bin ratio (1.0 ~ 30% of the objective).
  Start at 0.25. The docstring's "AlphaGenome 0.1" on the raw term would be ~98% of the loss.
- Headroom is small: a trained model beats an equal-share-per-track prediction by only 0.2%
  of the term (the rest is sampling noise). Judge on specificity, not the logged term.
- Same term on val.
- **Readouts:** specificity envelope at ep15+, long-range hierarchy-carried influence
  (should rise), train/val influence ratio (should stay ~1).

### D. Higher LR for the hierarchy's own parameters
The refresh gates/projections barely move; they are 1-D or small and see far fewer
effective updates than the L0 path. Separate optimizer group at a multiplier.
- **Risk (user):** too high bites — gates are sigmoid-ish scalars. Start at 3x, warm the
  multiplier in with the global warmup, log the group's update/weight ratio, and check
  `scripts/refresh_write_gain.py` at ep5 against the baseline.
- Run together with 1 (incentive + speed are complementary), then ablate D if 1+D wins.

### 2. PC + glob400 (+1)
glob400 ran `hier_pc_enable: false`. The two strongest levers have never been combined.
Config-only arm.

### E. Per-track readout from the coarse nodes
Cheap first half: `hier_copredict_l0: true` (off in glob400) already bypasses the decaying
descent path — coarse summaries go straight into L0's pre-head features. But all 153
tracks still read one shared feature. The full version gives each track a learned query
that cross-attends the coarse nodes near its bin (Perceiver-IO output queries), a direct
per-track route from distal context. Do copredict first; the per-track head only if the
cross-track loss shows distal information exists but is not reaching the output.

### F. Random context length during training
Crop training windows to 128-524 kb at random. Every level then fills differently and the
global block's reach varies, so the hierarchy cannot settle into fixed POSITION-based
routes and has to route by content (a transformer would not care; this one might). Also
yields specificity-vs-context-length from one model. Needs checking against the flex
BlockMask cache (keyed per skeleton, so one compile per length bucket).

### Deferred
- **C. Masked-centre prediction** — covered by Pinball Omni's masked training; run once the
  hierarchy is learning well.
- **G. Hi-C prior on coarse attention** — soon, with the triangular Hi-C formulation (no full
  contact matrix instantiated; see the Hi-C memories).
- **Content-adaptive pooling** (dynamic chunking) — gives up the fixed hierarchy; only after
  A-F have shown what the fixed version can reach.

## Protocol for every arm
Specificity on the SAME central bins for all arms being compared
(`eval_specificity --crop-target-bins 6144` when any arm is cropped), ep15+, best-so-far
envelopes, both BN modes, `flex_union_failed_modules == 0`, arms dated after the
2026-09-14 Muon decay fix. Val loss is not a selector (it bottoms 16+ epochs early).

## Status

| item | status | result |
|---|---|---|
| A | done 2026-09-24 | **no.** avg(30,35,40) 0.4571, avg(25..45) 0.4449 vs ep35 alone 0.4679 (both BN modes). Not one basin; no weight-EMA. |
| B | done 2026-09-24 | **real, sub-1 kb.** Uncropped control ep25, shift (bp) -> spec: 0 .4684, 128 .4676, 256 .4681, 384 .4701, 512 .4708, 640 .4714, **768 .4721**, 896 .4696, 1024 .4690; 2 kb .4686, 8 kb .4681. Smooth hump with ~1 kb period (L1 stride), amplitude 0.0045 vs a noise floor of ~0.001 (0/1024/2048 and +512/-512 pairs). The TRAINING phase (0) sits near the trough; the loader only shifts +/-3 bp, so the model has only seen phase 0. Coarse grid (>= 2 kb) costs nothing. (Hard-crop arm invalid for this test: shifts move scored bins into its untrained flank.) |
| 1 | wired 2026-09-24, off | notebook `cross_track_weight` (per-bin units; raw term is ~470x the main loss, now divided by region_bins). Start 0.25. |
| D | wired 2026-09-24, off | `optimizer_utils.build_training_optimizer(hier_lr_mult=)`, notebook `hier_lr_mult`. 1.0 = identical optimizer (verified). Start 3.0. |
| B-fix: shift aug | wired 2026-09-24, off | ep20 sweep confirms the hump (peak 512 bp, 0.4691 vs 0.4667 at phase 0), peak phase moves between checkpoints -> train on all phases. `shift_aug_bp` (notebook) -> loader shifts train batches by k*128 bp, |shift| <= 512, + jitter, coverage moved with it (DNAaugment `cov_shift_bins`, before RC). Needs `loss_crop_bins = 16352` when uncropped (checked at startup). Verified: off == old (function + loader RNG replay), on: seq/cov aligned both strands. |
| 2 | text arms written 2026-09-24 | `configs/pinball_wikitext_pack_glob400_pc.yaml` = glob400 + the text PC block (from `cleaner_pack_pc`, verbatim). DNA arm not started. |
| D (text) | ported 2026-09-24 | `cli._build_optimizer` reads `hier_lr_mult` (same `HIERARCHY_PARAM_MARKERS` as ChromScape; 7.09M of 144M params on glob400). 1.0 = identical groups (verified, muon_hybrid and adamw). Arm `configs/pinball_wikitext_pack_glob400_pc_hlr3.yaml` = the PC arm + `hier_lr_mult: 3.0`, so PC and D separate: control -> PC -> PC+D. |
| WD split (text) | key added 2026-09-24, unset | `adamw_weight_decay` in `cli._build_optimizer`: text trains the AdamW group (norms, biases, 1-D gates, embeddings, head) at `weight_decay` 0.1, DNA at 1e-4 since the 09-14 fix. Unset = identical optimizer (verified). Changing it needs a fresh control + transformer; next round. |
| E (copredict) | not started | |
| F | not started | |
