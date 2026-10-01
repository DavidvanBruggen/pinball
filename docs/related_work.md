# Related work: hierarchical / sparse long-context attention

Collected 2026-09-30 while positioning the hierarchy-nominated slots
(`local_pack_global_nominator: head`, `local_pack_global_boost: key`). Each entry: what it
does, how it differs from pinball, and what (if anything) is worth borrowing.

## Pinball's claim, stated against these

One learned, overlapping, per-layer-refreshed hierarchy (L1-L3) that is at the same time
(a) the long-context router — its summaries are keys in the same causal softmax as the local
window — and (b) the nominator — parent->child heads give one global importance ranking over
tokens *and* summaries, trained through the task (value gate + softmax), not a dense teacher.
Nominated rows join the same softmax; each query decides via q.u (key-space boost) how much it
wants them. Linear cost: fixed-budget global block + window + fixed slot budget per chunk.

## Closest

- **Simplified Sparse Attention via Gist Tokens (SSA / H-SSA)** — arXiv 2604.20920 (2026-04).
  <https://arxiv.org/html/2604.20920>
  Gist (and meta-gist) tokens in the residual stream, attended JOINTLY with raw tokens in one
  softmax; per-query top-k chunks by q.gist-key are unfolded to raw tokens. Sub-quadratic,
  not linear: prefill O(nL + n^2/L^2). **Must be cited and compared**: it shares the
  joint-softmax hierarchy idea. Differences: per-query chunk scoring vs a hierarchy-produced
  global ranking; unfolding vs a fixed shared slot set; quadratic term vs linear.

- **Hierarchical Sparse Attention Done Right (HiLS-Attention)** — arXiv 2607.02980 (2026-07).
  <https://arxiv.org/html/2607.02980v1>
  Landmark per chunk with an entropy-calibrated compressed key; two-level hierarchical softmax
  trained end to end. Summaries are used for ROUTING only, never attended as keys.
  Borrow: calibrate a summary's key to its chunk's attention mass (log-sum-exp + entropy
  term) so summaries and tokens compete fairly in the joint softmax.

- **Every Token Counts: Generalizing 16M Ultra-Long Context (HSA)** — arXiv 2511.23319.
  <https://arxiv.org/html/2511.23319v1>
  Chunks + landmarks (bidirectional chunk encoder with CLS at layer L/2); each token retrieves
  top-k past chunks, attends each separately, fuses by retrieval score. Linear memory, 16M
  tokens. Borrow: **NoPE on the long-range path, RoPE only in the sliding window** (their
  ablation: key to length extrapolation) -> in pinball as partial RoPE (far keys keep only
  the unrotated dims); warm-up from dense-ish selection; a small window strengthens
  long-range learning; long-document data (PG19) matters.

- **Understanding and Improving Length Generalization in Hierarchical Sparse Attention
  Models** — arXiv 2510.17196 (ICLR 2026). <https://arxiv.org/abs/2510.17196>
  Three principles: expressive non-linear chunk encoder with CLS; a bypassing residual path
  for retrieved info; enforced selection sparsity in pre-training (train/test gap). 4K -> 32M
  training-free extrapolation. Borrow: keep the slot budget K fixed across lengths. Contrast
  (not borrowed): the bypass path is the opposite of pinball's joint softmax.

## Also relevant

- **NSA: Hardware-Aligned and Natively Trainable Sparse Attention** (DeepSeek) — arXiv
  2502.11089. <https://arxiv.org/pdf/2502.11089>
  Compressed + selected + sliding-window branches, separate softmaxes, gated merge; block
  selection from compressed-attention scores at kernel-tile granularity. Borrow: select
  contiguous BLOCKS (copying needs neighbours) -> nominate L1 windows and unfold them.

- **DSA / lightning indexer** (DeepSeek-V3.2). Learned per-token importance predictor
  distilled (KL) from dense attention, top-k per query. Contrast: pinball's nominator is
  trained by the task, with no dense teacher.

- **Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs** —
  arXiv 2510.24606. <https://arxiv.org/abs/2510.24606>

- **H-Transformer-1D: Fast One-Dimensional Hierarchical Attention** — arXiv 2107.11906.
  <https://arxiv.org/pdf/2107.11906>
  Hierarchical-matrix attention, log-linear; hierarchy without learned selection.

- Overview page: <https://www.emergentmind.com/topics/hierarchical-sparse-attention-hsa>

## Not yet checked in detail

Landmark attention, Routing Transformer, Quest, Hourglass / multi-scale transformers,
memory-token approaches — all query-dependent selection or hierarchy-without-nomination;
worth a pass before writing up.
