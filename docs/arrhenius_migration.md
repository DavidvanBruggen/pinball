# Arrhenius (NAISS/EuroHPC) migration plan

Status: **planning**. Nothing installed yet. Opened 2026-09-16.

Target system: Arrhenius, NAISS + EuroHPC, run by NSC.
382 GPU nodes x 4 NVIDIA GH200 superchips (72-core ARM Grace + 96 GB HBM3 + 128 GB LPDDR
per module), 4x Slingshot 200 Gb/s, 1.8 TB local NVMe per node.
Docs: https://www.naiss.se/resource/arrhenius/

---

## Why this is worth doing

Every DNA arm currently runs sequentially on one Blackwell at ~10 h each, so the queue
(PC arm, noprenorm, post-fix transformer at 1024, 4M) is weeks of wall-clock. **One Arrhenius
node runs four arms concurrently.** The post-fix transformer baseline in particular has been
hardware-blocked for the whole PC investigation.

Secondary: 96 GB HBM per GPU against the ~38 GiB the 32768-block arm uses on the Blackwell
means the 4M-context arm stops being memory-gated.

---

## Verified facts (do not re-derive)

**Architecture.** GH200 is **aarch64**. The local machine is x86_64. A container image is
architecture-specific, so the existing conda env **cannot** be converted or shipped — the image
must be built natively on Arrhenius. Only the *version spec* transfers.

**Local version spec to reproduce:**

| | ChromScape-bw | pinball-cs / pinball-bw |
|---|---|---|
| python | 3.11.10 | 3.11 |
| torch | 2.7.0+cu128 | 2.11.0+cu130 |
| triton | 3.3.0 | 3.6 |
| flash-attn | 2.8.3 | (bw wheel is sm_120-only) |
| other | torch-geometric 2.7.0, transformers 4.44.2, numpy 1.26.4, scipy 1.17.0 | |

**Who actually needs the `flash-attn` package** — the same `attn_backend: flash` key means
different things on different model types:

- **pinball arms: YES.** All 14 `configs/pinball_dna_*.yaml` set `attn_backend: flash`, commented
  `alias -> l0_local_backend`. That routes to the L0 local window attention in
  `src/pinball/model/layers/hierarchical_message_passing.py`, which imports `flash_attn` (FA2) or
  `flash_attn_interface` (FA3) and probes for flash's `window_size` parameter. The sliding window
  is what makes local attention O(N) rather than O(N^2). **This is a hard dependency.**
- **pinball transformer baseline: NO.** `transformer_baseline.py:198` maps
  `attn_backend in {auto, flash, sdpa}` onto `F.scaled_dot_product_attention` and reports
  `backend_used = "sdpa"`. No `flash_attn` import. Ships inside the torch wheel.
- **ChromScape borzoi host: NO.** The live config module
  `ChromScape_config_PINBALL_NO_FLASH_1024dim` sets `flashed = False`, and `FlashAttention` in
  `pytorch_borzoi_transformer_basenji_small_rotary_dim_flash.py` imports `flash_attn` *inside*
  `__init__` (line 153), so with `flashed=False` the package is never imported.

Consequence: converting the transformer to flex would **not** remove the flash dependency — the
transformer was never the consumer. Dropping flash-attn entirely would require moving pinball's
L0 local windows onto flex, and the text measurements argue against that (the flex path lacks
attention dropout; additive beat flex late-run in two matched pairs).

**FA3 becomes available for the first time.** `hierarchical_message_passing.py` gates
FlashAttention-3 to `cap[0] == 9` (Hopper-only; on the wrong arch the launch *aborts the process*
rather than raising). Neither local GPU is Hopper (sm_89, sm_120), so that branch has never
executed. GH200 is sm_90 — its intended target, on the hot local-window path.

**JIT/triton risk is low.** sm_90 is Hopper (2022) and supported by every triton in use. The
Blackwell `ptxas-blackwell` symlink hack was needed because sm_120 was *newer* than triton 3.3;
that does not recur here. Likewise `torch.compile(flex_attention)` failing with
`PassManager::run failed` is an sm_120 bug and should not appear on sm_90.

**No compiled PyG extensions needed.** `hierarchical_message_passing.py:28` treats
`torch_scatter` as an optional fallback and prefers PyG's native scatter. `torch-geometric` is
pure Python.

**TensorFlow is on the critical path but CPU-only.** `eval_specificity.py:169` imports
`TFRecord_to_Pytorch_parallel_loader.py`, which does `tf.config.set_visible_devices([], 'GPU')`
at line 8 and uses only `tf.data` (Options, AUTOTUNE, OutOfRangeError). A CPU-only aarch64 TF
wheel suffices — CUDA-enabled TF for ARM does not exist. This also justifies keeping
`numpy==1.26.4`.

**Login is password + 2FA**, no SSH keys for NAISS accounts. There is no `~/.ssh/config` and no
keypair on the local machine yet.

**Quotas.** `$HOME` = 30 GiB and **1 million files**. A conda env is 200k-500k files; two would
blow the inode limit. An Apptainer `.sif` is one file. Project storage is
`/nobackup/proj/disk/<project>`; check with `storagequota`.

---

## Decisions taken

1. **Container (Apptainer), not conda.** Driven by the 1M-file inode limit, by flash-attn being a
   hard dependency best taken prebuilt, and by NSC supporting Apptainer directly.
2. **Build the image on Arrhenius**, inside `interactive -A <project>`. Cross-architecture build
   is not possible.
3. **Do not restructure attention to avoid flash-attn.** Install it.
4. **Do not transfer Claude sessions.** Drive the cluster from the local session over an SSH
   ControlMaster socket; rsync `CLAUDE.md` / `AGENTS.md` / memory instead.

## Open decisions

- **cu128 vs cu130.** The DNA stack runs on ChromScape-bw (torch 2.7 / cu128); flex needs
  pinball-cs (torch 2.11 / cu130). The site's recommended GPU env is
  `GPU/buildenv-nvhpc/25.9-cu13.0`, which matches cu130. Collapsing both into one torch-2.11
  image would end the split and give flex + scoring in one environment for the first time — but
  the ChromScape side (TF-cpu, PyG, the borzoi CNN, BN scoring) has never been run against
  torch 2.11 and must be validated. **Decide deliberately, not by accident.**
- NGC arm64 base image vs a site-provided image/uenv. Check NSC docs for a recommended GH200
  PyTorch image before defaulting to NGC.
- How many TB of TFRecords must move. If large, start the transfer early — it parallelises with
  everything else.

---

## Phases

### Phase 0 — recon (no installs)
- [ ] Run `scratchpad/cluster_probe.sh` on a **compute** node (`interactive -A <project>`).
      Needed answers: `uname -m`, compute cap, `apptainer` present, outbound network reachable
      (some sites are gated -> offline wheelhouse plan), where `$SCRATCH`/project storage points.
- [ ] `storagequota` — confirm project storage size and inode budget.
- [ ] `ml avail` for cuda / python / apptainer versions.

### Phase 1 — path de-hardcoding (architecture-independent, do it locally, blocks everything)
- [ ] `PINBALL_ROOT` / `CHROMSCAPE_DATA` / `CHROMSCAPE_CKPT` env vars, resolved as
      `os.environ.get(..., <current default>)` so local behaviour stays bit-identical.
- [ ] 23 sites in `pinball/configs/*.yaml` (data + checkpoint dirs).
- [ ] 127 sites across ~40 `ChromScape/ChromScape/bin/*.py`, including
      `build_pinball(cfg_path="/home/david/Projects/pinball/configs/...")` in the 3kernel model.
- [ ] The training notebooks under `ChromScape/Notebooks/`.

### Phase 2 — SSH + editor
- [ ] `~/.ssh/config` with `ControlMaster auto` / `ControlPath ~/.ssh/cm-%r@%h:%p` /
      `ControlPersist 8h`. Without multiplexing, every VS Code connection re-prompts for 2FA.
- [ ] `ssh -fN arrhenius` once per day (interactive; password + verification code).
- [ ] VS Code `remote.SSH.serverInstallPath` pointed at project storage, **not** `$HOME`.
- [ ] Kernel on a **compute** node, either:
      (a) Jupyter inside an sbatch job + `ssh -fN -L 8888:<node>:8888 arrhenius` (reuses the
          socket, no 2FA) + VS Code "Existing Jupyter Server"; or
      (b) Remote-SSH with `ProxyJump arrhenius` straight to an allocated node, if NSC permits it.

### Phase 3 — container
- [ ] Write the `.def` recipe. Base: NGC arm64 PyTorch (or site image).
- [ ] Layer: torch-geometric, transformers, datasets, pytorch_optimizer, numpy==1.26.4, scipy,
      pyyaml, tqdm, h5py, pybigwig, pysam, **tensorflow-cpu**, ipykernel.
- [ ] flash-attn: **check for a prebuilt `linux_aarch64` wheel first** — GH200 + flash-attn is a
      very common pairing. Only build from source as a fallback, with `MAX_JOBS=16` (not
      `nproc`; each nvcc job is GBs) and `TORCH_CUDA_ARCH_LIST="9.0"`, inside an allocation.
- [ ] `pip install -e .` both repos against a clone on project storage.
- [ ] Store the `.sif` on `/nobackup/proj/disk/<project>` — one file, no inode cost.

### Phase 4 — validation before any real run
- [ ] `pytest tests/ -q` inside the container.
- [ ] Confirm the flash backend actually resolved: the resolver logs which backend it picked.
      Expect `fa2` (or `fa3`); a silent fall-through to eager invalidates every speed number.
- [ ] `flex_union_failed_modules == 0` on the first eval row. The failure mode is **silent** on
      any hardware, and a failed flex union silently turns the flexhier arm into the 3kernel arm.
- [ ] One short DNA run, scored under **both** BN modes, against a known local checkpoint.
      Numerics will not be bit-identical across architecture; establish the tolerance before
      trusting cross-machine comparisons.

### Phase 5 — production
- [ ] `scratchpad/four_arms.sbatch` — 4 arms, one per superchip.
      Traps encoded there: **NUMA-bind each task** (4 GH200 = 4 NUMA domains; unbound dataloaders
      cross domains and lose most of the LPDDR bandwidth), **stage TFRecords to node-local NVMe**
      (Lustre + many small reads is the classic dataloader stall, x4 concurrent readers), and
      **write checkpoints to shared storage** — node-local NVMe is wiped at job end.
- [ ] Per-arm checkpoint directories. See the `smoke-runs-clobber-checkpoints` memory: a config
      run always writes into its own `checkpoint_dir` on exit, even after two steps.
- [ ] Then benchmark FA3 against FA2 on the local-window path.

---

## Artifacts already written

- `scratchpad/cluster_probe.sh` — run on a compute node; reports everything Phase 0 needs.
- `scratchpad/four_arms.sbatch` — 4-arms-on-one-node skeleton (config names, account and
  partition still TODO).

Both are in the session scratchpad and should be copied somewhere durable before the session ends.

---

## Update 2026-09-17 — RNA data is likely a new requirement

The window-only negative control showed the hierarchy buys ~0.002 specificity on 500 kb ATAC while
demonstrably moving distal information (100x the control's block-shuffle effect at 240-256 kb).
The reading is that the ATAC objective does not need the hierarchy, so the next real test is RNA,
where expression level depends on distal enhancers. See the `dna-hierarchy-vs-window` memory.

Consequences for this migration:

- **Local disk cannot host a new multi-modal dataset.** `/mnt/Data_Storage_Disk1` is at 100%
  (156 GB free of 19 TB) and `/` at 92% (291 GB free); only `/media/david/SSD_Data` has room
  (1.6 TB). This makes the cluster the venue for the RNA work, not a convenience.
- **Add RNA/multi-modal data to the Phase 0 transfer estimate.** The existing question "how many TB
  of TFRecords must move" now has a second dataset behind it, possibly fetched on the cluster
  directly rather than transferred.
- `/home/david/Projects/.../Borsenji_CancerexpertActiRNA` is 377 GB of 198 checkpoints from
  January 2025, not data — the obvious reclaim if local headroom is ever wanted.

---

## Update 2026-09-23 — FA4 support, and a dropout blocker for FA3/FA4

**FA3 and FA4 have no attention dropout; FA2 does.** Both glob400 configs set `dropout: 0.1`,
which the L0 flash windows use. Before this update, on a GH200 the picker chose FA3
(Hopper), its smoke test passed (it runs at dropout 0), and the **first training step
failed**: `attention_forward` raised "does not support dropout" and `_flash_win_lse` hit a
`TypeError`. Without FA2 installed, the picker instead fell to SDPA with only a warning.

New keys (all default to the old behaviour; verified bit-identical, max|diff| 0.000e+00,
DNA + text glob400, eval and train):

| key | values | meaning |
|---|---|---|
| `flash_impl` | `auto` \| `fa2` \| `fa3` \| `fa4` | L0 window kernel. `auto` = fa3 on Hopper else fa2 (historical). Env `PINBALL_FLASH_IMPL` overrides. Process-wide. |
| `flash_nodropout_mode` | `error` \| `token_v` | what `dropout > 0` means on fa3/fa4. `token_v` = the flex path's token-wise V dropout — **not the same regulariser as fa2**, so a token_v arm is not comparable to the existing arms as-is. Env `PINBALL_FLASH_NODROPOUT`. |
| `local_pack_flex_backend` | `triton` \| `flash` | kernel under the flex union. `flash` = torch's FA4 CuTe template (`kernel_options BACKEND="FLASH"`, torch ≥ 2.10ish, sm_90/sm_100). Probed eagerly before compile; any failure → Triton flex with an ERROR banner (same function, speed only). |

Options for the cluster, in order of comparability with existing results:
1. `flash_impl: fa2` + `local_pack_flex_backend: flash` — keeps per-edge dropout on the L0
   windows, takes FA4 only under flex. Needs FA2 built for sm_90 in the image.
2. `flash_impl: fa4` + `flash_nodropout_mode: token_v` + flex `flash` — fastest, but a new
   regulariser on the L0 windows. Run it as its own arm.
3. `dropout: 0.0` — also a new arm.

**Verify before any real run:** `python scripts/check_fa4.py` on a GPU node. It checks the
picker resolves fa4, fa4 windows vs dense SDPA (fwd + all grads), LSE under no_grad (the
eval-time lse merge needs it; older FA4 builds return None there), flex FLASH vs TRITON on a
pinball-shaped mask with timing, and the real DNA config: `flex_union_status()` must show
`flash_live_modules > 0` and `flash_failed_modules == 0`.
