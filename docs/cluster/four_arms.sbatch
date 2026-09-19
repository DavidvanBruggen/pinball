#!/bin/bash
#SBATCH --job-name=pinball-arms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GH200 superchip
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72           # one Grace CPU per task
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j-%t.out
# TODO: --account=... --partition=...  (site-specific)

set -euo pipefail

# --- NUMA: each GH200 is its own CPU+GPU pair. Without this the 4 dataloaders
# --- fight across NUMA domains and you lose most of the LPDDR bandwidth.
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="9.0"

# --- Stage data to node-local NVMe once, shared by all 4 tasks.
LOCAL=${SLURM_TMPDIR:-/tmp}/$SLURM_JOB_ID
if [ "$SLURM_PROCID" = "0" ]; then
  mkdir -p "$LOCAL/data"
  rsync -a --info=progress2 "$CHROMSCAPE_DATA_SHARED/" "$LOCAL/data/"
fi
# crude barrier; replace with srun-level staging if the site provides one
while [ ! -f "$LOCAL/data/.staged" ]; do sleep 5; done

export CHROMSCAPE_DATA="$LOCAL/data"
# CRITICAL: checkpoints go to SHARED storage. Node-local NVMe is wiped when the
# job ends -- a crash after 20 epochs would lose everything.
export CHROMSCAPE_CKPT="$SCRATCH/pinball_ckpt"

ARMS=(
  pinball_dna_bidi_linear_flexhier_pc.yaml
  pinball_dna_bidi_linear_flexhier_noprenorm.yaml
  transformer_dna_baseline.yaml            # TODO: real name
  pinball_dna_bidi_linear_flexhier_4M.yaml # TODO: real name
)
CFG=${ARMS[$SLURM_PROCID]}
echo "[task $SLURM_PROCID] $CFG on $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# each arm gets its OWN checkpoint dir -- see smoke-runs-clobber-checkpoints
export CHROMSCAPE_CKPT="$CHROMSCAPE_CKPT/${CFG%.yaml}"
mkdir -p "$CHROMSCAPE_CKPT"

python -m pinball.cli --config "$PINBALL_ROOT/configs/$CFG"
