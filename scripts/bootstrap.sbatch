#!/bin/bash
# Session-surviving bootstrap: runs bootstrap.sh then pre-downloads the four HF
# model snapshots.
#
# Submit with your cluster's account and GPU partition, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/bootstrap.sbatch
# (#SBATCH directives cannot expand environment variables, so account/partition
#  are given on the command line.)
#
# Lives in its own SLURM step so it survives the submitting srun disconnecting.
# CPU-only (request --gpus=0 — the env build and HF downloads don't need a GPU;
# the GPU-bound verification steps stay in Phase 0 checklist).

#SBATCH --job-name=aspect-setup
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

set -euo pipefail

echo "==> host=$(hostname)  job=${SLURM_JOB_ID:-?}  date=$(date)"
echo "==> cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}"

REPO_DIR="${ASPECT_REPO:?export ASPECT_REPO (see README)}"
ASPECT_SCRATCH="${ASPECT_SCRATCH:?export ASPECT_SCRATCH (see README)}"
ENV_PREFIX="${ASPECT_ENV:-${ASPECT_SCRATCH}/envs/aspect}"

cd "$REPO_DIR"

echo "==> running bootstrap.sh"
bash scripts/bootstrap.sh

# Set up env vars for the verification + downloads (we can't `source` activate.sh
# reliably in sbatch; just set what's needed directly).
export ASPECT_SCRATCH HF_HOME="${ASPECT_SCRATCH}/.cache/huggingface"
export PIP_CACHE_DIR="${ASPECT_SCRATCH}/.cache/pip"
export TMPDIR="${ASPECT_SCRATCH}/tmp"
export WANDB_DIR="${ASPECT_SCRATCH}/wandb"
export PATH="${ENV_PREFIX}/bin:$PATH"

echo "==> verifying imports"
python - <<'PY'
import torch, sae_lens, nnsight, transformers
print("torch         ", torch.__version__)
print("torch.cuda    ", torch.version.cuda)
print("sae_lens      ", sae_lens.__version__)
print("nnsight       ", nnsight.__version__)
print("transformers  ", transformers.__version__)
PY

echo "==> downloading models"
python scripts/download_models.py

echo "==> all done  date=$(date)"
