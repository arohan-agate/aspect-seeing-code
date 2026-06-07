#!/bin/bash
# Phase 4 — causal steering at CLIP layer 22 inside LLaVA.
#
# Submit with your cluster's account and GPU partition, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/phase4_steering.sbatch
# (#SBATCH directives cannot expand environment variables, so account/partition
#  are given on the command line.)

#SBATCH --job-name=phase4-steering
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00

set -euo pipefail
echo "==> host=$(hostname) job=${SLURM_JOB_ID:-?} date=$(date)"
nvidia-smi -L || true

REPO_DIR="${ASPECT_REPO:?export ASPECT_REPO (see README)}"
ASPECT_SCRATCH="${ASPECT_SCRATCH:?export ASPECT_SCRATCH (see README)}"
ENV_PREFIX="${ASPECT_ENV:-${ASPECT_SCRATCH}/envs/aspect}"

export ASPECT_SCRATCH
export HF_HOME="${ASPECT_SCRATCH}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="${ASPECT_SCRATCH}/.cache/pip"
export TMPDIR="${ASPECT_SCRATCH}/tmp"
export WANDB_DIR="${ASPECT_SCRATCH}/wandb"
export PATH="${ENV_PREFIX}/bin:$PATH"

module load cuda/12.6.3 2>/dev/null || module load cuda/12.4.1 2>/dev/null || true

echo "==> launching phase4_steering.py (high-superposition groups)"
python -u "${REPO_DIR}/scripts/phase4_steering.py" \
    --groups duck_rabbit young_old_woman hidden_face

echo "==> done date=$(date)"
