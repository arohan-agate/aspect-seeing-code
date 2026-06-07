#!/bin/bash
# Cache OpenAI CLIP-L/14-336 layer-22 patch activations for own-SAE training.
#
# =====================================================================
# SIZE CEILING — TARGET: ~240 GB on scrubbed.
# If projected final size exceeds 500 GB, STOP and ask before continuing.
# 200K images × 576 patches × 1024 dim × 2 bytes ≈ 236 GB.
# The Python script enforces a 500 GB hard stop based on running average.
# =====================================================================
#
# Submit with your cluster's account and GPU partition, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/cache_activations.sbatch
# (#SBATCH directives cannot expand environment variables, so account/partition
#  are given on the command line.)
#
# Prefer a large-memory, non-preemptible A100 partition if available. A
# preemptible partition also works: the caching script is preemption-tolerant —
# progress.json checkpoints after each tar, so resume loses at most one
# tar's worth (~10k images) of work. Pass --requeue is set so SLURM auto-requeues.

#SBATCH --job-name=cache-clip-l22
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail

echo "==> host=$(hostname)  job=${SLURM_JOB_ID:-?}  date=$(date)"
nvidia-smi -L || true

REPO_DIR="${ASPECT_REPO:?export ASPECT_REPO (see README)}"
ASPECT_SCRATCH="${ASPECT_SCRATCH:?export ASPECT_SCRATCH (see README)}"
ENV_PREFIX="${ASPECT_ENV:-${ASPECT_SCRATCH}/envs/aspect}"

export ASPECT_SCRATCH
export HF_HOME="${ASPECT_SCRATCH}/.cache/huggingface"
export HF_DATASETS_CACHE="${ASPECT_SCRATCH}/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="${ASPECT_SCRATCH}/.cache/huggingface/hub"
export PIP_CACHE_DIR="${ASPECT_SCRATCH}/.cache/pip"
export TMPDIR="${ASPECT_SCRATCH}/tmp"
export PATH="${ENV_PREFIX}/bin:$PATH"

# CUDA module — cu126 torch; cuda/12.6.3 is the cleanest match. Fall back to
# default cuda/12.4.1 if 12.6.3 not available on this node.
module load cuda/12.6.3 2>/dev/null || module load cuda/12.4.1 2>/dev/null || true

mkdir -p "${ASPECT_SCRATCH}/activations/clip-L14-layer22"

echo "==> running cache_activations.py"
python -u "${REPO_DIR}/scripts/cache_activations.py" \
    --target-images 200000 \
    --shard-size 10000 \
    --batch-size 128 \
    --max-output-gb 500

echo "==> done  date=$(date)"
