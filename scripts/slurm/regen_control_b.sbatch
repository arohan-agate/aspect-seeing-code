#!/bin/bash
# Regenerate Schroeder/Necker control_b images with better strategies than
# the line-art prompts that failed in review. Schroeder → photographic SDXL
# (shadow/depth cues). Necker → programmatic matplotlib 3D solid cubes
# (Lambert shading; unambiguous 'from below' by construction).
#
# Submit with your cluster's account and GPU partition, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/regen_control_b.sbatch
# (#SBATCH directives cannot expand environment variables, so account/partition
#  are given on the command line.)

#SBATCH --job-name=regen-control-b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --requeue

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
export PATH="${ENV_PREFIX}/bin:$PATH"

module load cuda/12.6.3 2>/dev/null || module load cuda/12.4.1 2>/dev/null || true

echo "==> launching regen_control_b.py"
python -u "${REPO_DIR}/scripts/regen_control_b.py"

echo "==> done date=$(date)"
