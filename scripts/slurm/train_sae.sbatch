#!/bin/bash
# Train our own TopK SAE on cached CLIP-L/14 layer-22 activations.
#
# Submit with your cluster's account and GPU partition, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/train_sae.sbatch
# (#SBATCH directives cannot expand environment variables, so account/partition
#  are given on the command line.)
#
# Resume-safe: scripts/train_sae.py writes a rolling 100k-step checkpoint
# window. On requeue the sbatch re-invokes the script with --resume-from
# pointing to the most recent checkpoint.

#SBATCH --job-name=train-sae-clipL22
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail

echo "==> host=$(hostname) job=${SLURM_JOB_ID:-?} date=$(date)"
nvidia-smi -L || true

REPO_DIR="${ASPECT_REPO:?export ASPECT_REPO (see README)}"
ASPECT_SCRATCH="${ASPECT_SCRATCH:?export ASPECT_SCRATCH (see README)}"
ENV_PREFIX="${ASPECT_ENV:-${ASPECT_SCRATCH}/envs/aspect}"
OUT_DIR="${ASPECT_SCRATCH}/models/own-sae-clip-L14-layer22"

export ASPECT_SCRATCH
export HF_HOME="${ASPECT_SCRATCH}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="${ASPECT_SCRATCH}/.cache/pip"
export TMPDIR="${ASPECT_SCRATCH}/tmp"
export WANDB_DIR="${ASPECT_SCRATCH}/wandb"
export PATH="${ENV_PREFIX}/bin:$PATH"

# Pass HF + W&B secrets through from submit-time environment (if present)
# or source the local overrides file.
if [[ -z "${WANDB_API_KEY:-}" && -f "${REPO_DIR}/scripts/activate.local.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/scripts/activate.local.sh"
fi

module load cuda/12.6.3 2>/dev/null || module load cuda/12.4.1 2>/dev/null || true

mkdir -p "$OUT_DIR"

# Resume from latest rolling checkpoint, if any
RESUME_ARG=""
LATEST_CKPT=$(ls -1t "$OUT_DIR"/checkpoint_step*.pt 2>/dev/null | head -n 1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "==> resuming from $LATEST_CKPT"
    RESUME_ARG="--resume-from $LATEST_CKPT"
fi

# Pin wandb run id across preemptions so all chunks log to the same run.
WANDB_RUN_ID_FILE="${OUT_DIR}/wandb_run_id.txt"
if [[ -f "$WANDB_RUN_ID_FILE" ]]; then
    export WANDB_RUN_ID="$(cat "$WANDB_RUN_ID_FILE")"
    echo "==> WANDB_RUN_ID=$WANDB_RUN_ID (resumed)"
fi

echo "==> launching train_sae.py"
python -u "${REPO_DIR}/scripts/train_sae.py" \
    --wandb-project aspect-seeing-sae-training \
    --wandb-name own-sae-v1 \
    --total-steps 500000 \
    --checkpoint-every 100000 \
    --val-every 5000 \
    --log-every 200 \
    $RESUME_ARG

echo "==> done date=$(date)"
