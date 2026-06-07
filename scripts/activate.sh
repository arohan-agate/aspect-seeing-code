# Source this file to set up a session for the aspect-seeing project.
# Usage:   source scripts/activate.sh
#
# - Derives the repo root from this script's location.
# - Sets all cache/tmp/wandb env vars under the scratch root (ASPECT_SCRATCH).
# - Loads cuda/12.4.1 (best-effort; ignored where modules are unavailable).
# - Activates the conda env at $ASPECT_ENV, if ASPECT_ENV is set.
# - Sources scripts/activate.local.sh if present (gitignored; holds HF_TOKEN etc.)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "activate.sh: must be sourced, not executed."
    exit 1
fi

# Repo root, inferred from this script's location (mirrors paths.py).
ASPECT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ASPECT_REPO

# Scratch root. Defaults to <repo>/scratch (mirrors paths.py); on a cluster,
# point ASPECT_SCRATCH at fast/large storage with plenty of quota.
export ASPECT_SCRATCH="${ASPECT_SCRATCH:-${ASPECT_REPO}/scratch}"

export HF_HOME="${ASPECT_SCRATCH}/.cache/huggingface"
export HF_DATASETS_CACHE="${ASPECT_SCRATCH}/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="${ASPECT_SCRATCH}/.cache/huggingface/hub"
export PIP_CACHE_DIR="${ASPECT_SCRATCH}/.cache/pip"
export TMPDIR="${ASPECT_SCRATCH}/tmp"
export WANDB_DIR="${ASPECT_SCRATCH}/wandb"
export WANDB_CACHE_DIR="${ASPECT_SCRATCH}/.cache/wandb"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
         "$PIP_CACHE_DIR" "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# CUDA module (best-effort; skip silently if `module` is unavailable).
if command -v module >/dev/null 2>&1; then
    module load cuda/12.4.1 2>/dev/null || true
fi

# Conda env activation. Only if ASPECT_ENV is set; otherwise the user manages
# their own environment. Prefer `conda activate`; fall back to PATH export.
if [[ -n "${ASPECT_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        conda activate "$ASPECT_ENV" 2>/dev/null || \
            export PATH="${ASPECT_ENV}/bin:$PATH"
    else
        export PATH="${ASPECT_ENV}/bin:$PATH"
    fi
fi

# Local-only overrides (gitignored): HF_TOKEN, WANDB_API_KEY, etc.
if [[ -f "${ASPECT_REPO}/scripts/activate.local.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ASPECT_REPO}/scripts/activate.local.sh"
fi

echo "aspect-seeing env ready."
echo "  ASPECT_REPO    = $ASPECT_REPO"
echo "  ASPECT_SCRATCH = $ASPECT_SCRATCH"
echo "  ASPECT_ENV     = ${ASPECT_ENV:-<unset; using current environment>}"
echo "  python         = $(command -v python 2>/dev/null || echo 'not found')"
echo "  HF_HOME        = $HF_HOME"
