#!/usr/bin/env bash
# Idempotent bootstrap for the aspect-seeing project.
# Creates a conda env under ${ASPECT_SCRATCH}/envs/aspect/, installs
# env/requirements.txt (including pinned torch cu124 via extra index), then
# installs this repo in -e mode.
#
# Re-running is safe: conda create is skipped if env exists, pip installs are
# idempotent. To force a clean rebuild: rm -rf $ENV_PREFIX, then re-run.
#
# Override CONDA_BIN if `conda` is not on PATH, and ASPECT_SCRATCH to place the
# env/caches on fast/large storage.

set -euo pipefail

# Repo root, inferred from this script's location.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Scratch root. Defaults to <repo>/scratch (mirrors paths.py).
ASPECT_SCRATCH="${ASPECT_SCRATCH:-${REPO_DIR}/scratch}"
ENV_PREFIX="${ASPECT_SCRATCH}/envs/aspect"
CONDA_BIN="${CONDA_BIN:-conda}"
PY_VERSION="3.10"
TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu124"

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
    echo "ERROR: conda not found (CONDA_BIN='$CONDA_BIN')." >&2
    echo "       Install conda/miniforge, or set CONDA_BIN to its absolute path." >&2
    exit 1
fi

echo "==> bootstrap.sh starting"
echo "    repo:         $REPO_DIR"
echo "    env prefix:   $ENV_PREFIX"

# Project-scoped caches so nothing lands in $HOME.
export HF_HOME="${ASPECT_SCRATCH}/.cache/huggingface"
export PIP_CACHE_DIR="${ASPECT_SCRATCH}/.cache/pip"
export TMPDIR="${ASPECT_SCRATCH}/tmp"
export WANDB_DIR="${ASPECT_SCRATCH}/wandb"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$WANDB_DIR" \
         "${ASPECT_SCRATCH}/envs" "${ASPECT_SCRATCH}/models" \
         "${ASPECT_SCRATCH}/data" "${ASPECT_SCRATCH}/logs" \
         "${ASPECT_SCRATCH}/outputs/images" \
         "${ASPECT_SCRATCH}/outputs/activations" \
         "${ASPECT_SCRATCH}/outputs/features" \
         "${ASPECT_SCRATCH}/outputs/captions" \
         "${ASPECT_SCRATCH}/outputs/figures"

# 1. Create conda env if missing.
if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "==> conda env already exists, skipping create"
else
    echo "==> creating conda env (python ${PY_VERSION}) at ${ENV_PREFIX}"
    "$CONDA_BIN" create -y -p "$ENV_PREFIX" "python=${PY_VERSION}" pip
fi

PIP="${ENV_PREFIX}/bin/pip"

# 2. Upgrade pip tooling.
echo "==> upgrading pip/setuptools/wheel"
"$PIP" install --upgrade pip setuptools wheel

# 3. Single pip resolve: torch pins + all other deps together, with
#    --extra-index-url so cu124 torch wheels are found alongside PyPI.
echo "==> installing env/requirements.txt (torch pinned, cu124 extra index)"
"$PIP" install \
    --extra-index-url "$TORCH_EXTRA_INDEX" \
    -r "${REPO_DIR}/env/requirements.txt"

# 4. Editable install of this repo's package.
if [[ -f "${REPO_DIR}/pyproject.toml" ]]; then
    echo "==> pip install -e (editable) this repo"
    "$PIP" install -e "$REPO_DIR"
else
    echo "==> no pyproject.toml yet; skipping editable install"
fi

echo "==> bootstrap.sh done"
echo "    next: source scripts/activate.sh  (in a GPU srun session for model work)"
