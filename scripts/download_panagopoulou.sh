#!/usr/bin/env bash
# Fetch the Panagopoulou bistable image zip + annotations from Google Drive.
# The images are NOT in the upstream GitHub repo — they live on Drive behind
# gdown IDs hardcoded in the original notebooks. See the upstream Panagopoulou
# et al. repo (https://github.com/artemisp/Bistable-Illusions-MLLMs) for provenance.
#
# Requires: the aspect-seeing conda env active (provides gdown).
# Usage:    source scripts/activate.sh && bash scripts/download_panagopoulou.sh

set -euo pipefail

DATA_DIR="${ASPECT_SCRATCH:?export ASPECT_SCRATCH or source scripts/activate.sh}/data/panagopoulou"
IMAGES_ZIP_ID="1L8Tn1bK_I_0c9YxLlXQuJCkTDlxqX90c"
ANNOTATIONS_ID="1JXQogZHvQbSEU4GwuSAyIB0WPDGMb49y"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found — is the conda env active? (source scripts/activate.sh)" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [[ ! -f bistable_images.zip ]]; then
    echo "==> downloading bistable_images.zip"
    gdown "$IMAGES_ZIP_ID" -O bistable_images.zip
else
    echo "==> bistable_images.zip already present, skipping"
fi

if [[ ! -d images ]]; then
    echo "==> extracting"
    mkdir -p images
    unzip -q bistable_images.zip -d images/
else
    echo "==> images/ already unpacked, skipping"
fi

if [[ ! -f bistable_dataset.json ]]; then
    echo "==> downloading bistable_dataset.json"
    gdown "$ANNOTATIONS_ID" -O bistable_dataset.json
else
    echo "==> bistable_dataset.json already present, skipping"
fi

echo "==> done"
echo "    images:      $DATA_DIR/images/"
echo "    annotations: $DATA_DIR/bistable_dataset.json"
ls "$DATA_DIR/images/" 2>/dev/null | head -5 || true
