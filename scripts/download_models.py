"""Pre-download HF model snapshots into the project-scoped models dir.

Run once after the conda env is built:
    source scripts/activate.sh
    python scripts/download_models.py

Stops on any 403 (gated-model license not accepted). Re-runs are cheap —
snapshot_download checks local_dir and resumes.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from aspect_seeing.paths import MODELS_DIR

MODELS_ROOT = MODELS_DIR

# Ordered — stop on first 403. None of these should be gated in April 2026,
# but we defend anyway.
MODELS = [
    ("openai/clip-vit-large-patch14-336", "clip-vit-large-patch14-336"),
    ("llava-hf/llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-7b-hf"),
    ("lewington/CLIP-ViT-L-scope",       "CLIP-ViT-L-scope"),
    ("Qwen/Qwen3-8B",                    "Qwen3-8B"),
]


def fetch(repo_id: str, dirname: str) -> Path:
    target = MODELS_ROOT / dirname
    target.mkdir(parents=True, exist_ok=True)
    print(f"==> {repo_id} -> {target}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
    )
    # size report
    size_bytes = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"    done ({size_bytes / 1e9:.2f} GB)", flush=True)
    return Path(path)


def main() -> int:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for repo_id, dirname in MODELS:
        try:
            fetch(repo_id, dirname)
        except GatedRepoError as e:
            print(
                f"!! 403 on {repo_id}: {e}\n"
                f"   Accept the license at https://huggingface.co/{repo_id} "
                f"with an HF_TOKEN account, then re-run.",
                file=sys.stderr,
                flush=True,
            )
            return 2
        except RepositoryNotFoundError as e:
            print(f"!! repo not found: {repo_id}: {e}", file=sys.stderr, flush=True)
            return 3
    print("==> all models downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
