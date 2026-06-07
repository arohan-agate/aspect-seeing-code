"""Cache OpenAI CLIP-L/14-336 layer-22 residual-stream activations on a training
corpus, for later own-SAE training (per Phase 0 SAE-quality flag).

Stores one memmap float16 .npy shard per SHARD_SIZE images, shape
(N, 576, 1024) — patch tokens only (CLS dropped, matching LLaVA's
vision_feature_layer=-2). Progress is checkpointed per-tar so preemption
can resume cleanly.

Default corpus: CC3M at $ASPECT_SCRATCH/data/cc3m (332 tars, ~3.3M
image-text pairs; we can stop early). Batch size 128 on A100.

Size ceiling: --max-output-gb enforces a hard stop if the cache directory
size exceeds the budget (checked after each tar). Default 500 GB.
200K images ≈ 236 GB at (576, 1024, fp16).

Run via scripts/slurm/cache_activations.sbatch.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterator

from aspect_seeing.paths import ASPECT_SCRATCH, DATA_DIR, MODELS_DIR

CLIP_PATH = MODELS_DIR / "clip-vit-large-patch14-336"
OUT_DIR = ASPECT_SCRATCH / "activations" / "clip-L14-layer22"
DEFAULT_TARS_DIR = DATA_DIR / "cc3m"
PROGRESS_PATH = OUT_DIR / "progress.json"
KEYS_PATH = OUT_DIR / "keys.jsonl"

# CLIP ViT-L/14-336: 24 layers, 1024 hidden, 336^2/14^2 = 24*24 = 576 patches.
# Wait — 336/14 = 24, so 24*24 = 576 patches + 1 CLS = 577 tokens per image.
# (Not the 256+CLS=257 we see at 224 resolution. At 336 res: 576+1=577.)
HIDDEN_DIM = 1024
PATCHES_PER_IMAGE = 576   # (336/14)**2 = 24*24
TOKENS_PER_IMAGE = 577    # CLS + 576 patches


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"completed_tars": [], "total_images": 0, "total_shards": 0}


def _save_progress(p: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(p, indent=2))


def _iterate_webdataset_tar(tar_path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (key, image_bytes) for every .jpg in a webdataset-format tar."""
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if member.isreg() and member.name.endswith((".jpg", ".jpeg", ".png", ".webp")):
                f = tf.extractfile(member)
                if f is None:
                    continue
                key = member.name.rsplit(".", 1)[0]
                try:
                    yield key, f.read()
                except Exception:
                    continue


def _load_image_batch(batch_bytes: list[bytes], processor):
    """Decode + preprocess a batch of image bytes via CLIPImageProcessor."""
    from PIL import Image
    imgs = []
    for b in batch_bytes:
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            imgs.append(img)
        except Exception:
            imgs.append(None)
    # Drop any that failed to decode
    keep = [i for i, x in enumerate(imgs) if x is not None]
    imgs = [imgs[i] for i in keep]
    if not imgs:
        return None, keep
    inputs = processor(images=imgs, return_tensors="pt")
    return inputs, keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars-dir", type=Path, default=DEFAULT_TARS_DIR)
    ap.add_argument("--target-images", type=int, default=200_000)
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-output-gb", type=float, default=500.0,
                    help="hard stop if cache dir exceeds this size (default 500 GB)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] clip_path={CLIP_PATH}", flush=True)
    print(f"[init] tars_dir={args.tars_dir}", flush=True)
    print(f"[init] out_dir={OUT_DIR}", flush=True)
    print(f"[init] target={args.target_images}  shard={args.shard_size}  batch={args.batch_size}", flush=True)

    import numpy as np
    import torch
    from transformers import CLIPVisionModel, CLIPImageProcessor

    print("[load] CLIP-L/14-336 bf16", flush=True)
    t0 = time.time()
    processor = CLIPImageProcessor.from_pretrained(str(CLIP_PATH))
    model = CLIPVisionModel.from_pretrained(str(CLIP_PATH), dtype=torch.bfloat16).cuda().eval()
    print(f"       loaded in {time.time()-t0:.1f}s", flush=True)

    progress = _load_progress()
    completed = set(progress["completed_tars"])
    total_images = progress["total_images"]
    shard_idx = progress["total_shards"]
    print(f"[resume] {len(completed)} tars done, {total_images} images cached, "
          f"next shard = {shard_idx}", flush=True)

    tar_paths = sorted(args.tars_dir.glob("*.tar"))
    print(f"[discover] {len(tar_paths)} tars in corpus", flush=True)

    # Shard buffer — grows until SHARD_SIZE, then flushes.
    shard_buf = np.empty((args.shard_size, PATCHES_PER_IMAGE, HIDDEN_DIM), dtype=np.float16)
    shard_keys: list[str] = []
    shard_fill = 0

    def flush_shard() -> None:
        nonlocal shard_fill, shard_idx, total_images
        if shard_fill == 0:
            return
        out_path = OUT_DIR / f"activations_{shard_idx:05d}.npy"
        # Save only the filled portion
        np.save(out_path, shard_buf[:shard_fill])
        # Append keys to keys.jsonl
        with KEYS_PATH.open("a") as f:
            for i, k in enumerate(shard_keys):
                f.write(json.dumps({"key": k, "shard": shard_idx, "row": i}) + "\n")
        size_gb = out_path.stat().st_size / 1e9
        print(f"[flush] shard {shard_idx} → {out_path.name}  "
              f"({shard_fill} imgs, {size_gb:.2f} GB)", flush=True)
        shard_idx += 1
        total_images += shard_fill
        shard_fill = 0
        shard_keys.clear()

    start = time.time()
    try:
        for tar_path in tar_paths:
            if total_images >= args.target_images:
                print(f"[stop] reached target {args.target_images} images", flush=True)
                break
            if tar_path.name in completed:
                continue

            t_tar = time.time()
            batch_bytes: list[bytes] = []
            batch_keys: list[str] = []
            n_in_tar = 0

            def _flush_batch():
                nonlocal n_in_tar, shard_fill
                if not batch_bytes:
                    return
                inputs, keep = _load_image_batch(batch_bytes, processor)
                if inputs is None:
                    batch_bytes.clear(); batch_keys.clear()
                    return
                inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs, output_hidden_states=True)
                # Layer 22 output = hidden_states[-2] (LLaVA's vision_feature_layer=-2 convention)
                layer22 = out.hidden_states[-2]           # (B, 577, 1024)
                patches = layer22[:, 1:].to(torch.float16).cpu().numpy()  # (B, 576, 1024)

                # Write into shard buffer; flush whenever it fills.
                for i, local_i in enumerate(keep):
                    if shard_fill >= shard_buf.shape[0]:
                        flush_shard()
                    shard_buf[shard_fill] = patches[i]
                    shard_keys.append(batch_keys[local_i])
                    shard_fill += 1
                    n_in_tar += 1
                batch_bytes.clear(); batch_keys.clear()

            for key, img_bytes in _iterate_webdataset_tar(tar_path):
                batch_bytes.append(img_bytes)
                batch_keys.append(key)
                if len(batch_bytes) >= args.batch_size:
                    _flush_batch()
                if total_images + shard_fill >= args.target_images:
                    break
            _flush_batch()

            # Mark tar complete even if under-target — no partial resume inside a tar.
            completed.add(tar_path.name)
            progress["completed_tars"] = sorted(completed)
            progress["total_images"] = total_images + shard_fill  # incl. unflushed
            progress["total_shards"] = shard_idx
            _save_progress(progress)
            elapsed = time.time() - t_tar
            rate = n_in_tar / max(elapsed, 1e-3)
            # Hard size ceiling — prevents runaway disk usage if the estimate is off.
            out_size_bytes = sum(p.stat().st_size for p in OUT_DIR.iterdir()
                                 if p.is_file() and p.suffix == ".npy")
            out_size_gb = out_size_bytes / (1024 ** 3)
            print(f"[tar ] {tar_path.name}  +{n_in_tar} imgs  "
                  f"({elapsed:.1f}s, {rate:.0f} imgs/s)  "
                  f"running total={total_images + shard_fill}  "
                  f"disk={out_size_gb:.1f} GB", flush=True)
            if out_size_gb > args.max_output_gb:
                print(f"!! STOP: cache dir {out_size_gb:.1f} GB exceeds "
                      f"--max-output-gb={args.max_output_gb:.0f} GB. "
                      f"Flushing partial and exiting. Re-run with higher --max-output-gb "
                      f"if this is intended.", flush=True)
                flush_shard()
                progress["total_images"] = total_images
                progress["total_shards"] = shard_idx
                _save_progress(progress)
                return 10
    except KeyboardInterrupt:
        print("[sigterm] flushing partial shard before exit", flush=True)
        flush_shard()
        progress["total_images"] = total_images
        progress["total_shards"] = shard_idx
        _save_progress(progress)
        return 130

    # Final flush
    flush_shard()
    progress["completed_tars"] = sorted(completed)
    progress["total_images"] = total_images
    progress["total_shards"] = shard_idx
    _save_progress(progress)

    total_elapsed = time.time() - start
    print(f"[done] {total_images} imgs in {shard_idx} shards, "
          f"{total_elapsed:.0f}s ({total_images / max(total_elapsed,1e-3):.0f} imgs/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
