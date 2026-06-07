"""Evaluate an own-SAE checkpoint on the held-out activation shards.

Loads best.pt (half-precision weights), reconstructs the TopKSAE module,
runs reconstruction metrics on the last 2 shards of the cached activations
(the held-out val split train_sae.py reserved).

Reports per-shard and pooled MSE / SSE-per-token / explained variance /
active-feature count. Intended as the cutoff evaluation before Phase 2 v2.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from aspect_seeing.paths import ASPECT_SCRATCH, MODELS_DIR


DEFAULT_ACTS = ASPECT_SCRATCH / "activations" / "clip-L14-layer22"
DEFAULT_CKPT = MODELS_DIR / "own-sae-clip-L14-layer22" / "best.pt"


def build_sae(d_in: int, n_features: int, k: int, device: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TopKSAE(nn.Module):
        def __init__(self):
            super().__init__()
            W_enc = torch.randn(d_in, n_features) / math.sqrt(d_in)
            self.W_enc = nn.Parameter(W_enc.clone())
            self.W_dec = nn.Parameter(W_enc.t().contiguous().clone())
            self.b_enc = nn.Parameter(torch.zeros(n_features))
            self.b_pre = nn.Parameter(torch.zeros(d_in))
            self.k = k

        def encode(self, x):
            pre = x - self.b_pre
            hidden = F.relu(pre @ self.W_enc + self.b_enc)
            top_vals, top_idx = hidden.topk(self.k, dim=-1)
            out = torch.zeros_like(hidden)
            out.scatter_(-1, top_idx, top_vals)
            return out

        def decode(self, latent):
            return latent @ self.W_dec + self.b_pre

        def forward(self, x):
            latent = self.encode(x)
            recon = self.decode(latent)
            return recon, latent

    return TopKSAE().to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--acts-dir", type=Path, default=DEFAULT_ACTS)
    ap.add_argument("--val-shards", type=int, default=2)
    ap.add_argument("--batch-images", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=200,
                    help="number of (batch_images × 576) patch-token batches")
    args = ap.parse_args()

    import torch

    if not args.ckpt.exists():
        print(f"!! checkpoint missing: {args.ckpt}")
        return 1
    print(f"[1/3] loading checkpoint {args.ckpt}", flush=True)
    # weights_only=False because we saved argparse.Namespace-derived paths
    # (pathlib.PosixPath) in config — not in the unpickler's safe-globals list.
    # Source is our own training script, so this is fine.
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    print(f"       step={ckpt['step']}  d_in={cfg['d_in']}  n_features={cfg['n_features']}  k={cfg['k']}",
          flush=True)
    if "val" in ckpt:
        v = ckpt["val"]
        print(f"       checkpoint's training-time val metrics: "
              f"EV={v['ev']:.3f}  SSE/tok={v['sse_per_tok']:.2f}  "
              f"active={v['active_features']:.1f}", flush=True)

    sae = build_sae(cfg["d_in"], cfg["n_features"], cfg["k"], "cuda")
    # Load state dict; entries were saved as fp16 in best.pt — upcast for compute.
    sd = {k: v.float() for k, v in ckpt["model"].items()}
    sae.load_state_dict(sd)
    sae.eval()
    print(f"[2/3] SAE loaded onto GPU", flush=True)

    shard_paths = sorted(args.acts_dir.glob("activations_*.npy"))
    val_paths = shard_paths[-args.val_shards:]
    print(f"       val shards: {[p.name for p in val_paths]}", flush=True)
    val_shards = [np.load(p, mmap_mode="r") for p in val_paths]

    print(f"[3/3] running {args.n_batches} batches × {args.batch_images} images × 576 patches", flush=True)
    rng = np.random.default_rng(2026)
    agg = {"mse": 0.0, "sse_per_tok": 0.0, "ev_num": 0.0, "ev_denom": 0.0,
           "active": 0.0, "n_batches": 0}
    t0 = time.time()
    with torch.no_grad():
        for b in range(args.n_batches):
            si = int(rng.integers(0, len(val_shards)))
            shard = val_shards[si]
            idx = rng.integers(0, shard.shape[0], size=args.batch_images)
            imgs = np.ascontiguousarray(shard[idx])
            patches = imgs.reshape(-1, cfg["d_in"])
            x = torch.from_numpy(patches).float().cuda()
            recon, latent = sae(x)
            err = recon - x
            agg["mse"]         += (err * err).mean().item()
            agg["sse_per_tok"] += (err * err).sum(dim=-1).mean().item()
            agg["ev_num"]      += (err * err).sum().item()
            agg["ev_denom"]    += (x * x).sum().item()
            agg["active"]      += (latent > 0).float().sum(dim=-1).mean().item()
            agg["n_batches"]   += 1

    n = agg["n_batches"]
    ev = 1 - agg["ev_num"] / max(agg["ev_denom"], 1e-12)
    print()
    print("=" * 60, flush=True)
    print(f"val MSE/elem  : {agg['mse']/n:.6f}", flush=True)
    print(f"val SSE/token : {agg['sse_per_tok']/n:.2f}   "
          f"(CLIP-Scope CLS baseline ≈ 300 → we're at "
          f"{'MUCH BETTER' if agg['sse_per_tok']/n < 200 else 'similar'})", flush=True)
    print(f"val EV        : {ev:.4f}        "
          f"(CLIP-Scope CLS baseline ≈ 0.68)", flush=True)
    print(f"active/token  : {agg['active']/n:.1f}   (expected {cfg['k']})", flush=True)
    print(f"batches       : {n}  ({n * args.batch_images * 576} tokens)", flush=True)
    print(f"elapsed       : {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
