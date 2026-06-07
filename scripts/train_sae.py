"""Train a TopK SAE on cached CLIP-L/14 layer-22 patch activations.

Architecture matches CLIP-Scope (Ewington-Pitsos & Goyal 2024):
    - input:          d_in=1024   (CLIP-L/14 hidden dim)
    - hidden:         n_features=65536  (expansion factor 16)
    - sparsity:       TopK with k=32
    - decoder:        unit-norm columns (renormalized each step)

Target: match or beat CLIP-Scope's reported baseline on OpenAI-CLIP held-out
split: mean CLS SSE ≈ 300, EV ≈ 0.68 (their LAION numbers; should be similar
or better on OpenAI-CLIP since that's LLaVA's backbone).

Data: 20 memmap shards at $ASPECT_SCRATCH/activations/
clip-L14-layer22/ (shape (N, 576, 1024) fp16 per shard, ~200K images).
Train on shards 0..N-3, validate on the last 2 shards (held-out images).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from aspect_seeing.paths import ASPECT_SCRATCH, MODELS_DIR


# ---- TopK SAE ------------------------------------------------------------

def build_sae(d_in: int, n_features: int, k: int, device: str):
    import torch
    import torch.nn as nn

    class TopKSAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Kaiming-style init for W_enc; W_dec is the transpose, unit-normed
            W_enc = torch.randn(d_in, n_features) / math.sqrt(d_in)
            self.W_enc = nn.Parameter(W_enc.clone())
            self.W_dec = nn.Parameter(W_enc.t().contiguous().clone())
            self.b_enc = nn.Parameter(torch.zeros(n_features))
            self.b_pre = nn.Parameter(torch.zeros(d_in))
            self.d_in = d_in
            self.n_features = n_features
            self.k = k

        def normalize_decoder_(self) -> None:
            with torch.no_grad():
                norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-12)
                self.W_dec.data /= norms

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            import torch.nn.functional as F
            pre = x - self.b_pre
            hidden = F.relu(pre @ self.W_enc + self.b_enc)
            top_vals, top_idx = hidden.topk(self.k, dim=-1)
            out = torch.zeros_like(hidden)
            out.scatter_(-1, top_idx, top_vals)
            return out

        def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
            return latent @ self.W_dec + self.b_pre

        def forward(self, x: "torch.Tensor"):
            latent = self.encode(x)
            recon = self.decode(latent)
            return recon, latent

    return TopKSAE().to(device)


# ---- Shard iterator ------------------------------------------------------

class ShardBatches:
    """Samples random (shard, image) pairs and yields (B*576, d_in) float32
    tensors on the device. Memory-maps shards lazily."""

    def __init__(self, shard_paths, batch_images: int, device: str, seed: int = 42):
        self.shards = [np.load(p, mmap_mode="r") for p in shard_paths]
        self.sizes  = [s.shape[0] for s in self.shards]
        self.batch_images = batch_images
        self.device = device
        self.rng = np.random.default_rng(seed)

    def sample(self):
        import torch
        si = int(self.rng.integers(0, len(self.shards)))
        shard = self.shards[si]
        idx = self.rng.integers(0, self.sizes[si], size=self.batch_images)
        # Force read-out to ndarray; memmap copy is cheap for small chunks.
        imgs = np.ascontiguousarray(shard[idx])           # (B, 576, 1024) fp16
        patches = imgs.reshape(-1, shard.shape[-1])       # (B*576, 1024)
        t = torch.from_numpy(patches).float().to(self.device, non_blocking=True)
        return t


# ---- Metrics -------------------------------------------------------------

def recon_metrics(x, recon, latent):
    import torch
    with torch.no_grad():
        err  = recon - x
        sq   = err * err
        mse_per_elem = sq.mean().item()
        sse_per_tok  = sq.sum(dim=-1).mean().item()
        total_var = x.var(dim=0, unbiased=False).sum().item()
        resid_var = err.var(dim=0, unbiased=False).sum().item()
        ev = 1 - resid_var / max(total_var, 1e-12)
        active = (latent > 0).float().sum(dim=-1).mean().item()
    return {"mse_per_elem": mse_per_elem, "sse_per_tok": sse_per_tok,
            "ev": ev, "active_features": active}


def evaluate(sae, val_sampler, n_batches: int = 80):
    import torch
    sae.eval()
    agg = {"mse_per_elem": 0.0, "sse_per_tok": 0.0,
           "ev_num": 0.0, "ev_denom": 0.0, "active": 0.0}
    with torch.no_grad():
        for _ in range(n_batches):
            x = val_sampler.sample()
            recon, latent = sae(x)
            err = recon - x
            agg["mse_per_elem"] += (err * err).mean().item()
            agg["sse_per_tok"]  += (err * err).sum(dim=-1).mean().item()
            agg["ev_num"]   += (err * err).sum().item()
            agg["ev_denom"] += (x * x).sum().item()
            agg["active"]   += (latent > 0).float().sum(dim=-1).mean().item()
    sae.train()
    return {
        "mse_per_elem": agg["mse_per_elem"] / n_batches,
        "sse_per_tok":  agg["sse_per_tok"]  / n_batches,
        "ev":           1 - agg["ev_num"] / max(agg["ev_denom"], 1e-12),
        "active_features": agg["active"] / n_batches,
    }


# ---- Training loop -------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path,
                    default=ASPECT_SCRATCH / "activations" / "clip-L14-layer22")
    ap.add_argument("--out-dir", type=Path,
                    default=MODELS_DIR / "own-sae-clip-L14-layer22")
    ap.add_argument("--d-in", type=int, default=1024)
    ap.add_argument("--n-features", type=int, default=65536)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch-images", type=int, default=8,
                    help="images per step; patch tokens/step = 576 * this")
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--warmup-steps", type=int, default=1_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-shards", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=100_000)
    ap.add_argument("--val-every", type=int, default=5_000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--wandb-project", default="aspect-seeing-sae-training")
    ap.add_argument("--wandb-name",    default="own-sae-v1")
    ap.add_argument("--resume-from", default=None,
                    help="path to a checkpoint .pt to resume from")
    args = ap.parse_args()

    import torch
    import wandb

    # Setup
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  activations={args.activations_dir}", flush=True)

    shard_paths = sorted(args.activations_dir.glob("activations_*.npy"))
    assert len(shard_paths) > args.val_shards, f"need > {args.val_shards} shards"
    train_paths = shard_paths[:-args.val_shards]
    val_paths   = shard_paths[-args.val_shards:]
    print(f"[data] {len(train_paths)} train shards, {len(val_paths)} val shards", flush=True)

    # Dimension sanity
    sanity = np.load(shard_paths[0], mmap_mode="r")
    assert sanity.shape[-1] == args.d_in, f"shard d_in {sanity.shape[-1]} != args {args.d_in}"
    print(f"[data] shard0 shape={sanity.shape} dtype={sanity.dtype}", flush=True)
    del sanity

    train_sampler = ShardBatches(train_paths, args.batch_images, device, seed=42)
    val_sampler   = ShardBatches(val_paths,   args.batch_images, device, seed=999)

    # Model
    sae = build_sae(args.d_in, args.n_features, args.k, device)
    sae.normalize_decoder_()
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"[model] TopK SAE  d_in={args.d_in}  n_features={args.n_features}  k={args.k}  "
          f"params={n_params/1e6:.1f}M", flush=True)

    optim = torch.optim.Adam(sae.parameters(), lr=args.lr)

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_scale)

    step0 = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"[resume] loading {args.resume_from}", flush=True)
        ckpt = torch.load(args.resume_from, map_location=device)
        sae.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step0 = ckpt["step"] + 1
        print(f"[resume] starting at step {step0}", flush=True)

    # wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        dir=os.environ.get("WANDB_DIR"),
        config=vars(args),
        resume="allow",
        id=os.environ.get("WANDB_RUN_ID"),
    )
    print(f"[wandb] run id={run.id}  url={run.url}", flush=True)
    (args.out_dir / "wandb_run_id.txt").write_text(run.id + "\n")

    # Loop
    t0 = time.time()
    best_val_ev = -float("inf")
    for step in range(step0, args.total_steps):
        x = train_sampler.sample()                   # (B*576, 1024) fp32 on GPU
        sae.normalize_decoder_()
        recon, latent = sae(x)
        mse = (recon - x).pow(2).mean()
        optim.zero_grad(set_to_none=True)
        mse.backward()
        optim.step()
        sched.step()

        if step % args.log_every == 0:
            m = recon_metrics(x, recon, latent)
            tokens_seen = (step - step0 + 1) * args.batch_images * 576
            tok_per_sec = tokens_seen / max(time.time() - t0, 1e-3)
            lr_now = sched.get_last_lr()[0]
            print(f"[{step:>7}] loss={mse.item():.4f}  sse/tok={m['sse_per_tok']:.2f}  "
                  f"ev={m['ev']:.3f}  active={m['active_features']:.1f}  "
                  f"tok/s={tok_per_sec:.0f}  lr={lr_now:.2e}", flush=True)
            wandb.log({
                "train/loss_mse": mse.item(),
                "train/sse_per_tok": m["sse_per_tok"],
                "train/ev": m["ev"],
                "train/active_features": m["active_features"],
                "train/tokens_per_sec": tok_per_sec,
                "train/lr": lr_now,
            }, step=step)

        if step > step0 and step % args.val_every == 0:
            v = evaluate(sae, val_sampler, n_batches=80)
            print(f"   [VAL @ {step}] sse/tok={v['sse_per_tok']:.2f}  ev={v['ev']:.3f}  "
                  f"active={v['active_features']:.1f}", flush=True)
            wandb.log({f"val/{kk}": vv for kk, vv in v.items()}, step=step)
            if v["ev"] > best_val_ev:
                best_val_ev = v["ev"]
                best_path = args.out_dir / "best.pt"
                torch.save({
                    "step": step, "config": vars(args),
                    "model": {k: p.detach().half().cpu() for k, p in sae.state_dict().items()},
                    "val":   v,
                }, best_path)
                print(f"   new best EV={v['ev']:.3f}; saved {best_path}", flush=True)

        if step > step0 and step % args.checkpoint_every == 0:
            ckpt_path = args.out_dir / f"checkpoint_step{step:07d}.pt"
            torch.save({
                "step": step,
                "model": sae.state_dict(),        # full-precision for resume correctness
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "config": vars(args),
            }, ckpt_path)
            print(f"   saved checkpoint: {ckpt_path}", flush=True)
            # Keep only the 2 most recent checkpoints on disk (rolling window)
            older = sorted(args.out_dir.glob("checkpoint_step*.pt"))[:-2]
            for p in older:
                p.unlink(missing_ok=True)

    # Final save — half-precision weights only (per spec)
    final_path = args.out_dir / "final.pt"
    final_state = {k: p.detach().half().cpu() for k, p in sae.state_dict().items()}
    torch.save({
        "step": args.total_steps,
        "model": final_state,
        "config": vars(args),
    }, final_path)
    print(f"[done] saved {final_path}  best_val_ev={best_val_ev:.3f}", flush=True)
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
