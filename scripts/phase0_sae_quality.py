"""Phase 0 SAE-quality sweep: per-image MSE + EV across all 29 Panagopoulou
bistable stimuli using the CLIP-Scope layer-22 TopK SAE.

Outputs:
  outputs/phase0_sae_quality.csv  — per-image CLS + patch MSE / SSE / EV
  outputs/figures/phase0_sae_mse.pdf  — histogram + per-image scatter

Compares the per-image SSE distribution against CLIP-Scope's reported
average MSE = 299.96 / EV = 0.684 at layer 22 (CLS token, LAION CLIP).
If our median CLS SSE on bistable stimuli is > 400, that's our signal
to schedule training our own SAE on OpenAI-CLIP layer 22 in week 1.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from aspect_seeing.paths import DATA_DIR, OUTPUTS_DIR, FIG_DIR

IMAGES_DIR = DATA_DIR / "panagopoulou" / "images" / "Bistable Images Original"
OUT_CSV  = OUTPUTS_DIR / "phase0_sae_quality.csv"
OUT_PDF  = FIG_DIR / "phase0_sae_mse.pdf"

CLIPSCOPE_REPO = "lewington/CLIP-ViT-L-scope"
SAE_CHECKPOINT_REL = "22_resid/1200013184.pt"
LAYER_INDEX = 22

README_BASELINE_SSE = 299.96   # CLIP-Scope README, layer 22, CLS, LAION test set
README_BASELINE_EV = 0.684


def main() -> int:
    print("[1/4] importing", flush=True)
    import torch
    import PIL.Image
    from clipscope import ConfiguredViT, TopKSAE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    print("[2/4] loading SAE + LAION CLIP", flush=True)
    sae = TopKSAE.from_pretrained(
        checkpoint=SAE_CHECKPOINT_REL,
        repo_id=CLIPSCOPE_REPO,
        device="cuda",
    )
    sae.eval()
    transformer = ConfiguredViT([(LAYER_INDEX, "resid")], device="cuda")

    image_paths = sorted([p for p in IMAGES_DIR.iterdir()
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    print(f"     {len(image_paths)} images found", flush=True)

    print(f"[3/4] sweeping {len(image_paths)} images", flush=True)
    rows = []
    t_total = time.time()
    for img_path in image_paths:
        img = PIL.Image.open(img_path).convert("RGB")
        with torch.no_grad():
            acts = transformer.all_activations(img)[(LAYER_INDEX, "resid")]   # (1,257,1024)

        # CLS round-trip
        cls = acts[:, 0]                                                       # (1, 1024)
        out_cls = sae.forward_verbose(cls)
        recon_cls = out_cls["reconstruction"]
        sse_cls   = ((recon_cls - cls) ** 2).sum().item()
        mse_cls   = sse_cls / cls.numel()
        # Explained variance = 1 - SS_res / SS_tot, with SS_tot defined w.r.t. mean of CLS itself
        ss_tot_cls = ((cls - cls.mean()) ** 2).sum().item()
        ev_cls    = 1 - sse_cls / max(ss_tot_cls, 1e-12)

        # Patch round-trip (mean over 256 patch tokens)
        patches = acts[:, 1:].reshape(-1, 1024)                                # (256, 1024)
        out_p   = sae.forward_verbose(patches)
        recon_p = out_p["reconstruction"]
        sq_err  = ((recon_p - patches) ** 2)
        sse_per_token = sq_err.sum(dim=-1).mean().item()                       # mean over tokens
        mse_patch = sq_err.mean().item()
        ss_tot_p  = ((patches - patches.mean(dim=-1, keepdim=True)) ** 2).sum().item()
        ev_patch  = 1 - sq_err.sum().item() / max(ss_tot_p, 1e-12)

        latent_active_cls = (out_cls["latent"] != 0).sum(dim=-1).float().mean().item()

        rows.append({
            "image": img_path.name,
            "cls_sse": round(sse_cls, 4),
            "cls_mse_per_elem": round(mse_cls, 6),
            "cls_ev": round(ev_cls, 4),
            "patch_sse_per_token": round(sse_per_token, 4),
            "patch_mse_per_elem": round(mse_patch, 6),
            "patch_ev": round(ev_patch, 4),
            "latent_active_cls": round(latent_active_cls, 1),
        })
        print(f"     {img_path.name:<32}  CLS SSE={sse_cls:7.2f}  EV={ev_cls:.3f}  "
              f"patchSSE/tok={sse_per_token:7.2f}", flush=True)
    elapsed = time.time() - t_total
    print(f"     swept in {elapsed:.1f}s", flush=True)

    # Write CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"     wrote {OUT_CSV}", flush=True)

    print("[4/4] plotting + summary", flush=True)
    cls_sse  = np.array([r["cls_sse"] for r in rows])
    cls_ev   = np.array([r["cls_ev"] for r in rows])
    patch_sse = np.array([r["patch_sse_per_token"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(cls_sse, bins=12, color="steelblue", edgecolor="white")
    ax.axvline(README_BASELINE_SSE, color="red", linestyle="--",
               label=f"README baseline = {README_BASELINE_SSE:.0f}")
    ax.axvline(np.median(cls_sse), color="black", linestyle=":",
               label=f"median = {np.median(cls_sse):.0f}")
    ax.set_xlabel("CLS reconstruction SSE (sum over 1024 dims)")
    ax.set_ylabel("# images")
    ax.set_title("CLIP-Scope L22 SAE — CLS SSE on Panagopoulou bistable stimuli")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(cls_sse, patch_sse, s=28, color="darkorange", alpha=0.85)
    ax.axhline(np.median(patch_sse), color="black", linestyle=":", linewidth=1)
    ax.axvline(np.median(cls_sse), color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("CLS SSE")
    ax.set_ylabel("Patch SSE per token (mean over 256)")
    ax.set_title("Per-image: CLS vs patch reconstruction error")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f"     wrote {OUT_PDF}", flush=True)

    # Summary
    median_cls = float(np.median(cls_sse))
    median_ev  = float(np.median(cls_ev))
    median_patch = float(np.median(patch_sse))
    print()
    print("=" * 60, flush=True)
    print(f"  N images               : {len(rows)}", flush=True)
    print(f"  CLS SSE  median / mean : {median_cls:.1f} / {cls_sse.mean():.1f}   "
          f"(README {README_BASELINE_SSE:.1f})", flush=True)
    print(f"  CLS SSE  min / max     : {cls_sse.min():.1f} / {cls_sse.max():.1f}", flush=True)
    print(f"  CLS EV   median / mean : {median_ev:.3f} / {cls_ev.mean():.3f}   "
          f"(README {README_BASELINE_EV:.3f})", flush=True)
    print(f"  patch SSE/tok median   : {median_patch:.1f}", flush=True)
    if median_cls > 400:
        print(f"  ==> FLAG: median CLS SSE {median_cls:.1f} > 400 — schedule own-SAE training", flush=True)
    else:
        print(f"  ==> median CLS SSE {median_cls:.1f} ≤ 400 — CLIP-Scope baseline acceptable", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
