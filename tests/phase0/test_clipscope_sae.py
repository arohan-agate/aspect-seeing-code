"""Phase 0 smoke test 2: CLIP-Scope layer-22 SAE round-trip MSE.

Uses LAION CLIP (laion/CLIP-ViT-L-14-laion2B-s32B-b82K) — the exact backbone
CLIP-Scope was trained on. README baseline MSE for layer 22 = 299.96 (CLS
token). We expect to land in that ballpark on a real bistable image. Patch-
token MSE is reported separately (this is what we'll actually intervene on
in Phase 5; CLS is dropped by LLaVA's vision_feature_layer=-2 path).

Run on an A100/A40 after `source scripts/activate.sh`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from aspect_seeing.paths import DATA_DIR, MODELS_DIR

CLIPSCOPE_DIR = MODELS_DIR / "CLIP-ViT-L-scope"
SAE_CHECKPOINT_REL = "22_resid/1200013184.pt"   # latest (most training tokens)
IMAGE_PATH = (
    DATA_DIR / "panagopoulou" / "images"
    / "Bistable Images Original" / "duck_rabbit_1.png"
)
LAYER_INDEX = 22


def main() -> int:
    print("[1/5] importing", flush=True)
    import torch
    import PIL.Image
    from clipscope import ConfiguredViT, TopKSAE

    print("[2/5] loading SAE", flush=True)
    # clipscope expects a HF-style relative path; we have it locally so
    # download_dir lets it use the snapshot we already pulled.
    sae = TopKSAE.from_pretrained(
        checkpoint=SAE_CHECKPOINT_REL,
        repo_id="lewington/CLIP-ViT-L-scope",
        device="cuda",
    )
    sae.eval()
    print(f"      sae class={sae.__class__.__name__}", flush=True)

    print("[3/5] loading LAION CLIP via ConfiguredViT", flush=True)
    locations = [(LAYER_INDEX, "resid")]
    transformer = ConfiguredViT(locations, device="cuda")

    print(f"[4/5] loading image: {IMAGE_PATH.name}", flush=True)
    if not IMAGE_PATH.exists():
        print(f"!! image not found: {IMAGE_PATH}", file=sys.stderr)
        return 1
    img = PIL.Image.open(IMAGE_PATH).convert("RGB")

    print("[5/5] forward + reconstruct", flush=True)
    with torch.no_grad():
        acts = transformer.all_activations(img)[locations[0]]   # (1, 257, 1024)
    print(f"      activations shape: {tuple(acts.shape)} dtype={acts.dtype}", flush=True)
    assert acts.shape[-1] == 1024, "expected hidden dim 1024 for CLIP-L/14"
    assert acts.shape[1] == 257, "expected 256 patches + 1 CLS = 257 tokens"

    # CLS round-trip (matches README baseline MSE 299.96)
    cls = acts[:, 0]                                            # (1, 1024)
    with torch.no_grad():
        out_cls = sae.forward_verbose(cls)
    recon_cls = out_cls["reconstruction"]                       # (1, 1024)
    mse_cls = torch.nn.functional.mse_loss(recon_cls, cls).item()
    sse_cls = ((recon_cls - cls) ** 2).sum().item()
    var_cls = ((cls - cls.mean()) ** 2).sum().item()
    ev_cls = 1 - sse_cls / max(var_cls, 1e-12)
    print(f"      CLS  : MSE-per-elem={mse_cls:.4f}  SSE={sse_cls:.2f}  EV~{ev_cls:.3f}", flush=True)

    # Patch-token round-trip — this is what we actually use downstream.
    patches = acts[:, 1:].reshape(-1, 1024)                     # (256, 1024)
    with torch.no_grad():
        out_p = sae.forward_verbose(patches)
    recon_p = out_p["reconstruction"]
    mse_patch = torch.nn.functional.mse_loss(recon_p, patches).item()
    sse_patch = ((recon_p - patches) ** 2).sum(dim=-1).mean().item()  # SSE-per-token
    print(f"      patch: MSE-per-elem={mse_patch:.4f}  SSE-per-token={sse_patch:.2f}", flush=True)

    # README reports SSE-style MSE (sum of squared errors, not mean over dims).
    # The 299.96 figure is sum-over-1024-dims of squared errors. Report both
    # so we can sanity-check against the README directly.
    print(f"      ↳ CLS SSE-over-dims = {sse_cls:.2f}  (README baseline ≈ 299.96)", flush=True)

    # Latent sparsity sanity (TopK sae with k=32)
    latent = out_cls["latent"]
    nonzero = (latent != 0).sum(dim=-1).float().mean().item()
    print(f"      latent: shape={tuple(latent.shape)}  active/sample={nonzero:.1f} (expect ~32)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
