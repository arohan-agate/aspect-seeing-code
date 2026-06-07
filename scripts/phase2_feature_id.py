"""Phase 2 — feature identification with tie-corrected rank AUROC.

This script computes per-feature AUROC with proper tie handling via
`scipy.stats.rankdata` (cf. paper Appendix D). With sparse TopK SAE features
(most cells are exactly zero), an argsort-based ranking that resolves ties by
input row order would systematically bias AUROC, generating phantom
"B-preferring" candidates whose features never actually activate on aspect-B
controls. The pipeline below avoids that.

Pipeline:

  1. **Proper tie correction**: `scipy.stats.rankdata(method='average', axis=0)`
     gives all tied values the average of the rank positions they would
     occupy, eliminating the row-order bias entirely.

  2. **Activation floor**: only retain features with `mean_match > 0.005`
     (matching-aspect controls). Kills the empty-feature category at the
     source.

  3. **Distractor AUROC** as the specificity measure. For each candidate
     feature, compute AUROC of "matching control" vs "10k random CC3M
     patches". Higher = more specific. Reuses `_auroc_full`.

  4. **Rank by distractor AUROC descending**, cap at 15 per aspect.

Outputs:
    $ASPECT_SCRATCH/outputs/phase2/features_<group>_v3.csv
    $ASPECT_SCRATCH/outputs/figures/phase2_features_<group>_v3.pdf
    $ASPECT_SCRATCH/outputs/phase2/feature_id_v3_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from aspect_seeing.paths import (
    ASPECT_SCRATCH, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIG_DIR,
)


INVENTORY_CSV = DATA_DIR / "dataset_inventory.csv"
CACHE_DIR     = ASPECT_SCRATCH / "activations" / "clip-L14-layer22"
SAE_CKPT      = MODELS_DIR / "own-sae-clip-L14-layer22" / "best.pt"
CLIP_PATH     = MODELS_DIR / "clip-vit-large-patch14-336"

OUT_DIR_CSV = OUTPUTS_DIR / "phase2"
OUT_DIR_FIG = FIG_DIR

LAYER_INDEX_FROM_END = -2
D_MODEL              = 1024
N_FEATURES           = 65_536
K_TOPK               = 32

MIN_AUROC            = 0.85
MIN_MEAN_MATCH       = 0.005   # filter empty-feature ties
MIN_DISTRACTOR_AUROC = 0.85    # specificity threshold (mirror of MIN_AUROC)
TOP_N_PER_ASPECT     = 15
N_DISTRACTORS        = 10_000  # random CC3M patches for the specificity AUROC
MAX_ACT_K            = 10
N_GRID_COLS          = 10
SEED                 = 2026

GROUP_ASPECT_PAIR_PATTERNS: dict[str, list[str]] = {
    "duck_rabbit": ["Duck-Rabbit", "Duck / Rabbit"],
    "face_vase":   ["Rubin Vase", "Takashima", "Vase / Profiles", "Face ↔ Vase"],
    "hidden_face": ["Tree / Faces", "Hidden-Face", "Plant / Faces", "Leaf / Face",
                    "Rock Formation / Face", "Sealife / Faces", "Flower / Faces",
                    "Wolf / Face", "Face / Building", "Face / Head"],
    "young_old_woman": ["Young-Old Woman", "Young Woman / Old Woman",
                        "Elderly Man / Young Woman"],
    "schroeder_stairs": ["Schroeder Stairs"],
    "necker_cube":   ["Necker Cube"],
}
GROUPS_DEFAULT = list(GROUP_ASPECT_PAIR_PATTERNS.keys())


# ---------- SAE module ----------

def build_sae(device: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TopKSAE(nn.Module):
        def __init__(self):
            super().__init__()
            W_enc = torch.randn(D_MODEL, N_FEATURES) / math.sqrt(D_MODEL)
            self.W_enc = nn.Parameter(W_enc.clone())
            self.W_dec = nn.Parameter(W_enc.t().contiguous().clone())
            self.b_enc = nn.Parameter(torch.zeros(N_FEATURES))
            self.b_pre = nn.Parameter(torch.zeros(D_MODEL))

        def encode(self, x):
            pre = x - self.b_pre
            hidden = F.relu(pre @ self.W_enc + self.b_enc)
            top_vals, top_idx = hidden.topk(K_TOPK, dim=-1)
            out = torch.zeros_like(hidden)
            out.scatter_(-1, top_idx, top_vals)
            return out
    return TopKSAE().to(device)


def load_own_sae(device: str):
    import torch
    ckpt = torch.load(SAE_CKPT, map_location=device, weights_only=False)
    print(f"[sae] loaded best.pt from step {ckpt['step']}", flush=True)
    if "val" in ckpt:
        v = ckpt["val"]
        print(f"[sae] training-time val: EV={v['ev']:.3f}  SSE/tok={v['sse_per_tok']:.2f}",
              flush=True)
    sae = build_sae(device)
    sd = {k: v.float() for k, v in ckpt["model"].items()}
    sae.load_state_dict(sd)
    sae.eval()
    return sae


# ---------- Tie-corrected AUROC ----------

def _auroc_full(X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
    """Vectorized Mann-Whitney AUROC using scipy.stats.rankdata for proper
    tie correction (rather than argsort-based ranks that resolve ties by
    row order)."""
    n_a, F = X_a.shape
    n_b, _ = X_b.shape
    if n_a == 0 or n_b == 0:
        return np.full(F, 0.5, dtype=np.float64)
    X = np.concatenate([X_a, X_b], axis=0).astype(np.float64)
    # Fall back to per-column rankdata if axis= isn't supported (pre-scipy-1.10).
    try:
        ranks = rankdata(X, method="average", axis=0)
    except TypeError:
        ranks = np.apply_along_axis(lambda v: rankdata(v, method="average"), 0, X)
    is_zero = (X.sum(axis=0) == 0)
    R_a = ranks[:n_a].sum(axis=0)
    U_a = R_a - n_a * (n_a + 1) / 2.0
    auroc = U_a / (n_a * n_b)
    auroc[is_zero] = 0.5
    return auroc


def loo_auroc(X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
    """LOO AUROC per feature: average AUROC after removing each sample once."""
    n_a, n_b = X_a.shape[0], X_b.shape[0]
    N = n_a + n_b
    acc = np.zeros(X_a.shape[1], dtype=np.float64)
    for k in range(n_a):
        mask = np.ones(n_a, dtype=bool); mask[k] = False
        acc += _auroc_full(X_a[mask], X_b)
    for k in range(n_b):
        mask = np.ones(n_b, dtype=bool); mask[k] = False
        acc += _auroc_full(X_a, X_b[mask])
    return acc / N


# ---------- CLIP forward ----------

def load_openai_clip(device: str):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    import torch
    print(f"[clip] loading OpenAI-CLIP-L/14-336 (bf16)", flush=True)
    processor = CLIPImageProcessor.from_pretrained(str(CLIP_PATH))
    model = CLIPVisionModel.from_pretrained(
        str(CLIP_PATH), dtype=torch.bfloat16,
    ).to(device).eval()
    return processor, model


def forward_images_to_patches(paths, processor, model, device, batch_size: int = 8):
    import torch
    from PIL import Image
    out_list = []
    t0 = time.time()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[clip] open fail {p}: {e}", flush=True)
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**inputs, output_hidden_states=True)
        layer22 = out.hidden_states[LAYER_INDEX_FROM_END]
        patches = layer22[:, 1:].float().cpu().numpy()
        out_list.append(patches)
        if start + batch_size >= len(paths) or (start // batch_size) % 5 == 0:
            print(f"[clip] {start + len(imgs)}/{len(paths)} in {time.time()-t0:.1f}s", flush=True)
    return np.concatenate(out_list, axis=0).astype(np.float32)


def sae_encode_image_features(patches_batch: np.ndarray, sae, device: str):
    """(N, 576, 1024) → (N, 65536) mean-pool latents and (N, 576, 65536) per-patch latents."""
    import torch
    N = patches_batch.shape[0]
    image_feats = np.zeros((N, N_FEATURES), dtype=np.float32)
    patch_feats: list[np.ndarray | None] = [None] * N
    chunk = 4
    with torch.no_grad():
        for i in range(0, N, chunk):
            x = torch.from_numpy(patches_batch[i:i + chunk]).to(device)
            c, P, D = x.shape
            flat = x.reshape(c * P, D)
            latent = sae.encode(flat)
            latent = latent.reshape(c, P, N_FEATURES).float().cpu().numpy()
            for j in range(c):
                patch_feats[i + j] = latent[j]
                image_feats[i + j] = latent[j].mean(axis=0)
    return image_feats, patch_feats


# ---------- Distractor activation ----------

def encode_distractor_patches(n: int, sae, device: str, rng: np.random.Generator) -> np.ndarray:
    """Return (n, 65536) SAE-encoded latents on n random patches from the cache."""
    import torch
    shard_paths = sorted(CACHE_DIR.glob("activations_*.npy"))
    shards = [np.load(p, mmap_mode="r") for p in shard_paths]
    sizes = [s.shape[0] for s in shards]
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    chunk = 1000
    print(f"[distract] sampling + encoding {n} patches in chunks of {chunk}", flush=True)
    t0 = time.time()
    for start in range(0, n, chunk):
        c = min(chunk, n - start)
        # Sample c (shard, image, patch) triples
        patches = np.zeros((c, D_MODEL), dtype=np.float32)
        for i in range(c):
            si = int(rng.integers(0, len(shards)))
            ii = int(rng.integers(0, sizes[si]))
            pj = int(rng.integers(0, 576))
            patches[i] = np.array(shards[si][ii, pj], dtype=np.float32)
        with torch.no_grad():
            x = torch.from_numpy(patches).to(device)
            latent = sae.encode(x).float().cpu().numpy()
        out[start:start + c] = latent
        if start + c >= n or (start // chunk) % 2 == 0:
            print(f"[distract] {start + c}/{n}  ({time.time()-t0:.1f}s)", flush=True)
    return out


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=GROUPS_DEFAULT)
    ap.add_argument("--skip-figure", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    OUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = load_own_sae(device)
    processor, clip = load_openai_clip(device)

    inv = list(csv.DictReader(INVENTORY_CSV.open()))
    rng = np.random.default_rng(SEED)

    # ---- distractor encoding (one-time, reused across all groups) ----
    print(f"[1/4] encoding {N_DISTRACTORS} distractor patches", flush=True)
    distractor_latent = encode_distractor_patches(N_DISTRACTORS, sae, device, rng)
    n_active_distract = (distractor_latent > 0).any(axis=0).sum()
    print(f"       {n_active_distract} of {N_FEATURES} features fired on at least 1 distractor", flush=True)

    summary: list[dict] = []

    for g in args.groups:
        print(f"\n[group {g}] " + "-" * 50, flush=True)
        patterns = set(GROUP_ASPECT_PAIR_PATTERNS.get(g, []))
        rows = [r for r in inv if r["aspect_pair"] in patterns]
        ctrl_a = [r for r in rows if r["category"] == "control" and r["aspect_label"] == "a"]
        ctrl_b = [r for r in rows if r["category"] == "control" and r["aspect_label"] == "b"]
        bistable = [r for r in rows if r["category"] == "bistable"]
        n_a, n_b = len(ctrl_a), len(ctrl_b)
        print(f"[{g}] {n_a} control_a + {n_b} control_b + {len(bistable)} bistable", flush=True)
        if n_a < 5 or n_b < 5:
            print(f"[{g}] !! skipping — fewer than 5 controls on one side", flush=True)
            summary.append({"group": g, "status": "skipped-thin", "n_a": n_a, "n_b": n_b})
            continue

        paths_a = [Path(r["file_path"]) for r in ctrl_a]
        paths_b = [Path(r["file_path"]) for r in ctrl_b]
        paths_bist = [Path(r["file_path"]) for r in bistable]
        all_paths = paths_a + paths_b + paths_bist

        print(f"[{g}] forwarding {len(all_paths)} images through OpenAI-CLIP", flush=True)
        patches = forward_images_to_patches(all_paths, processor, clip, device,
                                            batch_size=args.batch_size)
        print(f"[{g}] SAE-encoding (N={patches.shape[0]})", flush=True)
        image_feats, patch_feats = sae_encode_image_features(patches, sae, device)
        del patches

        X_a = image_feats[:n_a]
        X_b = image_feats[n_a:n_a + n_b]
        mean_a = X_a.mean(axis=0)
        mean_b = X_b.mean(axis=0)

        # ---- LOO AUROC (now tie-corrected) ----
        print(f"[{g}] LOO AUROC (N={n_a + n_b} folds, tie-corrected)", flush=True)
        t0 = time.time()
        loo = loo_auroc(X_a, X_b)
        print(f"[{g}] LOO done in {time.time()-t0:.1f}s", flush=True)

        # ---- Filter: AUROC strength + activation floor ----
        cand_a = np.where((loo > MIN_AUROC) & (mean_a > MIN_MEAN_MATCH))[0]
        cand_b = np.where((loo < 1 - MIN_AUROC) & (mean_b > MIN_MEAN_MATCH))[0]
        n_pre_floor_a = int(((loo > MIN_AUROC)).sum())
        n_pre_floor_b = int(((loo < 1 - MIN_AUROC)).sum())
        print(f"[{g}] candidates after AUROC: A={n_pre_floor_a} B={n_pre_floor_b}; "
              f"after mean_match floor (>{MIN_MEAN_MATCH}): A={len(cand_a)} B={len(cand_b)}",
              flush=True)

        # ---- Distractor AUROC (specificity) ----
        # For each retained candidate, compute AUROC of its values on matching
        # controls vs distractors. High AUROC = features fire much more on
        # matching aspect than on random patches.
        print(f"[{g}] specificity AUROC against {N_DISTRACTORS} distractors", flush=True)

        def _spec_auroc(cands: np.ndarray, X_match: np.ndarray) -> np.ndarray:
            if len(cands) == 0:
                return np.array([], dtype=float)
            return _auroc_full(X_match[:, cands], distractor_latent[:, cands])

        spec_a = _spec_auroc(cand_a, X_a)
        spec_b = _spec_auroc(cand_b, X_b)

        # Apply specificity threshold + rank descending
        keep_a_mask = spec_a > MIN_DISTRACTOR_AUROC
        keep_b_mask = spec_b > MIN_DISTRACTOR_AUROC
        kept_a = cand_a[keep_a_mask]
        kept_a_spec = spec_a[keep_a_mask]
        kept_b = cand_b[keep_b_mask]
        kept_b_spec = spec_b[keep_b_mask]
        # Sort descending by specificity AUROC
        oa = (-kept_a_spec).argsort()
        ob = (-kept_b_spec).argsort()
        kept_a = kept_a[oa][:TOP_N_PER_ASPECT]
        kept_a_spec = kept_a_spec[oa][:TOP_N_PER_ASPECT]
        kept_b = kept_b[ob][:TOP_N_PER_ASPECT]
        kept_b_spec = kept_b_spec[ob][:TOP_N_PER_ASPECT]
        print(f"[{g}] after specificity > {MIN_DISTRACTOR_AUROC}: "
              f"A={len(kept_a)}, B={len(kept_b)}; capped at {TOP_N_PER_ASPECT}", flush=True)

        # ---- Max-activating examples ----
        feature_rows: list[dict] = []
        retained = list(zip(kept_a.tolist(), ["a"] * len(kept_a),
                            kept_a_spec.tolist(),
                            [mean_a[fid] for fid in kept_a])) + \
                   list(zip(kept_b.tolist(), ["b"] * len(kept_b),
                            kept_b_spec.tolist(),
                            [mean_b[fid] for fid in kept_b]))

        for (feat_id, lbl, spec_score, m_match) in retained:
            best: list[tuple[float, int, int]] = []
            for ii, lat in enumerate(patch_feats):
                if lat is None:
                    continue
                col = lat[:, feat_id]
                pj = int(np.argmax(col))
                v = float(col[pj])
                if v <= 0:
                    continue
                best.append((v, ii, pj))
            best.sort(key=lambda x: -x[0])
            top = best[:MAX_ACT_K]
            max_imgs = ",".join(Path(all_paths[ii]).name for (_, ii, _) in top)
            max_vals = ",".join(f"{v:.3f}" for (v, _, _) in top)

            feature_rows.append({
                "aspect_pair": g,
                "aspect_label": lbl,
                "feature_id": int(feat_id),
                "loo_auroc": round(float(loo[feat_id]), 4),
                "loo_auroc_strength": round(float(max(loo[feat_id], 1 - loo[feat_id])), 4),
                "mean_match_activation": round(float(m_match), 4),
                "specificity_auroc": round(float(spec_score), 4),
                "n_controls_a": n_a,
                "n_controls_b": n_b,
                "max_activating_image_ids": max_imgs,
                "max_activating_values": max_vals,
                "manual_label": "",
            })

        csv_path = OUT_DIR_CSV / f"features_{g}_v3.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(feature_rows[0].keys()) if feature_rows else [
                    "aspect_pair","aspect_label","feature_id","loo_auroc",
                    "loo_auroc_strength","mean_match_activation","specificity_auroc",
                    "n_controls_a","n_controls_b","max_activating_image_ids",
                    "max_activating_values","manual_label",
                ],
            )
            w.writeheader()
            w.writerows(feature_rows)
        print(f"[{g}] wrote {csv_path}  ({len(feature_rows)} features)", flush=True)

        if not args.skip_figure and feature_rows:
            _plot_group_figure(g, feature_rows, all_paths)

        summary.append({
            "group": g, "status": "ok",
            "n_controls_a": n_a, "n_controls_b": n_b,
            "n_pre_floor_auroc_a": n_pre_floor_a, "n_pre_floor_auroc_b": n_pre_floor_b,
            "n_after_mean_match_a": int(len(cand_a)), "n_after_mean_match_b": int(len(cand_b)),
            "n_retained_a": int(len(kept_a)), "n_retained_b": int(len(kept_b)),
            "median_specificity_auroc_a": float(np.median(kept_a_spec)) if len(kept_a_spec) else None,
            "median_specificity_auroc_b": float(np.median(kept_b_spec)) if len(kept_b_spec) else None,
        })

    summary_path = OUT_DIR_CSV / "feature_id_v3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n[done] wrote summary: {summary_path}", flush=True)
    return 0


def _plot_group_figure(group: str, feature_rows: list[dict], all_paths: list[Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import PIL.Image
    n_feats = len(feature_rows)
    if n_feats == 0:
        return
    fig = plt.figure(figsize=(N_GRID_COLS * 1.4 + 1.7, n_feats * 1.55))
    gs = GridSpec(n_feats, N_GRID_COLS + 1, figure=fig,
                  width_ratios=[1.7] + [1.0] * N_GRID_COLS,
                  hspace=0.25, wspace=0.05)
    lookup = {Path(p).name: p for p in all_paths}
    for row, feat in enumerate(feature_rows):
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.axis("off")
        ax_lbl.text(0, 0.5,
                    f"feat {feat['feature_id']}\n"
                    f"pref {feat['aspect_label']}\n"
                    f"AUROC {feat['loo_auroc_strength']:.3f}\n"
                    f"specAUROC {feat['specificity_auroc']:.3f}\n"
                    f"mean {feat['mean_match_activation']:.3f}",
                    ha="left", va="center", fontsize=8, family="monospace")
        names = feat["max_activating_image_ids"].split(",")
        for col in range(N_GRID_COLS):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.axis("off")
            if col >= len(names) or not names[col].strip():
                continue
            p = lookup.get(names[col].strip())
            if p is None or not p.exists():
                continue
            try:
                im = PIL.Image.open(p).convert("RGB")
                im.thumbnail((160, 160))
                ax.imshow(im)
            except Exception:
                pass
    fig.suptitle(f"Phase 2 — {group} — own-SAE/OpenAI-CLIP, tie-corrected LOO + dist-AUROC",
                 fontsize=10, y=0.995)
    out_pdf = OUT_DIR_FIG / f"phase2_features_{group}_v3.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
