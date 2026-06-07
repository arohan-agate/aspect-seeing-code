"""Phase 3 — superposition vs dominance analysis (design doc §4.3).

For each bistable stimulus in each of the 6 groups, classify whether the
two aspect-feature pools (15 aspect_a + 15 aspect_b features from the
Phase 2 v3 manifest) co-activate (superposition), only one activates
(dominance-A or dominance-B), or neither does (dissolution).

Threshold calibration (per the user spec):
  - Threshold for aspect_a feature pool = median of (mean of the 15
    aspect_a feature activations) computed across the aspect_B controls.
    This is the "noise floor" — what the A-feature pool looks like when
    shown a non-A image.
  - Symmetric for aspect_b.
A bistable image is considered "active in pool X" iff its mean
pool-X activation exceeds threshold X.

Outputs:
    outputs/phase3/superposition_<group>.csv
    outputs/phase3/superposition_summary.json
    outputs/figures/phase3_superposition_<group>.pdf
    outputs/figures/phase3_superposition_overview.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from aspect_seeing.paths import DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FIG_DIR

INVENTORY_CSV       = DATA_DIR / "dataset_inventory.csv"
PHASE1_DOMINANCE    = OUTPUTS_DIR / "phase1" / "dominance_per_stimulus.csv"
PHASE2_FEATURES_DIR = OUTPUTS_DIR / "phase2"
SAE_CKPT            = MODELS_DIR / "own-sae-clip-L14-layer22" / "best.pt"
CLIP_PATH           = MODELS_DIR / "clip-vit-large-patch14-336"

OUT_DIR_CSV = OUTPUTS_DIR / "phase3"
OUT_DIR_FIG = FIG_DIR

LAYER_INDEX_FROM_END = -2
D_MODEL              = 1024
N_FEATURES           = 65_536
K_TOPK               = 32

# Same patterns as Phase 2 v3 — must stay in sync
GROUP_ASPECT_PAIR_PATTERNS: dict[str, list[str]] = {
    "duck_rabbit":      ["Duck-Rabbit", "Duck / Rabbit"],
    "face_vase":        ["Rubin Vase", "Takashima", "Vase / Profiles", "Face ↔ Vase"],
    "hidden_face":      ["Tree / Faces", "Hidden-Face", "Plant / Faces", "Leaf / Face",
                         "Rock Formation / Face", "Sealife / Faces", "Flower / Faces",
                         "Wolf / Face", "Face / Building", "Face / Head"],
    "young_old_woman":  ["Young-Old Woman", "Young Woman / Old Woman",
                         "Elderly Man / Young Woman"],
    "schroeder_stairs": ["Schroeder Stairs"],
    "necker_cube":      ["Necker Cube"],
}
GROUPS_DEFAULT = list(GROUP_ASPECT_PAIR_PATTERNS.keys())


# ---------- model loading ----------

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
    print(f"[sae] loaded best.pt step={ckpt['step']}", flush=True)
    sae = build_sae(device)
    sd = {k: v.float() for k, v in ckpt["model"].items()}
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def load_openai_clip(device: str):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    import torch
    print(f"[clip] loading OpenAI-CLIP-L/14-336 (bf16)", flush=True)
    processor = CLIPImageProcessor.from_pretrained(str(CLIP_PATH))
    model = CLIPVisionModel.from_pretrained(
        str(CLIP_PATH), dtype=torch.bfloat16,
    ).to(device).eval()
    return processor, model


# ---------- forward ----------

def forward_images_to_imgfeats(paths, processor, clip, sae, device, batch_size: int = 8):
    """Run images through CLIP layer 22 → SAE encoder → mean-pool over patches.
    Returns dict[Path -> np.ndarray of shape (N_FEATURES,)]."""
    import torch
    from PIL import Image
    out: dict[Path, np.ndarray] = {}
    t0 = time.time()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        imgs, kept = [], []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                kept.append(p)
            except Exception as e:
                print(f"[fwd] open fail {p}: {e}", flush=True)
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cl_out = clip(**inputs, output_hidden_states=True)
        layer22 = cl_out.hidden_states[LAYER_INDEX_FROM_END]      # (B, 577, 1024)
        patches = layer22[:, 1:].float()                          # (B, 576, 1024)
        B, P, D = patches.shape
        flat = patches.reshape(B * P, D)
        with torch.no_grad():
            latent = sae.encode(flat)                             # (B*P, 65536)
        latent = latent.reshape(B, P, N_FEATURES).mean(dim=1).float().cpu().numpy()
        for p, vec in zip(kept, latent):
            out[p] = vec
        if (start // batch_size) % 5 == 0 or start + batch_size >= len(paths):
            print(f"[fwd] {len(out)}/{len(paths)} in {time.time()-t0:.1f}s", flush=True)
    return out


# ---------- threshold + classification ----------

def classify(a_act: float, b_act: float, thr_a: float, thr_b: float) -> str:
    a_on = a_act > thr_a
    b_on = b_act > thr_b
    if a_on and b_on:
        return "superposition"
    if a_on:
        return "dominance_a"
    if b_on:
        return "dominance_b"
    return "neither"


# ---------- per-group runner ----------

def run_group(g: str, inv_rows, phase1_lookup, img_feats, feature_csv: Path) -> dict:
    patterns = set(GROUP_ASPECT_PAIR_PATTERNS[g])
    rows = [r for r in inv_rows if r["aspect_pair"] in patterns]
    bistable = [r for r in rows if r["category"] == "bistable"]
    ctrl_a   = [r for r in rows if r["category"] == "control" and r["aspect_label"] == "a"]
    ctrl_b   = [r for r in rows if r["category"] == "control" and r["aspect_label"] == "b"]
    print(f"[{g}] {len(bistable)} bistable, {len(ctrl_a)} ctrl_a, {len(ctrl_b)} ctrl_b", flush=True)

    # Load Phase 2 v3 retained features for this group
    feats = list(csv.DictReader(feature_csv.open()))
    a_ids = [int(r["feature_id"]) for r in feats if r["aspect_label"] == "a"]
    b_ids = [int(r["feature_id"]) for r in feats if r["aspect_label"] == "b"]
    print(f"[{g}] feature pools: A={len(a_ids)} B={len(b_ids)}", flush=True)

    def pool_act(image_path: Path, ids: list[int]) -> float:
        vec = img_feats.get(image_path)
        if vec is None:
            return float("nan")
        return float(np.mean([vec[i] for i in ids]))

    # ---- threshold calibration on controls ----
    # thr_A = median over CONTROL_B images of (mean of A-feature activations)
    # thr_B = median over CONTROL_A images of (mean of B-feature activations)
    a_on_b_ctrl = [pool_act(Path(r["file_path"]), a_ids) for r in ctrl_b]
    b_on_a_ctrl = [pool_act(Path(r["file_path"]), b_ids) for r in ctrl_a]
    a_on_b_ctrl = [v for v in a_on_b_ctrl if not math.isnan(v)]
    b_on_a_ctrl = [v for v in b_on_a_ctrl if not math.isnan(v)]
    thr_a = float(np.median(a_on_b_ctrl)) if a_on_b_ctrl else 0.0
    thr_b = float(np.median(b_on_a_ctrl)) if b_on_a_ctrl else 0.0

    # Sanity diagnostics: A-pool on its matching A controls should be MUCH higher than thr_A
    a_on_a_ctrl = [pool_act(Path(r["file_path"]), a_ids) for r in ctrl_a if pool_act(Path(r["file_path"]), a_ids) == pool_act(Path(r["file_path"]), a_ids)]
    b_on_b_ctrl = [pool_act(Path(r["file_path"]), b_ids) for r in ctrl_b if pool_act(Path(r["file_path"]), b_ids) == pool_act(Path(r["file_path"]), b_ids)]
    print(f"[{g}] thr_A={thr_a:.5f}  (A-pool on B-controls median); "
          f"A-pool on A-controls median={np.median(a_on_a_ctrl) if a_on_a_ctrl else 0:.5f}  "
          f"({(np.median(a_on_a_ctrl) / max(thr_a, 1e-12) if a_on_a_ctrl else 0):.1f}× threshold)", flush=True)
    print(f"[{g}] thr_B={thr_b:.5f}  (B-pool on A-controls median); "
          f"B-pool on B-controls median={np.median(b_on_b_ctrl) if b_on_b_ctrl else 0:.5f}  "
          f"({(np.median(b_on_b_ctrl) / max(thr_b, 1e-12) if b_on_b_ctrl else 0):.1f}× threshold)", flush=True)

    # ---- per-bistable classification ----
    out_rows = []
    for r in bistable:
        p = Path(r["file_path"])
        a_act = pool_act(p, a_ids)
        b_act = pool_act(p, b_ids)
        cls = classify(a_act, b_act, thr_a, thr_b) if not (math.isnan(a_act) or math.isnan(b_act)) else "missing"
        ph1 = phase1_lookup.get(r["id"], {})
        out_rows.append({
            "stimulus_id": r["id"],
            "filename": p.name,
            "aspect_pair": r["aspect_pair"],
            "aspect_a_activation":  round(a_act, 6) if not math.isnan(a_act) else "",
            "aspect_b_activation":  round(b_act, 6) if not math.isnan(b_act) else "",
            "thr_a": round(thr_a, 6),
            "thr_b": round(thr_b, 6),
            "classification": cls,
            "phase1_dominance":  ph1.get("dominance", ""),
            "phase1_p_neither":  ph1.get("p_neither", ""),
            "phase1_p_aspect_a": ph1.get("p_aspect_a", ""),
            "phase1_p_aspect_b": ph1.get("p_aspect_b", ""),
        })

    csv_path = OUT_DIR_CSV / f"superposition_{g}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[{g}] wrote {csv_path}", flush=True)

    # Summary counts
    counts = Counter(r["classification"] for r in out_rows)
    return {
        "group": g,
        "n_bistable": len(out_rows),
        "n_controls_a": len(ctrl_a),
        "n_controls_b": len(ctrl_b),
        "n_features_a": len(a_ids),
        "n_features_b": len(b_ids),
        "thr_a": thr_a,
        "thr_b": thr_b,
        "a_pool_on_a_ctrl_median": float(np.median(a_on_a_ctrl)) if a_on_a_ctrl else 0.0,
        "b_pool_on_b_ctrl_median": float(np.median(b_on_b_ctrl)) if b_on_b_ctrl else 0.0,
        "classification_counts": dict(counts),
        "rows": out_rows,
    }


# ---------- figures ----------

CLASS_COLORS = {
    "superposition": "#7c3aed",  # purple
    "dominance_a":   "#2563eb",  # blue
    "dominance_b":   "#dc2626",  # red
    "neither":       "#6b7280",  # grey
    "missing":       "#000000",
}


def _scatter_panel(ax, group_data: dict, title: str | None = None):
    rows = group_data["rows"]
    thr_a = group_data["thr_a"]
    thr_b = group_data["thr_b"]
    xs, ys, sizes, colors = [], [], [], []
    for r in rows:
        if r["aspect_a_activation"] == "" or r["aspect_b_activation"] == "":
            continue
        xs.append(float(r["aspect_a_activation"]))
        ys.append(float(r["aspect_b_activation"]))
        # marker size: encode Phase 1 dominance (larger = more dominant); fall back to 60
        try:
            d = float(r["phase1_dominance"])
            sizes.append(40 + 200 * d)
        except (ValueError, TypeError):
            sizes.append(60)
        colors.append(CLASS_COLORS.get(r["classification"], "#000"))
    if not xs:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or group_data["group"])
        return
    ax.axhline(thr_b, color="black", lw=0.6, linestyle=":", alpha=0.6)
    ax.axvline(thr_a, color="black", lw=0.6, linestyle=":", alpha=0.6)
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.78, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("aspect-A pool mean activation")
    ax.set_ylabel("aspect-B pool mean activation")
    title_str = title or f"{group_data['group']}  (n={len(rows)})"
    counts = group_data["classification_counts"]
    title_str += "\n" + "  ".join(f"{k.split('_')[-1] if k.startswith('dominance') else k}={counts.get(k,0)}"
                                  for k in ("superposition", "dominance_a", "dominance_b", "neither"))
    ax.set_title(title_str, fontsize=9)
    # Pad axis to include thresholds
    xmax = max(max(xs), thr_a) * 1.15
    ymax = max(max(ys), thr_b) * 1.15
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)


def render_per_group_figures(group_results: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    for gd in group_results:
        fig, ax = plt.subplots(figsize=(6.5, 5.6))
        _scatter_panel(ax, gd)
        legend_handles = [
            Line2D([0],[0], marker="o", color="w", markerfacecolor=CLASS_COLORS[k],
                   markersize=9, label=k)
            for k in ("superposition", "dominance_a", "dominance_b", "neither")
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="best",
                  framealpha=0.85, title="classification")
        out_pdf = OUT_DIR_FIG / f"phase3_superposition_{gd['group']}.pdf"
        fig.tight_layout()
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[fig] wrote {out_pdf}", flush=True)


def render_overview_figure(group_results: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = len(group_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.6))
    axes = np.atleast_2d(axes).flatten()
    for ax, gd in zip(axes, group_results):
        _scatter_panel(ax, gd)
    for ax in axes[len(group_results):]:
        ax.set_visible(False)
    legend_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=CLASS_COLORS[k],
               markersize=10, label=k)
        for k in ("superposition", "dominance_a", "dominance_b", "neither")
    ]
    fig.legend(handles=legend_handles, ncol=4, loc="upper center", fontsize=10,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9, title="classification")
    fig.suptitle(
        "Phase 3 — superposition vs dominance per stimulus, all 6 groups\n"
        "Marker size ∝ Phase 1 (neutral) dominance score",
        fontsize=11, y=1.05,
    )
    fig.tight_layout()
    out_pdf = OUT_DIR_FIG / "phase3_superposition_overview.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_pdf}", flush=True)


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=GROUPS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    OUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load inventory + Phase 1 dominance ----
    inv = list(csv.DictReader(INVENTORY_CSV.open()))
    phase1_lookup: dict[str, dict] = {}
    if PHASE1_DOMINANCE.exists():
        for r in csv.DictReader(PHASE1_DOMINANCE.open()):
            phase1_lookup[r["stimulus_id"]] = r
        print(f"[init] loaded {len(phase1_lookup)} Phase 1 dominance rows", flush=True)
    else:
        print(f"[init] WARNING: no Phase 1 dominance CSV at {PHASE1_DOMINANCE}", flush=True)

    # ---- collect all paths needed across requested groups (dedup) ----
    needed_paths: dict[Path, None] = {}
    for g in args.groups:
        patterns = set(GROUP_ASPECT_PAIR_PATTERNS[g])
        for r in inv:
            if r["aspect_pair"] not in patterns:
                continue
            if r["category"] not in ("bistable", "control"):
                continue
            p = Path(r["file_path"])
            if p.exists():
                needed_paths[p] = None
            else:
                print(f"[init] missing image: {p}", flush=True)
    paths = list(needed_paths)
    print(f"[init] {len(paths)} unique images to forward", flush=True)

    # ---- load models + forward ----
    sae = load_own_sae(device)
    processor, clip = load_openai_clip(device)
    print(f"[fwd] running CLIP→SAE on {len(paths)} images", flush=True)
    img_feats = forward_images_to_imgfeats(paths, processor, clip, sae, device,
                                           batch_size=args.batch_size)

    # ---- per-group analysis ----
    group_results = []
    for g in args.groups:
        feat_csv = PHASE2_FEATURES_DIR / f"features_{g}_v3.csv"
        if not feat_csv.exists():
            print(f"[{g}] !! missing {feat_csv} — skipping", flush=True)
            continue
        gd = run_group(g, inv, phase1_lookup, img_feats, feat_csv)
        group_results.append(gd)

    # ---- summary ----
    summary = []
    for gd in group_results:
        summary.append({
            "group": gd["group"],
            "n_bistable": gd["n_bistable"],
            "n_controls_a": gd["n_controls_a"],
            "n_controls_b": gd["n_controls_b"],
            "n_features_a": gd["n_features_a"],
            "n_features_b": gd["n_features_b"],
            "threshold_a": gd["thr_a"],
            "threshold_b": gd["thr_b"],
            "a_pool_on_a_ctrl_median": gd["a_pool_on_a_ctrl_median"],
            "b_pool_on_b_ctrl_median": gd["b_pool_on_b_ctrl_median"],
            "classification_counts": gd["classification_counts"],
        })
    sum_path = OUT_DIR_CSV / "superposition_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n[done] wrote {sum_path}", flush=True)

    # ---- figures ----
    if not args.skip_figures and group_results:
        render_per_group_figures(group_results)
        render_overview_figure(group_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
