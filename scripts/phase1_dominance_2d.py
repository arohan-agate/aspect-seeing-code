"""Two-axis Phase 1 summary figure — destined to be the paper's Figure 1.

Each of the 83 stimuli is plotted at (dominance, P(neither)):
  - x = dominance = |P(aspect_a) - P(aspect_b)|
  - y = P(neither)

Three regions are shaded to make the two-population finding legible:
  - bottom-right:  high dominance, low neither  → canonical steering targets
  - top-left:      low dominance, high neither  → abstention / aspect-blind
  - middle / lower: intermediate                → partial / mixed

Points are colored by a coarse semantic cluster of aspect pair (duck-rabbit,
rubin-vase / takashima, necker-cube, young-old, schroeder, other).

Writes both PDF (for paper) and PNG (for wandb).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from aspect_seeing.paths import OUTPUTS_DIR, FIG_DIR

DOMINANCE_CSV = OUTPUTS_DIR / "phase1" / "dominance_per_stimulus.csv"
OUT_PDF = FIG_DIR / "phase1_dominance_2d.pdf"
OUT_PNG = OUT_PDF.with_suffix(".png")


# Semantic clustering by substring match on aspect_pair name.
# Order matters: earlier rules take precedence.
CLUSTER_RULES: list[tuple[str, tuple[str, ...], str]] = [
    # label               signatures                                    color
    ("duck-rabbit",      ("duck-rabbit", "duck / rabbit"),            "#2563eb"),
    ("rubin / vase / faces",
                         ("rubin vase", "takashima", "vase", "tree / faces", "flower / faces",
                          "plant / faces", "sealife / faces"),        "#dc2626"),
    ("necker cube",      ("necker cube",),                            "#059669"),
    ("young / old",      ("young-old woman", "young woman / old woman",
                          "elderly man / young woman"),                "#7c3aed"),
    ("schroeder stairs", ("schroeder stairs",),                        "#d97706"),
    ("spinning dancer",  ("spinning dancer",),                         "#db2777"),
]
OTHER_LABEL = "other"
OTHER_COLOR = "#94a3b8"


def _cluster_for(pair_name: str) -> tuple[str, str]:
    p = pair_name.lower()
    for label, sigs, color in CLUSTER_RULES:
        if any(s in p for s in sigs):
            return label, color
    return OTHER_LABEL, OTHER_COLOR


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = list(csv.DictReader(DOMINANCE_CSV.open()))
    dom  = np.array([float(r["dominance"]) for r in rows])
    pn   = np.array([float(r["p_neither"]) for r in rows])
    pair = [r["aspect_pair"] for r in rows]
    sids = [r["stimulus_id"] for r in rows]

    # Group by cluster for plotting with distinct handles
    clusters: dict[str, dict] = {}
    for i, p in enumerate(pair):
        label, color = _cluster_for(p)
        clusters.setdefault(label, {"color": color, "idx": []})["idx"].append(i)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    # Region shading (subtle)
    # Thresholds: 'high dominance' = dom > 0.6, 'low neither' = pn < 0.2
    #             'high neither'   = pn  > 0.6, 'low dominance' = dom < 0.2
    ax.add_patch(Rectangle((0.6, 0.0), 0.4, 0.2, facecolor="#dcfce7", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, 0.6), 0.2, 0.4, facecolor="#fee2e2", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.2, 0.2), 0.4, 0.4, facecolor="#fef3c7", edgecolor="none",
                           alpha=0.45, zorder=0))

    ax.text(0.98, 0.03, "steering targets\n(high dom · low neither)",
            ha="right", va="bottom", fontsize=8.5, color="#166534", alpha=0.9)
    ax.text(0.02, 0.97, "abstention / aspect-blind\n(low dom · high neither)",
            ha="left", va="top", fontsize=8.5, color="#991b1b", alpha=0.9)
    ax.text(0.40, 0.40, "intermediate / partial", ha="center", va="center",
            fontsize=8.5, color="#92400e", alpha=0.85)

    # Threshold guide lines
    for t in (0.2, 0.6):
        ax.axvline(t, color="#cbd5e1", linestyle=":", linewidth=0.8, zorder=1)
        ax.axhline(t, color="#cbd5e1", linestyle=":", linewidth=0.8, zorder=1)

    # Scatter per cluster, in a consistent order (keep 'other' behind the named ones)
    order = [OTHER_LABEL] + [lbl for lbl, _, _ in CLUSTER_RULES]
    for label in order:
        if label not in clusters:
            continue
        c = clusters[label]
        ii = c["idx"]
        size = 36 if label != OTHER_LABEL else 22
        alpha = 0.8 if label != OTHER_LABEL else 0.55
        edge = "white" if label != OTHER_LABEL else "none"
        # Add jitter: many stimuli sit at identical (0, 1) or (1, 0) → stack visually
        rng = np.random.default_rng(abs(hash(label)) % 2**32)
        jx = rng.uniform(-0.008, 0.008, size=len(ii))
        jy = rng.uniform(-0.008, 0.008, size=len(ii))
        ax.scatter(dom[ii] + jx, pn[ii] + jy, s=size, c=c["color"], alpha=alpha,
                   edgecolors=edge, linewidths=0.6, zorder=3,
                   label=f"{label} (n={len(ii)})")

    ax.set_xlabel(r"dominance $|P(\mathrm{aspect}_A) - P(\mathrm{aspect}_B)|$")
    ax.set_ylabel(r"$P(\mathrm{neither})$")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Phase 1 — dominance × abstention (83 stimuli, LLaVA-1.6-7B, Qwen3-8B judge)",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.98),
              framealpha=0.9, edgecolor="#e5e7eb")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=180)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")

    # Quick text summary of what landed in each region
    mask_target     = (dom > 0.6) & (pn < 0.2)
    mask_abstain    = (dom < 0.2) & (pn > 0.6)
    mask_intermed   = (~mask_target) & (~mask_abstain)
    print()
    print(f"steering-target region : {mask_target.sum():>3} stimuli")
    print(f"abstention region      : {mask_abstain.sum():>3} stimuli")
    print(f"intermediate           : {mask_intermed.sum():>3} stimuli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
