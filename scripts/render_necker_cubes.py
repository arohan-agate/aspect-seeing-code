"""Programmatically render canonical Necker cubes as line drawings.

Output: 8 variants at different line widths, sizes, rotations, and
edge-style combinations. By construction every variant is mechanically
bistable — the wireframe is the Necker cube: 12 straight edges with
equal foreshortening so front-face / back-face parsing is ambiguous.

No SDXL, no Wikimedia — just matplotlib. Stronger evidence than any
generative attempt for this specific illusion.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aspect_seeing.paths import DATA_DIR

OUT_DIR = DATA_DIR / "expansion" / "programmatic" / "necker_cube"


def _cube_edges(rotation_deg: float = 30, tilt_deg: float = 20, size: float = 1.0):
    """Return a list of (x0, y0)->(x1, y1) 2-D line endpoints for a cube
    projected isometrically. rotation_deg rotates around the vertical axis;
    tilt_deg tilts around the horizontal (camera pitch)."""
    # 3-D corners of the cube
    s = size / 2
    corners = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
    ])
    rot = math.radians(rotation_deg)
    tilt = math.radians(tilt_deg)
    Rz = np.array([[ math.cos(rot), -math.sin(rot), 0],
                   [ math.sin(rot),  math.cos(rot), 0],
                   [ 0,              0,             1]])
    Rx = np.array([[1, 0, 0],
                   [0,  math.cos(tilt), -math.sin(tilt)],
                   [0,  math.sin(tilt),  math.cos(tilt)]])
    rotated = (corners @ Rz.T) @ Rx.T
    # Orthographic projection — keep x and z (or y, depending)
    proj = rotated[:, [0, 2]]
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # bottom face
        (4,5),(5,6),(6,7),(7,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # vertical edges
    ]
    return [(proj[a], proj[b]) for a, b in edges]


def render_variant(out_path: Path, *, line_width: float, rotation_deg: float,
                   tilt_deg: float, size: float, canvas_px: int,
                   dashed: bool = False, bg: str = "white",
                   ink: str = "black") -> None:
    fig, ax = plt.subplots(figsize=(canvas_px / 100, canvas_px / 100), dpi=100)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ls = "--" if dashed else "-"
    for (p0, p1) in _cube_edges(rotation_deg=rotation_deg, tilt_deg=tilt_deg, size=size):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                color=ink, linewidth=line_width, linestyle=ls, solid_capstyle="round")
    pad = 0.18 * size
    ax.set_xlim(-size - pad, size + pad)
    ax.set_ylim(-size - pad, size + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, facecolor=bg)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 8 canonical variants — line width × rotation × tilt × dashed variants
    variants = [
        # idx, line_w, rot, tilt, size, canvas, dashed, bg, ink, slug
        ( 1, 2.5, 30, 20, 1.0, 512, False, "white", "black", "necker_w25_r30_t20"),
        ( 2, 4.0, 30, 20, 1.0, 512, False, "white", "black", "necker_w40_r30_t20_thick"),
        ( 3, 1.5, 30, 20, 1.0, 512, False, "white", "black", "necker_w15_r30_t20_thin"),
        ( 4, 2.5, 45, 20, 1.0, 512, False, "white", "black", "necker_r45"),
        ( 5, 2.5, 15, 20, 1.0, 512, False, "white", "black", "necker_r15"),
        ( 6, 2.5, 30, 30, 1.0, 512, False, "white", "black", "necker_t30"),
        ( 7, 2.5, 30, 20, 1.0, 512, True,  "white", "black", "necker_dashed"),
        ( 8, 2.5, 30, 20, 1.0, 768, False, "white", "black", "necker_canvas768"),
    ]

    print(f"[1/2] rendering {len(variants)} Necker cubes to {OUT_DIR}", flush=True)
    for (idx, lw, rot, tilt, size, canvas, dashed, bg, ink, slug) in variants:
        path = OUT_DIR / f"{slug}.png"
        render_variant(path, line_width=lw, rotation_deg=rot, tilt_deg=tilt,
                       size=size, canvas_px=canvas, dashed=dashed, bg=bg, ink=ink)
        print(f"   [{idx}] {path.name}  "
              f"(w={lw}, rot={rot}°, tilt={tilt}°, canvas={canvas}px, dashed={dashed})",
              flush=True)
    print(f"[2/2] wrote {len(variants)} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
