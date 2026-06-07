"""Regenerate control_b (aspect-B) images for Schroeder Stairs and Necker Cube,
both of which were blocked in the first review pass because their original
line-art prompts were structurally ambiguous.

Two strategies, applied in one run:

  Schroeder control_b — SDXL with PHOTOGRAPHIC prompts that use shadow /
  perspective / context to disambiguate descent. Five prompt templates × 6
  seeds each = 30 candidates.

  Necker control_b   — PROGRAMMATIC matplotlib 3D solid-cube renders
  with explicit directional lighting. Face shading comes from the dot product
  of face normal and light direction, so "cube from below" has the bottom
  face visible and lit differently than the top. 24 canonical variants
  across rotation × tilt × light-angle × face-color.

Output: $ASPECT_SCRATCH/data/expansion/regen_v2/
          schroeder_stairs/control_b/*.png
          necker_cube/control_b/*.png
        <repo>/docs/regen_v2_review.html
          (keep/reject review HTML)
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from aspect_seeing.paths import ASPECT_REPO, DATA_DIR, MODELS_DIR


OUT_ROOT   = DATA_DIR / "expansion" / "regen_v2"
OUT_SHROE  = OUT_ROOT / "schroeder_stairs/control_b"
OUT_NECKER = OUT_ROOT / "necker_cube/control_b"
REVIEW_HTML = ASPECT_REPO / "docs" / "regen_v2_review.html"

SDXL_PATH = MODELS_DIR / "sdxl-base-1.0"


# ---------- Schroeder: photographic SDXL ----------

SCHROEDER_PROMPTS: list[tuple[str, str, str]] = [
    ("basement_concrete",
     "a photograph of a real concrete staircase descending into a basement, "
     "wide-angle lens, viewer at top of the stairs looking down, handrails on "
     "both sides, strong overhead lighting casting clear shadows under each "
     "step, visible basement floor at the bottom, architectural photography",
     "line drawing, wireframe, ambiguous orientation, upward, ascending, "
     "geometric diagram, sketch"),

    ("subway_descending",
     "a photograph of a subway station staircase going down, tile walls, "
     "overhead fluorescent lighting, viewer at top looking down, platform "
     "visible at the bottom, clear downward perspective, realistic urban "
     "scene",
     "line drawing, wireframe, upward, going up, sketch, illustration"),

    ("wooden_descent",
     "a photograph of a wooden staircase descending into a lower room, "
     "warm interior lighting, viewer standing at the top step looking down, "
     "hardwood floor visible at the bottom, shadows clearly showing depth, "
     "realistic interior photography",
     "line drawing, wireframe, upward, going up, geometric diagram"),

    ("spiral_descending",
     "a photograph of a spiral staircase descending into a basement, viewer "
     "at the top looking down into the spiral, curving handrail visible, "
     "clear sense of downward motion and depth, architectural realism",
     "line drawing, wireframe, flat illustration, sketch, ambiguous"),

    ("garden_steps_down",
     "a photograph of stone garden steps descending down a hillside, clear "
     "downward perspective, long shadows from late afternoon sunlight, "
     "visible ground below, realistic landscape photograph",
     "line drawing, wireframe, ambiguous, sketch"),
]
SCHROEDER_SEEDS_PER_PROMPT = 6


def gen_schroeder_photos() -> list[Path]:
    import torch
    from diffusers import StableDiffusionXLPipeline
    OUT_SHROE.mkdir(parents=True, exist_ok=True)
    print(f"[schro/1] loading SDXL from {SDXL_PATH}", flush=True)
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(SDXL_PATH), torch_dtype=torch.bfloat16, use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[schro/1] loaded in {time.time()-t0:.1f}s  "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Save prompt sidecars once per slug
    for slug, prompt, neg in SCHROEDER_PROMPTS:
        sc = OUT_SHROE / f"{slug}.prompt.txt"
        sc.write_text(f"# slug={slug}\nprompt: {prompt}\nnegative_prompt: {neg}\n")

    paths: list[Path] = []
    seed_base = 100_000
    i = 0
    for (slug, prompt, neg) in SCHROEDER_PROMPTS:
        for s in range(SCHROEDER_SEEDS_PER_PROMPT):
            seed = seed_base + i
            out_path = OUT_SHROE / f"{slug}_seed{seed:06d}.png"
            if out_path.exists():
                paths.append(out_path); i += 1; continue
            gen = torch.Generator(device="cuda").manual_seed(seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                img = pipe(prompt=prompt, negative_prompt=neg,
                            height=1024, width=1024, num_inference_steps=30,
                            generator=gen).images[0]
            img.save(out_path)
            paths.append(out_path)
            print(f"[schro/2] {out_path.name} saved", flush=True)
            i += 1

    # Drop pipe before matplotlib pass to free VRAM
    del pipe
    torch.cuda.empty_cache()
    return paths


# ---------- Necker: programmatic 3D solid cubes ----------

def _cube_faces_with_normals():
    """Return a list of (face_corners, face_normal) for a unit cube at origin."""
    # Each face is a list of 4 (x,y,z) corners, CCW when viewed from outside.
    # Normal points outward.
    s = 0.5
    V = np.array([[-s,-s,-s], [ s,-s,-s], [ s, s,-s], [-s, s,-s],
                  [-s,-s, s], [ s,-s, s], [ s, s, s], [-s, s, s]], dtype=float)
    faces_idx = [
        ([0,3,2,1], (0,0,-1)),   # bottom (-z)
        ([4,5,6,7], (0,0, 1)),   # top    (+z)
        ([0,1,5,4], (0,-1,0)),   # front  (-y)
        ([2,3,7,6], (0, 1,0)),   # back   (+y)
        ([1,2,6,5], ( 1,0,0)),   # right  (+x)
        ([3,0,4,7], (-1,0,0)),   # left   (-x)
    ]
    return [(V[list(idx)], np.array(n, dtype=float)) for (idx, n) in faces_idx]


def _shade(normal: np.ndarray, light_dir: np.ndarray, ambient: float = 0.25) -> float:
    """Lambert + ambient. light_dir should point FROM light TO surface."""
    lit = max(0.0, float(-np.dot(normal / (np.linalg.norm(normal) + 1e-12),
                                  light_dir / (np.linalg.norm(light_dir) + 1e-12))))
    return min(1.0, ambient + (1 - ambient) * lit)


def render_necker_solid(out_path: Path, *, elev: float, azim: float,
                        light_dir: tuple[float, float, float],
                        base_color=(0.85, 0.6, 0.4), canvas_px: int = 512) -> None:
    """Render a solid cube from viewpoint (elev, azim) with directional light.
    elev: degrees; negative = from below (bottom face visible + lit by overhead)."""
    faces = _cube_faces_with_normals()
    ld = np.array(light_dir, dtype=float)
    fig = plt.figure(figsize=(canvas_px / 100, canvas_px / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    # Build polygons, colored by shading
    polys = []
    colors = []
    for corners, n in faces:
        shade = _shade(n, ld, ambient=0.22)
        polys.append(corners)
        rgb = tuple(shade * c for c in base_color)
        colors.append(rgb)
    pc = Poly3DCollection(polys, facecolors=colors, edgecolors="#222",
                          linewidths=0.8, zsort="average")
    ax.add_collection3d(pc)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    # Tighter margin
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def gen_necker_solid_cubes() -> list[Path]:
    """24 variants of 'cube from below' (elev < 0), with different azimuths,
    light directions, and colors."""
    OUT_NECKER.mkdir(parents=True, exist_ok=True)
    variants = []
    # All have negative elevation (viewer below cube) so the BOTTOM face is visible.
    # Vary azim, light, color.
    azimuths = [30, 45, 60, 75, 120, 150]      # 6
    light_configs = [
        (0, 0, -1),     # pure top-down (bottom face in shadow)
        (0.3, 0.3, -1), # angled overhead
        (-0.3, 0.3, -1),
        (0, 0.5, -0.9), # from front-top
    ]
    colors = [(0.85, 0.6, 0.4), (0.6, 0.75, 0.9),
              (0.75, 0.75, 0.75), (0.55, 0.75, 0.55)]
    elevs = [-15, -30]                          # 2 viewpoints (both "from below")

    paths: list[Path] = []
    i = 0
    for elev in elevs:
        for azim in azimuths:
            # Pick 2 light/color combos per azim to stay at 24 total
            for (ld, col) in zip(light_configs[:2], colors[:2]):
                slug = (f"necker_solid_e{elev:+d}_a{azim:d}_"
                        f"l{ld[0]:.1f}_{ld[1]:.1f}_{ld[2]:.1f}_"
                        f"c{col[0]:.1f}_{col[1]:.1f}_{col[2]:.1f}")
                slug = re.sub(r"[^a-zA-Z0-9_]+", "_", slug)
                out_path = OUT_NECKER / f"{slug}.png"
                if not out_path.exists():
                    render_necker_solid(out_path, elev=elev, azim=azim,
                                        light_dir=ld, base_color=col,
                                        canvas_px=512)
                paths.append(out_path)
                i += 1
    # Sidecar describing the batch
    (OUT_NECKER / "README.txt").write_text(
        "Programmatic 3D solid-cube renders for Necker control_b.\n"
        "All rendered with negative elevation (viewer below cube) so the\n"
        "bottom face is visible, and directional overhead lighting makes the\n"
        "shading asymmetric — unambiguous 'from below' by construction,\n"
        "mirroring the programmatic Neckers in data/expansion/programmatic/.\n"
        "Generated by scripts/regen_control_b.py.\n"
    )
    print(f"[necker] rendered {len(paths)} solid-cube variants → {OUT_NECKER}", flush=True)
    return paths


# ---------- review HTML ----------

def build_review_html(schroeder_paths: list[Path], necker_paths: list[Path]) -> None:
    from PIL import Image
    def b64(path: Path) -> str:
        im = Image.open(path).convert("RGB")
        im.thumbnail((360, 360))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def tile(path: Path, source: str, group: str) -> str:
        try:
            d = b64(path)
            img = f'<img src="data:image/jpeg;base64,{d}" alt="{path.name}">'
        except Exception:
            img = '<div class="missing">(encode failed)</div>'
        return (f'<div class="tile" data-source="{source}" data-group="{group}" '
                f'data-kind="control_b" data-filename="{path.name}">'
                f'{img}'
                f'<div class="meta">{path.name}</div>'
                f'</div>')

    css = """
    body{font-family:-apple-system,sans-serif;max-width:1400px;margin:1.5rem auto;padding:0 1rem;}
    h1{margin:0 0 0.3rem 0;} .summary{color:#666;}
    h2{margin:2rem 0 0.6rem 0;padding:0.5rem 0.8rem;background:#f1f5f9;border-left:4px solid #3b82f6;font-size:1.05rem;}
    .topbar{display:flex;gap:1rem;align-items:center;padding:0.6rem 0.8rem;background:#fef3c7;border:1px solid #fcd34d;
            border-radius:6px;position:sticky;top:0;z-index:20;box-shadow:0 2px 6px rgba(0,0,0,0.04);}
    .topbar button{padding:0.4rem 0.8rem;background:#1f2937;color:#fff;border:0;border-radius:4px;cursor:pointer;font-weight:600;}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:0.7rem;}
    .tile{border:2px solid transparent;border-radius:6px;overflow:hidden;background:white;cursor:pointer;position:relative;}
    .tile img{display:block;width:100%;height:220px;object-fit:cover;background:#f4f4f4;}
    .tile .meta{padding:0.25rem 0.45rem;font-size:0.7rem;font-family:ui-monospace,monospace;color:#555;}
    .tile.keep{border-color:#059669;box-shadow:0 0 0 3px rgba(5,150,105,0.2);}
    .tile.reject{border-color:#dc2626;opacity:0.45;}
    .tile.keep::after{content:"✓ KEEP";position:absolute;top:4px;right:4px;background:#059669;color:white;padding:2px 6px;border-radius:3px;font-size:0.7rem;font-weight:600;}
    .tile.reject::after{content:"✗ REJECT";position:absolute;top:4px;right:4px;background:#dc2626;color:white;padding:2px 6px;border-radius:3px;font-size:0.7rem;font-weight:600;}
    """
    js = r"""
    const tiles = Array.from(document.querySelectorAll('.tile'));
    const MAP = {unset:'keep', keep:'reject', reject:'unset'};
    function set(t, s){t.classList.remove('keep','reject'); if(s!=='unset') t.classList.add(s); t.dataset.decision = s; count();}
    tiles.forEach(t => {t.dataset.decision = 'unset'; t.addEventListener('click', () => set(t, MAP[t.dataset.decision]));});
    let hov=null; tiles.forEach(t=>{t.addEventListener('mouseenter',()=>hov=t); t.addEventListener('mouseleave',()=>{if(hov===t) hov=null;});});
    document.addEventListener('keydown', e=>{ if(!hov) return; const k=e.key.toLowerCase(); if(k==='k') set(hov,'keep'); if(k==='r') set(hov,'reject'); if(k==='u') set(hov,'unset'); });
    function count(){
      let k=0,r=0,u=0; tiles.forEach(t=>{const d=t.dataset.decision; if(d==='keep') k++; else if(d==='reject') r++; else u++;});
      document.getElementById('ck').textContent=k; document.getElementById('cr').textContent=r; document.getElementById('cu').textContent=u;
    }
    document.getElementById('exp').addEventListener('click', ()=>{
      const rows = [['source','group','kind','slug','seed','filename','decision']];
      tiles.forEach(t => rows.push(['regen_v2', t.dataset.group, 'control_b', '', '', t.dataset.filename, t.dataset.decision]));
      const csv = rows.map(r => r.map(x => /[,"\n]/.test(String(x)) ? '"'+String(x).replace(/"/g,'""')+'"' : String(x)).join(',')).join('\n');
      const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'}); const a=document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = `regen_v2_decisions_${new Date().toISOString().slice(0,10)}.csv`;
      document.body.appendChild(a); a.click(); a.remove();
    });
    count();
    """
    out = [
        "<!DOCTYPE html>", "<html><head><meta charset=\"UTF-8\">",
        "<title>regen v2 review — Schroeder + Necker control_b</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Regen v2 review — Schroeder (photographic SDXL) + Necker (3D renders)</h1>",
        '<p class="summary">Click/keyboard-shortcut per tile: click to cycle keep/reject, or K/R/U while hovered. Export CSV from the top bar.</p>',
        '<div class="topbar">',
        '<div>keep:<b id="ck">0</b> · reject:<b id="cr">0</b> · unset:<b id="cu">0</b></div>',
        '<button id="exp">⬇ export CSV</button></div>',
        f"<h2>schroeder_stairs · control_b · photographic SDXL ({len(schroeder_paths)})</h2>",
        '<div class="grid">',
        *[tile(p, "regen_v2", "schroeder_stairs") for p in schroeder_paths],
        "</div>",
        f"<h2>necker_cube · control_b · programmatic 3D solid cubes ({len(necker_paths)})</h2>",
        '<div class="grid">',
        *[tile(p, "regen_v2", "necker_cube") for p in necker_paths],
        "</div>",
        f"<script>{js}</script></body></html>",
    ]
    REVIEW_HTML.write_text("\n".join(out))
    size_mb = REVIEW_HTML.stat().st_size / 1024 / 1024
    print(f"[review] wrote {REVIEW_HTML} ({size_mb:.2f} MB)", flush=True)


def main() -> int:
    print("[regen] starting Schroeder photographic SDXL + Necker 3D solid-cube", flush=True)
    try:
        schroeder_paths = gen_schroeder_photos()
    except Exception as e:
        print(f"!! Schroeder generation failed: {type(e).__name__}: {e}", flush=True)
        schroeder_paths = []
    try:
        necker_paths = gen_necker_solid_cubes()
    except Exception as e:
        print(f"!! Necker generation failed: {type(e).__name__}: {e}", flush=True)
        necker_paths = []

    build_review_html(schroeder_paths, necker_paths)
    print(f"[done] schro={len(schroeder_paths)}  necker={len(necker_paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
