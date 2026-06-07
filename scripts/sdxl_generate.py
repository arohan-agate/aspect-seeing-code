"""Generate SDXL bistable-variant and pure-aspect-control images (stimulus expansion).

Two modes:
    --mode=sample   one image per prompt (for user approval of prompt wording)
    --mode=bulk     N_CANDIDATES images per prompt, different seeds (for hand-verification)

Output layout:
    data/expansion/sdxl/<group>/<kind>/<prompt_slug>_seed<N>.png
    data/expansion/sdxl/<group>/<kind>/<prompt_slug>.prompt.txt   (single file per prompt,
                                                                   records the prompt text)

kind ∈ {bistable, control_a, control_b}.
Prompts are hard-coded in the tables below.

Run on a GPU (SDXL needs ~10 GB VRAM in bf16).
    python scripts/sdxl_generate.py --mode=sample
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from aspect_seeing.paths import DATA_DIR, MODELS_DIR

SDXL_PATH = MODELS_DIR / "sdxl-base-1.0"
OUT_ROOT  = DATA_DIR / "expansion" / "sdxl"

# ---------- prompt tables ----------

# Bistable variants we want to add via SDXL.
# (group, prompt_slug, prompt, negative_prompt, count_needed_after_approval)
BISTABLE_PROMPTS: list[tuple[str, str, str, str, int]] = [
    ("duck_rabbit", "jastrow_style_lithograph",
     "black-and-white 19th-century lithograph of a duck-rabbit illusion, simple line art, "
     "ambiguous head, visible beak and long ears, strong contrast, isolated on white",
     "color, photograph, 3d, texture, multiple figures",
     1),
    ("hidden_face", "tree_with_hidden_faces",
     "a tree with human faces subtly hidden in the bark and branch patterns, photorealistic, "
     "naturalistic color, daylight, no obvious face outlines, ambiguous faces only visible on close inspection",
     "cartoon, obvious face, portrait, 3d render",
     1),
    ("schroeder_stairs", "schroeder_illusion_line",
     "black-and-white line drawing of Schroeder's staircase illusion, visible as going up or "
     "coming down depending on viewing angle, clean geometric perspective lines, isolated on white",
     "color, photograph, 3d, escher, penrose stairs, shading",
     1),
    ("schroeder_stairs", "schroeder_illusion_minimal",
     "Schroeder staircase orientation illusion, alternative viewing angle, minimalist line art, "
     "no shading, precise geometric lines, black on white",
     "color, photograph, 3d, escher, shaded, textured",
     1),
    ("necker_cube", "necker_ambiguous",
     "classical Necker cube line drawing, 3D ambiguous wireframe, white background, thin black lines, "
     "equal foreshortening on front and back faces, perfectly symmetric",
     "solid, shaded, colored, complex background",
     1),
    ("necker_cube", "necker_minimal",
     "Necker cube optical illusion, minimalist wireframe, 12 equal edges, ambiguous orientation",
     "solid, shaded, colored, 3d render, noisy",
     1),
    ("young_old_woman", "hill_style_illusion",
     "black-and-white portrait illusion that can be seen as either a young woman in a feathered hat "
     "looking away or an elderly woman with a headscarf in profile, high contrast, ambiguous chin and nose",
     "color, photograph, single interpretation, clear age",
     1),
    # grimace_begger dropped — synthetic-only evidence too weak to anchor any claim.
]

# Disambiguated controls.
# (group, kind, prompt_slug, prompt, negative_prompt, count_needed)
CONTROL_PROMPTS: list[tuple[str, str, str, str, str, int]] = [
    # Duck-Rabbit
    ("duck_rabbit", "control_a", "pure_duck",
     "a realistic photograph of a duck swimming in a pond, only a duck visible, clear beak and webbed feet, "
     "water and reeds, naturalistic daylight, no other animals",
     "rabbit, long ears, mammal, fur, grass", 15),
    ("duck_rabbit", "control_b", "pure_rabbit",
     "a realistic photograph of a rabbit in a grassy meadow, only a rabbit visible, long ears and fur clearly "
     "visible, no beaks, no water, naturalistic daylight",
     "duck, beak, bird, feathers, water", 15),

    # Face ↔ Vase
    ("face_vase", "control_a", "pure_faces_profile",
     "two human faces in profile facing each other, 19th-century silhouette style, solid black on white background, "
     "their noses pointed toward each other but no vase shape in the negative space, portrait poses",
     "vase, cup, object between them, color, texture", 15),
    ("face_vase", "control_b", "pure_vase",
     "a decorative ceramic vase on a plain neutral background, classical shape, one central vase, no human figures, "
     "no profiles, straight-on product photography",
     "faces, silhouettes, profiles, people", 15),

    # Hidden-Face
    ("hidden_face", "control_a", "pure_nature_scene",
     "a photograph of a tree trunk and leaves in a forest, close up on bark texture, no human figures, no faces, "
     "no recognizable features other than natural foliage and wood grain",
     "face, human, portrait, human features", 15),
    ("hidden_face", "control_b", "pure_face",
     "a close-up portrait photograph of a human face, studio lighting, neutral background, no natural elements, "
     "no foliage, front-facing neutral expression",
     "tree, leaves, nature, landscape, foliage", 15),

    # Schroeder Stairs
    ("schroeder_stairs", "control_a", "pure_upward_stairs",
     "black-and-white line drawing of a simple staircase, clearly going upward, viewer at bottom looking up, "
     "steps rising away from the viewer, minimal perspective lines, clean white background",
     "downward, descending, color, shaded, 3d render", 15),
    ("schroeder_stairs", "control_b", "pure_downward_stairs",
     "black-and-white line drawing of a simple staircase descending, clearly viewed from above looking down, "
     "steps going away and downward, bird's-eye perspective, clean white background",
     "upward, ascending, color, shaded, 3d render", 15),

    # Necker Cube
    ("necker_cube", "control_a", "pure_cube_from_above",
     "3D rendered solid opaque cube photographed from above, top face clearly visible and illuminated, "
     "unambiguous viewing angle, plain background, realistic lighting and shadows",
     "wireframe, line drawing, ambiguous, below", 15),
    ("necker_cube", "control_b", "pure_cube_from_below",
     "3D rendered solid opaque cube photographed from below, bottom face clearly visible, unambiguous "
     "low-angle viewpoint, plain background, consistent lighting and shadows",
     "wireframe, line drawing, ambiguous, above", 15),

    # Grimace-Begger dropped from primary AND controls (see plan §6 footnote).

    # Young-Old Woman
    ("young_old_woman", "control_a", "pure_young_woman",
     "a black-and-white portrait of a young woman in a large feathered hat, looking away from the viewer, "
     "clean line art style, smooth skin, youthful features",
     "elderly, wrinkled, old, headscarf", 15),
    ("young_old_woman", "control_b", "pure_elderly_woman",
     "a black-and-white profile portrait of an elderly woman wearing a headscarf, wrinkled face visible, "
     "prominent nose and chin, clean line art style, aged features",
     "young, smooth skin, feathered hat", 15),
]

HEIGHT = 1024   # SDXL's native resolution
WIDTH = 1024
STEPS = 30      # 30 is a good bf16 quality/speed tradeoff on SDXL


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")[:50]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sample", "bulk"), required=True,
                    help="sample: 1 image per prompt. bulk: generate `count_needed` per prompt.")
    ap.add_argument("--seed-base", type=int, default=42,
                    help="base torch seed; each image uses seed-base + image-index")
    ap.add_argument("--only-group", default=None,
                    help="restrict to one group (e.g. duck_rabbit)")
    ap.add_argument("--skip-bistable", action="store_true")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()

    print("[1/4] importing", flush=True)
    import torch
    from diffusers import StableDiffusionXLPipeline
    from PIL import Image

    if not SDXL_PATH.exists():
        print(f"!! SDXL weights missing at {SDXL_PATH}", file=sys.stderr)
        print("   download with: python -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', "
              f"local_dir='{SDXL_PATH}')\"", file=sys.stderr)
        return 1

    print(f"[2/4] loading SDXL (bf16) from {SDXL_PATH}", flush=True)
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(SDXL_PATH), torch_dtype=torch.bfloat16, use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"       loaded in {time.time()-t0:.1f}s  resident VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB",
          flush=True)

    # Build the flat work list ----------------------------------------------------
    jobs: list[tuple[str, str, str, str, str, int]] = []
    if not args.skip_bistable:
        for group, slug, prompt, neg, count in BISTABLE_PROMPTS:
            if args.only_group and group != args.only_group:
                continue
            jobs.append((group, "bistable", slug, prompt, neg, count))
    if not args.skip_controls:
        for group, kind, slug, prompt, neg, count in CONTROL_PROMPTS:
            if args.only_group and group != args.only_group:
                continue
            jobs.append((group, kind, slug, prompt, neg, count))

    # In sample mode we generate 1 per prompt regardless of `count`
    # (we over-generate 1.8× in bulk to allow hand-rejection; see plan §9)
    per_prompt = 1 if args.mode == "sample" else None
    OVERGEN = 1.8   # bulk over-generation factor

    total = sum(1 if per_prompt else int(round(c * OVERGEN)) for (_, _, _, _, _, c) in jobs)
    print(f"[3/4] {len(jobs)} prompts, mode={args.mode}, total images to generate={total}",
          flush=True)
    print(f"       over-gen factor (bulk only): {OVERGEN}", flush=True)

    # Generate --------------------------------------------------------------------
    manifest = []
    counter = 0
    t_start = time.time()
    for (group, kind, slug, prompt, neg, count) in jobs:
        out_dir = OUT_ROOT / group / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        n_images = per_prompt if per_prompt is not None else int(round(count * OVERGEN))
        # Save the prompt text alongside the images (one file per prompt)
        (out_dir / f"{slug}.prompt.txt").write_text(
            f"# group={group} kind={kind} slug={slug}\n"
            f"prompt: {prompt}\n"
            f"negative_prompt: {neg}\n"
        )
        for i in range(n_images):
            seed = args.seed_base + counter
            out_path = out_dir / f"{slug}_seed{seed:06d}.png"
            if out_path.exists():
                counter += 1
                continue
            gen = torch.Generator(device="cuda").manual_seed(seed)
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                img = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    height=HEIGHT, width=WIDTH,
                    num_inference_steps=STEPS,
                    generator=gen,
                ).images[0]
            img.save(out_path)
            dt = time.time() - t0
            counter += 1
            total_elapsed = time.time() - t_start
            eta = (total - counter) * (total_elapsed / max(counter, 1))
            print(f"[{counter:>4}/{total}] {group:<18} {kind:<10} {slug:<30} "
                  f"seed={seed:>6}  {dt:5.1f}s  eta={eta/60:5.1f} min", flush=True)
            manifest.append({"group": group, "kind": kind, "slug": slug,
                             "seed": seed, "path": str(out_path)})

    manifest_path = OUT_ROOT / f"manifest_{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[4/4] wrote {manifest_path}  ({len(manifest)} images)", flush=True)
    print(f"       total elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
