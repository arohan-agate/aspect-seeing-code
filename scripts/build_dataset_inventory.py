"""Build the unified bistable-stimulus inventory.

Combines Panagopoulou (29 images) + AmbiBench bistable + bistable-suggestive
geometric (~56 images) into one CSV with a consistent schema:
    id, source, category, aspect_pair, aspect_a, aspect_b, file_path, license, notes

Writes:
    $ASPECT_SCRATCH/data/dataset_inventory.csv
    docs/dataset_inventory.md  (markdown summary + reference)

Curation rules:
- AmbiBench bistable_open / bistable_choice: take all 55 unique files (every
  one is a canonical bistable stimulus by construction).
- AmbiBench geometric: take only files whose names match bistable-suggestive
  signatures (necker, schroeder, rubin, ambiguous, illusion, impossible,
  duck_rabbit, young_old) — most "geometric" rows are about line/shape
  perception, not aspect-seeing.
- Panagopoulou: take all 29 (already curated by the original authors).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from aspect_seeing.paths import ASPECT_REPO, DATA_DIR

REPO_DIR = ASPECT_REPO
OUT_CSV = DATA_DIR / "dataset_inventory.csv"
OUT_MD  = REPO_DIR / "docs" / "dataset_inventory.md"

PANAGOPOULOU_DIR = DATA_DIR / "panagopoulou" / "images" / "Bistable Images Original"
AMBIBENCH_DIR = DATA_DIR / "ambibench" / "test"
AMBIBENCH_META = AMBIBENCH_DIR / "metadata.jsonl"

EXPANSION_FINAL_DIR = DATA_DIR / "expansion" / "final"

# Group → (aspect_pair, aspect_a, aspect_b) for expansion rows. Keys are the
# group slugs used in the expansion pipeline.
EXPANSION_GROUP_METADATA: dict[str, tuple[str, str, str]] = {
    "duck_rabbit":      ("Duck-Rabbit",     "duck",              "rabbit"),
    "face_vase":        ("Face ↔ Vase",     "two faces",         "vase"),
    "hidden_face":      ("Hidden-Face",     "nature scene",      "human face"),
    "schroeder_stairs": ("Schroeder Stairs","upward",            "downward"),
    "necker_cube":      ("Necker Cube",     "cube from above",   "cube from below"),
    "young_old_woman":  ("Young-Old Woman", "young woman",       "old woman"),
}

# Canonical-group mapping from the (often different) aspect_pair strings in the
# raw Panagopoulou / AmbiBench inventories onto the canonical group slugs used
# for Phase 2 analysis. Anything unmapped gets bucketed as 'other_singleton'.
CANONICAL_GROUP: dict[str, str] = {
    # Duck-Rabbit
    "Duck-Rabbit":            "duck_rabbit",
    "Duck / Rabbit":          "duck_rabbit",
    # Face ↔ Vase family
    "Rubin Vase":             "face_vase",
    "Takashima":              "face_vase",
    "Vase / Profiles":        "face_vase",
    "Face ↔ Vase":            "face_vase",
    # Hidden-Face family
    "Tree / Faces":           "hidden_face",
    "Hidden-Face":            "hidden_face",
    "Plant / Faces":          "hidden_face",
    "Leaf / Face":            "hidden_face",
    "Rock Formation / Face":  "hidden_face",
    "Sealife / Faces":        "hidden_face",
    "Flower / Faces":         "hidden_face",
    "Wolf / Face":            "hidden_face",
    "Face / Building":        "hidden_face",
    "Face / Head":            "hidden_face",
    # Schroeder Stairs
    "Schroeder Stairs":       "schroeder_stairs",
    # Necker Cube
    "Necker Cube":            "necker_cube",
    # Young-Old Woman family
    "Young-Old Woman":          "young_old_woman",
    "Young Woman / Old Woman":  "young_old_woman",
    "Elderly Man / Young Woman": "young_old_woman",
    # Dropped groups we still want to name
    "Grimace-Begger":         "grimace_begger_DROPPED",
    "Spinning Dancer":        "spinning_dancer_DROPPED",
    "geometric-illusion":     "geometric_ambibench_singleton",
}

PRIMARY_GROUPS   = ("duck_rabbit", "face_vase", "hidden_face", "schroeder_stairs")
SECONDARY_GROUPS = ("necker_cube", "young_old_woman")


def _canonical_group(aspect_pair: str) -> str:
    return CANONICAL_GROUP.get(aspect_pair, "other_singleton")

PANAGOPOULOU_AT_PAIRS = {
    # Aspect pairs from the Panagopoulou et al. release
    # (github.com/artemisp/Bistable-Illusions-MLLMs)
    "rubin_vase":         ("Rubin Vase",          "two faces", "vase"),
    "necker_cube":        ("Necker Cube",         "cube above", "cube below"),
    "necker_lattice":     ("Necker Cube",         "cube above", "cube below"),
    "duck_rabbit":        ("Duck-Rabbit",         "duck", "rabbit"),
    "young_old_woman":    ("Young-Old Woman",     "young woman", "old woman"),
    "cat_dog":            ("Cat-Dog",             "cat", "dog"),
    "grimace_begger":     ("Grimace-Begger",      "face grimacing", "begger"),
    "idaho_face":         ("Idaho-Face",          "state of Idaho", "face"),
    "lion_gorilla_tree":  ("Lion-Gorilla-Tree",   "lion and gorilla", "tree"),
    "spining_dancer":     ("Spinning Dancer",     "clockwise", "counter-clockwise"),
    "woman_trumpeter":    ("Woman-Trumpeter",     "woman face", "saxophonist"),
    "schroeder_stairs":   ("Schroeder Stairs",    "upright", "sideways"),
    "raven-bear":         ("Raven-Bear",          "bird", "bear"),
    "raven_bear":         ("Raven-Bear",          "bird", "bear"),
    "takashima":          ("Takashima",           "two faces", "vase"),
}

GEOMETRIC_BISTABLE_SIGNATURES = (
    # Tight: only canonical bistable / seeing-as figures, not generic optical
    # illusions (Müller-Lyer, Kanizsa triangle, etc.) which are perceptual but
    # not bistable.
    "necker", "schroeder", "rubin", "impossible",
    "duck_rabbit", "young_old",
)


def _panagopoulou_pair_for(filename: str) -> tuple[str, str, str]:
    """Map a Panagopoulou filename to (aspect_pair, aspect_a, aspect_b)."""
    base = filename.lower()
    for prefix, (pair, a, b) in PANAGOPOULOU_AT_PAIRS.items():
        if base.startswith(prefix):
            return pair, a, b
    return ("unknown", "", "")


def _ambibench_aspect_pair(answer: str) -> tuple[str, str, str]:
    """Parse an AmbiBench bistable answer like 'duck, rabbit' or 'duck; rabbit'."""
    parts = re.split(r"[,;]\s*", answer.strip(), maxsplit=1)
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        pair = f"{a} / {b}".title()
        return pair, a, b
    return (answer.strip().title() or "unknown", answer.strip(), "")


def collect_panagopoulou() -> list[dict]:
    rows = []
    images = sorted(p for p in PANAGOPOULOU_DIR.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for i, p in enumerate(images):
        pair, a, b = _panagopoulou_pair_for(p.name)
        rows.append({
            "id": f"pana_{i:03d}",
            "source": "panagopoulou_2024",
            "category": "bistable",
            "aspect_pair": pair,
            "aspect_a": a,
            "aspect_b": b,
            "aspect_label": "bistable",
            "file_path": str(p),
            "license": "research-only (paper distribution)",
            "notes": "",
        })
    return rows


def collect_expansion_final() -> list[dict]:
    """Rows for everything under data/expansion/final/<group>/<category>/<file>.
    category ∈ {bistable, control_a, control_b}."""
    if not EXPANSION_FINAL_DIR.exists():
        return []
    rows: list[dict] = []
    # Deterministic order: group, category, filename
    for group_dir in sorted(EXPANSION_FINAL_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        meta = EXPANSION_GROUP_METADATA.get(group)
        if meta is None:
            continue
        pair, a, b = meta
        for category_dir in sorted(group_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name  # 'bistable' | 'control_a' | 'control_b'
            aspect_label = {
                "bistable":  "bistable",
                "control_a": "a",
                "control_b": "b",
            }.get(category, "bistable")
            row_category = "bistable" if category == "bistable" else "control"
            for i, img in enumerate(sorted(p for p in category_dir.iterdir()
                                           if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})):
                # Source inferred from filename prefix set by ingest script:
                #   wiki_, prog_, sdxlsamp_, sdxlbulk_
                name = img.name
                if name.startswith("wiki_"):
                    source = "wikimedia_commons"
                    license_ = "public domain / CC (see sidecar .yaml)"
                elif name.startswith("prog_"):
                    source = "programmatic"
                    license_ = "generated by scripts/render_necker_cubes.py"
                elif name.startswith("sdxlsamp_"):
                    source = "sdxl"
                    license_ = "SDXL-generated (CreativeML OpenRAIL-M)"
                elif name.startswith("sdxlbulk_"):
                    source = "sdxl"
                    license_ = "SDXL-generated (CreativeML OpenRAIL-M)"
                else:
                    source = "unknown"
                    license_ = ""
                id_prefix = {
                    "bistable":  "exp",
                    "control_a": "ca",
                    "control_b": "cb",
                }[category]
                rows.append({
                    "id": f"{id_prefix}_{group[:4]}_{i:03d}",
                    "source": source,
                    "category": row_category,
                    "aspect_pair": pair,
                    "aspect_a": a,
                    "aspect_b": b,
                    "aspect_label": aspect_label,
                    "file_path": str(img),
                    "license": license_,
                    "notes": f"expansion/{group}/{category}/{name}",
                })
    return rows


def collect_ambibench() -> list[dict]:
    metadata = []
    with AMBIBENCH_META.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metadata.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Bistable: take all unique files. PREFER bistable_open over bistable_choice
    # because bistable_choice answers are multiple-choice letters ("A", "B")
    # while bistable_open has the actual descriptive aspect pair ("duck, rabbit").
    bistable: dict[str, dict] = {}
    for r in metadata:
        t = r.get("type", "")
        if not t.startswith("bistable"):
            continue
        fn = r["file_name"]
        existing = bistable.get(fn)
        if existing is None:
            bistable[fn] = r
        elif existing.get("type") == "bistable_choice" and t == "bistable_open":
            bistable[fn] = r   # upgrade to the open-ended answer row

    # Geometric: filter to bistable-suggestive names AND require local file
    geometric = {}
    for r in metadata:
        if r.get("type") == "geometric":
            fn_lower = r["file_name"].lower()
            if any(s in fn_lower for s in GEOMETRIC_BISTABLE_SIGNATURES):
                local_path = AMBIBENCH_DIR / r["file_name"]
                if not local_path.exists():
                    continue   # don't include files we haven't downloaded
                fn = r["file_name"]
                if fn not in geometric:
                    geometric[fn] = r

    rows = []
    for i, (fn, r) in enumerate(sorted(bistable.items())):
        pair, a, b = _ambibench_aspect_pair(r["answer"])
        local_path = AMBIBENCH_DIR / fn
        rows.append({
            "id": f"ambi_b_{i:03d}",
            "source": "ambibench_2025",
            "category": "bistable",
            "aspect_pair": pair,
            "aspect_a": a,
            "aspect_b": b,
            "aspect_label": "bistable",
            "file_path": str(local_path),
            "license": "MIT",
            "notes": f"q: {r['question'][:80]}{'...' if len(r['question'])>80 else ''}",
        })
    for i, (fn, r) in enumerate(sorted(geometric.items())):
        local_path = AMBIBENCH_DIR / fn
        rows.append({
            "id": f"ambi_g_{i:03d}",
            "source": "ambibench_2025",
            "category": "geometric",
            "aspect_pair": "geometric-illusion",
            "aspect_a": "",
            "aspect_b": "",
            "aspect_label": "bistable",
            "file_path": str(local_path),
            "license": "MIT",
            "notes": f"answer: {r['answer'][:60]}",
        })
    return rows


def main() -> int:
    print("[1/4] collecting Panagopoulou", flush=True)
    pana = collect_panagopoulou()
    print(f"      {len(pana)} rows", flush=True)

    print("[2/4] collecting AmbiBench (bistable + filtered geometric)", flush=True)
    ambi = collect_ambibench()
    n_bist = sum(1 for r in ambi if r["category"] == "bistable")
    n_geom = sum(1 for r in ambi if r["category"] == "geometric")
    print(f"      {n_bist} bistable + {n_geom} geometric = {len(ambi)} rows", flush=True)

    print("[2b/4] collecting expansion/final (bistable + controls)", flush=True)
    exp = collect_expansion_final()
    exp_bist = sum(1 for r in exp if r["category"] == "bistable")
    exp_ctrl = sum(1 for r in exp if r["category"] == "control")
    print(f"      {exp_bist} bistable + {exp_ctrl} controls = {len(exp)} rows", flush=True)

    rows = pana + ambi + exp

    # Verify all paths exist
    missing = [r for r in rows if not Path(r["file_path"]).exists()]
    if missing:
        print(f"      WARNING: {len(missing)} rows reference missing files:", flush=True)
        for r in missing[:5]:
            print(f"        {r['file_path']}", flush=True)

    print(f"[3/4] writing {OUT_CSV}", flush=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary stats for the markdown
    by_source = Counter(r["source"] for r in rows)
    by_pair = defaultdict(int)
    for r in rows:
        by_pair[(r["source"], r["aspect_pair"])] += 1

    # Per-canonical-group, per-aspect-label counts across all sources
    by_cgroup_cat: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        cg = _canonical_group(r["aspect_pair"])
        r["_canonical_group"] = cg
        by_cgroup_cat[(cg, r["aspect_label"])] += 1

    print(f"[4/4] writing {OUT_MD}", flush=True)
    md_lines = [
        "# Unified stimulus inventory",
        "",
        f"Total: **{len(rows)}** rows across **{len(by_source)}** sources.",
        f"Bistable rows: **{sum(1 for r in rows if r['category']=='bistable')}**. "
        f"Control rows: **{sum(1 for r in rows if r['category']=='control')}**. "
        f"Geometric rows: **{sum(1 for r in rows if r['category']=='geometric')}**.",
        "",
        "Source CSV (regenerated by `scripts/build_dataset_inventory.py`):",
        f"`{OUT_CSV}` — gitignored, lives under $ASPECT_SCRATCH.",
        "",
        "## Counts by source",
        "",
        "| Source | Count |",
        "|---|---|",
    ]
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        md_lines.append(f"| `{src}` | {n} |")
    md_lines.append(f"| **total** | **{len(rows)}** |")

    md_lines += [
        "",
        "## Per-canonical-group × aspect-label counts (Phase 2 source of truth)",
        "",
        "For Phase 2 feature-identification, **primary** groups need ≥ 4 bistable and ≥ 10 of each pure-aspect control. The three remaining buckets (`other_singleton`, `*_DROPPED`, `geometric_ambibench_singleton`) are not targeted by Phase 2.",
        "",
        "| Canonical group | Role | bistable | pure-A | pure-B | Flag |",
        "|---|---|---:|---:|---:|---|",
    ]
    all_cgroups = sorted({k[0] for k in by_cgroup_cat.keys()})
    def _role(g: str) -> str:
        if g in PRIMARY_GROUPS:   return "primary"
        if g in SECONDARY_GROUPS: return "secondary"
        if g.endswith("_DROPPED"): return "dropped"
        return "other"
    def _flag(g: str, bistable: int, pa: int, pb: int) -> str:
        if g in PRIMARY_GROUPS or g in SECONDARY_GROUPS:
            flags = []
            if bistable < 4: flags.append("bistable<4")
            if pa < 10:      flags.append(f"pure-A<10 (n={pa})")
            if pb < 10:      flags.append(f"pure-B<10 (n={pb})")
            return "; ".join(flags) if flags else "✅"
        return ""
    for g in all_cgroups:
        bistable = by_cgroup_cat.get((g, "bistable"), 0)
        pa = by_cgroup_cat.get((g, "a"), 0)
        pb = by_cgroup_cat.get((g, "b"), 0)
        role = _role(g)
        flag = _flag(g, bistable, pa, pb)
        md_lines.append(f"| `{g}` | {role} | {bistable} | {pa} | {pb} | {flag} |")

    md_lines += [
        "",
        "## Schema",
        "",
        "Columns in `dataset_inventory.csv`:",
        "",
        "- `id` — stable per-row id (`pana_NNN`, `ambi_b_NNN`, `exp_<group>_NNN`, `ca_/cb_<group>_NNN`)",
        "- `source` — provenance: `panagopoulou_2024`, `ambibench_2025`, `wikimedia_commons`, `programmatic`, `sdxl`",
        "- `category` — `bistable` (ambiguous image), `control` (disambiguated single-aspect), `geometric` (legacy)",
        "- `aspect_pair` — human-readable name of the aspect pair (e.g. `Duck-Rabbit`)",
        "- `aspect_a`, `aspect_b` — the two competing percepts as used in Phase 1 prompting",
        "- `aspect_label` — `bistable` for bistable rows; `a` or `b` for control rows (identifies which aspect the control image depicts)",
        "- `file_path` — absolute path under $ASPECT_SCRATCH (regenerate if the scratch tier is cleared)",
        "- `license` — upstream license text",
        "- `notes` — free-text",
        "",
        "## Curation rules",
        "",
        "- **Panagopoulou** (29 images): all kept — already curated by original authors.",
        "- **AmbiBench bistable** (55 unique files): all kept.",
        f"- **AmbiBench geometric** (112 unique files in raw): filtered to **{n_geom}** matching bistable-suggestive name signatures: `{GEOMETRIC_BISTABLE_SIGNATURES}`.",
        "- **Expansion (Wikimedia + programmatic + SDXL samples + SDXL bulk)**: curated by user keep/reject pass, 2026-04-24. Decisions CSV archived at `data/expansion/final/decisions_2026-04-24.csv`.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "source scripts/activate.sh",
        "python scripts/build_dataset_inventory.py",
        "```",
    ]
    OUT_MD.write_text("\n".join(md_lines) + "\n")

    print()
    print(f"==> wrote {len(rows)} rows total: {len(pana)} Panagopoulou + {n_bist} AmbiBench bistable + {n_geom} AmbiBench geometric", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
