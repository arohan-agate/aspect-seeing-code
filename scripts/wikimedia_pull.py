"""Pull bistable-illusion source images from Wikimedia Commons in three passes.

This script merges three single-pass pullers into one CLI. The passes are
independent and intended to be run in order; later passes read the manifest
written by earlier passes to avoid re-downloading the same file.

  curated  (pass 1) — Hard-coded Commons file titles (the TARGETS list), one
                      per canonical illusion. Resolves each title to a direct
                      image URL, skips SVGs (CLIP needs raster), records the
                      license, and writes an attribution .yaml sidecar. All
                      titles are expected to be public-domain by age or on
                      stable PD/CC pages. This pass *overwrites* the manifest.

  search   (pass 2) — Generic per-group MediaWiki search queries (the SEARCHES
                      list) to fill gaps left by the curated pass. Filters to
                      raster formats at least 400 px on each side and at most
                      25 MB. Appends to (and de-duplicates) the manifest.

  targeted (pass 3) — Schröder-stairs / Necker-cube-specific multilingual
                      queries plus 19th-century psychology-plate searches, with
                      a looser 300 px minimum (small PD scans are still usable
                      for CLIP's 336^2 input). Appends to / de-duplicates the
                      manifest.

The three passes deliberately keep distinct constants and slightly different
image-info / download / sidecar behavior; shared helpers (file-title search,
filename sanitizing, manifest de-duplication) are factored out, while the parts
that genuinely differ between passes are kept per-pass.

Run (no GPU needed):
    python scripts/wikimedia_pull.py                       # all three, in order
    python scripts/wikimedia_pull.py --passes curated      # just pass 1
    python scripts/wikimedia_pull.py --passes search targeted
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import requests

from aspect_seeing.paths import DATA_DIR

OUT_ROOT = DATA_DIR / "expansion" / "wikimedia"
# Wikimedia requires a real contact in the User-Agent. Set this to your own
# project/contact before running, or Commons may rate-limit or block requests.
UA = "aspect-seeing/0.0.1 (research; your-contact@example.com)"
API = "https://commons.wikimedia.org/w/api.php"


# ---------------------------------------------------------------------------
# Pass 1 ("curated"): hard-coded Commons file titles.
# ---------------------------------------------------------------------------

# Curated list: (group_slug, file_title_on_commons, rationale)
# Titles are best-effort; the script prints which ones 404 so the list
# can be updated. All are expected to be public-domain-by-age.
TARGETS: list[tuple[str, str, str]] = [
    # Duck-Rabbit
    ("duck_rabbit", "File:Duck-Rabbit_illusion.jpg",
     "Canonical duck-rabbit illusion (public domain, pre-1928 origin)."),
    ("duck_rabbit", "File:Kaninchen_und_Ente.png",
     "Fliegende Blätter 1892 original — predates Jastrow 1899."),
    ("duck_rabbit", "File:Jastrow-Hase-Ente.svg",
     "SVG redraw of Jastrow 1899; PD by age."),
    ("duck_rabbit", "File:Duck-Rabbit_illusion.png",
     "Alternate rendering of the canonical figure."),

    # Face <-> Vase (Rubin original)
    ("face_vase",  "File:Rubin2.jpg",
     "Rubin 1915 original vase illusion; PD by age."),
    ("face_vase",  "File:Cup_or_faces_paradox.svg",
     "SVG redraw of Rubin's vase; PD."),

    # Hidden-Face (Arcimboldo)
    ("hidden_face", "File:Giuseppe_Arcimboldo_-_Vertumnus_-_Google_Art_Project.jpg",
     "Arcimboldo's Vertumnus (1591) — Emperor Rudolf II as composite of fruits/vegetables; PD by age."),
    ("hidden_face", "File:Arcimboldo_Four_Seasons_01.jpg",
     "Arcimboldo Four Seasons (1563); PD by age."),
    ("hidden_face", "File:Arcimboldo_Summer_1563.jpg",
     "Arcimboldo Summer portrait (1563); PD by age."),

    # Schroeder Stairs
    ("schroeder_stairs", "File:Schroeder_stairs.svg",
     "Schröder 1858 stairs illusion; PD by age."),
    ("schroeder_stairs", "File:Schroder_stairs_animation.gif",
     "Schröder stairs animation — single frame usable; PD by age."),
    ("schroeder_stairs", "File:Schroeder-Treppe.jpg",
     "German wiki version of the Schröder stairs."),
    ("schroeder_stairs", "File:Schroeder_staircase.png",
     "Line-drawing version of the Schröder stairs."),

    # Necker Cube
    ("necker_cube", "File:Necker_cube.svg",
     "Necker 1832 cube; PD by age."),
    ("necker_cube", "File:Necker_cube.png",
     "Alternate raster Necker cube."),
    ("necker_cube", "File:Necker-Würfel.svg",
     "German wiki version of the Necker cube."),

    # Young-Old Woman
    ("young_old_woman", "File:My_Wife_and_My_Mother-in-Law.png",
     "Hill 1915 'My Wife and My Mother-in-Law'; PD by age."),
    ("young_old_woman", "File:My_Wife_and_My_Mother-in-Law.jpg",
     "Alternate raster version."),
    ("young_old_woman", "File:Boring_figure.png",
     "Boring 1930 popularization of Hill's illusion; PD by age."),
]


# ---------------------------------------------------------------------------
# Pass 2 ("search"): generic per-group search queries.
# ---------------------------------------------------------------------------

# Per-group search queries and how many to keep: (group, query, keep_n)
SEARCH_QUERIES: list[tuple[str, str, int]] = [
    ("duck_rabbit",      "duck rabbit illusion",        3),
    ("face_vase",        "rubin vase illusion",         3),
    ("hidden_face",      "arcimboldo",                  3),
    ("hidden_face",      "hidden face illusion painting", 2),
    ("schroeder_stairs", "schroeder stairs",            3),
    ("schroeder_stairs", "reversible staircase",        2),
    ("necker_cube",      "necker cube",                 3),
    ("young_old_woman",  "my wife my mother in law",    2),
    ("young_old_woman",  "boring figure illusion",      2),
]

SEARCH_MIN_WIDTH = 400
SEARCH_MIN_HEIGHT = 400


# ---------------------------------------------------------------------------
# Pass 3 ("targeted"): Schroeder/Necker-specific multilingual queries.
# ---------------------------------------------------------------------------

# (group, query, keep_n)
TARGETED_QUERIES: list[tuple[str, str, int]] = [
    # Suggested queries
    ("schroeder_stairs", "Schröder Treppe",               3),
    ("schroeder_stairs", "reversible stairs illusion",    3),
    ("schroeder_stairs", "Schroeder staircase illusion",  3),
    ("necker_cube",      "Necker cube 1832",              3),
    ("necker_cube",      "Philosophical Magazine Necker", 2),
    # 19th-c. PD psychology plates
    ("schroeder_stairs", "Popular Science Monthly illusion staircase", 2),
    ("necker_cube",      "Popular Science Monthly cube illusion",      2),
    ("necker_cube",      "Würfel Kippfigur",                            2),
    ("schroeder_stairs", "escalier réversible illusion",                2),
    ("necker_cube",      "cube de Necker",                              2),
]

TARGETED_MIN_WIDTH = 300       # looser than pass 2: 19th-c. plates can be small but still usable for CLIP 336²
TARGETED_MIN_HEIGHT = 300


# Shared by passes 2 and 3.
RASTER_EXTS = {".jpg", ".jpeg", ".png"}
MAX_BYTES = 25 * 1024 * 1024   # 25 MB cap


# ---------------------------------------------------------------------------
# Shared helpers (identical across passes).
# ---------------------------------------------------------------------------

def _safe_filename(title: str) -> str:
    """Drop the "File:" prefix and any path separators."""
    base = title[5:] if title.lower().startswith("file:") else title
    return base.replace("/", "_").replace(" ", "_")


def search_files(query: str, limit: int = 8) -> list[str]:
    """Search Commons' File: namespace; returns a list of file titles."""
    r = requests.get(API, headers={"User-Agent": UA}, params={
        "action": "query",
        "list": "search",
        "srnamespace": 6,         # File namespace
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }, timeout=30)
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    return [h["title"] for h in hits]


def _dedup_successes(successes: list[dict]) -> list[dict]:
    """Collapse manifest successes keyed by (title, group)."""
    return list({(s["title"], s["group"]): s for s in successes}.values())


# ---------------------------------------------------------------------------
# Pass 1 helpers (curated). These differ from the search passes: a richer
# imageinfo query (mediatype + metadata + description), an explicit
# "missing" check, and an uncapped download.
# ---------------------------------------------------------------------------

def resolve_image_info(title: str) -> dict | None:
    """Ask MediaWiki API for the direct URL + metadata for a file title."""
    r = requests.get(API, headers={"User-Agent": UA}, params={
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mediatype|metadata|extmetadata",
        "format": "json",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values())) if pages else None
    if page is None or page.get("missing") is not None or "imageinfo" not in page:
        return None
    info = page["imageinfo"][0]
    extmeta = info.get("extmetadata", {})
    return {
        "title": title,
        "url":   info["url"],
        "mime":  info.get("mime"),
        "size":  info.get("size"),
        "width": info.get("width"),
        "height": info.get("height"),
        "license":    (extmeta.get("LicenseShortName") or {}).get("value"),
        "artist":     (extmeta.get("Artist") or {}).get("value"),
        "credit":     (extmeta.get("Credit") or {}).get("value"),
        "description": (extmeta.get("ImageDescription") or {}).get("value"),
    }


def download_uncapped(url: str, dest: Path) -> None:
    """Curated-pass download: no byte cap (curated titles are vetted)."""
    r = requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 14):
            f.write(chunk)


def run_curated() -> int:
    """Pass 1: download the hard-coded TARGETS list and (re)write the manifest."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    succ: list[dict] = []
    fail: list[tuple[str, str]] = []

    for (group, title, rationale) in TARGETS:
        print(f"[try ] {group:<20} {title}", flush=True)
        try:
            info = resolve_image_info(title)
        except Exception as e:
            print(f"       !! API error: {e}", flush=True)
            fail.append((group, f"{title}: API error {e}"))
            continue
        if info is None:
            print(f"       !! not found on Commons", flush=True)
            fail.append((group, f"{title}: not found"))
            continue

        # SVG is problematic for CLIP (rasterization required). Skip SVGs, take raster only.
        if (info.get("mime") or "").endswith("svg+xml"):
            print(f"       -- SVG: skipping (need raster)", flush=True)
            fail.append((group, f"{title}: SVG (skipped)"))
            continue

        ext = (info["url"].rsplit(".", 1)[-1] or "jpg").lower()
        fname = _safe_filename(title.rsplit(".", 1)[0]) + f".{ext}"
        out_path = OUT_ROOT / group / fname
        sidecar  = out_path.with_suffix(out_path.suffix + ".yaml")

        try:
            download_uncapped(info["url"], out_path)
            size_kb = out_path.stat().st_size / 1024
            print(f"       ✓ downloaded {fname} ({size_kb:.0f} KB, {info.get('width')}×{info.get('height')})",
                  flush=True)
        except Exception as e:
            print(f"       !! download failed: {e}", flush=True)
            fail.append((group, f"{title}: download {e}"))
            continue

        # Write sidecar metadata
        sidecar.write_text(
            "# attribution and license for the sibling image\n"
            f"source_api: wikimedia_commons\n"
            f"wikimedia_title: '{title}'\n"
            f"rationale: '{rationale}'\n"
            f"download_url: '{info['url']}'\n"
            f"license: {info.get('license')!r}\n"
            f"artist: {info.get('artist')!r}\n"
            f"credit: {info.get('credit')!r}\n"
        )
        succ.append({"group": group, "title": title, "out_path": str(out_path),
                     "license": info.get("license"),
                     "artist":  info.get("artist")})
        time.sleep(0.5)   # be polite to Commons

    # Summary
    print()
    print("=" * 60, flush=True)
    print(f"successes: {len(succ)} / {len(TARGETS)}", flush=True)
    per_group = Counter(s["group"] for s in succ)
    for g in sorted({t[0] for t in TARGETS}):
        print(f"  {g:<20} {per_group.get(g, 0)}", flush=True)
    if fail:
        print(f"\nfailures ({len(fail)}):", flush=True)
        for g, reason in fail:
            print(f"  {g:<20} {reason}", flush=True)

    (OUT_ROOT / "pull_manifest.json").write_text(
        json.dumps({"successes": succ, "failures": [{"group": g, "reason": r} for g, r in fail]},
                   indent=2)
    )
    print(f"\nwrote {OUT_ROOT / 'pull_manifest.json'}", flush=True)
    return 0 if succ else 1


# ---------------------------------------------------------------------------
# Search-pass helpers (passes 2 and 3). These share a leaner imageinfo query
# (mime instead of mediatype/metadata, no description) and a byte-capped
# download. Pass 2 and pass 3 differ only in their size thresholds, the
# sidecar source_api label, and the summary column width — captured via args.
# ---------------------------------------------------------------------------

def image_info(title: str) -> dict | None:
    """Leaner imageinfo lookup used by both search passes."""
    r = requests.get(API, headers={"User-Agent": UA}, params={
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "format": "json",
    }, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values())) if pages else None
    if page is None or "imageinfo" not in page:
        return None
    info = page["imageinfo"][0]
    extmeta = info.get("extmetadata", {})
    return {
        "title": title,
        "url": info["url"],
        "mime": info.get("mime"),
        "size": info.get("size"),
        "width": info.get("width"),
        "height": info.get("height"),
        "license":     (extmeta.get("LicenseShortName") or {}).get("value"),
        "artist":      (extmeta.get("Artist") or {}).get("value"),
        "credit":      (extmeta.get("Credit") or {}).get("value"),
    }


def download_capped(url: str, dest: Path, max_bytes: int = MAX_BYTES) -> int:
    """Search-pass download: aborts past max_bytes (untrusted search hits)."""
    r = requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=90)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 15):
            f.write(chunk)
            total += len(chunk)
            if total > max_bytes:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"exceeded {max_bytes} B")
    return total


def _run_search_pass(
    searches: list[tuple[str, str, int]],
    *,
    min_width: int,
    min_height: int,
    source_api: str,
    label_width: int,
) -> int:
    """Shared driver for the two search passes (pass 2 and pass 3).

    Both passes search per group, filter to raster formats above a minimum
    size, download with a byte cap, write an attribution sidecar, and
    append/de-duplicate the manifest. They differ only in the inputs passed
    here (queries, size thresholds, sidecar source label, column width).
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "pull_manifest.json"
    already: set[str] = set()
    if manifest_path.exists():
        for s in json.loads(manifest_path.read_text()).get("successes", []):
            already.add(s["title"])
    print(f"[init] {len(already)} titles on disk from previous passes", flush=True)

    succ_new: list[dict] = []
    for (group, query, keep_n) in searches:
        print(f"\n[search] {group:<{label_width}} '{query}'  keep_n={keep_n}", flush=True)
        try:
            titles = search_files(query, limit=10)
        except Exception as e:
            print(f"        !! {e}", flush=True)
            continue
        if not titles:
            print(f"        !! no hits", flush=True)
            continue

        n_kept = 0
        for title in titles:
            if n_kept >= keep_n:
                break
            if title in already:
                print(f"        -- {title}: have", flush=True)
                continue
            info = image_info(title)
            if info is None:
                continue
            ext = "." + (info["url"].rsplit(".", 1)[-1] or "").lower()
            if ext not in RASTER_EXTS:
                print(f"        -- {title}: {ext}", flush=True)
                continue
            if (info.get("width") or 0) < min_width or (info.get("height") or 0) < min_height:
                print(f"        -- {title}: too small ({info.get('width')}×{info.get('height')})",
                      flush=True)
                continue
            if (info.get("size") or 0) > MAX_BYTES:
                print(f"        -- {title}: too big ({info.get('size')} B)", flush=True)
                continue

            fname = _safe_filename(title.rsplit(".", 1)[0]) + ext
            out_path = OUT_ROOT / group / fname
            sidecar  = out_path.with_suffix(out_path.suffix + ".yaml")
            try:
                n = download_capped(info["url"], out_path)
            except Exception as e:
                print(f"        !! download {title}: {e}", flush=True)
                continue
            print(f"        ✓ {title} → {fname} ({n/1024:.0f} KB, "
                  f"{info.get('width')}×{info.get('height')})", flush=True)
            sidecar.write_text(
                "# attribution and license for the sibling image\n"
                f"source_api: {source_api}\n"
                f"wikimedia_title: '{title}'\n"
                f"search_query: '{query}'\n"
                f"download_url: '{info['url']}'\n"
                f"license: {info.get('license')!r}\n"
                f"artist: {info.get('artist')!r}\n"
                f"credit: {info.get('credit')!r}\n"
            )
            succ_new.append({"group": group, "title": title, "out_path": str(out_path),
                             "license": info.get("license"), "artist": info.get("artist")})
            already.add(title)
            n_kept += 1
            time.sleep(0.4)

    # Merge into manifest
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else \
               {"successes": [], "failures": []}
    manifest["successes"].extend(succ_new)
    manifest["successes"] = _dedup_successes(manifest["successes"])
    manifest_path.write_text(json.dumps(manifest, indent=2))

    per = Counter(s["group"] for s in succ_new)
    print()
    print("=" * 60, flush=True)
    print(f"new successes this pass: {len(succ_new)}", flush=True)
    for g in sorted({s[0] for s in searches}):
        print(f"  {g:<{label_width}} +{per.get(g, 0)}", flush=True)
    print(f"\nmanifest updated: {manifest_path}", flush=True)
    return 0


def run_search() -> int:
    """Pass 2: generic per-group search queries, >=400 px."""
    return _run_search_pass(
        SEARCH_QUERIES,
        min_width=SEARCH_MIN_WIDTH,
        min_height=SEARCH_MIN_HEIGHT,
        source_api="wikimedia_commons (search)",
        label_width=20,
    )


def run_targeted() -> int:
    """Pass 3: Schroeder/Necker-specific multilingual queries, looser >=300 px."""
    return _run_search_pass(
        TARGETED_QUERIES,
        min_width=TARGETED_MIN_WIDTH,
        min_height=TARGETED_MIN_HEIGHT,
        source_api="wikimedia_commons (v2 targeted search)",
        label_width=18,
    )


PASSES = {
    "curated": run_curated,
    "search": run_search,
    "targeted": run_targeted,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--passes",
        nargs="+",
        choices=list(PASSES),
        default=list(PASSES),
        help="Which passes to run, in order. Default: all three (curated search targeted).",
    )
    args = parser.parse_args()

    rc = 0
    for name in args.passes:
        print(f"\n########## pass: {name} ##########", flush=True)
        rc = PASSES[name]() or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
