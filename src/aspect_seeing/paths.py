"""Filesystem layout. Set ASPECT_SCRATCH (and optionally ASPECT_REPO) via
environment; see README. No personal defaults are baked in."""
import os
from pathlib import Path

ASPECT_REPO = Path(os.environ.get("ASPECT_REPO", Path(__file__).resolve().parents[2]))
ASPECT_SCRATCH = Path(os.environ["ASPECT_SCRATCH"]) if os.environ.get("ASPECT_SCRATCH") \
    else ASPECT_REPO / "scratch"

DATA_DIR    = ASPECT_SCRATCH / "data"
OUTPUTS_DIR = ASPECT_SCRATCH / "outputs"
MODELS_DIR  = ASPECT_SCRATCH / "models"
LOGS_DIR    = ASPECT_SCRATCH / "logs"
FIG_DIR     = OUTPUTS_DIR / "figures"
