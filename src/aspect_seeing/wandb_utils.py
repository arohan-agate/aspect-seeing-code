"""wandb init helpers scoped to this project.

Per design doc §8: each run logs (phase, stimulus id, feature id, activation
vector path, metric snapshot). All runs share the same W&B project so we can
compare across phases.
"""
from __future__ import annotations

import os
from typing import Any

WANDB_PROJECT = "aspect-seeing"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")  # optional; defaults to user


def init_run(
    phase: str,
    run_name: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """Start a W&B run for this project.

    phase:    'phase1' | 'phase2' | ... | 'phase5' (matches design-doc §4).
    run_name: short descriptor, e.g. '2026-04-24-llava-dominance-v1'.
    config:   serialized run config (stimulus set, prompts, seeds, etc.).
    tags:     optional extra tags on top of the phase tag.
    """
    import wandb

    merged_tags = [phase] + (tags or [])
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config or {},
        tags=merged_tags,
        dir=os.environ.get("WANDB_DIR"),
    )
