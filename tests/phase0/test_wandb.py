"""Phase 0 smoke test 5: W&B init + log + finish.

Logs one dummy metric to wandb.ai under project='aspect-seeing'. Requires
WANDB_API_KEY in the environment (set via scripts/activate.local.sh, not
checked into git).

Run after `source scripts/activate.sh`.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("WANDB_API_KEY"):
        print("!! WANDB_API_KEY missing; cannot reach wandb.ai", file=sys.stderr)
        return 1

    print("[1/3] importing wandb", flush=True)
    import wandb
    print(f"      wandb {wandb.__version__}", flush=True)
    print(f"      WANDB_DIR = {os.environ.get('WANDB_DIR', '<unset>')}", flush=True)

    print("[2/3] init run", flush=True)
    run = wandb.init(
        project="aspect-seeing",
        name="phase0-smoke",
        tags=["phase0", "smoke"],
        config={"purpose": "phase0 smoke test", "stimulus": "duck_rabbit_1.png"},
        dir=os.environ.get("WANDB_DIR"),
        reinit="finish_previous",
    )
    print(f"      run.id   = {run.id}", flush=True)
    print(f"      run.url  = {run.url}", flush=True)

    print("[3/3] log dummy metric and finish", flush=True)
    for step in range(5):
        wandb.log({"smoke/dummy": step * 0.1, "smoke/sq": step ** 2}, step=step)
    wandb.finish()
    print("      ==> run finished cleanly; check wandb.ai for visibility", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
