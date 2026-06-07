# aspect-seeing

Mechanistic-interpretability study of *aspect commitment* in vision-language models.
We caption bistable images (duck–rabbit, Rubin's vase, Necker cube, …) with
LLaVA-1.6-Vicuna-7B, train a TopK sparse autoencoder on the CLIP ViT-L/14 layer
that LLaVA actually consumes (residual stream, layer 22, patch tokens), and ask
where the commitment to one aspect is made: a behavioral baseline over 83 stimuli
(Phase 1), per-aspect SAE feature identification with tie-corrected rank AUROC
(Phase 2), superposition-vs-dominance analysis at the vision tower (Phase 3), and
causal steering of the layer-22 residual stream during captioning (Phase 4).
The framing follows Wittgenstein's seeing / seeing-as distinction.

Paper: [<ARXIV_LINK>](<ARXIV_LINK>)

## Setup

```bash
git clone <REPO_URL> aspect-seeing && cd aspect-seeing
pip install -r env/requirements.txt   # env/requirements.lock.txt has the exact pins
pip install -e .
```

Python ≥ 3.10. `env/requirements.lock.txt` records the exact environment the
results were produced with (torch 2.7.1 + CUDA 12.x).

Environment variables (see `src/aspect_seeing/paths.py`):

| Variable | Required | Meaning |
|---|---|---|
| `ASPECT_SCRATCH` | recommended | Root for all large artifacts (data, activations, models, outputs). Defaults to `./scratch` inside the repo — point it at fast, large storage on a cluster. |
| `ASPECT_REPO` | optional | Repo root override; inferred from the package location otherwise. |
| `ASPECT_ENV` | optional | Conda/venv prefix that `scripts/activate.sh` and the SLURM scripts activate. |
| `HF_TOKEN` | if needed | Hugging Face token for gated model downloads. |
| `WANDB_API_KEY`, `WANDB_ENTITY` | optional | Training/eval runs log to the `aspect-seeing` W&B project; omit to run without logging. |

`source scripts/activate.sh` sets cache/tmp dirs under `$ASPECT_SCRATCH` and
sources a gitignored `scripts/activate.local.sh` if you create one for secrets.

On a SLURM cluster, submit the provided batch scripts with your own account and
GPU partition (the `#SBATCH` directives intentionally omit them):

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/train_sae.sbatch
```

## Models & data

No model weights, activations, or the trained SAE are shipped in this repo.
`$ASPECT_SCRATCH/models/` (= `MODELS_DIR`) should hold local snapshots of:

| Model | HF id | Used for |
|---|---|---|
| LLaVA-1.6-Vicuna-7B | `llava-hf/llava-v1.6-vicuna-7b-hf` | captioning + steering (vision_feature_layer = −2 = CLIP layer 22) |
| CLIP ViT-L/14-336 | `openai/clip-vit-large-patch14-336` | activation caching / SAE training |
| Qwen3-8B | `Qwen/Qwen3-8B` | caption judge (`enable_thinking=False`) |
| CLIP-Scope layer-22 SAE | `lewington/CLIP-ViT-L-scope` | Phase 0 reference SAE quality check |
| SDXL base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | pure-aspect control generation |

`python scripts/download_models.py` fetches them into `MODELS_DIR`.

Stimuli and controls live under `$ASPECT_SCRATCH/data/`:

- **Panagopoulou et al. 2024** bistable set (29 images): `bash scripts/download_panagopoulou.sh`
  (upstream: <https://github.com/artemisp/Bistable-Illusions-MLLMs>; images are not redistributed here)
- **AmbiBench** (bistable subset; MIT): download `https://huggingface.co/datasets/BLNL/AmbiBench`
  into `$ASPECT_SCRATCH/data/ambibench/` (the scripts expect `ambibench/test/` + `metadata.jsonl`)
- **Wikimedia Commons** historical originals: `python scripts/wikimedia_pull.py`
  (set a real contact e-mail in the script's User-Agent first)
- **SDXL pure-aspect controls**: `python scripts/sdxl_generate.py --mode bulk` (then hand-verify —
  generated controls can leak the opposite aspect), `python scripts/regen_control_b.py` for the
  regenerated control-B sets
- **Programmatic Necker renders**: `python scripts/render_necker_cubes.py`
- **Inventory** (the per-image source/license/retention table): `python scripts/build_dataset_inventory.py`

SAE training additionally needs a local copy of **CC3M** as webdataset tars
(pass `--tars-dir`); the 200K-image activation cache is ~236 GB in fp16.

## Reproducing the pipeline

Run order (each step also has a matching `scripts/slurm/*.sbatch` where long-running):

| Step | Command | Output (under `$ASPECT_SCRATCH/outputs/`) |
|---|---|---|
| 0. SAE reference check (optional) | `python scripts/phase0_sae_quality.py` | `phase0_sae_quality.csv` |
| 1. Cache CLIP layer-22 activations | `python scripts/cache_activations.py --tars-dir /path/to/cc3m` | `…/activations/clip-L14-layer22/` shards |
| 2. Train the TopK SAE (k=32, 65 536 features) | `python scripts/train_sae.py` or `sbatch … scripts/slurm/train_sae.sbatch` | `MODELS_DIR/own-sae-clip-L14-layer22/best.pt` |
| 3. Evaluate the SAE (held-out EV) | `python scripts/eval_sae.py` | metrics to stdout / W&B |
| 4. Phase 1 — behavioral baseline | `python scripts/phase1_behavioral.py` then `python scripts/phase1_force_choice.py` | `phase1/*.csv` |
| 5. Figure 1 | `python scripts/phase1_dominance_2d.py` | `figures/phase1_dominance_2d.pdf` |
| 6. Phase 2 — per-aspect feature ID (tie-corrected AUROC) | `python scripts/phase2_feature_id.py` or `sbatch … scripts/slurm/phase2_feature_id.sbatch` | `phase2/features_<group>_v3.csv` |
| 7. Phase 3 — superposition vs dominance | `python scripts/phase3_superposition.py` | `phase3/superposition_*.csv` |
| 8. Phase 4 — causal steering at CLIP layer 22 | `python scripts/phase4_steering.py` | `phase4/success_vs_alpha.csv` |

Phases 1 and 4 keep LLaVA and the Qwen3 judge resident simultaneously
(~32 GB peak); an 80 GB GPU is comfortable, a 48 GB GPU works for the
single-model steps. GPU smoke tests for every component are under
`tests/phase0/`.

## Repo layout

```
src/aspect_seeing/     package: paths config, Qwen3 judge, W&B helpers
scripts/               pipeline + dataset-construction scripts (entry points above)
scripts/slurm/         SLURM batch wrappers for the long-running steps
tests/phase0/          per-component GPU smoke tests
env/                   requirements.txt (loose) and requirements.lock.txt (exact pins)
docs/design_doc.md     the experimental design the paper executed
docs/dataset_inventory.md  stimulus/control inventory and curation rules
```
