# aspect-seeing

Where does a vision-language model commit to one reading of a bistable image? We caption duck–rabbit, Rubin's-vase, Necker-cube and similar stimuli with LLaVA-1.6-Vicuna-7B, train a TopK sparse autoencoder on the CLIP ViT-L/14 layer LLaVA consumes (layer 22, patch tokens), and locate the commitment with a behavioral baseline, per-aspect feature identification, a superposition-vs-dominance test, and causal steering of the layer-22 residual stream. Framing follows Wittgenstein's seeing / seeing-as distinction.

Paper: https://arxiv.org/abs/2606.08031

## Setup

```bash
git clone <REPO_URL> aspect-seeing && cd aspect-seeing
pip install -r env/requirements.txt   # exact pins in requirements.lock.txt
pip install -e .
```

Python ≥ 3.10, torch 2.7.1 (CUDA 12.x). Set `ASPECT_SCRATCH` to large/fast storage for data, activations, models, and outputs (defaults to `./scratch`). Optional: `HF_TOKEN` for gated downloads; `WANDB_API_KEY` / `WANDB_ENTITY` for logging.

## Models & data

Weights, activations, and the trained SAE are not shipped. `python scripts/download_models.py` fetches models into `$ASPECT_SCRATCH/models/`:

| Model | HF id | Role |
|---|---|---|
| LLaVA-1.6-Vicuna-7B | `llava-hf/llava-v1.6-vicuna-7b-hf` | captioning + steering |
| CLIP ViT-L/14-336 | `openai/clip-vit-large-patch14-336` | activation caching / SAE training |
| Qwen3-8B | `Qwen/Qwen3-8B` | caption judge |
| CLIP-Scope SAE | `lewington/CLIP-ViT-L-scope` | Phase 0 reference check |
| SDXL base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | control generation |

Stimuli/controls go under `$ASPECT_SCRATCH/data/`. Helpers: `download_panagopoulou.sh` (29-image bistable set, not redistributed here); AmbiBench bistable subset from `huggingface.co/datasets/BLNL/AmbiBench`; `wikimedia_pull.py` (historical originals); `sdxl_generate.py` + `regen_control_b.py` (pure-aspect controls — hand-verify); `render_necker_cubes.py`. SAE training needs CC3M as webdataset tars.

## Pipeline

| Step | Command |
|---|---|
| Train SAE (k=32, 65 536 feats) | `cache_activations.py --tars-dir <cc3m>` → `train_sae.py` → `eval_sae.py` |
| Phase 1 — behavioral | `phase1_behavioral.py`, `phase1_force_choice.py`, `phase1_dominance_2d.py` |
| Phase 2 — feature ID (tie-corrected AUROC) | `phase2_feature_id.py` |
| Phase 3 — superposition vs dominance | `phase3_superposition.py` |
| Phase 4 — causal steering | `phase4_steering.py` |

Long-running steps have matching `scripts/slurm/*.sbatch` wrappers; outputs land under `$ASPECT_SCRATCH/outputs/`. Component smoke tests are in `tests/phase0/`.

## Layout

```
src/aspect_seeing/   package: paths config, Qwen3 judge, W&B helpers
scripts/             pipeline + dataset scripts; scripts/slurm/ batch wrappers
tests/phase0/        per-component smoke tests
env/                 requirements (loose + locked)
docs/                dataset inventory + curation rules
```
