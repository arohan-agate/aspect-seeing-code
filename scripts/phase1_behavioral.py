"""Phase 1 — Behavioral baseline (design doc §4.1).

For each bistable stimulus, caption with LLaVA-1.6-7B under two neutral
prompts × 20 seeds = 40 samples. Classify each caption with the Qwen3-8B
judge into {aspect_a, aspect_b, both, neither}. Compute per-stimulus
dominance score = |P(aspect_a) - P(aspect_b)|.

Outputs:
    outputs/phase1/behavioral_baseline.csv
    outputs/figures/phase1_dominance.pdf

Runs on one A100 80GB (model coexistence verified in Phase 0:
LLaVA+Qwen3 peak ~32 GB). Uses do_sample=True, temperature=0.7, and
num_return_sequences=10 × 2 calls per (stimulus, prompt) for the 20
seeds. wandb run name: phase1-behavioral-baseline.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from aspect_seeing.paths import ASPECT_REPO, DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FIG_DIR

REPO_DIR = ASPECT_REPO
INVENTORY_CSV = DATA_DIR / "dataset_inventory.csv"
OUT_DIR = OUTPUTS_DIR / "phase1"
OUT_CSV = OUT_DIR / "behavioral_baseline.csv"
OUT_FIG = FIG_DIR / "phase1_dominance.pdf"
OUT_SUMMARY_JSON = OUT_DIR / "dominance_summary.json"

LLAVA_PATH = MODELS_DIR / "llava-v1.6-vicuna-7b-hf"

PROMPTS = {
    "neutral_what": "What is in this image?",
    "neutral_describe": "Describe this image in one sentence.",
}

N_SEEDS = 20
TEMPERATURE = 0.7
SEQS_PER_CALL = 10           # 2 calls × 10 = 20 seeds per (stimulus, prompt)
MAX_NEW_TOKENS = 48
CSV_HEADERS = [
    "stimulus_id", "aspect_pair", "aspect_a", "aspect_b",
    "prompt_type", "seed", "caption", "judge_label", "judge_raw_output",
]


def _build_llava_prompt(user_text: str) -> str:
    return (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\n{user_text} ASSISTANT:"
    )


def _trim_after_assistant(text: str) -> str:
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1]
    return text.strip()


def _load_stimuli() -> list[dict]:
    rows = []
    with INVENTORY_CSV.open() as f:
        for r in csv.DictReader(f):
            # Skip rows without a clean two-aspect classification target
            # (e.g. the one 'geometric-illusion' row has empty aspect_a/aspect_b).
            if not r["aspect_a"] or not r["aspect_b"]:
                continue
            rows.append(r)
    return rows


def main() -> int:
    import numpy as np
    import torch
    import PIL.Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    import wandb

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    stimuli = _load_stimuli()
    n_stim = len(stimuli)
    n_total = n_stim * len(PROMPTS) * N_SEEDS
    print(f"[init] {n_stim} stimuli × {len(PROMPTS)} prompts × {N_SEEDS} seeds = {n_total} generations",
          flush=True)

    # ---- load models ----
    print("[load] LLaVA-1.6-7B (bf16)", flush=True)
    t0 = time.time()
    processor = LlavaNextProcessor.from_pretrained(str(LLAVA_PATH), use_fast=True)
    llava = LlavaNextForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH), dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True,
    )
    llava.eval()
    print(f"       loaded in {time.time()-t0:.1f}s", flush=True)

    print("[load] Qwen3-8B judge", flush=True)
    from aspect_seeing.eval.judge import Judge, JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, _parse_label
    judge = Judge()
    judge._ensure_loaded()
    print(f"       peak VRAM after both models: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    # ---- wandb ----
    print("[wandb] init", flush=True)
    run = wandb.init(
        project="aspect-seeing",
        name="phase1-behavioral-baseline",
        tags=["phase1", "behavioral"],
        dir=os.environ.get("WANDB_DIR"),
        config={
            "n_stimuli": n_stim,
            "n_prompts": len(PROMPTS),
            "n_seeds": N_SEEDS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seqs_per_call": SEQS_PER_CALL,
            "inventory_csv": str(INVENTORY_CSV),
        },
    )

    # ---- generate + judge ----
    # CSV resume: if partial file exists, skip already-done (stimulus_id, prompt_type) combos
    done_keys: set[tuple[str, str]] = set()
    if OUT_CSV.exists():
        with OUT_CSV.open() as f:
            for r in csv.DictReader(f):
                done_keys.add((r["stimulus_id"], r["prompt_type"]))
        print(f"[resume] found {len(done_keys)} already-done (stimulus, prompt) pairs", flush=True)

    csv_exists = OUT_CSV.exists() and OUT_CSV.stat().st_size > 0
    f_csv = OUT_CSV.open("a", newline="")
    writer = csv.DictWriter(f_csv, fieldnames=CSV_HEADERS)
    if not csv_exists:
        writer.writeheader()

    t_start = time.time()
    n_done = 0
    n_skipped = 0
    for si, stim in enumerate(stimuli):
        img_path = Path(stim["file_path"])
        if not img_path.exists():
            print(f"[skip] {stim['id']}: missing {img_path}", flush=True)
            n_skipped += N_SEEDS * len(PROMPTS)
            continue
        img = PIL.Image.open(img_path).convert("RGB")

        for prompt_type, user_text in PROMPTS.items():
            if (stim["id"], prompt_type) in done_keys:
                continue

            full_prompt = _build_llava_prompt(user_text)
            try:
                inputs = processor(images=img, text=full_prompt, return_tensors="pt").to(llava.device)
            except Exception as e:
                print(f"[skip] processor fail for {stim['id']}: {e}", flush=True)
                continue

            # Two generate calls × SEQS_PER_CALL for the 20 seeds.
            captions: list[str] = []
            for call_idx in range(N_SEEDS // SEQS_PER_CALL):
                torch.manual_seed(si * 1000 + hash(prompt_type) % 1000 + call_idx)
                try:
                    with torch.no_grad():
                        out = llava.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            num_return_sequences=SEQS_PER_CALL,
                        )
                    texts = processor.batch_decode(out, skip_special_tokens=True)
                    captions.extend(_trim_after_assistant(t) for t in texts)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[oom] {stim['id']} call {call_idx}, retrying with 1-seq batches", flush=True)
                    # Fallback: single-seq generation × SEQS_PER_CALL (slower but safer)
                    for k in range(SEQS_PER_CALL):
                        torch.manual_seed(si * 1000 + hash(prompt_type) % 1000 + call_idx * 100 + k)
                        with torch.no_grad():
                            out = llava.generate(
                                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True, temperature=TEMPERATURE, num_return_sequences=1,
                            )
                        t = processor.batch_decode(out, skip_special_tokens=True)[0]
                        captions.append(_trim_after_assistant(t))

            # Truncate / pad to exactly N_SEEDS
            captions = captions[:N_SEEDS]

            # Judge each caption
            aspects = (stim["aspect_a"], stim["aspect_b"])
            for seed, caption in enumerate(captions):
                # Inline judge call with raw output capture (matches Judge.classify internals)
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                        aspect_a=aspects[0], aspect_b=aspects[1], caption=caption)},
                ]
                q_inputs = judge._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_tensors="pt", enable_thinking=False,
                ).to(judge._model.device)
                with torch.no_grad():
                    q_out = judge._model.generate(
                        q_inputs, max_new_tokens=judge.max_new_tokens,
                        do_sample=False, pad_token_id=judge._tokenizer.eos_token_id,
                    )
                raw = judge._tokenizer.decode(q_out[0][q_inputs.shape[1]:], skip_special_tokens=True)
                label = _parse_label(raw)

                writer.writerow({
                    "stimulus_id": stim["id"],
                    "aspect_pair": stim["aspect_pair"],
                    "aspect_a": aspects[0],
                    "aspect_b": aspects[1],
                    "prompt_type": prompt_type,
                    "seed": seed,
                    "caption": caption,
                    "judge_label": label,
                    "judge_raw_output": raw,
                })
                n_done += 1
            f_csv.flush()

            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1e-3)
            eta = (n_total - n_done - n_skipped) / max(rate, 1e-3)
            print(f"[{n_done:>5}/{n_total}] {stim['id']:<14} {prompt_type:<18} "
                  f"elapsed={elapsed:7.0f}s  rate={rate:5.2f}/s  eta={eta/60:5.1f}min",
                  flush=True)
            wandb.log({
                "generations_done": n_done,
                "rate_per_sec": rate,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            })

    f_csv.close()
    elapsed = time.time() - t_start
    print(f"[done] {n_done} generations in {elapsed:.0f}s ({n_done/max(elapsed,1):.2f}/s)", flush=True)

    # ---- per-stimulus dominance + plot ----
    print("[summary] computing dominance + plotting", flush=True)
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    # Per stimulus: distribution over {aspect_a, aspect_b, both, neither}
    per_stim = df.groupby(["stimulus_id", "aspect_pair"])
    rows = []
    for (stim_id, pair), g in per_stim:
        counts = Counter(g["judge_label"])
        n = len(g)
        p_a = counts.get("aspect_a", 0) / n
        p_b = counts.get("aspect_b", 0) / n
        p_both = counts.get("both", 0) / n
        p_neither = counts.get("neither", 0) / n
        dominance = abs(p_a - p_b)
        # entropy over the 4-way distribution
        probs = np.array([p_a, p_b, p_both, p_neither])
        probs_nz = probs[probs > 0]
        entropy = float(-(probs_nz * np.log(probs_nz)).sum()) if probs_nz.size else 0.0
        rows.append({
            "stimulus_id": stim_id,
            "aspect_pair": pair,
            "n_samples": n,
            "p_aspect_a": p_a, "p_aspect_b": p_b,
            "p_both": p_both, "p_neither": p_neither,
            "dominance": dominance,
            "entropy": entropy,
        })
    sum_df = pd.DataFrame(rows)
    sum_csv = OUT_DIR / "dominance_per_stimulus.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"          wrote {sum_csv}", flush=True)

    mean_dom = float(sum_df["dominance"].mean())
    median_dom = float(sum_df["dominance"].median())
    frac_strong_dom = float((sum_df["dominance"] > 0.5).mean())
    print(f"          mean dominance   = {mean_dom:.3f}", flush=True)
    print(f"          median dominance = {median_dom:.3f}", flush=True)
    print(f"          frac > 0.5       = {frac_strong_dom:.3f}", flush=True)

    OUT_SUMMARY_JSON.write_text(json.dumps({
        "n_stimuli": len(sum_df), "n_generations": int(len(df)),
        "mean_dominance": mean_dom, "median_dominance": median_dom,
        "frac_dominance_gt_0.5": frac_strong_dom,
        "elapsed_seconds": elapsed,
    }, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.hist(sum_df["dominance"].values, bins=12, color="steelblue", edgecolor="white")
    ax.axvline(median_dom, color="black", linestyle=":", label=f"median = {median_dom:.2f}")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.6, label="success threshold (0.5)")
    ax.set_xlabel(r"dominance $|P(A)-P(B)|$")
    ax.set_ylabel("# stimuli")
    ax.set_title(f"Phase 1 dominance distribution ({len(sum_df)} stimuli)")
    ax.legend(fontsize=9)

    ax = axes[1]
    by_pair = sum_df.groupby("aspect_pair")["dominance"].agg(["mean", "count"]).sort_values("mean")
    # Keep pairs with ≥2 stimuli (singleton pairs are noisy)
    by_pair_keep = by_pair[by_pair["count"] >= 2]
    y = np.arange(len(by_pair_keep))
    ax.barh(y, by_pair_keep["mean"], color="darkorange")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p} (n={int(n)})" for p, n in zip(by_pair_keep.index, by_pair_keep["count"])],
                        fontsize=8)
    ax.set_xlabel("mean dominance")
    ax.set_title("Per aspect pair (n ≥ 2)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    # wandb.Image needs a raster; save a sibling PNG for upload.
    out_png = OUT_FIG.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    print(f"          wrote {OUT_FIG}", flush=True)
    print(f"          wrote {out_png}", flush=True)

    wandb.log({
        "summary/mean_dominance": mean_dom,
        "summary/median_dominance": median_dom,
        "summary/frac_dom_gt_0.5": frac_strong_dom,
        "summary/n_generations": int(len(df)),
        "figures/dominance_overview": wandb.Image(str(out_png)),
    })
    wandb.finish()
    print("[done] Phase 1 behavioral baseline complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
