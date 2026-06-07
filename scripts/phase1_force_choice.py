"""Phase 1 follow-up: force-choice re-prompt on high-abstention stimuli.

The main Phase 1 run (scripts/phase1_behavioral.py) found a population of
stimuli — Necker Cube, Young-Old Woman, Schroeder Stairs, Spinning Dancer,
etc. — where LLaVA neither picks aspect_a nor aspect_b but instead gives
aspect-agnostic captions ("a cube", "a woman") that the judge correctly
labels "neither". That could mean either:

  (a) LLaVA literally can't see the two aspects (genuine aspect-blindness)
  (b) LLaVA sees them but won't commit unless prompted to choose

This script distinguishes (a) from (b). Same 20-seed protocol, but with a
forced-choice prompt that names the aspect pair directly:
    "Which interpretation does this image support: {aspect_a} or {aspect_b}?
     Answer with one word."

If forced-choice shifts the label distribution away from 'neither' to
{aspect_a, aspect_b}, the stimulus is a 'won't commit' case — still a
candidate for Phase 5 steering (the aspect representations exist, the LM
just isn't using them by default). If forced-choice stays near-100%
'neither', the stimulus is aspect-blind at the model level — likely
unrecoverable for Phase 5, but interesting for the paper.

Output: outputs/phase1/forced_choice.csv with the same schema as
behavioral_baseline.csv, plus a summary at outputs/phase1/forced_choice_summary.csv.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from aspect_seeing.paths import ASPECT_REPO, DATA_DIR, OUTPUTS_DIR, MODELS_DIR

REPO_DIR = ASPECT_REPO
INVENTORY_CSV   = DATA_DIR / "dataset_inventory.csv"
DOMINANCE_CSV   = OUTPUTS_DIR / "phase1" / "dominance_per_stimulus.csv"
OUT_DIR         = OUTPUTS_DIR / "phase1"
OUT_CSV         = OUT_DIR / "forced_choice.csv"
OUT_SUMMARY_CSV = OUT_DIR / "forced_choice_summary.csv"

LLAVA_PATH = MODELS_DIR / "llava-v1.6-vicuna-7b-hf"

# Selection: top-N stimuli by P(neither).
N_HIGH_NEITHER = 10
N_SEEDS = 20
TEMPERATURE = 0.7
SEQS_PER_CALL = 10
MAX_NEW_TOKENS = 24
CSV_HEADERS = [
    "stimulus_id", "aspect_pair", "aspect_a", "aspect_b",
    "prompt_type", "seed", "caption", "judge_label", "judge_raw_output",
]


def _force_choice_prompt(aspect_a: str, aspect_b: str) -> str:
    # One-word constraint keeps the judge prompt clean and latency low.
    return (f"Which interpretation does this image support: "
            f"{aspect_a} or {aspect_b}? Answer with one word.")


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


def _select_high_neither() -> list[dict]:
    """Return the N_HIGH_NEITHER stimuli (from Phase 1) with the highest P(neither),
    joined with inventory rows so we have aspect labels + file paths.
    """
    inventory = {r["id"]: r for r in csv.DictReader(INVENTORY_CSV.open())}
    dom_rows = []
    with DOMINANCE_CSV.open() as f:
        for r in csv.DictReader(f):
            r["p_neither"] = float(r["p_neither"])
            r["dominance"] = float(r["dominance"])
            dom_rows.append(r)
    dom_rows.sort(key=lambda r: -r["p_neither"])

    selected = []
    for r in dom_rows[:N_HIGH_NEITHER]:
        inv = inventory.get(r["stimulus_id"])
        if inv is None:
            continue
        selected.append({
            "id": r["stimulus_id"],
            "aspect_pair": inv["aspect_pair"],
            "aspect_a": inv["aspect_a"],
            "aspect_b": inv["aspect_b"],
            "file_path": inv["file_path"],
            "p_neither_phase1": r["p_neither"],
            "dominance_phase1": r["dominance"],
        })
    return selected


def main() -> int:
    import torch
    import PIL.Image
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stimuli = _select_high_neither()
    print(f"[1/4] {len(stimuli)} high-neither stimuli selected:", flush=True)
    for s in stimuli:
        print(f"      {s['id']:<14} {s['aspect_pair']:<28} "
              f"P(neither)_phase1={s['p_neither_phase1']:.3f}  "
              f"dom_phase1={s['dominance_phase1']:.3f}", flush=True)

    # ---- load models ----
    print("[2/4] loading LLaVA-1.6-7B (bf16)", flush=True)
    t0 = time.time()
    processor = LlavaNextProcessor.from_pretrained(str(LLAVA_PATH), use_fast=True)
    llava = LlavaNextForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH), dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True,
    )
    llava.eval()
    print(f"       LLaVA loaded in {time.time()-t0:.1f}s", flush=True)

    print("       loading Qwen3 judge", flush=True)
    from aspect_seeing.eval.judge import Judge, JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, _parse_label
    judge = Judge()
    judge._ensure_loaded()
    print(f"       peak VRAM with both models: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    # ---- generate + judge ----
    csv_exists = OUT_CSV.exists() and OUT_CSV.stat().st_size > 0
    done_keys: set[tuple[str, str]] = set()
    if csv_exists:
        with OUT_CSV.open() as f:
            for r in csv.DictReader(f):
                done_keys.add((r["stimulus_id"], r["prompt_type"]))
        print(f"[resume] found {len(done_keys)} done (stimulus, prompt) pairs", flush=True)

    f_csv = OUT_CSV.open("a", newline="")
    writer = csv.DictWriter(f_csv, fieldnames=CSV_HEADERS)
    if not csv_exists:
        writer.writeheader()

    print("[3/4] running forced-choice generations", flush=True)
    t_start = time.time()
    for si, stim in enumerate(stimuli):
        prompt_type = "force_choice"
        if (stim["id"], prompt_type) in done_keys:
            continue

        img_path = Path(stim["file_path"])
        if not img_path.exists():
            print(f"[skip] {stim['id']}: missing {img_path}", flush=True)
            continue
        img = PIL.Image.open(img_path).convert("RGB")

        user_text = _force_choice_prompt(stim["aspect_a"], stim["aspect_b"])
        full_prompt = _build_llava_prompt(user_text)
        inputs = processor(images=img, text=full_prompt, return_tensors="pt").to(llava.device)

        captions = []
        for call_idx in range(N_SEEDS // SEQS_PER_CALL):
            torch.manual_seed(si * 1000 + call_idx)
            with torch.no_grad():
                out = llava.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=TEMPERATURE,
                    num_return_sequences=SEQS_PER_CALL,
                )
            texts = processor.batch_decode(out, skip_special_tokens=True)
            captions.extend(_trim_after_assistant(t) for t in texts)

        captions = captions[:N_SEEDS]
        aspects = (stim["aspect_a"], stim["aspect_b"])
        for seed, caption in enumerate(captions):
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
        f_csv.flush()
        elapsed = time.time() - t_start
        print(f"       [{si+1}/{len(stimuli)}] {stim['id']:<14} {stim['aspect_pair']:<28} "
              f"done ({elapsed:.0f}s elapsed)", flush=True)
    f_csv.close()

    # ---- summary comparing phase1 vs force-choice ----
    print("[4/4] computing before/after summary", flush=True)
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    rows = []
    for (stim_id, pair), g in df.groupby(["stimulus_id", "aspect_pair"]):
        counts = Counter(g["judge_label"])
        n = len(g)
        p_a_fc       = counts.get("aspect_a", 0) / n
        p_b_fc       = counts.get("aspect_b", 0) / n
        p_both_fc    = counts.get("both",     0) / n
        p_neither_fc = counts.get("neither",  0) / n
        # Pull Phase 1 baseline for contrast
        prior = next((s for s in stimuli if s["id"] == stim_id), None)
        rows.append({
            "stimulus_id": stim_id,
            "aspect_pair": pair,
            "p_neither_phase1":     prior["p_neither_phase1"] if prior else None,
            "dominance_phase1":     prior["dominance_phase1"] if prior else None,
            "p_aspect_a_fc":  p_a_fc,
            "p_aspect_b_fc":  p_b_fc,
            "p_both_fc":      p_both_fc,
            "p_neither_fc":   p_neither_fc,
            "dominance_fc":   abs(p_a_fc - p_b_fc),
            "delta_neither":  p_neither_fc - (prior["p_neither_phase1"] if prior else 0.0),
        })
    sum_df = pd.DataFrame(rows).sort_values("p_neither_phase1", ascending=False)
    sum_df.to_csv(OUT_SUMMARY_CSV, index=False)

    print(f"       wrote {OUT_CSV}", flush=True)
    print(f"       wrote {OUT_SUMMARY_CSV}", flush=True)
    print()
    print("=== force-choice vs neutral-Phase 1 (P(neither) and dominance) ===", flush=True)
    print(sum_df[[
        "stimulus_id", "aspect_pair",
        "p_neither_phase1", "p_neither_fc", "delta_neither",
        "dominance_phase1", "dominance_fc",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x,float) else x), flush=True)
    n_still_abstaining = int((sum_df["p_neither_fc"] > 0.5).sum())
    n_rescued = len(sum_df) - n_still_abstaining
    print()
    print(f"       {n_rescued}/{len(sum_df)} stimuli flipped from abstention under forced choice", flush=True)
    print(f"       {n_still_abstaining}/{len(sum_df)} remain abstaining (aspect-blind)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
