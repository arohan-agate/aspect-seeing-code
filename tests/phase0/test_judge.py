"""Phase 0 smoke test 6: Qwen3-8B judge end-to-end + VRAM coexistence.

Two halves:

(A) Run the judge on 5 hand-written captions covering the full label space
    {aspect_a, aspect_b, both, neither} and verify the parsed labels match
    expectations. Also report the raw judge response so we can see what
    Qwen3 actually emits with enable_thinking=False.

(B) Load LLaVA-1.6-7B alongside Qwen3-8B (both bf16) and report combined
    peak VRAM. Phase 3 / Phase 5 need both models live in the same process
    (caption + judge per stimulus); we want to know whether they coexist
    on a single A100 80GB or whether we need a sequential
    caption-then-judge workflow.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from aspect_seeing.paths import DATA_DIR, MODELS_DIR

# Test cases: (caption, (aspect_a, aspect_b), expected_label).
TEST_CASES = [
    ("This image shows a duck in water.",
     ("duck", "rabbit"),                     "aspect_a"),
    ("I see a rabbit with long ears.",
     ("duck", "rabbit"),                     "aspect_b"),
    ("A young woman is looking over her shoulder.",
     ("young woman", "old woman"),           "aspect_a"),
    ("There's nothing recognizable here.",
     ("duck", "rabbit"),                     "neither"),
    ("It could be either a duck or a rabbit.",
     ("duck", "rabbit"),                     "both"),
]


def _classify_with_raw(judge, caption: str, aspects: tuple[str, str]):
    """Like Judge.classify but also returns the raw model output for diagnostics."""
    judge._ensure_loaded()
    import torch
    from aspect_seeing.eval.judge import (
        JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, _parse_label,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            aspect_a=aspects[0], aspect_b=aspects[1], caption=caption,
        )},
    ]
    inputs = judge._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(judge._model.device)

    with torch.no_grad():
        out = judge._model.generate(
            inputs,
            max_new_tokens=judge.max_new_tokens,
            do_sample=False,
            pad_token_id=judge._tokenizer.eos_token_id,
        )
    raw = judge._tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return _parse_label(raw), raw


def part_a_judge_only() -> tuple[int, int, float]:
    print("=" * 60, flush=True)
    print("Part A: Qwen3-8B judge on 5 hand-written captions", flush=True)
    print("=" * 60, flush=True)
    import torch
    from aspect_seeing.eval.judge import Judge

    torch.cuda.reset_peak_memory_stats()

    print("[A1] loading Qwen3-8B (bf16, lazy load via Judge.classify)", flush=True)
    judge = Judge()
    # Trigger load now so the timing column doesn't include it.
    judge._ensure_loaded()
    qwen3_vram = torch.cuda.memory_allocated() / 1e9
    print(f"     loaded; resident VRAM = {qwen3_vram:.2f} GB", flush=True)

    print("[A2] running 5 captions", flush=True)
    print(f"  {'idx':<3} {'expected':<10} {'got':<10} {'OK':<3} caption", flush=True)
    correct = 0
    misparses: list[tuple[int, str, str, str, str]] = []
    for i, (caption, aspects, expected) in enumerate(TEST_CASES):
        t0 = time.time()
        got, raw = _classify_with_raw(judge, caption, aspects)
        dt = time.time() - t0
        ok = got == expected
        if ok:
            correct += 1
        else:
            misparses.append((i, caption, expected, got, raw))
        print(f"  {i:<3} {expected:<10} {got:<10} {'✓' if ok else '✗'}   "
              f"{dt:5.2f}s  '{caption}'", flush=True)
        print(f"      raw: {raw!r}", flush=True)

    accuracy = correct / len(TEST_CASES)
    print(f"\n[A3] accuracy: {correct}/{len(TEST_CASES)} = {accuracy*100:.0f}%", flush=True)
    if misparses:
        print("\nmisparses:", flush=True)
        for i, cap, exp, got, raw in misparses:
            print(f"  case {i}: expected {exp!r} got {got!r} raw={raw!r}", flush=True)
            print(f"           caption: {cap!r}", flush=True)

    judge_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[A4] judge-only peak VRAM = {judge_peak:.2f} GB", flush=True)
    return correct, len(TEST_CASES), judge_peak


def part_b_coexistence(judge_peak: float) -> float:
    print("\n" + "=" * 60, flush=True)
    print("Part B: LLaVA + Qwen3 coexistence on the same GPU", flush=True)
    print("=" * 60, flush=True)
    import torch
    import PIL.Image
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    from aspect_seeing.eval.judge import Judge

    LLAVA_PATH = MODELS_DIR / "llava-v1.6-vicuna-7b-hf"
    IMAGE_PATH = (
        DATA_DIR / "panagopoulou" / "images"
        / "Bistable Images Original" / "duck_rabbit_1.png"
    )

    torch.cuda.reset_peak_memory_stats()

    print("[B1] loading Qwen3-8B (judge, bf16)", flush=True)
    judge = Judge()
    judge._ensure_loaded()
    after_qwen = torch.cuda.memory_allocated() / 1e9
    print(f"     resident after Qwen3: {after_qwen:.2f} GB", flush=True)

    print("[B2] loading LLaVA-1.6-7B (bf16) on the same device", flush=True)
    processor = LlavaNextProcessor.from_pretrained(str(LLAVA_PATH), use_fast=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    after_llava = torch.cuda.memory_allocated() / 1e9
    print(f"     resident after LLaVA: {after_llava:.2f} GB", flush=True)

    print("[B3] caption duck_rabbit_1.png + judge in one process", flush=True)
    img = PIL.Image.open(IMAGE_PATH).convert("RGB")
    conv = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        "USER: <image>\nWhat is in this image? ASSISTANT:"
    )
    inputs = processor(images=img, text=conv, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0]
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:", 1)[1].strip()
    print(f"     LLaVA caption: {caption!r}", flush=True)

    label = judge.classify(caption=caption, aspects=("duck", "rabbit"))
    print(f"     judge label  : {label}", flush=True)

    coexist_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[B4] coexistence peak VRAM = {coexist_peak:.2f} GB", flush=True)
    print(f"     headroom on A100 80GB  ≈ {80 - coexist_peak:.1f} GB", flush=True)
    return coexist_peak


def main() -> int:
    correct, total, judge_peak = part_a_judge_only()
    coexist_peak = part_b_coexistence(judge_peak)

    print("\n" + "=" * 60, flush=True)
    print("Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  judge accuracy        : {correct}/{total}", flush=True)
    print(f"  judge-only peak VRAM  : {judge_peak:.2f} GB", flush=True)
    print(f"  LLaVA+judge peak VRAM : {coexist_peak:.2f} GB", flush=True)
    if coexist_peak < 70:
        print("  ==> both models coexist comfortably on A100 80GB", flush=True)
    else:
        print("  ==> tight coexistence on A100 80GB; consider sequential caption-then-judge", flush=True)
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
