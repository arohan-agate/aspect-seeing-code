"""Phase 0 smoke test 3: LLaVA-1.6-7B caption + peak VRAM.

Loads LlavaNextForConditionalGeneration from the pre-downloaded snapshot in
bf16, captions duck_rabbit_1.png with a neutral prompt, prints the caption
and peak GPU memory.

Run on an A100/A40 after `source scripts/activate.sh`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from aspect_seeing.paths import DATA_DIR, MODELS_DIR

LLAVA_PATH = MODELS_DIR / "llava-v1.6-vicuna-7b-hf"
IMAGE_PATH = (
    DATA_DIR / "panagopoulou" / "images"
    / "Bistable Images Original" / "duck_rabbit_1.png"
)
PROMPT = "What is in this image?"


def main() -> int:
    print("[1/4] importing", flush=True)
    import torch
    import PIL.Image
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

    if not LLAVA_PATH.exists():
        print(f"!! LLaVA snapshot missing at {LLAVA_PATH}", file=sys.stderr)
        return 1
    if not IMAGE_PATH.exists():
        print(f"!! image missing at {IMAGE_PATH}", file=sys.stderr)
        return 1

    torch.cuda.reset_peak_memory_stats()

    print(f"[2/4] loading processor + model (bf16) from {LLAVA_PATH.name}", flush=True)
    t0 = time.time()
    processor = LlavaNextProcessor.from_pretrained(str(LLAVA_PATH))
    model = LlavaNextForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)

    # vision_feature_layer is the layer LLaVA actually consumes (layer -2 = 22 of CLIP-L).
    vfl = getattr(model.config, "vision_feature_layer", None)
    print(f"      config.vision_feature_layer = {vfl}", flush=True)

    print(f"[3/4] preparing prompt + image", flush=True)
    img = PIL.Image.open(IMAGE_PATH).convert("RGB")
    # LLaVA-1.6 vicuna chat template (per official model card)
    conv = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\n{PROMPT} ASSISTANT:"
    )
    inputs = processor(images=img, text=conv, return_tensors="pt").to(model.device)

    print("[4/4] generate (greedy, max 64 tokens)", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
    elapsed = time.time() - t0
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    # Trim everything before "ASSISTANT:" so we just print the model's reply.
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1].strip()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"      generated in {elapsed:.1f}s", flush=True)
    print(f"      peak VRAM: {peak_gb:.2f} GB", flush=True)
    print(f"      caption  : {text!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
