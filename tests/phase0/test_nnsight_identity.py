"""Phase 0 smoke test 4: nnsight identity hook at CLIP layer 22.

Wraps LLaVA-1.6-7B in nnsight, installs a passthrough (identity) hook at
the residual-stream output of CLIP-L/14 layer 22 (the layer LLaVA actually
consumes via vision_feature_layer=-2), generates a caption with temperature 0,
and asserts the output token IDs match the unhooked baseline.

This validates the intervention path before Phase 2 / Phase 5 use real
non-trivial hooks. If nnsight's LLaVA generate() integration is fragile,
falls back to a transformers forward_hook (the design-doc fallback) and
prints which path actually worked — that distinction matters for Phase 2
planning.

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
LAYER_INDEX = 22
MAX_NEW_TOKENS = 32   # short for fast equality check


def _build_inputs(processor, img):
    conv = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\n{PROMPT} ASSISTANT:"
    )
    return processor(images=img, text=conv, return_tensors="pt")


def _decode(processor, out_ids):
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1].strip()
    return text


def main() -> int:
    print("[1/6] importing", flush=True)
    import torch
    import PIL.Image
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

    print("[2/6] loading LLaVA (bf16)", flush=True)
    processor = LlavaNextProcessor.from_pretrained(str(LLAVA_PATH), use_fast=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()

    img = PIL.Image.open(IMAGE_PATH).convert("RGB")

    print("[3/6] generating baseline (no hook)", flush=True)
    inputs = _build_inputs(processor, img).to(model.device)
    with torch.no_grad():
        baseline_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    baseline_text = _decode(processor, baseline_ids)
    print(f"      baseline: {baseline_text!r}", flush=True)

    # ---- Path A: nnsight identity hook ----
    print(f"[4/6] trying nnsight identity hook at layer {LAYER_INDEX}", flush=True)
    nnsight_ok = False
    nnsight_text = None
    try:
        from nnsight import NNsight
        nn = NNsight(model)
        # Build inputs again to reset (some processors mutate state).
        inputs2 = _build_inputs(processor, img).to(model.device)
        # nnsight 0.6 generate ctx: the assignment 'output = output' is the
        # identity intervention; nnsight installs the hook and proxies the
        # forward through it. LLaVA's processor returns a BatchFeature (dict-
        # like); pass its data dict so HF generate can unpack input_ids /
        # pixel_values / image_sizes properly.
        gen_kwargs = dict(inputs2.data, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        with nn.generate(**gen_kwargs) as tracer:
            layer_out = nn.vision_tower.vision_model.encoder.layers[LAYER_INDEX].output
            nn.vision_tower.vision_model.encoder.layers[LAYER_INDEX].output = layer_out
            nn_out = nn.generator.output.save()
        nnsight_text = _decode(processor, nn_out)
        nnsight_ok = True
        print(f"      nnsight  : {nnsight_text!r}", flush=True)
    except Exception as e:
        # Truncate the message — nnsight likes to print the full model repr
        # in its NNsightException payload, which is hundreds of lines.
        msg = str(e).splitlines()[0] if str(e) else "(empty)"
        if len(msg) > 200:
            msg = msg[:200] + "...(truncated)"
        print(f"      nnsight FAILED ({type(e).__name__}): {msg}", flush=True)

    # ---- Path B: transformers forward_hook fallback ----
    print(f"[5/6] running transformers forward_hook identity at layer {LAYER_INDEX}", flush=True)
    layer_module = model.vision_tower.vision_model.encoder.layers[LAYER_INDEX]
    fired = {"n": 0}

    def identity(module, inp, out):
        # CLIPEncoderLayer returns a tuple (hidden_states, ...) — leave it alone.
        fired["n"] += 1
        return out

    handle = layer_module.register_forward_hook(identity)
    try:
        inputs3 = _build_inputs(processor, img).to(model.device)
        with torch.no_grad():
            hooked_ids = model.generate(**inputs3, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        hooked_text = _decode(processor, hooked_ids)
    finally:
        handle.remove()
    print(f"      hook fired {fired['n']}x during generate", flush=True)
    print(f"      tf-hook  : {hooked_text!r}", flush=True)

    print("[6/6] verifying identity preserved output", flush=True)
    tf_match = hooked_text == baseline_text
    print(f"      transformers hook == baseline?  {tf_match}", flush=True)
    if nnsight_ok:
        nn_match = nnsight_text == baseline_text
        print(f"      nnsight       == baseline?  {nn_match}", flush=True)
        ok = nn_match
        print(f"      ==> nnsight path WORKS for layer-{LAYER_INDEX} interventions" if nn_match
              else "      ==> nnsight RAN but output diverged — investigate before Phase 2", flush=True)
    else:
        ok = tf_match
        print("      ==> nnsight path BROKEN; will use transformers forward_hook for Phase 2", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
