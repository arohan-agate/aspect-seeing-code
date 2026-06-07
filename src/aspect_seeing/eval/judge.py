"""Qwen3-8B LLM-judge for caption-aspect classification.

We use Qwen/Qwen3-8B (Apache 2.0, ungated) as the LLM-judge for
classifying LLaVA captions into {aspect_a, aspect_b, both, neither}. Key gotcha:
Qwen3 emits a <think>...</think> block by default; we pass enable_thinking=False
to apply_chat_template so outputs are parseable.

Usage:
    from aspect_seeing.eval.judge import Judge
    judge = Judge()  # loads Qwen3-8B on GPU
    label = judge.classify(
        caption="A small brown rabbit peeking out from behind a fence.",
        aspects=("duck", "rabbit"),
    )
    # -> "rabbit"

Intentionally thin: the real model load/inference logic needs a live GPU
session. This module defines the API and the prompt template so callers can
be written before GPU time is available.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from aspect_seeing.paths import MODELS_DIR

Label = Literal["aspect_a", "aspect_b", "both", "neither"]

JUDGE_SYSTEM_PROMPT = (
    "You are a careful classifier. Given a caption describing an image and "
    "two competing aspects, decide which aspect the caption primarily describes. "
    "Respond with exactly one token from: aspect_a, aspect_b, both, neither."
)

JUDGE_USER_TEMPLATE = (
    "Aspects:\n"
    "  aspect_a = {aspect_a}\n"
    "  aspect_b = {aspect_b}\n\n"
    "Caption: {caption}\n\n"
    "Which aspect does the caption primarily describe? Answer with one of: "
    "aspect_a, aspect_b, both, neither."
)

VALID_LABELS: set[str] = {"aspect_a", "aspect_b", "both", "neither"}


@dataclass
class Judge:
    """Qwen3-8B LLM-judge. Loads on first `classify` call."""

    model_path: str = field(
        default_factory=lambda: os.environ.get(
            "QWEN3_MODEL_PATH",
            str(MODELS_DIR / "Qwen3-8B"),
        )
    )
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_new_tokens: int = 8

    _tokenizer: object | None = None
    _model: object | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Import inside so the module can be imported on CPU-only nodes.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Qwen3-8B weights not found at {self.model_path}. "
                "Run `python scripts/download_models.py` first."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype),
            device_map=self.device,
        )
        self._model.eval()

    def classify(
        self,
        caption: str,
        aspects: tuple[str, str],
    ) -> Label:
        """Classify a caption against an (aspect_a, aspect_b) pair."""
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    aspect_a=aspects[0], aspect_b=aspects[1], caption=caption,
                ),
            },
        ]
        # enable_thinking=False is the whole reason we use Qwen3 instead of its
        # sibling models — without it, output is prefixed with <think>...</think>
        # which makes `re.search` for the label unreliable.
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return _parse_label(text)


def _parse_label(text: str) -> Label:
    """Pull one of VALID_LABELS out of the judge's free-text output."""
    m = re.search(r"\b(aspect_a|aspect_b|both|neither)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()  # type: ignore[return-value]
    # Fallback: if the model said the aspect name instead of aspect_a/aspect_b,
    # the caller should handle that. Default to 'neither' to be conservative.
    return "neither"
