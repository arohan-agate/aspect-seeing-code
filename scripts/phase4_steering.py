"""Phase 4 — causal steering at CLIP layer 22 inside LLaVA (design doc §4.5).

Uses Phase 2 v3 features to construct unit-norm steering directions per
aspect pool, then additively boosts the matching layer-22 patch tokens
during LLaVA's vision_tower forward via a `register_forward_hook` (the
design-doc fallback after Phase 0 found nnsight 0.6.3 broken on LLaVA
multimodal generate).

For each superposition stimulus in (duck_rabbit, young_old_woman,
hidden_face), we run:
  - α = 0  (baseline, no hook)
  - α ∈ {0.5, 1, 2, 4, 8, 16} × {direction = +v_a, +v_b}

Greedy decoding (do_sample=False, temperature=0) per the design-doc
recommendation — eliminates seed variance and isolates the steering
effect. Per-α flip rate aggregated over stimuli.

Per generated caption we record:
  - judge_label from Qwen3 (with enable_thinking=False)
  - perplexity under LLaVA's language_model (= Vicuna 7B v1.5)

Outputs:
  outputs/phase4/steering_<group>.csv
  outputs/phase4/steering_summary.json
  outputs/figures/phase4_flip_vs_alpha.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from aspect_seeing.paths import DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FIG_DIR

INVENTORY_CSV          = DATA_DIR / "dataset_inventory.csv"
PHASE2_FEATURES_DIR    = OUTPUTS_DIR / "phase2"
PHASE3_DIR             = OUTPUTS_DIR / "phase3"
SAE_CKPT               = MODELS_DIR / "own-sae-clip-L14-layer22" / "best.pt"
LLAVA_PATH             = MODELS_DIR / "llava-v1.6-vicuna-7b-hf"

OUT_DIR_CSV = OUTPUTS_DIR / "phase4"
OUT_DIR_FIG = FIG_DIR

ALPHAS              = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LAYER_INDEX_MOD     = 22                # encoder.layers[22] -- LLaVA reads its output
PROMPT              = "What is in this image?"
MAX_NEW_TOKENS      = 48
D_MODEL             = 1024
N_FEATURES          = 65_536
K_TOPK              = 32

GROUPS_DEFAULT = ["duck_rabbit", "young_old_woman", "hidden_face"]


# ---------- model loading ----------

def build_sae(device: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TopKSAE(nn.Module):
        def __init__(self):
            super().__init__()
            W_enc = torch.randn(D_MODEL, N_FEATURES) / math.sqrt(D_MODEL)
            self.W_enc = nn.Parameter(W_enc.clone())
            self.W_dec = nn.Parameter(W_enc.t().contiguous().clone())
            self.b_enc = nn.Parameter(torch.zeros(N_FEATURES))
            self.b_pre = nn.Parameter(torch.zeros(D_MODEL))

        def encode(self, x):
            pre = x - self.b_pre
            hidden = F.relu(pre @ self.W_enc + self.b_enc)
            top_vals, top_idx = hidden.topk(K_TOPK, dim=-1)
            out = torch.zeros_like(hidden)
            out.scatter_(-1, top_idx, top_vals)
            return out
    return TopKSAE().to(device)


def load_own_sae(device: str):
    import torch
    ckpt = torch.load(SAE_CKPT, map_location=device, weights_only=False)
    print(f"[sae] loaded best.pt step={ckpt['step']}", flush=True)
    sae = build_sae(device)
    sd = {k: v.float() for k, v in ckpt["model"].items()}
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def load_llava(device: str):
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    import torch
    print("[llava] loading LLaVA-1.6-7B (bf16)", flush=True)
    processor = LlavaNextProcessor.from_pretrained(str(LLAVA_PATH), use_fast=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        str(LLAVA_PATH),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


# ---------- steering hook ----------

def compute_steering_vector(sae, ids: list[int]) -> "torch.Tensor":
    """Unit-norm mean of the SAE decoder columns for the given feature ids."""
    import torch
    W_dec = sae.W_dec.detach().float()                  # (65536, 1024)
    v = W_dec[ids].mean(dim=0)                          # (1024,)
    n = v.norm()
    if float(n) < 1e-8:
        raise RuntimeError("steering vector has near-zero norm")
    return (v / n).contiguous()


def make_steering_hook(steering_vec, alpha: float, dtype):
    """Returns a forward hook that adds α·v to the patch tokens of the layer
    output (skipping CLS at position 0). Works on the (hidden_states,) tuple
    returned by CLIPEncoderLayer."""
    def hook(module, inputs, output):
        # output is (hidden_states,) or just hidden_states depending on flags
        if isinstance(output, tuple):
            hs = output[0]
            add = alpha * steering_vec.to(hs.device, hs.dtype)
            new_hs = hs.clone()
            new_hs[:, 1:, :] = new_hs[:, 1:, :] + add
            return (new_hs,) + output[1:]
        hs = output
        add = alpha * steering_vec.to(hs.device, hs.dtype)
        new_hs = hs.clone()
        new_hs[:, 1:, :] = new_hs[:, 1:, :] + add
        return new_hs
    return hook


# ---------- LLaVA generation + perplexity + judge ----------

def _build_llava_prompt(user_text: str) -> str:
    return (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\n{user_text} ASSISTANT:"
    )


def _trim(text: str) -> str:
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1]
    return text.strip()


def caption_image(processor, llava, img, max_new_tokens=MAX_NEW_TOKENS):
    import torch
    inputs = processor(
        images=img, text=_build_llava_prompt(PROMPT), return_tensors="pt"
    ).to(llava.device)
    with torch.no_grad():
        out = llava.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    txt = processor.batch_decode(out, skip_special_tokens=True)[0]
    return _trim(txt)


def _resolve_llm_head(llava):
    """Return (backbone, head) for the language model, robust to LlavaNext's
    layouts across transformers versions:
      - llava.language_model = LlamaModel (headless), llava.lm_head = Linear
      - llava.language_model = LlamaForCausalLM (head included; .logits)
      - llava.model.language_model = LlamaModel, llava.lm_head = Linear
    """
    backbone = None
    head = None
    if hasattr(llava, "language_model") and llava.language_model is not None:
        backbone = llava.language_model
    elif hasattr(llava, "model") and hasattr(llava.model, "language_model"):
        backbone = llava.model.language_model
    if hasattr(llava, "lm_head") and llava.lm_head is not None:
        head = llava.lm_head
    return backbone, head


def perplexity(llava, tokenizer, text: str) -> float:
    """Perplexity of `text` under LLaVA's underlying language model (= Vicuna)
    on the text alone (no image conditioning)."""
    import torch
    import torch.nn.functional as F
    if not text or not text.strip():
        return float("inf")
    ids = tokenizer(text, return_tensors="pt").input_ids.to(llava.device)
    if ids.shape[1] < 2:
        return float("inf")
    backbone, head = _resolve_llm_head(llava)
    if backbone is None:
        return float("nan")
    with torch.no_grad():
        out = backbone(ids)
        if hasattr(out, "logits") and out.logits is not None:
            logits = out.logits                                # CausalLM-style
        else:
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            if head is None:
                return float("nan")
            logits = head(hidden)
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    return float(torch.exp(loss).item())


# ---------- group runner ----------

def load_v3_features(group: str) -> tuple[list[int], list[int]]:
    p = PHASE2_FEATURES_DIR / f"features_{group}_v3.csv"
    rows = list(csv.DictReader(p.open()))
    a_ids = [int(r["feature_id"]) for r in rows if r["aspect_label"] == "a"]
    b_ids = [int(r["feature_id"]) for r in rows if r["aspect_label"] == "b"]
    return a_ids, b_ids


def load_phase3_superposition(group: str) -> list[dict]:
    p = PHASE3_DIR / f"superposition_{group}.csv"
    rows = list(csv.DictReader(p.open()))
    return [r for r in rows if r["classification"] == "superposition"]


def run_group(g: str, sae, llava, processor, judge, layer_module,
              all_inv_by_id: dict[str, dict], writer, baseline_perplexities: dict):
    import torch
    a_ids, b_ids = load_v3_features(g)
    print(f"[{g}] feature pools: A={len(a_ids)} B={len(b_ids)}", flush=True)
    v_a = compute_steering_vector(sae, a_ids)
    v_b = compute_steering_vector(sae, b_ids)
    print(f"[{g}] steering vector norms: |v_a|=1, |v_b|=1 (unit)", flush=True)

    superpos = load_phase3_superposition(g)
    print(f"[{g}] {len(superpos)} superposition stimuli to steer", flush=True)

    from PIL import Image

    # JUDGE inputs need aspect_a / aspect_b labels; pull from inventory
    inv_by_id = {r["id"]: r for r in csv.DictReader(INVENTORY_CSV.open())}

    n_done = 0
    t_g0 = time.time()
    for s in superpos:
        stim_id = s["stimulus_id"]
        inv = inv_by_id.get(stim_id, {})
        if not inv:
            print(f"[{g}] !! missing inventory row for {stim_id}", flush=True)
            continue
        path = Path(inv["file_path"])
        if not path.exists():
            print(f"[{g}] !! missing image {path}", flush=True)
            continue
        aspect_a = inv.get("aspect_a", "")
        aspect_b = inv.get("aspect_b", "")

        img = Image.open(path).convert("RGB")

        # Baseline (no hook)
        cap0 = caption_image(processor, llava, img)
        ppl0 = perplexity(llava, processor.tokenizer, cap0)
        lbl0 = judge.classify(cap0, aspects=(aspect_a, aspect_b))
        baseline_perplexities[stim_id] = ppl0
        writer.writerow({
            "group": g, "stimulus_id": stim_id, "aspect_pair": inv["aspect_pair"],
            "aspect_a": aspect_a, "aspect_b": aspect_b,
            "direction": "baseline", "alpha": 0.0,
            "caption": cap0, "judge_label": lbl0,
            "perplexity": round(ppl0, 4),
            "perplexity_ratio": 1.0,
        })

        # Steered conditions
        for direction, vec in (("a", v_a), ("b", v_b)):
            for alpha in ALPHAS:
                handle = layer_module.register_forward_hook(
                    make_steering_hook(vec, alpha, dtype=torch.bfloat16)
                )
                try:
                    cap = caption_image(processor, llava, img)
                except Exception as e:
                    cap = f"<<exception during generate: {type(e).__name__}>>"
                finally:
                    handle.remove()
                ppl = perplexity(llava, processor.tokenizer, cap)
                lbl = judge.classify(cap, aspects=(aspect_a, aspect_b))
                ppl_ratio = ppl / max(ppl0, 1e-9)
                writer.writerow({
                    "group": g, "stimulus_id": stim_id, "aspect_pair": inv["aspect_pair"],
                    "aspect_a": aspect_a, "aspect_b": aspect_b,
                    "direction": direction, "alpha": alpha,
                    "caption": cap, "judge_label": lbl,
                    "perplexity": round(ppl, 4),
                    "perplexity_ratio": round(ppl_ratio, 4),
                })
        n_done += 1
        elapsed = time.time() - t_g0
        rate = n_done / max(elapsed, 1e-3)
        eta = (len(superpos) - n_done) / max(rate, 1e-3)
        print(f"[{g}] {n_done}/{len(superpos)} stim {stim_id:<14} "
              f"baseline=`{cap0[:55].replace(chr(10),' ')}…` lbl={lbl0}  "
              f"ppl0={ppl0:.1f}  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)


# ---------- aggregation + plot ----------

def _flatten_csv() -> "pd.DataFrame":
    import pandas as pd
    frames = []
    for p in sorted(OUT_DIR_CSV.glob("steering_*.csv")):
        if p.name.startswith("steering_") and p.suffix == ".csv":
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_and_plot():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = _flatten_csv()
    if df.empty:
        print("[plot] no data — skipping", flush=True)
        return None

    # Per (group, direction, alpha): success rate = label == direction's aspect
    df["target_label"] = df["direction"].map({"a": "aspect_a", "b": "aspect_b"})
    df["is_steered_aspect"] = (df["judge_label"] == df["target_label"]).astype(int)

    # Baseline counts (alpha=0): treat as "fraction labeled aspect_a" / "aspect_b"
    base = df[df["direction"] == "baseline"].groupby("group").agg(
        n=("stimulus_id", "count"),
        p_a=("judge_label", lambda x: (x == "aspect_a").mean()),
        p_b=("judge_label", lambda x: (x == "aspect_b").mean()),
        p_neither=("judge_label", lambda x: (x == "neither").mean()),
        ppl=("perplexity", "mean"),
    ).reset_index()

    steer = df[df["direction"] != "baseline"].groupby(
        ["group", "direction", "alpha"]
    ).agg(
        n_stim=("stimulus_id", "count"),
        success=("is_steered_aspect", "mean"),
        ppl_mean=("perplexity", "mean"),
        ppl_ratio_mean=("perplexity_ratio", "mean"),
    ).reset_index()

    # Persist tidy aggregates
    base_csv  = OUT_DIR_CSV / "baseline_per_group.csv"
    steer_csv = OUT_DIR_CSV / "success_vs_alpha.csv"
    base.to_csv(base_csv, index=False)
    steer.to_csv(steer_csv, index=False)
    print(f"[plot] wrote {base_csv} and {steer_csv}", flush=True)

    # ---- figure ----
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(2, len(groups), figsize=(5.0 * len(groups), 8.4),
                             sharex=True, gridspec_kw={"hspace": 0.32})
    if len(groups) == 1:
        axes = axes.reshape(2, 1)

    color_a = "#2563eb"  # blue (steer toward A)
    color_b = "#dc2626"  # red (steer toward B)

    for ci, g in enumerate(groups):
        ax_succ = axes[0, ci]
        ax_ppl  = axes[1, ci]

        for direction, color in (("a", color_a), ("b", color_b)):
            sub = steer[(steer["group"] == g) & (steer["direction"] == direction)] \
                  .sort_values("alpha")
            if sub.empty:
                continue
            ax_succ.plot(sub["alpha"], sub["success"], "o-", color=color,
                         label=f"steer →{direction}", linewidth=1.7, markersize=5.5)
            ax_ppl.plot(sub["alpha"], sub["ppl_ratio_mean"], "o-", color=color,
                        linewidth=1.7, markersize=5.5)

        # Baseline reference: P(aspect_a) and P(aspect_b) on neutral prompt
        b = base[base["group"] == g].iloc[0] if (base["group"] == g).any() else None
        if b is not None:
            ax_succ.axhline(b["p_a"], color=color_a, linestyle=":", alpha=0.5,
                            linewidth=1.0)
            ax_succ.axhline(b["p_b"], color=color_b, linestyle=":", alpha=0.5,
                            linewidth=1.0)
            ax_ppl.axhline(1.0, color="black", linestyle=":", alpha=0.5, linewidth=1.0)
            # Mark fluency budget = 1.2× baseline (design-doc 20% guard)
            ax_ppl.axhline(1.2, color="grey", linestyle="--", alpha=0.7, linewidth=1.0,
                           label="fluency guard (1.2×)")

        ax_succ.set_title(f"{g}  (n={int(b['n']) if b is not None else 0})", fontsize=10)
        ax_succ.set_ylabel("P(caption labeled steered aspect)")
        ax_succ.set_ylim(-0.04, 1.04)
        ax_succ.set_xscale("log", base=2)
        ax_succ.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
        ax_succ.legend(fontsize=8, loc="best")

        ax_ppl.set_xlabel(r"steering coefficient $\alpha$")
        ax_ppl.set_ylabel("perplexity / baseline")
        ax_ppl.set_xscale("log", base=2)
        ax_ppl.set_yscale("log", base=10)
        ax_ppl.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, which="both")
        if ci == 0:
            ax_ppl.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Phase 4 — steering success rate and fluency vs α at CLIP layer 22\n"
        "Top: P(caption labeled the steered aspect) per α.  "
        "Bottom: perplexity ratio (steered ÷ baseline).  "
        "Dashed grey = 1.2× fluency guard.",
        fontsize=11, y=1.0,
    )
    out_pdf = OUT_DIR_FIG / "phase4_flip_vs_alpha.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_pdf}", flush=True)

    # Summary JSON
    summary = []
    for g in groups:
        steer_g = steer[steer["group"] == g]
        b = base[base["group"] == g].iloc[0] if (base["group"] == g).any() else None
        # Find best α per direction (success high AND fluency under guard)
        def best_for(direction: str):
            sub = steer_g[steer_g["direction"] == direction]
            ok = sub[sub["ppl_ratio_mean"] <= 1.2]
            if ok.empty:
                return {"alpha": None, "success": None, "ppl_ratio": None}
            top = ok.sort_values("success", ascending=False).iloc[0]
            return {"alpha": float(top["alpha"]), "success": float(top["success"]),
                    "ppl_ratio": float(top["ppl_ratio_mean"])}
        summary.append({
            "group": g,
            "n_stimuli": int(b["n"]) if b is not None else 0,
            "baseline_p_a": float(b["p_a"]) if b is not None else None,
            "baseline_p_b": float(b["p_b"]) if b is not None else None,
            "baseline_p_neither": float(b["p_neither"]) if b is not None else None,
            "best_steer_to_a_under_guard": best_for("a"),
            "best_steer_to_b_under_guard": best_for("b"),
            "max_success_to_a": float(steer_g[steer_g["direction"]=="a"]["success"].max()) if not steer_g.empty else None,
            "max_success_to_b": float(steer_g[steer_g["direction"]=="b"]["success"].max()) if not steer_g.empty else None,
        })
    sj = OUT_DIR_CSV / "steering_summary.json"
    sj.write_text(json.dumps(summary, indent=2))
    print(f"[plot] wrote {sj}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=GROUPS_DEFAULT)
    ap.add_argument("--skip-aggregation", action="store_true")
    args = ap.parse_args()

    OUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae = load_own_sae(device)
    processor, llava = load_llava(device)

    print("[judge] loading Qwen3-8B", flush=True)
    from aspect_seeing.eval.judge import Judge
    judge = Judge()
    judge._ensure_loaded()
    print(f"[judge] peak VRAM with all 3 models: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    # Reach into LLaVA to get encoder.layers[22] of the CLIP vision tower.
    layer_module = llava.vision_tower.vision_model.encoder.layers[LAYER_INDEX_MOD]
    print(f"[hook] target = vision_tower.vision_model.encoder.layers[{LAYER_INDEX_MOD}]  "
          f"({type(layer_module).__name__})", flush=True)

    fields = [
        "group", "stimulus_id", "aspect_pair", "aspect_a", "aspect_b",
        "direction", "alpha", "caption", "judge_label",
        "perplexity", "perplexity_ratio",
    ]

    baseline_ppls: dict[str, float] = {}
    for g in args.groups:
        out_csv = OUT_DIR_CSV / f"steering_{g}.csv"
        f = out_csv.open("w", newline="")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        try:
            run_group(g, sae, llava, processor, judge, layer_module,
                      {}, writer, baseline_ppls)
        finally:
            f.close()
        print(f"[{g}] wrote {out_csv}", flush=True)

    if not args.skip_aggregation:
        aggregate_and_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
