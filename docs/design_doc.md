# Design Doc — Aspect-Seeing in Vision-Language Models via Sparse Autoencoders

**Author:** [user]
**Target venues:** PhilML ICML 2026 (May 11, 4 pages, primary) and Mechanistic Interpretability Workshop at ICML 2026 (May 8, 4 or 8 pages, dual)
**Start date:** April 23, 2026
**Compute:** a SLURM GPU cluster, A40-class GPUs (48 GB each)
**Status:** v1, post-literature-review

---

## 0. One-paragraph summary

We extract sparse-autoencoder features from CLIP ViT-L/14 layer 22 (the layer LLaVA-1.6-7B consumes) on bistable images and disambiguated controls. A 3,320-generation behavioral baseline across 83 stimuli surfaces three regimes under neutral prompting: **default-dominant** (LLaVA reliably reports one aspect, e.g. duck-rabbit → duck), **forced-balanced** (aspect-agnostic captions under neutral prompts, but a roughly even split between the two aspects under forced-choice prompts, e.g. Necker cube), and **intermediate**. Forced-choice prompting recovers aspect labels that neutral prompting misses on every abstention stimulus (10/10), showing that the gap between *reporting* an image and *committing to an interpretation* of it is behavioral, not representational. Using contrastive SAE analysis on the controls we identify candidate per-aspect features, test whether bistable stimuli represent both aspects in superposition or only the dominant one, and causally steer along the identified directions — to **flip** caption aspect on default-dominant pairs and to **re-bias** the ~50/50 split on forced-balanced pairs, under a fluency guard. The neutral-vs-forced-choice prompt contrast empirically operationalizes Wittgenstein's distinction between "seeing" and "seeing-as" (*Philosophical Investigations*, PPF §§111–136) in a VLM for the first time.

## 1. Background and contribution

### 1.1 Positioning against prior work

Three neighboring literatures converge on this project.

**Bistable images in VLMs.** Panagopoulou et al. (arXiv:2405.19423, ACL CMCL 2024) introduce a 29-image bistable benchmark and show 12 VLMs exhibit strong dominance biases driven by language-model priors, but their analysis is purely behavioral. AmbiBench (anonymous, under double-blind review at ICLR 2026 at the time of writing) extends this to 2,238 images and probes "perceptual-switch heads" at the attention level, raising InternVL3-2B bistable accuracy from 29% to 42% via head-level intervention. **We go below the attention-head level to SAE features**, which are finer-grained and interpretable by construction, and we demonstrate **causal aspect-flipping** at the caption level on a per-stimulus basis — neither done previously.

**SAE-based CLIP → LLaVA pipelines.** Pach et al. (arXiv:2504.02821) established the pipeline of CLIP-side SAE interventions propagating to LLaVA output for general monosemanticity; Joseph et al. (arXiv:2504.08729) quantified steerability (10–15% of features are reliably steerable). **Our contribution is not the pipeline but its application to representational competition**: bistable stimuli create the minimal experimental condition under which feature competition (rather than detection) is the phenomenon of interest.

**Philosophy of aspect-seeing.** Wittgenstein (PI, PPF §§111–136) distinguishes "seeing" from "seeing-as"; Hanson (1958) and Kuhn (1962) generalize this to theory-ladenness of perception; Nanay (2016) develops an attention-based account of aspect-perception. **No prior ML paper operationalizes this literature**; the closest analogs (Millière & Buckner 2024; Williams et al. 2025 "Mechanistic Interpretability Needs Philosophy") argue for philosophy-ML integration without empirical aspect-seeing experiments.

### 1.2 Claimed contributions

1. **A three-regime behavioral taxonomy** — default-dominant / forced-balanced / intermediate — discovered in a 3,320-generation baseline on 83 bistable stimuli. Default-dominant replicates Panagopoulou-style language-prior bias; forced-balanced is novel: LLaVA gives aspect-agnostic captions under neutral prompts but splits evenly between aspects under forced-choice prompts (e.g. Necker cube ~50/50), showing abstention is behavioral, not representational.
2. **Empirical feature characterization** — contrastively identified per-aspect SAE features in CLIP-L/14 layer 22, with manual and max-activating-example verification, across both default-dominant and forced-balanced stimulus groups.
3. **Causal aspect steering, two regimes** — per-stimulus interventions that **flip** LLaVA's caption aspect for default-dominant pairs and **re-bias** the forced-choice distribution for forced-balanced pairs, evaluated under a fluency guard.
4. **Empirical operationalization of aspect-seeing** — the neutral-vs-forced-choice prompt contrast makes Wittgenstein's "seeing" / "seeing-as" distinction measurable in a VLM; the first empirical engagement with Wittgensteinian aspect-seeing via mechanistic interpretability, grounded in a reproducible behavioral protocol rather than used purely as framing.

## 2. Research questions and hypotheses

### 2.1 RQ1: Do VLMs represent bistable stimuli with separable aspect features?

**Operational question.** When we contrastively compare SAE feature activations on disambiguated duck images vs disambiguated rabbit images, do we find features that (a) activate strongly for one aspect and weakly for the other, (b) recur across stimulus exemplars of the same aspect, and (c) are distinct from low-level texture features?

**H1.0 (null):** No reliable per-aspect features exist; activations on bistable images look like averages or neither-class.

**H1.1 (positive):** Per-aspect features exist and can be identified by contrastive top-k analysis on ≥3 matched disambiguated pairs per bistable stimulus.

**Falsifiable prediction:** If H1.1 holds, for each of the top-10 candidate duck-features and rabbit-features per stimulus, the AUROC of distinguishing pure-duck from pure-rabbit controls must exceed 0.85 on held-out exemplars.

### 2.2 RQ2: Are both aspects represented, or only the dominant one?

**Operational question.** On a bistable image, are the duck-feature and rabbit-feature both active (superposition), only the dominant-aspect feature active (dominance), or neither strongly active (dissolution)?

**H2.superposition:** Both feature sets activate above their individual control baselines.

**H2.dominance:** Only the behaviorally-preferred aspect's features activate (matching Panagopoulou's language-prior finding).

**H2.dissolution:** Activations are below control baselines for both aspects.

**Primary metric — the superposition index:**
> S = min(mean(active ducks), mean(active rabbits)) / max(mean(active ducks), mean(active rabbits))
>
> Computed per bistable stimulus. S ≈ 1 = full superposition; S < 0.3 = strong dominance; intermediate = partial.

### 2.3 RQ3: Does linguistic priming shift which aspect is represented?

**Operational question.** When LLaVA is prompted "Describe the duck in this image" vs "Describe the rabbit in this image", do the SAE feature activations on an identical input image shift toward the primed aspect?

**H3.0 (null):** Prompt has no effect on vision-side feature activation.

**H3.1 (positive):** Text prompt induces a shift in the ratio of duck-feature to rabbit-feature activation at the vision-tower output, independent of image content.

This is the direct empirical form of Wittgenstein's claim that aspect-seeing is "halfway between thought and perception."

### 2.4 RQ4: Can we causally induce aspect-switching?

**Operational question.** If we steer along the identified aspect-feature direction at CLIP layer 22 before the projector, can we force LLaVA to caption the non-dominant aspect for a given bistable stimulus?

**H4.0 (null):** Steering either has no effect or destroys caption fluency before producing an aspect flip.

**H4.1 (positive):** There exists a steering coefficient range in which caption aspect flips from dominant to non-dominant on ≥50% of stimuli with BLEU against held-out descriptions remaining within 20% of baseline.

## 3. System and data design

### 3.1 Models

| Component | Model | Source |
|---|---|---|
| Vision-language model | `llava-hf/llava-v1.6-vicuna-7b-hf` | HuggingFace |
| Decomposed vision tower | `openai/clip-vit-large-patch14-336` | HuggingFace |
| Decomposed language model | `lmsys/vicuna-7b-v1.5` | HuggingFace |

LLaVA-1.6-7B uses `vision_feature_layer=-2` — the penultimate (layer 22) of CLIP ViT-L/14, patch tokens only, 576 tokens per 336² tile. This is the **only intervention site that cleanly propagates from vision features to captions** without going through CLIP's own final projection.

### 3.2 SAE choice

**Primary: saev (Stevens et al., OSU-NLP, arXiv:2502.06755)** — BatchTopK and Matryoshka BatchTopK SAEs trained on CLIP ViT-L/14-336px at layers {11, 17, 22, 23}. Layer 22 is our target.

**Backup: CLIP-Scope (Ewington-Pitsos & Goyal, 2024, lewington/CLIP-ViT-L-scope)** — TopK SAEs (k=32, 65,536 features) at residual-stream layers {2, 5, 8, 11, 14, 17, 20, 22}. If saev loading breaks, swap in.

Prisma toolkit (arXiv:2504.19475) provides the hook infrastructure (HookedViT) but its public pretrained SAEs are CLIP-B/32 only — not usable for LLaVA. We use Prisma for hooks and visualization, not for SAE weights.

### 3.3 Dataset

**Bistable stimuli (target: 60 images across 8 aspect pairs):**

| Pair | Canonical image | Count |
|---|---|---|
| Duck / rabbit | Jastrow 1899 + variants | 8 |
| Face / vase | Rubin 1915 + variants | 8 |
| Necker cube (front/back) | Necker 1832 | 8 |
| Young / old woman | Hill 1915 | 8 |
| Rat / man | Bugelski-Alampay 1961 | 6 |
| Schroeder stairs | Schröder 1858 | 6 |
| Cat / dog | Panagopoulou 2024 | 8 |
| AI-generated bistable | SDXL + manual curation | 8 |

Sources: Panagopoulou et al.'s 29-image release (https://github.com/artemisp/Bistable-Illusions-MLLMs) for behavioral baseline; Wikimedia Commons for historical originals; AmbiBench (https://huggingface.co/datasets/BLNL/AmbiBench, MIT) for extended coverage.

**Disambiguated controls (target: 120 images, 15 per aspect pair × 2 aspects):**

For each aspect pair we need unambiguous exemplars of each side (pure duck, pure rabbit). Generation procedure:
1. SDXL with prompts like "a realistic photograph of only a duck in a pond, high detail, no rabbit features" and adversarial negative prompt.
2. Hand-filter: reject any image where the non-target aspect is visible or the subject is ambiguous (solo-researcher pass, target ≥90% retention).
3. Pre-register 15 per side before running any probe.

**Distractors (target: 60 images):** ordinary COCO photographs, one per control class, matched for composition/color. Used as a null-control for feature activation baselines.

**Total dataset: ~240 images.** Stored at `$ASPECT_SCRATCH/data/{bistable,pure-A,pure-B,distractor}/*.png`.

### 3.4 Dataset versioning

Every image gets a hash-based ID. A single CSV (`dataset.csv`) stores: `id, category, aspect_pair, aspect_label, source, license, prompt_if_generated, retained_flag`. Frozen before week 2; any additions logged with date.

## 4. Experimental design

### 4.1 Phase 1 — Behavioral baseline (days 1–4)

**Goal:** Replicate Panagopoulou et al.'s dominance finding on our image set and confirm our stimuli behave as bistable in LLaVA-1.6-7B.

**Procedure.** For each bistable image, query LLaVA with a neutral prompt ("What is in this image?" and "Describe this image in one sentence.") across 20 seeds. Classify each output as duck, rabbit, neither, or both via an LLM-as-judge rubric (Claude Opus 4.7) with manual spot-check on 30 random outputs.

**Metrics.**
- *Dominance score* per stimulus: |P(aspect A) − P(aspect B)| ∈ [0, 1].
- *Aspect entropy*: H(aspect distribution), low = strong dominance.
- *Model agreement*: cross-model correlation with CLIP ViT-L/14 zero-shot classification.

**Success criterion.** Mean dominance score >0.5 across ≥40 bistable stimuli, confirming a dominant aspect exists to be flipped later.

**Output.** Table of (stimulus, dominant aspect, dominance score). Locks in reference numbers.

### 4.2 Phase 2 — Contrastive feature identification (days 5–8)

**Goal:** Identify candidate per-aspect SAE features at CLIP layer 22.

**Procedure.**
1. Load saev layer-22 SAE on CLIP ViT-L/14-336px.
2. For each aspect pair (e.g., duck/rabbit), extract CLIP layer-22 patch activations on all pure-A and pure-B controls.
3. Pool features: mean-pool across patches for each image → one activation vector per image per feature.
4. For each feature *f*, compute AUROC for separating pure-A from pure-B controls (held-out split).
5. Retain top 20 features per aspect (A-features: high activation on A, low on B; B-features: reverse) with AUROC > 0.85.

**Validation.**
- Max-activating-examples inspection: pull the top-10 patches across a 50K-image COCO subset for each retained feature. Manually label. Retain only features whose max-activating patches visually align with the aspect (duck-beak-like shapes, rabbit-ear-like shapes, etc.). Expected retention: 5–15 features per aspect after visual sanity check.
- Specificity test: retained features should show low activation on distractors (COCO non-animal images).

**Output.** Feature manifest CSV: `aspect_pair, aspect_label, feature_id, auroc, manual_label, top_max_activating_image_ids`.

### 4.3 Phase 3 — Superposition vs dominance analysis (days 9–10)

**Goal:** Test RQ2.

**Procedure.**
1. For each bistable image, run CLIP forward pass and record activations on the retained A-features and B-features.
2. Compute per-stimulus: A-score = mean activation of retained A-features; B-score = mean of B-features.
3. Normalize each by the median activation of that feature on its matching-aspect pure controls (so A-score = 1.0 means "as active as on a typical pure A").
4. Compute superposition index S (§2.2) per stimulus.
5. Classify each stimulus into {superposition if S>0.7, dominance if S<0.3, partial else}.

**Primary figure.** Scatter plot: each bistable image is a point, x-axis = A-score, y-axis = B-score. Pure-A controls along the x-axis, pure-B along the y-axis, bistable images in the quadrant revealing the regime.

**Secondary analysis.** Correlate S with behavioral dominance from Phase 1. Hypothesis: low S (dominance regime) correlates with high behavioral dominance.

### 4.4 Phase 4 — Priming experiment (days 11–12)

**Goal:** Test RQ3.

**Procedure.**
1. For each bistable image and each prompt P ∈ {neutral, prime-A, prime-B}:
   - Run LLaVA forward pass; record CLIP layer-22 patch activations at the moment they're passed to the projector.
   - Measure A-score and B-score as in Phase 3.
2. Compute priming shift: Δ_prime = (A-score | prime-A) − (A-score | prime-B). Positive = priming effective.
3. Test Δ_prime > 0 across stimuli (paired t-test, Bonferroni-corrected per aspect pair).

**Critical control.** Does prompt affect vision-tower activations **at all**? In LLaVA-1.6 the CLIP vision tower is run once per image *before* text tokens are processed — so naively Δ_prime should be zero. If it's non-zero, either (a) there is cross-attention leakage we missed, or (b) we measured at the wrong point. Document carefully; this is where the experiment could be revealing *or* confounded.

**If prompt has no effect on vision tower** (likely), move priming to the LM side: measure Vicuna residual-stream activations at image-token positions across prompts, and ask whether Vicuna reprojects the vision features differently under priming. This shifts RQ3 from "does priming affect perception" to "does priming affect interpretation," which is still Wittgensteinian but methodologically cleaner.

### 4.5 Phase 5 — Causal steering (days 13–16)

**Goal:** Test RQ4.

**Procedure.**
1. For each bistable image with a clear dominant aspect (from Phase 1):
   - Compute steering vector v = mean(retained non-dominant-aspect feature directions in SAE decoder space).
   - Apply at CLIP layer 22 (before projector) across a sweep of coefficients α ∈ {0, 0.5, 1, 2, 4, 8, 16}.
   - Generate 5 captions per α per stimulus.
   - Classify each caption with the LLM judge (Phase 1 rubric).
2. Steering success rate per stimulus: fraction of captions at optimal α that describe the non-dominant aspect.

**Fluency guard.**
- Per α, compute BLEU-4 against held-out human-written captions for pure-A and pure-B images of the target aspect.
- Compute perplexity of the generated caption under Vicuna.
- Reject α if BLEU drops >20% from α=0 baseline or perplexity doubles.

**Primary success criterion.** At least one α exists per stimulus where (a) aspect-flip rate ≥50% and (b) fluency guard passes. Across the stimulus set, report mean flip rate and fluency delta at the per-stimulus-optimal α.

**Ablations (time permitting).**
- Steering at layers 17 and 11 (earlier) vs 22 (primary) — where does aspect representation become manipulable?
- Steering with random feature directions (null control): flip rate should be near-zero.
- Steering with dominant-aspect direction (positive control): should strengthen dominance, not flip.
- Number of features in the steering vector: 1, 5, 10, 20. Scaling curve.

### 4.6 Phase 6 — Writing (days 17–20)

See §7 timeline.

## 5. Metrics summary

| RQ | Primary metric | Threshold for positive result |
|---|---|---|
| RQ1 (separable features exist) | Feature AUROC on held-out pure controls | ≥0.85 for ≥5 features per aspect after manual verification |
| RQ2 (superposition vs dominance) | Superposition index S per stimulus | Mean S > 0.7 = superposition regime; < 0.3 = dominance regime |
| RQ3 (priming effect) | Priming shift Δ_prime | Significant (p < 0.05 corrected) and positive |
| RQ4 (causal flip) | Aspect-flip rate at fluency-preserving α | ≥50% flip rate on ≥50% of stimuli |

### 5.1 ICLR expansion track (deferred from workshop scope)

The workshop submission stops at vision-side causal steering (Phase 4). Phase 4 already produced one boundary finding — vision-side steering at CLIP layer 22 cannot break the linguistic abstention on force-balanced stimuli (young_old_woman: 0/7 flips at any α under fluency guard, despite Phase 3 confirming feature-level superposition). That result motivates a follow-up package for the ICLR submission, scoped here to keep workshop and conference plans separate.

1. **Language-side steering.** Re-run the steering experiment with the forward hook moved from `vision_tower.encoder.layers[22]` to (a) the `multi_modal_projector` output and (b) one or more layers inside `language_model` (Vicuna). The hypothesis from Phase 4 is that the seeing-vs-seeing-as collapse lives in the LM, not in CLIP — language-side steering on force-balanced stimuli should rescue the 0% flip rate that vision-side steering can't. Direct test: same SAE-derived steering vectors mapped through the projector, applied to image-token positions in Vicuna's residual stream, sweep α, measure flip / fluency / judge as in Phase 4. This pinpoints *where* the dominance collapse happens.

2. **Cross-model verification.** Replicate the full pipeline (Phase 1 behavioural baseline → Phase 2 feature ID → Phase 3 superposition → Phase 4 steering) on at least one second VLM with a different vision-language adapter, e.g. `Qwen2-VL-7B` or `InternVL3-2B` (the latter is what AmbiBench's switch-head paper used — direct comparison). Goal: show the three-regime taxonomy and the vision-vs-language steering split aren't LLaVA-specific quirks. Each model needs its own SAE + own controls; reuse the bistable stimulus set unchanged.

3. **Vicuna text-only aspect-seeing probe.** Run the text-only follow-up flagged in §11: ask Vicuna without any image whether words like *bank*, *light*, *bat*, *crane* are bistable, and contrast its behaviour against its image-grounded responses on the same lexical items. Tests whether aspect-seeing as a phenomenon generalises from vision to language within the same model — direct empirical engagement with Wittgenstein's claim that aspect-seeing is "halfway between thought and perception."

These three items together turn the workshop result into a fuller paper: workshop = "we can flip vision-side aspect features in CLIP and partially flip captions"; ICLR = "we can localise the dominance collapse to the language model, replicate cross-model, and connect it to text-side aspect-seeing." The workshop submission deliberately does not depend on these.

## 6. Risks, mitigations, and decision points

### 6.1 Hard-stop decision points

**End of day 4 — behavioral replication:** if mean dominance score < 0.3 (stimuli are *not* bistable for LLaVA), either the model is too aspect-blind or our stimuli fail. Fix: switch to AmbiBench's larger set; revalidate with Qwen2-VL-7B (which Panagopoulou showed is slightly less aspect-dominant).

**End of day 10 — feature identification:** if no aspect has ≥3 features passing AUROC+manual verification, SAE resolution is insufficient at layer 22. Fix options: (a) try layer 17 or 23, (b) swap saev for CLIP-Scope (different L0/expansion factor), (c) widen to Matryoshka SAEs for finer granularity. Budget: 2 days of slack.

**End of day 12 — priming:** if priming has no effect at CLIP layer (expected) AND no effect at Vicuna-side (surprising), RQ3 collapses. **This is recoverable** — report "VLMs exhibit aspect-blindness to priming at the vision tower and partial sensitivity at the language model," which is itself a publishable Wittgensteinian claim.

**End of day 16 — causal steering:** if no α satisfies flip+fluency jointly, pivot to a negative-result framing: "VLMs represent multiple aspects in superposition but the LM is insensitive to vision-side aspect steering," which still engages AmbiBench's switch-heads finding by contrast.

### 6.2 Specific risks

**R1 — AmbiBench ICLR camera-ready lands before May 8.** Monitor OpenReview weekly. Mitigation: the three differentiators (SAE-level, causal flipping, philosophy) are robust to any behavioral extension they might add; preserve these as the contribution headline.

**R2 — saev layer-22 SAE fails to load or reconstruct poorly.** saev v2 resolved v1's ViT-L/14 scaling issues, but verify immediately. Fallback: CLIP-Scope layer 22 or layer 20 (lewington/CLIP-ViT-L-scope). Budget day 1 for this.

**R3 — LLaVA-1.6-7B + nnsight integration is finicky.** No canonical example in the literature for SAE intervention inside LLaVA-1.6. Most analogous: Pach et al. 2025 (arXiv:2504.02821). Read their code first. Test with a trivial intervention (zero ablation) before any scientific run.

**R4 — AI-generated "pure" controls leak the other aspect.** Hand-verify every generated image. If SDXL cannot produce clean exemplars for some aspect pair (realistic concern for rat/man and Schroeder stairs), drop that pair rather than introduce confounds.

**R5 — LLM-judge rubric has systematic biases.** Include manual spot-checks on 30 outputs per phase; calibrate against 2 human raters (self + one friend) on a 50-sample gold set.

**R6 — The superposition index S is ambiguous at intermediate values.** Preregister cutoffs (0.3, 0.7) before seeing Phase 3 results.

## 7. Timeline

| Days | Phase | Key deliverable |
|---|---|---|
| 1–2 | Setup | Cluster env, models loaded, saev SAE verified at layer 22 |
| 3–4 | Phase 1 | Behavioral dominance table |
| 5–8 | Phase 2 | Feature manifest CSV |
| 9–10 | Phase 3 | Superposition index figure |
| 11–12 | Phase 4 | Priming result (possibly null, documented) |
| 13–16 | Phase 5 | Steering curves + flip-rate table + fluency guard |
| 17 | Slack / write intro + related work | — |
| 18 | Write MI Workshop 8-page version | — |
| **Day 18: submit MI Workshop (May 8)** | | |
| 19–20 | Trim to 4-page PhilML, expand Wittgenstein framing | — |
| **Day 21: submit PhilML (May 11)** | | |
| 22–30 | Optional: extend steering ablations for AI4Law-relevant follow-up or public release | — |

## 8. Infrastructure

- **Compute:** 1–2 A40s sufficient for all phases; no distributed training.
- **Storage:** `$ASPECT_SCRATCH/` — images, activation caches, feature manifests, logs.
- **Code:** project repository (private until submission).
- **Tracking:** Weights & Biases project; each run logs (phase, stimulus id, feature id, activation vector path, metric snapshot).
- **Determinism:** fix seeds per run; LLaVA inference uses temperature 0.7 with 20 seeds for behavioral runs, temperature 0 for steering runs.
- **Environment:** `transformers==4.52`, `sae-lens>=6.0`, `nnsight>=0.3`, `torch==2.3`, `openai-clip`, `prisma-vit==0.4.2`, `saev==0.2`.

## 9. Claude Code division of labor

**Claude Code handles:**
- SDXL disambiguated-control generation pipeline (Phase 0).
- Dataset CSV assembly, license audit, hash-ID generation.
- SAE loading glue + saev/Prisma hooks with CLIP-L/14.
- nnsight LLaVA-1.6 wrapper with clean intervention API.
- Per-phase evaluation scripts.
- Plotting (matplotlib) with consistent styling.
- Draft paper sections from experimental logs.

**Human (you) handles:**
- All scientific decisions (RQ refinement, threshold preregistration, go/no-go at decision points).
- Manual verification of max-activating-example labels (Phase 2) — the single highest-leverage human task.
- Final caption-quality judgments (Phase 5).
- Philosophy framing (Wittgenstein, Nanay, Wollheim citations) — don't let Claude Code draft the philosophical sections without tight supervision.
- Differentiation paragraphs against AmbiBench and Pach et al. — high-stakes reviewer-facing text.

## 10. Paper outline (MI Workshop 8pp version)

1. **Introduction** (0.5 pp) — duck-rabbit as canonical stimulus; the three prior streams; our three contributions; explicit differentiation from AmbiBench and Pach et al. in the first paragraph.
2. **Background** (0.75 pp) — CLIP+LLaVA pipeline; SAE interpretability fundamentals; the Wittgensteinian "seeing" vs "seeing-as" distinction in one paragraph, cited but not belabored.
3. **Method** (1.5 pp) — dataset, models, SAE choice, the superposition index, steering protocol, fluency guard.
4. **Results** (3 pp) — behavioral baseline, feature identification (fig 1), superposition analysis (fig 2, the main money plot), priming (fig 3), steering curves and flip rates (fig 4).
5. **Discussion** (1 pp) — what the taxonomy reveals; relation to predictive-coding and competition accounts of bistable perception; candidate Wittgensteinian interpretation in a dedicated paragraph.
6. **Limitations and future work** (0.5 pp).
7. **References** (0.75 pp).

PhilML 4pp trim: keep figures 1, 2, 4; collapse method; expand discussion with Wittgenstein/Nanay/Wollheim; move behavioral baseline to appendix.

## 11. Open questions flagged for further thought

- Should we compare against a *text-only* aspect-seeing probe — ask Vicuna "does the word 'bank' mean financial institution or riverbank?" — to see if aspect-seeing generalizes from vision to language within the same VLM? (Strong philosophical hook, probably out of scope for 2 weeks, good v2 direction.)
- Is there a natural "aspect dawning" probe — induce a gradient from pure A to bistable and watch features activate temporally? Hard because CLIP is a single forward pass, but could be approximated with interpolated SDXL images.
- Should we file a pre-registration on OSF at day 4 (after Phase 1 freezes stimuli and cutoffs)? Low-cost insurance against reviewer pushback on p-hacking concerns.
