# TruthLens Phase 7 Protocol: Shortcut-Resistant Detection via Matched In-Memory Augmentation

**Repository:** `https://github.com/deepanshu70001/Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**Milestone:** Phase 7 — Research Candidate Training, Calibration & Generalization Audit  
**Date:** November 2026  
**Status:** **PRE-REGISTERED PROTOCOL (LOCKED BEFORE TRAINING)**  

---

## 1. Context and Problem Statement

Phase 6 evaluation rejected candidate E3 (`truthlens/siglip-base-tiny-genimage-e3_partial_finetuning`). While E3 achieved high in-distribution validation accuracy on Tiny-GenImage ($0.8520$ balanced accuracy), held-out benchmark evaluation on Frontier Benchmark October 2026 revealed severe degradation compared to the frozen baseline ($0.5497$ vs $0.6228$ balanced accuracy; $0.5546$ vs $0.6501$ ROC-AUC).

### The Root Cause
The Phase 7 Step 1 audit (`experiments/reports/phase7_format_audit.json`) confirmed the underlying shortcut:
- **Phase 4 Training Manifest ($N=6,995$):** Exactly 100.0% of real images are JPEG (3,515 / 3,515) from ImageNet-Camera, and exactly 100.0% of AI images are PNG (3,480 / 3,480) from generative synthesis benchmarks.
- **Dimensional Asymmetry:** Real images vary widely in aspect ratio (mean $1.22$, range $0.39 - 3.89$) and dimensions (median $500 \times 375$), while AI images are strictly square (aspect ratio $1.000$, dimensions $128 \times 128$, $256 \times 256$, or $512 \times 512$).
- **Metadata Solvability:** A simple metadata classifier (format, width, height, aspect ratio, quality) achieves a **1.0000 ROC-AUC** and **1.0000 Balanced Accuracy** on Tiny-GenImage without inspecting image pixels.
- **Causal Vulnerability:** Transcoding AI PNG images to JPEG ($Q=90$) causes $45.6\%$ of predictions to flip from AI to Real, with an enormous average logit drop of $6.12$ points.

Phase 7 investigates whether new detector architectures (SigLIP-base and ResNet-50 ImageNet V2) trained using **only the existing approved 6,995 training samples** can achieve shortcut resistance when trained under a strictly matched in-memory pipeline applied identically to both classes.

---

## 2. Matched Augmentation Pipeline Specification

To eliminate container-level and resolution shortcuts, the pipeline applies the following transformations on-the-fly identically to both classes:

1. **Random Rescale of Shorter Side:**
   - Shorter dimension is rescaled to a size sampled uniformly: $S \sim \text{Uniform}(224, 340)$ with random interpolation (`BILINEAR` or `BICUBIC`).
   - Ensures neither class retains fixed original dimensions or aspect ratio cues.
2. **Random Crop to $224 \times 224$:**
   - A $224 \times 224$ crop is extracted from the randomly rescaled image.
   - Ensures all inputs to the vision model have identical spatial resolution ($224 \times 224$) and aspect ratio ($1.0$).
3. **Random In-Memory Transcoding (Identical for Both Classes):**
   - Transcoding is executed in-memory via `io.BytesIO()`.
   - **JPEG Transcoding ($p = 0.70$):** Image is re-encoded as JPEG with random quality factor $Q \sim \text{Uniform}(40, 95)$.
   - **WebP Transcoding ($p = 0.15$):** Image is re-encoded as WebP with random quality factor $Q \sim \text{Uniform}(50, 95)$.
   - **Lossless PNG Transcoding ($p = 0.15$):** Image is re-encoded as PNG.
4. **Gaussian Blur Perturbation ($p = 0.15$):**
   - Applied with radius $r \sim \text{Uniform}(0.5, 1.2)$ to disrupt residual high-frequency camera vs diffusion noise cues.
5. **Canonical Normalization:**
   - For SigLIP models: mean $(0.5, 0.5, 0.5)$, std $(0.5, 0.5, 0.5)$.
   - For ResNet-50 models: ImageNet standard mean $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$.

### Residual Confounds (Documented Limitations)
The matched in-memory pipeline eliminates container format, image dimensions, aspect ratio, and single-pass compression shortcuts. However, two intrinsic data limitations cannot be removed without new data:
1. **Double Compression in Real Images:** Real nature images were already JPEG-compressed at source (ImageNet ILSVRC2012). Re-encoding them as JPEG introduces second-generation DCT quantization artifacts, whereas AI images experience first-generation JPEG compression.
2. **Interpolation History:** AI images generated at native $128 \times 128$ or $256 \times 256$ have different sampling and anti-aliasing histories when upscaled compared to natural photographs downscaled from $500 \times 375$.

---

## 3. Pre-Training Shortcut Gate Verification

Prior to model training, the matched pipeline was applied to an $N = 1,000$ sample of the training manifest. A metadata classifier was trained with 5-fold Stratified Cross-Validation on the resulting metadata (format, width, height, aspect ratio, JPEG quality):

- **Measured 5-Fold CV ROC-AUC:** **$0.5028$**
- **Shortcut Gate Threshold:** $\text{ROC-AUC} < 0.5500$
- **Gate Status:** **PASS** (Container format shortcuts are completely neutralized).

---

## 4. Controlled Model Run Specifications

All runs use the identical approved Phase 4 training pool (6,995 images: 3,515 authentic, 3,480 synthetic), seed `42`, FP16 mixed precision, and RTX 3050 4GB VRAM hardware constraints:

| Run ID | Model ID | Backbone Architecture | Pretrained Source | Trainable Parameters | Optimizer & Learning Rates | Training Recipe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | `truthlens/siglip-base-v2-s1` | `google/siglip-base-patch16-224` | Revision `7fd15f0689...` | Head only (769 params, 0.0008%) | AdamW, Head LR = $1 \times 10^{-4}$ | Linear probe on $K=5$ precomputed matched augmented views per training image. |
| **S2** | `truthlens/siglip-base-v2-s2` | `google/siglip-base-patch16-224` | Revision `7fd15f0689...` | Layers 10, 11 + Head (14,176,513 params, 15.26%) | AdamW, Backbone LR = $1 \times 10^{-5}$, Head LR = $1 \times 10^{-4}$ | On-the-fly matched augmentation, gradient accumulation 2, early stop on val. |
| **R1** | `truthlens/resnet50-v2-r1` | `torchvision.models.resnet50` | `IMAGENET1K_V2` | Full backbone + Head (23,510,081 params, 100.0%) | AdamW, Backbone LR = $1 \times 10^{-5}$, Head LR = $1 \times 10^{-4}$ | Full fine-tune with on-the-fly matched augmentation. |
| **R2** | `truthlens/resnet50-v2-r2` | `torchvision.models.resnet50` | `IMAGENET1K_V2` | Stage 4 (`layer4` + `fc`) (14,983,169 params, 63.73%) | AdamW, Stage 4 LR = $1 \times 10^{-5}$, Head LR = $1 \times 10^{-4}$ | Partial fine-tune with on-the-fly matched augmentation. |
| **A1** | `truthlens/siglip-base-v2-a1` | `google/siglip-base-patch16-224` | Revision `7fd15f0689...` | Layers 10, 11 + Head (14,176,513 params, 15.26%) | AdamW, Backbone LR = $1 \times 10^{-5}$, Head LR = $1 \times 10^{-4}$ | **Ablation Run:** Identical to S2 but **WITHOUT** matched augmentation (raw canonical transform). |

---

## 5. Evaluation Metrics & Robustness Probes

### 5.1 Primary Metrics
- **Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1 Score, ROC-AUC, PR-AUC, MCC.**
- **Real-Image False Positive Rate ($\text{FPR}_{5\%}$):** Operating threshold chosen on validation splits to ensure $\text{FPR} \le 5\%$ on real images.
- **Cross-Generator Recall:** AI detection rates tracked separately for seen generators (ADM, BigGAN, SD 1.5, GLIDE) and held-out generators (Midjourney, Wukong, VQDM).

### 5.2 Validation Robustness Probes (Applied to BOTH Classes)
- **JPEG-90 Probe:** In-memory re-encode at $Q=90$. Measure prediction flip rate (%) and mean $|\Delta \text{Logit}|$.
- **JPEG-75 Probe:** In-memory re-encode at $Q=75$. Measure prediction flip rate (%) and mean $|\Delta \text{Logit}|$.
- **JPEG-50 Probe:** In-memory re-encode at $Q=50$. Measure prediction flip rate (%) and mean $|\Delta \text{Logit}|$.
- **Gaussian Blur Probe:** Blur with $r=1.0$. Measure prediction flip rate (%) and mean $|\Delta \text{Logit}|$.

---

## 6. Pre-Stated Model Selection Rule

Selection of the single best candidate prior to locked test evaluation is determined **strictly by validation splits** (Tiny-GenImage validation, $N = 5,958$, and Frontier validation, $N = 297$):

$$\text{Score}_{\text{val}} = 0.50 \times \text{BalancedAccuracy}_{\text{tiny\_val}} + 0.50 \times \text{BalancedAccuracy}_{\text{frontier\_val}}$$

- **Tie-Breaker:** Frontier validation ROC-AUC.
- **Frontier Validation Uncertainty:** Because Frontier validation contains $N = 297$ samples, 95% bootstrap confidence intervals are computed and reported for all validation candidates.

---

## 7. Pre-Registered Criteria for Locked Test Evaluation

The selected candidate will be evaluated on the locked test partitions (Tiny-GenImage, Frontier Benchmark, WildAIGC-Bench) against the frozen baseline. Promotion to Phase 8 integration review requires meeting **ALL** of the following pre-registered conditions:

1. **Frontier Test Generalization:** Candidate Frontier test Balanced Accuracy and ROC-AUC must exceed the frozen baseline, with a paired group-aware bootstrap 95% confidence interval for the difference strictly excluding zero:
   $$\text{CI}_{95\%}(\Delta \text{BalancedAccuracy}) > 0 \quad \text{and} \quad \text{CI}_{95\%}(\Delta \text{ROC-AUC}) > 0$$
2. **Real-Image False Positive Rate:** Real-image false alarm rate on the Frontier test partition must satisfy $\text{FPR} \le 5.0\%$.
3. **JPEG Transcoding Robustness:** Under JPEG-90 transcoding, the prediction flip rate must be $< 25.0\%$ across both authentic and AI classes (compared to E3's failure rate of $45.6\%$).
4. **No Unseen Generator Collapse:** AI detection recall on each held-out generator (Midjourney, Wukong, VQDM) must remain $\ge 50.0\%$.

If the candidate fails any of these criteria, it is designated `REJECT_CANDIDATE_FOR_PRODUCTION` and the frozen baseline remains retained.
