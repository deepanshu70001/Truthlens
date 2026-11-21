# TruthLens Phase 6: Calibration, Held-Out Benchmark Evaluation & Generalization Audit Report

**Repository:** `https://github.com/deepanshu70001/Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**Milestone:** Phase 6 — Calibration, Held-Out Evaluation & Generalization Audit  
**Date:** November 2026  
**Candidate Evaluated:** `truthlens/siglip-base-tiny-genimage-e3_partial_finetuning`  
**Baseline Evaluated:** `truthlens/siglip-cifake-frozen-head`  
**Final Governance Decision:** **`RETAIN_BASELINE`** (`REJECT_CANDIDATE_FOR_PRODUCTION`)  

---

## 1. Executive Summary

Phase 6 executed a controlled, leakage-free calibration and held-out evaluation of the fine-tuned candidate model (`e3_partial_finetuning`) across the three locked benchmark test partitions:
1. **Tiny-GenImage Locked Test Partition** ($N = 7,000$: 3,500 Authentic ImageNet, 3,500 AI-Generated across 7 generators).
2. **Frontier Benchmark October 2026 Locked Test Partition** ($N = 2,091$: 1,380 Authentic, 711 AI-Generated across modern 2026 generators).
3. **WildAIGC-Bench Locked Test Partition** ($N = 1,294$: 449 Authentic, 845 AI-Generated in-the-wild web images).

The evaluation yielded a **critical scientific finding** demonstrating the vital necessity of held-out benchmark audits and domain confounding investigations:

- **In-Distribution Tiny-GenImage Test Set:** E3 appeared to achieve outstanding performance: **0.8520 Balanced Accuracy** (vs 0.5523 baseline, $\Delta = +29.97\%$) and **0.9453 ROC-AUC** (vs 0.6123 baseline, $\Delta = +33.30\%$).
- **Cross-Dataset Generalization (Frontier Benchmark):** When evaluated on modern frontier generators (Flux 1.1 Pro, Midjourney v6, SD 3.5, Ideogram 2.0, DALL-E 3) under realistic web distribution, **E3 suffered a severe collapse**, dropping to **0.5497 Balanced Accuracy** (vs **0.6228** baseline, $\Delta = -7.31\%$) and **0.5546 ROC-AUC** (vs **0.6501** baseline, $\Delta = -9.55\%$).
- **Causal Mechanism (Domain Confounding):** Controlled matched-format transcoding experiments revealed that E3 learned low-level file compression artifacts (JPEG vs PNG) rather than true generative synthesis patterns. When AI-generated PNG images were transcoded to JPEG (quality=90), **45.6% of predictions flipped** from AI to Real, with an enormous mean logit shift of **6.12 points**. Under JPEG-75, **49.5% flipped**.
- **Unseen Generator Recall Collapse:** On held-out generators never seen in training, E3's AI detection recall collapsed from near-perfection on seen generators (~99%) down to **51.60% on Midjourney**, **54.00% on Wukong**, and **20.60% on VQDM**.
- **Formal Governance Disposition:** The candidate model `e3_partial_finetuning` is **REJECTED FOR PRODUCTION DEPLOYMENT**. The frozen CIFAKE baseline (`truthlens/siglip-cifake-frozen-head`) is **RETAINED IN PRODUCTION**.

---

## 2. Model Identity & Validation Pipeline Audit (Steps 1 & 2)

Prior to opening the locked test partitions, model identity, checkpoint integrity, and validation pipeline metrics were audited:

### 2.1 Cryptographic Checkpoint Verification
- **E3 Checkpoint Path:** `experiments/checkpoints/phase5/e3_partial_finetuning/best_checkpoint.pt`
  - Expected SHA-256: `541956a1f766d5093aaec4733f616afaf6deaf5aad1e0bbd6e59f95687a4b96d`
  - Actual SHA-256: `541956a1f766d5093aaec4733f616afaf6deaf5aad1e0bbd6e59f95687a4b96d` (**MATCH**)
- **E3 Head Path:** `experiments/checkpoints/phase5/e3_partial_finetuning/head.pt`
  - Expected SHA-256: `38b12fb1931158a771cb6b64bd5745ef7f7796b27df6b9eb292c281566689b68`
  - Actual SHA-256: `38b12fb1931158a771cb6b64bd5745ef7f7796b27df6b9eb292c281566689b68` (**MATCH**)
- **Baseline Head Path:** `models/baselines/siglip_frozen_head.pt`
  - Expected SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`
  - Actual SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` (**MATCH - PRESERVED UNCHANGED**)
- **Backbone Model ID:** `google/siglip-base-patch16-224` (Revision: `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`).
- **Audit Artifact:** [phase6_model_identity_audit.json](..\experiments\reports\phase6_model_identity_audit.json).

### 2.2 Numerical Validation Reproducibility
Evaluation on the 5,958 validation samples confirmed exact numerical reproducibility with 0 variance:
- Validation Loss: `0.3990`
- Accuracy: `0.8823`
- Balanced Accuracy: `0.8667`
- Precision: `0.9574`
- Recall: `0.7586`
- Specificity: `0.9748`
- ROC-AUC: `0.9577`
- Confusion Matrix: $TN = 3,324, FP = 86, FN = 615, TP = 1,933$.

---

## 3. Confidence Calibration (Step 3)

Temperature scaling was fitted **exclusively on calibration partitions** ($N = 13,000$ total samples across Tiny-GenImage, Frontier, and WildAIGC). Zero validation or test data was used during fitting.

### Calibration Summary Table

| Calibration Dataset | Sample Count | Optimal Temperature $T^*$ | Convergence | Pre-Cal NLL | Post-Cal NLL | Pre-Cal Brier | Post-Cal Brier | Pre-Cal ECE | Post-Cal ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny-GenImage** | 12,048 | **2.0898** | True | 0.3668 | **0.2654** | 0.0915 | **0.0798** | 0.0847 | **0.0571** |
| **Frontier Benchmark** | 588 | **6.9975** | True | 1.2899 | **0.6586** | 0.3051 | **0.2324** | 0.2845 | **0.1597** |
| **WildAIGC-Bench** | 364 | **50.0000** | True (Bound) | 1.6730 | **0.6947** | 0.4528 | **0.2508** | 0.4529 | **0.1627** |
| **Pooled Calibration** | 13,000 | **2.2729** | True | 0.4451 | **0.3090** | 0.1113 | **0.0951** | 0.1010 | **0.0570** |

### Calibration Insights vs Baseline
- On Tiny-GenImage calibration data, the E3 candidate is vastly better calibrated out-of-the-box than the baseline, requiring $T = 2.0898$ (compared to the baseline's extreme $T = 14.0373$). Uncalibrated NLL dropped from $1.1033$ (baseline) to $0.3668$ (E3).
- On WildAIGC-Bench, the optimizer hit the upper search bound ($T=50.0$) to flatten probabilities to $0.50$, directly confirming the Phase 3.5 finding that margins do not correlate with ground truth on WildAIGC.
- Artifacts: [phase6_calibration.json](..\experiments\reports\phase6_calibration.json) and [phase6_reliability_data.json](..\experiments\reports\phase6_reliability_data.json).

---

## 4. Locked Test Set Evaluation Results (Step 4)

Both models were evaluated on the exact, unchanged locked test sets. Zero test samples were discarded or modified.

### 4.1 Complete Test Performance Comparison Table

| Dataset | Model | Sample Count | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | FAR | FRR | F1 Score | ROC-AUC | PR-AUC | MCC | Cohen's Kappa |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny-GenImage** | Baseline | 7,000 | 0.5523 | 0.5523 | 0.5357 | **0.7849** | 0.3197 | 0.6803 | 0.2151 | 0.6368 | 0.6123 | 0.5898 | 0.1173 | 0.1046 |
| | **E3 Candidate** | 7,000 | **0.8520** | **0.8520** | **0.9692** | 0.7271 | **0.9769** | **0.0231** | 0.2729 | **0.8309** | **0.9453** | **0.9423** | **0.7297** | **0.7040** |
| | *Paired Difference* | | *+0.2997* | *+0.2997* | *+0.4335* | *-0.0578* | *+0.6572* | *-0.6572* | *+0.0578* | *+0.1941* | *+0.3330* | *+0.3525* | *+0.6124* | *+0.5994* |
| **Frontier Benchmark** | **Baseline** | 2,091 | 0.5801 | **0.6228** | **0.4459** | **0.7689** | 0.4767 | 0.5233 | 0.2311 | **0.5645** | **0.6501** | **0.5387** | **0.2372** | **0.2185** |
| | E3 Candidate | 2,091 | **0.5863** | 0.5497 | 0.4170 | 0.4243 | **0.6751** | **0.3249** | 0.5757 | 0.4206 | 0.5546 | 0.4199 | 0.0993 | 0.0970 |
| | *Paired Difference* | | *+0.0062* | *-0.0731* | *-0.0289* | *-0.3446* | *+0.1984* | *-0.1984* | *+0.3446* | *-0.1439* | *-0.0955* | *-0.1188* | *-0.1379* | *-0.1215* |
| **WildAIGC-Bench** | Baseline | 1,294 | **0.4490** | 0.4294 | 0.5940 | **0.4935** | 0.3653 | 0.6347 | 0.5065 | **0.5391** | 0.4226 | 0.6033 | -0.1420 | -0.1378 |
| | E3 Candidate | 1,294 | 0.4042 | **0.4780** | **0.6135** | 0.2367 | **0.7194** | **0.2806** | 0.7633 | 0.3416 | **0.5181** | **0.6698** | -0.0487 | -0.0388 |
| | *Paired Difference* | | *-0.0448* | *+0.0486* | *+0.0195* | *-0.2568* | *+0.3541* | *-0.3541* | *+0.2568* | *-0.1975* | *+0.0955* | *+0.0665* | *+0.0933* | *+0.0990* |

### 4.2 Confusion Matrix Breakdown
- **Tiny-GenImage ($N=7,000$):**
  - Baseline: $TN = 1,119, FP = 2,381, FN = 753, TP = 2,747$
  - E3 Candidate: $TN = 3,419, FP = 81, FN = 955, TP = 2,545$
- **Frontier Benchmark ($N=2,091$):**
  - Baseline: $TN = 658, FP = 722, FN = 164, TP = 547$
  - E3 Candidate: $TN = 931, FP = 449, FN = 410, TP = 301$
- **WildAIGC-Bench ($N=1,294$):**
  - Baseline: $TN = 164, FP = 285, FN = 428, TP = 417$
  - E3 Candidate: $TN = 323, FP = 126, FN = 645, TP = 200$

> [!WARNING]
> **WildAIGC-Bench Label Validity Limitation:**  
> As formally proven in Phase 3.5, WildAIGC-Bench ground truth labels classify whether an image was shared in response to an AI-prompt context, and contains heavy proportions of human-edited, natural, or watermarked imagery. Neither model achieves meaningful discrimination above chance ($AUC \approx 0.42 - 0.52$). WildAIGC metrics must be interpreted with this proven limitation.

---

## 5. Domain Confounding & Transcoding Investigation (Step 5)

### 5.1 The Confounding Hypothesis
In the Tiny-GenImage benchmark:
- 100% of authentic nature images are **JPEG** (ImageNet ILSVRC2012).
- 100% of synthetic images are **PNG** (lossless exports from ADM, BigGAN, SD 1.5, GLIDE, Midjourney, Wukong, VQDM).
- Class prefix overlap is **0**.

### 5.2 Controlled Matched-Format Transcoding Experiment
To test whether the model learned compression format rather than generative artifacts, 1,000 paired test images (500 Real JPEG, 500 AI PNG) were transcoded in memory into matched formats (JPEG-90, JPEG-75, and lossless PNG) while preserving semantic pixel content:

| Transcoding Transformation | Baseline Prediction Flips | Baseline Flip Rate (%) | Baseline Mean $|\Delta \text{Logit}|$ | E3 Candidate Prediction Flips | E3 Candidate Flip Rate (%) | E3 Candidate Mean $|\Delta \text{Logit}|$ | E3 Candidate Max $|\Delta \text{Logit}|$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **JPEG Quality 90** | 142 / 1,000 | 14.2% | 0.7078 | **456 / 1,000** | **45.6%** | **6.1198** | **15.9258** |
| **JPEG Quality 75** | 173 / 1,000 | 17.3% | 0.8294 | **495 / 1,000** | **49.5%** | **6.7081** | **15.9336** |
| **PNG (Lossless)** | 0 / 1,000 | 0.0% | 0.0000 | 0 / 1,000 | 0.0% | 0.0000 | 0.0000 |

### 5.3 Scientific Proof of Shortcut Learning
1. **Catastrophic Sensitivity to JPEG Transcoding:** When synthetic PNG images are compressed with standard JPEG ($Q=90$), nearly half (**45.6%**) immediately flip their prediction to "Real". At $Q=75$, **49.5% flip**.
2. **Extreme Logit Shifts:** E3's logits dropped by an average of **$6.12$ to $6.71$ points** purely from JPEG encoding (with peak perturbations exceeding $15.9$ points). In contrast, the frozen baseline shifted by only $0.71$ to $0.83$ points.
3. **Causal Interpretation:** Fine-tuning unfreezing transformer layer 11 on the unaugmented Tiny-GenImage dataset severely overfitted to the high-frequency spectral signature of uncompressed PNG files. The candidate treats JPEG compression artifacts as proof of authenticity.
4. **Artifacts:** [phase6_domain_confounding.json](..\experiments\reports\phase6_domain_confounding.json) and [phase6_encoding_robustness.csv](..\experiments\reports\phase6_encoding_robustness.csv).

---

## 6. Generator Generalization Audit (Step 6)

Evaluating performance by generator on the Tiny-GenImage test set reveals the exact boundary of generalization:

### Generator Performance Breakdown

| Generator | Protocol Category | Sample Count (AI) | Baseline AI Detection Recall | E3 Candidate AI Detection Recall | Baseline Balanced Accuracy | E3 Candidate Balanced Accuracy | $\Delta \text{BalAcc}$ (E3 - Base) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ADM** | `SEEN_TRAINING` | 500 | 67.40% | **99.40%** | 0.4969 | **0.9854** | **+0.4885** |
| **BigGAN** | `SEEN_TRAINING` | 500 | 60.40% | **99.60%** | 0.4619 | **0.9864** | **+0.5245** |
| **GLIDE** | `SEEN_TRAINING` | 500 | 89.40% | **99.40%** | 0.6069 | **0.9854** | **+0.3785** |
| **SD v1.5** | `SEEN_TRAINING` | 500 | 92.80% | **84.40%** | 0.6239 | **0.9104** | **+0.2865** |
| **Midjourney** | `HELD_OUT_UNSEEN` | 500 | **89.00%** | **51.60%** | 0.6049 | **0.7464** | **+0.1415** |
| **Wukong** | `HELD_OUT_UNSEEN` | 500 | **87.40%** | **54.00%** | 0.5969 | **0.7584** | **+0.1615** |
| **VQDM** | `HELD_OUT_UNSEEN` | 500 | **63.00%** | **20.60%** | 0.4749 | **0.5914** | **+0.1165** |

### Key Takeaways:
- On **seen generators** (ADM, BigGAN, GLIDE), E3 approaches 100% detection accuracy.
- On **held-out unseen generators**, E3's AI recall drops precipitously: **Midjourney drops by -37.40%**, **Wukong drops by -33.40%**, and **VQDM drops by -42.40%** (detecting only 20.6% of VQDM images).
- While balanced accuracy remains higher than the baseline due to E3's high specificity on ImageNet nature images (97.7% vs 32.0%), E3 fails to reliably identify synthetic content from unseen generative architectures.
- Artifact: [phase6_generator_generalization.json](..\experiments\reports\phase6_generator_generalization.json).

---

## 7. Frontier Compression Robustness (Step 7)

Evaluating social media compression variants on the Frontier benchmark:

| Social Media Variant | Sample Count | Baseline Balanced Accuracy | Baseline ROC-AUC | E3 Candidate Balanced Accuracy | E3 Candidate ROC-AUC | $\Delta \text{ROC-AUC}$ (E3 - Base) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original (Web Uncompressed)** | 1,046 | 0.6164 | 0.6395 | 0.5689 | 0.5750 | **-0.0645** |
| **WhatsApp Compression** | 349 | 0.6175 | 0.6444 | 0.5496 | 0.5591 | **-0.0853** |
| **Instagram Compression** | 348 | 0.6375 | 0.6647 | 0.5355 | 0.5403 | **-0.1244** |
| **Reshared (Multiple recompressions)** | 348 | 0.6200 | 0.6519 | 0.5446 | 0.5435 | **-0.1084** |

Under every compression variant, E3 lags behind the frozen baseline by $6.5$ to $12.4$ ROC-AUC points.
Artifact: [phase6_compression_robustness.json](..\experiments\reports\phase6_compression_robustness.json).

---

## 8. Statistical Comparison & Group Bootstrap (Step 8)

Group-aware stratified bootstrap resampling ($B = 1,000$ resamples) computed 95% confidence intervals for all paired metric differences:

### Stratified Bootstrap Differences (E3 - Baseline) [95% CI]

| Dataset | Metric | Mean Difference | 95% Bootstrap Confidence Interval | Statistical Significance ($p < 0.05$) |
| :--- | :--- | :--- | :--- | :--- |
| **Tiny-GenImage** | $\Delta \text{Balanced Accuracy}$ | **+0.3032** | `[+0.2444, +0.3457]` | Significant Improvement (In-Domain) |
| | $\Delta \text{ROC-AUC}$ | **+0.3342** | `[+0.3155, +0.3553]` | Significant Improvement (In-Domain) |
| | $\Delta \text{F1 Score}$ | **+0.2020** | `[+0.0754, +0.2889]` | Significant Improvement (In-Domain) |
| **Frontier Benchmark** | $\Delta \text{Balanced Accuracy}$ | **-0.0740** | `[-0.1152, -0.0315]` | **Significant Degradation (Cross-Domain)** |
| | $\Delta \text{ROC-AUC}$ | **-0.0957** | `[-0.1453, -0.0436]` | **Significant Degradation (Cross-Domain)** |
| | $\Delta \text{F1 Score}$ | **-0.1454** | `[-0.2026, -0.0905]` | **Significant Degradation (Cross-Domain)** |
| **WildAIGC-Bench** | $\Delta \text{Balanced Accuracy}$ | **+0.0488** | `[+0.0058, +0.0910]` | Borderline (Noise / Label Limit) |
| | $\Delta \text{ROC-AUC}$ | **+0.0956** | `[+0.0429, +0.1505]` | Borderline (Noise / Label Limit) |
| | $\Delta \text{F1 Score}$ | **-0.1974** | `[-0.2531, -0.1407]` | Significant Drop |

On the modern Frontier benchmark, the upper bound of the 95% confidence interval for $\Delta \text{Balanced Accuracy}$ is **$-0.0315$** (strictly negative). The performance drop is statistically proven and not an artifact of random sampling.
Artifact: [phase6_statistical_comparison.json](..\experiments\reports\phase6_statistical_comparison.json).

---

## 9. Final Candidate Decision (Step 9)

### Formal Decision: **`RETAIN_BASELINE`** (`REJECT_CANDIDATE_FOR_PRODUCTION`)

### Justification:
1. **Cross-Dataset Failure:** The candidate fails cross-dataset generalization, underperforming the frozen CIFAKE baseline on the modern Frontier benchmark by $-7.40\%$ Balanced Accuracy and $-9.57\%$ ROC-AUC.
2. **Vulnerability to Format Shortcut:** Matched transcoding confirmed that E3 learned JPEG vs PNG compression artifacts as a shortcut for authenticity. Compressing an AI image with standard JPEG causes a **45.6% - 49.5% failure rate**, making the model dangerously susceptible to false negatives on social media.
3. **Unseen Generator Fragility:** E3's AI recall drops down to $20.6\%$ on VQDM, $51.6\%$ on Midjourney, and $54.0\%$ on Wukong.
4. **Safety & Abstention:** Under TruthLens safety standards, a candidate model that introduces domain shortcuts and worsens performance on modern frontier generators cannot be promoted to production.

Artifact: [phase6_candidate_decision.json](..\experiments\reports\phase6_candidate_decision.json).

### Recommendations for Phase 7:
1. **Retain the frozen CIFAKE baseline** (`models/baselines/siglip_frozen_head.pt`) in the production detector.
2. **Data Augmentation:** In any subsequent fine-tuning research, implement aggressive compression-invariant data augmentation during training (random JPEG compression with variable quality $Q \in [40, 95]$, WebP transcoding, and Gaussian blur) to prevent the network from learning container-level artifacts.
3. **Format-Balanced Datasets:** Ensure all training classes contain equal proportions of uncompressed PNG and compressed JPEG images.

---

## 10. Publication-Quality Figures Generated

All figures are serialized in `experiments/reports/figures/`:
1. `phase6_roc_curves.png` — Side-by-side ROC curves across Tiny-GenImage, Frontier, and WildAIGC.
2. `phase6_pr_curves.png` — Precision-Recall curves across all three benchmarks.
3. `phase6_confusion_matrices.png` — Normalized confusion matrices for Baseline vs E3.
4. `phase6_reliability_diagrams.png` — Reliability diagrams showing pre- and post-calibration calibration curves.
5. `phase6_generator_performance.png` — Cross-generator detection rates comparing seen vs held-out generators.
6. `phase6_encoding_robustness.png` — Bar plots of prediction flip rates and logit perturbation shifts under JPEG and PNG transcoding.
7. `phase6_compression_robustness.png` — ROC-AUC across WhatsApp, Instagram, and Reshared social media variants.
8. `phase6_paired_metric_differences.png` — Forest plot of paired differences with 95% bootstrap confidence intervals.

---

## 11. Explicit Answers to the 14 Stop Condition Questions

1. **Did model identity verification pass?**  
   **PASS.** E3 checkpoint SHA-256 (`541956a1...`), head SHA-256 (`38b12fb1...`), and baseline head SHA-256 (`e39afc55...`) matched expected hashes identically.
2. **Was E3 validation performance reproduced?**  
   **PASS.** All metrics on the 5,958 validation samples matched Phase 5 reported values exactly (Balanced Acc: 0.8667, ROC-AUC: 0.9577, Accuracy: 0.8823).
3. **What calibration temperature was fitted?**  
   $T = 2.0898$ on Tiny-GenImage calibration; $T = 6.9975$ on Frontier calibration; $T = 50.0000$ on WildAIGC; pooled $T = 2.2729$.
4. **What are the calibrated NLL, Brier score, and ECE?**  
   On Tiny-GenImage calibration: NLL dropped from $0.3668 \rightarrow 0.2654$; Brier score dropped from $0.0915 \rightarrow 0.0798$; ECE dropped from $0.0847 \rightarrow 0.0571$.
5. **What are E3's Tiny-GenImage test results?**  
   Accuracy: 0.8520, Balanced Accuracy: 0.8520, Precision: 0.9692, Recall: 0.7271, Specificity: 0.9769, F1: 0.8309, ROC-AUC: 0.9453.
6. **What are E3's Frontier test results?**  
   Accuracy: 0.5863, Balanced Accuracy: 0.5497, Precision: 0.4170, Recall: 0.4243, Specificity: 0.6751, F1: 0.4206, ROC-AUC: 0.5546.
7. **What are E3's WildAIGC results, with label limitations clearly stated?**  
   Accuracy: 0.4042, Balanced Accuracy: 0.4780, ROC-AUC: 0.5181. *Limitation:* Labels reflect contextual prompt associations rather than synthetic image pixels; both models show chance-level performance.
8. **How does E3 compare with the original baseline?**  
   E3 outperforms baseline on Tiny-GenImage (+29.97% BalAcc, +33.30% AUC), but degrades significantly on the modern Frontier benchmark (-7.31% BalAcc, -9.55% AUC).
9. **How does it perform on Midjourney, Wukong, and VQDM?**  
   AI detection recall collapses to 51.60% on Midjourney, 54.00% on Wukong, and 20.60% on VQDM (compared to ~99% on seen generators).
10. **Does JPEG/PNG transcoding change predictions?**  
    **YES, CATASTROPHICALLY.** JPEG transcoding of AI PNG images causes **45.6% - 49.5% prediction flips** with an average logit drop of $6.12$ to $6.71$ points.
11. **Does compression reduce performance?**  
    **Yes.** Across social media compression variants on Frontier, E3's ROC-AUC drops to $0.54 - 0.56$ (below baseline's $0.64 - 0.66$).
12. **Were test samples kept unchanged and isolated?**  
    **PASS.** 100% of test samples, labels, and splits were preserved without any alteration.
13. **What is the final candidate decision?**  
    **`RETAIN_BASELINE`** (`REJECT_CANDIDATE_FOR_PRODUCTION`).
14. **Did all relevant tests pass?**  
    **PASS.** All regression tests pass with 0 errors.
