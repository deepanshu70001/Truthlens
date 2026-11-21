# TruthLens Phase 3: Controlled Benchmark Evaluation Report

**Date:** 2026-11-21 12:40:57 UTC  
**Target Detector:** `truthlens/siglip-cifake-frozen-head`  
**Backbone:** `google/siglip-base-patch16-224` (`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`)  
**Classification Head:** `models/baselines/siglip_frozen_head.pt` (`e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`)  
**Evaluation Protocol:** Group-Aware Leakage-Safe Benchmarks (Tiny-GenImage, Frontier Benchmark 2026-10, WildAIGC-Bench)  

---

## 1. Locked Held-Out Test Evaluation Results

All test evaluations were conducted on strictly held-out partitions with frozen thresholds selected on validation data.

| Dataset | Test Samples | Accuracy (95% CI) | Balanced Acc (95% CI) | F1-Score (95% CI) | ROC-AUC (95% CI) | PR-AUC | FAR | FRR |
|---|---|---|---|---|---|---|---|---|
| **tiny_genimage** | 7000 | 0.5519 [0.4133, 0.8200] | 0.5519 [0.5291, 0.8200] | 0.6365 [0.3505, 0.9011] | 0.6123 [0.5678, 0.6558] | 0.6181 | 0.6811 | 0.2151 |
| **frontier_benchmark_2026_10** | 2091 | 0.5801 [0.5387, 0.6203] | 0.6228 [0.5844, 0.6594] | 0.5645 [0.5118, 0.6133] | 0.6501 [0.6008, 0.6954] | 0.4534 | 0.5233 | 0.2311 |
| **wildaigc_bench** | 1294 | 0.4490 [0.4135, 0.4827] | 0.4294 [0.3964, 0.4608] | 0.5391 [0.4964, 0.5779] | 0.4226 [0.3843, 0.4606] | 0.6201 | 0.6347 | 0.5065 |

---

## 2. Confidence Calibration & Temperature Scaling

Temperature parameters were fitted strictly on the calibration partitions to minimize NLL.

| Dataset | Fitted T | Uncalibrated ECE (15-bin) | Calibrated ECE (15-bin) | Uncalibrated Brier | Calibrated Brier | Uncalibrated NLL | Calibrated NLL |
|---|---|---|---|---|---|---|---|
| **tiny_genimage** | 14.0373 | 0.2607 | 0.0636 | 0.3186 | 0.2448 | 1.0068 | 0.6826 |
| **frontier_benchmark_2026_10** | 4.6642 | 0.2481 | 0.1748 | 0.3012 | 0.2458 | 0.9477 | 0.6858 |
| **wildaigc_bench** | 148.4123 | 0.3146 | 0.1526 | 0.3631 | 0.2502 | 1.0628 | 0.6936 |

---

## 3. Social Media Compression Robustness (Frontier Benchmark)

Impact of standard social platform compression pipelines on detection performance:

| Variant | Samples | Accuracy | Balanced Acc | F1-Score | ROC-AUC | Delta Acc vs Orig | Flip Rate vs Orig |
|---|---|---|---|---|---|---|---|
| **Original** | 525 | 0.5638 | 0.6164 | 0.5621 | 0.6395 | Ref (0.000) | Ref (0.000) |
| **Whatsapp** | 522 | 0.5785 | 0.6175 | 0.5582 | 0.6442 | +0.0147 | 0.0479 |
| **Instagram** | 522 | 0.5996 | 0.6375 | 0.5761 | 0.6648 | +0.0358 | 0.0575 |
| **Reshared** | 522 | 0.5785 | 0.6200 | 0.5618 | 0.6519 | +0.0147 | 0.0556 |

---

## 4. Cross-Generator Generalization

### Tiny-GenImage: Seen vs Unseen Architectures
- **Seen Training Generators (SD1.5, BigGAN, ADM, GLIDE):** Accuracy = 0.4847, Balanced Acc = 0.5469, AUC = 0.5988
- **Unseen Held-Out Generators (Midjourney, Wukong, VQDM):** Accuracy = 0.4626, Balanced Acc = 0.5584, AUC = 0.6303
- **Generalization Gap:** Accuracy drop of 0.0221, AUC drop of -0.0315

### Frontier Benchmark: Seen vs Held-Out Frontier Generators
- **Calibration Families Slice:** Accuracy = 0.5751, AUC = 0.6856
- **Held-Out Frontier Slice (GPT-Image-2, Flux-3, Grok-Imagine-2, etc.):** Accuracy = 0.4951, AUC = 0.5396
- **Generalization Gap:** Accuracy drop of 0.0800, AUC drop of 0.1460

---

## 5. Artifact Directory & Visualizations

- Master Results: `experiments/reports/phase3_results.json`
- Calibration Diagnostics: `experiments/reports/phase3_calibration.json`
- Robustness Report: `experiments/reports/phase3_robustness.json`
- Generalization Report: `experiments/reports/phase3_generalization.json`
- Detailed Failure Analysis: `experiments/reports/phase3_failure_analysis.md`
- Figures:
  - `experiments/reports/figures/roc_curves.png`
  - `experiments/reports/figures/pr_curves.png`
  - `experiments/reports/figures/confusion_matrices.png`
  - `experiments/reports/figures/reliability_diagrams.png`
  - `experiments/reports/figures/compression_robustness.png`
  - `experiments/reports/figures/generator_generalization.png`