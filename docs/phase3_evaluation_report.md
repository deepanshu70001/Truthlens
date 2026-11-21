# TruthLens Phase 3: Controlled Benchmark Evaluation, Calibration & Generalization Analysis

**Date:** 2026-11-21  
**Working Branch:** `fix/truthlens-remediation`  
**Repository:** `https://github.com/deepanshu70001/Truthlens`  
**Target Detector:** `truthlens/siglip-cifake-frozen-head`  
**Model Architecture:** Vision Transformer (`google/siglip-base-patch16-224`) with frozen trained binary linear classification head (`models/baselines/siglip_frozen_head.pt`)  
**Status:** Completed and Audited  

---

## 1. Executive Summary

TruthLens Phase 3 conducted a scientifically rigorous, leakage-safe empirical evaluation of the existing frozen authenticity detector across three externally ingested image benchmarks:
1. **Tiny-GenImage (1.0-tiny)**: Standardized synthetic generator slices and real ImageNet images (25,006 evaluated images; 9,994 held in reserve).
2. **WasItAIGenerated Frontier Benchmark (October 2026)**: Cutting-edge late-2026 generative models and 4-way social media compression variants (2,976 evaluated images).
3. **WildAIGC-Bench (v3)**: In-the-wild viral fact-checked news claims and multi-stage edited images (1,839 evaluated images).

### Core Research Objectives & Key Findings:
- **Strict Model Identity Enforced**: Evaluation verified runtime backbone `google/siglip-base-patch16-224` (revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`), head hash `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`, and preprocessing pipeline `siglip-frozen-v1`.
- **Strict Isolation of Calibration & Thresholds**:
  - Scalar temperature $T$ was fitted **exclusively on the calibration split** minimizing Negative Log-Likelihood (NLL).
  - Operating decision thresholds (Balanced Accuracy, F1, High Specificity, and Selective Abstention bounds) were chosen **exclusively on the validation split**.
  - Held-out test splits were strictly locked and evaluated once with frozen parameters.
- **Cross-Generator Generalization Gap**:
  - The detector exhibits high discriminative power on generator architectures similar to its training distribution (e.g. Stable Diffusion 1.5, BigGAN), but suffers significant performance degradation when confronted with unseen diffusion models (e.g., Midjourney, Wukong, VQDM in Tiny-GenImage) and cutting-edge 2026 foundation models (e.g. OpenAI GPT-Image-2, Flux-3, Grok Imagine 2 in Frontier Benchmark).
- **Social Media Compression Fragility**:
  - WhatsApp, Instagram, and Reshared compression pipelines consistently degrade classification accuracy and cause authentic-prediction flips on AI-generated images, confirming that JPEG quantization and chroma subsampling attenuate subtle synthetic high-frequency artifacts.
- **Statistical Rigour**:
  - All test metrics are reported with 95% confidence intervals derived from **1,000 group-level block bootstrap resamples**, guaranteeing that intra-group correlations (e.g. compression variant quadruplets in Frontier, connected fact-check components in WildAIGC) do not artificially deflate confidence intervals.

---

## 2. Methodology & Experimental Protocol

### 2.1 Model Identity Contract
Every evaluation run executes a cryptographic handshake verifying the runtime detector against `experiments/configs/phase3_evaluation_config.yaml`:
```yaml
evaluation_target:
  model_id: "truthlens/siglip-cifake-frozen-head"
  backbone_model: "google/siglip-base-patch16-224"
  backbone_revision: "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
  head_checkpoint_path: "models/baselines/siglip_frozen_head.pt"
  expected_head_sha256: "e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538"
  preprocessing_version: "siglip-frozen-v1"
  label_mapping:
    0: "real"
    1: "ai_generated"
```

### 2.2 Leakage-Safe Three-Tier Split Protocol
To ensure valid generalization claims, dataset records were partitioned in Phase 2 Step 3 into three non-overlapping roles:
- **Calibration Split (Role: `calibration`)**: Used solely to fit temperature scaling $T$ via NLL minimization.
- **Validation Split (Role: `validation`)**: Used to verify calibration generalization and select candidate operating thresholds (max balanced accuracy, max F1, 95% specificity, selective abstention bands).
- **Held-Out Test Split (Role: `test`)**: Strictly reserved. Evaluated once with frozen parameters and operating thresholds.

### 2.3 Group-Aware Block Bootstrap (1,000 Resamples)
Standard i.i.d. bootstrap resampling over-optimistically underestimates variance when samples are clustered. To preserve statistical independence:
1. Images are clustered by their canonical `group_id` (e.g. `frontier_{source_id}` for Frontier compression families, `wildaigc_cc_{id}` for WildAIGC connected components).
2. 1,000 bootstrap iterations sample independent `group_id`s with replacement.
3. All image records in each sampled group are gathered to evaluate Accuracy, Balanced Accuracy, F1-Score, and ROC-AUC.
4. Percentile confidence bounds ($2.5\%$ and $97.5\%$) define the two-sided 95% confidence intervals.

---

## 3. Benchmark Datasets Overview

| Dataset | Total Records | Calibration | Validation | Held-Out Test | Reserve (Excluded) | Grouping Strategy |
|---|---|---|---|---|---|---|
| **Tiny-GenImage** | 35,000 | 12,048 | 5,958 | 7,000 | 9,994 | Official test partition retained as test; train partitioned 50/25/25 |
| **Frontier Benchmark** | 2,976 | 588 | 297 | 2,091 | 0 | 4-way variant families (original + whatsapp + instagram + reshared) |
| **WildAIGC-Bench** | 1,839 | 364 | 181 | 1,294 | 0 | Multi-sample fact-check case groups (connected components) |
| **Total** | **39,815** | **13,000** | **6,436** | **10,385** | **9,994** | **Zero cross-role hash, group, or variant leakage** |

---

## 4. Confidence Calibration Analysis

Temperature scaling maps raw logit margins $z = \text{logit}_{\text{ai}} - \text{logit}_{\text{real}}$ to calibrated probabilities:
$$\hat{p}_{\text{ai}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z / T}}$$
where $T > 0$ is fitted by minimizing binary cross-entropy on calibration data:
$$\mathcal{L}_{\text{NLL}}(T) = -\frac{1}{N_{\text{cal}}} \sum_{i=1}^{N_{\text{cal}}} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

### Key Calibration Properties:
- **Order Preserving**: Because $\sigma(z / T)$ is strictly monotonic, temperature scaling **does not alter the ranking of samples, ROC curves, or ROC-AUC scores**.
- **Probabilistic Alignment**: It aligns raw softmax outputs with empirical true positive frequencies, significantly reducing Expected Calibration Error (ECE) and Brier scores.
- **Safety**: Uncalibrated predictions must not be used as confidence estimates or application decisions.

---

## 5. Summary of Deliverables & Artifacts

All experiment scripts, configurations, test suites, and generated artifacts are tracked and reproducible:
- **Evaluation Config**: `experiments/configs/phase3_evaluation_config.yaml`
- **Inference Script**: `experiments/evaluate_authenticity.py`
- **Calibration Script**: `experiments/calibrate_evaluation.py`
- **Results Analysis Script**: `experiments/analyze_results.py`
- **Unit Test Suite**: `backend/tests/test_phase3_evaluation.py` (10 passed)
- **Prediction Datasets**:
  - `experiments/results/phase3/tiny_genimage_predictions.csv`
  - `experiments/results/phase3/frontier_predictions.csv`
  - `experiments/results/phase3/wildaigc_predictions.csv`
  - `experiments/results/phase3/inference_summary.json`
- **Master Quantitative Reports**:
  - `experiments/reports/phase3_calibration.json`
  - `experiments/reports/phase3_robustness.json`
  - `experiments/reports/phase3_generalization.json`
  - `experiments/reports/phase3_results.json`
  - `experiments/reports/phase3_results.md`
  - `experiments/reports/phase3_failure_analysis.md`
- **Visualizations (`experiments/reports/figures/`)**:
  - `roc_curves.png`
  - `pr_curves.png`
  - `confusion_matrices.png`
  - `reliability_diagrams.png`
  - `compression_robustness.png`
  - `generator_generalization.png`
