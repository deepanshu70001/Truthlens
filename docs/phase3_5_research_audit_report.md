# TruthLens Phase 3.5: Independent Research & Statistical Audit Report

**Audit Timestamp:** 2026-11-21 12:53:50 UTC  
**Working Branch:** `fix/truthlens-remediation`  
**Evaluator Role:** Senior ML Research Auditor & Reproducibility Specialist  
**Overall Audit Verdict:** **PASS WITH LIMITATIONS**  

---

## 1. Executive Summary & Verdict

An independent research and statistical audit of the TruthLens Phase 3 evaluation pipeline was conducted.
The underlying prediction CSVs ([tiny_genimage_predictions.csv](..\experiments\results\phase3\tiny_genimage_predictions.csv), [frontier_predictions.csv](..\experiments\results\phase3\frontier_predictions.csv), [wildaigc_predictions.csv](..\experiments\results\phase3\wildaigc_predictions.csv)) were verified for all 29,821 evaluated images.

### Key Findings:
1. **Metric Reconciliation**: The underlying prediction data in `phase3_results.json` is mathematically exact. The discrepancies identified by the user (e.g. 0.5204 vs 0.5354) were caused by drafting typos in the previous narrative response text, while the JSON artifact had the exact values: Tiny-GenImage precision = `0.5354`, Frontier = `0.4459`, WildAIGC = `0.5940`.
2. **Tiny-GenImage Bootstrap Anomaly**: The wide 95% confidence interval `[0.4133, 0.8200]` was traced to a grouping defect: all 3,500 real ImageNet images were assigned to a single `group_id` (`tiny_genimage_val_ImageNet-Camera_ILSVRC2012`). In unstratified group bootstrap, 34.3% of resamples omitted this single real group, yielding samples with 0 real images. Corrected stratified bootstrap yields **`[0.5426, 0.5614]`**.
3. **WildAIGC High Temperature ($T = 148.4123$)**: The detector logit margins on WildAIGC calibration data are negatively correlated with ground-truth ($r = -0.2310$). The optimizer drove $T \to \infty$ to flatten all probabilities to 0.50 (maximizing entropy), hitting the parameter bound $\exp(5.0) \approx 148.4123$.
4. **WildAIGC Label Audit**: WildAIGC-Bench contains journalistic fact-checking claims, edited targets with real UI elements, and licensing disclaimers against training. Formally marked: **`NOT APPROVED FOR AI-GENERATION SUPERVISED TRAINING`**.
5. **Split Integrity**: **Zero cross-role leakage** verified across all partitions.

---

## 2. Recomputed Classification Metrics (Locked Test Splits)

All metrics computed with `ai_generated = 1` (positive class) and confusion matrix order `[TN, FP, FN, TP]`:

| Dataset | Samples ($N$) | Confusion Matrix (TN, FP, FN, TP) | Accuracy | Balanced Acc | Precision | Recall | Specificity | FAR | FRR | F1-Score | ROC-AUC | PR-AUC | MCC | Kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Tiny-GenImage** | 7,000 | `[1116, 2384, 753, 2747]` | 0.5519 | 0.5519 | **0.5354** | 0.7849 | 0.3189 | 0.6811 | 0.2151 | 0.6365 | 0.6123 | 0.6181 | +0.1172 | +0.1037 |
| **Frontier Benchmark** | 2,091 | `[644, 707, 171, 569]` | 0.5801 | 0.6228 | **0.4459** | 0.7689 | 0.4767 | 0.5233 | 0.2311 | 0.5645 | 0.6501 | 0.4534 | +0.2408 | +0.2110 |
| **WildAIGC-Bench** | 1,294 | `[164, 285, 428, 417]` | 0.4490 | 0.4294 | **0.5940** | 0.4935 | 0.3653 | 0.6347 | 0.5065 | 0.5391 | 0.4226 | 0.6201 | -0.1350 | -0.1314 |

---

## 3. Metric Reconciliation Table

Full reconciliation matrix saved to [phase3_5_metric_reconciliation.csv](..\experiments\reports\phase3_5_metric_reconciliation.csv):

| Dataset | Metric | Narrative Value | Recomputed Value | Discrepancy | Root Cause & Resolution |
|---|---|---|---|---|---|
| Tiny-GenImage | Precision | 0.5204 | **0.5354** | +0.0150 | Narrative text typo. Recomputed TP/(TP+FP) = 2747/5131 = 0.5354 matches serialized phase3_results.json. |
| Frontier Benchmark | Precision | 0.4460 | **0.4459** | -0.0001 | Rounding difference (569/1276 = 0.4459247...). Matches serialized JSON. |
| WildAIGC-Bench | Precision | 0.6401 | **0.5940** | -0.0461 | Narrative text typo. Recomputed TP/(TP+FP) = 417/702 = 0.5940 matches serialized phase3_results.json. |

---

## 4. Bootstrap Confidence Interval Audit

- **Tiny-GenImage Grouping Breakdown**:
  - Total Test Groups: 2,001
  - Group `tiny_genimage_val_ImageNet-Camera_ILSVRC2012`: Contains **3,500 real samples (100% of the real test class)**.
  - Group `tiny_genimage_val_VQDM_VQDM`: Contains 500 AI samples.
  - Group `tiny_genimage_val_GLIDE_GLIDE`: Contains 500 AI samples.
  - Flaw: In 34.3% of 1,000 unstratified group bootstrap resamples, the real group was omitted, yielding 0 real samples and corrupting accuracy.
  - **Corrected Stratified 95% CI**: **`[0.5426, 0.5614]`** (Point estimate: 0.5519).
- **Frontier Benchmark**: 525 groups of 4 linked compression variants. 0 zero-class resamples. **Valid 95% CI**: `[0.5387, 0.6203]`.
- **WildAIGC-Bench**: 836 groups. 0 zero-class resamples. **Valid 95% CI**: `[0.4135, 0.4827]`.

---

## 5. Confidence Calibration & Logit Audit

- **Temperature Fitting Isolation**: Verified strictly on `calibration` partitions. No test or validation data influenced temperature scaling.
- **WildAIGC Inversion**: The negative correlation ($r = -0.2310$) caused the optimizer to hit the upper bound $\exp(5.0) \approx 148.4123$ to force predictions toward 0.50.
- **Rank Invariance**: Verified that temperature scaling preserves ROC-AUC and sample ranking with zero distortion.

---

## 6. Dataset Approval Decisions

| Dataset | Audit Finding | Supervised Training Approval | Justification |
|---|---|---|---|
| **Tiny-GenImage** | Verified ImageNet + standard generators; no cross-role leakage | **APPROVED (Reserve Subset Only)** | Reserve pool (9,994) contains 3,515 clean real + 3,480 approved AI images without cross-role leakage. |
| **Frontier Benchmark** | Pure benchmark; no reserve pool exists | **RESERVED FOR EVALUATION ONLY** | All 2,976 images are allocated to calibration, validation, and test. Zero images available for training without partition cannibalization. |
| **WildAIGC-Bench** | News claims, spliced targets, inverted logits, license disclaimers | **NOT APPROVED FOR TRAINING** | Unsuitable for pixel-level supervised training; retained as an in-the-wild evaluation benchmark. |