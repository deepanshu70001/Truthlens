# TruthLens Scientific Uncertainty & Calibration Research Report

**Document Version:** `1.0.0`  
**Date:** 2026-11-24  
**Benchmark Suite:** `experiments/evaluate_uncertainty_calibration.py`  
**Research Artifact:** `experiments/reports/uncertainty_calibration_research.json`  
**Dataset Scope:** Authorized training split ($n = 6,995$; 3,515 authentic, 3,480 AI-generated)  
**Dataset Source:** `research/experiment2_forensic_screening/siglip_logits.csv` & `experiments/reports/phase9_feature_vectors.npz`  
**Model Identity:** `truthlens/siglip-cifake-frozen-head` (SHA-256: `e39afc5536a0...`)

---

## 1. Executive Summary & Research Motivation

A critical requirement of multimodal verification systems is that confidence scores presented to forensic analysts must correspond to actual empirical probabilities of authenticity or generation. Prior evaluations established that deep neural networks trained with Cross-Entropy / Binary Cross-Entropy loss frequently become overconfident on ambiguous samples.

This research report establishes:
1. Quantitative measurement of **uncalibrated SigLIP probability distortion** (ECE, MCE, Brier score, and reliability diagrams).
2. Characterization of predictive uncertainty through **Maximum Softmax Probability (MSP)** and **Shannon Predictive Entropy**.
3. Exploratory post-hoc **temperature scaling optimization ($T^*$)** on authorized non-locked training data.
4. An empirical **error analysis stratified across confidence deciles**.
5. **Selective prediction and risk-coverage trade-offs** under entropy-based abstention.

> [!IMPORTANT]
> **Strict Non-Negotiable Invariance Preservation:**  
> The temperature parameter fitted in this empirical study ($T^* = 5.9432$) is strictly an offline research characterization. In compliance with Package C master rules, no production calibration parameter, threshold, checkpoint, or evidence fusion rule (`e11.conservative.v1`) was modified.

---

## 2. Experimental Setup & Mathematical Formulations

### 2.1 Dataset Inventory
- **Authorized Training Split ($n = 6,995$):** 3,515 authentic images and 3,480 AI-generated images.
- **Logit Provenance:** Extracted from the frozen SigLIP linear head without unsealing protected calibration or held-out test partitions.

### 2.2 Metrics Formulations
- **Maximum Softmax Probability (MSP):**
  $$c_i = \max(p_i, 1 - p_i) \in [0.5, 1.0]$$
- **Shannon Predictive Entropy:**
  $$\mathcal{H}(p_i) = -p_i \log_2(p_i) - (1 - p_i) \log_2(1 - p_i) \in [0.0, 1.0]\ \text{bits}$$
- **Brier Score (Mean Squared Error on Probabilities):**
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Maximum Calibration Error (MCE):**
  $$\text{MCE} = \max_{m \in \{1,\dots,M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Temperature Scaling (Post-Hoc Optimization):**
  $$p_i^{\text{cal}} = \sigma\left(\frac{z_i}{T}\right), \quad T^* = \arg\min_T \text{NLL}(z, y; T)$$

---

## 3. Quantitative Calibration & Uncertainty Results

### 3.1 Uncalibrated Baseline vs. Exploratory Temperature Scaling

| Metric | Uncalibrated Baseline ($T=1.0$) | Temperature Scaled ($T^*=5.9432$) | Delta ($\Delta$) | Research Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Negative Log-Likelihood (NLL)** | 0.9767 | **0.6725** | -0.3042 (-31.1%) | Substantial improvement in probabilistic fit |
| **Expected Calibration Error (10 bins)** | 0.2472 (24.72%) | **0.0313 (3.13%)** | **-0.2159 (-87.3%)** | Severe overconfidence eliminated |
| **Expected Calibration Error (15 bins)** | 0.2472 (24.72%) | **0.0313 (3.13%)** | -0.2159 (-87.3%) | Invariant across bin resolutions |
| **Maximum Calibration Error (MCE)** | 0.3474 (34.74%) | **0.1985 (19.85%)** | -0.1489 (-42.9%) | Worst-case bin discrepancy reduced |
| **Brier Score** | 0.3092 | **0.2400** | -0.0692 (-22.4%) | Outperforms uninformative baseline ($0.2500$) |
| **Mean Confidence (MSP)** | 0.8121 (81.21%) | **0.5574 (55.74%)** | -0.2547 | Aligns confidence with actual $56.48\%$ accuracy |
| **Mean Predictive Entropy** | 0.5821 bits | **0.9702 bits** | +0.3881 bits | Accurately reflects high decision boundary entropy |

---

### 3.2 Reliability Diagram Analysis (10 Bins)

#### Uncalibrated Model Reliability
The uncalibrated model exhibits severe systematic overconfidence across upper confidence deciles:
- Samples in bin $[0.95, 1.00]$ ($n = 2,674$, $38.2\%$ of dataset) display a mean confidence of **$98.54\%$**, but achieve an empirical accuracy of only **$63.80\%$** (a calibration gap of **$34.74\%$**).
- Samples in bin $[0.85, 0.90]$ display a mean confidence of **$87.72\%$**, yet achieve an accuracy of only **$54.67\%$** (a calibration gap of **$33.05\%$**).

#### Temperature-Scaled Model Reliability ($T^* = 5.9432$)
Applying temperature softening contracts logit magnitudes:
- 6,009 samples ($85.9\%$) are mapped into the $[0.50, 0.60]$ confidence range with an empirical accuracy of **$55.22\%$** and mean confidence of **$54.51\%$** (an exceptional calibration gap of only **$0.71\%$**).
- ECE is compressed to **$3.13\%$**, proving that temperature scaling resolves overconfidence distortion without altering class ranking.

---

### 3.3 Error Analysis Stratified Across Confidence Deciles

| Confidence Decile | Sample Count | Percentage of Split | Error Count | Empirical Accuracy | Mean Entropy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$[0.50, 0.55)$** | 561 | 8.02% | 277 | 50.62% | 0.9984 bits |
| **$[0.55, 0.60)$** | 536 | 7.66% | 258 | 51.87% | 0.9839 bits |
| **$[0.60, 0.65)$** | 499 | 7.13% | 240 | 51.90% | 0.9577 bits |
| **$[0.65, 0.70)$** | 467 | 6.68% | 224 | 52.03% | 0.9168 bits |
| **$[0.70, 0.75)$** | 443 | 6.33% | 215 | 51.47% | 0.8584 bits |
| **$[0.75, 0.80)$** | 442 | 6.32% | 205 | 53.62% | 0.7766 bits |
| **$[0.80, 0.85)$** | 436 | 6.23% | 196 | 55.05% | 0.6698 bits |
| **$[0.85, 0.90)$** | 450 | 6.43% | 204 | 54.67% | 0.5362 bits |
| **$[0.90, 0.95)$** | 487 | 6.96% | 257 | 47.23% | 0.3804 bits |
| **$[0.95, 1.00]$** | 2,674 | 38.23% | 968 | **63.80%** | 0.1704 bits |

---

### 3.4 Selective Prediction & Risk-Coverage Dynamics

Selective prediction allows TruthLens to abstain on uncertain images (returning `outcome="inconclusive"` or `direction="informational"`), evaluating only samples where confidence exceeds a threshold.

| Target Coverage | Decided Samples | Abstained Samples | Accuracy on Decided | Error Rate (Risk) | Confidence Threshold | Max Entropy Threshold |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100.0%** | 6,995 | 0 | 56.48% | 43.52% | 0.5000 | 1.0000 bits |
| **90.0%** | 6,295 | 700 | 57.06% | 42.94% | 0.5694 | 0.9856 bits |
| **80.0%** | 5,596 | 1,399 | 57.77% | 42.23% | 0.6534 | 0.9312 bits |
| **70.0%** | 4,896 | 2,099 | 58.74% | 41.26% | 0.7494 | 0.8122 bits |
| **60.0%** | 4,197 | 2,798 | 60.04% | 39.96% | 0.8576 | 0.5901 bits |
| **50.0%** | 3,497 | 3,498 | **62.51%** | **37.49%** | **0.9490** | **0.3012 bits** |
| **40.0%** | 2,798 | 4,197 | 63.69% | 36.31% | 0.9668 | 0.2104 bits |
| **30.0%** | 2,098 | 4,897 | 64.73% | 35.27% | 0.9791 | 0.1462 bits |
| **20.0%** | 1,399 | 5,596 | **66.19%** | **33.81%** | 0.9888 | 0.0901 bits |

- **Effective Error Reduction (EER at 50% Coverage):** Abstaining on the $50\%$ most uncertain inputs reduces classification error from **$43.52\%$ to $37.49\%$** (a **$13.8\%$ relative risk reduction**).
- **High-Precision Operating Point (20% Coverage):** At $20\%$ coverage ($\mathcal{H} \le 0.09$ bits), accuracy rises to **$66.19\%$**.

---

## 4. Operational Takeaways for Production Architecture

1. **Raw Confidence Rejection:** Raw SigLIP softmax scores must never be presented as empirical probabilities in user interfaces or operational reports without prominent calibration disclaimers.
2. **Conservative Fusion Validation:** The $24.72\%$ calibration gap provides empirical justification for `e11.conservative.v1`, which refuses to cast a directional authenticity vote whenever calibrated probabilities are unavailable.
3. **Selective Abstention Protocol:** Implementing entropy thresholds ($\mathcal{H} \le 0.30$ bits) provides a principled statistical filter to isolate confident predictions while safely designating borderline cases as inconclusive.

---

## 5. Status Classification

In accordance with the approved status vocabulary:
- **Previous Status:** `PARTIALLY IMPLEMENTED` / `EXPLORATORY_ONLY`
- **Package C Certified Status:** **`EXPLORATORY_ONLY`**  
  *Justification:* The empirical calibration and uncertainty dynamics are comprehensively quantified on authorized data, but temperature scaling remains an offline exploratory research finding that does not modify production runtime parameters.
