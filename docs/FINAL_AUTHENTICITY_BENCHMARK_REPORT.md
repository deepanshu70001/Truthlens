# TruthLens Final Authenticity Benchmark & Partition Audit Report

**Document Version:** `1.0.0`  
**Date:** 2026-11-24  
**Evaluated Model Identity:** `truthlens/siglip-cifake-frozen-head`  
**Model Architecture:** Google SigLIP ViT-B/16 Vision Backbone (`7fd15f0689c79...`) + Frozen Linear Head  
**Checkpoint File:** `models/baselines/siglip_frozen_head.pt`  
**Checkpoint SHA-256:** `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` (Bit-exact verified)  
**Historical Evaluation Artifact:** `experiments/reports/phase3_results.json`  
**Governance Protocol:** `docs/phase2_split_protocol.md`, `docs/phase10_blocker_report.md`, and `docs/r10_data_governance_audit.md`

---

## 1. Executive Summary & Governance Disposition

Package C requires establishing rigorous, statistically defensible evidence for the TruthLens authenticity classifier without violating partition boundaries or misrepresenting historical results.

### 1.1 Partition Release & Unsealing Governance Audit
In accordance with master project directives:
> *"The current authorization covers the final authenticity benchmark workstream, but does not permit guessing which protected partition to open. If the exact final split cannot be identified from an authoritative protocol, stop before accessing it and report the required authorization. Do not use the reserve partition unless its release is explicitly authorized. Do not relabel the historical 10,385-image result as a fresh final benchmark."*

**Governance Audit Finding:**
1. **Fresh Split Unsealing Prohibited:** The sealed reserve partition (2,999 primary reserve images, 9,994 total reserve records) and the protected calibration split (13,000 images) are strictly sealed under cryptographic and protocol protection. No authorized unsealing manifest was provided for a fresh test run.
2. **Certified Benchmark Status:** In strict accordance with the approved status vocabulary, fresh independent unsealing is formally classified as:
   $$\mathbf{DATASET\_ACCESS\_BLOCKED} \quad / \quad \mathbf{EVALUATION\_PROTOCOL\_REQUIRED}$$
3. **Historical Evidence Reporting:** To provide full scientific transparency, the historical held-out test evaluation conducted in Phase 3 ($n = 10,385$ samples across Tiny-GenImage, Frontier Benchmark October 2026, and WildAIGC-Bench v3) is comprehensively re-analyzed below with complete statistical rigor and 95% bootstrap confidence intervals, and is explicitly identified as **historical evaluation evidence**.

---

## 2. Experimental Partitions & Group-Aware Leakage Control

As specified in `docs/phase2_split_protocol.md`, image evaluations enforce group-level clustering to prevent near-duplicate leakage, social media variant leakage, and shared-event reference contamination:

```
+---------------------------------------------------------------------------------------------------+
|                           HISTORICAL HELD-OUT TEST EVALUATION INVENTORY                           |
+---------------------------------------------------------------------------------------------------+
| 1. Tiny-GenImage (v1.0-tiny)     | 7,000 images (3,500 authentic, 3,500 AI) | 2,001 clusters      |
| 2. Frontier Benchmark (Oct 2026) | 2,091 images (1,351 authentic, 740 AI)   | 525 clusters        |
| 3. WildAIGC-Bench v3             | 1,294 images (449 authentic, 845 AI)     | 836 clusters        |
+----------------------------------+------------------------------------------+---------------------+
| TOTAL POOLED EVALUATION POPULATION | 10,385 images (5,300 authentic, 5,085 AI)| 3,362 clusters      |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Quantitative Evaluation Results (Historical Baseline)

### 3.1 Dataset-by-Dataset Benchmark Performance (Default Threshold $\tau = 0.50$)

| Evaluation Metric | Tiny-GenImage ($n=7,000$) | Frontier 2026-10 ($n=2,091$) | WildAIGC-Bench ($n=1,294$) | Pooled Total ($n=10,385$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 55.19% | 58.01% | 44.90% | **54.47%** |
| **Balanced Accuracy** | 55.19% | 62.28% | 42.94% | **54.86%** |
| **Precision (AI)** | 53.54% | 44.59% | 59.40% | **52.51%** |
| **Recall / TPR (AI)** | 78.49% | 76.89% | 49.35% | **73.41%** |
| **Specificity / TNR** | 31.89% | 47.67% | 36.53% | **36.30%** |
| **False Alarm Rate (FAR / FPR)** | 68.11% | 52.33% | 63.47% | **63.70%** |
| **False Rejection Rate (FRR / FNR)**| 21.51% | 23.11% | 50.65% | **26.59%** |
| **F1-Score** | 63.65% | 56.45% | 53.91% | **61.22%** |
| **ROC-AUC** | **0.6123** | **0.6501** | **0.4226** | — |
| **PR-AUC** | 0.6181 | 0.4534 | 0.6201 | — |

---

### 3.2 Detailed Confusion Matrices ($\tau = 0.50$)

| Benchmark Partition | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tiny-GenImage** | 1,116 | 2,384 | 753 | 2,747 | 7,000 |
| **Frontier 2026-10** | 644 | 707 | 171 | 569 | 2,091 |
| **WildAIGC-Bench** | 164 | 285 | 428 | 417 | 1,294 |
| **Combined Total** | **1,924** | **3,376** | **1,352** | **3,733** | **10,385** |

---

### 3.3 Group-Aware Bootstrap 95% Confidence Intervals (1,000 Iterations)

To prevent cluster-induced variance underestimation, bootstrap resampling was conducted across **independent variant family clusters** rather than individual images:

| Benchmark Partition | Metric | Bootstrap Mean | Standard Deviation ($\sigma$) | 95% Bootstrap CI [Lower, Upper] |
| :--- | :--- | :---: | :---: | :---: |
| **Tiny-GenImage** ($2,001$ clusters) | Accuracy | 0.6063 | 0.1363 | **[0.4133, 0.8200]** |
| | Balanced Accuracy | 0.6325 | 0.1115 | **[0.5291, 0.8200]** |
| | F1-Score | 0.6714 | 0.1702 | **[0.3505, 0.9011]** |
| | ROC-AUC | 0.6144 | 0.0233 | **[0.5678, 0.6558]** |
| **Frontier 2026-10** ($525$ clusters) | Accuracy | 0.5809 | 0.0207 | **[0.5387, 0.6203]** |
| | Balanced Accuracy | 0.6232 | 0.0198 | **[0.5844, 0.6594]** |
| | F1-Score | 0.5645 | 0.0257 | **[0.5118, 0.6133]** |
| | ROC-AUC | 0.6506 | 0.0239 | **[0.6008, 0.6954]** |
| **WildAIGC-Bench** ($836$ clusters) | Accuracy | 0.4491 | 0.0179 | **[0.4135, 0.4827]** |
| | Balanced Accuracy | 0.4294 | 0.0166 | **[0.3964, 0.4608]** |
| | F1-Score | 0.5390 | 0.0211 | **[0.4964, 0.5779]** |
| | ROC-AUC | 0.4224 | 0.0193 | **[0.3843, 0.4606]** |

---

## 4. Scientific Failure Mode & Generalization Analysis

### 4.1 Cross-Generator Generalization Collapse
- **Tiny-GenImage (Controlled Synthetic Baseline):** Achieves moderate discriminative ability ($\text{ROC-AUC} = 0.6123, 95\%\ \text{CI}\ [0.5678, 0.6558]$).
- **Frontier Benchmark (October 2026 Diffusion & Flow Models):** Maintains $\text{ROC-AUC} = 0.6501, 95\%\ \text{CI}\ [0.6008, 0.6954]$.
- **WildAIGC-Bench (In-The-Wild Generators & Photorealistic Subjects):** Suffers complete discriminative failure ($\text{ROC-AUC} = 0.4226, 95\%\ \text{CI}\ [0.3843, 0.4606]$). This sub-random performance indicates that the model relies heavily on high-frequency generation artifacts that invert or vanish across diverse unconstrained generators.

### 4.2 Severe False Alarm Rate (FAR = 63.70%)
At the default classification threshold ($\tau = 0.50$), the model misclassifies **63.70% of authentic images as AI-generated** (3,376 false alarms out of 5,300 authentic images).
- **Scientific Implication:** Deploying raw model scores with $\tau = 0.50$ produces unacceptable false accusations of synthetic manipulation.
- **Production Defense:** This empirical vulnerability confirms the strict necessity of TruthLens's production **conservative evidence fusion policy (`e11.conservative.v1`)**, which:
  1. Gates the model's directional vote when calibration is absent.
  2. Enforces selective prediction and non-voting status.
  3. Displays explicit caveats on raw authenticity scores in the frontend UI.

---

## 5. Status Summary

- **Fresh Final Benchmark Split:** **`DATASET_ACCESS_BLOCKED`** / **`EVALUATION_PROTOCOL_REQUIRED`**  
  *Justification:* Sealed reserve and calibration partitions are protected; no authorized release protocol was issued.
- **Historical Authenticity Classifier Capability:** **`EVALUATED_WITH_LIMITATIONS`**  
  *Justification:* Fully quantified across 10,385 historical held-out test images with cluster-bootstrap confidence intervals; performance exhibits elevated false alarms (FAR 63.70%) and cross-generator vulnerability (WildAIGC ROC-AUC 0.4226).
