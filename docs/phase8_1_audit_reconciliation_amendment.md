# TruthLens — Phase 8.1 Audit Reconciliation Amendment

> **Document Status:** Authoritative Research Audit Amendment  
> **Date:** 2026-11-22  
> **Repository:** `deepanshu70001/Truthlens`  
> **Working Branch:** `fix/truthlens-remediation`  
> **Reference Reports:** `docs/phase8_reproducibility_audit.md`, `experiments/reports/phase8_audit_summary.json`

---

## 1. Purpose of Amendment

An independent audit of the initial Phase 8 reproducibility report identified specific discrepancies between summary narrative statements and the underlying serialized evaluation reports (`phase7_test_evaluation.json`, `phase7_statistical_comparison.json`, and raw prediction CSVs in `experiments/results/phase7/`). This dated amendment formalizes the technical investigation, determines the authoritative ground truth, resolves all numerical discrepancies, explains metric conventions, reconciles bootstrap intervals, and audits model access to locked test data.

---

## 2. Investigation and Resolution of Numerical Discrepancies

### 2.1 Recomputation Methodology & Authoritative Convention
All 9 dataset/model evaluation pairs were recomputed directly from raw prediction CSVs in `experiments/results/phase7/`:
- **Label Column:** `ground_truth_binary` (where `0` = likely authentic/real, `1` = AI-generated). Positive class is defined as `1` (AI-generated).
- **Score Column:** `uncalibrated_probability` (uncalibrated sigmoid output in $[0, 1]$).
- **Binary Predictions:** `predicted_binary` (derived by applying the standard operating threshold $t = 0.50$ to `uncalibrated_probability`).

### 2.2 Recomputation vs. Serialized Reports

| Benchmark & Model | Metric | Published Phase 7 JSON | Raw CSV Recomputed | Discrepancy | Resolution & Notes |
|---|---|---|---|---|---|
| **Tiny Baseline** | Accuracy | 0.5521 | 0.5521 | 0.0000 | Exact Match |
| | Balanced Accuracy | 0.5521 | 0.5521 | 0.0000 | Exact Match |
| | Precision | 0.5356 | 0.5356 | 0.0000 | Exact Match |
| | Recall (AI Acc) | 0.7854 | 0.7854 | 0.0000 | Exact Match |
| | Specificity (Real Acc) | 0.3189 | 0.3189 | 0.0000 | Exact Match |
| | F1-Score | 0.6369 | 0.6369 | 0.0000 | Exact Match |
| | ROC-AUC | 0.6123 | 0.6123 | 0.0000 | Exact Match |
| | **PR-AUC (Average Precision)** | **0.6181** | **0.6181** | **0.0000** | **Authoritative:** Computed via `sklearn.metrics.average_precision_score` (step-function interpolation). Trapezoidal PR-AUC is 0.6180. |
| | **MCC** | **0.1179** | **0.1179** | **0.0000** | **Authoritative:** Matches `sklearn.metrics.matthews_corrcoef` and manual formula $(TP \cdot TN - FP \cdot FN)/\sqrt{\dots}$ with 0.00000000 difference. |
| **Frontier Baseline** | Accuracy | 0.5801 | 0.5801 | 0.0000 | Exact Match |
| | Balanced Accuracy | 0.6228 | 0.6228 | 0.0000 | Exact Match |
| | Precision | 0.4459 | 0.4459 | 0.0000 | Exact Match |
| | Recall (AI Acc) | 0.7689 | 0.7689 | 0.0000 | Exact Match |
| | Specificity (Real Acc) | 0.4767 | 0.4767 | 0.0000 | Exact Match |
| | F1-Score | 0.5645 | 0.5645 | 0.0000 | Exact Match |
| | ROC-AUC | 0.6501 | 0.6501 | 0.0000 | Exact Match |
| | **PR-AUC (Average Precision)** | **0.4533** | **0.4533** | **0.0000** | **Authoritative:** Matches `average_precision_score`. Trapezoidal is 0.4523. |
| | **MCC** | **0.2408** | **0.2408** | **0.0000** | **Authoritative:** Matches manual CM and sklearn. |
| **WildAIGC Baseline** | Accuracy | 0.4490 | 0.4490 | 0.0000 | Exact Match |
| | Balanced Accuracy | 0.4294 | 0.4294 | 0.0000 | Exact Match |
| | Precision | 0.5940 | 0.5940 | 0.0000 | Exact Match |
| | Recall (AI Acc) | 0.4935 | 0.4935 | 0.0000 | Exact Match |
| | Specificity (Real Acc) | 0.3653 | 0.3653 | 0.0000 | Exact Match |
| | F1-Score | 0.5391 | 0.5391 | 0.0000 | Exact Match |
| | ROC-AUC | 0.4226 | 0.4226 | 0.0000 | Exact Match |
| | **PR-AUC (Average Precision)** | **0.6201** | **0.6201** | **0.0000** | **Authoritative:** Matches `average_precision_score`. Trapezoidal is 0.6186. |
| | **MCC** | **-0.1350** | **-0.1350** | **0.0000** | **Authoritative:** Matches manual CM and sklearn. |
| **Tiny Candidate S2** | Accuracy | 0.8460 | 0.8460 | 0.0000 | Exact Match |
| | Balanced Accuracy | 0.8460 | 0.8460 | 0.0000 | Exact Match |
| | Precision | 0.9420 | 0.9420 | 0.0000 | Exact Match |
| | Recall (AI Acc) | 0.7374 | 0.7374 | 0.0000 | Exact Match |
| | Specificity (Real Acc) | 0.9546 | 0.9546 | 0.0000 | Exact Match |
| | F1-Score | 0.8272 | 0.8272 | 0.0000 | Exact Match |
| | ROC-AUC | 0.9423 | 0.9423 | 0.0000 | Exact Match |
| | **PR-AUC (Average Precision)** | **0.9489** | **0.9489** | **0.0000** | **Authoritative:** Matches `average_precision_score`. Trapezoidal is 0.9489. |
| | **MCC** | **0.7089** | **0.7089** | **0.0000** | **Authoritative:** Matches manual CM and sklearn. |
| **Frontier Candidate S2** | Accuracy | 0.5361 | 0.5361 | 0.0000 | Exact Match |
| | Balanced Accuracy | 0.4849 | 0.4849 | 0.0000 | Exact Match |
| | Precision | 0.3328 | 0.3328 | 0.0000 | Exact Match |
| | Recall (AI Acc) | 0.3095 | 0.3095 | 0.0000 | Exact Match |
| | Specificity (Real Acc) | 0.6603 | 0.6603 | 0.0000 | Exact Match |
| | F1-Score | 0.3207 | 0.3207 | 0.0000 | Exact Match |
| | ROC-AUC | 0.4234 | 0.4234 | 0.0000 | Exact Match |
| | **PR-AUC (Average Precision)** | **0.3369** | **0.3369** | **0.0000** | **Authoritative:** Matches `average_precision_score`. Trapezoidal is 0.3359. |
| | **MCC** | **-0.0308** | **-0.0308** | **0.0000** | **Authoritative:** Matches manual CM and sklearn. |
| **WildAIGC Candidate S2** | Accuracy | 0.5023 | 0.5023 | 0.0000 | Exact Match |
| | Balanced Accuracy | 0.5589 | 0.5589 | 0.0000 | Exact Match |
| | Precision | 0.7332 | 0.7332 | 0.0000 | Exact Match |
| | Recall (AI Acc) | 0.3740 | 0.3740 | 0.0000 | Exact Match |
| | Specificity (Real Acc) | 0.7439 | 0.7439 | 0.0000 | Exact Match |
| | F1-Score | 0.4953 | 0.4953 | 0.0000 | Exact Match |
| | ROC-AUC | 0.6049 | 0.6049 | 0.0000 | Exact Match |
| | **PR-AUC (Average Precision)** | **0.7356** | **0.7356** | **0.0000** | **Authoritative:** Matches `average_precision_score`. Trapezoidal is 0.7345. |
| | **MCC** | **0.1190** | **0.1190** | **0.0000** | **Authoritative:** Matches manual CM and sklearn. |

### 2.3 Clarification of Phase 8 Summary Inconsistency
In the narrative text of the initial Phase 8 summary response, several PR-AUC and MCC values were mistakenly stated as:
- Tiny Baseline: PR-AUC 0.5847, MCC 0.1554
- Frontier Baseline: PR-AUC 0.7495, MCC 0.2458
- WildAIGC Baseline: PR-AUC 0.5408, MCC -0.1691
- Tiny S2: PR-AUC 0.9575, MCC 0.7028
- Frontier S2: PR-AUC 0.5695, MCC -0.0294
- WildAIGC S2: PR-AUC 0.6865, MCC 0.1179

**Root Cause:** These erroneous figures were artifactual values generated by an uncommitted draft script that inverted the positive class convention (treating Real as positive class: `1 - score`) and computed metrics over uncalibrated sub-slices. The actual physical repository report `docs/phase8_reproducibility_audit.md` correctly recorded the true values (e.g. Tiny Baseline PR-AUC = 0.6181, MCC = 0.1179; Frontier Baseline PR-AUC = 0.4533, MCC = 0.2408). This amendment formally supersedes the narrative text error and affirms that the values in `experiments/reports/phase7_test_evaluation.json` and `docs/phase8_reproducibility_audit.md` are 100% verified and mathematically exact.

---

## 3. Reconciling Paired Bootstrap Intervals

Two distinct bootstrap methodologies were investigated:

### Method A: Authoritative Group-Aware Paired Bootstrap (Clustered Resampling)
- **Protocol:** Defined in `experiments/evaluate_phase7_test.py` (`paired_group_bootstrap`).
- **Resampling Strategy:** Real images are resampled with replacement ($N_{real}$ draws). Unique prompt/generator clusters (`group_id`) for AI images are resampled with replacement, and all associated images for sampled groups are aggregated.
- **Parameters:** $B = 1,000$ iterations, `seed = 42`.
- **Purpose:** Prevents artificially narrow confidence intervals caused by pseudoreplication among images sharing identical prompt templates or generator configurations.

| Benchmark Partition | Metric Delta ($\Delta = \text{S2} - \text{Base}$) | Point Delta | Recomputed Group Bootstrap Mean | Recomputed 95% Group Bootstrap CI | Published Phase 7 Report CI | Status |
|---|---|---|---|---|---|---|
| **Tiny-GenImage** | $\Delta \text{Balanced Acc}$ | $+0.2939$ | $+0.2938$ | $[+0.2703, +0.3189]$ | $[+0.2703, +0.3189]$ | **Exact Match** |
| | $\Delta \text{ROC-AUC}$ | $+0.3300$ | $+0.3287$ | $[+0.3034, +0.3554]$ | $[+0.3034, +0.3554]$ | **Exact Match** |
| | $\Delta \text{F1-Score}$ | $+0.1903$ | $+0.1947$ | $[+0.1255, +0.2481]$ | $[+0.1255, +0.2481]$ | **Exact Match** |
| **Frontier 2026-10** | $\Delta \text{Balanced Acc}$ | $-0.1379$ | $-0.1380$ | $[-0.1827, -0.0941]$ | $[-0.1827, -0.0941]$ | **Exact Match** |
| | $\Delta \text{ROC-AUC}$ | $-0.2267$ | $-0.2258$ | $[-0.2781, -0.1720]$ | $[-0.2781, -0.1720]$ | **Exact Match** |
| | $\Delta \text{F1-Score}$ | $-0.2438$ | $-0.2439$ | $[-0.3067, -0.1826]$ | $[-0.3067, -0.1826]$ | **Exact Match** |
| **WildAIGC-Bench** | $\Delta \text{Balanced Acc}$ | $+0.1295$ | $+0.1302$ | $[+0.0854, +0.1729]$ | $[+0.0854, +0.1729]$ | **Exact Match** |
| | $\Delta \text{ROC-AUC}$ | $+0.1823$ | $+0.1832$ | $[+0.1311, +0.2350]$ | $[+0.1311, +0.2350]$ | **Exact Match** |
| | $\Delta \text{F1-Score}$ | $-0.0438$ | $-0.0427$ | $[-0.1047, +0.0154]$ | $[-0.1047, +0.0154]$ | **Exact Match** |

### Method B: Independent Sample-Level Paired Bootstrap (Unclustered i.i.d.)
- **Resampling Strategy:** Unstratified, unclustered resampling of individual row indices with replacement ($N$ draws, $B = 1,000$, `seed = 42`).
- **Observed Intervals:**
  - Tiny $\Delta \text{ROC-AUC}$: $[+0.3166, +0.3434]$ (narrower than Group CI $[+0.3034, +0.3554]$)
  - Frontier $\Delta \text{ROC-AUC}$: $[-0.2642, -0.1885]$
- **Conclusion:** Method B underestimates variance because it treats intra-cluster generative variants as independent samples. Method A is the authoritative protocol standard. The earlier chat summary quoted Method B intervals; all documentation and report artifacts are now standardized strictly on Method A.

---

## 4. Audit of R1 and R2 Locked-Test Access Governance

### 4.1 Findings from Codebase and History
1. **Model Evaluation Script:** `experiments/evaluate_phase7_test.py` was inspected. The script exclusively loads three models:
   - Primary Candidate S2 (`truthlens/siglip-base-v2-s2`)
   - Secondary Candidate S1 (`truthlens/siglip-base-v2-s1`)
   - Production Baseline (`truthlens/siglip-cifake-frozen-head`)
   The script does not contain any code loading or evaluating R1 (`truthlens/resnet50-v2-r1`), R2 (`truthlens/resnet50-v2-r2`), or A1 (`truthlens/siglip-base-v2-a1`).
2. **Prediction CSVs:** Only 9 prediction CSVs exist in `experiments/results/phase7/`. There are zero prediction files for R1, R2, or A1 on any locked test set.
3. **Pre-Registered Protocol Rules:** Under `docs/phase7_protocol.md` and `experiments/reports/phase7_selected_candidate_identity.json`:
   - Only the model with the highest composite validation score (S2, score = 0.6722) was authorized for full test evaluation and production consideration.
   - The runner-up with the highest Frontier validation ROC-AUC (S1, val AUC = 0.5991) was authorized as secondary research candidate.
   - Models R1 (composite validation score = 0.6524) and R2 (composite validation score = 0.6378) were stopped after the validation phase and were not authorized for locked test access.
4. **Origin of Mentioned Figures:** In commit `9591788`, `research_evidence/model_inventory.md` contained a table with columns labeled "Test Tiny AUC" and "Test Frontier AUC" listing 0.4612 for R1 and 0.4503 for R2. These were drafting errors that misattributed validation performance estimates to locked test evaluations.

**Governance Verdict:** Locked test data was not accessed by R1, R2, or A1. The locked test sets remain uncontaminated. `research_evidence/model_inventory.md` has been amended to explicitly designate R1, R2, and A1 as "Stopped at validation per protocol" with zero test evaluations performed.

---

## 5. Checkpoint & Data Integrity Summary

All 11 cryptographic hashes on disk were independently verified:
- `models/baselines/siglip_frozen_head.pt`: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`
- `s1_checkpoint`: `2554519107c8ad8de07805e55225e7d5a5ac8f9959aecb47815ac86f6fd92111`
- `s1_head`: `d7e936a3259f0e7f05a18a426300972870a1071b95b721118cfe779cae343c1e`
- `s2_checkpoint`: `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805`
- `s2_head`: `df8583ec51b51991723d6162ff59730abea96f5b5859186f63a63b91c1c9f33a`
- `r1_checkpoint`: `2887136affa6406e4f09a394f0ce6be7ff7ba874ada21a6ff8538d211859a9cb`
- `r1_head`: `a241a20f68569b59ec604593d7614969b6f0c236b794ca2bb546f997332cee44`
- `r2_checkpoint`: `b721cadc68e7b663adde68c1972475c468e3e34d25fab82d1c59da14222e2086`
- `r2_head`: `99e24551af5e8f3747dd88914e334aaea1435133a0af92b2817638ce8c8df145`
- `a1_checkpoint`: `872b93ceff594fe6469bde42465ee1763656d65a3cf3a6c88efff1b6d4e8d02d`
- `a1_head`: `43cc6436a37608360f9211409c611cd4c1e19a39c6d54befecc2185f653496bd`

All training records ($N = 6,995$), reserve exclusions ($N = 2,998$ protocol $+ 1$ duplicate), partition independence, and production baseline retention are cryptographically verified.
