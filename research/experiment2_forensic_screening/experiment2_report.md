# TruthLens — Experiment 2: Univariate Forensic Feature Screening Report
## Protocol Version 1.2.1 Pre-Registered Research Evaluation

**Date:** 2026-11-23  
**Execution Status:** Completed & Audited  
**Authorized Partition:** `experiments/manifests/phase4_training_manifest.csv` ($N = 6,995$)  
**Baseline Model:** `truthlens/siglip-cifake-frozen-head` (`models/baselines/siglip_frozen_head.pt`)  
**Output Directory:** `research/experiment2_forensic_screening/`  

---

### 1. Executive Summary

Experiment 2 evaluated whether any of TruthLens's six classical forensic indicators provides statistically significant complementary predictive information beyond the frozen primary SigLIP authenticity detector.

The experiment was conducted strictly on the **6,995 approved research training samples** using **stratified 5-fold group-aware nested cross-validation**. No diagnostic validation samples, calibration samples, or held-out test benchmarks were accessed.

### Primary Results Summary:

| Forensic Probe | Applicable Subgroup ($N$) | Coverage | Class Overlap ($N_\text{real} / N_\text{ai}$) | Baseline Log-Loss ($M0$) | Augmented Log-Loss ($M1$) | $\Delta \text{Log-Loss}$ ($M0 - M1$) | 95% Bootstrap CI | Adjusted $q$-value | Statistical Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Error Level Analysis (ELA)** | 3,515 | 50.25% | 3,515 / 0 | N/A | N/A | N/A | N/A | N/A | **STATISTICALLY NON-IDENTIFIABLE** |
| **JPEG Block Artifacts** | 3,515 | 50.25% | 3,515 / 0 | N/A | N/A | N/A | N/A | N/A | **STATISTICALLY NON-IDENTIFIABLE** |
| **Noise Consistency** | 6,995 | 100.0% | 3,515 / 3,480 | 0.757890 | 0.759610 | -0.001721 | [-0.003023, 6.8e-05] | 0.292000 | **STATISTICALLY_INCONCLUSIVE** |
| **Edge Consistency** | 6,995 | 100.0% | 3,515 / 3,480 | 0.757890 | 0.765480 | -0.007590 | [-0.014074, 0.00233] | 0.412000 | **STATISTICALLY_INCONCLUSIVE** |
| **Texture Consistency (Gabor)** | 6,995 | 100.0% | 3,515 / 3,480 | 0.757890 | 0.755603 | +0.002287 | [-0.004348, 0.012509] | 0.498000 | **STATISTICALLY_INCONCLUSIVE** |
| **Resampling Clues** | 6936 | 99.16% | 3512 / 3424 | 0.756050 | 0.767281 | -0.011231 | [-0.023516, 0.003665] | 0.498000 | **STATISTICALLY_INCONCLUSIVE** |

---

### 2. Methodological Invariants Verified

1. **Manifest Integrity:** `experiments/manifests/phase4_training_manifest.csv` verified with SHA-256 `22ea2825c7ab75ef110c7a6e2a7c67e69459a3afd3ed6159adbcbed7abc168ed`. Exact 6,995 rows (3,515 Real / 3,480 AI).
2. **Model Immutability:** `models/baselines/siglip_frozen_head.pt` verified with SHA-256 `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` before and after execution (`Diff == 0.0`).
3. **Strict Partition Blindness:** Zero samples from diagnostic validation ($N=6,436$), calibration ($N=13,000$), held-out benchmark ($N=10,385$), or protected reserve ($N=2,999$) were accessed.
4. **Leakage-Free Nested Cross-Validation:** All 903 canonical `group_id` clusters were partitioned across 5 outer folds. Zero groups cross folds. All scaling and hyperparameter selection ($L_2$ penalty $C$) were executed strictly on inner training folds.
5. **No Imputation Safeguard:** Native-JPEG-specific probes (ELA, JPEG artifacts) and abstained resampling samples were strictly kept null; no zeros or dummy features were imputed.
