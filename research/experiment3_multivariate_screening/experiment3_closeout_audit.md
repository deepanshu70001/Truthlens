# TruthLens — Experiment 3 Closeout and Governance Audit

**Document Identity:** `research/experiment3_multivariate_screening/experiment3_closeout_audit.md`<br>
**Repository:** `https://github.com/deepanshu70001/Truthlens`<br>
**Research Branch:** `research/experiment3-multivariate-screening`<br>
**Baseline Remediation Merge-Base:** `fix/truthlens-remediation` at `f4baf15e67d2e837a17df7da0383cd0ede1a8b0a`<br>
**Implementation Commit:** `8981f8b052cb08e2cffb6c130b5b2fdd44f8290b`<br>
**Results-Artifact Commit:** `c02c1d7218b59e615c4e79409bd936bafb25d430`<br>
**Research Protocol:** [protocol.md](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/protocol.md) (Exploratory Protocol v1.0.0)<br>
**Date:** 2026-11-24<br>
**Study Status Classification:** `EXPLORATORY_POST_HOC_AFTER_EXPERIMENT2_GATE_FAILURE`

---

## Executive Summary

Experiment 3 (Multivariate Forensic Complementarity Screening) was executed under explicit user authorization following the closeout of Experiment 2. Experiment 2 established that univariate classical forensic indicators provided no statistically significant incremental predictive power beyond frozen SigLIP (all Benjamini-Hochberg FDR $q \ge 0.292$), and explicitly recorded that Experiment 3 was **not recommended**. Consequently, this investigation is strictly classified as an **exploratory, post hoc analysis** rather than a confirmatory preregistered study or independent replication.

The investigation was conducted strictly on the authorized Phase 4 training partition ([phase4_training_manifest.csv](file:///D:/Projects/Truthlens/experiments/manifests/phase4_training_manifest.csv), $N=6,995$) using immutable group-aware 5-fold cross-validation inherited directly from Experiment 2. All 18 research deliverables were generated, validated, and verified against [reproducibility_manifest.json](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/reproducibility_manifest.json) with exact bit-for-bit SHA-256 and byte-size integrity.

### Primary Scientific Conclusions

1. **Failure to Reject Null Hypothesis Across Candidate Models:** Across 5-fold group-aware nested cross-validation on the common complete-case cohort ($N=6936$), none of the four candidate forensic fusion models ($M_1, M_2, M_3, M_4$) achieved a statistically significant positive incremental predictive log-score ($\Delta\mathcal{L} > 0$) after Benjamini-Hochberg False Discovery Rate (FDR) control at $\alpha = 0.05$ (all adjusted $q \ge 0.336183$).
2. **Systematic Out-of-Fold Log-Loss Degradation:** Across all evaluated configurations, point estimates for out-of-fold log-loss improvement over the SigLIP-logit-only logistic reference model ($M_0$) were negative:
   - **$M_1$ (Noise + Edge):** $\Delta\mathcal{L} = -0.008603$, 95% CI $[-0.014552, +0.000300]$, permutation $p = 0.084046$, adjusted $q = 0.336183$.
   - **$M_2$ (Texture + Resampling):** $\Delta\mathcal{L} = -0.011358$, 95% CI $[-0.031312, +0.012351]$, permutation $p = 0.995000$, adjusted $q = 0.995000$.
   - **$M_3$ (All Four Continuous Probes):** $\Delta\mathcal{L} = -0.015188$, 95% CI $[-0.035340, +0.009455]$, permutation $p = 0.838558$, adjusted $q = 0.995000$.
   - **$M_4$ (Correlation-Adjusted Fusion):** $\Delta\mathcal{L} = -0.015188$, 95% CI $[-0.035340, +0.009455]$, permutation $p = 0.838558$, adjusted $q = 0.995000$.
   Adding classical forensic features to the SigLIP reference model degraded pooled out-of-fold log loss from 0.755336 ($M_0$) to 0.763939 ($M_1$), 0.766694 ($M_2$), and 0.770524 ($M_3$/$M_4$).
3. **M3 vs M4 Empirical Identity on Training Folds:** Complete-linkage Spearman correlation clustering at threshold $|r_s| \ge 0.85$ revealed that across all 5 outer training folds, no pair of continuous forensic features exceeded the threshold. Consequently, complete-linkage clustering produced four single-element clusters (`[[noise], [edge], [texture], [resampling]]`), causing $M_4$'s correlation adjustment to reduce to standard z-score standardization identical to $M_3$ on this dataset partition.
4. **Severe Native Format Confounding:** In the authorized Phase 4 training partition, 100% of the 3,515 authentic images are native JPEGs, whereas 100% of the 3,480 AI images are native PNGs. Error Level Analysis (ELA) and JPEG block boundary artifacts were excluded from multivariate screening because Experiment 2 established that they have zero AI-class overlap in their applicable subgroups.
5. **Generator Diversity Imbalance:** The 3,480 AI images are dominated by specific generator families (GLIDE, etc.), limiting independent generator diversity and precluding universal claims.
6. **Reaffirmation of Production Fusion Architecture:** Directional fusion of classical forensic indicators is definitively **REJECTED**. Classical forensic indicators must remain strictly non-directional, contextual heuristics in `backend/app/services/fusion/engine.py`.
7. **Research Gate Enforcements:** Experiment 4 (final evaluation) remains strictly **UNAUTHORIZED**. The 10,385-sample held-out benchmark and the 2,999-sample retained reserve partition remain locked and unaccessed.
8. **Production Safety & Invariance:** Production SigLIP baseline head weights ([siglip_frozen_head.pt](file:///D:/Projects/Truthlens/models/baselines/siglip_frozen_head.pt)) remain bit-exact identical to the approved hash (`e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`). All 17 regression tests in `test_experiment3_multivariate_screening.py` and 622 backend tests pass.

---

## Part A — Independent Audit of Experiment 3 Deliverables

### 1. Artifact Verification against Reproducibility Manifest

All 18 experiment deliverables located in [research/experiment3_multivariate_screening/](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/) were verified against [reproducibility_manifest.json](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/reproducibility_manifest.json):

| # | Artifact File | Size (Bytes) | SHA-256 Digest | Status |
|---|---|---:|---|:---:|
| 1 | [`bootstrap_confidence_intervals.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/bootstrap_confidence_intervals.json) | 1,393 | `dca889ad5aeaa74c7b93fbe273cdc72366ac2e7765c008577e1de6f156d6cf32` | EXACT MATCH |
| 2 | [`correlation_clusters_by_outer_fold.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/correlation_clusters_by_outer_fold.json) | 5,687 | `c9abad60546557c0af69bd09fc92d93f5ae1b9435ae562352744d1ec210242d1` | EXACT MATCH |
| 3 | [`coverage.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/coverage.json) | 486 | `3b650f1fd9f76f4b73acff0fd1359ba72f138c5d3e8d734ce570a87920440ec3` | EXACT MATCH |
| 4 | [`dataset_integrity.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/dataset_integrity.json) | 882 | `617b36b44c9ba3142e59762e8dec179081fa8aee421f4498835f39059b9cd843` | EXACT MATCH |
| 5 | [`environment_versions.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/environment_versions.json) | 202 | `ef40abd5549152be7b364b04bde78e0731d3881f1c6675e49cd454b2088dfde7` | EXACT MATCH |
| 6 | [`execution.log`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/execution.log) | 281 | `c9931db408065f8200446c0b13125fbe3a3ffc39d16e86d0e3add8f1dfee7f36` | EXACT MATCH |
| 7 | [`experiment3_config.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/experiment3_config.json) | 2,055 | `f70e35d22ff02763d47679fdb5ba138d2e13e7b7ba949af3490d26453b21608b` | EXACT MATCH |
| 8 | [`experiment3_report.md`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/experiment3_report.md) | 5,369 | `7998c6e9bb88e032108f010808b256b9f774e1e0a90e9be174faf335f30fa12e` | EXACT MATCH |
| 9 | [`experiment3_results.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/experiment3_results.json) | 32,780 | `8e63af4d4723432a05714e040043a4f48c41d6024329821641e94d464b060189` | EXACT MATCH |
| 10 | [`fold_summary.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/fold_summary.json) | 2,943 | `f3c17d6d9811ea1c9b41eb27bf3b58123c5513fcc582c192ec81e4cde81d245e` | EXACT MATCH |
| 11 | [`input_integrity.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/input_integrity.json) | 3,678 | `0b67f3faecc7b44d9d2dfc5de1f78be712536881d7d3e895b856017e1e787c24` | EXACT MATCH |
| 12 | [`model_fit_details.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/model_fit_details.json) | 23,759 | `c9c77268f55588a1c9c7eb1b64ce171ed64825cf1c5812556ca12295fe52bc4e` | EXACT MATCH |
| 13 | [`multiple_testing_results.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/multiple_testing_results.json) | 815 | `6eb3ace624073d1fa0c36aba0964b6076d25f48acf34b2d5a0a0ce65fd2ab40e` | EXACT MATCH |
| 14 | [`oof_predictions.csv`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/oof_predictions.csv) | 1,262,153 | `fda476dba49d54142a5123fa4f13fc316fd06a04eee2a5d879ac57d3b7258fca` | EXACT MATCH |
| 15 | [`paired_metrics.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/paired_metrics.json) | 20,566 | `ab9251b7d4bc285157e0a2b377bc0ee81c6b10c9ce2d72042b16f750ce1fbb56` | EXACT MATCH |
| 16 | [`permutation_results.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/permutation_results.json) | 1,288 | `f097f6d83281a56073b8e4daa32d123442115718b98932de212158af7318d317` | EXACT MATCH |
| 17 | [`protocol.md`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/protocol.md) | 3,626 | `1d35b1228502c038dec7afded0bd4e2f869243a03da99ad77875fe3caa9537b7` | EXACT MATCH |
| 18 | [`reproducibility_manifest.json`](file:///D:/Projects/Truthlens/research/experiment3_multivariate_screening/reproducibility_manifest.json) | 2,873 | Self-referential index | VERIFIED |

---

### 2. Cohort Definition, Outer-Fold Imbalance, and Data Isolation

- **Authorized Partition:** `experiments/manifests/phase4_training_manifest.csv` ($N=6,995$, SHA-256 `22ea2825c7ab75ef110c7a6e2a7c67e69459a3afd3ed6159adbcbed7abc168ed`).
- **Complete-Case Cohort:** $N=6936$ samples (99.16% coverage). Exactly 59 samples were excluded due to probe non-success or non-finite values (all due to `resampling_clues_outcome_not_applicable;resampling_clues_score_missing_or_nonfinite`; zero imputation):
  - **Authentic (Real):** 3,512 samples (50.63%)
  - **AI-Generated:** 3,424 samples (49.37%)
  - **Unique Source Groups:** 903 groups
- **Severe Outer-Fold and Group-Size Imbalance:**
  The immutable 5 outer folds inherited from Experiment 2 exhibit substantial structural imbalance in cluster size and class base rates:

| Partition | Total N | Real N (Rate) | AI N (Rate) | Source Groups | Avg N / Group |
|---|---:|---:|---:|---:|---:|
| Outer Fold 1 | 2703 | 703 (26.0%) | 2000 (74.0%) | 51 | 53.0 |
| Outer Fold 2 | 1073 | 703 (65.5%) | 370 (34.5%) | 212 | 5.1 |
| Outer Fold 3 | 1073 | 703 (65.5%) | 370 (34.5%) | 213 | 5.0 |
| Outer Fold 4 | 1073 | 703 (65.5%) | 370 (34.5%) | 213 | 5.0 |
| Outer Fold 5 | 1073 | 703 (65.5%) | 370 (34.5%) | 214 | 5.0 |

*Methodological Impact:*
Outer Fold 1 contains 2,703 samples across only 51 source groups (average ~53.0 samples/group), with a strong AI majority (73.99%). In contrast, Outer Folds 2–5 each contain exactly 1,073 samples across 212–214 source groups (average ~5.0 samples/group) with an authentic majority (65.52%). Because logistic regression intercepts naturally shift with local base rates, pooled cross-fold ROC-AUC (0.440684 for $M_0$) is subject to rank-ordering distortions across folds. Therefore, stratified macro outer-fold ROC-AUC (0.590587 for $M_0$) is reported as the primary discrimination metric.

- **Protected Partitions Unaccessed:** Diagnostic validation split, calibration split ($N=13,000$), held-out benchmark ($N=10,385$), and protected reserve partition ($N=2,999$) were never opened. Source image files were not read; execution operated exclusively on pre-extracted feature tables and metadata.

---

### 3. Primary Multivariate Screening Results

Primary endpoint: $\Delta\mathcal{L} = \text{log-loss}(M_0) - \text{log-loss}(M_j)$, where positive values denote improvement over the SigLIP-logit-only logistic reference model ($M_0$):

| Model Candidate | $\Delta\mathcal{L}$ (Log-Loss Gain) | 95% Group-Bootstrap CI | Permutation $p$ | BH FDR $q$-value | Decision Verdict |
|---|---:|---:|---:|---:|:---:|
| **M1_noise_edge** | -0.008603 | [-0.014552, +0.000300] | 0.084046 | 0.336183 | `EXPLORATORY_INCONCLUSIVE` |
| **M2_texture_resampling** | -0.011358 | [-0.031312, +0.012351] | 0.995000 | 0.995000 | `EXPLORATORY_INCONCLUSIVE` |
| **M3_all_four** | -0.015188 | [-0.035340, +0.009455] | 0.838558 | 0.995000 | `EXPLORATORY_INCONCLUSIVE` |
| **M4_correlation_adjusted** | -0.015188 | [-0.035340, +0.009455] | 0.838558 | 0.995000 | `EXPLORATORY_INCONCLUSIVE` |

*Statistical Parameters:*
- **Reference Model ($M_0$):** Univariate logistic regression fit solely on the scalar SigLIP AI logit $Z_{\text{SigLIP}}$.
- **Bootstrap:** 5,000 paired source-group cluster replicates sampled with replacement within fixed outer folds (seed `314859`).
- **Permutation:** 20,000 two-sided cluster sign-flip permutations within fixed outer folds (seed `315059`).
- **Multiplicity:** Benjamini-Hochberg procedure controlling FDR at $\alpha = 0.05$ across all 4 comparisons.
- **Decision Rule:** Supported improvement requires adjusted $q < 0.05$ and 95% CI lower bound $> 0$. Supported detriment requires adjusted $q < 0.05$ and 95% CI upper bound $< 0$. All other outcomes are exploratory-inconclusive.

---

### 4. Secondary Calibration and Discrimination Metrics

| Metric | $M_0$ (SigLIP Ref) | $M_1$ (Noise+Edge) | $M_2$ (Texture+Resamp) | $M_3$ (All Four) | $M_4$ (Corr-Adjusted) |
|---|---:|---:|---:|---:|---:|
| **Pooled Log-Loss** | 0.755336 | 0.763939 | 0.766694 | 0.770524 | 0.770524 |
| **Pooled Brier Score** | 0.279680 | 0.283505 | 0.282767 | 0.284742 | 0.284742 |
| **Macro Fold ROC-AUC** | 0.590587 | 0.572129 | 0.589788 | 0.579066 | 0.579066 |
| **Macro Fold PR-AUC** | 0.505718 | 0.490309 | 0.506612 | 0.499359 | 0.499359 |
| **Macro Fold Balanced Acc** | 0.529410 | 0.528272 | 0.554586 | 0.548558 | 0.548558 |
| **Descriptive Pooled ROC-AUC** | 0.440684 | 0.412670 | 0.442556 | 0.436316 | 0.436316 |

### Non-Monotonicity of Secondary Metrics Analysis

Contrary to premature assertions of uniform monotonic degradation across all metrics:
1. **Pooled Log-Loss degrades monotonically:** Log loss increases steadily from 0.755336 ($M_0$) to 0.763939 ($M_1$), 0.766694 ($M_2$), and 0.770524 ($M_3$/$M_4$).
2. **Secondary ranking and classification metrics exhibit non-monotonic behavior:**
   - **Brier Score:** $M_2$ (0.282767) is slightly lower (better calibrated) than $M_1$ (0.283505), though both remain worse than $M_0$ (0.279680).
   - **Macro Fold ROC-AUC:** $M_2$ (0.589788) outperforms both $M_1$ (0.572129) and $M_3$ (0.579066), remaining close to $M_0$ (0.590587).
   - **Macro Fold PR-AUC:** $M_2$ (0.506612) is slightly higher than baseline $M_0$ (0.505718) and $M_1$ (0.490309).
   - **Macro Fold Balanced Accuracy:** $M_2$ (0.554586) and $M_3$/$M_4$ (0.548558) achieve higher balanced accuracy than $M_0$ (0.529410) due to threshold shifts at $p=0.5$.

*Statistical Summary:* While localized threshold-dependent metrics (such as balanced accuracy at 0.5) fluctuate and do not degrade monotonically, probabilistic log-loss calibration systematically deteriorates. None of the positive secondary shifts achieve statistical significance under multiplicity control (all BH $q \ge 0.336183$, all paired bootstrap intervals span zero).

---

## Part B — Governance, Architectural Disposition, and Gate Enforcements

### 1. Architectural Disposition: Directional Forensic Fusion Rejected

The empirical findings of Experiment 3 corroborate and finalize the negative screening results of Experiment 2:
1. Classical forensic indicators fail to provide statistically defensible complementary predictive information beyond the SigLIP reference model in out-of-fold log loss.
2. In practice, incorporating classical forensic features increases out-of-fold log loss by 0.0086 to 0.0152, degrading probabilistic calibration.
3. Severe format confounding (JPEG vs. PNG) means that any observed residual correlations reflect file container artifacts rather than genuine manipulation signals.

**Disposition:** Classical forensic indicators remain strictly **non-directional, contextual heuristics**. They must NEVER participate as directional voting inputs in the TruthLens authenticity decision pipeline (`backend/app/services/fusion/engine.py`).

### 2. Research Gate Enforcements

1. **Experiment 3 Closes the Forensic Investigation:** No further univariate or multivariate forensic screening experiments are authorized on the Phase 4 training partition.
2. **Experiment 4 Remains Strictly Unauthorized:** Final evaluation on the 10,385-sample held-out benchmark and 2,999-sample retained reserve partition is **BLOCKED**. Accessing these partitions without formal preregistration and gate approval constitutes a protocol violation.
3. **Production Weights and Configuration Invariance:** Production model weights (`siglip_frozen_head.pt`), preprocessing pipelines, decision thresholds ($\tau = 0.50$), and calibration parameters are unmodified and frozen.
