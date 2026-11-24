# TruthLens — Experiment 3: Multivariate Forensic Complementarity Screening

## Exploratory Research Protocol 1.0.0

**Execution date:** 2026-11-24 UTC  
**Status:** COMPLETED — EXPLORATORY  
**Authorized partition:** experiments/manifests/phase4_training_manifest.csv (N=6995)  
**Common complete-case cohort:** 6936 (99.16% coverage)  
**Baseline:** truthlens/siglip-cifake-frozen-head  
**Output:** research/experiment3_multivariate_screening/

> Experiment 2's uploaded closeout audit explicitly recorded that Experiment 3 was not recommended because no univariate probe passed the pre-defined gate. The user subsequently authorized this follow-on. It is therefore exploratory and post hoc, not confirmatory. No result authorizes production directional forensic voting.

## 1. Research question

Do combinations of the four continuous Experiment 2 probes provide incremental predictive information beyond the SigLIP AI logit on a common complete-case group-aware nested-CV cohort?

ELA and JPEG block artifacts are excluded because Experiment 2 found zero AI-class overlap in their applicable subgroups.

## 2. Data integrity and isolation

- Authorized manifest rows: 6995; source groups: 903.
- Group clusters crossing the immutable Experiment 2 outer folds: 0.
- All input CSVs and checkpoint hash were verified against the current checked-in Experiment 2 manifest and approved identities.
- Source images were not opened; only the authorized manifest metadata and Experiment 2 derived artifacts were used.
- Diagnostic validation, calibration, held-out benchmarks, and protected reserve partitions were not accessed.
- Format by class: {'likely_authentic': {'JPEG': 3515}, 'ai_generated': {'PNG': 3480}}. The training set is fully format-confounded; any gain may reflect file format or dataset source, not a universal forensic authenticity cue.

## 3. Cohort

- Complete-case N: 6936
- Real: 3512
- AI-generated: 3424
- Unique source groups: 903
- Excluded rows: 59
- Imputation: none.

## 4. Candidate models

| Model | Added inputs | Adjustment |
|---|---|---|
| M0 | None | SigLIP logit baseline |
| M1 | Noise + edge | Standardized linear features |
| M2 | Texture + resampling | Standardized linear features |
| M3 | All four | Standardized linear features |
| M4 | All four | Correlation clustering and signed alignment |

All models use L2 logistic regression. C is selected by 4-fold StratifiedGroupKFold within each outer training fold. Five outer folds are inherited unchanged from Experiment 2. Scaling and correlation transformation are fit within training folds only.

M4 uses absolute Spearman correlation, complete linkage, threshold 0.85, and sign-aligns cluster members to a deterministic training-fold anchor before averaging.

## 5. Primary results

Primary endpoint: Δ log-loss = log-loss(M0) − log-loss(Mj); positive values favor the augmented model.

| Comparison | Δ Log-Loss | 95% Group Bootstrap CI | Raw p | BH q | Verdict |
|---|---:|---:|---:|---:|---|
| M1_noise_edge | -0.008603 | [-0.014552, +0.000300] | 0.084046 | 0.336183 | EXPLORATORY_INCONCLUSIVE |
| M2_texture_resampling | -0.011358 | [-0.031312, +0.012351] | 0.995000 | 0.995000 | EXPLORATORY_INCONCLUSIVE |
| M3_all_four | -0.015188 | [-0.035340, +0.009455] | 0.838558 | 0.995000 | EXPLORATORY_INCONCLUSIVE |
| M4_correlation_adjusted | -0.015188 | [-0.035340, +0.009455] | 0.838558 | 0.995000 | EXPLORATORY_INCONCLUSIVE |

Uncertainty: 5,000 paired source-group bootstrap replicates sampled within fixed outer folds.  
Hypothesis test: 20,000 two-sided cluster sign-flip permutations within fixed outer-fold strata.  
Multiplicity: Benjamini-Hochberg FDR across M1–M4, alpha=0.05.

Supported improvement requires q < 0.05 and a 95% group-bootstrap interval entirely above zero. Supported detriment uses the symmetric condition. All other outcomes are exploratory-inconclusive.

## 6. Secondary results

paired_metrics.json includes pooled log-loss and Brier score, macro outer-fold ROC-AUC/PR-AUC/balanced accuracy, fold-level metrics, and confusion matrices. Pooled cross-fold ROC-AUC is descriptive only due possible fold-specific intercept/base-rate shifts.

## 7. Limitations

1. Post hoc exploratory follow-on, not an independent confirmatory replication.
2. All approved real samples are JPEG and all AI samples are PNG. Class and native format are perfectly confounded.
3. Source-group size imbalance limits effective independent generator information.
4. Any observed gain applies only to this partition and does not establish a universal forensic cue or external generalization.
5. Classical forensic indicators remain non-directional in production fusion regardless of these results.
6. The uploaded Experiment 2 closeout audit records commit b988e2f; the live GitHub branch has since advanced. This run pins inputs to the live branch's current Experiment 2 reproducibility manifest instead of assuming the older inventory is interchangeable.

## 8. Reproducibility

See experiment3_config.json, input_integrity.json, dataset_integrity.json, coverage.json, fold_summary.json, model_fit_details.json, oof_predictions.csv, paired_metrics.json, bootstrap_confidence_intervals.json, permutation_results.json, multiple_testing_results.json, environment_versions.json, and reproducibility_manifest.json.

Runtime: 5.621 seconds.
