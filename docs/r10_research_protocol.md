# R10 Preregistered Hybrid Experiment Protocol

**Protocol ID:** `truthlens-r10-siglip-frequency-residual-v1`  
**Protocol version:** `1.0.0`  
**Status:** Preregistered before this R10 run.  
**Date:** 2026-11-22  
**Research only:** Results cannot change production inference, calibration, routing, or E11 fusion.

## Research question and hypotheses

**Primary question:** Does a small, training-fitted frequency residual improve the frozen production SigLIP head's ranking on the permitted Frontier validation partition?

- **H1 (primary):** The R10 candidate's Frontier validation ROC-AUC exceeds the unchanged frozen production head's ROC-AUC.
- **H0:** The paired Frontier validation ROC-AUC difference is zero or below.
- **Secondary:** Measure the candidate's Tiny validation discrimination and fixed-threshold error rates, its diagnostic WildAIGC behavior, its missing-feature fallback, and CPU fit/runtime cost.
- **Complexity check:** The 35-parameter residual branch must not add a validation improvement that is explainable only by one small Frontier source group. Report group counts and a paired source-group bootstrap interval.

No calibration, held-out test, reserve, or additional data is used. No metric or threshold is selected from those protected partitions.

## Explicit input allowlist and permitted operations

| Exact path | Contents and scope documented by the project | Permitted R10 operation |
|---|---|---|
| `experiments/reports/phase9_feature_vectors.npz` | Per-sample 768-dimensional frozen SigLIP vectors, 34 frequency features, labels, sample/source-group metadata, and split role for 6,995 approved training plus 6,436 validation records only | Read the archive metadata and feature arrays; fit on the 6,995 approved training rows; evaluate only on Tiny, Frontier, and separately labeled WildAIGC validation rows. Abort if any other role appears or documented counts do not match. |
| `models/baselines/siglip_frozen_head.pt` | Frozen production linear head; its SHA-256 is recorded in the completion report | Read weights to reproduce the baseline logit from the matching SigLIP vectors. Never write or replace it. |
| `experiments/forensics/phase9_frequency_features.py` | Deterministic 34-dimensional feature implementation | Read source for feature semantics only. No feature extraction or image access is needed for this run. |

The training manifest `experiments/manifests/phase4_training_manifest.csv` is the project's explicitly approved 6,995-row training manifest; it will not be opened because the allowlisted Phase 9 feature archive already contains those training vectors and labels. The phase2 split protocol assigns validation roles for development diagnostics/decision exploration. The archive's validation rows are limited to Tiny-GenImage (5,958), Frontier 2026-10 (297), and WildAIGC-Bench (181). WildAIGC is a contextual-label diagnostic and is excluded from H1 and all model selection.

The combined manifests `experiments/manifests/tiny_genimage_evaluation_splits.csv`, `experiments/manifests/frontier_benchmark_2026_10_evaluation_splits.csv`, and `experiments/manifests/wildaigc_bench_evaluation_splits.csv` contain protected calibration/test/excluded roles alongside validation. They are explicitly outside the R10 read allowlist and will not be opened. No raw image, path list, calibration record, test prediction, test label, reserve row, or other experiment directory is read.

Before fitting, the runner must verify the archive's exact role counts and aggregate class counts against this protocol, the feature matrix widths, finite values, group identifiers, and the checkpoint hash. The governance preflight may aggregate labels only to verify these counts; it must not calculate model metrics or expose row-level labels. If a required field/count is absent or any unapproved split role is present, stop without fitting and record the blocker. Per-row validation labels may be used only after the candidate is fixed, for the preregistered metrics.

## Candidate architecture

The visual branch is the unchanged 768-dimensional SigLIP representation multiplied by the existing frozen production linear head to produce `z_base`. The forensic branch uses the 34 Phase 9 frequency features. A `StandardScaler` is fit on approved training rows only. A regularized linear residual is added:

```text
z_hybrid = z_base + (w · x_frequency_standardized + b)
```

Fit `w` and `b` by minimizing training binary cross-entropy on `z_hybrid` plus `0.5 * ||w||²` (fixed L2 coefficient `1.0`). The production head coefficient is fixed at `1`; the baseline is not refit. There are 35 trainable residual parameters. The only outcome label is `AI-generated=1`, `authentic=0`, as documented by the phase4 and phase9 reports. No OCR, language, third-party fact-check, YOLO, C2PA, or XAI outputs enter the candidate.

The baseline comparison uses the exact same SigLIP vectors and rows with `z_base` alone. The candidate ID is `truthlens/r10-siglip-frequency-residual-v1`; its code, config, output, and coefficients are stored under the new `research/r10_e11/` directory. It is never loaded by production code.

## Fit and evaluation rules

- Fit the scaler and residual branch using only the 6,995 approved training rows.
- Use one fixed configuration; no hyperparameter, feature, epoch, or threshold selection is performed on validation.
- Use logits only. The residual candidate is not calibrated, and logits/sigmoid outputs are not probabilities.
- Primary metric: Frontier validation ROC-AUC difference, `AUC(hybrid) - AUC(frozen baseline)`.
- Estimate a two-sided 95% paired source-group bootstrap interval with 2,000 replicates and seed `2026`; resample groups, keeping rows from each selected group together.
- Secondary metrics: accuracy, balanced accuracy, precision, recall, FPR, and FNR at the fixed zero-logit threshold; Tiny ROC-AUC and WildAIGC diagnostic ROC-AUC; missing-frequency fallback equivalence to baseline; fit and inference time. Report every class count and group count.
- WildAIGC remains a separately labeled contextual-label diagnostic, never part of the primary metric or selection.
- Hypothesis support requires a positive Frontier AUC delta whose paired group-bootstrap 95% interval excludes zero, Tiny balanced accuracy no more than `0.01` below baseline, and Frontier FPR no more than `0.02` above baseline. Otherwise report the hypothesis as unsupported or inconclusive; do not adjust the candidate after seeing validation results.
- No calibration, held-out testing, model promotion, production routing change, or R11/R12/R13 work is permitted in this experiment.

## Reproducibility and stopping rules

Record the source commit, dirty-tree state, Python/library versions, CPU/GPU availability, dataset archive SHA-256, permitted split counts, feature names/config hash, production head SHA-256, cached SigLIP revision, solver/objective, random seed, exact command, runtime, result JSON checksum, and candidate coefficient checksum.

Stop before fitting if the approved feature archive includes any partition beyond the listed training and validation roles, if its role/class counts do not match, if features are non-finite or schema-incompatible, or if the baseline checkpoint hash differs from the supplied identity. Never attempt to repair a mismatch by reading the protected split manifests.
