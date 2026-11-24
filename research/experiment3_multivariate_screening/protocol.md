# TruthLens Experiment 3 — Exploratory Protocol 1.0.0

**Authorization:** 2026-11-24  
**Live branch baseline:** fix/truthlens-remediation at f4baf15e67d2e837a17df7da0383cd0ede1a8b0a  
**Partition:** Phase 4 training manifest, 6,995 samples

## Status

The uploaded Experiment 2 closeout audit explicitly states Experiment 3 was not recommended because no univariate forensic probe passed the gate. The user subsequently authorized this follow-on. This is an exploratory, post hoc analysis, not a confirmatory preregistration or independent replication.

## Inputs

Only the Phase 4 training manifest, the baseline checkpoint (hash only), and Experiment 2's current checked-in reproducibility manifest, manifest summary, model identity, forensic feature CSV, fold assignments CSV, and SigLIP logits CSV are accessed. The script verifies hashes and byte sizes against the current branch artifacts. It accepts exact hashes or CRLF reconstruction only; any other mismatch fails closed.

Expected manifest SHA-256: 22ea2825c7ab75ef110c7a6e2a7c67e69459a3afd3ed6159adbcbed7abc168ed.

Expected frozen baseline checkpoint SHA-256: e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538.

No source images are opened. No validation, calibration, held-out benchmark, or protected reserve partition is read.

## Cohort

Use the common complete-case intersection where all four continuous probes (noise, edge, texture, resampling) are successful and finite. Do not impute missing or not-applicable values. Exclude ELA and JPEG block artifacts because they had zero AI-class overlap in Experiment 2.

## Models

- M0: SigLIP AI logit only.
- M1: M0 + noise + edge.
- M2: M0 + texture + resampling.
- M3: M0 + all four.
- M4: M0 + all four with train-fold-only correlation clustering and sign alignment.

L2 logistic regression; C grid {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}; 5 immutable Experiment 2 outer folds; 4-fold StratifiedGroupKFold inside each outer training fold.

Scaling, correlation clustering, signed alignment, and C selection are fitted only within training folds. Any inner fold lacking either class or any group leakage stops the experiment. OOF predictions must cover every common-cohort row exactly once per model.

## Endpoints

Primary: paired OOF log-loss improvement, log_loss(M0) minus log_loss(Mj), positive means improvement.

Uncertainty: 5,000 source-group bootstrap replicates, sampled with replacement within each fixed outer fold.

Hypothesis test: two-sided paired group-cluster sign-flip randomization, 20,000 permutations, sign flips within fixed outer-fold strata.

Multiplicity: Benjamini-Hochberg over M1–M4, alpha 0.05. A supported improvement requires adjusted q < 0.05 and the bootstrap 95% CI lower bound above zero.

Secondary: Brier, macro outer-fold ROC-AUC, PR-AUC, balanced accuracy, fold-level metrics and confusion matrices. Pooled cross-fold ROC-AUC is descriptive only.

## Critical interpretation limits

The Phase 4 manifest has all real images in JPEG and all AI images in PNG. Class and native format are perfectly confounded. A gain cannot be interpreted as a universal forensic authenticity cue. Source-group imbalance reduces effective independent generator diversity. No result changes the production rule that classical forensic indicators remain non-directional. Any positive signal requires independent replication on balanced, held-out data.

## Gate deviation

This experiment is run only because the user explicitly authorized it after the Experiment 2 closeout. It does not retroactively satisfy the univariate research gate. Experiment 4 remains unauthorized.
