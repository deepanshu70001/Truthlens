# Phase 6 experiment setup and reproducibility protocol

## Current environment

Recorded validation environment: Windows 10.0.26200, Python 3.11.9, PyTorch 2.14.1+cpu, 8 logical CPU cores, CUDA unavailable, Node workspace runtime v24.22.1. This is an environment record, not a deployment profile. See `phase5_performance_report.md` for the narrowly scoped five-run synthetic microbenchmark.

## Reproduction

From repository root:

```powershell
python -m pytest backend/tests -q
python experiments/dataset_audit.py --root test_data/originals --output experiments/results/test_originals_inventory.json
python experiments/evaluate_authenticity.py --manifest experiments/manifests/authenticity.csv --split validation --output experiments/results/authenticity_validation.json
python experiments/evaluate_authenticity.py --manifest experiments/manifests/authenticity.csv --split test --robustness --output experiments/results/authenticity_test.json
```

The current labeled manifest does not exist. The checked-in example is header-only; execution returns `NOT_EVALUABLE` without loading the model. Add a reviewed manifest only after source, license/permission, labels, and grouping are established. Do not use the held-out test split for threshold/model selection.

## Protocol controls

1. Freeze dataset version, label guide, annotator roles, event/source/generator group assignment and exclusions before training or threshold selection.
2. Hash exact duplicate files and investigate dHash candidates; group duplicates and event variants into one split. Record exclusions with reason rather than deleting silently.
3. Use validation for any threshold choice. Keep test held out. Report the operational 0.35/0.65 abstention band as pre-existing configuration, plus fixed 0.50 diagnostic; do not call scores calibrated probabilities.
4. Compare only on identical examples, transformations, labels and split. Include majority baseline and an appropriate published detector only if compatible code, model license and protocol can be reproduced.
5. Save JSON outputs, source revision and dirty-tree hash, package versions, hardware, seed, checkpoint ID/revision, manifest version and run command. Never store credentials or personal images in result artifacts.
6. For bootstrap confidence intervals, resample independent event/source groups rather than images when observations are clustered. With insufficient group count, omit inferential claims.

## Metrics and decision rules

For binary authenticity use per-class count, confusion matrix, precision, recall, specificity, F1, ROC-AUC and average-precision PR-AUC where both classes exist. Report `coverage` and uncertainty abstention separately from metrics on covered cases. Report fixed-threshold behavior and score ranking separately. The script can compute raw-score Brier/ECE diagnostics and a fixed transformation sensitivity check, but it does not calibrate or claim calibrated probabilities. Reliability plots and any calibration method require an independent labeled validation set and declared binning/procedure.

For OCR, annotate ground-truth text and report normalized CER/WER per language/script and image-quality stratum. For extraction, annotate spans/types and report precision/recall/F1. For retrieval, independently label relevant results and report P@K/MRR; evaluate verdict attribution only on human-reviewed evidence and distinguish provider outage. For XAI, use masks or reviewer protocols and sanity checks, not visual inspection alone.

## Current experiment artifacts

- `experiments/results/test_originals_inventory.json`: aggregate audit of test originals, no labels/inference.
- `experiments/results/authenticity_test.json`: actual run status `NOT_EVALUABLE`, zero eligible rows, no model metrics.
- `experiments/results/test_data_readiness.png`: generated plot of aggregate audit counts, not model performance.
- `experiments/results/fusion_controlled_scenarios.json`: deterministic rule-path outputs from constructed evidence, not sample accuracy.

The audit plot requires matplotlib, listed in `backend/requirements-dev.txt`. The working tree was modified during implementation; a Git commit should be created before final submission to make the exact source state citable.
