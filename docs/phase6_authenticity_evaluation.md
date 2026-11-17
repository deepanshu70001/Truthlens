# Authenticity detector evaluation

## Task and implementation

The configured detector, `prithivMLmods/Deepfake-Real-Class-Siglip2`, returns fake/real softmax scores. The service maps labels from checkpoint configuration and applies settings thresholds: AI-generated when fake score ≥ 0.65; likely authentic when ≤ 0.35; otherwise inconclusive. The score is explicitly uncalibrated. Manipulation is not a separate class and `manipulation_score` is null. These are operational code facts, not validated thresholds.

## Experiment performed

`experiments/evaluate_authenticity.py` was invoked with the header-only reviewed-manifest template and `--split test`. It returned `NOT_EVALUABLE`, **0 eligible examples**, with no model inference and no accuracy, precision, recall, F1, specificity, AUC, calibration, or robustness number. Machine result: `experiments/results/authenticity_test.json`.

The directory `test_data/originals` contains 35 readable files, 31 JPEG and 4 PNG, with no labels/provenance/license record; the aggregate audit found 31 exact duplicate extra copies and 33 exact dHash candidate repeats. These assets cannot supply an independent test set. Neither the 2026-11-17 synthetic model smoke image nor the uncontrolled Phase 5 browser upload was used as evaluation evidence.

## Threshold and calibration status

The code thresholds (0.35/0.65) are confirmed by source inspection only. There is no measured false-positive/false-negative tradeoff, coverage curve, validation-set threshold selection, reliability diagram, Brier score or ECE. Do not say “65% probability” or “calibrated confidence.” The fixed-threshold sweep script reports metrics only if an eligible manifest is supplied; it does not tune a threshold. A validation split must be used for threshold selection, with the test split kept unopened until the protocol is fixed.

## Robustness status

No labeled independent test data supports resize/compression/crop/blur/brightness/noise robustness. Transforms can change detector scores, but synthetic perturbation tests alone would not establish real-world robustness. Future runs should apply fixed transformation parameters to every test item, keep group/source leakage controlled, and report per-transform metric and abstention coverage.

## Baselines and calibration

No direct comparative benchmark is reported. Published metrics on other datasets are not comparable. A majority baseline can be computed when a valid held-out set exists; a detector baseline requires same images, labels, splits and preprocessing. No calibration claim is justified.

## Result

Authenticity effectiveness is **not evaluable with the available approved data**. This is a scientifically meaningful readiness result, not an accuracy result.
