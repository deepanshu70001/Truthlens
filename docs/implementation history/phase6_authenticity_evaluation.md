# Authenticity detector evaluation

## Task and implementation

## Current detector configuration (2026-11-19)

The default detector pairs the locally supplied `models/baselines/siglip_frozen_head.pt` linear head with the frozen `google/siglip-base-patch16-224` vision encoder. The head checkpoint SHA-256 is `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`. Training used the CIFAKE real/AI dataset, extracted SigLIP `pooler_output` features, and optimized a one-logit BCE head; AI-generated is the positive class. The UI tag follows the requested raw-score rule: AI-generated if its score is greater than the authentic score, otherwise Real/authentic. It is always a binary model tag when prediction succeeds; it is not a calibrated probability or a validated claim about arbitrary real-world images. Manipulation is not a separate class and `manipulation_score` is null.

The remote `ml/models` branch reports a 200-image CIFAKE test subset result for this model (100 images per class; accuracy 0.87, AI recall 0.83, ROC AUC 0.9613). These branch-reported metrics have not been reproduced from a reviewed manifest in this checkout, do not represent an independently collected benchmark, and do not establish generalization to newer generators or in-the-wild images. The synthetic API smoke run confirms integration only and is not evaluation evidence.

The prior Phase 6 manifest-based evaluation below returned `NOT_EVALUABLE` with 0 eligible examples. It remains the current independent-evaluation conclusion; choosing a binary UI tag does not change that result.

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
