# TruthLens Phase 6 experiments

This directory contains reproducible evaluation tools. It contains no user-image data and does not silently treat repository images as labeled research data.

## Current status

- `dataset_audit.py` inventories an explicitly selected directory using aggregate counts, image dimensions, file validity, SHA-256 duplicates and dHash candidate pairs (Hamming distance ≤2). It does not print image content or copy images. Candidate pairs require manual/source-group review and are not automatically excluded.
- `evaluate_authenticity.py` accepts a reviewed CSV manifest and runs the configured detector, threshold analysis, image transformations, and standard binary metrics. It rejects unapproved/unlabeled manifests and flags duplicate groups crossing splits. It never tunes thresholds on the test split.
- `calibrate_authenticity.py` fits a temperature-scaling artifact from validation rows only. It requires separate non-empty train/validation/test splits, source and rights references, at least 30 validation examples per class, no cross-split group or exact-hash leakage, and an exact checkpoint revision. The output remains a candidate unless the operator explicitly confirms human review. Application use is optional and is never implied by running the script.
- The repository has no approved labeled manifest yet. Consequently, scientific model metrics are not evaluable. Test fixtures and local uploads are excluded.

## Commands

From repository root, with the project environment active:

```powershell
python experiments/dataset_audit.py --root test_data --output experiments/results/repository_test_data_audit.json
python experiments/evaluate_authenticity.py --manifest experiments/manifests/authenticity.csv --split test --output experiments/results/authenticity_test.json
python experiments/calibrate_authenticity.py --manifest experiments/manifests/authenticity.csv --output models/configs/authenticity-calibration.json
```

The manifest schema is `experiments/manifests/authenticity.example.csv`. Before use, copy it and complete source URL, license/permission reference, label protocol, split, and duplicate/event group IDs. Do not place sensitive uploads in experiment artifacts. Group related images and duplicates in a single split. Use `validation` for threshold selection; reserve `test` for one final evaluation.

After independently reviewing data labels, source rights, generator categories, and split isolation, an operator may add `--confirm-reviewed-validation-data` to produce an artifact accepted by the runtime. Then configure `AUTHENTICITY_CALIBRATION_PATH` and pin `AUTHENTICITY_MODEL_REVISION` to the checkpoint revision recorded by the run. The runtime rejects artifacts for another model/revision. The calibration artifact records validation NLL, Brier score, ECE, temperature, selected abstention thresholds, and limitations. These are calibration-fit diagnostics, not held-out test accuracy. Use the reserved test split once with `evaluate_authenticity.py` after the calibration and thresholds are frozen. A calibrated probability can then be displayed as scoped to the validation distribution, but the overall application label remains inconclusive until input-domain membership is independently validated; TruthLens currently has no validated OOD detector.

Without an accepted calibration artifact, the detector exposes its raw model argmax and softmax scores but sets the application classification to `inconclusive`; raw softmax magnitude cannot create a directional fusion vote. The detector does not claim OOD detection, domain validation, empirical accuracy, or manipulation classification. Running a script does not make a dataset licensed or independent; these remain human-reviewed prerequisites.

No performance number should be reported in a paper until a valid labeled manifest, label protocol, permission basis, and independent grouping have been reviewed and recorded.
