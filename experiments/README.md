# TruthLens Phase 6 experiments

This directory contains reproducible evaluation tools. It contains no user-image data and does not silently treat repository images as labeled research data.

## Current status

- `dataset_audit.py` inventories an explicitly selected directory using aggregate counts, image dimensions, file validity, SHA-256 duplicates and dHash candidate pairs (Hamming distance ≤2). It does not print image content or copy images. Candidate pairs require manual/source-group review and are not automatically excluded.
- `evaluate_authenticity.py` accepts a reviewed CSV manifest and runs the configured detector, threshold analysis, image transformations, and standard binary metrics. It rejects unapproved/unlabeled manifests and flags duplicate groups crossing splits. It never tunes thresholds on the test split.
- The repository has no approved labeled manifest yet. Consequently, scientific model metrics are not evaluable. Test fixtures and local uploads are excluded.

## Commands

From repository root, with the project environment active:

```powershell
python experiments/dataset_audit.py --root test_data --output experiments/results/repository_test_data_audit.json
python experiments/evaluate_authenticity.py --manifest experiments/manifests/authenticity.csv --split test --output experiments/results/authenticity_test.json
```

The manifest schema is `experiments/manifests/authenticity.example.csv`. Before use, copy it and complete source URL, license/permission reference, label protocol, split, and duplicate/event group IDs. Do not place sensitive uploads in experiment artifacts. Group related images and duplicates in a single split. Use `validation` for threshold selection; reserve `test` for one final evaluation.

Each output records timestamp, repository revision, software/hardware metadata where available, configuration, sample counts, metrics, and exclusions. The detector uses the repository's configured checkpoint and operational thresholds. Scores are uncalibrated. Running the script does not make a dataset licensed or independent; those are human-reviewed prerequisites.

No performance number should be reported in a paper until a valid labeled manifest, label protocol, permission basis, and independent grouping have been reviewed and recorded.
