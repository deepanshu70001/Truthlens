# Phase 9: Generator-Level Error Analysis

## Scope and method

This report uses only the approved 6,995-sample training manifest and the existing validation roles: Tiny-GenImage (5,958), Frontier Benchmark 2026-10 (297), and WildAIGC-Bench (181). Calibration and locked test roles were excluded before image loading. The benchmark files, manifests, and labels were not changed.

The diagnostic classifier is a logistic regression on frozen SigLIP-base features. Training metrics are out-of-fold predictions from five `StratifiedGroupKFold` folds, grouping by the manifest's source group. For each validation partition, the probe is fitted on approved training features and labels; validation labels are used only to calculate diagnostics. It is a research probe, not the production detector or a Phase 10 selection decision. Scores use a 0.5 threshold. Confidence intervals use 500 source-group bootstrap replicates and seed 242 for validation metrics.

## Aggregate results

| Partition | n (authentic / AI) | Accuracy | Balanced accuracy | Precision | Recall | FPR | FNR | ROC-AUC (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Approved training, grouped out-of-fold | 6,995 (3,515 / 3,480) | 0.878 | 0.878 | 0.895 | 0.855 | 0.099 | 0.145 | 0.948 [0.941, 0.956] |
| Tiny-GenImage validation | 5,958 (3,410 / 2,548) | 0.821 | 0.810 | 0.832 | 0.729 | 0.110 | 0.271 | 0.899 [0.885, 0.909] |
| Frontier 2026-10 validation | 297 (193 / 104) | 0.589 | 0.549 | 0.413 | 0.413 | 0.316 | 0.587 | 0.604 [0.474, 0.738] |
| WildAIGC-Bench validation | 181 (63 / 118) | 0.519 | 0.568 | 0.738 | 0.407 | 0.270 | 0.593 | 0.552 [0.452, 0.650] |

The Frontier interval is broad. WildAIGC has an additional contextual-label caveat: labels reflect the dataset's context and are not objective provenance labels. Neither partition supports a precise population estimate.

## Existing checkpoint validation results

The exact production baseline and Phase 7 checkpoint weights were also evaluated on validation only. This pass did no fitting or threshold selection. The table shows Frontier validation metrics (n=297); complete Tiny, Frontier, and WildAIGC generator/source breakdowns with confidence intervals are in [`phase9_candidate_validation_errors.json`](../experiments/reports/phase9_candidate_validation_errors.json).

| Model | Accuracy | Balanced accuracy | Precision | Recall | FPR | FNR | ROC-AUC (95% group-bootstrap CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production frozen baseline | 0.687 | 0.726 | 0.533 | 0.856 | 0.404 | 0.144 | 0.750 [0.635, 0.847] |
| S1 SigLIP linear probe | 0.539 | 0.579 | 0.409 | 0.712 | 0.554 | 0.288 | 0.599 [0.468, 0.722] |
| S2 SigLIP partial fine-tuning | 0.562 | 0.517 | 0.373 | 0.365 | 0.332 | 0.635 | 0.477 [0.343, 0.628] |
| R1 ResNet-50 full fine-tuning | 0.690 | 0.598 | 0.625 | 0.288 | 0.093 | 0.712 | 0.524 [0.378, 0.678] |
| R2 ResNet-50 final-stage fine-tuning | 0.697 | 0.601 | 0.659 | 0.279 | 0.078 | 0.721 | 0.505 [0.363, 0.651] |
| A1 SigLIP unaugmented ablation | 0.694 | 0.669 | 0.560 | 0.587 | 0.249 | 0.413 | 0.750 [0.651, 0.860] |

The frozen production head retains ranking signal on this validation role but has a high 0.404 false-positive rate at 0.5. S2 and the two ResNet fine-tuned checkpoints have lower Frontier AUCs. Frontier confidence intervals are broad, and A1 is an unaugmented shortcut-prone ablation rather than a clean candidate.

## Generator and source groups

Training generator diagnostics are stored with counts and group-bootstrap intervals in [`phase9_generator_error_analysis.json`](../experiments/reports/phase9_generator_error_analysis.json). The 6,995 approved records include ADM, BigGAN, GLIDE, Stable Diffusion v1.5, and ImageNet-Camera real images. Per-generator training predictions are out-of-fold, so each source group stays in one fold.

The Frontier validation AI rows cover 19 named generator categories, with only 4 or 8 samples per category in this validation role. Their centroid and error summaries are descriptive and too small for category-level conclusions. Tiny validation contains larger groups (roughly 480–540 AI samples for ADM, BigGAN, Midjourney, Stable Diffusion v1.5, and Wukong). Tiny generator groups absent from the training generator list are useful held-out development diagnostics, but remain within the same benchmark and should not be mistaken for external validation.

## Acquisition and format differences

The authorized training records remain strongly class-correlated in their source metadata: all 3,515 authentic training images are JPEG, while all 3,480 AI training images are PNG. Mean recorded aspect ratio is 1.220 for authentic images and 1.000 for AI images. Decoded color modes are RGB/L for authentic and RGB/RGBA for AI. This is a documented property of the approved dataset, not evidence that encoding causes a model error. The Phase 7 matched in-memory augmentation specification was designed to reduce direct format, dimension, and aspect-ratio shortcuts during training.

Validation metadata distributions are recorded separately by dataset and class in the JSON report. The image loading and canonical transform use Pillow RGB decoding, shorter-side resize to 224, center crop to 224, and backbone-specific normalization. No filenames, paths, source names, labels, or manifest metadata are model input features.

## Failure cases and interpretation

[`phase9_failure_cases.csv`](../experiments/reports/phase9_failure_cases.csv) records validation false positives and false negatives for the frozen-feature probe, production baseline, and S1/S2/R1/R2/A1 checkpoints, with sample identifiers and available source/acquisition metadata. It contains 12,122 model-error rows across the seven diagnostic models. It does not contain locked-test predictions.

The strongest operational diagnostic is the increase in both false-positive and false-negative rates on the small Frontier validation partition relative to Tiny validation. This is consistent with a cross-generator/domain gap. It does not identify a cause: generator identity, prompt/style composition, source pipeline, resolution, compression history, and label construction vary together. Small Frontier category counts make per-generator rankings unstable.
