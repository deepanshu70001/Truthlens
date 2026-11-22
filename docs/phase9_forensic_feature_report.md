# Phase 9: Forensic Frequency Feature Analysis

## Extractor

[`experiments/forensics/phase9_frequency_features.py`](../experiments/forensics/phase9_frequency_features.py) implements a deterministic, in-memory, pixel-only extractor with 34 features covering FFT spectrum bands, 8×8 block-DCT bands, high-pass residual statistics, edge/texture measures, and a resampling-periodicity proxy. It applies the Phase 7 Pillow decode, shorter-side resize, and center-crop geometry equally to both labels. It does not accept a path, label, or dataset-source argument. Inputs smaller than 8 pixels on either side are marked unsupported; decode and numerical errors are returned with a failure status. Tests cover deterministic output, undersized images, and closed/corrupt and wrong-type inputs.

Original image files are never written or modified. All 13,431 authorized records produced finite vectors. Per-sample feature arrays and associated metadata are in [`phase9_feature_vectors.npz`](../experiments/reports/phase9_feature_vectors.npz). The 34 univariate grouped-CV results and cluster-bootstrap intervals are in [`phase9_frequency_feature_ablation.csv`](../experiments/reports/phase9_frequency_feature_ablation.csv); aggregate and validation details are in [`phase9_frequency_feature_analysis.json`](../experiments/reports/phase9_frequency_feature_analysis.json).

## Results

Frequency-only logistic probes achieved 0.895 ROC-AUC on grouped training out-of-fold predictions (95% source-group bootstrap CI 0.830–0.919). The leading univariate training features were FFT spectral flatness (AUC 0.876; CI 0.726–0.933), DCT high/low energy ratio (0.852; 0.706–0.909), and residual vertical correlation (0.815; 0.656–0.876). These estimates indicate usable signal in this dataset and pipeline. They do not establish a universal AI-generation signature; the training classes also differ in source, resolution, encoding, and acquisition history.

On the independent diagnostic validation roles, frequency-only AUC was 0.762 on Tiny-GenImage validation, 0.623 on Frontier validation, and 0.564 on WildAIGC-Bench validation. The Frontier interval is wide due to n=297; WildAIGC additionally retains its contextual-label caveat.

## Compression and resize stability

On a deterministic, class-balanced training sample of 400 images, mean featurewise Pearson correlation before versus after each in-memory transformation was:

| Transformation | Mean correlation across 34 features |
|---|---:|
| JPEG quality 90 | 0.970 |
| JPEG quality 75 | 0.966 |
| JPEG quality 50 | 0.963 |
| Downscale to 50%, then canonical resize | 0.851 |
| Upscale to 200%, then canonical resize | 0.962 |

Most aggregate features were stable under these specific JPEG encodings. DCT AC sparsity was unstable, and downscaling affected several high-frequency and residual measurements. These correlations measure feature stability, not classification stability or causal robustness; they do not cover every JPEG encoder, resizing kernel, social-media pipeline, or repeated compression history.
