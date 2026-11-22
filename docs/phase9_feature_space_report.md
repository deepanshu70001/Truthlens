# Phase 9: Visual Feature-Space Analysis

## Data and feature extraction

Features were extracted from the approved 6,995 training samples and 6,436 validation samples (Tiny-GenImage 5,958; Frontier 297; WildAIGC-Bench 181). No calibration or locked test images were opened. Both frozen backbones use the Phase 7 decoding and geometry contract: Pillow RGB decode, bilinear shorter-side resize to 224, center crop to 224, then the appropriate SigLIP or ImageNet normalization. SigLIP-base uses revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`; the second representation is torchvision ResNet-50 ImageNet-1K V2. Both are in evaluation mode and are not fitted on validation data.

Per-sample vectors and metadata are in [`phase9_feature_vectors.npz`](../experiments/reports/phase9_feature_vectors.npz). Metadata includes sample identifier, label, generator/source group, dimensions, aspect ratio, color mode, encoding, split, and transformation contract. The identifiers and other metadata are stored beside the arrays, never concatenated into model features.

## Grouped linear-probe diagnostics

Five-fold grouped cross-validation uses only the approved training partition. A standardized logistic regression is refitted inside each fold; source groups never cross folds. Results are out-of-fold estimates, not trained detector candidates.

| Representation | Dimensions | Accuracy | Balanced accuracy | ROC-AUC (95% group-bootstrap CI) |
|---|---:|---:|---:|---:|
| SigLIP-base | 768 | 0.878 | 0.878 | 0.948 [0.941, 0.956] |
| ResNet-50 ImageNet V2 | 2,048 | 0.615 | 0.615 | 0.659 [0.634, 0.685] |

Training class-centroid Euclidean distance is 3.154 for SigLIP and 1.282 for ResNet-50. Within-class variance traces and within-to-between ratios are recorded in [`phase9_feature_space_analysis.json`](../experiments/reports/phase9_feature_space_analysis.json). In the SigLIP space, Tiny validation generator centroids sit about 2.57–2.99 units from the approved training AI centroid; Frontier validation category centroids range about 8.26–13.34 units. Each Frontier category has only 4 or 8 AI samples, so this is a descriptive distance pattern with substantial sampling uncertainty, not proof of distinct generator clusters.

## Figures and projection limits

- [SigLIP PCA](../experiments/reports/figures/phase9_siglip_pca.png)
- [ResNet-50 PCA](../experiments/reports/figures/phase9_resnet50_pca.png)

PCA is fitted on training features and then applied to validation features. UMAP was unavailable in the environment (`umap-learn` is not installed). A two-dimensional projection discards most feature dimensions and may distort distances, density, overlap, and apparent clusters. These figures are descriptive only; the grouped linear probes and high-dimensional distance summaries are the quantitative diagnostics.
