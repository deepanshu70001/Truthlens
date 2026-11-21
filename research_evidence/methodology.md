# TruthLens Methodology

> **Document Status:** Comprehensive System & Evaluation Methodology
> **Source Files:** `backend/app/services/orchestrator.py`, `backend/app/services/authenticity/detector.py`, `experiments/train_phase7.py`, `experiments/calibrate_phase7_candidate.py`, `experiments/create_evaluation_splits.py`

---

## 1. System Pipeline Architecture

TruthLens employs an asynchronous, multi-stage processing pipeline coordinated by a central orchestrator (`backend/app/services/orchestrator.py`). An uploaded image undergoes 19 discrete evaluation stages:
1. **Validation:** Checks MIME type, file signature magic bytes, byte length ($\le 20$ MB), and maximum pixel count ($\le 40$ MP).
2. **Persistence & Hashing:** Computes SHA-256 cryptographic digest and 64-bit perceptual hash (pHash); persists `Analysis` and `ImageAsset` database records.
3. **Preprocessing:** Resolves EXIF orientation, converts color space to normalized RGB, generates thumbnail and resized derivatives, and computes basic image statistics (entropy, sharpness, dynamic range).
4. **Authenticity Detection:** Executes the production authenticity detector (`truthlens/siglip-cifake-frozen-head`), producing class probabilities and calibrated confidence.
5. **Classical Forensics:** Evaluates spatial and transform-domain heuristics: Error Level Analysis (ELA), Laplacian noise variance, JPEG ghosting, edge density, and resampling artifacts.
6. **Optical Character Recognition (OCR):** Extracts text blocks and coordinates via EasyOCR with optional Tesseract fallback.
7. **Object Detection & Visual Explainability (Optional):** Emits object bounding boxes via Ultralytics YOLO and gradient-weighted class activation heatmaps via Grad-CAM.
8. **Text Normalization & Claim Extraction:** Cleans OCR text, standardizes Unicode, and extracts candidate factual statements, named entities, and temporal tokens.
9. **External Verification Lookups (Optional):** Queries Google Fact Check Tools API for corroborating reviews and Google Vision Web Detection for reverse-image provenance.
10. **Evidence Normalization & Status-Aware Fusion:** Normalizes stage outputs into typed evidence items and aggregates directional votes under strict eligibility rules.
11. **Report Generation:** Assembles a structured JSON document recording findings, stage execution statuses, confidence ratings, and limitations.

---

## 2. Pre-Training Container Format Audit

To detect format confounding in training datasets, TruthLens trains Random Forest classifiers on non-pixel metadata:
- Extracted features: file extension, MIME type, file size bytes, byte entropy, width, height, aspect ratio.
- Cross-validation on raw Tiny-GenImage yielded an ROC-AUC of 1.0000, revealing that 100% of authentic samples were JPEG photographs and 100% of synthetic samples were PNG exports.

---

## 3. Matched In-Memory Augmentation Pipeline

To neutralize container-format shortcuts without mutating disk datasets, Phase 7 implemented dynamic in-memory transformations applied symmetrically to both classes during training:
1. **Dynamic Transcoding:** Pillow encodes images in-memory to randomly selected containers: JPEG (40% probability), PNG (30% probability), or WebP (30% probability).
2. **Quality Variation:** Quality factors are randomly sampled: $Q \in [30, 95]$ for JPEG/WebP, and compression level $\in [1, 9]$ for PNG.
3. **Spatial Jitter:** Multi-scale random cropping ($[0.8, 1.0]$ area ratio) and horizontal flipping ($p = 0.5$).
4. **Bicubic Normalization:** Resizing to $224 \times 224$ and normalizing with SigLIP mean and standard deviation:
   $$\mu = [0.5, 0.5, 0.5], \quad \sigma = [0.5, 0.5, 0.5]$$

This intervention collapsed metadata Random Forest ROC-AUC to 0.5028 (chance level).

---

## 4. Group-Aware Split Governance

Dataset splits (`experiments/create_evaluation_splits.py`) strictly avoid evaluation data leakage:
- **Disjoint Grouping:** Clustered identical prompt text, source events, and near-duplicate images into cohesive groups.
- **Generator Isolation:** Partitions preserved balanced representation of distinct generator engines (BigGAN, GLIDE, SD 1.5, Midjourney, DALL-E 3, FLUX.1) across train, val, calib, and test sets.
- **Sealed Test Set:** The 10,385-sample locked test partition was frozen prior to candidate model selection.

---

## 5. Post-Hoc Temperature Calibration

To mitigate out-of-domain overconfidence, TruthLens fits a single scalar parameter $T > 0$ on logit predictions $z$ using disjoint calibration partitions:
$$\hat{p}_{\text{calibrated}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z / T}}$$
The parameter $T$ is optimized via L-BFGS to minimize Negative Log-Likelihood (NLL) with bounds $T \in [0.1, 100.0]$:
$$\mathcal{L}_{\text{NLL}}(T) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i(T) + (1 - y_i) \log(1 - \hat{p}_i(T)) \right]$$

---

## 6. Status-Aware Multi-Modal Evidence Fusion

The fusion engine (`backend/app/services/fusion/engine.py`) enforces strict scientific boundaries:
- **Eligibility:** Only stages reporting status `completed` or `partial` may submit evidence. Failed, skipped, or unavailable stages are excluded from scoring.
- **Directional Voting:** Authenticity direction is determined exclusively by model evidence. Classical forensics, OCR claims, and contextual cues are designated `informational` and cannot alter the authenticity score.
- **Uncertainty Gating:** Raw scores within $[0.35, 0.65]$ are tagged `inconclusive` to prevent forced binary misclassifications.
