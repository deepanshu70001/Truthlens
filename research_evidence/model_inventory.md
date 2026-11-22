# TruthLens Model Inventory

> **Document Status:** Complete & Traced to Checkpoint Digests and Execution Registries
> **Source Directory:** `models/`, `experiments/checkpoints/`
> **Registry Manifests:** `experiments/reports/phase7_selected_candidate_identity.json`, `experiments/reports/phase7_training_summary.json`, `experiments/reports/phase7_test_evaluation.json`

---

## 1. Production Model (Locked in Production)

| Attribute | Specification |
|---|---|
| **Model ID** | `truthlens/siglip-cifake-frozen-head` |
| **Role** | Production authenticity classifier |
| **Backbone** | `google/siglip-base-patch16-224` (Hugging Face) |
| **Backbone Revision** | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` |
| **Head Weights Path** | `models/baselines/siglip_frozen_head.pt` |
| **Head SHA-256 Digest** | `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` |
| **Architecture** | Frozen Vision Transformer (12 layers, 768-dim) + Linear Head ($768 \to 1$ logit) |
| **Total Parameters** | 86,432,257 |
| **Trainable Parameters** | 769 (0.0009%) |
| **Input Preprocessing** | `siglip-frozen-v1` (Normalized RGB $224 \times 224$, Bicubic interpolation) |
| **Calibration Status** | Post-hoc temperature scaling ($T=1.3187$ pooled) |
| **Operational Governance** | **RETAINED IN PRODUCTION** |

---

## 2. Research Candidates (Phase 7 Shortcut-Resistant Training & Governance)

The Phase 7 pre-registered protocol authorized only the primary candidate (highest composite validation score) and secondary candidate (highest Frontier validation ROC-AUC) to be evaluated on the locked held-out test splits. Other architectures (R1, R2, A1) were governed by validation metrics only.

| Candidate Model ID | Architecture & Strategy | Checkpoint Path | Checkpoint SHA-256 | Trainable Parameters | Peak VRAM | Tiny Val AUC | Frontier Val AUC | Locked Test Tiny AUC | Locked Test Frontier AUC | Decision / Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **`truthlens/siglip-base-v2-s2`** (Primary) | SigLIP Layers 10, 11 + Head FT | `experiments/checkpoints/phase7/s2_siglip_partial_ft/best_checkpoint.pt` | `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805` | 14,178,817 (16.4%) | 839.1 MB | 0.9276 | 0.4769 | **0.9423** | 0.4234 | **REJECTED FOR PRODUCTION** (Baseline retained) |
| **`truthlens/siglip-base-v2-s1`** (Secondary) | SigLIP Linear Probe ($K=5$ views) | `experiments/checkpoints/phase7/s1_siglip_linear_probe/best_checkpoint.pt` | `2554519107c8ad8de07805e55225e7d5a5ac8f9959aecb47815ac86f6fd92111` | 769 (0.0009%) | 700.5 MB | 0.8373 | **0.5991** | 0.8741 | 0.5464 | Research Candidate |
| **`truthlens/resnet50-v2-r1`** | ResNet-50 Full FT | `experiments/checkpoints/phase7/r1_resnet50_full_ft/best_checkpoint.pt` | `2887136affa6406e4f09a394f0ce6be7ff7ba874ada21a6ff8538d211859a9cb` | 23,510,081 (100%) | 1,121.5 MB | 0.7828 | 0.5241 | *Not evaluated* | *Not evaluated* | Research Candidate (Stopped at validation) |
| **`truthlens/resnet50-v2-r2`** | ResNet-50 Stage 4 FT | `experiments/checkpoints/phase7/r2_resnet50_last_stage/best_checkpoint.pt` | `b721cadc68e7b663adde68c1972475c468e3e34d25fab82d1c59da14222e2086` | 14,976,001 (63.7%) | 606.9 MB | 0.7421 | 0.5047 | *Not evaluated* | *Not evaluated* | Research Candidate (Stopped at validation) |
| **`truthlens/siglip-base-v2-a1`** | SigLIP Ablation (No Augmentation) | `experiments/checkpoints/phase7/a1_siglip_no_aug_ablation/best_checkpoint.pt` | `872b93ceff594fe6469bde42465ee1763656d65a3cf3a6c88efff1b6d4e8d02d` | 14,178,817 (16.4%) | 839.1 MB | 0.9258 | 0.7495 | *Not evaluated* | *Not evaluated* | Shortcut Ablation Baseline |

---

## 3. Historical Rejected Candidate (Phase 5/6)

| Candidate Model ID | Architecture | Checkpoint SHA-256 | Reason for Rejection |
|---|---|---|---|
| **`truthlens/siglip-base-tiny-genimage-e3_partial_finetuning`** | SigLIP Layer 11 + Head (Unaugmented) | `541956a1f766d5093aaec4733f616afaf6deaf5aad1e0bbd6e59f95687a4b96d` | Learned container-format shortcut (100% JPEG real vs 100% PNG AI). Flipped 45.6% of predictions under JPEG-90 transcoding. |

---

## 4. Auxiliary Pipeline Services

- **EasyOCR / Tesseract:** Character recognition engine extracting embedded text in English and Hindi (`backend/app/services/ocr/service.py`).
- **Ultralytics YOLO:** Object detection model for scene entity tagging (`backend/app/services/vision/detector.py`).
- **Grad-CAM:** Gradient-weighted class activation mapping targeting visual attention layers (`backend/app/services/xai/gradcam.py`).
