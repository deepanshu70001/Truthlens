# TruthLens — Production SigLIP Grad-CAM Runtime Verification Report

**Date:** 2026-11-23  
**Repository:** `D:\Projects\Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**HEAD Commit:** `7f97a72ccc48802ccc251df6a60389e65cd1df2b`  
**Final Status Classification:** **`VERIFIED — PRODUCTION SIGLIP GRAD-CAM`**

---

## 1. Executive Summary

A comprehensive runtime verification of the Grad-CAM explainable AI (XAI) implementation was conducted using the actual production TruthLens authenticity detector (`FineTunedSiglipClassifier`) and its active frozen checkpoint assets.

- **Outcome:** **`VERIFIED — PRODUCTION SIGLIP GRAD-CAM`**
- **Core Findings:**
  1. The production authenticity model loads completely offline from local cache: Hugging Face `SiglipVisionModel` (`google/siglip-base-patch16-224`, snapshot `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`) coupled with the trained binary classification head `models/baselines/siglip_frozen_head.pt` (SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`).
  2. A target layer traversal defect was identified: `_find_target_layer` assumed a flat `vision_model.encoder` hierarchy (matching the mock toy model in tests), whereas real Hugging Face `SiglipVisionModel` wraps `SiglipVisionTransformer` in its internal `.vision_model` child attribute.
  3. A minimal 3-line structural fix in `_find_target_layer` resolved the traversal to locate `model.vision_model.vision_model.encoder.layers[-1].layer_norm2` (`LayerNorm((768,), eps=1e-06)`).
  4. Real Grad-CAM inference was executed on the permitted fixture (`backend/tests/fixtures/test_photo.jpg`). It generated full-resolution 640×480 attribution and overlay maps without altering model weights, predictions, or logits.
  5. Deterministic verification confirmed **0.0 logit difference** between pre- and post-Grad-CAM inferences and bit-exact preservation of all 86 model parameter tensors.

---

## 2. Production Model Identity & Architecture

- **Model Class:** `backend.app.services.authenticity.detector.FineTunedSiglipClassifier`
- **Configured Model ID:** `truthlens/siglip-cifake-frozen-head`
- **Model Revision String:** `backbone:google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed;head_sha256:e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`
- **Active Backbone:** `google/siglip-base-patch16-224`
- **Backbone Revision / Git Commit:** `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`
- **Active Checkpoint Path:** `D:\Projects\Truthlens\models\baselines\siglip_frozen_head.pt`
- **Head Architecture:** `torch.nn.Linear(in_features=768, out_features=1, bias=True)`
- **Class Mapping:** `{0: 'real', 1: 'ai_generated'}`
- **Preprocessing Implementation:** `FineTunedSiglipProcessor` (`siglip-frozen-v1`: Resize 224, CenterCrop 224, ToTensor, Normalize mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
- **Execution Device & Precision:** `cuda:0` / `torch.float32`

---

## 3. Checkpoint Files & Cryptographic Verification

| Asset | Path | File Size | SHA-256 Checksum |
|---|---|---|---|
| Frozen Head Weights | `models/baselines/siglip_frozen_head.pt` | 5,053 bytes | `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` |
| Backbone Model | `models/cache/.../7fd15f06.../model.safetensors` | 812,672,320 bytes | `2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8` |
| Backbone Config | `models/cache/.../7fd15f06.../config.json` | 432 bytes | `cd85b3d28829722820bcb89a2cfbb4160e55fd359249a3044da724166a8d9688` |

---

## 4. Runtime & Hardware Environment

- **Python Version:** `3.12.10 (tags/v3.12.10:0cc8128, Apr 8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- **PyTorch Version:** `2.7.1+cu118`
- **Transformers Version:** `4.57.6`
- **CUDA Available:** `True` (Device count: 1)
- **GPU Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB dedicated VRAM, 8 GB shared memory)
- **Model Loading Duration:** 7.25 seconds
- **VRAM Utilization:** ~354 MB allocated / ~402 MB reserved (well within the 4 GB limit)

---

## 5. Grad-CAM Target Layer Discovery & Implementation Correction

### 5.1 Root Cause of Previous Failure
In `backend/app/services/xai/gradcam.py`, `_find_target_layer` performed:
```python
vision = getattr(self.model, "vision_model", None)
encoder = getattr(vision, "encoder", None)
```
In mock tests, `_ToySiglip` defined `self.vision_model = SimpleNamespace(encoder=...)`. However, Hugging Face `transformers.SiglipVisionModel` wraps its transformer in an internal `self.vision_model` attribute (`SiglipVisionTransformer`). Consequently, `getattr(vision, "encoder", None)` evaluated to `None`, causing Grad-CAM to return `status="unavailable"`.

### 5.2 Corrected Layer Resolution
`_find_target_layer` was updated with minimal traversal logic:
```python
def _find_target_layer(self):
    vision = getattr(self.model, "vision_model", getattr(self.model, "siglip", self.model))
    inner = getattr(vision, "vision_model", vision)
    encoder = getattr(inner, "encoder", getattr(vision, "encoder", None))
    layers = getattr(encoder, "layers", None)
    if layers:
        last = layers[-1]
        return getattr(last, "layer_norm2", last)
    return None
```
- **Discovered Target Module:** `model.vision_model.vision_model.encoder.layers[11].layer_norm2`
- **Target Module Type:** `torch.nn.modules.normalization.LayerNorm((768,), eps=1e-06, elementwise_affine=True)`
- **Cleanup Enhancement:** `GradCAM.cleanup()` was updated to explicitly reset `self.activations = self.gradients = None` and call `model.zero_grad(set_to_none=True)` to ensure zero gradient retention.

---

## 6. Real Production Verification Sequence

**Permitted Fixture:** `backend/tests/fixtures/test_photo.jpg` (640×480 RGB JPEG)

### Step A: Baseline Prediction
- **Execution Device:** `cuda`
- **Inference Duration:** 541.55 ms
- **Raw Logits:** `{'fake': 1.4130399227142334, 'real': -1.4130399227142334}`
- **Predicted Class (`model_prediction`):** `likely_ai_generated`
- **Overall Decision Label:** `inconclusive` (governed by configured decision gating thresholds)
- **Raw Softmax Scores:** `ai_generated_score: 0.9441`, `authentic_score: 0.0559`
- **Calibrated Probability:** `None` (`calibration_status: "not_calibrated"`)

### Step B: Real Grad-CAM Generation
- **Target Module:** `SiglipEncoderLayer[11].layer_norm2`
- **Target Class:** `ai_generated` (index 1)
- **Target Score:** `0.944068968296051`
- **Activation Tensor Shape:** `[1, 196, 768]` (196 tokens = 14×14 grid for 224×224 input / patch size 16)
- **Gradient Tensor Shape:** `[1, 196, 768]`
- **Raw CAM Dimensions:** `(14, 14)`
- **Raw CAM Min / Max:** `0.0` / `1.0` (100% finite values)
- **Heatmap Image Dimensions:** `(640, 480)` (matches source photo)
- **Overlay Image Dimensions:** `(640, 480)` (matches source photo)
- **Grad-CAM Execution Duration:** 311.57 ms
- **Salient Regions Detected:** 2 regions:
  - Region 1: Bounding Box `[260, 234, 443, 296]`, Area Fraction `0.03693`, Max Activation `0.98931`
  - Region 2: Bounding Box `[152, 234, 236, 292]`, Area Fraction `0.01586`, Max Activation `0.85595`

### Step C: Prediction Preservation Check
- **Post-Grad-CAM Raw Logits:** `{'fake': 1.4130399227142334, 'real': -1.4130399227142334}`
- **Logit Difference:** `0.0` (bit-exact identical)
- **Post-Grad-CAM Label:** `inconclusive` (identical)
- **Post-Grad-CAM Model Prediction:** `likely_ai_generated` (identical)
- **Post-Grad-CAM Softmax Scores:** `0.944068968296051` / `0.055931031703948975` (identical)

### Step D: Model Integrity & Cleanup
- **Parameter Tensors Check:** All 86 parameter tensors evaluated with `torch.equal(old, new)` — **100% bit-exact match**.
- **Model Training State:** `model.training` was `False` before and remained `False` after.
- **Hook Cleanup:** Forward and backward hooks removed; `_hooks` list length is `0`.
- **Gradients Cleared:** `all(p.grad is None for p in model.parameters())` evaluates to `True`.
- **Checkpoint Immutability:** Head SHA-256 and Backbone SHA-256 verified identical before and after.

---

## 7. Mock vs. Real-Runtime Test Distinction

- **Mock / Contract Tests:**
  - Mock `_ToySiglip` exercises target layer registration, coordinate alignment, and backward pass on CPU without requiring GPU or model downloads.
  - Mock tests verify graceful fallback to `status="unavailable"` when target layers are missing.
- **Real-Runtime Verification:**
  - Full Hugging Face `SiglipVisionModel` (86 million parameters) loaded into GPU memory.
  - Forward-backward pass through genuine 12-layer ViT attention and MLP blocks on real image fixture.
  - Real OpenCV jet-colormap overlay rendering and contour-based region extraction.

---

## 8. Regression Suite Results

1. **Focused E6/E7 Contract Tests:**
   ```bash
   python -m pytest backend/tests/test_e6_e7_contract.py -q
   ```
   **Result:** `51 passed in 5.05s` (100% passing)

2. **Full Safe Backend Regression Suite:**
   ```bash
   python -m pytest backend/tests -q --ignore=backend/tests/test_phase2_dataset_splits.py --ignore=backend/tests/test_phase3_evaluation.py --ignore=backend/tests/test_phase4_and_phase3_5.py --ignore=backend/tests/test_phase5_training.py --ignore=backend/tests/test_phase6_evaluation.py --ignore=backend/tests/test_phase7.py
   ```
   **Result:** `409 passed, 12 skipped, 6 warnings in 28.47s` (100% passing)

3. **Bytecode Compilation Check:**
   ```bash
   python -m compileall -q backend/app
   ```
   **Result:** Clean (0 compilation errors)

4. **Git Formatting / Diff Check:**
   ```bash
   git diff --check
   ```
   **Result:** Clean (0 formatting / whitespace issues)

---

## 9. Final Git Status & Modifications

```
 M backend/app/services/forensics/analyzer.py
 M backend/app/services/provenance/service.py
 M backend/app/services/xai/gradcam.py
 M backend/tests/test_e6_e7_contract.py
 M docs/MODULE_STATUS_REGISTER.md
 M docs/e6_e7_forensics_provenance_report.md
?? docs/siglip_gradcam_verification_report.md
?? docs/yolo_verification_report.md
```
- Strictly no commits or pushes.
- No modifications to SigLIP weights, preprocessing, calibration, fusion, or decision policy.
