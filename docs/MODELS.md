# TruthLens — Model Registry

> This document catalogues all ML models investigated for TruthLens.
> Each entry records availability, license, task, size, input format, VRAM, and limitations.
> Models marked **SELECTED** are planned for integration. Others are documented as alternatives.

---

## Model Selection Principles

1. **No single model is the truth oracle.** All models are probabilistic classifiers whose outputs are treated as *signals*, not verdicts.
2. **Concept drift matters.** Image generation advances quickly. Prefer models trained on diverse, recent data.
3. **VRAM budget is 4 GB** (RTX 3050 Laptop). Models must be loadable within this constraint — use sequential loading, not simultaneous.
4. **Abstraction first.** Every model is accessed through the `BaseModel` interface, so swapping is a config change.

---

## 1. Authenticity / Synthetic Image Detection

### 1.1 `prithivMLmods/Deepfake-Real-Class-Siglip2` — **SELECTED (Primary)**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Binary classification: Fake vs Real                                   |
| **Architecture**  | SiglipForImageClassification (SigLIP2-base)                           |
| **Base Model**    | `google/siglip2-base-patch16-224`                                     |
| **Input**         | RGB image → preprocessed to 224×224 tensors                           |
| **Output**        | 2-class logits → softmax probabilities                                |
| **Parameters**    | ~86M (SigLIP2-base)                                                   |
| **VRAM**          | ~400 MB (inference, fp32)                                             |
| **License**       | Apache 2.0                                                            |
| **Framework**     | Hugging Face Transformers (`AutoImageProcessor` + model)              |
| **Availability**  | ✅ Available on Hugging Face Hub                                       |
| **Strengths**     | Modern vision-language encoder; good at general fake-vs-real           |
| **Limitations**   | Binary only (no manipulation-type breakdown); concept drift risk       |
| **Why Selected**  | Lightweight, Apache 2.0 license, easy integration, modern architecture |

### 1.2 `buildborderless/CommunityForensics-DeepfakeDet-ViT` — **SELECTED (Secondary / Ensemble)**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Binary: fake probability (single sigmoid output)                      |
| **Architecture**  | ViT-Small (patch16, 384×384)                                         |
| **Base Model**    | `timm/vit_small_patch16_384.augreg_in21k_ft_in1k`                    |
| **Input**         | RGB, shortest edge → 440px, center-crop 384×384, CLIP-normalized     |
| **Output**        | Single logit → sigmoid → fake probability                            |
| **Parameters**    | ~22M (ViT-Small)                                                      |
| **VRAM**          | ~200 MB (inference, fp32)                                             |
| **License**       | MIT                                                                    |
| **Framework**     | Hugging Face Transformers / timm                                      |
| **Availability**  | ✅ Available on Hugging Face Hub                                       |
| **Training Data** | 2.7M images from 4,803 generators (CommunityForensics dataset)        |
| **Strengths**     | Massive training diversity; excellent generalization; MIT license       |
| **Limitations**   | Requires `transformers >= 5.4.0` for correct preprocessing (verify)    |
| **Why Selected**  | Best-in-class training diversity; strong complement to SigLIP2         |
| **Note**          | Config/weight fix applied July 2026 — ensure latest version is used   |

### 1.3 `prithivMLmods/open-deepfake-detection` — ALTERNATIVE

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Binary: Deepfake vs Real                                              |
| **Architecture**  | SigLIP2-based                                                         |
| **License**       | Apache 2.0                                                            |
| **Status**        | Alternative — similar architecture to 1.1, can be swapped if needed   |

### 1.4 `computervisionpro/convnextv2-real-fake` — ALTERNATIVE

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Binary: Real vs Fake                                                  |
| **Architecture**  | ConvNeXt V2 (variant unclear — Tiny/Base)                             |
| **Input**         | Typically 224×224 or 384×384                                          |
| **License**       | Varies (check repository)                                             |
| **Status**        | Alternative — no official model card found; verify before using       |
| **Risk**          | Possibly older training data → concept drift concern                  |

### 1.5 `prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2` — ALTERNATIVE

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | 3-class: AI-generated vs Deepfake vs Real                             |
| **Architecture**  | SigLIP2-based                                                         |
| **License**       | Apache 2.0                                                            |
| **Status**        | Interesting 3-class variant; could replace 1.1 if finer granularity needed |

### 1.6 `king1oo1/deepfake-model` — ALTERNATIVE

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Binary: Deepfake vs Real                                              |
| **Architecture**  | SigLIP2-based                                                         |
| **Training Data** | 330k+ images (Midjourney, SD, DALL-E, etc.)                           |
| **Status**        | Alternative — large curated training set, good generalization          |

---

## 2. Vision-Language Model (VLM)

### 2.1 `Qwen/Qwen2.5-VL-3B-Instruct` — **SELECTED**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Visual question answering, contextual analysis, image description     |
| **Architecture**  | Multimodal LLM + ViT encoder (3B parameters)                         |
| **Input**         | Image + text prompt                                                   |
| **Output**        | Free-form text response                                               |
| **Parameters**    | ~3B                                                                    |
| **VRAM (BF16)**   | ~7.5 GB                                                               |
| **VRAM (Q4_K_M)** | ~1.9 GB                                                               |
| **VRAM (Q8_0)**   | ~4.1 GB                                                               |
| **License**       | Apache 2.0                                                            |
| **Framework**     | Transformers (`Qwen2_5_VLForConditionalGeneration`)                   |
| **Availability**  | ✅ Available on Hugging Face Hub                                       |
| **Strengths**     | Strong visual reasoning; structured JSON output; multilingual          |
| **Limitations**   | Full precision won't fit in 4 GB VRAM — must use quantized version    |
| **Usage in TruthLens** | Contextual analysis, visual inconsistency detection, scene description. **NOT used as sole fake detector.** |
| **Quantization Plan** | Use Q4_K_M GGUF (~1.9 GB) or int4 via bitsandbytes for RTX 3050  |

> ⚠️ **Important**: This model is used for *contextual reasoning* (describing scenes, identifying visual inconsistencies, answering forensic questions). It is NOT the authenticity classifier.

---

## 3. Language Detection

### 3.1 `papluca/xlm-roberta-base-language-detection` — **SELECTED**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Text language classification (20 languages)                           |
| **Architecture**  | XLM-RoBERTa-base + classification head                                |
| **Input**         | Text string (tokenized)                                               |
| **Output**        | Language code + confidence for 20 languages                           |
| **Languages**     | ar, bg, de, el, en, es, fr, hi, it, ja, nl, pl, pt, ru, sw, th, tr, ur, vi, zh |
| **Parameters**    | ~278M (XLM-RoBERTa-base)                                             |
| **VRAM**          | ~1.1 GB (inference, fp32); ~550 MB (fp16)                             |
| **License**       | Not explicitly stated (assumed research-friendly)                     |
| **Accuracy**      | 99.6% on test set                                                     |
| **Framework**     | Hugging Face Transformers pipeline                                    |
| **Availability**  | ✅ Available on Hugging Face Hub                                       |
| **Strengths**     | High accuracy; covers Hindi and Urdu (key for India context)          |
| **Limitations**   | Only 20 languages; weak on very short text (1-5 words)                |
| **Why Selected**  | Covers key Indian languages (hi, ur); lightweight; high accuracy       |

---

## 4. Translation

### 4.1 `facebook/nllb-200-distilled-600M` — **PLANNED (Future)**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Machine translation (200 languages)                                   |
| **Architecture**  | Transformer seq2seq (distilled)                                       |
| **Parameters**    | ~600M                                                                  |
| **VRAM**          | ~2.1 GB (inference)                                                   |
| **License**       | CC-BY-NC-4.0 (non-commercial only)                                   |
| **Availability**  | ✅ Available on Hugging Face Hub                                       |
| **Strengths**     | 200 languages including many Indian languages; distilled = efficient   |
| **Limitations**   | Non-commercial license; max 512 tokens; not for production deployment  |
| **Status**        | PLANNED — will integrate if OCR text translation is needed             |
| **License Risk**  | CC-BY-NC limits commercial use. For a research project this is acceptable. |

---

## 5. Object Detection

### 5.1 Ultralytics YOLO26s — **SELECTED**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Object detection (COCO classes)                                       |
| **Architecture**  | YOLO26 (Small variant)                                                |
| **Parameters**    | ~9.5M                                                                  |
| **Input**         | RGB image (any resolution, auto-resized)                              |
| **Output**        | Bounding boxes + class labels + confidence scores                     |
| **VRAM**          | ~500 MB (inference)                                                   |
| **License**       | AGPL-3.0 (Ultralytics)                                               |
| **Framework**     | `ultralytics` Python package                                          |
| **Availability**  | ✅ Available (needs `pip install ultralytics`)                         |
| **Strengths**     | Fast, accurate, well-documented, NMS-free option                      |
| **Limitations**   | AGPL license (copyleft); only COCO classes by default                  |
| **Why Selected**  | Best speed/accuracy balance for the RTX 3050; well-maintained          |

---

## 6. OCR

### 6.1 EasyOCR — **SELECTED**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Optical character recognition (80+ languages)                         |
| **Architecture**  | CRAFT detector + recognition network                                  |
| **Input**         | Image (any format)                                                    |
| **Output**        | Bounding boxes + text + confidence per text region                    |
| **Languages**     | 80+ including Hindi, Urdu, Bengali, Tamil, and many more              |
| **VRAM**          | ~300-500 MB (depends on languages loaded)                             |
| **License**       | Apache 2.0                                                            |
| **Framework**     | `easyocr` Python package                                              |
| **Availability**  | ✅ Available (needs `pip install easyocr`)                             |
| **Strengths**     | Wide language support; good on scene text; Apache 2.0 license          |
| **Limitations**   | Slower than PaddleOCR on large images; detection can miss small text   |
| **Why Selected**  | Excellent Indian language support; permissive license; simple API       |

### 6.2 PaddleOCR — ALTERNATIVE

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | OCR (80+ languages)                                                   |
| **License**       | Apache 2.0                                                            |
| **Status**        | Alternative — faster but requires PaddlePaddle framework               |

---

## 7. Explainable AI (XAI) / Heatmaps

### 7.1 Captum (Grad-CAM compatible) — **SELECTED**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Attribution maps for PyTorch models                                   |
| **Techniques**    | GradCAM, Integrated Gradients, SHAP, Occlusion, etc.                 |
| **License**       | BSD-3-Clause                                                           |
| **Framework**     | PyTorch (via `captum` package)                                        |
| **Availability**  | Needs `pip install captum`                                             |
| **Why Selected**  | Official PyTorch attribution library; comprehensive technique set      |

### 7.2 Error Level Analysis (ELA) — **BUILT-IN (Custom)**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | JPEG compression artifact analysis                                    |
| **Input**         | Image file                                                            |
| **Output**        | ELA difference heatmap                                                |
| **Dependencies**  | Pillow, NumPy (already available)                                     |
| **License**       | N/A (custom implementation)                                           |
| **Limitations**   | Only meaningful for JPEG; re-saved images reduce effectiveness         |

### 7.3 Noise Analysis — **BUILT-IN (Custom)**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Noise pattern consistency analysis                                    |
| **Dependencies**  | OpenCV, NumPy, scikit-image (all available)                           |
| **License**       | N/A (custom implementation)                                           |

---

## 8. Face Detection

### 8.1 YOLO26s (Face mode) or RetinaFace — **PLANNED**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Task**          | Face detection and landmark extraction                                |
| **Status**        | Will evaluate YOLO-face models vs RetinaFace at integration time      |
| **Note**          | Face detection is used to identify faces for forensic analysis, not for facial recognition / identification |

---

## 9. Image Similarity / Hashing

### 9.1 Built-in Hash Functions — **SELECTED**

| Field             | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| **Algorithms**    | pHash, dHash, aHash, wHash                                           |
| **Dependencies**  | `imagehash` Python package (needs install) + Pillow                   |
| **License**       | BSD                                                                    |
| **Use in TruthLens** | Duplicate detection, near-duplicate matching, provenance linking   |

---

## VRAM Budget Planning (RTX 3050 Laptop — 4 GB)

> Models CANNOT all be loaded simultaneously. The orchestrator must manage sequential loading.

| Model                          | Approx. VRAM | Load Priority |
| ------------------------------ | ------------- | ------------- |
| YOLO26s (object detection)     | ~500 MB       | Early         |
| SigLIP2 (authenticity)         | ~400 MB       | Early         |
| CommunityForensics ViT         | ~200 MB       | Early         |
| EasyOCR                        | ~400 MB       | On-demand     |
| XLM-RoBERTa (language)         | ~550 MB       | After OCR     |
| Qwen2.5-VL-3B (Q4 quantized)  | ~1,900 MB     | Last, alone   |
| Captum (runs on existing model)| ~0 extra      | With model    |
| **Total if all loaded**        | **~3,950 MB** | —             |

**Strategy**: Load models in groups, unload before loading next group:
1. **Group A** (parallel): SigLIP2 + CommunityForensics ViT + YOLO26s ≈ 1,100 MB
2. **Group B** (sequential): EasyOCR → XLM-RoBERTa ≈ 950 MB
3. **Group C** (solo): Qwen2.5-VL-3B (quantized) ≈ 1,900 MB

---

## Model Installation Requirements

```bash
# Required packages not yet installed
pip install easyocr ultralytics captum imagehash grad-cam
pip install qwen-vl-utils  # For Qwen2.5-VL preprocessing
pip install bitsandbytes    # For int4 quantization (if using transformers)
```

---

## Versioning & Reproducibility

Every model prediction is logged with:
- Model ID (Hugging Face repo or local path)
- Model version / commit hash
- Timestamp (UTC)
- Inference runtime (ms)
- Device used (cpu / cuda:0)
- Input image SHA-256 hash
- Raw output (logits / probabilities)
- Error message (if any)

This ensures full reproducibility and auditability of every analysis.
