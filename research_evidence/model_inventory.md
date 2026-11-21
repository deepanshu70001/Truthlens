# TruthLens Model Inventory

> **Document Status:** Verified Against Repository & Experimental Logs  
> **Inspection Date:** 2026-10-20  
> **Primary References:** `docs/MODELS.md`, `results/ml/`, `backend/app/services/authenticity/`, `scripts/ml/train_baselines.py`

---

## 1. Overview of Evaluated & Integrated Models

TruthLens evaluates and incorporates multiple neural network architectures across computer vision, vision-language modeling, and optical character recognition. To maintain strict scientific rigor, models are divided into:
1. **Integrated Primary Models** (active in the running system)
2. **Empirically Benchmarked Baselines** (trained/evaluated on the CIFAKE benchmark subset)
3. **Investigated & Alternative Models** (evaluated for feasibility or deferred due to hardware constraints)

---

## 2. Comprehensive Model Master Table

| Model Identifier / Name | Primary Task | Backbone Architecture | Total Params | Trainable Params | VRAM (Inference) | Source / License | Repository Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`prithivMLmods/Deepfake-Real-Class-Siglip2`** | Synthetic / AI Image Detection | SigLIP2-Base (`google/siglip2-base-patch16-224`) | ~86M | N/A (Pretrained) | ~400 MB (fp32) | Hugging Face (Apache 2.0) | **Integrated (Primary)** |
| **`cnn_scratch`** | AI vs Real Detection Baseline | 4-layer Conv2D + BatchNorm + MaxPool + FC | 356,097 | 356,097 | ~80 MB | Custom (`scripts/ml/`) | **Benchmarked Baseline** |
| **`resnet18_frozen`** | AI vs Real Detection Baseline | Torchvision ResNet-18 (ImageNet-1k weights) | 11,177,025 | 513 (Linear head) | ~150 MB | PyTorch (BSD-3-Clause) | **Benchmarked Baseline** |
| **`siglip_frozen`** | AI vs Real Detection Baseline | `google/siglip-base-patch16-224` | 92,884,993 | 769 (Linear head) | ~450 MB | Hugging Face (Apache 2.0) | **Benchmarked Baseline** |
| **`buildborderless/CommunityForensics-DeepfakeDet-ViT`** | Secondary Ensemble Detection | ViT-Small (`timm/vit_small_patch16_384`) | ~22M | N/A (Pretrained) | ~200 MB | Hugging Face (MIT) | Investigated Alternative |
| **`ultralytics/yolov8n.pt`** | Scene Object & Entity Detection | YOLOv8 Nano (Anchor-free CNN) | ~3.2M | N/A (COCO-pretrain) | ~120 MB | Ultralytics (AGPL-3.0) | **Integrated (Primary)** |
| **EasyOCR (CRAFT + CRNN)** | Multilingual Scene Text Extraction | CRAFT (detection) + ResNet-LSTM-CTC (recognition) | ~15M | N/A (Pretrained) | ~350 MB | JaidedAI (Apache 2.0) | **Integrated (Primary)** |
| **`Qwen/Qwen2.5-VL-3B-Instruct`** | Contextual Scene & Claim Reasoning | Multimodal Vision-Language Transformer | ~3.0B | N/A (Pretrained) | ~6.5 GB (fp16) / ~3.8 GB (int4) | Alibaba Cloud (Apache 2.0) | Investigated / Service Drafted (Deferred) |
| **XLM-RoBERTa / FastText** | Language & Script Identification | Multilingual Masked Language Model / n-gram | ~278M | N/A (Pretrained) | ~300 MB | Meta / Hugging Face (MIT) | Investigated Alternative |

---

## 3. Deep Dive: Authenticity & Deepfake Detection Models

### 3.1 Primary Model: SigLIP2 (`prithivMLmods/Deepfake-Real-Class-Siglip2`)
- **Role:** Primary inference model in `backend/app/services/authenticity/detector.py`.
- **Input Dimensions:** RGB image resized and normalized to $224 \times 224 \times 3$.
- **Output:** Softmax distribution over binary classes (`Real` vs. `Fake`).
- **Strengths:** Employs sigmoid loss for language-image pretraining (SigLIP2), yielding superior zero-shot representation of subtle high-frequency generator artifacts compared to standard CLIP.
- **Identified Failure Modes & Concept Drift:** Highly vulnerable to emerging generative pipelines not represented in training (e.g. FLUX.1, SD3, Midjourney v6). Compression, aggressive JPEG re-encoding, and social platform downscaling attenuate high-frequency signals.

### 3.2 Benchmarked Baselines (Trained on Local CIFAKE Subset)
To establish rigorous empirical grounding for the research paper, three models were trained on an identical split (800 train, 200 validation, 200 test) using `scripts/ml/train_baselines.py`:

1. **`cnn_scratch` (Convolutional Baseline):**
   - 4 convolutional blocks (channels: 32 → 64 → 128 → 256) with Batch Normalization, ReLU activation, $2 \times 2$ MaxPooling, and a dense classification head.
   - Trained for 20 epochs with Adam optimizer ($\text{lr}=10^{-3}$).
   - Reached **89.5% test accuracy** and **0.9585 ROC-AUC** in 500 seconds.
2. **`resnet18_frozen` (Transfer Learning Baseline):**
   - Standard ResNet-18 initialized with ImageNet-1k weights. Feature backbone frozen; single linear classification layer trained.
   - Trained for 10 epochs. Reached **81.5% test accuracy** and **0.9255 ROC-AUC** in 382 seconds.
3. **`siglip_frozen` (Foundation Model Linear Probe):**
   - `google/siglip-base-patch16-224` vision encoder with frozen transformer weights, tuning only a 769-parameter linear projection head.
   - Trained for 50 epochs. Reached **87.0% test accuracy** and **0.9613 ROC-AUC** in 73 seconds (GPU-accelerated cached features).

---

## 4. Hardware Constraints and Model Sizing Tradeoffs

The development environment operates on an **NVIDIA GeForce RTX 3050 Laptop GPU with 4,096 MB VRAM**.

```
VRAM Budget Breakdown (Total Capacity: 4,096 MB):
├── Windows OS & Display Overhead:      ~700 MB
├── Available Workstation Headroom:     ~3,396 MB
├── SigLIP2 Authenticity Model:         ~400 MB (Active Stage 3 & 7)
├── YOLOv8n Object Detector:            ~120 MB (Active Stage 6)
└── EasyOCR Text Recognizer:            ~350 MB (Active Stage 5)
```

**Reason for Deferring Qwen2.5-VL-3B:**  
A 3-billion-parameter Vision-Language Model requires a minimum of 6.5 GB VRAM in half-precision (fp16) or ~3.8 GB when quantized to 4-bit (bitsandbytes int4). Loading Qwen2.5-VL concurrently with the FastAPI server and operating system exceeded available physical VRAM, causing CUDA Out-Of-Memory (OOM) faults. Consequently, VLM context reasoning was architected as a decoupled optional service rather than a mandatory stage in the primary 19-stage pipeline.
