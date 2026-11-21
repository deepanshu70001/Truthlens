# TruthLens Implementation Summary for Research Paper Drafting

> **Document Status:** High-Density Author Reference  
> **Inspection Date:** 2026-10-20  
> **Target Use:** Rapid factual reference while drafting paper sections

---

## 1. Quick Facts & Numerical Cheat Sheet

| Parameter | Value / Metric | Source / Reference |
| :--- | :--- | :--- |
| **Project Full Title** | *TruthLens: AI System for Image Verification, Analysis and Virality Estimation Including Social Impact* | Repository Root |
| **Core Architecture** | Asynchronous 19-Stage Orchestration Pipeline with Fail-Soft Boundaries | `backend/app/services/orchestrator.py` |
| **Primary Authenticity Model** | SigLIP2 (`prithivMLmods/Deepfake-Real-Class-Siglip2`) | `backend/app/services/authenticity/` |
| **Best Baseline ROC-AUC** | **0.9613** (SigLIP Frozen Linear Probe) | `results/ml/siglip_frozen.json` |
| **Best Baseline Accuracy** | **89.50%** (Custom 4-Layer CNN from Scratch) | `results/ml/cnn_scratch.json` |
| **Best Baseline Precision** | **92.47%** (CNN from Scratch; 7% False Positive Rate) | `results/ml/cnn_scratch.json` |
| **Trainable Parameter Efficiency** | SigLIP probe trained only **769 parameters** (vs. 356k for CNN) | `results/ml/comparison.csv` |
| **Fastest Training Time** | **73.0 seconds** (SigLIP over 50 epochs using cached embeddings) | `results/ml/siglip_frozen.json` |
| **Benchmark Dataset Split** | CIFAKE Subset: 800 train, 200 validation, 200 test (Seed: `20261014`) | `data/external/cifake/manifest.json` |
| **Explainable AI (XAI)** | Layer-Targeted Grad-CAM on Transformer Visual Encoder | `backend/app/services/xai/gradcam.py` |
| **Optical Character Recognition** | EasyOCR (CRAFT + CRNN) with Indic & English multi-script support | `backend/app/services/ocr/service.py` |
| **Object Detection Backbone** | YOLOv8n (`ultralytics/yolov8n.pt`, 3.2M parameters) | `backend/app/services/vision/detector.py` |
| **Regional Coverage** | 36 Indian States and Union Territories (Census 2011 + TRAI 2024 data) | `backend/app/services/regional/engine.py` |
| **Virality Data Declaration** | Explicit `DATA_UNAVAILABLE` status flag (zero fabricated predictions) | `backend/app/services/virality/engine.py` |
| **Hardware Constraint** | 4,096 MB VRAM (NVIDIA GeForce RTX 3050 Laptop GPU) | Host System Specs |

---

## 2. Section-by-Section Paper Drafting Guide

### Abstract
- Open with the dual challenges of multimodal disinformation: generative diffusion artifacts and deceptive out-of-context image recycling ("cheapfakes").
- Highlight the 19-stage modular architecture.
- Cite the empirical baseline comparison: SigLIP achieving **0.9613 ROC-AUC** with only 769 trainable parameters, alongside the 4-layer CNN achieving **89.5% accuracy** on native pixel grids.
- Emphasize the Indian regional exposure mapping across 36 states and the epistemic honesty of declaring `DATA_UNAVAILABLE` on proprietary forward cascades.

### Introduction
- Establish the threat model: synthetic media combined with encrypted WhatsApp/Telegram viral forwarding in multilingual democracies.
- Clarify the paper's central thesis: **Single-model classifiers are insufficient; robust verification requires layered evidence fusion across visual, forensic, linguistic, and provenance channels.**

### Methodology (Sections III & IV)
- Detail the 19-stage pipeline from cryptographic SHA-256 validation through Error Level Analysis (Q=95), multi-perceptual hashing (pHash/dHash/aHash), EasyOCR, and Google Fact Check querying.
- Present the mathematical formulation of the Multi-Criteria Evidence Fusion Engine:
  $$S = \frac{\sum w_i \cdot \left(\frac{1 + p_i \cdot c_i}{2}\right)}{\sum w_i}$$
- Explain the Grad-CAM visual interpretability mechanism applied to transformer attention blocks.

### Results & Discussion (Section V)
- Feature **Table 1** (Comparative Baselines) and **Figures 3 & 4** (ROC curves and loss convergence).
- Address the *High-Frequency Resolution Paradox*: why CNN-from-scratch on native $32 \times 32$ pixels attained 89.5% accuracy compared to ResNet-18 (81.5%), which suffered from interpolation smoothing upon upsampling.
- Emphasize the compute-efficiency narrative: SigLIP converged in 73 seconds.

### Limitations & Ethics (Section VI & VII)
- Acknowledge the 4 GB VRAM ceiling requiring sequential execution and deferral of 3B+ parameter VLMs.
- Disclose the 1,200-sample CIFAKE baseline scale and discuss generative concept drift (FLUX.1, Midjourney v6).
- Reiterate ethical safeguards against automation bias and false-positive chilling of citizen journalism.
