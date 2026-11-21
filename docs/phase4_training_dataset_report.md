# TruthLens Phase 4: Training Dataset Preparation & Validation Report

**Date:** 2026-11-21 12:55:34 UTC  
**Target Architecture:** Vision Transformer (`google/siglip-base-patch16-224`)  
**Hardware Target:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Status:** Complete and Verified (Zero Leakage)  

---

## 1. Executive Summary & Policy Overview

In preparation for Phase 5 (SigLIP head re-training), Phase 4 audited and prepared a fully reproducible, leakage-free training dataset.

### Core Governance & Eligibility Decisions:
1. **Tiny-GenImage Reserve Pool Audit**:
   - Total reserve pool: **9,994 samples**.
   - **`APPROVED_TRAINING`**: **6,995 samples** (3,515 Real ImageNet-Camera + 3,480 AI from ADM, BigGAN, SD 1.5, GLIDE).
   - **`EXCLUDED_PROTOCOL`**: **2,998 samples** (`Midjourney`: 507, `Wukong`: 491, `VQDM`: 2,000) held out to preserve the strict cross-generator generalization protocol.
   - **`EXCLUDED_DUPLICATE`**: **1 sample** (SHA-256 `3eba56bc...` redundant ImageNet nature photograph).
   - **Class Balance**: 3,480 AI (49.75%) vs 3,515 Real (50.25%) — **near-perfect 1:1 balance**.
2. **Frontier Benchmark Policy**: **0 samples in training**. All 2,976 samples remain strictly allocated to evaluation and calibration.
3. **WildAIGC-Bench Policy**: **0 samples in training**. Formally marked `NOT APPROVED FOR AI-GENERATION SUPERVISED TRAINING` due to journalistic claims, spliced visual targets, and licensing disclaimers.
4. **Strict Partition Protection**: **Zero hash or group overlaps** exist between the approved training manifest and existing calibration, validation, or locked test splits.

---

## 2. Approved Training Dataset Quality Metrics

Summary of the approved training dataset ([phase4_training_manifest.csv](..\experiments\manifests\phase4_training_manifest.csv)):

| Attribute | Value | Status |
|---|---|---|
| **Total Approved Images** | **6,995** | Verified 100% Readable |
| **AI-Generated Images** | 3,480 (49.75%) | Latent & Pixel Diffusion, GAN |
| **Likely Authentic Images** | 3,515 (50.25%) | ImageNet-Camera Nature Photographs |
| **Class Balance Ratio** | 0.99 : 1.00 | **Optimal (No Class Imbalance)** |
| **Image Width Range** | 128 px to 500 px (Mean: 202.9 px, Median: 128 px) | Scaled to 224x224 during training |
| **Image Height Range** | 128 px to 500 px (Mean: 198.5 px, Median: 128 px) | Scaled to 224x224 during training |
| **Aspect Ratio Range** | 0.44 to 2.27 (Mean: 1.05, Median: 1.00) | Center cropped to 224x224 |
| **File Formats** | JPEG (86.0%), PNG (14.0%) | 0 Corrupted Images |
| **Duplicate Count** | 0 in approved manifest | 1 duplicate purged during audit |

### Generator Breakdown in Approved Training Set:
- **ImageNet-Camera** (Real): 3,515 samples
- **GLIDE** (AI - Latent Diffusion): 2,000 samples
- **ADM** (AI - Pixel Diffusion): 502 samples
- **Stable Diffusion v1.5** (AI - Latent Diffusion): 496 samples
- **BigGAN** (AI - GAN): 482 samples

---

## 3. Tiny-GenImage Reserve Pool Eligibility Audit

Full audit table saved to [phase4_exclusion_audit.csv](..\experiments\reports\phase4_exclusion_audit.csv):

| Eligibility Decision | Sample Count | Percentage | Rationale |
|---|---|---|---|
| **`APPROVED_TRAINING`** | **6,995** | **70.0%** | Verified readable, clean in-domain generators and authentic ImageNet photographs with zero partition overlap. |
| **`EXCLUDED_PROTOCOL`** | **2,998** | **30.0%** | Reserved held-out generator architectures (`Midjourney`: 507, `Wukong`: 491, `VQDM`: 2,000) under `phase2_split_protocol.yaml` to ensure cross-generator testing remains strictly unseen. |
| **`EXCLUDED_DUPLICATE`** | **1** | **<0.1%** | Exact byte-duplicate instance of SHA-256 `3eba56bc...` nature photo. |
| **Total Audited** | **9,994** | **100.0%** | Exhaustive classification of all reserve instances. |

---

## 4. Hardware-Aware Training Configuration (RTX 3050 4GB VRAM)

Configuration saved to [phase4_training_config.yaml](..\experiments\configs\phase4_training_config.yaml):
- **Target Model**: Frozen `google/siglip-base-patch16-224` backbone with newly initialized linear head (`768 -> 1`).
- **Batch Size**: 32 per GPU forward step.
- **Gradient Accumulation**: 2 steps (Effective Batch Size: 64).
- **Precision**: `fp16` mixed precision (fits comfortably within 4 GB VRAM, using ~1.8 GB).
- **Optimizer**: AdamW ($lr = 1e-4$, weight decay = $1e-2$, cosine decay with warmup).
- **Early Stopping**: Monitored on `validation` split loss ($patience = 3$ epochs).

---

## 5. Figures & Deliverables Index

- Training Manifest: `experiments/manifests/phase4_training_manifest.csv`
- Exclusion Audit: `experiments/reports/phase4_exclusion_audit.csv`
- Dataset Quality Report: `experiments/reports/phase4_dataset_quality.json`
- Hardware Config: `experiments/configs/phase4_training_config.yaml`
- Validation Script: `experiments/validate_phase4_training_data.py`
- Visualizations (`experiments/reports/figures/`):
  - `phase4_class_distribution.png`
  - `phase4_generator_distribution.png`
  - `phase4_resolution_distribution.png`
  - `phase4_aspect_ratio_distribution.png`
  - `phase4_exclusion_breakdown.png`