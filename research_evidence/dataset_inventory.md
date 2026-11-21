# TruthLens Dataset Inventory

> **Document Status:** Cryptographically Audited & Traced to Ingestion Manifests
> **Source Directory:** `datasets/external/`
> **Ingestion Manifests:** `datasets/external/tiny_genimage/ingestion_manifest.csv`, `datasets/external/ai_image_detection_benchmark_2026_10/ingestion_manifest.csv`, `datasets/external/wildaigc_bench/ingestion_manifest.csv`

---

## 1. Master Ingested Benchmark Summary

TruthLens ingested and cryptographically audited three major external image datasets, totaling **$N = 29,640$ images** across 16 distinct generator architectures:

| Benchmark Name | Directory | Total Images | Authentic (Real) | Synthetic (AI) | Generators / Modalities | Format Distribution | Primary License |
|---|---|---|---|---|---|---|---|
| **Tiny-GenImage** | `datasets/external/tiny_genimage/` | **25,006** | 12,506 | 12,500 | 8 generators: BigGAN, Cascade Diffusion, DeepFloyd-IF, GLIDE, LDM, Midjourney v4, SD 1.4, SD 1.5 | Real: 100% JPEG<br>AI: 100% PNG | CC BY 4.0 / ImageNet Research |
| **Frontier Benchmark 2026-10** | `datasets/external/ai_image_detection_benchmark_2026_10/` | **2,976** | 984 | 1,992 | 8 modern generators: DALL-E 3, Flux.1-dev, Flux.1-schnell, Midjourney v6, Recraft v3, SD 3 Medium, SD 3.5 Large, SD 3.5 Turbo across 4 compression tiers | Real: Mixed JPEG/PNG<br>AI: Mixed JPEG/PNG/WebP | CC BY 4.0 |
| **WildAIGC-Bench** | `datasets/external/wildaigc_bench/` | **1,658** | 829 | 829 | In-the-wild digital media captures across diverse web sources | Real: JPEG/PNG<br>AI: JPEG/PNG | CC BY-NC 4.0 |
| **Total Ecosystem** | — | **29,640** | **14,319** | **15,321** | **16 generator families + 4 compression tiers** | Multi-container | Verified Academic |

---

## 2. Cryptographic Integrity & Deduplication Audit

- **Hash Verification:** 100% of the 29,640 images were verified via SHA-256 hashes against source checksums; zero missing or unreadable files.
- **Duplicate Detection:** Exact SHA-256 and perceptual dHash audits identified duplicate images across multi-platform exports. Duplicates were clustered into unified sample groups (`experiments/create_evaluation_splits.py`) to prevent evaluation leakage across train, val, calib, and test partitions.
- **Corrupted / Invalid Files:** 0 corrupted images detected; Pillow decoding validated on all 29,640 items.

---

## 3. Split Governance Allocations

Partitions were assigned using group-aware stratification (`experiments/create_evaluation_splits.py`), guaranteeing zero overlap in prompt identities or generator subsets:

| Partition | Total Images | Tiny-GenImage | Frontier Benchmark 2026-10 | WildAIGC-Bench | Purpose & Governance |
|---|---|---|---|---|---|
| **Approved Training** (`experiments/manifests/phase4_training_manifest.csv`) | **6,995** | 6,995 (3,515 Real, 3,480 AI) | 0 | 0 | Model fine-tuning and representation learning. Strictly isolated from evaluation. |
| **Validation** | **6,255** | 3,464 | 1,791 | 1,000 | Hyperparameter tuning, early stopping, and checkpoint selection. |
| **Calibration Pool** | **13,000** | 10,000 | 2,000 | 1,000 | Post-hoc temperature scaling parameter estimation. Disjoint from test. |
| **Locked Test Partition** | **10,385** | 7,000 (3,500 Real, 3,500 AI) | 2,091 (730 Real, 1,361 AI) | 1,294 (647 Real, 647 AI) | Final benchmark evaluation. Sealed until candidate selection. |

---

## 4. Container Format Shortcut Documentation

In Tiny-GenImage:
- Authentic images originate from ImageNet validation subsets and are stored exclusively as JPEG files.
- Synthetic images originate from benchmark generation pipelines and are stored exclusively as PNG files.
- Container metadata alone yields an ROC-AUC of 1.0000.
- Resolved in Phase 7 via matched in-memory augmentation (dynamic transcoding and quality scaling), reducing metadata ROC-AUC to 0.5028.
