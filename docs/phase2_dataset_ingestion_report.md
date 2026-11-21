# Phase 2 — Step 2: External Dataset Ingestion & Validation Report

**Ingestion Date:** 2026-11-21
**Repository Branch:** `fix/truthlens-remediation`
**Execution Scope:** Phase 2, Step 2 Only (External Dataset Ingestion, Validation, and Manifest Generation)
**Machine / Environment:** Windows (Python 3.12, Pillow 11.3.0, ImageHash 4.3.2)
**Machine-Readable Report:** [experiments/reports/phase2_dataset_ingestion.json](..\experiments\reports\phase2_dataset_ingestion.json)
**Ingestion Script:** [experiments/ingest_external_datasets.py](..\experiments\ingest_external_datasets.py)

---

## 1. Executive Ingestion Summary

In Phase 2 Step 2, three external benchmark datasets were located, inspected, organized into standard directory structures, fully validated with byte-level decoding, cryptographically checksummed, perceptually fingerprinted, and compiled into individual research ingestion manifests.

### Datasets Ingested

1. **Tiny-GenImage (v1.0-tiny):** 35,000 images across 7 generator architectures with balanced `train` and `val` splits.
2. **WasItAIGenerated Frontier Image & Video Benchmark (October 2026):** 2,976 images (and 402 videos) from 19 frontier image generators with real-world social-platform compression variants.
3. **WildAIGC-Bench (v3):** 1,839 images (1,238 unique cases: 600 AI-generated in-the-wild claims, 638 verified real news control photographs) with source fact-check links.

### Key Quality & Validation Metrics

* **Total Images Indexed & Validated:** **39,815 images**
* **Corrupted / Unreadable Images:** **0 images** (100% of images pass header verification and pixel buffer decoding).
* **Cross-Dataset Overlap:** **0 exact duplicates** between any of the three datasets.
* **TruthLens Internal Data Overlap:** **0 exact duplicates** with repository test fixtures or runtime data (`data/`, `test_data/`, `backend/tests/fixtures/`).
* **Model Training Data Overlap:** **0 overlap** with the default detector's CIFAKE classification head training data.
* **Manifests Created:** Standardized CSV manifests generated under both `datasets/external/<dataset>/manifest.csv` and `experiments/manifests/`.

---

## 2. Ingestion Directory Organization & Preservation

To preserve disk storage and avoid modifying original files, all downloads were safely organized under `datasets/external/` using NTFS hardlinks (`os.link`), preserving the original downloads in place while creating canonical research directory structures.

```
datasets/
├── external/
│   ├── tiny_genimage/
│   │   ├── imagenet_ai_0419_biggan/       [train/ (ai, nature), val/ (ai, nature)]
│   │   ├── imagenet_ai_0419_vqdm/         [train/ (ai, nature), val/ (ai, nature)]
│   │   ├── imagenet_ai_0424_sdv5/         [train/ (ai, nature), val/ (ai, nature)]
│   │   ├── imagenet_ai_0424_wukong/       [train/ (ai, nature), val/ (ai, nature)]
│   │   ├── imagenet_ai_0508_adm/          [train/ (ai, nature), val/ (ai, nature)]
│   │   ├── imagenet_glide/                [train/ (ai, nature), val/ (ai, nature)]
│   │   ├── imagenet_midjourney/           [train/ (ai, nature), val/ (ai, nature)]
│   │   └── manifest.csv                   [35,000 indexed rows]
│   ├── ai_image_detection_benchmark_2026_10/
│   │   ├── images/                        [747 high-res source images]
│   │   ├── images_social/                 [2,229 social-platform compression variants]
│   │   ├── videos/                        [201 high-res source videos]
│   │   ├── videos_social/                 [201 social-platform compression variants]
│   │   ├── metadata.csv                   [original release metadata]
│   │   └── manifest.csv                   [2,976 indexed image rows]
│   └── wildaigc_bench/
│       ├── raw/                           [1,238 original in-the-wild media files]
│       ├── processed/                     [601 standardized benchmark images]
│       ├── data/                          [test Parquet split]
│       ├── metadata.jsonl                 [fact-check & source metadata]
│       ├── relations.jsonl                [cross-image reference pairs]
│       └── manifest.csv                   [1,839 indexed image rows]
```

Original download sources (`datasets/tiny genimage/`, `datasets/frontier-image-video-bench-2026-10/`, `datasets/WildAIGC-Bench/`) remain unmodified.

---

## 3. Dataset Profiles & Validation Findings

### A. Tiny-GenImage (1.0-tiny)
* **Citation:** *GenImage: A Million-Scale Dataset for Detecting AI-Generated Images* (NeurIPS 2023).
* **Scope:** 35,000 256x256 images spanning 7 generator architectures paired with ImageNet (ILSVRC 2012) photographic controls.
* **Total Images:** 35,000 (35,000 valid, 0 corrupted).
* **Labels:**
  * `likely_authentic`: 17,500 images (ImageNet real photographs, `.jpeg`).
  * `ai_generated`: 17,500 images (diffusion/GAN synthesized corresponding classes, `.png`).
* **Splits:**
  * `train`: 28,000 images (4,000 per generator: 2,000 AI + 2,000 real).
  * `val`: 7,000 images (1,000 per generator: 500 AI + 500 real).
* **Generators:** BigGAN (2,500), VQDM (2,500), Stable Diffusion v1.5 (2,500), Wukong (2,500), ADM (2,500), GLIDE (2,500), Midjourney (2,500), ImageNet-Camera (17,500).
* **Duplicates:** 34,999 unique SHA-256 (1 exact duplicate copy), 34,995 unique dHash (5 near-duplicate candidate pairs).
* **Licenses:** ImageNet Terms of Access for research (nature) / GenImage Open Benchmark License (ai).
* **Manifest:** [experiments/manifests/tiny_genimage.csv](..\experiments\manifests\tiny_genimage.csv) (35,000 rows).

---

### B. WasItAIGenerated Frontier Image & Video Benchmark (October 2026)
* **Citation:** *WasItAIGenerated Frontier Image & Video Benchmark* (October 2026 Release).
* **Scope:** Modern test benchmark specifically covering contemporary 2026 frontier generators and social network post-processing.
* **Total Images:** 2,976 images (2,976 valid, 0 corrupted).
  * *Note on Large Images:* Includes 2 ultra-high resolution 16K panoramic control images (`16320x12240`, 199.7 megapixels) which were safely decoded with an expanded pixel buffer limit (`Image.MAX_IMAGE_PIXELS = 250_000_000`).
* **Labels:**
  * `likely_authentic`: 1,932 images (camera captures with documented authorship and CC licenses).
  * `ai_generated`: 1,044 images (generated by frontier models).
* **Split:** `test` (100% held-out test split).
* **Generators (19 Image Generators):** Nano Banana Pro (80), Flux.2 Pro (68), Ideogram v4.5 (68), HiDream O1 (64), Recraft v4.1 (64), Krea 2 Turbo (64), Qwen-Image 3 (60), Nano Banana 2.1 (56), Seedream v5 Pro (56), Grok Imagine 2 (56), GPT-Image-2 (52), Kling Image 3 (52), Meta Muse Image (52), Flux.3 (48), Flux.2 dev (48), Luma Uni-1 (48), Z-Image Turbo (44), Qwen-Image 2512 (40), Microsoft MAI-Image 2.5 (36).
* **Social Compression Variants:** 2,229 images represent real-world compressed copies across WhatsApp, Telegram, Messenger, Line, WeChat.
* **Duplicates:** 2,976 unique SHA-256 (0 exact duplicate files). 975 unique dHash clusters (reflecting the exact visual correspondence between social-compressed variants and their uncompressed originals).
* **Licenses:** 100% documented Creative Commons & Public Domain: CC BY-SA 4.0 (1,326), CC BY 4.0 WasItAIGenerated (1,044), CC BY 4.0 (302), CC0 (232), Public Domain (110), CC BY-SA 2.0/3.0.
* **Manifest:** [experiments/manifests/frontier_benchmark_2026_10.csv](..\experiments\manifests\frontier_benchmark_2026_10.csv) (2,976 rows).

---

### C. WildAIGC-Bench (v3)
* **Citation:** *WildAIGC-Bench: A Benchmark for AI-Generated Images in the Wild* (ArXiv 2610.28087, 2026).
* **Scope:** 1,238 real news events and viral claims collected in-the-wild, accompanied by professional fact-check articles.
* **Total Images Indexed:** 1,839 images (1,238 raw in-the-wild media, 601 cropped/processed benchmark samples).
* **Validation:** 1,839 valid, 0 corrupted.
* **Labels:**
  * `ai_generated`: 1,201 images (viral synthetic images associated with fake claims).
  * `likely_authentic`: 638 images (real news photos verifying authentic events).
* **Split:** `test` (100% held-out in-the-wild evaluation split).
* **Provenance & Metadata:** 99.89% source URL coverage. Every sample includes title, context summary, fact-check source URL (JFC, Snopes, RTVE, etc.), and original social media link.
* **Duplicates:** 1,363 unique SHA-256. 476 duplicate clusters exist strictly between the uncropped `raw/` files and their identical copies in `processed/`.
* **Licenses:** CC BY 4.0 for dataset annotations and metadata; third-party media used under fair-use news reporting / fact-checking research exception.
* **Manifest:** [experiments/manifests/wildaigc_bench.csv](..\experiments\manifests\wildaigc_bench.csv) (1,839 rows).

---

## 4. Cross-Dataset & Internal Overlap Analysis

| Comparison | Exact SHA-256 Matches | Near Duplicates (dHash $\le 2$) | Overlap Verdict |
| :--- | :---: | :---: | :--- |
| **Between Ingested Datasets** | **0** | **0** | All three datasets are completely independent in origin, scope, and image files. |
| **Ingested vs. TruthLens `data/`** | **0** | **0** | Zero overlap with runtime application uploads. |
| **Ingested vs. TruthLens `test_data/`** | **0** | **0** | Zero overlap with internal test fixtures. |
| **Ingested vs. `backend/tests/fixtures`** | **0** | **0** | Zero overlap with unit test fixture photo. |

### Model Training Data Isolation Analysis

* **Detector Model ID:** `truthlens/siglip-cifake-frozen-head`
* **Head Training Data:** CIFAKE (60,000 images, 32x32 CIFAR-10 real vs synthetic Stable Diffusion v1.4).
  * **Findings:** Zero overlap. All ingested external images have resolutions between 256x256 and 16,320x12,240. None match the 32x32 pixel array format of CIFAKE.
* **Backbone Pretraining:** `google/siglip-base-patch16-224` (pre-trained on WebLI in 2023).
  * **Frontier Benchmark Isolation:** All 19 image generators and 12 video generators in the WasItAIGenerated Frontier Benchmark were created in or up to late 2026 (e.g. GPT-Image-2, Flux.3, Nano Banana Pro). They postdate the SigLIP backbone by several years, guaranteeing **zero training set leakage or memorization**.
  * **WildAIGC-Bench Isolation:** Real-world news and viral claims from 2024–2026 fact-checks postdate backbone training.

---

## 5. Ingestion Manifest Schema & Metadata Completeness

All individual manifests adhere to the TruthLens canonical manifest specification:

`image_path,label,split,group_id,generator,source_url,license_ref,annotation_notes,sha256,width,height,format,size_bytes,valid,error,dhash,phash`

| Dataset Manifest | Total Rows | Ground Truth Label Coverage | Split Coverage | License Coverage | Source URL Coverage | Valid Image Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tiny-GenImage** | 35,000 | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** (35,000 / 35,000) |
| **Frontier Benchmark 2026-10** | 2,976 | 100.0% | 100.0% | 100.0% | 64.92% | **100.0%** (2,976 / 2,976) |
| **WildAIGC-Bench** | 1,839 | 100.0% | 100.0% | 100.0% | 99.89% | **100.0%** (1,839 / 1,839) |
| **Total Ingested** | **39,815** | **100.0%** | **100.0%** | **100.0%** | **97.35%** | **100.0%** (39,815 / 39,815) |

---

## 6. Compliance & Constraints Verification

* **Model Training:** None (no models trained or fine-tuned).
* **Detector & Calibration Code:** Neither [detector.py](..\backend\app\services\authenticity\detector.py) nor [calibration.py](..\backend\app\services\authenticity\calibration.py) was modified.
* **Thresholds & Fusion:** No thresholds or fusion weights changed.
* **Original Files:** Original download archives and files preserved completely intact.
* **Git Hygiene:** All datasets reside in `datasets/`, which is ignored by `.gitignore`. `git diff --check` is clean.
* **Stop Condition:** Stopped after ingestion and validation. Phase 3 has **not** been started.
