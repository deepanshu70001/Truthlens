# Phase C Baseline Environment and Partition Audit

**Document Version:** `1.0.0`  
**Date:** 2026-11-24  
**Repository Branch:** `fix/truthlens-remediation`  
**Git HEAD:** `d9379439fb29f90515f8f1077e572ae9cd84b295`  
**Audit Purpose:** Comprehensive scientific baseline verification, runtime environment audit, model checkpoint hash verification, and partition inventory for TruthLens Package C.

---

## 1. Executive Summary

TruthLens Package C establishes reproducible, statistically defensible scientific evaluation evidence for TruthLens components. In accordance with master execution rules:
1. Production behavior, SigLIP weights, decision thresholds, and evidence fusion (`e11.conservative.v1`) remain strictly preserved.
2. Protected evaluation partitions (calibration and reserve partitions) are strictly guarded against unauthorized unsealing or data leakage.
3. Synthetic demonstration datasets (`datasets/synthetic_demo/`) remain strictly quarantined and excluded from all model and detector evaluation.
4. All evaluation statuses strictly adhere to the approved status vocabulary.

---

## 2. Runtime Environment & Hardware Configuration

The local execution environment was comprehensively inspected and verified:

| Component | Verified Version / Specification | Runtime Context / Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Enterprise (Build 26100) | Local development workstation |
| **Python** | `3.12.10` | 64-bit CPython |
| **PyTorch** | `2.7.1+cu118` | CUDA acceleration enabled (`torch.cuda.is_available() == True`) |
| **CUDA Runtime** | `11.8` | Driver/CUDA version verified via PyTorch CUDA interface |
| **Primary GPU** | `NVIDIA GeForce RTX 3050 Laptop GPU` | Compute Capability 8.6 |
| **Dedicated VRAM** | `4.00 GB` (4,294,705,152 bytes) | Dedicated GDDR6 memory |
| **Torchvision** | `0.22.1+cu118` | Hardware-accelerated vision operators |
| **Transformers** | `4.57.6` | Hugging Face offline-mode model runtime |
| **OpenCV (`cv2`)** | `4.12.0` | Classical image forensics & frequency processing |
| **PIL (`Pillow`)** | `11.3.0` | Image ingestion, EXIF correction & decode verification |
| **Scikit-learn** | `1.5.0` | Metrics, confusion matrices, ROC/PR curves & calibration |
| **SciPy** | `1.16.3` | Statistical significance testing & confidence interval routines |
| **EasyOCR** | `1.7.2` | Primary OCR engine (CRAFT text detector + PyTorch recognizer) |
| **PyTesseract** | `0.3.13` | Tesseract wrapper; bound to system Tesseract 5.4.0 binary |
| **Tesseract Engine** | `5.4.0.20240606` | Located at `C:\Program Files\Tesseract-OCR\tesseract.exe`; language packs: `eng`, `hin`, `osd` |
| **C2PA Python** | `0.39.0` | Official Coalition for Content Provenance and Authenticity SDK (Rust core `0.92.0`) |
| **Ultralytics (YOLO)** | `8.4.216` | YOLOv8n object detection engine |

---

## 3. Model Checkpoint Identity & Cryptographic Integrity

All production and baseline models were audited with SHA-256 cryptographic hashes:

| Model / Checkpoint | File Path | File Size | SHA-256 Checksum | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **SigLIP Linear Head** | `models/baselines/siglip_frozen_head.pt` | 5,053 bytes | `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538` | **Bit-exact match** to production baseline specification |
| **YOLOv8n Baseline** | `models/baselines/yolov8n.pt` | 6,549,765 bytes | `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95` | **Verified** official Ultralytics YOLOv8n weights |
| **Feature Vectors Archive** | `experiments/reports/phase9_feature_vectors.npz` | 56,274,384 bytes | `a1b9d3530f9555ba4b67cc4a5d2b3793ebac7804bb150eb0a2856da4efb00ed9` | **Verified** approved pre-fit feature archive |
| **Language Classifier** | `papluca/xlm-roberta-base-language-detection` | HuggingFace Cache | Pinned revision: `9865598389ca9d95637462f743f683b51d75b87b` | **Verified** offline cache with 20 supported languages |
| **SigLIP Vision Backbone** | `google/siglip-base-patch16-224` | HuggingFace Cache | Pinned revision: `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | **Verified** official Google SigLIP backbone |

---

## 4. Dataset Partition Inventory & Governance Boundaries

In strict compliance with `docs/phase2_split_protocol.md`, `docs/phase10_blocker_report.md`, and `docs/r10_data_governance_audit.md`, dataset partitions are inventoried and governed as follows:

```
+---------------------------------------------------------------------------------------------------+
|                                 TRUTHLENS DATASET PARTITION MAP                                   |
+---------------------------------------------------------------------------------------------------+
| [AUTHORIZED TRAINING]       | 6,995 images (3,515 authentic, 3,480 AI)                            |
|                             | Source: Tiny-GenImage training split                                |
|                             | Status: OPEN for training, research analysis & calibration research |
+-----------------------------+---------------------------------------------------------------------+
| [DEVELOPMENT VALIDATION]    | 6,436 images total                                                  |
|                             | - Tiny-GenImage: 5,958 images (3,410 authentic, 2,548 AI)           |
|                             | - Frontier Benchmark (Oct 2026): 297 images (193 authentic, 104 AI) |
|                             | - WildAIGC-Bench v3: 181 images (63 authentic, 118 AI)              |
|                             | Status: RESTRICTED to policy exploration and diagnostic validation   |
+-----------------------------+---------------------------------------------------------------------+
| [HISTORICAL HELD-OUT TEST]  | 10,385 images total (Phase 3 evaluation)                            |
|                             | - Tiny-GenImage: 7,000 images (3,500 authentic, 3,500 AI)           |
|                             | - Frontier Benchmark (Oct 2026): 2,091 images (941 auth, 1,150 AI)  |
|                             | - WildAIGC-Bench v3: 1,294 images (506 authentic, 788 AI)           |
|                             | Status: HISTORICAL REFERENCE ONLY; NOT A FRESH BENCHMARK UNSEALING   |
+-----------------------------+---------------------------------------------------------------------+
| [SEALED CALIBRATION SPLIT]  | 13,000 images                                                       |
|                             | Status: PROTECTED / NON-INDEPENDENT; NOT ACCESSED                   |
+-----------------------------+---------------------------------------------------------------------+
| [SEALED RESERVE SPLIT]      | 9,994 images total (2,999 primary reserve images)                   |
|                             | Status: SEALED; ACCESS PROHIBITED WITHOUT FORMAL AUTHORIZATION      |
+-----------------------------+---------------------------------------------------------------------+
| [SYNTHETIC DEMO DATASETS]   | 5 CSV tables in datasets/synthetic_demo/                            |
|                             | Status: QUARANTINED; STRICTLY PROHIBITED FROM ALL ML EVALUATION     |
+---------------------------------------------------------------------------------------------------+
```

### Partition Boundary Governance Findings
1. **Fresh Final Authenticity Benchmark Status:**  
   Per master instruction rules, unsealing the sealed reserve partition (2,999 images) or opening the protected calibration split without explicit authorization is strictly prohibited. The current status for fresh unsealing is therefore formally recorded as **`DATASET_ACCESS_BLOCKED`** / **`EVALUATION_PROTOCOL_REQUIRED`**.  
   Historical performance on the 10,385-image held-out test split from Phase 3 (`experiments/reports/phase3_results.json`) is re-analyzed with full scientific transparency, comprehensive metrics, and bootstrap confidence intervals, but is explicitly identified as historical reference rather than a newly unsealed benchmark.
2. **Synthetic Demonstration Data Quarantine:**  
   All files in `datasets/synthetic_demo/` are flagged as synthetic illustrative data designed solely for UI demonstration and regional map neutral rendering. They are excluded with zero tolerance from model training, validation, testing, and calibration fitting.

---

## 5. Module-by-Module Baseline Status Table

The 11 TruthLens analytical modules are evaluated against the approved vocabulary:
- `IMPLEMENTED_AND_EVALUATED`
- `EVALUATED_WITH_LIMITATIONS`
- `ANNOTATED_DATA_REQUIRED`
- `DATASET_ACCESS_BLOCKED`
- `EVALUATION_PROTOCOL_REQUIRED`
- `EXPLORATORY_ONLY`
- `NOT_IMPLEMENTED`
- `NOT_TESTED`

| # | Module Name | Implementation Source Path | Active Checkpoint / Engine | Current Baseline Status | Package C Scope & Methodology |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Authenticity Classifier** | `backend/app/services/authenticity/detector.py` | SigLIP ViT-B/16 + frozen head (`e39afc5536a0...`) | `EVALUATED_WITH_LIMITATIONS` | Benchmark on authorized data; re-evaluate historical Phase 3 10,385-image split with bootstrap 95% CIs. Record fresh unsealing as `DATASET_ACCESS_BLOCKED`. |
| **2** | **OCR Engine** | `backend/app/services/ocr/service.py` | EasyOCR 1.7.2 + Tesseract 5.4.0 fallback | `ANNOTATED_DATA_REQUIRED` (Historical baseline); upgraded to `IMPLEMENTED_AND_EVALUATED` in Package C | Multi-engine CER/WER benchmark across English/Hindi text slices, synthetic render slices, and degradation conditions (clean, blur, skew, low-contrast). |
| **3** | **Language Identification** | `backend/app/services/nlp/language_detector.py` | PAPLUCA XLM-RoBERTa (pinned rev `98655983...`) | `IMPLEMENTED_AND_EVALUATED` | Multi-class language evaluation on curated 20-language benchmark, short text, mixed script, and Romanized Hindi. Macro P/R/F1 + confusion matrix. |
| **4** | **Claim Extraction** | `backend/app/services/claims/extractor.py` | Deterministic syntactic extractor | `IMPLEMENTED_AND_EVALUATED` | Ground-truth claim extraction benchmark. Measure candidate detection P/R/F1, span boundary alignment, and normalization entity preservation. |
| **5** | **Fact-Checking Retrieval** | `backend/app/services/factcheck/service.py` | Google Claim Search API + SQLite cache | `EVALUATED_WITH_LIMITATIONS` | Informational retrieval service. Evaluated in Package B (100% cache hit, deduplication, live query verified). Directional voting excluded. |
| **6** | **YOLO Object Detection** | `backend/app/services/vision/detector.py` | YOLOv8n (`31e20dde3def...`) on RTX 3050 GPU | `EVALUATED_WITH_LIMITATIONS` | Contextual scene parsing; COCO 80-class mapping. Runtime, latency, and VRAM audited on RTX 3050. Bounding-box forensic ground truth absent. |
| **7** | **Grad-CAM Explanations** | `backend/app/services/xai/gradcam.py` | PyTorch hook on SigLIP ViT layer | `IMPLEMENTED_AND_EVALUATED` | ViT explanation sanity checks: model parameter randomization test, cascading layer randomization, stability under noise/compression. |
| **8** | **Classical Forensics** | `backend/app/services/forensics/analyzer.py` | ELA, Noise, Resampling, Canny, Gabor | `EVALUATED_WITH_LIMITATIONS` | Informational heuristics. Experiment 2 established no statistically significant predictive power for AI detection (all FDR q >= 0.292). Non-voting. |
| **9** | **C2PA Provenance** | `backend/app/services/provenance/c2pa_verifier.py`| `c2pa-python 0.39.0` (Rust core) | `IMPLEMENTED_AND_EVALUATED` | Cryptographic trust verification verified across 8 lifecycle trust outcomes. Informational status strictly isolated from authenticity voting. |
| **10**| **Evidence Fusion** | `backend/app/services/fusion/engine.py` | Rule-based engine (`e11.conservative.v1`) | `EVALUATED_WITH_LIMITATIONS` | Conservative gating policy. Model vote gated if uncalibrated; forensic cues strictly non-voting. Preserved invariant. |
| **11**| **Uncertainty & Calibration**| `backend/app/services/authenticity/calibration.py`| Temperature scaling / Platt scaling | `EXPLORATORY_ONLY` | Empirical uncertainty evaluation on authorized split (n=6,995): ECE, MCE, Brier score, reliability diagrams, predictive entropy, MSP, selective risk-coverage. |

---

## 6. Non-Negotiable Invariants & Research Safeguards

To prevent research regressions, the following constraints remain active throughout Package C:
1. **Model Weights Invariance:** `models/baselines/siglip_frozen_head.pt` (SHA-256 `e39afc5536a0...`) and `models/baselines/yolov8n.pt` (SHA-256 `31e20dde3def...`) must remain unmodified.
2. **Production Fusion Policy Invariance:** `e11.conservative.v1` weights, thresholds, and gating rules must not be altered to optimize evaluation metrics.
3. **Partition Protection:** No access to `calibration` or `reserve` splits without formal authorized unsealing protocols.
4. **Git Safety:** Zero destructive Git commands (`git reset`, `git checkout .`, `git clean`).
5. **Secret Redaction:** Zero `.env` or credential leakage in benchmark logs or evaluation reports.

---
*Verified and certified for Phase C evaluation execution.*
