# Project Inventory — TruthLens Repository

> Inspected: 2026-10-20 | Git HEAD: `270afbb` | 6 commits

## Top-Level Structure

```
final year project/
├── backend/              # FastAPI backend (Python)
│   ├── app/
│   │   ├── api/v1/       # REST endpoints (analyze, analysis, health, datasets, stubs)
│   │   ├── core/         # Logging, exceptions, security
│   │   ├── db/           # SQLAlchemy async session, base model
│   │   ├── models/       # ORM: Analysis, ImageAsset, ModelRun, SystemEvent
│   │   ├── schemas/      # Pydantic: AnalysisOut, HealthResponse, etc.
│   │   ├── services/     # 27 service modules (see below)
│   │   ├── utils/        # Device resolution, GPU memory
│   │   └── workers/      # (directory exists, contents unknown)
│   ├── tests/            # pytest suite: 5 test files + conftest
│   ├── requirements.txt  # 38 dependencies
│   └── requirements-dev.txt
├── frontend/             # React + TypeScript + Vite
│   └── src/
│       ├── App.tsx       # 729-line single-file application
│       ├── App.css       # Custom styles
│       ├── index.css     # Base styles
│       └── services/     # API client (api.ts)
├── data/
│   ├── synthetic/        # 3 CSV datasets (500 records each) + manifest.json
│   ├── external/cifake/  # manifest.json only (images not committed to git)
│   ├── heatmaps/         # EMPTY
│   ├── artifacts/        # (storage)
│   └── reports/          # (storage)
├── docs/                 # 5 docs: ARCHITECTURE, EVALUATION, IMPLEMENTATION_PLAN, MODELS, VIRALITY_DATA
├── models/
│   ├── cache/            # HuggingFace model cache (git-ignored)
│   └── configs/          # Model configuration
├── results/ml/           # 3 experiment results (JSON + CSV predictions)
├── scripts/
│   ├── ml/               # train_baselines.py, make_comparison.py, make_plots.py, run_all.sh
│   ├── download_cifake_subset.py
│   ├── download_models.py
│   ├── generate_synthetic_data.py
│   └── validate_datasets.py
├── storage/              # uploads/ + artifacts/ (runtime storage)
├── test_data/            # fake_news_test.jpg (940 KB) + empty subdirs
├── docker/               # EMPTY
├── truthlens.db          # SQLite database (40 KB)
├── test_stage3_debug.db  # Debug database (40 KB)
└── requirements-ml.txt   # ML-specific requirements (178 bytes)
```

## Backend Service Modules (27 directories under `backend/app/services/`)

| Module | Key File | Size | Responsibility |
|--------|----------|------|---------------|
| `analysis/` | — | — | Analysis coordination |
| `authenticity/` | `detector.py` (8.4 KB) | 245 lines | SigLIP2 deepfake/real classification |
| `claims/` | `extractor.py` (8.4 KB) | — | Claim extraction from OCR text |
| `evidence/` | `schema.py` (17.2 KB) | 493 lines | Evidence item/collection data model + builders |
| `explanation/` | `generator.py` (8.5 KB) | 287 lines | Natural-language explanation from evidence |
| `factcheck/` | `service.py` | — | Google Fact-Check Tools API integration |
| `forensics/` | `analyzer.py` (18 KB) | 501 lines | 6 classical forensic analyses |
| `fusion/` | `engine.py` (9.1 KB) | 260 lines | Weighted evidence aggregation |
| `hashing/` | `service.py` (5.7 KB) | 201 lines | SHA-256 + pHash/aHash/dHash/wHash |
| `metadata/` | `service.py` | — | EXIF extraction |
| `ml/` | — | — | ML abstractions |
| `nlp/` | `language_detector.py` (5.2 KB) | — | XLM-RoBERTa language detection |
| `ocr/` | `service.py` (7.6 KB) | — | EasyOCR text extraction |
| `preprocessing/` | `service.py` | — | Thumbnail, resize, quality stats |
| `provenance/` | `service.py` (8.2 KB) | 266 lines | Google Vision Web Detection API |
| `quality/` | `service.py` | — | Image quality assessment |
| `regional/` | `engine.py` (11.4 KB) | 295 lines | India regional spread risk (heuristic) |
| `reporting/` | `builder.py` (5.1 KB) | — | JSON report assembly |
| `search/` | — | — | Search services |
| `social_impact/` | `engine.py` (11.1 KB) | 321 lines | Social impact assessment (heuristic) |
| `text/` | `normalizer.py` (5.3 KB) | — | Text cleaning, sentence splitting |
| `virality/` | `engine.py` (12.4 KB) | 341 lines | Virality estimation (heuristic) |
| `vision/` | `detector.py` (4.9 KB) | — | YOLO object detection |
| `vlm/` | `service.py` (4.6 KB) | — | Qwen2.5-VL context analysis |
| `xai/` | `gradcam.py` (11.4 KB) | 334 lines | Grad-CAM heatmap generation |

## Database Models (4 ORM classes)

| Model | Table | Fields |
|-------|-------|--------|
| `Analysis` | `analyses` | id, status, truth_score, truth_confidence, truth_label, current_step, completed_steps, processing_time_ms, error_message, report_json, timestamps |
| `ImageAsset` | `image_assets` | id, analysis_id, original_filename, storage_filename, storage_path, sha256_hash, perceptual_hash, mime_type, file_size_bytes, width, height, format, thumbnail_path |
| `ModelRun` | `model_runs` | id, analysis_id, model_id, model_version, task, device, input_hash, result_json, confidence, runtime_ms, error_message |
| `SystemEvent` | `system_events` | id, analysis_id, event_type, severity, source, message, metadata_json |

## Frontend (Single-Page Application)

- **Framework**: React 18+ with TypeScript
- **Build**: Vite
- **Styling**: Custom CSS (dark theme)
- **Architecture**: Monolithic `App.tsx` (729 lines) — upload, progress tracking, report display
- **Visualizations**: Recharts (radar/bar charts), React-Leaflet (India map), Grad-CAM overlay
- **API client**: `services/api.ts` — `uploadImage()`, `getAnalysisStatus()`, `getAnalysisReport()`, `getHeatmapUrl()`
