# TruthLens — Implementation Plan

> **TruthLens**: AI-Assisted Image Verification, Analysis and Virality Estimation Including Social Impact
>
> ⚠️ TruthLens is an **AI-assisted verification system**, NOT an absolute truth detector.
> No model prediction should ever be presented as proof that an image is fake or real.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Findings](#2-repository-findings)
3. [Technology Stack](#3-technology-stack)
4. [Folder Structure](#4-folder-structure)
5. [Module Plan (24 Modules)](#5-module-plan-24-modules)
6. [API Plan](#6-api-plan)
7. [Database Plan](#7-database-plan)
8. [Security Plan](#8-security-plan)
9. [Data Availability Assessment](#9-data-availability-assessment)
10. [Implementation Stages](#10-implementation-stages)
11. [VRAM Management Strategy](#11-vram-management-strategy)
12. [Testing Strategy](#12-testing-strategy)
13. [Deployment Plan](#13-deployment-plan)

---

## 1. Project Overview

TruthLens accepts an image and produces a comprehensive analysis covering:

- **Authenticity assessment** — AI-generated or authentic (probabilistic, multi-model ensemble)
- **Manipulation detection** — splicing, copy-move, inpainting indicators
- **Visual forensics** — ELA, noise analysis, JPEG artifact maps
- **Object & face detection** — scene understanding
- **OCR** — text extraction with language detection
- **Contextual analysis** — VLM-powered scene reasoning
- **Explainable AI** — Grad-CAM heatmaps highlighting suspicious regions
- **Provenance search** — reverse image lookup
- **Fact-checking** — claim extraction and verification
- **Virality estimation** — spread risk assessment
- **Regional spread risk** — India-specific geographic risk map
- **Social impact** — potential societal impact indicators
- **Comprehensive report** — all evidence fused into an auditable report

---

## 2. Repository Findings

### 2.1 Initial State
- **Repository**: Empty directory (no existing code)
- **No existing**: README, package files, source code, database, Docker config, or assets

### 2.2 System Environment

| Component        | Details                                              |
| ---------------- | ---------------------------------------------------- |
| **OS**           | Windows                                              |
| **CPU**          | Intel i5-12500H (12 cores / 16 threads)              |
| **RAM**          | 16 GB                                                |
| **GPU**          | NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM, CUDA)    |
| **CUDA**         | Available (CUDA 11.8 via PyTorch)                    |
| **Python**       | 3.12.10                                              |
| **Node.js**      | v22.23.2                                             |
| **npm**          | 10.9.8                                               |
| **Docker**       | 29.8.1 + Compose v5.5.1                              |
| **PostgreSQL**   | NOT installed locally (will use Docker)               |

### 2.3 Pre-installed Python Packages (Key)

| Package          | Version   | Status      |
| ---------------- | --------- | ----------- |
| Python           | 3.12.10   | ✅ Installed |
| PyTorch          | 2.7.1+cu118 | ✅ Installed |
| torchvision      | 0.22.1+cu118 | ✅ Installed |
| transformers     | 4.57.6    | ✅ Installed |
| timm             | 1.0.3     | ✅ Installed |
| FastAPI          | 0.115.6   | ✅ Installed |
| SQLAlchemy       | 2.0.45    | ✅ Installed |
| Pydantic         | 2.10.4    | ✅ Installed |
| OpenCV           | 4.12.0    | ✅ Installed |
| Pillow           | 11.3.0    | ✅ Installed |
| NumPy            | 1.26.4    | ✅ Installed |
| Pandas           | 2.2.3     | ✅ Installed |
| scikit-learn     | 1.5.0     | ✅ Installed |
| scikit-image     | 0.25.2    | ✅ Installed |
| huggingface-hub  | 0.36.0    | ✅ Installed |
| accelerate       | 1.13.0    | ✅ Installed |
| uvicorn          | 0.34.0    | ✅ Installed |
| redis            | 5.1.0     | ✅ Installed |
| python-dotenv    | 1.0.1     | ✅ Installed |
| Starlette        | 0.41.3    | ✅ Installed |
| slowapi          | 0.1.9     | ✅ Installed |
| spacy            | 3.8.14    | ✅ Installed |
| pytesseract      | 0.3.13    | ✅ Installed |
| streamlit        | 1.52.2    | ✅ Installed |
| easyocr          | —         | ❌ Not installed |
| ultralytics      | —         | ❌ Not installed |
| captum           | —         | ❌ Not installed |
| imagehash        | —         | ❌ Not installed |
| alembic          | —         | ❌ Not installed |
| asyncpg          | —         | ❌ Not installed |
| aiosqlite        | —         | ❌ Not installed |

---

## 3. Technology Stack

### Backend
| Layer            | Technology                                           |
| ---------------- | ---------------------------------------------------- |
| Web Framework    | FastAPI 0.115+                                       |
| Validation       | Pydantic v2                                          |
| ORM              | SQLAlchemy 2.0 (async mode)                          |
| Database         | PostgreSQL 16 (Docker) / SQLite (fallback)           |
| Migrations       | Alembic                                              |
| Server           | Uvicorn (ASGI)                                       |
| Image Processing | OpenCV, Pillow, scikit-image                         |
| ML Framework     | PyTorch 2.7, torchvision, timm                       |
| NLP              | Hugging Face Transformers, spaCy                     |
| OCR              | EasyOCR                                              |
| Object Detection | Ultralytics YOLO26                                   |
| XAI              | Captum, custom Grad-CAM                              |
| Hashing          | imagehash                                            |
| Task Queue       | asyncio (primary), optional Redis + Celery           |
| Rate Limiting    | SlowAPI                                              |

### Frontend
| Layer            | Technology                                           |
| ---------------- | ---------------------------------------------------- |
| Framework        | React 18+ with TypeScript                            |
| Build Tool       | Vite                                                 |
| Styling          | Tailwind CSS                                         |
| Charts           | Recharts                                             |
| Maps             | Leaflet / React-Leaflet                              |
| HTTP             | Axios                                                |
| Routing          | React Router v6                                      |

### Infrastructure
| Layer            | Technology                                           |
| ---------------- | ---------------------------------------------------- |
| Containers       | Docker + Docker Compose                              |
| Secrets          | `.env` file (dotenv)                                 |
| CI/CD            | GitHub Actions (future)                              |

---

## 4. Folder Structure

```
truthlens/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app factory
│   │   ├── config.py                  # Settings via pydantic-settings
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── router.py          # Main API router
│   │   │       ├── endpoints/
│   │   │       │   ├── analyze.py     # POST /analyze
│   │   │       │   ├── analysis.py    # GET /analysis/{id}
│   │   │       │   ├── ocr.py         # POST /ocr
│   │   │       │   ├── factcheck.py   # POST /fact-check
│   │   │       │   ├── provenance.py  # POST /provenance
│   │   │       │   ├── models.py      # GET /models
│   │   │       │   └── health.py      # GET /health
│   │   │       └── deps.py            # Dependency injection
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── security.py           # File validation, sanitization
│   │   │   ├── exceptions.py         # Custom exception handlers
│   │   │   └── logging.py            # Privacy-conscious logging
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── session.py            # Async DB session factory
│   │   │   ├── base.py               # SQLAlchemy Base
│   │   │   └── migrations/           # Alembic migrations
│   │   ├── models/                    # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── image_asset.py
│   │   │   ├── model_run.py
│   │   │   ├── forensic_signal.py
│   │   │   ├── ocr.py
│   │   │   ├── claims.py
│   │   │   ├── factcheck.py
│   │   │   ├── provenance.py
│   │   │   ├── xai.py
│   │   │   ├── virality.py
│   │   │   ├── regional_risk.py
│   │   │   ├── social_impact.py
│   │   │   └── system_event.py
│   │   ├── schemas/                   # Pydantic request/response schemas
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── image.py
│   │   │   ├── ocr.py
│   │   │   ├── forensics.py
│   │   │   ├── factcheck.py
│   │   │   ├── provenance.py
│   │   │   ├── virality.py
│   │   │   ├── report.py
│   │   │   └── health.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py        # Main analysis orchestrator
│   │   │   ├── analysis/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ingestion.py       # Module 1: Image ingestion
│   │   │   │   ├── preprocessing.py   # Module 2: Preprocessing
│   │   │   │   ├── metadata.py        # Module 3: Metadata analysis
│   │   │   │   ├── quality.py         # Module 4: Quality analysis
│   │   │   │   ├── authenticity.py    # Module 5: AI-generated detection
│   │   │   │   ├── manipulation.py    # Module 6: Manipulation detection
│   │   │   │   ├── forensics.py       # Module 7: Visual forensics (ELA, noise)
│   │   │   │   ├── objects.py         # Module 8: Object detection
│   │   │   │   ├── faces.py           # Module 9: Face detection
│   │   │   │   └── context.py         # Module 10: Background/context analysis
│   │   │   ├── ml/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py            # BaseModel abstract class
│   │   │   │   ├── registry.py        # Model registry & lifecycle
│   │   │   │   ├── device_manager.py  # GPU/CPU management
│   │   │   │   ├── siglip_detector.py # SigLIP2 authenticity model
│   │   │   │   ├── community_forensics.py  # CommunityForensics ViT
│   │   │   │   ├── vlm_analyzer.py    # Qwen2.5-VL integration
│   │   │   │   ├── yolo_detector.py   # YOLO26 object detection
│   │   │   │   └── xai_engine.py      # Grad-CAM / attribution maps
│   │   │   ├── nlp/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ocr_service.py     # Module 11: OCR
│   │   │   │   ├── language.py        # Module 12: Language detection
│   │   │   │   ├── normalizer.py      # Module 13: Text normalization
│   │   │   │   └── claims.py          # Module 14: Claim extraction
│   │   │   ├── search/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── factcheck.py       # Module 15: Fact-check search
│   │   │   │   ├── reverse_image.py   # Module 16: Reverse image search
│   │   │   │   └── image_hash.py      # Module 17: Image hash/similarity
│   │   │   └── reporting/
│   │   │       ├── __init__.py
│   │   │       ├── heatmap.py         # Module 19: Heatmap generation
│   │   │       ├── evidence_fusion.py # Module 20: Evidence fusion
│   │   │       ├── virality.py        # Module 21: Virality estimation
│   │   │       ├── regional.py        # Module 22: India regional spread
│   │   │       ├── social_impact.py   # Module 23: Social impact
│   │   │       └── report_gen.py      # Module 24: Report generation
│   │   ├── workers/
│   │   │   ├── __init__.py
│   │   │   └── tasks.py              # Background task definitions
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── file_utils.py         # Safe file handling
│   │       ├── image_utils.py        # Image processing helpers
│   │       └── hash_utils.py         # Hashing utilities
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── unit/
│   │   └── integration/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── alembic.ini
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/                   # Reusable UI components
│   │   │   ├── analysis/             # Analysis-specific components
│   │   │   └── layout/               # Layout components
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/                 # API client
│   │   ├── types/                    # TypeScript types
│   │   ├── utils/
│   │   └── assets/
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── models/
│   ├── configs/                      # Model configuration files
│   └── cache/                        # Downloaded model weights (gitignored)
├── storage/
│   ├── uploads/                      # Uploaded images (gitignored)
│   └── artifacts/                    # Generated artifacts (gitignored)
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── scripts/
│   ├── setup.py                      # One-time setup script
│   └── download_models.py            # Model pre-download script
├── docs/
│   ├── IMPLEMENTATION_PLAN.md        # This file
│   ├── ARCHITECTURE.md               # System architecture
│   └── MODELS.md                     # Model registry
├── .env.example
├── .gitignore
└── README.md
```

---

## 5. Module Plan (24 Modules)

### Module 1: Image Ingestion
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/ingestion.py`                              |
| **Input**   | Uploaded file (multipart form)                                |
| **Output**  | Validated file path, SHA-256 hash, storage ID                 |
| **Logic**   | MIME check → magic-byte check → size limit → sanitize name → store with random UUID |
| **Deps**    | `python-magic` or `filetype`, Pillow                          |

### Module 2: Image Preprocessing
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/preprocessing.py`                          |
| **Input**   | Raw image path                                                |
| **Output**  | Standardized image tensors at multiple resolutions             |
| **Logic**   | Load → color-space normalize → resize variants (224, 384, original) → compute basic stats |
| **Deps**    | OpenCV, Pillow, torchvision transforms                        |

### Module 3: Metadata Analysis
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/metadata.py`                               |
| **Input**   | Image file                                                    |
| **Output**  | EXIF data, GPS (if present), camera model, software, timestamps |
| **Logic**   | Extract EXIF → check for inconsistencies (e.g., software = "Photoshop") → flag anomalies |
| **Deps**    | Pillow (ExifTags), piexif or exifread                         |
| **Signals** | Missing EXIF (common in AI images), editing software traces    |

### Module 4: Image Quality Analysis
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/quality.py`                                |
| **Input**   | Image array                                                   |
| **Output**  | Resolution, DPI, blur score, noise level, compression level    |
| **Logic**   | Laplacian variance (blur) → noise estimation → JPEG quality estimation |
| **Deps**    | OpenCV, NumPy                                                 |

### Module 5: AI-Generated Image Detection
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/authenticity.py`                           |
| **Models**  | SigLIP2 (primary) + CommunityForensics ViT (secondary)       |
| **Input**   | Preprocessed image tensors                                    |
| **Output**  | Ensemble probability: P(real), P(AI-generated)                |
| **Logic**   | Run both models → weighted average → report individual + ensemble scores |
| **Deps**    | transformers, timm, torch                                     |
| **Note**    | Output is ALWAYS a probability with confidence interval, NEVER a binary verdict |

### Module 6: Manipulation Detection
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/manipulation.py`                           |
| **Input**   | Image                                                         |
| **Output**  | Manipulation indicators (splicing, copy-move, inpainting)      |
| **Logic**   | ELA analysis + noise inconsistency + copy-move detection       |
| **Deps**    | OpenCV, NumPy, scikit-image                                   |
| **Note**    | Primarily algorithmic (non-ML) forensic techniques             |

### Module 7: Visual Forensic Analysis
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/forensics.py`                              |
| **Input**   | Image                                                         |
| **Output**  | ELA map, noise map, JPEG ghost map, edge inconsistency map     |
| **Logic**   | Multiple classical forensic algorithms                         |
| **Deps**    | OpenCV, Pillow, NumPy, scikit-image                           |

### Module 8: Object Detection
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/objects.py`                                |
| **Model**   | YOLO26s                                                       |
| **Input**   | Image                                                         |
| **Output**  | Detected objects with bounding boxes, classes, confidence      |
| **Deps**    | ultralytics                                                   |

### Module 9: Face Detection
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/faces.py`                                  |
| **Input**   | Image                                                         |
| **Output**  | Face bounding boxes, count, landmarks                         |
| **Deps**    | YOLO-face or OpenCV DNN face detector                         |
| **Note**    | Detection only — NO identification / recognition              |

### Module 10: Background/Context Analysis
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/analysis/context.py`                                |
| **Model**   | Qwen2.5-VL-3B-Instruct (quantized)                           |
| **Input**   | Image + structured prompts                                    |
| **Output**  | Scene description, visual inconsistencies, contextual flags    |
| **Logic**   | Prompt VLM for: scene description, inconsistency check, suspicious elements |
| **Deps**    | transformers, qwen-vl-utils                                   |
| **Note**    | VLM is a reasoning tool, NOT the authenticity classifier       |

### Module 11: OCR
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/nlp/ocr_service.py`                                 |
| **Model**   | EasyOCR                                                       |
| **Input**   | Image                                                         |
| **Output**  | List of OCRBlock (bbox, text, confidence, language)            |
| **Deps**    | easyocr                                                       |

### Module 12: Language Detection
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/nlp/language.py`                                    |
| **Model**   | `papluca/xlm-roberta-base-language-detection`                 |
| **Input**   | Text string                                                   |
| **Output**  | Detected language code + confidence                           |
| **Deps**    | transformers                                                  |

### Module 13: Text Normalization
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/nlp/normalizer.py`                                  |
| **Input**   | Raw OCR text                                                  |
| **Output**  | Cleaned, normalized text                                      |
| **Logic**   | Unicode normalization → whitespace cleanup → optional translation |
| **Deps**    | Built-in Python, optionally NLLB-200 for translation          |

### Module 14: Claim Extraction
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/nlp/claims.py`                                      |
| **Input**   | Normalized OCR text + VLM context                             |
| **Output**  | List of extracted claims (text, type, confidence)              |
| **Logic**   | Pattern-based extraction + VLM-assisted claim identification   |
| **Deps**    | spaCy, regex, VLM service                                     |

### Module 15: Fact-Check Search
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/search/factcheck.py`                                |
| **Input**   | Extracted claims                                               |
| **Output**  | Matching fact-check results from external APIs                 |
| **Logic**   | Query Google Fact Check API / ClaimBuster → return matches     |
| **Deps**    | requests/httpx, external API keys                             |
| **Data**    | REAL DATA (from live APIs) — requires API keys                 |

### Module 16: Reverse Image / Provenance Search
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/search/reverse_image.py`                            |
| **Input**   | Image                                                         |
| **Output**  | Matching URLs, earliest known appearance, source domains       |
| **Logic**   | Google Custom Search API (image) or TinEye API                 |
| **Deps**    | requests/httpx, external API keys                             |
| **Data**    | REAL DATA (from live APIs) — requires API keys                 |

### Module 17: Image Similarity / Hash Analysis
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/search/image_hash.py`                               |
| **Input**   | Image                                                         |
| **Output**  | Perceptual hashes (pHash, dHash, aHash), similarity scores    |
| **Logic**   | Compute hashes → compare against stored hashes for duplicates  |
| **Deps**    | imagehash, Pillow                                             |

### Module 18: Explainable AI
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/ml/xai_engine.py`                                   |
| **Input**   | Model + input image                                           |
| **Output**  | Attribution maps (which regions influenced the model decision) |
| **Logic**   | Grad-CAM on authenticity model → generate overlay              |
| **Deps**    | captum, torch, matplotlib                                     |

### Module 19: Heatmap Generation
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/reporting/heatmap.py`                               |
| **Input**   | Attribution maps + forensic maps + original image              |
| **Output**  | Composite heatmap overlays (PNG)                               |
| **Logic**   | Normalize maps → colormap → alpha-blend with original          |
| **Deps**    | OpenCV, matplotlib, NumPy                                     |

### Module 20: Evidence Fusion
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/reporting/evidence_fusion.py`                       |
| **Input**   | All ForensicSignals from modules 3-19                         |
| **Output**  | Unified TruthScore + confidence + signal breakdown             |
| **Logic**   | Weighted aggregation → conflict detection → uncertainty estimation |
| **Deps**    | NumPy, custom logic                                           |
| **Note**    | Weights are configurable and documented                        |

### Module 21: Virality Estimation

| Aspect          | Detail                                                    |
| --------------- | --------------------------------------------------------- |
| **File**        | `services/reporting/virality.py`                          |
| **Input**       | Analysis results (objects, faces, text, sentiment, topic)  |
| **Output**      | Virality risk score + contributing factors                 |
| **Data Status** | ⚠️ **DATA_UNAVAILABLE for training**                      |
| **Approach**    | Heuristic-based scoring using known virality indicators    |
| **Indicators**  | Emotional content, faces (especially children/politicians), sensational text, health/political topics |
| **Honest State**| System will clearly label this as "heuristic estimate, not data-driven prediction" |
| **Future**      | Can be improved with real social media spread data if obtained |

### Module 22: India Regional Spread Estimation

| Aspect          | Detail                                                    |
| --------------- | --------------------------------------------------------- |
| **File**        | `services/reporting/regional.py`                          |
| **Input**       | Language detected, content type, topic indicators          |
| **Output**      | Risk scores per Indian region/state                        |
| **Data Status** | ⚠️ **DATA_UNAVAILABLE for real spread data**              |
| **Approach**    | Language-region mapping heuristic (e.g., Hindi text → Hindi belt states) |
| **Demo Data**   | Optional static demographic/language-distribution data for visualization |
| **Honest State**| System will label as "language-based regional mapping, not empirical spread prediction" |
| **Future**      | Requires real social media propagation dataset for India   |

### Module 23: Social Impact Analysis

| Aspect          | Detail                                                    |
| --------------- | --------------------------------------------------------- |
| **File**        | `services/reporting/social_impact.py`                     |
| **Input**       | All analysis results                                       |
| **Output**      | Impact indicators (communal risk, health risk, political sensitivity, etc.) |
| **Data Status** | ⚠️ **DATA_UNAVAILABLE for empirical impact modeling**     |
| **Approach**    | Rule-based assessment using content indicators             |
| **Categories**  | Communal tension, health misinformation, political manipulation, financial fraud, child safety |
| **Honest State**| System will label as "indicator-based risk assessment, not predictive model" |

### Module 24: Report Generation
| Aspect      | Detail                                                        |
| ----------- | ------------------------------------------------------------- |
| **File**    | `services/reporting/report_gen.py`                            |
| **Input**   | Complete analysis results                                      |
| **Output**  | Structured JSON report + optional PDF                          |
| **Logic**   | Compile all signals → structure into report sections → generate |
| **Deps**    | Pydantic models, reportlab (PDF), Jinja2 (templates)          |

---

## 6. API Plan

### Endpoints

| Method | Endpoint                              | Description                          |
| ------ | ------------------------------------- | ------------------------------------ |
| POST   | `/api/v1/analyze`                     | Submit image for full analysis       |
| GET    | `/api/v1/analysis/{analysis_id}`      | Get analysis status and results      |
| GET    | `/api/v1/analysis/{analysis_id}/report` | Get formatted analysis report      |
| GET    | `/api/v1/analysis/{analysis_id}/heatmap` | Get heatmap overlay image         |
| POST   | `/api/v1/ocr`                         | Standalone OCR extraction            |
| POST   | `/api/v1/fact-check`                  | Standalone fact-check query          |
| POST   | `/api/v1/provenance`                  | Standalone reverse image search      |
| GET    | `/api/v1/models`                      | List loaded/available models         |
| GET    | `/api/v1/health`                      | System health check                  |
| GET    | `/api/v1/health/models`               | Model-specific health/readiness      |

### Request/Response Patterns

#### POST `/api/v1/analyze`
```json
// Request: multipart/form-data with field "image"
// Response:
{
  "analysis_id": "uuid",
  "status": "processing",
  "created_at": "ISO-8601",
  "estimated_duration_seconds": 45
}
```

#### GET `/api/v1/analysis/{analysis_id}`
```json
{
  "analysis_id": "uuid",
  "status": "completed",  // pending | processing | completed | failed
  "progress": {
    "current_step": "evidence_fusion",
    "completed_steps": ["ingestion", "preprocessing", "metadata", "..."],
    "total_steps": 18,
    "percentage": 85
  },
  "results": {
    "truth_score": {
      "score": 0.72,
      "confidence": 0.65,
      "label": "likely_authentic",
      "caveat": "This is an AI-assisted probabilistic assessment, not proof."
    },
    "authenticity": { /* ... */ },
    "manipulation": { /* ... */ },
    "forensics": { /* ... */ },
    "metadata": { /* ... */ },
    "objects": { /* ... */ },
    "faces": { /* ... */ },
    "ocr": { /* ... */ },
    "context": { /* ... */ },
    "provenance": { /* ... */ },
    "fact_check": { /* ... */ },
    "xai": { /* ... */ },
    "virality": { /* ... */ },
    "regional_risk": { /* ... */ },
    "social_impact": { /* ... */ }
  },
  "model_runs": [ /* audit trail */ ],
  "processing_time_seconds": 42.3
}
```

#### GET `/api/v1/health`
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "gpu_memory_total_mb": 4096,
  "gpu_memory_used_mb": 1200,
  "database": "connected",
  "uptime_seconds": 3600
}
```

---

## 7. Database Plan

### Entity-Relationship Overview

```
Analysis ──────────── 1:1 ──── ImageAsset
    │
    ├── 1:N ──── ModelRun
    ├── 1:N ──── ForensicSignal
    ├── 1:1 ──── OCRResult ──── 1:N ──── OCRBlock
    ├── 1:N ──── Claim ──────── 1:N ──── FactCheckResult
    ├── 1:N ──── ProvenanceResult
    ├── 1:N ──── XAIArtifact
    ├── 1:1 ──── ViralityResult
    ├── 1:N ──── RegionalRisk
    └── 1:1 ──── SocialImpact

SystemEvent (independent audit log)
```

### Table Definitions

#### Analysis
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Analysis identifier                |
| image_asset_id      | UUID (FK)    | Link to uploaded image             |
| status              | Enum         | pending/processing/completed/failed |
| truth_score         | Float        | Overall score (0-1)                |
| truth_confidence    | Float        | Confidence in score (0-1)          |
| truth_label         | String       | Human-readable label               |
| processing_time_ms  | Integer      | Total processing time              |
| error_message       | Text (null)  | Error details if failed            |
| created_at          | DateTime     | Creation timestamp (UTC)           |
| completed_at        | DateTime     | Completion timestamp (UTC)         |

#### ImageAsset
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Storage identifier                 |
| original_filename   | String       | Sanitized original name            |
| storage_path        | String       | Internal storage path              |
| sha256_hash         | String(64)   | File content hash                  |
| mime_type           | String       | Validated MIME type                |
| file_size_bytes     | Integer      | File size                          |
| width               | Integer      | Image width (px)                   |
| height              | Integer      | Image height (px)                  |
| format              | String       | Image format (JPEG, PNG, etc.)     |
| created_at          | DateTime     | Upload timestamp                   |

#### ModelRun
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Run identifier                     |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| model_id            | String       | Model identifier (HF repo path)   |
| model_version       | String       | Model version / commit hash        |
| task                | String       | Task name (e.g., "authenticity")   |
| device              | String       | Device used (cpu/cuda:0)           |
| input_hash          | String(64)   | SHA-256 of input image             |
| result_json         | JSONB        | Raw model output                   |
| confidence          | Float        | Output confidence                  |
| runtime_ms          | Integer      | Inference time (ms)                |
| error_message       | Text (null)  | Error if failed                    |
| created_at          | DateTime     | Run timestamp                      |

#### ForensicSignal
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Signal identifier                  |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| signal_type         | String       | Type (authenticity/manipulation/metadata/etc.) |
| source_module       | String       | Which module generated this        |
| value               | Float        | Numeric signal value (0-1)         |
| confidence          | Float        | Signal confidence (0-1)            |
| evidence_json       | JSONB        | Supporting evidence data           |
| description         | Text         | Human-readable description         |
| created_at          | DateTime     | Creation timestamp                 |

#### OCRResult
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | OCR result identifier              |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| full_text           | Text         | Concatenated OCR text              |
| primary_language    | String       | Detected primary language          |
| language_confidence | Float        | Language detection confidence      |
| block_count         | Integer      | Number of text blocks detected     |
| created_at          | DateTime     | Creation timestamp                 |

#### OCRBlock
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Block identifier                   |
| ocr_result_id       | UUID (FK)    | Parent OCR result                  |
| text                | Text         | Block text content                 |
| confidence          | Float        | OCR confidence for this block      |
| bbox_x              | Integer      | Bounding box X                     |
| bbox_y              | Integer      | Bounding box Y                     |
| bbox_w              | Integer      | Bounding box width                 |
| bbox_h              | Integer      | Bounding box height                |
| language            | String       | Detected language for this block   |

#### Claim
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Claim identifier                   |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| text                | Text         | Claim text                         |
| claim_type          | String       | Type (factual/opinion/statistic)   |
| source              | String       | Source (ocr/vlm/user)              |
| confidence          | Float        | Extraction confidence              |
| created_at          | DateTime     | Creation timestamp                 |

#### FactCheckResult
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Result identifier                  |
| claim_id            | UUID (FK)    | Parent claim                       |
| source_name         | String       | Fact-check source (e.g., "Google") |
| source_url          | String       | URL to fact-check article          |
| rating              | String       | Rating (true/false/misleading/etc.)|
| rating_value        | Float (null) | Numeric rating if available        |
| review_date         | DateTime     | When the fact-check was published  |
| created_at          | DateTime     | Creation timestamp                 |

#### ProvenanceResult
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Result identifier                  |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| source_url          | String       | Where the image was found          |
| source_domain       | String       | Domain name                        |
| first_seen          | DateTime     | Earliest known appearance          |
| match_type          | String       | exact/near-duplicate/similar       |
| similarity_score    | Float        | Similarity to input image          |
| context_text        | Text         | Text around the image on the page  |
| created_at          | DateTime     | Creation timestamp                 |

#### XAIArtifact
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Artifact identifier                |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| artifact_type       | String       | Type (gradcam/ela/noise/heatmap)   |
| model_run_id        | UUID (FK, null) | Associated model run (if any)   |
| storage_path        | String       | Path to generated image/artifact   |
| description         | Text         | What this artifact shows           |
| created_at          | DateTime     | Creation timestamp                 |

#### ViralityResult
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Result identifier                  |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| virality_score      | Float        | Overall virality risk (0-1)        |
| confidence          | Float        | Estimation confidence              |
| data_status         | String       | DATA_AVAILABLE/DEMO_DATA/DATA_UNAVAILABLE |
| methodology         | String       | How the score was computed         |
| contributing_factors| JSONB        | Factors that increased the score   |
| created_at          | DateTime     | Creation timestamp                 |

#### RegionalRisk
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Result identifier                  |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| region_name         | String       | State or region name               |
| region_code         | String       | ISO code or custom code            |
| risk_score          | Float        | Regional risk (0-1)                |
| confidence          | Float        | Estimation confidence              |
| data_status         | String       | DATA_AVAILABLE/DEMO_DATA/DATA_UNAVAILABLE |
| methodology         | String       | How the score was computed         |
| contributing_factors| JSONB        | Why this region is at risk         |
| created_at          | DateTime     | Creation timestamp                 |

#### SocialImpact
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Result identifier                  |
| analysis_id         | UUID (FK)    | Parent analysis                    |
| overall_risk        | Float        | Overall social impact risk (0-1)   |
| communal_risk       | Float        | Communal tension risk              |
| health_risk         | Float        | Health misinformation risk         |
| political_risk      | Float        | Political manipulation risk        |
| financial_risk      | Float        | Financial fraud risk               |
| child_safety_risk   | Float        | Child safety concern risk          |
| data_status         | String       | DATA_AVAILABLE/DEMO_DATA/DATA_UNAVAILABLE |
| methodology         | String       | How scores were computed           |
| indicators_json     | JSONB        | Detailed indicator breakdown       |
| created_at          | DateTime     | Creation timestamp                 |

#### SystemEvent
| Column              | Type         | Description                        |
| ------------------- | ------------ | ---------------------------------- |
| id                  | UUID (PK)    | Event identifier                   |
| event_type          | String       | Type (model_load/error/warning)    |
| severity            | String       | info/warning/error/critical        |
| source              | String       | Component that generated event     |
| message             | Text         | Event description                  |
| metadata_json       | JSONB        | Additional event data              |
| created_at          | DateTime     | Event timestamp                    |

---

## 8. Security Plan

### File Upload Security
| Measure                | Implementation                                     |
| ---------------------- | -------------------------------------------------- |
| MIME validation        | Check `Content-Type` header against whitelist       |
| Magic-byte validation  | Read file header bytes to confirm actual type       |
| File size limit        | Reject files > 20 MB                               |
| Filename sanitization  | Strip path components, special chars, enforce ext   |
| Random storage IDs     | Store with UUID filename, never user-provided name  |
| Path traversal         | Reject `..`, validate resolved paths stay in upload dir |
| Image validity         | Attempt to open with Pillow — reject invalid images |

### Application Security
| Measure                | Implementation                                     |
| ---------------------- | -------------------------------------------------- |
| Secrets management     | All secrets in `.env`, loaded via pydantic-settings |
| `.gitignore`           | `.env`, weights, uploads, artifacts all excluded    |
| CORS                   | Explicit origin whitelist (configurable)            |
| API errors             | Sanitized responses; no tracebacks in production    |
| Rate limiting          | SlowAPI: 30 requests/min per IP                     |
| Input validation       | Pydantic models on all endpoints                    |
| Logging                | No PII, no raw image data in logs                   |
| SQL injection          | SQLAlchemy parameterized queries (never raw SQL)    |

---

## 9. Data Availability Assessment

> **Critical Scientific Rule**: Predictions without data are dishonest. This section explicitly classifies each module's data dependency.

| Module                        | Data Status             | Notes                                              |
| ----------------------------- | ----------------------- | -------------------------------------------------- |
| Image Ingestion               | N/A                     | No training data needed                            |
| Image Preprocessing           | N/A                     | No training data needed                            |
| Metadata Analysis             | N/A                     | Rule-based, no training data                       |
| Image Quality Analysis        | N/A                     | Algorithmic, no training data                      |
| AI-Generated Detection        | ✅ **REAL DATA**        | Pre-trained on large datasets (CommunityForensics: 2.7M images) |
| Manipulation Detection        | N/A                     | Algorithmic forensic techniques                    |
| Visual Forensic Analysis      | N/A                     | Algorithmic (ELA, noise analysis)                  |
| Object Detection              | ✅ **REAL DATA**        | Pre-trained on COCO dataset                        |
| Face Detection                | ✅ **REAL DATA**        | Pre-trained models                                 |
| Context Analysis (VLM)        | ✅ **REAL DATA**        | Pre-trained Qwen2.5-VL                             |
| OCR                           | ✅ **REAL DATA**        | Pre-trained EasyOCR                                |
| Language Detection            | ✅ **REAL DATA**        | Pre-trained XLM-RoBERTa (70k+ samples)             |
| Text Normalization            | N/A                     | Rule-based                                         |
| Claim Extraction              | 🟡 **OPTIONAL DEMO**   | Rule-based + VLM-assisted; no dedicated training data |
| Fact-Check Search             | ✅ **REAL DATA**        | From live APIs (Google Fact Check)                  |
| Reverse Image Search          | ✅ **REAL DATA**        | From live APIs                                      |
| Image Hash/Similarity         | N/A                     | Algorithmic                                        |
| XAI / Grad-CAM                | N/A                     | Derived from model activations                     |
| Heatmap Generation            | N/A                     | Visualization, no training data                    |
| Evidence Fusion               | 🟡 **OPTIONAL DEMO**   | Weights are heuristic; could be learned with data   |
| **Virality Estimation**       | ⚠️ **DATA_UNAVAILABLE** | No real social media spread dataset available       |
| **Regional Spread (India)**   | ⚠️ **DATA_UNAVAILABLE** | No real regional propagation dataset available      |
| **Social Impact Analysis**    | ⚠️ **DATA_UNAVAILABLE** | No empirical impact model; rule-based indicators    |
| Report Generation             | N/A                     | Template-based                                     |

### Honest System Behavior for DATA_UNAVAILABLE modules

When a module has `DATA_UNAVAILABLE` status, the system will:

1. **Always display** the data status prominently in the report
2. **Label outputs** as "heuristic estimate" or "indicator-based assessment"
3. **Never claim** predictive accuracy without empirical validation
4. **Provide transparency** about the methodology used (rule-based, language-mapping, etc.)
5. **Support future data integration** — the architecture allows plugging in real models when data becomes available

---

## 10. Implementation Stages

### Stage 1 (Current — Planning) ✅
- [x] Repository inspection
- [x] Environment assessment
- [x] Model investigation
- [x] Architecture design
- [x] Create `docs/IMPLEMENTATION_PLAN.md`
- [x] Create `docs/ARCHITECTURE.md`
- [x] Create `docs/MODELS.md`
- [x] Create project folder structure
- [x] Create `.env.example`, `.gitignore`, `README.md`

### Stage 2 — Backend Foundation ✅
- [x] FastAPI app factory (`main.py`, `config.py`)
- [x] Database setup (SQLAlchemy async, models, Alembic)
- [x] API router skeleton (all endpoints returning 501)
- [x] File upload endpoint with full security validation
- [x] Image ingestion + preprocessing modules
- [x] Metadata analysis module
- [x] Image quality analysis module
- [x] Health endpoint
- [x] Error handling + logging framework
- [x] `requirements.txt`
- [x] Basic Docker setup

### Stage 3 — Core ML Pipeline ✅
- [x] Model abstraction layer (`BaseModel`, `ModelRegistry`)
- [x] Device manager (GPU memory tracking)
- [x] SigLIP2 authenticity detector integration
- [x] CommunityForensics ViT integration
- [x] Ensemble authenticity scoring
- [x] Visual forensics (ELA, noise analysis)
- [x] YOLO26 object detection
- [x] Face detection
- [x] XAI / Grad-CAM heatmap generation
- [x] Model run auditing

### Stage 4 — NLP & Search Pipeline ✅
- [x] EasyOCR integration
- [x] Language detection
- [x] Text normalization
- [x] Claim extraction
- [x] Fact-check API integration
- [x] Reverse image search integration (Google Vision Web Detection)
- [x] Image hash computation

### Stage 5 — VLM & Context Analysis
- [ ] Qwen2.5-VL integration (quantized)
- [ ] Context analysis prompts
- [ ] Background/scene consistency analysis
- [ ] VLM-assisted claim extraction

### Stage 6 — Evidence Fusion & Reporting ✅
- [x] Evidence schema and collection
- [x] Evidence fusion algorithm
- [x] Explanation generator
- [x] Report generation (JSON)
- [x] Analysis orchestrator (full pipeline coordination)
- [x] Report API endpoint (`GET /api/v1/analysis/{id}/report`)
- [x] Virality estimation (heuristic + DATA_UNAVAILABLE flag)
- [x] India regional spread estimation (heuristic + DATA_UNAVAILABLE flag)
- [x] Social impact assessment (rule-based + DATA_UNAVAILABLE flag)
- [x] Data source investigation (`docs/VIRALITY_DATA.md`)
- [x] Evaluation framework (`docs/EVALUATION.md`)
- [x] DEMO_MODE support with clear labeling
- [ ] PDF report generation

### Stage 7 — Frontend
- [ ] Vite + React + TypeScript project setup
- [ ] Tailwind CSS configuration
- [ ] Upload page with drag-and-drop
- [ ] Analysis dashboard with real-time progress
- [ ] Report viewer with all sections
- [ ] Heatmap overlay viewer
- [ ] India regional map (Leaflet)
- [ ] Charts (Recharts)
- [ ] Responsive design

### Stage 8 — Integration & Polish
- [ ] End-to-end testing
- [ ] Performance optimization (model loading, caching)
- [ ] Docker Compose production setup
- [ ] Documentation finalization
- [ ] Demo preparation

---

## 11. VRAM Management Strategy

Given the 4 GB VRAM constraint of the RTX 3050 Laptop GPU:

### Sequential Model Loading Groups

```
Phase 1: Lightweight classifiers (can co-exist)
├── SigLIP2 authenticity      (~400 MB)
├── CommunityForensics ViT    (~200 MB)
└── YOLO26s                   (~500 MB)
    Total: ~1,100 MB
    → Unload all after completion

Phase 2: NLP models
├── EasyOCR                   (~400 MB)
└── XLM-RoBERTa language det  (~550 MB)
    Total: ~950 MB
    → Unload all after completion

Phase 3: VLM (largest, runs alone)
└── Qwen2.5-VL-3B (Q4)       (~1,900 MB)
    Total: ~1,900 MB
    → Unload after completion

Phase 4: XAI (reloads specific model)
└── Reload authenticity model + Captum
    Total: ~500 MB
```

### Memory Management Rules

1. **Never load all models simultaneously**
2. **Explicitly unload** (`model.cpu()` + `del model` + `torch.cuda.empty_cache()`) between phases
3. **Track VRAM** via `torch.cuda.memory_allocated()` before each load
4. **CPU fallback** if GPU memory insufficient for a model
5. **Configurable**: Users can force CPU-only mode via `DEVICE=cpu` in `.env`

---

## 12. Testing Strategy

| Level           | Tool                | Focus                              |
| --------------- | ------------------- | ---------------------------------- |
| Unit tests      | pytest              | Individual services, utilities      |
| Integration     | pytest + httpx      | API endpoints, DB operations        |
| ML model tests  | pytest              | Model loading, inference shape validation |
| Security tests  | pytest              | File validation, path traversal attacks |
| Frontend tests  | Vitest + React Testing Library | Component rendering, interactions |
| E2E tests       | Playwright (future) | Full pipeline browser tests         |

---

## 13. Deployment Plan

### Development
```bash
# Backend
cd backend && uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev

# Database (Docker)
docker run -d --name truthlens-pg -e POSTGRES_PASSWORD=truthlens -e POSTGRES_DB=truthlens -p 5432:5432 postgres:16
```

### Production (Docker Compose)
```yaml
services:
  backend:
    build: ./docker/Dockerfile.backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  frontend:
    build: ./docker/Dockerfile.frontend
    ports: ["80:80"]

  postgres:
    image: postgres:16
    volumes: ["pgdata:/var/lib/postgresql/data"]
    environment:
      POSTGRES_DB: truthlens
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

volumes:
  pgdata:
```

---

## Appendix A: External API Dependencies

| API                        | Required? | Purpose                           | Free Tier?     |
| -------------------------- | --------- | --------------------------------- | -------------- |
| Google Fact Check API      | Optional  | Fact-checking claims              | Yes (limited)  |
| Google Custom Search API   | Optional  | Reverse image search              | Yes (100/day)  |
| ClaimBuster API            | Optional  | Claim check-worthiness scoring    | Yes (research) |
| TinEye API                 | Optional  | Reverse image search              | Paid           |

If no API keys are configured, the system gracefully degrades — search modules return `"status": "api_not_configured"` instead of failing.

---

## Appendix B: Disclaimer Template

Every analysis report will include the following disclaimer:

> **DISCLAIMER**: TruthLens is an AI-assisted verification tool. Its outputs represent probabilistic assessments based on computational analysis, not definitive truth. Model predictions can be wrong. AI-generated content detection is an evolving field subject to concept drift. No result from this system should be cited as proof that an image is authentic or fabricated. Always cross-reference findings with additional sources and expert judgment.

---

*Document Version: 1.0.0*
*Created: 2026-10-01*
*Next Stage: Stage 2 — Backend Foundation*
