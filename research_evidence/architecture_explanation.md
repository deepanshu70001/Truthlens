# TruthLens System Architecture & Data-Flow Explanation

> **Document Status:** Complete & Traced to Codebase  
> **Inspection Date:** 2026-10-20  
> **Corresponding Diagram:** `research_evidence/architecture.mmd`  
> **Primary Source Code:** `backend/app/services/orchestrator.py`, `backend/app/main.py`, `backend/app/config.py`

---

## 1. Architectural Philosophy and Design Principles

TruthLens is architected around four core design principles:

1. **No Single Truth Oracle:** Real-world multimodal misinformation cannot be resolved by a single black-box classifier. TruthLens treats every model and algorithmic probe as an evidentiary signal with intrinsic uncertainty, error bounds, and domain constraints.
2. **Fail-Soft Asynchronous Orchestration:** A pipeline analyzing multimedia content against external web services, OCR extractors, and deep learning backbones must never crash due to a single component failure. If an external API (such as Google Fact Check) times out or EXIF data is scrubbed, the remaining 18 stages continue execution, recording an explicit warning in the audit event log.
3. **Hardware-Aware Sequential Execution:** Tailored for standard consumer/laptop workstation GPUs (specifically an NVIDIA GeForce RTX 3050 with 4 GB VRAM), the architecture enforces lazy loading, CPU offloading for lightweight preprocessing, and sequential execution with explicit memory cleanup (`torch.cuda.empty_cache()`).
4. **Epistemic Honesty Regarding Virality and Impact:** Where empirical training data is absent (specifically Indian regional social forward dynamics), the system declines to output pseudo-scientific regression scores, instead generating rule-based demographic hazard profiles accompanied by an explicit `DATA_UNAVAILABLE` declaration.

---

## 2. Layer-by-Layer Architectural Breakdown

### 2.1 Presentation Layer (Frontend SPA)
The client interface is implemented as a modern Single Page Application (SPA) using React 18, Vite, and Tailwind CSS.
- **Image Submission:** Users upload images via drag-and-drop. Client-side pre-validation validates image formats (JPEG, PNG, WebP) and size restrictions before network dispatch.
- **Real-Time Polling Dashboard:** Communicates with `/api/v1/analysis/{analysis_id}/status` to display incremental stage progression, runtime elapsed per stage, and system status flags.
- **Explainable Heatmap Viewer:** Renders interactive alpha-blended Grad-CAM overlays atop the original high-resolution image, highlighting specific pixel patches that triggered synthetic classification.
- **Geographic Risk Choropleth:** Integrates Leaflet to render a 36-region choropleth map of India, highlighting state-level linguistic and digital exposure to detected narrative claims.

### 2.2 API Gateway & Router Layer (FastAPI)
The backend exposes a versioned RESTful API (`/api/v1`) built with FastAPI, using asynchronous route handlers and Pydantic v2 validation models:
- `POST /api/v1/analyze/upload`: Accepts multipart image uploads, performs magic-byte validation, computes SHA-256 integrity hashes, records an `Analysis` entity in the database, and schedules the asynchronous pipeline.
- `GET /api/v1/analysis/{id}/status`: Returns the real-time execution state (`pending`, `processing`, `completed`, `failed`), currently executing stage name, and completed stage list.
- `GET /api/v1/analysis/{id}/report`: Returns the unified, audit-ready JSON report containing all evidentiary signals.
- `GET /api/v1/datasets/*`: Provides read-only access to synthetic scenarios and benchmark catalogs when demo mode is enabled.
- `GET /api/v1/health`: Returns system health, database connectivity, and GPU VRAM availability.

### 2.3 The 19-Stage Asynchronous Orchestrator
The central coordinator of TruthLens is the `AnalysisOrchestrator` (`backend/app/services/orchestrator.py`). Each stage runs within a managed wrapper `_run_stage` that captures execution time in milliseconds, tracks exceptions, and logs `SystemEvent` records.

```
Upload (Validations + SHA-256)
  │
  ├── Stage 1:  Metadata Extraction (EXIF, Camera Model, Software, Timestamps)
  ├── Stage 2:  Quality Assessment (Laplacian Blur, Noise Variance, Compression Artifacts)
  ├── Stage 3:  Authenticity Detection (SigLIP2 Backbone / Fallback Classifier)
  ├── Stage 4:  Classical Forensics (Error Level Analysis [ELA], Noise Residuals, JPEG Grid Analysis)
  ├── Stage 5:  Optical Character Recognition (EasyOCR / PyTesseract for Multilingual Scripts)
  ├── Stage 6:  Computer Vision (YOLOv8n Object & Entity Detection)
  ├── Stage 7:  Explainable AI (Grad-CAM Saliency Map Extraction via Target Layer)
  ├── Stage 8:  Text Normalization (Indic & English Text Cleansing, Script Identification)
  ├── Stage 9:  Claim Extraction (Rule-based & Linguistic Subject-Predicate Extraction)
  ├── Stage 10: Fact-Check Verification (Google Fact-Check Tools API Querying)
  ├── Stage 11: Provenance Search (Google Vision Web Detection / Reverse Image Query)
  ├── Stage 12: Image Hashing (Perceptual pHash, Difference dHash, Average aHash)
  ├── Stage 13: Evidence Collection (Ingestion into Unified EvidenceCollection Dataclass)
  ├── Stage 14: Evidence Fusion (Multi-Criteria Weighted Fusion Engine)
  ├── Stage 15: Natural Language Explanation (Contextual Rationale Synthesis)
  ├── Stage 16: Virality Risk Scoring (Multi-Factor Heuristic with DATA_UNAVAILABLE flag)
  ├── Stage 17: Regional Spread Estimation (Demographic Mapping Across 36 Indian States/UTs)
  ├── Stage 18: Social Impact Risk Assessment (Civilian & Community Risk Severity Matrix)
  └── Stage 19: Report Assembly (Schema Validation, Database Persistence & JSON Packaging)
```

---

## 3. Data Flow and Inter-Stage Dependencies

The stages exhibit both independent parallel execution potential and strict sequential dependencies:

1. **Extraction Dependencies:**
   - `Stage 5 (OCR)` → `Stage 8 (Text Normalization)` → `Stage 9 (Claim Extraction)` → `Stage 10 (Fact-Check Search)`.
   - `Stage 3 (Authenticity)` → `Stage 7 (Grad-CAM XAI)`: XAI hooks into the active PyTorch computation graph of the loaded authenticity model; if Stage 3 fails or is skipped, Stage 7 gracefully skips.
2. **Context & Impact Synthesis Dependencies:**
   - `Stage 16 (Virality)` consumes extracted claims, OCR text length, and detected visual objects.
   - `Stage 17 (Regional Spread)` consumes detected text language and recognized entities to cross-reference Indian demographic census records.
   - `Stage 18 (Social Impact)` fuses authenticity classification (Stage 14), fact-check ratings (Stage 10), regional exposure counts (Stage 17), and virality scores (Stage 16).
3. **Final Fusion Dependency:**
   - `Stage 14 (Fusion)` operates across all evidentiary signals collected in `Stage 13`. Each signal contributes a normalized confidence score and evidence weight.

---

## 4. Hardware and VRAM Management Strategy

Because the primary testbed operates on a 4 GB VRAM budget (RTX 3050 Laptop):
- **Model Isolation:** Deep neural models are not loaded simultaneously. When the authenticity detector finishes inference and Grad-CAM extraction, feature tensors are detached, and cache is freed before OCR or object detection inference executes.
- **CPU Offloading:** Classical forensics (ELA, Noise, JPEG grids), metadata extraction, and linguistic text normalization execute entirely on CPU threads via `asyncio.to_thread`, preserving GPU memory exclusively for transformer backbones.

---

## 5. Storage and Persistence Layer

- **Relational Metadata:** Stores `Analysis`, `ImageAsset`, `ModelRun`, and `SystemEvent` records. Configured for PostgreSQL 16 via `asyncpg`, with an automatic fallback to SQLite (`sqlite+aiosqlite`) for zero-dependency local verification.
- **Artifact Filesystem:** Raw uploaded images, intermediate thumbnails, and generated Grad-CAM heatmaps are persisted in deterministic, content-addressed paths under `data/artifacts/` and `data/heatmaps/`.
