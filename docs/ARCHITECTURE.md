# TruthLens — System Architecture

> **TruthLens** is an AI-assisted image verification, analysis, and virality estimation system.
> It is NOT an absolute truth detector — all outputs are probabilistic assessments with confidence scores and supporting evidence.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + Vite)                     │
│  Upload • Dashboard • Analysis Report • Heatmap Viewer • Map View   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ REST API (JSON)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (Python)                        │
│                                                                     │
│  ┌──────────┐  ┌──────────────────┐  ┌────────────────────┐        │
│  │ API      │  │ Analysis         │  │ Background Workers │        │
│  │ Router   │──│ Orchestrator     │──│ (optional Redis)   │        │
│  │ Layer    │  │                  │  │                    │        │
│  └──────────┘  └────────┬─────────┘  └────────────────────┘        │
│                         │                                           │
│  ┌──────────────────────┴──────────────────────────────────┐       │
│  │              ANALYSIS SERVICE LAYER                      │       │
│  │                                                          │       │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │       │
│  │  │ Image       │ │ ML Model     │ │ NLP Services     │  │       │
│  │  │ Processing  │ │ Services     │ │ (OCR, Claims,    │  │       │
│  │  │ Pipeline    │ │ (Detection,  │ │  Language, etc.) │  │       │
│  │  │             │ │  Forensics)  │ │                  │  │       │
│  │  └─────────────┘ └──────────────┘ └──────────────────┘  │       │
│  │                                                          │       │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │       │
│  │  │ Search      │ │ XAI /        │ │ Evidence Fusion  │  │       │
│  │  │ Services    │ │ Heatmap      │ │ & Scoring        │  │       │
│  │  │ (Reverse    │ │ Generation   │ │                  │  │       │
│  │  │  Image,     │ │              │ │                  │  │       │
│  │  │  Fact-Check)│ │              │ │                  │  │       │
│  │  └─────────────┘ └──────────────┘ └──────────────────┘  │       │
│  │                                                          │       │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │       │
│  │  │ Virality    │ │ Regional     │ │ Social Impact    │  │       │
│  │  │ Estimation  │ │ Spread Est.  │ │ Assessment       │  │       │
│  │  └─────────────┘ └──────────────┘ └──────────────────┘  │       │
│  └──────────────────────────────────────────────────────────┘       │
│                         │                                           │
│  ┌──────────────────────┴──────────────────────────────────┐       │
│  │              REPORT GENERATION                           │       │
│  │  Structured JSON • PDF • Visual Summary                  │       │
│  └──────────────────────────────────────────────────────────┘       │
│                         │                                           │
│  ┌──────────────────────┴──────────────────────────────────┐       │
│  │              DATA LAYER                                  │       │
│  │  PostgreSQL • File Storage • Model Cache                 │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Details

### 2.1 Frontend

| Aspect         | Choice                                    |
| -------------- | ----------------------------------------- |
| Framework      | React 18+ with TypeScript                 |
| Build Tool     | Vite                                      |
| Styling        | Tailwind CSS                              |
| Charts         | Recharts                                  |
| Maps           | Leaflet / React-Leaflet                   |
| HTTP Client    | Axios or native fetch                     |
| State          | React Context + hooks (no Redux needed)   |
| Routing        | React Router v6                           |

**Key Pages:**
- **Upload Page** — drag-and-drop image upload with preview
- **Analysis Dashboard** — real-time progress of analysis pipeline
- **Report View** — comprehensive analysis report with all signals
- **Heatmap Viewer** — overlay of suspicious regions on the image
- **Map View** — regional spread risk visualization for India
- **History** — past analyses (if database is available)

### 2.2 Backend (FastAPI)

| Aspect         | Choice                                    |
| -------------- | ----------------------------------------- |
| Framework      | FastAPI 0.115+                            |
| Validation     | Pydantic v2                               |
| ORM            | SQLAlchemy 2.0 (async)                    |
| Database       | PostgreSQL (via Docker or SQLite fallback) |
| Migrations     | Alembic                                   |
| Background     | asyncio tasks, optional Celery+Redis      |
| Server         | Uvicorn                                   |

### 2.3 Analysis Orchestrator

The orchestrator is the core coordination layer. It:
1. Receives an uploaded image
2. Validates and preprocesses it
3. Dispatches to independent analysis services (parallel where possible)
4. Collects all signals
5. Runs evidence fusion
6. Generates the final report

```
Image Upload
    │
    ▼
┌────────────────────────────────┐
│    INGESTION & PREPROCESSING   │
│  • MIME check • Magic bytes    │
│  • Resize • Hash • Metadata   │
└────────────┬───────────────────┘
             │
    ┌────────┴────────────────────────────────────────┐
    │         PARALLEL ANALYSIS DISPATCH               │
    │                                                  │
    │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
    │  │Authent.  │ │Manip.    │ │ Visual Forensics │ │
    │  │Detection │ │Detection │ │ (ELA, noise)     │ │
    │  └──────────┘ └──────────┘ └──────────────────┘ │
    │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
    │  │Object    │ │Face      │ │ Metadata         │ │
    │  │Detection │ │Detection │ │ Analysis         │ │
    │  └──────────┘ └──────────┘ └──────────────────┘ │
    │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
    │  │OCR +     │ │Context   │ │ Quality          │ │
    │  │Language  │ │Analysis  │ │ Assessment       │ │
    │  └──────────┘ └──────────┘ └──────────────────┘ │
    │  ┌──────────┐ ┌──────────┐                      │
    │  │Reverse   │ │Image     │                      │
    │  │Image     │ │Hashing   │                      │
    │  └──────────┘ └──────────┘                      │
    └────────────────────┬────────────────────────────┘
                         │
    ┌────────────────────┴────────────────────────────┐
    │          SEQUENTIAL POST-PROCESSING              │
    │                                                  │
    │  OCR Text → Claim Extraction → Fact-Check Search │
    │  XAI / Heatmap Generation                        │
    │  Evidence Fusion                                 │
    │  Virality Estimation                             │
    │  Regional Spread Risk                            │
    │  Social Impact Assessment                        │
    └────────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  REPORT GENERATION │
              └────────────────────┘
```

### 2.4 ML Model Abstraction

All ML models are accessed through a **Model Registry** abstraction:

```python
class BaseModel(ABC):
    """Abstract base for all ML models in TruthLens."""

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> ModelResult: ...

    @abstractmethod
    def get_info(self) -> ModelInfo: ...

    def unload(self) -> None: ...
```

This allows:
- Hot-swapping models without code changes
- Consistent result formatting across all models
- Centralized device management (CPU ↔ GPU)
- Auditable model runs (every prediction is logged)

### 2.5 Evidence Fusion

Evidence fusion combines signals from multiple analysis modules into a unified assessment using **weighted evidence aggregation**:

- Each analysis module produces `EvidenceItem` objects with type, source, strength, and direction
- Evidence strengths are weighted: low=0.1, medium=0.3, high=0.6
- Directions are: authentic, synthetic, manipulated, neutral, informational
- A weighted aggregate produces the overall assessment label
- **Conflicting strong evidence forces an "inconclusive" result** — the system does not force a determination
- All confidence scores are explicitly marked as **uncalibrated** unless independently validated
- No single model's output is treated as ground truth

Evidence categories:
- `model_evidence` — ML model predictions (authenticity, deepfake detection)
- `forensic_evidence` — Classical forensic signals (ELA, noise, JPEG artifacts)
- `ocr_evidence` — Extracted text content and language
- `contextual_evidence` — Object detection and scene understanding
- `metadata_evidence` — EXIF data and anomalies
- `factcheck_evidence` — External fact-check API results
- `provenance_evidence` — Web detection and reverse image search
- `xai_evidence` — Grad-CAM attribution maps
- `hash_evidence` — Image hash matching and deduplication

### 2.6 Data Layer

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   PostgreSQL     │     │   File Storage   │     │   Model Cache    │
│                  │     │                  │     │                  │
│  • Analysis      │     │  • Uploads       │     │  • HF weights    │
│  • ModelRun      │     │  • Heatmaps      │     │  • YOLO weights  │
│  • ForensicSignal│     │  • XAI artifacts │     │  • Config files  │
│  • OCRResult     │     │  • Reports (PDF) │     │                  │
│  • Claims        │     │                  │     │                  │
│  • FactCheck     │     │                  │     │                  │
│  • Provenance    │     │                  │     │                  │
│  • Virality      │     │                  │     │                  │
│  • RegionalRisk  │     │                  │     │                  │
│  • SocialImpact  │     │                  │     │                  │
│  • SystemEvent   │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 3. Security Architecture

| Concern                | Approach                                           |
| ---------------------- | -------------------------------------------------- |
| File upload            | MIME + magic-byte validation, size limit (20 MB)   |
| Filename safety        | Sanitize + replace with random UUID                |
| Storage paths          | Random IDs, no user-controlled paths               |
| Path traversal         | Whitelist allowed directories, reject `..`          |
| Secrets                | `.env` only, never hard-coded                      |
| API errors             | Sanitized error responses, no stack traces in prod |
| CORS                   | Explicit origin whitelist                           |
| Rate limiting          | Per-IP rate limiting via SlowAPI                    |
| Logging                | Privacy-conscious: no PII, no raw image data       |
| Input validation       | Pydantic models for all request/response schemas   |

---

## 4. Deployment Architecture

### Development
```
Frontend (Vite dev server :5173)  →  Backend (Uvicorn :8000)  →  PostgreSQL (Docker :5432)
```

### Production (Docker Compose)
```yaml
services:
  frontend    # Nginx serving built React app
  backend     # Uvicorn + FastAPI
  postgres    # PostgreSQL 16
  redis       # Redis 7 (optional, for task queue)
```

---

## 5. Key Design Decisions

1. **Modular services over monolithic**: Each analysis type is an independent service that can be developed, tested, and replaced individually.

2. **Model abstraction layer**: Models are loaded via a registry with a common interface. Swapping models requires only a config change.

3. **Honest uncertainty**: The system never claims to "prove" an image is fake or real. All outputs include confidence intervals and explicit caveats.

4. **Data availability honesty**: Modules that require training data (virality, regional spread) explicitly declare their data status. `DATA_UNAVAILABLE` is a valid and honest system state.

5. **GPU-aware scheduling**: The system detects GPU availability and manages VRAM budget across multiple models. Models are loaded/unloaded dynamically to stay within the 4 GB VRAM limit of the RTX 3050.

6. **Async-first**: FastAPI async handlers + asyncio task dispatch for parallel analysis. No blocking I/O in the main event loop.

7. **SQLite fallback**: While PostgreSQL is the primary database, the system falls back to SQLite for development when PostgreSQL is unavailable.
