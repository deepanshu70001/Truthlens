# Project Inventory — TruthLens Repository

> **Document Status:** Complete & Traced to Checked-Out Codebase
> **Repository:** `https://github.com/deepanshu70001/Truthlens`
> **Working Branch:** `fix/truthlens-remediation`
> **Workspace Path:** `d:\Projects\Truthlens`

---

## 1. Top-Level Repository Structure

```
Truthlens/
├── .github/                  # CI workflows (ci.yml)
├── backend/                  # FastAPI backend (Python 3.12)
│   ├── app/
│   │   ├── api/v1/           # REST endpoints (analyze, analysis, health, stubs)
│   │   ├── core/             # Logging, exceptions, security, upload bounds
│   │   ├── db/               # SQLAlchemy async session, declarative base
│   │   ├── models/           # ORM models: Analysis, ImageAsset, ModelRun, SystemEvent
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/         # Modular analysis services (authenticity, forensics, ocr, fusion, etc.)
│   │   └── utils/            # Device resolution, CUDA memory management
│   ├── tests/                # Automated pytest suite: 267 passed, 10 skipped
│   └── requirements.txt      # Backend Python dependencies
├── datasets/                 # Ingested external benchmarks (N = 29,640)
│   └── external/
│       ├── tiny_genimage/                         # 25,006 images + ingestion manifest
│       ├── ai_image_detection_benchmark_2026_10/  # 2,976 images + ingestion manifest
│       └── wildaigc_bench/                        # 1,658 images + ingestion manifest
├── docs/                     # Comprehensive architecture, methodology, and phase reports (Phases 1–7)
│   ├── TruthLens_IEEE_research_paper.md           # Research paper manuscript draft
│   └── phase*.md                                  # Phase 2–7 audit and evaluation reports
├── experiments/              # Training, calibration, splitting, and audit scripts
│   ├── configs/              # Phase 7 experiment configuration YAML
│   ├── manifests/            # Approved training, validation, calibration, and test manifests
│   ├── reports/              # Phase 2–7 evaluation reports and decision JSONs
│   ├── train_phase7.py       # Matched in-memory augmentation multi-architecture trainer
│   └── evaluate_phase7_test.py # Locked test evaluation across all external benchmarks
├── frontend/                 # React 19 + TypeScript + Vite + Tailwind CSS
│   ├── src/                  # App.tsx dashboard, components, services
│   └── package.json          # Frontend dependencies
├── models/
│   └── baselines/            # siglip_frozen_head.pt (SHA-256 e39afc5536...)
├── PROJECT_DOCUMENTATION/    # 7 authoritative project dossiers (Master, Architecture, Testing, etc.)
├── research_evidence/        # 22-document research evidence package
├── scripts/                  # check_doc_links.py, download_benchmarks.py, download_models.py
├── storage/                  # Runtime uploads and generated diagnostic artifacts
└── README.md                 # Primary project overview and quickstart
```

---

## 2. Key Backend Service Responsibilities

| Service Package | Key Module | Primary Responsibility |
|---|---|---|
| `authenticity/` | `detector.py`, `calibration.py` | Production SigLIP detector (`truthlens/siglip-cifake-frozen-head`), post-hoc temperature scaling, uncertainty gating |
| `forensics/` | `analyzer.py` | Classical image heuristics: ELA, noise variance, JPEG ghosting, edge density, resampling (informational cues) |
| `ocr/` | `service.py` | Optical character recognition via EasyOCR with optional Tesseract fallback |
| `claims/` | `extractor.py` | Heuristic extraction of candidate factual statements, temporal tokens, and location entities |
| `factcheck/` | `service.py` | Google Fact Check Tools API connector (config-gated) |
| `provenance/` | `service.py` | Google Vision Web Detection reverse lookup connector (config-gated) |
| `hashing/` | `service.py` | SHA-256 cryptographic hashing and 64-bit perceptual hashing (pHash, dHash, aHash, wHash) |
| `evidence/` | `schema.py` | Typed multi-modal evidence collection schema (`EvidenceItem`, `EvidenceCollection`) |
| `fusion/` | `engine.py` | Status-aware multi-modal evidence aggregation engine (Phase 6 rules) |
| `explanation/` | `generator.py` | Natural language explanation and limitations builder |
| `virality/` | `engine.py` | Rule-based content sensationalism and platform heuristics |
| `regional/` | `engine.py` | Canonical Indian geography mapping; returns explicit `DATA_UNAVAILABLE` status |
| `social_impact/` | `engine.py` | Keyword-based content sensitivity matching across 5 policy domains |
| `reporting/` | `builder.py` | Assembles final structured analysis report JSON |
| `orchestrator.py` | `orchestrator.py` | Central asynchronous coordinator executing stages with fail-soft boundaries |
