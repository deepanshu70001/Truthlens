# TruthLens

**AI-assisted image verification, analysis, and content-context indicators**

TruthLens is an academic research prototype that combines model outputs, image-forensic cues, OCR, claim lookup, provenance evidence, and structured reporting. It is not a definitive truth detector. A model score, forensic cue, heatmap, or external match is not proof that an image or claim is true or false.

## Current capabilities

- Securely validated image upload, hashes, image metadata and preprocessing.
- SigLIP2 classifier when its model dependencies and weights are available. Without an independently reviewed, checkpoint-matched calibration artifact, TruthLens shows its raw argmax and softmax scores separately and abstains with an inconclusive application classification. Its scores are not real-world probabilities; editing/manipulation classification is separate and unavailable from this detector.
- Classical forensic indicators with per-signal limitations; compression artifacts alone do not establish manipulation.
- OCR through EasyOCR or optional Tesseract, text normalization and heuristic claim eligibility/extraction.
- Optional Google Fact Check Tools and Google Vision Web Detection integrations.
- YOLO object detection when Ultralytics and its checkpoint are available; Grad-CAM where model support permits.
- Evidence collection/fusion and a structured report with per-stage status.
- Transparent content spread indicators and potential harm topic cues. Historical spread, state-level prediction, audience reach and observed harm outcomes are unavailable because suitable datasets are not installed.

Durable jobs, bearer authentication with ownership checks, and database migrations are implemented, with production infrastructure limitations documented. Translation and VLM analysis are unavailable; C2PA requires its optional SDK and trust configuration. See the [current status register](docs/MODULE_STATUS_REGISTER.md) and [roadmap](docs/PROJECT_ROADMAP.md).

## Architecture and stack

- UI: React 19, TypeScript, Vite, Tailwind, Recharts, Leaflet.
- API: FastAPI, Pydantic, async SQLAlchemy.
- Image/ML: Pillow, OpenCV, PyTorch/Transformers, EasyOCR optional, Ultralytics optional.
- Database: SQLite by default; PostgreSQL supported by async URL/driver.
- Storage: local filesystem under `STORAGE_PATH`.
- Background processing: database-backed durable job queue with a separate worker; delivery is at-least-once and real multi-process PostgreSQL recovery remains deployment verification work.

The API owns one orchestration pipeline and one report builder. See [system architecture](docs/system_architecture.md).

## Run locally

Requirements: Python 3.11+, Node 22.12+ recommended for Vite 8. OCR/model dependencies and weights require additional setup; CPU works but can be slower. CUDA is optional.

PowerShell, from repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to the local backend. API docs are at <http://127.0.0.1:8000/docs>; health is <http://127.0.0.1:8000/api/v1/health>. The detailed reproducible setup is in [deployment guide](docs/deployment_guide.md).

## Configuration and privacy

`.env.example` documents settings. Local SQLite and no external keys are sufficient for basic startup. Optional Google integrations send image bytes (Vision) or OCR-derived claim text (Fact Check Tools) outside the application when enabled. Review provider terms and obtain consent before use. Uploaded images and generated artifacts persist under `STORAGE_PATH`; no deletion/retention endpoint exists. The unauthenticated API is for trusted local use only.

## API and tests

- Actual endpoints and schemas: [API reference](docs/api_reference.md), generated OpenAPI `/openapi.json`.
- Backend tests: `python -m pytest backend/tests -q`.
- Frontend build: from `frontend/`, `npm ci` and `npm run build`.
- Focused synchronous cross-phase suite: `python -m pytest --noconftest -m "not asyncio" backend/tests/test_phase1.py backend/tests/test_phase2.py backend/tests/test_phase3.py backend/tests/test_stage3.py backend/tests/test_stage4.py backend/tests/test_phase4.py backend/tests/test_phase5.py backend/tests/test_stage5.py backend/tests/test_services.py -q`.

The test suite includes software verification across all phases. See the [Phase 6 implementation report](docs/implementation%20history/phase6_implementation_report.md) and [Phase 7 evaluation report](docs/phase7_evaluation_report.md).

## Dataset, evaluation and research status

External benchmarks (Tiny-GenImage, Frontier Benchmark October 2026, and WildAIGC-Bench) have been systematically ingested under cryptographic SHA-256 validation and partitioned using leakage-free group-aware protocols.

Controlled empirical investigations revealed that unaugmented fine-tuning learned container format shortcuts (100% JPEG real vs 100% PNG AI in Tiny-GenImage). In Phase 7, matched in-memory augmentation was developed to successfully eliminate format vulnerabilities on in-distribution data (achieving 0.8071 Balanced Accuracy and 0.9044 ROC-AUC under matched augmentation). However, evaluation against modern late-2026 generators on the Frontier Benchmark indicated that the research candidate did not satisfy pre-registered promotion criteria. In accordance with strict evaluation governance, **the production detector retains the frozen baseline (`truthlens/siglip-cifake-frozen-head`)**.

## Project status and team

TruthLens is an academic research prototype developed as a CSE-AI project. All models, protocols, and evaluation splits are fully reproducible via local scripts under `experiments/`.

## Project documentation

- [Documentation index](docs/README.md)
- [Current module status](docs/MODULE_STATUS_REGISTER.md)
- [Roadmap](docs/PROJECT_ROADMAP.md)
- [System architecture](docs/system_architecture.md)
- [API reference](docs/api_reference.md)
- [Deployment guide](docs/deployment_guide.md)
- [Migration runbook](docs/migration_runbook.md)
- [E11/R10 completion report](docs/e11_r10_completion_report.md)
- [Research and evaluation history](docs/README.md#research-evidence)
