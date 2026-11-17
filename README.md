# TruthLens

**AI-assisted image verification, analysis, and content-context indicators**

TruthLens is an academic research prototype that combines model outputs, image-forensic cues, OCR, claim lookup, provenance evidence, and structured reporting. It is not a definitive truth detector. A model score, forensic cue, heatmap, or external match is not proof that an image or claim is true or false.

## Current capabilities

- Securely validated image upload, hashes, image metadata and preprocessing.
- SigLIP2 authenticity classifier when its model dependencies and weights are available. Its softmax scores are uncalibrated; editing/manipulation classification is separate and unavailable from this detector.
- Classical forensic indicators with per-signal limitations; compression artifacts alone do not establish manipulation.
- OCR through EasyOCR or optional Tesseract, text normalization and heuristic claim eligibility/extraction.
- Optional Google Fact Check Tools and Google Vision Web Detection integrations.
- YOLO object detection when Ultralytics and its checkpoint are available; Grad-CAM where model support permits.
- Evidence collection/fusion and a structured report with per-stage status.
- Transparent content spread indicators and potential harm topic cues. Historical spread, state-level prediction, audience reach and observed harm outcomes are unavailable because suitable datasets are not installed.

Translation, VLM analysis, C2PA parsing, authenticated multi-user access, result deletion/retention, durable jobs, and a verified demonstration corpus are not implemented. See [system audit](docs/phase5_system_audit.md) and [Phase 4 dataset inventory](docs/phase4_dataset_inventory.md).

## Architecture and stack

- UI: React 19, TypeScript, Vite, Tailwind, Recharts, Leaflet.
- API: FastAPI, Pydantic, async SQLAlchemy.
- Image/ML: Pillow, OpenCV, PyTorch/Transformers, EasyOCR optional, Ultralytics optional.
- Database: SQLite by default; PostgreSQL supported by async URL/driver.
- Storage: local filesystem under `STORAGE_PATH`.
- Background processing: in-process async task; restart can interrupt active analyses.

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

The Phase 6 full backend suite passed 188 tests with 10 opt-in model tests skipped. This is software verification, not model evaluation. See the [Phase 6 implementation report](docs/phase6_implementation_report.md).

## Dataset, evaluation and research limits

The repository has no licensed, ground-truthed diverse image corpus for accuracy claims and no observed dissemination/outcome dataset for virality, regional or social-impact prediction. Existing files under `test_data/` lack source/license/ground-truth documentation and are not designated demo samples. Unit tests generate synthetic images for code paths only. Model timing and its limits are in [performance report](docs/phase5_performance_report.md). A publishable research paper requires properly licensed data, a documented protocol, baselines, statistical evaluation, and independent review; this repository does not claim those results.

## Project status and team

This is the existing CSE-AI student-team project described in the implementation prompts. Verified team-member names, institution, and paper authorship details were not present in the repository, so the paper draft leaves those fields for verification. Phase 6 prepares research-readiness materials and reproducible evaluation tooling; empirical model validation remains blocked by missing licensed labeled data and independent evaluations.

## Project documentation

- [System architecture](docs/system_architecture.md)
- [API reference](docs/api_reference.md)
- [Deployment guide](docs/deployment_guide.md)
- [Security audit](docs/phase5_security_audit.md)
- [System audit](docs/phase5_system_audit.md)
- [E2E test report](docs/phase5_e2e_test_report.md)
- [Performance report](docs/phase5_performance_report.md)
- [Phase 5 implementation report](docs/phase5_implementation_report.md)
- [Phase 6 research readiness audit](docs/phase6_research_readiness_audit.md)
- [Phase 6 final research report](docs/phase6_final_research_report.md)
- [Phase 6 IEEE paper draft](docs/TruthLens_IEEE_research_paper.md)
- [Phase 6 viva preparation](docs/phase6_viva_preparation.md)
- [Phase 6 implementation report](docs/phase6_implementation_report.md)
- [Phase 6 experiment tooling](experiments/README.md)
