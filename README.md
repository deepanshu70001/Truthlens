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

Translation, VLM analysis, C2PA parsing, authenticated multi-user access, result deletion/retention, durable jobs, and a verified demonstration corpus are not implemented. See [system audit](docs/implementation%20history/phase5_system_audit.md) and [Phase 4 dataset inventory](docs/implementation%20history/phase4_dataset_inventory.md).

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

The test suite includes software verification across all phases. See the [Phase 6 implementation report](docs/implementation%20history/phase6_implementation_report.md) and [Phase 7 evaluation report](docs/phase7_evaluation_report.md).

## Dataset, evaluation and research status

External benchmarks (Tiny-GenImage, Frontier Benchmark October 2026, and WildAIGC-Bench) have been systematically ingested under cryptographic SHA-256 validation and partitioned using leakage-free group-aware protocols.

Controlled empirical investigations revealed that unaugmented fine-tuning learned container format shortcuts (100% JPEG real vs 100% PNG AI in Tiny-GenImage). In Phase 7, matched in-memory augmentation was developed to successfully eliminate format vulnerabilities on in-distribution data (achieving 0.8071 Balanced Accuracy and 0.9044 ROC-AUC under matched augmentation). However, evaluation against modern late-2026 generators on the Frontier Benchmark indicated that the research candidate did not satisfy pre-registered promotion criteria. In accordance with strict evaluation governance, **the production detector retains the frozen baseline (`truthlens/siglip-cifake-frozen-head`)**.

## Project status and team

TruthLens is an academic research prototype developed as a CSE-AI project. All models, protocols, and evaluation splits are fully reproducible via local scripts under `experiments/`.

## Project documentation

- [Master Project Dossier](PROJECT_DOCUMENTATION/PROJECT_MASTER_DOSSIER.md) — Comprehensive master documentation suite
- [Documentation Index](docs/README.md) — Complete catalogue of all reference documents, audits, and phase reports
- [System architecture](docs/system_architecture.md) — Current pipeline and component map
- [API reference](docs/api_reference.md) — REST endpoints and structured report JSON schema
- [Deployment guide](docs/deployment_guide.md) — Local installation, configuration, and verification
- [Phase 7 evaluation report](docs/phase7_evaluation_report.md) — Locked test evaluation, group bootstrap CIs, and candidate decision
- [Phase 7 training report](docs/phase7_training_report.md) — Controlled training runs (S1, S2, R1, R2, A1) and robustness audit
- [Phase 7 protocol](docs/phase7_protocol.md) — Pre-registered protocol for matched in-memory augmentation
- [Phase 6 evaluation report](docs/phase6_evaluation_report.md) — Evaluation of E3 candidate and shortcut discovery
- [Phase 3 evaluation report](docs/phase3_evaluation_report.md) — Benchmark evaluation of the frozen baseline
- [Security audit](docs/implementation%20history/phase5_security_audit.md)
- [System audit](docs/implementation%20history/phase5_system_audit.md)
- [E2E test report](docs/implementation%20history/phase5_e2e_test_report.md)
- [Performance report](docs/implementation%20history/phase5_performance_report.md)
- [Research readiness audit](docs/implementation%20history/phase6_research_readiness_audit.md)
- [Final research report](docs/implementation%20history/phase6_final_research_report.md)
- [IEEE paper draft](docs/TruthLens_IEEE_research_paper.md)
- [Viva preparation](docs/implementation%20history/phase6_viva_preparation.md)
- [AI-image misclassification review](docs/ai_generation_misclassification_review.md)
