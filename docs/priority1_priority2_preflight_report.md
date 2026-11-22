# Priority 1 & 2 Completion Preflight Audit

Audit date: 2026-11-22 (Windows, PowerShell). Scope: read-only repository/runtime inspection. No dependencies, model files, databases, or provider services were changed or called. This report is the only file created by this audit.

## Repository state

- Branch: `fix/truthlens-remediation` (**READY**; expected branch matches).
- HEAD: `93b551cfe93c1cbeb91e943803da122c22f3b575` (**READY**).
- `git status --short`: pre-existing uncommitted E11/R10 work: modified `backend/app/services/{evidence/schema.py,explanation/generator.py,fusion/engine.py,orchestrator.py,reporting/builder.py}`, `backend/tests/{test_phase6.py,test_stage4.py}`, `docs/{MODULE_STATUS_REGISTER.md,PROJECT_ROADMAP.md}`, `frontend/src/App.tsx`, `frontend/src/__tests__/App.test.tsx`; untracked `backend/tests/test_e11_fusion.py`, E11/R10 reports/protocol docs, and `research/`. Preserved as found.
- `git status --short --ignored` additionally showed ignored `.pytest_cache`, `.ruff_cache`, Python caches, `datasets/`, model caches/checkpoints, `experiments/checkpoints/` local files, `frontend/node_modules/`, `frontend/dist/`, `research_evidence.zip`, `storage/`, and `truthlens.db`. These were not opened or modified. `models/cache/` existence was noted; individual protected or unrelated artifacts were not enumerated.
- `git diff --check`: exit 0; Git printed LF-to-CRLF working-copy notices for modified files.
- `git log -8 --oneline`: HEAD `93b551c` E8-E10; prior `63cce07` E6/E7; `aa9d7a3` E3/E4/E5; `6e188e0`; `8e7d551` Phase10 blocker; `ecc19c6` Phase9 diagnostics; `f0c2877` Phase9 diagnostics; `fde9800` frequency features.

## Python, CUDA, and GPU

Python 3.12.10; pip 26.2.2. `nvidia-smi` reports NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB VRAM, 56 MiB used at inspection. PyTorch 2.7.1+cu118; CUDA runtime reported by PyTorch 11.8; `torch.cuda.is_available()` true. This establishes device visibility only, not successful application inference.

| Package | Availability |
|---|---|
| torch | 2.7.1+cu118 |
| transformers | 4.57.6 |
| easyocr | 1.7.2 |
| pytesseract | 0.3.13 |
| ultralytics | missing |
| c2pa | missing (expected SDK import not available) |
| PIL | importable; distribution version not resolved in this probe |
| cv2 | 4.12.0.88 |
| sqlalchemy | importable; distribution version not resolved in this probe |
| alembic | 1.18.4 |
| psycopg | missing |
| asyncpg | 0.31.0 |
| pytest | 8.3.4 |

`pip list` was requested; pip emitted an invalid `~orch` distribution warning. Its long package listing was truncated by the command output limit; use `python -m pip list` locally if the complete inventory is needed. No packages were installed.

## E2 — OCR runtime

**BLOCKED — MISSING DEPENDENCY.** `where.exe tesseract` found no executable; `tesseract --version` and `--list-langs` could not run. Thus no installed language packs are verified. EasyOCR package is installed. Ignored `models/cache/` and `models/ocr/` directories exist, but cache contents were not enumerated during this pass; application configuration/cache-only behavior is present in repository code. A real EasyOCR inference result is not established by this audit. Required setup: install Tesseract and intended language packs, then run the existing OCR runtime verification with an approved fixture.

## E3 — PAPLUCA language model

**NOT VERIFIED.** The exact configured cache directory `models/cache/models--papluca--xlm-roberta-base-language-detection` and its snapshot/config/tokenizer/weight files were not independently inventoried. `transformers` is installed; repository code documents cache-only loading. A real inference test is not established here. Required setup: provision the exact approved checkpoint locally and verify its revision/config without downloading during audit.

## E6 — YOLO and Grad-CAM

- YOLO: **BLOCKED — MISSING DEPENDENCY**; `ultralytics` is absent. Repository configuration is `settings.VISION_MODEL`; detector refuses automatic download when configured local weights are absent. Exact checkpoint presence was not verified (no broad model/research scan). No real YOLO test can be claimed.
- Grad-CAM: **PARTIALLY READY**; code and regression tests exist, and CUDA is visible, but GPU memory is 4 GiB and production SigLIP layer/runtime smoke test was not run. SigLIP cache/revision was not re-enumerated in this audit. Do not infer successful production-model Grad-CAM from mocks. No detector was loaded or modified.

## E7 — C2PA and Google Vision provenance

- C2PA SDK: **BLOCKED — MISSING DEPENDENCY**; SDK import is unavailable. Config exposes a trust-anchor path, but presence/content was not checked (**NOT VERIFIED**). Permitted signed C2PA fixture: **NOT VERIFIED**; known fixture inventory included `backend/tests/fixtures/test_photo.jpg`, but it was not asserted signed.
- Google Vision credentials: **BLOCKED — MISSING CREDENTIAL** for live calls; presence was deliberately not printed or probed by reading credential content. No provider request was made.
- Endpoint/provider tests: **READY — MOCKED TESTS ONLY**; provenance service, endpoint, and unit/regression tests are present. No live provider or real signed asset validation is established.

## E8/E10 — PostgreSQL, migrations, worker, browser

- `where.exe psql` / `psql --version`: no `psql` executable (**BLOCKED — MISSING DEPENDENCY**). `where.exe docker` found Docker Desktop CLI, version 29.9.3; daemon availability was not checked. No PostgreSQL container/database was created.
- Existing default config is local SQLite (`sqlite+aiosqlite:///./truthlens.db`); ignored `truthlens.db` exists and was not opened. Production config validates PostgreSQL async URL. No existing database was connected to.
- Alembic revisions found: `20261122_01_legacy_schema_baseline.py`, `20261122_02_durable_jobs_security.py`; migration config and `backend/migrations/env.py` exist. Migration head was not applied or queried in this audit. SQLite migration tests are available; actual fresh/existing PostgreSQL upgrade is **BLOCKED — MISSING DEPENDENCY / ISOLATED DB NOT ESTABLISHED**.
- Worker entrypoint and durable queue code exist (`backend/app/services/job_queue.py`, `backend/app/workers/`); configured lease/retry controls are present. Local deterministic tests are **READY — MOCKED TESTS ONLY**. Real multi-process crash/restart with PostgreSQL is not verified.
- PostgreSQL driver: `asyncpg` installed; `psycopg` absent. Backup/restore executable availability was not established (**NOT VERIFIED**).
- Playwright package/module not installed in Python probe; browser configuration/dependency was not independently verified. Repository frontend workflow tests may be mocked; actual browser E2E readiness is **NOT VERIFIED** until package/config/browser are checked. No browser test was run.

## E9 — Security readiness

**READY — MOCKED TESTS ONLY** for focused local unit/API checks. Repository contains bearer token auth, ownership guards, DB-backed rate-limit code, upload validation, endpoint artifact access checks, production secret/host/CORS/database validation, and `backend/tests/test_e9_security.py`. `.env.example` uses blank secret/provider fields; actual secret values were not read. Config includes trusted hosts, CORS origins, proxy-related controls and separate rate limits. This source/test audit does not establish deployed proxy trust, production auth provisioning, shared PostgreSQL rate-limit behavior, or penetration-test security. Production/deployment checks require an isolated configured environment; destructive tests were not run.

## E4/E5 — Claims and fact checking

**READY — MOCKED TESTS ONLY** for local claim extraction, normalization, offsets/provenance, fact-check endpoint, and provider error handling. Relevant service and contract tests exist. Annotated ground-truth fixture availability was not verified beyond known general fixtures; no labels were fabricated. Fact Check Tools credential presence was not probed and no live call was made (**BLOCKED — MISSING CREDENTIAL** until operator provision/presence check). Standalone OCR route remains a stub per existing implementation history; OCR integration tests can run, standalone route completion cannot be assumed.

## Production protection

Production identity previously read-only verified in the E11/R10 work: model ID `truthlens/siglip-cifake-frozen-head`, backbone `google/siglip-base-patch16-224`, cached revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`, checkpoint SHA-256 `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`. The checkpoint was not loaded or changed in this audit. Inference routing and calibration artifacts were not edited. No training/tuning occurred; no calibration/held-out evaluation data were opened. Protected research directories were not broadly scanned. Current audit does not independently rehash the checkpoint; cited identity is **PARTIALLY READY** based on prior recorded verification.

## Immediate work, blockers, and recommended order

1. **Executable immediately (local/mocked):** E11/E1–E7 contract and service regression tests, OCR mocked fallback tests, claims/fact-check mocked tests, E8 SQLite queue/lease/recovery tests, E9 auth/ownership/upload/rate-limit unit/API tests, E10 SQLite migration tests, Python compilation, frontend typecheck/build/unit tests if existing frontend dependencies are intact.
2. **Requires local setup:** install Tesseract plus required packs for real fallback; provision PAPLUCA model cache for real E3 inference; install Ultralytics and provision only the configured YOLO checkpoint for real YOLO; provision C2PA SDK and a permitted signed fixture/trust anchors; verify Playwright/browser setup for browser E2E.
3. **Requires isolated infrastructure:** establish disposable PostgreSQL (Docker daemon or local server), apply migrations only to that disposable database, exercise worker restart/retry and shared rate-limit behavior; establish backup/restore tooling.
4. **Requires operator credentials:** Google Vision and Fact Check Tools credentials; verify only presence, then live calls only under separately authorized integration tests.
5. **Deployment-only verification:** production proxy/trusted-host behavior, secret provisioning, auth principal lifecycle, PostgreSQL concurrency, and production GPU memory/inference. Do not substitute mocked tests for these checks.

No repository source, package, checkpoint, database, environment secret, or provider state was changed. No commit or push was made. The existing uncommitted E11/R10 work and ignored assets were preserved.
