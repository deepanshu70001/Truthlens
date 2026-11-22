# Deployment and reproducibility guide

TruthLens now has a database-backed at-least-once analysis queue, pre-provisioned bearer authentication, ownership checks, shared database rate limits, migrations, and explicit per-analysis deletion. It is not a turnkey public deployment: production still requires a provisioned PostgreSQL service, TLS-terminating trusted proxy, secure secrets, a separately supervised worker, backup/restore procedures, and an operator-approved image retention policy. No production deployment was exercised as part of E8–E10.

## Prerequisites

- Python 3.11+ (validated in this checkout with 3.11.9 and 3.12.10).
- Node.js 22.12+ recommended for the current Vite 8 toolchain; `npm ci` from `frontend/`.
- 4+ CPU cores and sufficient RAM for CPU mode. CUDA is optional. The default fine-tuned authenticity backend uses `models/baselines/siglip_frozen_head.pt` plus the `google/siglip-base-patch16-224` vision encoder (downloaded from Hugging Face on first load unless already cached); other model packages/weights may download additional data.
- OCR uses EasyOCR with models stored in `OCR_MODEL_DIR` (default `models/ocr`) and does not download weights at runtime. Provision the detector and recognition weights required for the configured languages in that directory before use; missing weights make EasyOCR unavailable. Tesseract is an optional fallback and requires the Python wrapper, system executable, and configured language packs. Object detection requires Ultralytics and an available YOLO checkpoint. Optional API keys enable Google integrations.

## Local setup (PowerShell)

From repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

Keep default SQLite for a local demonstration. Edit `.env` only for storage/model settings or optional integrations. Set `DEVICE=cpu` if CUDA is not correctly installed. Configure optional keys only after reviewing privacy implications: Google Vision receives the uploaded image; Fact Check Tools receives extracted claim query text. Never commit `.env`.

Apply the current schema before starting the API or worker:

```powershell
python -m alembic upgrade head
```

For local development, the example `.env` explicitly enables `DEV_AUTH_BYPASS=true`; this is accepted only when `APP_ENV` is `development` or `test`. It grants a local development identity and must not be used outside those modes. Production requires `DEV_AUTH_BYPASS=false`, an explicit `SECRET_KEY` of at least 32 bytes, provisioned scrypt hashes in `AUTH_USERS_JSON`, explicit CORS origins and trusted hosts, and a PostgreSQL `postgresql+asyncpg://` URL. Generate a password hash with `python -m backend.app.security_cli --user <stable-user-id>` and place the resulting JSON in the secret configuration. Never pass a plaintext password through a command-line argument or commit generated credentials.

The repository includes the fine-tuned head at `models/baselines/siglip_frozen_head.pt`; this small file is included in source control. The default `AUTHENTICITY_BACKEND=finetuned_siglip` pairs that head with `AUTHENTICITY_BACKBONE_MODEL=google/siglip-base-patch16-224`; the first backend load needs Hugging Face access to retrieve the backbone. To select the previous hosted image classifier instead, set `AUTHENTICITY_BACKEND=huggingface` and configure `AUTHENTICITY_MODEL` (optionally `AUTHENTICITY_MODEL_REVISION`). The raw model prediction records the raw-score argmax ('likely_ai_generated' or 'likely_authentic'); the application classification label defaults strictly to 'inconclusive' unless a verified calibration artifact and domain validation evidence are provided. These scores are not calibrated probabilities. The trained head was evaluated on a small CIFAKE subset; that result does not establish generalization to current image generators or real-world images.

Start the API from the repository root:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, run the separate durable worker from the repository root:

```powershell
python -m backend.app.workers.analysis_worker
```

The worker owns a renewable database lease that serializes the full analysis pipeline (one active GPU/model analysis at a time). Multiple API processes share PostgreSQL-backed jobs, queue capacity, ownership data, and rate-limit counters; run one or more worker processes under a supervisor, but the shared lease intentionally keeps effective model concurrency at one. The worker should receive the same database, storage, model and secret configuration as the API. SIGINT/SIGTERM allows the current analysis to finish before the worker exits; an abrupt worker loss is recovered after lease expiry and uses bounded retries.

In a third terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to `http://localhost:8000`. Development docs: `http://127.0.0.1:8000/docs`; health: `/api/v1/health`. Production disables interactive API docs. For a separate frontend host, set `VITE_API_URL` to the API origin and include the exact frontend origin in `CORS_ORIGINS`.

## Demonstration flow

1. Confirm `/api/v1/health` returns a database connection.
2. Open the UI, select a JPEG/PNG/WebP/TIFF/GIF/BMP image below configured byte and pixel limits, and click Analyze.
3. Follow the backend-reported current/completed stages; do not interpret the stage count as model confidence.
4. Inspect authenticity, forensic cues, OCR/claims, evidence, provenance availability, content indicators, regional data status, category cues and limitations.
5. Open `/api/v1/analysis/{id}/report` and compare each result to its stage status and evidence IDs. Refreshing the UI restores the report while the server retains the analysis.

Use only images whose source/license and ground truth are documented. The repository's `test_data/fake_news_test.jpg` has no checked-in source/license/ground-truth record and is not approved as a presentation sample. Current unit tests generate synthetic images in memory; they are for code-path testing, not detector evaluation.

## Database, storage and operations

SQLite is the default for local development and tests. Production configuration requires PostgreSQL because the durable shared worker/rate-limit lifecycle depends on a multi-process database. Apply migrations with `python -m alembic upgrade head` before deploying either API or worker; production startup refuses a missing or outdated migration revision. Normal application startup does not call `Base.metadata.create_all`. Back up the database and `STORAGE_PATH` together. Deleting an analysis removes its database rows and files from the constrained per-analysis storage roots; the orphan scanner is report-only. There is no automatic analysis retention/deletion policy. Expired app-generated `.uploading` staging files are removed after `TEMP_UPLOAD_MAX_AGE_SECONDS`.

Job delivery is at-least-once. The report row is keyed by the stable analysis ID and artifact names are per-analysis, so recovery after a worker crash following report persistence acknowledges the terminal report instead of creating another one. An interruption before report persistence can repeat provider calls; exactly-once external execution is not guaranteed. Queue capacity is bounded by `ANALYSIS_QUEUE_MAX_OUTSTANDING`; a full queue returns HTTP 429 with `Retry-After`. Rate-limit state is shared in the database and fails closed with HTTP 503 if it is unavailable. There is no Dockerfile/Compose or cloud-specific supervisor configuration in this repository; deployment orchestration remains operator-owned.

## Troubleshooting

- `422`: file extension, MIME/signature, image decode, upload-byte limit, or maximum pixel limit failed.
- `DATA_UNAVAILABLE`: expected for unsupported OCR/model/region/data configuration; inspect stage warnings and install the required optional model/data only when its license and source are understood.
- OCR unavailable: install EasyOCR or system Tesseract plus every language code configured in `OCR_LANGUAGES`.
- YOLO unavailable: install the declared backend requirements and ensure `VISION_MODEL` points to an existing/local or permitted checkpoint.
- Model startup/download error: check `MODEL_CACHE_DIR`, available disk/RAM, Hugging Face access, and `HF_TOKEN` if model access requires it.
- Fact-check/provenance unavailable: these are optional and require their corresponding credentials and outbound connectivity.
- Heatmap unavailable: requires a loaded supported authenticity model and successful Grad-CAM; a heatmap shows model attribution, not proof of manipulation.
- Browser CORS error on a separately hosted UI: set exact frontend origin in `CORS_ORIGINS` and configure `VITE_API_URL`.

## Tests/build

From the repository root, run the protected-artifact-safe backend selection:

```powershell
python -m pytest backend/tests -q -rs --ignore-glob="backend/tests/test_phase*.py" --ignore=backend/tests/test_authenticity_calibration.py --ignore=backend/tests/integration
python -m compileall -q backend/app backend/migrations
```

The ignored phase/calibration tests may inspect protected research manifests, predictions or calibration records and are not part of routine CI. In `frontend/`, run `npm ci`, `npm run typecheck`, `npm test`, `npm run build`, and `npm run test:e2e` after installing the Playwright Chromium runtime with `npx playwright install chromium`.
