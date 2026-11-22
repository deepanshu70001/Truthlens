# Deployment and reproducibility guide

TruthLens is currently suitable for a trusted local academic demonstration. It has no authentication, retention/deletion workflow, durable background queue, or container deployment files. Do not expose it directly to the public internet. Protect it behind an authenticated, rate-limited reverse proxy and define image retention before any shared deployment.

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

The repository includes the fine-tuned head at `models/baselines/siglip_frozen_head.pt`; this small file is included in source control. The default `AUTHENTICITY_BACKEND=finetuned_siglip` pairs that head with `AUTHENTICITY_BACKBONE_MODEL=google/siglip-base-patch16-224`; the first backend load needs Hugging Face access to retrieve the backbone. To select the previous hosted image classifier instead, set `AUTHENTICITY_BACKEND=huggingface` and configure `AUTHENTICITY_MODEL` (optionally `AUTHENTICITY_MODEL_REVISION`). The raw model prediction records the raw-score argmax ('likely_ai_generated' or 'likely_authentic'); the application classification label defaults strictly to 'inconclusive' unless a verified calibration artifact and domain validation evidence are provided. These scores are not calibrated probabilities. The trained head was evaluated on a small CIFAKE subset; that result does not establish generalization to current image generators or real-world images.

Start the API from the repository root:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to `http://localhost:8000`. Backend docs: `http://127.0.0.1:8000/docs`; health: `/api/v1/health`. For a separate frontend host, set `VITE_API_URL` to the API origin and include the frontend origin in `CORS_ORIGINS`.

## Demonstration flow

1. Confirm `/api/v1/health` returns a database connection.
2. Open the UI, select a JPEG/PNG/WebP/TIFF/GIF/BMP image below configured byte and pixel limits, and click Analyze.
3. Follow the backend-reported current/completed stages; do not interpret the stage count as model confidence.
4. Inspect authenticity, forensic cues, OCR/claims, evidence, provenance availability, content indicators, regional data status, category cues and limitations.
5. Open `/api/v1/analysis/{id}/report` and compare each result to its stage status and evidence IDs. Refreshing the UI restores the report while the server retains the analysis.

Use only images whose source/license and ground truth are documented. The repository's `test_data/fake_news_test.jpg` has no checked-in source/license/ground-truth record and is not approved as a presentation sample. Current unit tests generate synthetic images in memory; they are for code-path testing, not detector evaluation.

## Database, storage and operations

SQLite is the default. PostgreSQL can be selected with an async SQLAlchemy URL, for example `postgresql+asyncpg://...`; provision it separately and keep credentials outside source control. The application currently creates tables using SQLAlchemy `create_all`; there are no migration scripts. Back up the database and `STORAGE_PATH` together. No automatic stale-file cleanup or delete endpoint exists; an operator-managed retention plan is required.

The pipeline runs as an in-process background task. Keep the API process running until analysis finishes. A restart can interrupt active jobs. There is no Dockerfile/Compose setup in this repository and Docker was unavailable in the validation environment; container deployment is not currently supported/documented as ready.

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

From the repository root, run `python -m pytest backend/tests -q` in a normal asyncio-capable environment; see `phase5_e2e_test_report.md` for the sandbox-specific API test startup blocker and the narrower passing command. Build the UI in `frontend/` with `npm ci` then `npm run build`.
