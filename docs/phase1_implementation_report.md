# TruthLens Phase 1 Implementation Report

## Summary

Phase 1 connects the existing upload route, 19-stage orchestrator, SigLIP2 detector, evidence fusion, persisted report, and React report view. The report retains the repository's legacy fields and adds input metadata, detector and fusion status, pipeline stage details, warnings, module availability, and generation timestamps. Optional-stage failures produce a `partial` report; a core pipeline failure after analysis creation produces a stable `failed` report.

No files were removed. The checkout is in `work/Truthlens`.

## Actual workflow

```text
POST /api/v1/analyze
  -> bounded upload read and content/type validation
  -> SHA-256, perceptual hash, original storage and preprocessing
  -> queued analysis record
  -> asynchronous pipeline task
  -> metadata, quality, SigLIP2, forensics, OCR, object detection
  -> dependent XAI / text normalization / claims / fact-check stages
  -> provenance and hashes
  -> evidence collection -> fusion -> explanation
  -> virality, regional spread, social impact
  -> structured report persisted as analysis.report_json
  -> GET /api/v1/analysis/{id} and GET /api/v1/analysis/{id}/report
```

Blocking analysis functions run in worker threads. Detector load and inference are serialized through a process-local lock. Claim checks and provenance lookups retain the existing synchronous integrations and now use configurable request timeouts. In-process upload tasks are tracked, exceptions are observed, and application shutdown waits for outstanding analyses.

## API contract

### Upload

`POST /api/v1/analyze` accepts multipart form data with an `image` file. It returns HTTP 201 and:

```json
{"analysis_id":"<uuid>","status":"queued"}
```

Invalid files return HTTP 422 with `detail` and `error_code`. The server checks detected content, supported type, matching extension when provided, Pillow decoding, and the configured size limit. It stores a generated UUID filename and SHA-256 digest.

### Status

`GET /api/v1/analysis/{analysis_id}` returns the existing `AnalysisOut` structure: `analysis_id`, `status`, nested `progress`, optional truth summary, image metadata, model runs, timing, and safe error message. Wire status can be `queued`, `processing`, `completed`, `partial`, or `failed`. The database retains its existing enum; a stored report with optional-stage failures maps the public status to `partial`.

### Report

`GET /api/v1/analysis/{analysis_id}/report` returns the persisted JSON report, including the existing fields (`image`, `authenticity`, `metadata`, `forensics`, `objects`, `ocr`, `fact_checks`, `provenance`, `evidence`, `assessment`, heuristic outputs, and disclaimer) plus:

- `input`, `submitted_at`, and `generated_at`
- `fusion`, `module_statuses`, `stages`, and top-level `warnings`
- `quality`

Unavailable detector output has null label and scores with an explicit status. Empty-evidence fusion is inconclusive with null score/confidence. Local storage paths, GPS internals, and secret-like fields are omitted. Stage status values use the existing lowercase vocabulary (`completed`, `failed`, `unavailable`, `skipped`, `running`, `pending`).

## Files changed

| File | Change |
|---|---|
| `backend/app/api/v1/endpoints/analyze.py`, `backend/app/main.py` | Bounded upload read, safe read/storage errors, and tracked background-task shutdown. |
| `backend/app/api/v1/endpoints/analysis.py` | Public partial status mapping and 19-stage progress total. |
| `backend/app/core/security.py` | Reject mismatched filename extensions when supplied. |
| `backend/app/core/exceptions.py` | Keep unhandled API error details out of responses, including debug mode. |
| `backend/app/config.py`, `.env.example` | Configure detector checkpoint and thresholds plus external API timeouts. |
| `backend/app/services/authenticity/detector.py` | Label-driven scores, safe uncertainty bands, invalid-output rejection, and inference lock. |
| `backend/app/services/orchestrator.py` | Stage lifecycle details, EXIF orientation handling, skipped/unavailable states, partial/failure reports, and safe persistence. |
| `backend/app/services/evidence/schema.py`, `backend/app/services/fusion/engine.py` | Evidence IDs/provenance, duplicate handling, authenticity-only voting, invalid evidence rejection, and null scores when unsupported. |
| `backend/app/services/reporting/builder.py` | Stable report metadata, module statuses, partial derivation, warnings, and sensitive-field/path filtering. |
| `backend/app/services/factcheck/service.py`, `backend/app/services/provenance/service.py` | Configurable timeouts and safer log messages. |
| `frontend/src/App.tsx` | Render actual fusion/detector output, evidence, module and pipeline statuses, and partial/failure reports. Removed unused declarations that blocked TypeScript compilation. |
| `backend/tests/conftest.py`, `backend/tests/test_api.py`, `backend/tests/test_stage3.py`, `backend/tests/test_stage4.py`, `backend/tests/test_phase1.py` | Deterministic detector mocks, temp-directory storage, background-task teardown, upload/report regressions, threshold and fusion checks, and empty-evidence semantics. |

## Test evidence

| Check | Result |
|---|---|
| `python -m pytest backend/tests -q` from repository root | **153 passed, 6 skipped** in 31.48 seconds. |
| `python -m pytest backend/tests/test_stage3.py::TestAuthenticity::test_inference_returns_result -q` | **1 passed** against the locally cached SigLIP2 checkpoint. |
| `git diff --check` | Passed. |
| TypeScript phase of `npm run build` | Passed after removing pre-existing unused declarations. |
| Full frontend production build | **Passed** with Node 24.22.1 / npm 11.19.1 after restoring lockfile dependencies, installing the matching Windows Rolldown binding, and updating the lockfile to patched Vite 8.3.5. Vite reports a CSS import-order warning and a large-chunk advisory; neither prevents the build. |
| `npm audit --prefix frontend` | **0 vulnerabilities** after the compatible lockfile update. |
| EasyOCR CPU inference | **Passed** on `test_data/fake_news_test.jpg`; extracted text with 26 words, English script, and no warnings. EasyOCR 1.7.2 and model weights are installed locally. |
| YOLO CPU inference | **Passed** on `test_data/fake_news_test.jpg`; loaded `yolov8n.pt` and returned 12 boxes across `person` and `traffic light`, with no warnings. Ultralytics 8.4.191 and weights are installed locally. |
| Browser-based rendering smoke check | Not performed; production bundle builds, but no browser-driven upload/report flow was run. |

The API integration test exercises upload through report retrieval with a deterministic mocked detector and verifies that optional-stage failures are represented as partial/unavailable. The direct detector, OCR, and object-detection runs are separate local-model checks. These smoke checks do not validate model accuracy or live external APIs.

An initial test attempt exposed a float32 threshold-edge assertion and an old integration assertion that incorrectly required `completed` when OCR was unavailable. Both were corrected. A subsequent rerun caught a SQLite race from unfinished background tasks; the tasks are now tracked and awaited during test teardown and application shutdown. The final suite passes. The first real integration attempt loaded the previously uncached checkpoint and created a local model cache; later API tests use the mock detector so fresh API tests do not download it. Test image artifacts use a system temporary directory, leaving tracked sample data unchanged.

## Module status

| Module | Status |
|---|---|
| Upload validation, digest, report retrieval | Implemented and covered by backend tests. |
| Orchestrator and fail-soft status reporting | Implemented; mocked API integration passed. |
| Primary detector | Implemented; direct cached-model inference test passed. |
| Evidence fusion | Implemented and covered for empty, duplicate, invalid, and directional evidence. |
| Structured report | Implemented and JSON-serialization/privacy covered by tests. |
| Frontend API/report connectivity | Implemented; TypeScript check passed, browser check blocked by bundler runtime. |
| OCR | Optional; EasyOCR 1.7.2 installed and CPU inference verified on the sample image. |
| Object detection | Optional; Ultralytics 8.4.191 installed and YOLOv8n CPU inference verified on the sample image. |
| Visual-context model | Not integrated; explicitly reported as `not_integrated`. |
| Fact-check and provenance APIs | Optional; live credentials/services were not verified. |
| Virality, regional spread, social impact | Existing heuristic modules retained; output remains labeled by their `data_status` and caveats. |

## Run instructions

From the repository root:

```sh
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Submit an image with:

```sh
curl -F "image=@/path/to/image.jpg" http://127.0.0.1:8000/api/v1/analyze
```

Use the returned `analysis_id` to poll `/api/v1/analysis/{analysis_id}` and retrieve `/api/v1/analysis/{analysis_id}/report`. Set optional credentials in `.env`; no credentials or model scores are embedded in source.

## Known limitations and next priority

- Frontend production build now passes with the portable Node 24.22.1 runtime; the lockfile resolves patched Vite 8.3.5 and the npm audit reports no vulnerabilities. The generated build still emits a CSS import-order warning and a large JavaScript chunk advisory.
- Visual-context analysis remains unintegrated by design. Live fact-check and provenance API calls were not verified because credentials are not configured. The project still uses the pre-existing non-durable in-process job runner.
- The model's class scores and fusion score are uncalibrated; this phase does not establish detector accuracy.
- Database status remains `completed` internally for partial runs to avoid a database enum migration; the API derives public `partial` from the persisted report.
- The orchestrator still uses the existing in-process background task and does not survive process restarts as a durable job.

Next priority: configure local API credentials, integrate a browser-driven upload/report smoke check, and decide whether to scope a separately resourced visual-language model integration. A browser upload/report check and live external API check remain outstanding.
