# TruthLens Functionality and Workflow

## 1. User roles and workflows

There is no login, account model or role-based access control in the code. The frontend has a general upload-and-inspect workflow. The following personas are inferred from the academic use case, not implemented as separate permissions:

| Persona | Expected need | Access in current implementation |
|---|---|---|
| Student/research demonstrator | Submit a sample and explain pipeline output | Browser UI, local API; no account |
| Researcher/developer | Inspect status, report JSON, artifacts and health | UI plus API/OpenAPI; no admin identity |
| Panel/reviewer | Understand results and limitations during a demo | Typically observes a user-operated session; no dedicated reviewer mode |

No administrative, notification, user-management or ownership workflows were found.

## 2. Main user journey

1. Open the Vite-hosted browser UI.
2. Choose an image supported by the UI/API; the upload must be within configured byte and pixel limits.
3. Submit it. The frontend sends multipart form data to `POST /api/v1/analyze`.
4. If validation succeeds, the API generates an analysis ID and responds with queued state.
5. The frontend polls status. A progress number represents completed stages out of the frontend/API's declared total, not confidence.
6. When a report is ready, the UI fetches full report JSON and any original/heatmap URL needed by the results view.
7. The reviewer inspects authenticity output, supporting signals, extracted text, possible candidate claims, external reviews if configured, provenance, stage statuses, warnings and limitations.

If the model or optional engines are absent, the report can be partial/unavailable. A missing external match is not evidence that a claim is false or true. The user should not treat an uploaded image's model tag as ground truth.

## 3. Feature inventory

### Upload, validation and image lifecycle — implemented

- **Input:** one uploaded image, original filename and request MIME metadata.
- **Validation:** reads at most configured max bytes plus one; checks extension, MIME, magic bytes and image decoding, plus maximum pixel constraints.
- **Processing:** sanitizes supplied name, creates opaque storage filename and analysis UUID, calculates SHA-256 and perceptual hash, reads dimensions/format, writes the original, then preprocessing generates thumbnail and resized files and quality statistics.
- **Persistence:** `Analysis`, `ImageAsset`, and `SystemEvent` records are committed before pipeline kickoff.
- **Output:** `analysis_id`, queued status, later status/report and image retrieval.
- **Errors/edges:** upload read, size, unsupported format or decode, and storage errors can reject the request. Preprocessing failure is recorded as a warning and later work can continue.
- **Code:** `backend/app/api/v1/endpoints/analyze.py`, `backend/app/core/security.py`, `backend/app/services/preprocessing/service.py`.

### Analysis progress and report retrieval — implemented

- `GET /api/v1/analysis/{id}` returns status, current step, completed steps, fixed total-step value, percentage, image metadata, model runs and timestamps.
- `GET /api/v1/analysis/{id}/report` returns persisted full report JSON or a status-specific interim/error response.
- The report can be `partial` where some stages fail or are unavailable. Stage state must be considered alongside output values.
- Code: `backend/app/api/v1/endpoints/analysis.py`, `backend/app/services/reporting/builder.py`, `frontend/src/services/api.ts`.

### Authenticity prediction — implemented & evaluated

- **Input:** preprocessed/decoded image in the expected model format ($224 \times 224$ normalized RGB).
- **Model:** production deployment retains the frozen baseline `models/baselines/siglip_frozen_head.pt` (SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`) paired with `google/siglip-base-patch16-224`.
- **Output:** class probabilities (calibrated or uncalibrated) and operational authenticity tag with explicit uncertainty band ($[0.35, 0.65]$).
- **Calibration:** post-hoc temperature scaling fitted on held-out calibration pools ($N=13,000$).
- **Benchmarked:** systematically benchmarked across 3 external datasets ($N=10,385$ locked test set); fine-tuned candidate S2 (`truthlens/siglip-base-v2-s2`) was rejected for production due to negative transfer on late-2026 generators.
- **Dependencies:** PyTorch/Transformers, checkpoint, accessible/cached backbone and compatible device.
- **Code:** `backend/app/services/authenticity/detector.py`, `backend/app/services/authenticity/calibration.py`, `docs/phase7_evaluation_report.md`.

### Metadata, quality, hashing — implemented

- Extracts available image format/size/EXIF data; not all images retain metadata.
- Calculates byte-level SHA-256 and perceptual hash for similarity workflows.
- Computes resolution, JPEG/quality indicators and basic image statistics.
- These values can support inspection and artifact identity; none proves source or manipulation.
- Code: `metadata/service.py`, `quality/service.py`, `hashing/service.py`, `preprocessing/service.py`.

### Classical forensics — implemented as heuristics

- Includes Error Level Analysis, noise, JPEG, edge, texture and resampling signals.
- Signals include limitations and are sensitive to format/encoding/history.
- No signal is a calibrated manipulation probability; no reliable edit classifier is claimed.
- ELA artifact is available via `GET /api/v1/analysis/{id}/forensics/ela` if created.
- Code: `backend/app/services/forensics/analyzer.py`; artifact route in `analysis.py`.

### OCR — pipeline feature, optional runtime dependencies

- Attempts EasyOCR first; can fall back to Tesseract where installed and configured.
- Default OCR languages are English and Hindi. The output includes engine, status, blocks/text and reasons for no readable text or unavailable processing.
- OCR output is not guaranteed correct. No normalized character/word error evaluation is present.
- Standalone `POST /api/v1/ocr` is a 501 stub; OCR is called from the analysis pipeline.
- Code: `backend/app/services/ocr/service.py`, `backend/app/services/orchestrator.py`; UI `OcrSection` in `frontend/src/App.tsx`.

### Text normalization and claim extraction — implemented heuristically

- Normalizes Unicode/whitespace and extracts structural items such as URLs, hashtags, mentions, numbers and sentences.
- Rule-based claim extraction prioritizes candidate statements and may identify surface entities, dates or locations.
- It does not determine truth and can miss implied, multilingual, sarcastic or context-dependent claims.
- Code: `backend/app/services/text/normalizer.py`, `backend/app/services/claims/extractor.py`, UI `ClaimsSection`.

### Fact-check retrieval — optional external evidence

- For eligible extracted claims, the Google Fact Check Tools adapter can search external reviews when a key and network are available.
- Bounded claim/result/time settings and retry configuration are defined in settings/service.
- Returned reviews/ratings are provider context; they can be stale, mismatched or unavailable. The system does not conduct a full independent fact investigation.
- No working standalone fact-check endpoint is exposed; `/api/v1/fact-check` returns 501.
- Code: `backend/app/services/factcheck/service.py`; UI `FactCheckSection`.

### Provenance lookup — optional external evidence

- Google Vision Web Detection may search for web entities/pages matching an image, if configured.
- Results are leads, not proof of the earliest/original source or date.
- Optional image transmission has privacy implications.
- Standalone `/api/v1/provenance` is a 501 stub; pipeline adapter lives in `backend/app/services/provenance/service.py`.

### Object detection and explainability — optional

- YOLO object detection can return object labels/boxes if Ultralytics and weights are available. It identifies objects, not factual accuracy.
- Grad-CAM can render target-class activation overlays if the model exposes compatible layers/gradients. The heatmap indicates attribution, not manipulation regions or causal proof.
- UI sections are conditional on output; endpoints serve existing files only.
- Code: `vision/detector.py`, `xai/gradcam.py`, UI `ForensicsSection` and `ImageViewer`.

### Evidence collection, fusion and explanations — implemented rules

- Converts model, forensic, OCR, metadata, fact-check, provenance, XAI and object outputs into typed evidence entries.
- Evidence records include source/type/status/direction/strength or related metadata as defined in `evidence/schema.py`.
- Fusion combines eligible completed/partial model-direction evidence and conflict/uncertainty rules. Phase 6 regression documentation says unavailable/failed/skipped evidence cannot vote and current forensic/context evidence remains informational.
- Explanations summarize evidence categories and caveats. There is no demonstrated accuracy benefit from fusion.
- Code: `evidence/schema.py`, `fusion/engine.py`, `explanation/generator.py`.

### Virality, regional and social impact — limited prototype indicators

- Virality uses hand-weighted text/content cues, not a trained model or observed propagation data.
- Regional numeric rankings remain unavailable; language or other contextual cues are not evidence of a region's susceptibility.
- Social-impact categories use exact-token/topic cues; they do not measure actual spread or harm.
- Datasets and validated predictive models for virality, content spread and regional spread are future scope. No observed outcome dataset is present.
- Code: `virality/engine.py`, `regional/engine.py`, `social_impact/engine.py`.

### Health, artifacts and model listing

- `/api/v1/health` checks database query and reports uptime/GPU/version/environment.
- `/api/v1/health/models` reports model/dependency/checkpoint state without necessarily loading all models.
- Image, heatmap and ELA routes check analysis and constrain resolved paths to the relevant storage directory.
- `/api/v1/models` returns an empty placeholder list; `/api/v1/ocr`, `/fact-check`, `/provenance` are not standalone implementations.

## 4. Functional and non-functional requirements

### Functional requirements evidenced in current implementation

1. Accept supported images through multipart upload.
2. Reject invalid, oversized, unsupported or over-pixel-limit files.
3. Create an analysis ID and persist status/image metadata.
4. Run analysis stages asynchronously in the server process and record progress.
5. Return summary status and structured report for an analysis ID.
6. Serve original/heatmap/ELA artifacts only when associated with an analysis and within configured paths.
7. Report stage warnings/unavailable states in output.
8. Provide health and model availability endpoints.

### Non-functional expectations and actual gaps

- **Traceability:** model-run and event records plus report stage statuses support auditing, but no external immutable audit log exists.
- **Privacy:** local persistence by default; optional providers may receive data; no deletion/retention control.
- **Reliability:** graceful partial stage capture exists, but in-process tasks can be lost on restart.
- **Security:** upload/path checks exist; authentication, ownership, rate limiting and public hardening are not evidenced.
- **Performance:** historical microbenchmarks are narrow and not service SLAs; no representative load test was conducted.
- **Usability/accessibility:** UI is present, but no user study or formal accessibility evaluation was found.
- **Scalability:** no queue or horizontal job coordination; deployment guide limits current use to trusted local academic demonstration.

## 5. API and frontend interaction map

| Frontend helper/component | Request | Use |
|---|---|---|
| `uploadImage(file)` | `POST /api/v1/analyze` | Submit image |
| `getAnalysisStatus(id)` | `GET /api/v1/analysis/{id}` | Poll state/progress |
| `getAnalysisReport(id)` | `GET /api/v1/analysis/{id}/report` | Load complete report |
| `getAnalysisImageUrl(id)` | `GET /api/v1/analysis/{id}/image` | Display stored original |
| `getHeatmapUrl(id)` | `GET /api/v1/analysis/{id}/heatmap` | Show heatmap if available |
| `getHealth()` | `GET /api/v1/health` | Health/status information |
| `getModels()` | `GET /api/v1/models` | Currently placeholder list |

The frontend uses a Vite development proxy for `/api`; production cross-origin configuration requires `VITE_API_URL` and backend CORS origins as documented in deployment guide.

## 6. Error handling and edge cases

- Invalid signature/MIME/extension, failed decode, excessive size/pixels → validation rejection.
- Storage write failure → storage error.
- Metadata/hash extraction can be best-effort with warning/fallback fields.
- Preprocessing can fail non-fatally; analysis may proceed with warning.
- Optional services may return unavailable/failed/skipped and final report can be partial.
- Unknown analysis ID → not found behavior.
- Invalid heatmap variant → 422; missing artifact → 404.
- Optional provider credentials/network issues must not be interpreted as negative evidence.
- Server restart can interrupt work because there is no durable queue.

## 7. Source map

The complete directory/module map and architecture are in `TECHNICAL_ARCHITECTURE.md`. Primary workflow sources are `backend/app/api/v1/endpoints/analyze.py`, `backend/app/api/v1/endpoints/analysis.py`, `backend/app/services/orchestrator.py`, `backend/app/services/reporting/builder.py`, `frontend/src/App.tsx`, and `frontend/src/services/api.ts`.
