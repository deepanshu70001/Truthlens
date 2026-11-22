# TruthLens E6/E7 — Forensics and Provenance Implementation Report

**Repository:** `D:/Projects/Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**Starting commit:** `aa9d7a3 feat: Implement E3 Language Identification and E4/E5 Claim Extraction`  
**Milestones:** E6 and E7 only

## 1. Result and preserved baseline

E6 and E7 are **PARTIALLY IMPLEMENTED**. Classical forensic indicators run and pass deterministic tests. The YOLO path and production-model Grad-CAM path are implemented but could not be run with their real local model resources. The C2PA adapter and provenance API are implemented and mocked/contract-tested, but the local C2PA SDK and a signed permitted image are unavailable. Google Vision was not called with live credentials.

At the start of this task, the worktree was clean and the branch was `fix/truthlens-remediation` at `aa9d7a3`. The repository contained ignored local caches, datasets, model assets, storage, and research archives. Their contents were not opened or changed. The only image fixture inspected for C2PA availability was the permitted `backend/tests/fixtures/test_photo.jpg`; it had no `c2pa`, `jumb`, or `manifest` byte marker. No locked-test, calibration, or research artifact was read or run.

No production SigLIP checkpoint, calibration file, authenticity detector, inference route, fusion weight, or decision policy was changed. No model or data set was downloaded. No commit or push was made.

## 2. Architecture audit

### E6

- `forensics/analyzer.py` already implemented ELA, Laplacian residual variation, JPEG block-boundary differences, Canny edge-density variation, Gabor responses, and second-difference zero-crossing rates. Before this change, non-JPEG inputs were still JPEG-recompressed for ELA, non-JPEG JPEG checks used a `skipped` status, image limits were not enforced in the analyzer, and per-signal method/runtime/output metadata was incomplete.
- `vision/detector.py` already used Ultralytics and had pipeline integration. It loaded the configured `yolov8n.pt` identifier through the normal YOLO loader, which can download weights. It did not hash the local model, validate every output box against input dimensions, or represent an empty detection outcome distinctly.
- `xai/gradcam.py` already hooked a final ViT layer and emitted heatmaps. Its unsupported target layer was reported as a failure, and the output dimensions/prediction-preservation behavior did not have focused E6 coverage.
- The repository has no dedicated manipulation-detection model. Classical probes are the only manipulation-adjacent signals and remain non-directional.

### E7

- `provenance/service.py` already queried Google Cloud Vision Web Detection and the orchestrator already passed original bytes to provenance. It had no C2PA parser, no original-file SHA field in the provenance result, and no bounded retry/response/concurrency behavior.
- The standalone `/api/v1/provenance` route was a 501 stub.
- Metadata extraction existed separately. EXIF fields are descriptive source metadata and are not cryptographic provenance.
- The evidence builder previously coupled the provenance card to aggregate Web Detection status and did not emit independent C2PA findings.
- E1 fusion already ignores failed/unavailable evidence and restricts directional authenticity voting to calibrated model evidence. The fusion implementation was left unchanged.

## 3. E6 implementation

### Classical forensic indicators

The analyzer validates non-empty bytes, upload size, decoding, image dimensions and `MAX_IMAGE_PIXELS`. It applies EXIF orientation to an independent working copy and retains the original bytes for other stages. Invalid, corrupt, unsupported, truncated, and over-limit images return safe failed results.

ELA now runs only for JPEG/JPG. Non-JPEG ELA and JPEG-specific checks on non-JPEG inputs return `completed` with `outcome: not_applicable`. JPEG block analysis also reports not-applicable below its minimum size. Noise, edge, and resampling checks report not-applicable when their minimum dimensions are not met. A single signal failure leaves successful signals visible and makes the aggregate partial; artifact-write failure is also partial.

Each signal reports an identifier, status/outcome, bounded uncalibrated score where available, measurement details, method/version, parameters/formula, runtime, limitations, safe error category, and report reference. Scores are indicators in `[0,1]`, never probabilities. The score formulas and parameters are included in the result. ELA uses JPEG quality 90; noise is quadrant Laplacian-residual variation; JPEG uses 8-pixel boundary differences; edge uses Canny `[50,150]` over a 3×3 grid; texture uses four-orientation Gabor response variation; resampling uses second-difference sign zero crossings. These heuristics do not prove editing, AI generation, or authenticity.

### YOLO

YOLO now loads only an existing local weights file and does not auto-download. It records the model filename and SHA-256, caps inference at `imgsz=640` and `max_det=100`, checks image limits, confidence and finite/in-bounds boxes, and states that boxes refer to EXIF-corrected input pixels. Empty output is `completed/no_results`. Object confidence remains detection confidence, not authenticity confidence. Local paths and raw exception text are not exposed.

The configured identity is `yolov8n.pt`. The local Python runtime lacks Ultralytics and no local YOLO weights were found, so only mocked initialization/detection and failure behavior were tested. No real YOLO inference is claimed.

### Grad-CAM

Grad-CAM remains optional and uses the already loaded authenticity model only after successful authenticity inference. The implementation validates source-aligned heatmap/overlay dimensions, preserves the explained model/class metadata, removes hooks, clears gradients and restores the model's prior train/eval mode. It does not alter model weights or classifier logits. Unsupported target layers are `unavailable`.

A toy SigLIP-shaped PyTorch model verified artifact size, unchanged weights, and unchanged pre/post predictions. The production SigLIP checkpoint was not loaded or changed; compatibility with that checkpoint remains unverified. Attribution is not independent truth or manipulation evidence.

## 4. E7 implementation

### C2PA/content credentials

The new `provenance/c2pa_verifier.py` adapter uses the official `c2pa-python` Reader/Context/Settings interface. The dependency is constrained to `c2pa-python>=0.39.0,<0.40.0`; the local runtime does not have that package installed. The adapter disables remote-manifest fetch. Trust verification remains explicitly unverified unless a readable `C2PA_TRUST_ANCHORS_PATH` is configured, in which case the SDK receives those anchors and trust checking is enabled. SDK diagnostics, rather than a local JUMBF/CBOR parser, determine validity. See the [official SDK usage](https://github.com/contentauth/c2pa-python/blob/main/docs/usage.md) and [context settings](https://github.com/contentauth/c2pa-python/blob/main/docs/context-settings.md).

The response keeps these properties separate:

| Property | Reported separately as |
|---|---|
| Manifest present/readable | `manifest_present`, `manifest_readable` |
| Signature validation | `signature_status` |
| Content binding | `content_binding_status` |
| Signer identity/trust | `signer_identity`, `signer_trust_status`, `trust_check_configured` |
| Edit/history context | assertion labels and ingredient count |
| Referenced binary resource | `referenced_resources_status` and counts; current scope is the active manifest thumbnail only |
| Implementation | SDK name/version and runtime |

`status` describes verifier execution. `verification_outcome` describes C2PA findings (`no_manifest`, `verified`, `verification_failed`, `unverified`, or `unavailable`) and is not an E1 stage status. An absent manifest is not evidence of manipulation. Valid signature/content binding does not establish signer trust unless trust is configured and does not establish event truth. A failed signature or content binding is returned as a finding while the verifier execution can still be `completed`.

Tests use a fake SDK module to exercise absent/malformed manifests, success/failure codes, trust configuration, signer identity, ingredients, thumbnail resources, and safe errors. They do not demonstrate real cryptographic verification. No signed permitted asset or installed local verifier was available, so real C2PA verification remains **unverified**.

### Original-file metadata and integrity

The provenance service hashes the original bytes with SHA-256 before any conversion and returns algorithm/value separately. It extracts EXIF camera/software/timestamp/dimension fields from those same bytes, distinguishes no metadata from extraction failure, omits GPS coordinates, and labels the source as EXIF rather than cryptographically verified provenance. Metadata and SDK text are bounded and sanitized for credentials and local paths.

### Google Vision Web Detection and endpoint

The existing provider is retained with credential-based `unavailable`, bounded timeout and total duration, limited retries, a four-request concurrency bound, a 2 MiB response cap, normalized HTTPS-only URLs, and safe provider errors. Sensitive query parameters are removed. Empty provider results are successful `no_results`; timeout and malformed responses are failures. Matching pages/images are external web-match context only.

`POST /api/v1/provenance` now validates size, MIME, magic bytes and decoding through the shared upload validator, retains original bytes, runs the same provenance service with a bounded endpoint wait, and returns typed normalized fields. No authentication or rate-limit policy was added; the route follows the existing API behavior. Mock tests cover provider timeout/success/empty/malformed and upload rejection. No live Google request or credential was used.

## 5. Pipeline, evidence, and frontend

The existing `forensics`, `vision`, `xai`, and `provenance` stages remain separate. Forensics, YOLO, Grad-CAM and provenance receive the unmodified original byte stream or the existing EXIF-corrected image view as appropriate. E6/E7 stages now apply configured runtime bounds, preserve service outcomes in `StageResult`, and include safe module identities. XAI is `unavailable` when its prerequisite model output is absent, not marked as a successful skip.

The provenance evidence builder emits separate C2PA and Web Detection items, even when only one method is available. Forensics, object detections, XAI and provenance findings remain informational. Failed/unavailable checks do not produce directional evidence. No fusion weights, voting eligibility, decision policy, detector output, calibration, or model routing changed. The end-to-end mocked report remains partial when optional E6/E7 checks fail and continues to serialize older report shapes.

The frontend report component now handles partial and unavailable Web Detection independently from C2PA, shows manifest/signature/content-binding/trust findings and the original hash, and tolerates older reports with missing fields. No UI redesign or frontend test framework was added. A production browser/E2E suite remains for E10.

## 6. Files changed

- `.env.example`
- `backend/requirements.txt`
- `backend/app/config.py`
- `backend/app/api/v1/endpoints/stubs.py`
- `backend/app/services/forensics/analyzer.py`
- `backend/app/services/vision/detector.py`
- `backend/app/services/xai/gradcam.py`
- `backend/app/services/provenance/c2pa_verifier.py` (new)
- `backend/app/services/provenance/service.py`
- `backend/app/services/evidence/schema.py`
- `backend/app/services/orchestrator.py`
- `backend/tests/test_e6_e7_contract.py` (new)
- `backend/tests/test_phase2.py`
- `backend/tests/test_phase3.py`
- `backend/tests/test_stage4.py`
- `frontend/src/App.tsx`
- `docs/MODULE_STATUS_REGISTER.md`
- `docs/PROJECT_ROADMAP.md`
- `docs/e6_e7_forensics_provenance_report.md` (this report)

## 7. Validation

Validation was run with the safe, nonprotected test files only. The locked evaluation/research files and calibration-specific tests were not opened or run. The exact final commands and counts are recorded here after the implementation pass:

### Backend

- Focused E6: `python -m pytest -q backend/tests/test_e6_e7_contract.py -k 'forensics or forensic or yolo or gradcam'`: **8 passed, 12 deselected**.
- Focused E7: `python -m pytest -q backend/tests/test_e6_e7_contract.py -k 'c2pa or provenance or web_detection or web_success or metadata'`: **11 passed, 9 deselected**.
- Integrated report: `python -m pytest -q backend/tests/test_e6_e7_contract.py -k integrated_report`: **1 passed, 19 deselected**.
- E6/E7 and relevant E1-E5 regression selection: `python -m pytest -q backend/tests/test_e6_e7_contract.py backend/tests/test_phase2.py backend/tests/test_stage_contract.py backend/tests/test_phase3.py backend/tests/test_stage4.py -k 'e6_e7 or provenance or forensics or yolo or gradcam or failed_directional_evidence or no_results_is_successful_execution'`: **34 passed, 70 deselected**.
- Safe backend regression set: `python -m pytest -q backend/tests/test_e6_e7_contract.py backend/tests/test_phase1.py backend/tests/test_phase2.py backend/tests/test_phase3.py backend/tests/test_phase4.py backend/tests/test_phase5.py backend/tests/test_stage_contract.py backend/tests/test_stage4.py backend/tests/test_services.py backend/tests/test_api.py backend/tests/test_ocr_runtime.py backend/tests/test_language_detector.py backend/tests/test_e4_e5_contract.py`: **224 passed, 2 skipped**. The skips are environment-gated OCR runtime cases. Locked evaluation/research and calibration-specific tests were excluded by scope.
- `python -m compileall -q backend/app`: **passed**.
- `git diff --check`: **passed**; Git printed only its existing LF-to-CRLF advisory for three edited text files.

### Frontend

- Changed component typecheck: `npx tsc --ignoreConfig --noEmit --jsx react-jsx --moduleResolution bundler --module esnext --target es2023 --skipLibCheck --allowSyntheticDefaultImports --types vite/client --allowArbitraryExtensions src/App.tsx`: **passed**.
- Production bundle: `npx vite build --outDir $env:TRUTHLENS_BUILD_OUT` with the output directed to a temporary directory: **passed**. Vite reported the existing CSS import-order and large-chunk warnings.
- Project-wide `npx tsc --noEmit --project tsconfig.app.json`: **failed on existing unrelated frontend issues**: missing `vitest` and `shared/india_regions.json` modules, an implicit `any` in `pipelineProgress.test.ts`, and a missing `AnalysisReport` export in `pipelineProgress.ts`. No reported error points to the changed `App.tsx`.

Pytest emitted the existing `pytest-asyncio` warning that `asyncio_default_fixture_loop_scope` is unset. An initial invocation through the Windows `pytest` launcher failed because it omitted the repository root from `sys.path`; the successful runs used `python -m pytest`.


Two existing `pytest` launcher invocations initially failed because the Windows script did not include the repository root on `sys.path`; all actual validation used `python -m pytest`. Pytest emitted the existing `pytest-asyncio` warning that `asyncio_default_fixture_loop_scope` is not set. The two skipped tests in the safe suite are environment-gated local OCR runtime cases; no downloads were attempted.

Mocked YOLO and C2PA results are test-double evidence only. The toy Grad-CAM run is not production SigLIP inference. Live Google Vision and real C2PA cryptographic verification were not executed. No full backend pass is claimed because protected evaluation/calibration tests were intentionally excluded.

## 8. Remaining blockers and stop boundary

- Install the pinned C2PA SDK in a permitted runtime and verify a permitted signed asset before claiming real C2PA behavior.
- Provision the existing approved YOLO weights and Ultralytics dependency to verify local inference and actual model hash.
- Verify Grad-CAM against the unchanged production SigLIP checkpoint in its approved runtime.
- Provision controlled Google Vision credentials only when a live provider check is authorized.
- A dedicated manipulation detector remains unimplemented and requires its own governed scope.
- Frontend browser/E2E coverage remains assigned to E10.

E8 and later work, research milestones, calibration, locked evaluation, and fusion-policy changes were not started.
