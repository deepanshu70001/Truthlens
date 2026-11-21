# TruthLens Phase 2 Implementation Report

## Audit and outcome

| Module | Existing implementation | Missing or incorrect behavior found | Change made |
|---|---|---|---|
| Classical forensics | Six existing OpenCV signals: `error_level_analysis`, `noise_consistency`, `jpeg_artifacts`, `edge_consistency`, `texture_consistency`, `resampling_clues`. | Caught failures emitted score `0`; ELA recompressed every format without saying so; no ELA artifact; JPEG small-image math could be undefined; six unrelated values were averaged into an apparent overall anomaly score. | Kept the six signal names and calculations, made failures null/explicit, mark JPEG checks skipped for non-JPEG sources, record ELA source format and limitations, optionally save analysis-linked ELA output, validate finite values, and leave `overall_anomaly_score` null. Every forensic evidence item is informational and cannot vote on authenticity. |
| Metadata, quality, hashing | Separate EXIF, quality, upload SHA-256, and pHash services; hash stage also computes aHash, dHash and wHash. | Missing EXIF was worded as common for AI images; hashing left its decoded image handle open. | Reworded metadata absence as non-directional, kept GPS private, and closed decoded images while preserving exact-byte SHA-256 and distinct perceptual hashes. Existing quality metrics remain informational and unchanged. |
| OCR | EasyOCR lazy-loaded for English and Hindi; raw and normalized text, boxes and confidence were returned. | No engine/status field, a load failure aborted the OCR stage, inference failures were not distinguished from no-text images, PyTesseract fallback was absent, and malformed outputs were not validated. | Added explicit status/engine/language configuration, finite confidence and box validation, truthful successful-empty output, and opportunistic PyTesseract fallback when both its Python package and executable are available. Default language configuration remains `en,hi`; only English was exercised in real inference. |
| YOLO | A singleton YOLOv8n object detector returning labels, confidence and integer boxes. | The checkpoint was hard-coded, `device` was ignored, load/output failures had no status, invalid outputs were accepted, and boxes lost fractional precision. | Added `VISION_MODEL` configuration (default `yolov8n.pt`), device-aware prediction, explicit completed/failed/unavailable states, finite/range/coordinate validation, empty-detection success, and original-image pixel boxes with fractional precision. |
| Grad-CAM and artifacts | A Grad-CAM service and orchestrator stage existed, but target-layer selection and token reshaping were generic; the heatmap API endpoint was still a stub. | Incorrect transformer token assumptions could make attribution fail or map the wrong spatial grid. Paths were stripped from reports without fetchable references. | Bound Grad-CAM to the last SigLIP vision block `layer_norm2`, handle square patch tokens and optional CLS tokens, target the detector's selected class, normalize a real gradient attribution, save grayscale map and overlay per analysis, return safe API URLs, and implement PNG routes for overlay/raw/ELA artifacts. Unsupported or failed attribution returns no artifact and does not stop the main report. |
| Orchestrator/report/UI | The existing 19-stage pipeline already invoked forensics, OCR, vision, XAI, hashing and evidence/fusion/report stages. | Returned module failure objects were recorded as completed; forensic scores could vote as manipulation signals; the frontend had no object boxes and showed a heatmap control even when XAI failed. | Stage status now follows structured module status, report partial-state derivation includes partial stages, forensic evidence is informational, OCR engine/status are displayed, YOLO boxes are drawn in original-image coordinates, and the heatmap control appears only for a successful attribution. |

## Result contracts

All statuses use lowercase values: `completed`, `partial`, `failed`, `unavailable`, or `skipped`. A successful OCR/YOLO run with no text or objects is `completed` with empty content.

- **Forensics:** `status`, six `signals` (`name`, `status`, nullable heuristic `score`, `score_type`, `interpretation`, `limitations`, `details`), `overall_anomaly_score: null`, `runtime_ms`, `warnings`, and `caveat`. ELA `details.artifact_url` is present only when an artifact was saved.
- **OCR:** `status`, `engine`, `language_configuration`, `text`/`raw_text`, `normalized_text`, script estimate, `blocks` (`text`, nullable confidence, four-corner bbox), nullable `total_confidence`, word count, runtime, and warnings. Normalization only applies Unicode NFC and whitespace collapsing.
- **YOLO:** `status`, model name, objects (`class_name`, model confidence, `[x1,y1,x2,y2]` original-image pixel coordinates), object count, image dimensions, runtime, warnings and a scene-context caveat.
- **XAI:** `status`, model, method, target class and model score, local paths for internal persistence, API URLs for raw map/overlay, attribution-region boxes, runtime, warnings and a clear non-proof caveat. Local paths are filtered from public report JSON; URLs are retained.

## Orchestrator stages

The existing stage names remain: `metadata`, `quality`, `authenticity`, `forensics`, `ocr`, `vision`, `xai`, `text_normalization`, `claim_extraction`, `fact_check`, `provenance`, `hashing`, `evidence_collection`, `fusion`, `explanation`, `virality`, `regional_spread`, `social_impact`, and `report`. XAI uses the authenticity model instance and only runs when that model and prediction are available. Evidence and report schemas remain in the existing pipeline.

## Files changed for Phase 2

- `.env.example`, `README.md`
- `backend/app/config.py`
- `backend/app/api/v1/endpoints/analysis.py`
- `backend/app/services/forensics/analyzer.py`
- `backend/app/services/metadata/service.py`
- `backend/app/services/hashing/service.py`
- `backend/app/services/ocr/service.py`
- `backend/app/services/vision/detector.py`
- `backend/app/services/xai/gradcam.py`
- `backend/app/services/orchestrator.py`
- `backend/app/services/evidence/schema.py`
- `backend/app/services/reporting/builder.py`
- `backend/tests/conftest.py`, `backend/tests/test_api.py`, `backend/tests/test_stage3.py`, `backend/tests/test_phase2.py`
- `frontend/src/App.tsx`

Earlier Phase 1 files in the checkout were preserved. No academic paper or synopsis was edited.

## Verification

**Mocked/offline tests:**

- `python -m pytest --noconftest backend/tests/test_phase2.py backend/tests/test_stage3.py -k "phase2 or forensics or xai" -q` — **16 passed, 1 skipped, 13 deselected**.
- `python -m pytest --noconftest backend/tests/test_stage4.py::TestEvidenceSchema backend/tests/test_stage4.py::TestFusion -q` — **11 passed**.
- Python `compileall` for changed services and routes — passed.
- `git diff --check` — passed.
- Frontend TypeScript + Vite production build — passed with Node 24.22.1 / Vite 8.3.5. Existing CSS import-order and large-chunk warnings remain.

The full API test suite was attempted, but pytest did not progress beyond its Windows `pytest-asyncio` loop-factory setup (before the first test body). The async API artifact test was added to `test_api.py`, but could not be run in this environment. The API route implementation therefore has compile coverage but not a completed HTTP integration test here. API tests now mock OCR and YOLO to avoid implicit model downloads once the harness is runnable.

**Real inference, separate from mocked tests:**

- Forensics ran all six existing signals on `test_data/fake_news_test.jpg`; it saved an ELA PNG and returned an analysis-linked route.
- EasyOCR 1.7.2 on CPU extracted 26 words with mean engine confidence `0.5606` using English. This is an OCR smoke result, not an accuracy benchmark.
- Ultralytics 8.4.191 / YOLOv8n on CPU returned 12 boxes, including `person` detections, with real confidences and original-image coordinates. This is not authenticity evidence.
- The configured SigLIP2 checkpoint predicted `likely_authentic` on this sample; its `Real` class score was `0.9903`. The actual Grad-CAM run completed for that class and saved both attribution map and overlay. This does not validate classifier accuracy or establish manipulation.

Example outputs from those actual module runs (shortened):

```json
{
  "forensics": {
    "status": "completed",
    "signal_names": ["error_level_analysis", "noise_consistency", "jpeg_artifacts", "edge_consistency", "texture_consistency", "resampling_clues"],
    "overall_anomaly_score": null,
    "ela_artifact_url": "/api/v1/analysis/4d4b1bdb-1bd3-4a95-862d-1cecb9140df9/forensics/ela"
  },
  "ocr": {"status": "completed", "engine": "EasyOCR", "language_configuration": ["en"], "word_count": 26, "mean_confidence": 0.5606},
  "vision": {"status": "completed", "model": "yolov8n.pt", "object_count": 12, "example": {"class_name": "person", "confidence": 0.8758, "bbox": [456.0, 785.0, 568.0, 1023.0]}},
  "authenticity": {"label": "likely_authentic", "scores": {"fake": 0.0097, "real": 0.9903}},
  "xai": {"status": "completed", "method": "grad_cam", "target_class": "Real", "target_score": 0.9903, "overlay_url": "/api/v1/analysis/<analysis-id>/heatmap?variant=overlay"}
}
```

Generated real-model overlay from this sample:

`work/phase2-real-smoke/heatmaps/f0a43604-2890-423b-a96c-2014ee303b5f/overlay.png`

## Remaining limitations

- The real OCR smoke used English only. Hindi/Devanagari remains configured but was not independently smoke-tested. Other Indian languages are not claimed as configured support.
- PyTesseract is an opportunistic fallback only; the Tesseract Python module and executable are not bundled or required. If unavailable, the service reports that state.
- The async API integration suite must be rerun in a working pytest/Windows event-loop environment to verify upload-to-artifact HTTP delivery end to end.
- The forensic indicators are simple content-dependent heuristics with no validated cutoffs. No aggregate manipulation probability is produced.
- Grad-CAM explains model influence, not manipulated pixels. YOLO is scene context only. Existing classifier calibration/accuracy remains unvalidated.
