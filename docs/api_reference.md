# API reference

Base prefix: `/api/v1`. FastAPI serves live OpenAPI at `/openapi.json`, Swagger UI at `/docs`, and ReDoc at `/redoc`; those are authoritative for exact generated models.

| Method and path | Request | Response / behavior |
|---|---|---|
| `POST /analyze` | Multipart form field `image`; JPEG, PNG, WebP, GIF, BMP, TIFF allowed subject to actual signature, extension, size and decode validation | `201` `{analysis_id, status:"queued"}`; `422` invalid upload; `500` storage failure |
| `GET /analysis/{analysis_id}` | UUID-like analysis ID | `AnalysisOut`: lifecycle status, progress current/completed/total, image metadata, model runs and times; `404` unknown ID |
| `GET /analysis/{analysis_id}/report` | Analysis ID | Stored structured report; queued/processing progress response; failed/missing/legacy report message; `404` unknown ID |
| `GET /analysis/{analysis_id}/image` | Analysis ID | Validated original image bytes, path confined beneath configured originals directory; `404` unavailable/unknown |
| `GET /analysis/{analysis_id}/heatmap?variant=overlay\|raw` | Variant defaults to `overlay`; allowed `overlay` or `raw` | PNG artifact; `422` unsupported variant, `404` missing artifact/analysis |
| `GET /analysis/{analysis_id}/forensics/ela` | Analysis ID | ELA visualization PNG; `404` if missing |
| `GET /health` | None | App version/environment, database connection status, GPU info, uptime |
| `GET /health/models` | None | Read-only readiness list for model load state, dependencies, local cache/checkpoint. Does not run inference |
| `GET /models` | Bearer token auth | Real model registry returning `ModelRegistryResponse`: inventories SigLIP, YOLOv8, EasyOCR, Tesseract, and XLM-RoBERTa status, readiness, and device without eager loading |
| `POST /ocr` | Multipart form field `image` (JPEG, PNG, WebP, etc.), optional query param `languages` (comma-separated ISO-639) | `200` `OCRResponse` with extracted text, bounding boxes, language, engine, runtime; `413` oversized upload, `422` invalid/empty/corrupt image, `500` OCR execution failure, `504` timeout |
| `POST /fact-check`, `/provenance` | No request schema implemented | `501 NOT_IMPLEMENTED` stubs; not usable standalone endpoints |

## Report response

The report is JSON with `analysis_id`, lifecycle `status`, submitted/generated timestamps, `input`, `image`, `preprocessing`, `authenticity`, `metadata`, `quality`, `forensics`, `objects`, `ocr`, `text_normalized`, `claims`, `fact_checks`, `provenance`, `xai`, `hashes`, `evidence`, `assessment`, `fusion`, `summary`, `explanation`, `virality`, `regional_spread`, `social_impact`, processing duration, `stages`, `module_statuses`, warnings, limitations and disclaimer. Missing modules remain null/unavailable; the `summary` keeps authenticity, claims, provenance, spread and potential impact distinct.

### Authenticity object schema

The `authenticity` dictionary distinguishes between model inference and application decisions:
- `label`: Application-level verdict (`inconclusive`, `likely_authentic`, or `likely_ai_generated`). Defaults to `inconclusive` when uncalibrated or when domain membership is unvalidated.
- `model_prediction`: Raw argmax binary classification (`likely_ai_generated`, `likely_authentic`, or `inconclusive`).
- `ai_generated_score` / `authentic_score`: Raw softmax scores ($0.0$ to $1.0$).
- `raw_logits`: Dict containing uncalibrated model logits (`real`, `fake`).
- `calibrated_ai_generated_probability`: Scaled probability populated when a valid calibration artifact is configured.
- `calibrated_candidate_label`: Candidate classification under temperature scaling thresholds.
- `calibration_status`: `not_calibrated`, `validated_temperature_scaling`, or `invalid_artifact`.
- `domain_validation_status`: `not_evaluated` or `validated_for_input_domain`.
- `model_name`, `model_version`, `runtime_ms`, `warnings`, `score_semantics`, `caveat`.

Pipeline stage states currently serialize lowercase: `pending`, `running`, `completed`, `partial`, `failed`, `skipped`, `unavailable`. Analysis lifecycle values include `queued`, `processing`, `completed`, `partial` (derived from stored report), and `failed`. Stage records contain timestamps, runtime, stable stage ID, analysis correlation ID, safe error category/message, warning/limitation arrays, and a report output reference. The pipeline accounts for 20 stages, including upload-time preprocessing. See `system_architecture.md` and `/docs` for current implementation details.

---

## Model registry endpoint (`GET /api/v1/models`)

The Model Registry endpoint provides structured, non-eager inspection of all configured machine learning and optical character recognition models in TruthLens.

### Endpoint signature
- **Method:** `GET`
- **Path:** `/api/v1/models`
- **Authentication:** Bearer token (`Principal = Depends(get_current_principal)`). Development bypass is permitted only when `DEV_AUTH_BYPASS=true` and `APP_ENV` is `development` or `test`.
- **Rate limiting:** Scoped to `"models"` using database-backed fixed-window rate limiter (`RATE_LIMIT_MODELS_PER_MINUTE`, default 60 requests/min).
- **Response model:** `ModelRegistryResponse`

### Critical status semantics
The endpoint distinguishes strictly between memory presence, disk cache, and readiness:

| Status (`status`) | Readiness (`readiness`) | Memory Resident (`loaded`) | Semantic Meaning |
|---|---|---|---|
| `loaded` | `ready` | `true` | The model weights are currently resident in RAM/VRAM and ready for immediate inference passes. |
| `available_not_loaded` | `available` | `false` | Required packages and local model checkpoints/binaries are verified on disk, but weights are not loaded into memory. Ready for on-demand initialization. |
| `cached_not_loaded` | `available` | `false` | Hugging Face model snapshot directory exists in `MODEL_CACHE_DIR`, but weights have not been loaded into memory. Ready for on-demand initialization. |
| `checkpoint_missing` | `unavailable` | `false` | A configured local checkpoint file (e.g. `siglip_frozen_head.pt` or `yolov8n.pt`) was not found at its configured location. |
| `dependency_unavailable` | `unavailable` | `false` | A required Python package (`transformers`, `ultralytics`, `easyocr`, `pytesseract`) or system executable (`tesseract`) is missing. |
| `download_required` | `unavailable` | `false` | The model is configured, but its weights are absent from local storage/cache and would require an external download. |
| `failed` | `failed` | `false` | The model previously encountered an unrecoverable exception during load/initialization. |
| `disabled` | `unavailable` | `false` | The model or engine is explicitly disabled in application settings. |

### Security and privacy guarantees
- **Zero eager loading:** Requesting `GET /api/v1/models` never calls `.load()` on any service or detector.
- **Zero inference passes:** No forward passes, dummy images, or GPU tensor allocations occur.
- **Strict path redaction:** Local absolute filesystem paths (such as `D:\...` or `/home/...`) are sanitized; only public filenames or repository IDs are returned.
- **Zero credential disclosure:** API keys, database credentials, secret signing keys, and exception stack traces are never exposed in response bodies.

### Model inventory
The registry reports five core engines:
1. `authenticity`: Production SigLIP authenticity detector (Frozen ViT-B/16 backbone + Linear classification head).
2. `object_detection`: Ultralytics YOLOv8n object detector for scene understanding.
3. `ocr_easyocr`: EasyOCR primary optical character recognition engine (CRAFT + CRNN).
4. `ocr_tesseract`: Tesseract OCR fallback engine (LSTM).
5. `language_detection`: XLM-RoBERTa 20-language document-level sequence classifier.

### Example success response (`200 OK`)
```json
{
  "status": "ok",
  "models": [
    {
      "id": "authenticity",
      "name": "SigLIP Authenticity Detector",
      "task": "authenticity_classification",
      "provider": "Google / TruthLens",
      "architecture": "SigLIP (Frozen ViT-B/16 Backbone + Linear Classifier Head)",
      "model_id": "truthlens/siglip-cifake-frozen-head",
      "version": "v1.0-frozen-head",
      "configured": true,
      "loaded": false,
      "readiness": "available",
      "status": "cached_not_loaded",
      "device": null,
      "capabilities": [
        "authenticity_prediction",
        "gradcam_explainability",
        "calibrated_probabilities"
      ],
      "details": {
        "backend": "finetuned_siglip",
        "backbone_model_id": "google/siglip-base-patch16-224",
        "checkpoint_present": true,
        "checkpoint_filename": "siglip_frozen_head.pt",
        "cached": true
      },
      "reason": "Model weights are cached locally; ready for on-demand loading."
    },
    {
      "id": "object_detection",
      "name": "YOLOv8 Object Detector",
      "task": "object_detection",
      "provider": "Ultralytics",
      "architecture": "YOLOv8n (Object Detection CNN)",
      "model_id": "yolov8n.pt",
      "version": "yolov8n-official",
      "configured": true,
      "loaded": false,
      "readiness": "available",
      "status": "available_not_loaded",
      "device": null,
      "capabilities": [
        "bounding_box_detection",
        "scene_context"
      ],
      "details": {
        "checkpoint_filename": "yolov8n.pt",
        "checkpoint_present": true,
        "ultralytics_available": true
      },
      "reason": "YOLO weights are available locally; ready for on-demand loading."
    },
    {
      "id": "ocr_easyocr",
      "name": "EasyOCR Text Recognition",
      "task": "optical_character_recognition",
      "provider": "JaidedAI / EasyOCR",
      "architecture": "CRAFT (Detection) + CRNN (Recognition)",
      "model_id": "easyocr",
      "version": "EasyOCR",
      "configured": true,
      "loaded": false,
      "readiness": "available",
      "status": "available_not_loaded",
      "device": null,
      "capabilities": [
        "multilingual_text_detection",
        "bounding_box_extraction"
      ],
      "details": {
        "configured_languages": [
          "en",
          "hi"
        ],
        "gpu_configured": true,
        "easyocr_available": true
      },
      "reason": "EasyOCR package is installed; ready for on-demand initialization."
    },
    {
      "id": "ocr_tesseract",
      "name": "Tesseract OCR Fallback",
      "task": "optical_character_recognition",
      "provider": "Google / Tesseract",
      "architecture": "Tesseract LSTM OCR Engine",
      "model_id": "tesseract",
      "version": "Tesseract",
      "configured": true,
      "loaded": false,
      "readiness": "available",
      "status": "available_not_loaded",
      "device": null,
      "capabilities": [
        "offline_text_fallback",
        "bounding_box_extraction"
      ],
      "details": {
        "pytesseract_available": true,
        "binary_present": true
      },
      "reason": "Tesseract binary and Python wrapper are available for fallback."
    },
    {
      "id": "language_detection",
      "name": "XLM-RoBERTa Language Identification",
      "task": "language_identification",
      "provider": "papluca / Hugging Face",
      "architecture": "XLM-RoBERTa Sequence Classification (20 Languages)",
      "model_id": "papluca/xlm-roberta-base-language-detection",
      "version": "9865598389ca9d95637462f743f683b51d75b87b",
      "configured": true,
      "loaded": false,
      "readiness": "available",
      "status": "cached_not_loaded",
      "device": null,
      "capabilities": [
        "20_language_classification",
        "offline_cache_only"
      ],
      "details": {
        "supported_languages_count": 20,
        "pinned_revision": "9865598389ca9d95637462f743f683b51d75b87b",
        "cached": true
      },
      "reason": "Model weights are cached locally; ready for on-demand loading."
    }
  ],
  "total_models": 5,
  "loaded_count": 0,
  "ready_count": 5,
  "environment": "development",
  "message": "Model registry reports configured services without eager loading."
}
```
### Partial availability / error examples

#### Checkpoint missing (`status="checkpoint_missing"`, `readiness="unavailable"`)
```json
{
  "id": "authenticity",
  "name": "SigLIP Authenticity Detector",
  "status": "checkpoint_missing",
  "readiness": "unavailable",
  "loaded": false,
  "device": null,
  "reason": "Checkpoint file 'siglip_frozen_head.pt' was not found on disk."
}
```
#### Dependency missing (`status="dependency_unavailable"`, `readiness="unavailable"`)
```json
{
  "id": "object_detection",
  "name": "YOLOv8 Object Detector",
  "status": "dependency_unavailable",
  "readiness": "unavailable",
  "loaded": false,
  "device": null,
  "reason": "Ultralytics package is not installed."
}
```
#### Initialization failed (`status="failed"`, `readiness="failed"`)
```json
{
  "id": "language_detection",
  "name": "XLM-RoBERTa Language Identification",
  "status": "failed",
  "readiness": "failed",
  "loaded": false,
  "device": null,
  "reason": "Language detector initialization failed (OSError)."
}
```
---

## Standalone OCR endpoint (`POST /api/v1/ocr`)

The Standalone OCR endpoint provides optical character recognition and text extraction for uploaded images, utilizing the existing EasyOCR engine with automatic fallback to Tesseract OCR when appropriate.

### Endpoint signature
- **Method:** `POST`
- **Path:** `/api/v1/ocr`
- **Authentication:** Bearer token (`Principal = Depends(get_current_principal)`). Development bypass is permitted only when `DEV_AUTH_BYPASS=true` and `APP_ENV` is `development` or `test`.
- **Rate limiting:** Scoped to `"ocr"` using database-backed fixed-window rate limiter (`RATE_LIMIT_OCR_PER_MINUTE`, default 20 requests/min).
- **Request parameters:**
  - `image` (form data file, required): Image file to extract text from (supported formats: JPEG, PNG, WebP, GIF, BMP, TIFF).
  - `languages` (query param, optional): Comma-separated ISO-639-1 language codes (e.g. `en`, `hi`, `en,hi`). If omitted, uses configured default OCR languages (`settings.OCR_LANGUAGES`).
- **Response model:** `OCRResponse`

### Response schema (`OCRResponse`)

| Field | Type | Description |
|---|---|---|
| `status` | string | Extraction status: `success`, `no_results`, `partial`, `failed`, or `unavailable`. |
| `text` | string | Full extracted text concatenated across detected text blocks. |
| `language` | string | Active configured languages string (e.g. `"en"` or `"en,hi"`). |
| `results` | array[`OCRBlockResponse`] | Array of detected text blocks with individual text, confidence, and bounding boxes. |
| `confidence` | float | Overall confidence score ($0.0 \dots 1.0$), averaged across extracted text blocks. |
| `engine` | string | Engine that generated the final result: `"easyocr"` or `"tesseract"`. |
| `processing_time_ms` | float | Wall-clock execution time in milliseconds. |
| `has_text` | boolean | True if at least one text block with usable text was detected. |
| `no_text_reason` | string or null | Reason code if no usable text was found (e.g. `"NO_TEXT_FOUND"`, `"IMAGE_TOO_SMALL"`). |
| `warnings` | array[string] | List of execution warnings (e.g. fallback triggered, bounding box coordinate clipping). |

#### Block schema (`OCRBlockResponse`)
| Field | Type | Description |
|---|---|---|
| `text` | string | Extracted text segment for this block. |
| `confidence` | float | Detection and recognition confidence score ($0.0 \dots 1.0$). |
| `box` | array[int] | Integer bounding box `[x1, y1, x2, y2]` in original image pixel coordinates. |

### Engine fallback behavior
1. **Primary engine (`easyocr`):** Text extraction is attempted first with EasyOCR using CRAFT text detection and CRNN character recognition.
2. **Fallback trigger:** If EasyOCR encounters an exception (e.g. CUDA OOM, corrupt model weights, initialization failure) or produces zero usable text blocks, the service automatically falls back to Tesseract OCR if enabled and available.
3. **Graceful degradation:** If neither engine produces text or Tesseract is unavailable, a clean `status="no_results"` response is returned without raising an HTTP 500 error.

### Error response status codes

| HTTP Status | Reason | Schema / Content |
|---|---|---|
| `413 Payload Too Large` | Uploaded file size exceeds `settings.MAX_UPLOAD_SIZE_MB` (default 10 MB). | `{"detail": "File size exceeds maximum allowed size (X bytes)"}` |
| `422 Unprocessable Content` | File fails security validation (empty file, invalid MIME type, mismatched magic bytes, unreadable/decompression bomb, or dimensions outside bounds). | `{"detail": "Image validation failed: <reason>"}` |
| `429 Too Many Requests` | Rate limit for `"ocr"` scope exceeded. | Standard rate limit headers and 429 detail. |
| `500 Internal Server Error` | Unexpected unhandled exception during OCR extraction. | Safe, sanitized error detail with no paths or stack traces leaked. |
| `504 Gateway Timeout` | OCR processing exceeded `settings.OCR_TIMEOUT_SECONDS` (default 60 seconds). | `{"detail": "OCR processing timed out after X seconds"}` |

### Example success response (`200 OK`)
```json
{
  "status": "success",
  "text": "BREAKING NEWS: FACT CHECK VERIFIED",
  "language": "en",
  "results": [
    {
      "text": "BREAKING NEWS:",
      "confidence": 0.942,
      "box": [12, 18, 185, 45]
    },
    {
      "text": "FACT CHECK VERIFIED",
      "confidence": 0.915,
      "box": [12, 52, 240, 80]
    }
  ],
  "confidence": 0.9285,
  "engine": "easyocr",
  "processing_time_ms": 42.15,
  "has_text": true,
  "no_text_reason": null,
  "warnings": []
}
```
### Example no-text response (`200 OK`)
```json
{
  "status": "no_results",
  "text": "",
  "language": "en",
  "results": [],
  "confidence": 0.0,
  "engine": "easyocr",
  "processing_time_ms": 18.3,
  "has_text": false,
  "no_text_reason": "NO_TEXT_FOUND",
  "warnings": []
}
```
