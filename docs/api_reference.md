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
| `POST /claims` | JSON `ClaimExtractionRequest` (`text`, `sentences`, `language`, `source`, `max_claims`, `ocr_confidence`) | `200` `ClaimExtractionResponse` with extracted claims, offsets, eligibility, confidence, and method; `422` invalid input; `429` rate limit exceeded; `500` internal extraction failure |
| `POST /fact-check` | JSON `FactCheckRequest` (`claims`, `language_code`, `max_results_per_claim`) | `200` `FactCheckResponse` with third-party reviews, publisher ratings, review URLs, claim verifications, and persistent caching; `401` unauthorized; `422` validation error; `429` rate limit exceeded |
| `POST /nlp/language` | JSON `LanguageDetectionRequest` (`text`, `top_k`) | `200` `LanguageDetectionResponse` with document-level language identification across 20 languages, top-k candidates, softmax confidence, runtime, warnings, limitations; `401` unauthorized; `422` invalid input; `429` rate limit exceeded |
| `POST /provenance` | No request schema implemented | `501 NOT_IMPLEMENTED` stub; not usable standalone endpoint |

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

### Readiness and lifecycle counts
- `total_models`: Total number of configured models reported (5).
- `loaded_count`: Number of models currently resident in memory (`loaded is True`).
- `available_count`: Number of models verified on disk or local cache and available for on-demand initialization (`readiness == "available"`).
- `ready_count`: Total operational count (`readiness in ("ready", "available")`), distinguishing on-disk availability from missing checkpoints or missing dependencies.

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
  "available_count": 5,
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
```### Example no-text response (`200 OK`)
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
---

## Standalone Claim Extraction endpoint (`POST /api/v1/claims`)

The Standalone Claim Extraction endpoint extracts candidate factual assertions, attributed quotations, and statistical propositions from submitted text using the deterministic TruthLens claim extraction engine (`conservative_rules_v2`).

> [!IMPORTANT]
> **Explicit Verification Boundary:** Claim extraction identifies candidate statements for downstream review; it does **NOT** decide whether claims are true or false, does not query external fact-check providers, does not call search engines, and does not execute language models.

### Endpoint signature
- **Method:** `POST`
- **Path:** `/api/v1/claims`
- **Authentication:** Bearer token (`Principal = Depends(get_current_principal)`). Development bypass is permitted only when `DEV_AUTH_BYPASS=true` and `APP_ENV` is `development` or `test`.
- **Rate limiting:** Scoped to `"claims"` using database-backed fixed-window rate limiter (`RATE_LIMIT_CLAIMS_PER_MINUTE`, default 30 requests/min).
- **Request model:** `ClaimExtractionRequest`
- **Response model:** `ClaimExtractionResponse`

### Request schema (`ClaimExtractionRequest`)

| Field | Type | Description |
|---|---|---|
| `text` | string (optional) | Source text string (max 10,000 characters). Required if `sentences` is omitted. Enables sentence splitting and original character offset computation. |
| `sentences` | array[string] (optional) | Pre-split list of sentence strings (max 100 items). |
| `language` / `language_code` | string (optional) | ISO 639-1 language code (e.g. `"en"`, `"hi"`). Validated against supported 20-language catalog. |
| `source` | string (optional) | Text source modality (defaults to `"text"`). |
| `ocr_confidence` | float (optional) | Upstream OCR confidence score ($0.0 \dots 1.0$), if extracted from OCR text. |
| `max_claims` | int (optional) | Maximum candidate claims to extract (clamped between 1 and 50, default 5). |
| `min_confidence` | float (optional) | Minimum heuristic extraction confidence threshold (default 0.1). |

### Response schema (`ClaimExtractionResponse`)

| Field | Type | Description |
|---|---|---|
| `status` | string | Execution status: `"completed"`, `"unavailable"`, or `"failed"`. |
| `claims` | array[`ClaimResponse`] | List of extracted candidate statements. |
| `total_claims` / `candidate_count` | int | Total number of candidate claims identified. |
| `accepted_count` / `eligible_count` | int | Number of claims deemed `ELIGIBLE` for fact-checking. |
| `method` / `extraction_method` | string | Extraction engine identifier (`"conservative_rules_v2"`). |
| `runtime_ms` / `processing_time_ms` | int | Processing duration in milliseconds. |
| `warnings` | array[string] | Operational caveats and non-fatal warnings. |

#### Claim item schema (`ClaimResponse`)

| Field | Type | Description |
|---|---|---|
| `claim_id` | string | Stable, deterministic ID derived from SHA-256 of normalized text (`"claim_<hex16>"`). |
| `claim` / `original_text` | string | Raw extracted sentence/statement. |
| `normalized_text` | string | NFC Unicode-normalized text with collapsed whitespace. |
| `classification` | string | Statement type: `"FACTUAL"`, `"QUOTE"`, `"OPINION"`, `"QUESTION"`, `"SLOGAN"`, or `"AMBIGUOUS"`. |
| `verification_eligibility` | string | Eligibility status: `"ELIGIBLE"` or `"INELIGIBLE"`. |
| `eligibility_reason` | string | Deterministic explanation for verification eligibility or rejection. |
| `confidence` | float | Heuristic extraction score ($0.0 \dots 1.0$). **Explicitly not a truth probability.** |
| `confidence_semantics` | string | Literal string explaining score semantics. |
| `entities` | array[string] | Proper noun entity candidates identified via capitalization patterns. |
| `dates` | array[string] | Extracted date expressions. |
| `locations` | array[string] | Recognized geographical location names. |
| `numbers` | array[string] | Detected numeric values and percentages. |
| `source_start` | int or null | 0-based start character offset into the original raw `text` input string. |
| `source_end` | int or null | 0-based end character offset into the original raw `text` input string. |
| `language` / `language_code` | string | Pinned language code (e.g. `"en"`, `"hi"`, or `"unknown"`). |
| `warnings` | array[string] | Claim-specific caveats (e.g. low OCR confidence, unverified language). |

### Offset indexing semantics
- **Raw input string indexing:** When `text` is supplied as a string, `source_start` and `source_end` index directly into the unmodified input text such that `text[source_start:source_end] == original_text`.
- **Pre-split sentence lists:** When `sentences` is supplied as a list of strings without raw string context, `source_start` and `source_end` are `null`.

### Eligibility and filtering rules
1. **Attributed quotations (`QUOTE`):** Sentences with quotes and attribution verbs (`said`, `announced`, `reported`) are marked `ELIGIBLE`. Unattributed quotes are marked `INELIGIBLE`.
2. **Factual assertions (`FACTUAL`):** Sentences with past/present auxiliary verbs or numbers are marked `ELIGIBLE` if word count $\ge 3$. Short fragments (< 3 words) are marked `AMBIGUOUS` and `INELIGIBLE`.
3. **Questions (`QUESTION`):** Sentences ending in `?` or starting with question words (`what`, `why`, `क्या`, etc.) are marked `INELIGIBLE`.
4. **Opinions (`OPINION`):** Sentences with subjective terms (`I think`, `should`, `best`, `worst`, `मुझे लगता है`) are marked `INELIGIBLE`.
5. **Slogans (`SLOGAN`):** Hashtags, promotional phrases, and calls to action (`#Save...`, `Boycott...`) are marked `INELIGIBLE`.

### Example success response (`200 OK`)
```json
{
  "status": "completed",
  "claims": [
    {
      "claim_id": "claim_a1b2c3d4e5f67890",
      "claim": "The Health Ministry announced 50 new hospitals in 2025.",
      "original_text": "The Health Ministry announced 50 new hospitals in 2025.",
      "normalized_text": "The Health Ministry announced 50 new hospitals in 2025.",
      "classification": "FACTUAL",
      "verification_eligibility": "ELIGIBLE",
      "eligibility_reason": "Text contains a factual assertion candidate.",
      "claim_type": "factual",
      "confidence": 0.81,
      "confidence_semantics": "heuristic extraction score; not truth probability",
      "entities": ["Health Ministry"],
      "dates": ["2025"],
      "locations": [],
      "numbers": ["50", "2025"],
      "source": "text",
      "source_start": 0,
      "source_end": 55,
      "source_text_reference": "ocr_sentence_0",
      "source_region_refs": [],
      "source_regions": [],
      "extraction_method": "conservative_rules_v2",
      "language": "en",
      "language_code": null,
      "language_name": null,
      "language_source": null,
      "language_model_id": null,
      "language_model_revision": null,
      "language_confidence": null,
      "language_detection_status": "provided_by_client",
      "language_detection_outcome": "provided_by_client",
      "ocr_confidence": null,
      "translated_text": null,
      "warnings": []
    }
  ],
  "total_claims": 1,
  "candidate_count": 1,
  "accepted_count": 1,
  "eligible_count": 1,
  "method": "conservative_rules_v2",
  "extraction_method": "conservative_rules_v2",
  "runtime_ms": 2,
  "processing_time_ms": 2,
  "warnings": []
}
```

---

## Standalone Fact-Checking endpoint (`POST /api/v1/fact-check`)

The Standalone Fact-Checking endpoint retrieves published third-party fact-check reviews matching submitted factual propositions using the Google Fact Check Tools API and a persistent, bounded local cache.

> [!IMPORTANT]
> **Attribution and Non-Fabrication Guarantees:**
> - TruthLens **NEVER** fabricates fact-check reviews. All reviews originate from external third-party fact-checking organizations indexed by Google Fact Check Tools.
> - Third-party publisher ratings reflect the independent findings of those organizations and are **NOT** TruthLens assessments or truth probabilities.
> - Review URLs are strictly sanitized to preserve HTTPS navigation while stripping authentication tokens, tracking tags (`utm_*`, `gclid`), and sensitive parameters.

### Endpoint signature
- **Method:** `POST`
- **Path:** `/api/v1/fact-check`
- **Authentication:** Bearer token (`Principal = Depends(get_current_principal)`). Development bypass is permitted only when `DEV_AUTH_BYPASS=true` and `APP_ENV` is `development` or `test`.
- **Rate limiting:** Scoped to `"fact-check"` using database-backed fixed-window rate limiter (`RATE_LIMIT_FACTCHECK_PER_MINUTE`, default 30 requests/min).
- **Request model:** `FactCheckRequest`
- **Response model:** `FactCheckResponse`

### Request schema (`FactCheckRequest`)

| Field | Type | Description |
|---|---|---|
| `claims` | array[string] | List of claim propositions to check (1 to 20 items; each item 5 to 2000 characters). |
| `language_code` | string (optional) | BCP-47 / ISO 639 language tag (e.g. `"en"`, `"hi"`, `"en-US"`). Must match regex `^[a-z]{2,3}(-[A-Z]{2})?$` and be present in supported language catalog. |
| `max_results_per_claim` | int (optional) | Maximum reviews to retrieve per claim (clamped between 1 and 10, default 5). |

### Response schema (`FactCheckResponse`)

| Field | Type | Description |
|---|---|---|
| `status` | string | Execution status: `"success"`, `"partial"`, `"failed"`, `"no_results"`, `"unavailable"`, or `"skipped"`. |
| `provider` | string | External provider name (`"google_fact_check_tools"`). |
| `claims` | array[`FactCheckClaimItem`] | Deduplicated list of third-party fact-check review records. |
| `total_results` | int | Total number of unique review records returned. |
| `queries_performed` | int | Count of live external provider queries executed during this call. |
| `cached_queries_count` | int | Count of queries resolved directly from persistent cache. |
| `runtime_ms` | int | Total processing duration in milliseconds. |
| `warnings` | array[string] | Operational caveats, skipped ineligible claim counts, or provider quota alerts. |
| `claim_verifications` | array[`ClaimVerificationItem`] | Per-claim verification summaries mapped 1-to-1 to input `claims`. |
| `note` | string | Explicit disclaimer emphasizing third-party attribution. |

#### Claim verification item schema (`ClaimVerificationItem`)

| Field | Type | Description |
|---|---|---|
| `request_index` | int | 0-based index matching the claim position in the input request. |
| `claim_id` | string or null | Deterministic SHA-256 identifier (`"claim_<hex16>"`). |
| `original_claim` | string | Raw claim proposition submitted in the request. |
| `normalized_claim` | string | Whitespace-collapsed, normalized claim text. |
| `eligibility_status` | string | `"ELIGIBLE"` or `"INELIGIBLE"`. |
| `eligibility_reason` | string or null | Reason why claim was accepted or skipped. |
| `verification_status` | string | `"COMPLETED"`, `"SKIPPED"`, or `"UNAVAILABLE"`. |
| `verdict` | string | Aggregate review verdict: `"SUPPORTED"`, `"CONTRADICTED"`, `"INSUFFICIENT_EVIDENCE"`, `"UNVERIFIABLE"`, or `"RETRIEVAL_UNAVAILABLE"`. |
| `publisher_rating_summary` | string or null | Summary of publisher ratings for this claim. |
| `verdict_semantics` | string | Explicit semantics notice explaining that verdict summarizes publisher ratings. |
| `reasoning_summary` | string or null | Plain-language synthesis of available reviews. |
| `supporting_evidence_ids` | array[string] | Evidence IDs of reviews rating the claim as true/accurate. |
| `contradicting_evidence_ids` | array[string] | Evidence IDs of reviews rating the claim as false/misleading. |
| `conflicting_evidence_ids` | array[string] | Evidence IDs of reviews with conflicting ratings. |
| `cache_hit` | bool | `true` if this claim was resolved from cache; `false` otherwise. |
| `confidence` | float or null | Confidence score (`null`, explicitly uncalibrated). |
| `retrieval_error_category` | string or null | Error classification if retrieval failed (`PROVIDER_RATE_LIMIT`, `PROVIDER_TIMEOUT`, etc.). |
| `limitations` | array[string] | Explicit operational boundaries and limitations. |

### Caching and deduplication architecture
- **Persistent storage:** SQLite database stored at `settings.STORAGE_PATH / "cache" / "factcheck_cache.sqlite3"`.
- **Cache key:** SHA-256 hash combining normalized claim query text, language code, and provider name (`"fc_cache_<hex32>"`).
- **TTL expiration:** Configured via `FACTCHECK_CACHE_TTL_SECONDS` (default: 86,400s / 24 hours). Expired entries are evicted automatically on read.
- **Bounded capacity:** Configured via `FACTCHECK_CACHE_MAX_ENTRIES` (default: 5,000 entries). LRU eviction discards oldest entries when capacity is exceeded.
- **In-flight request deduplication:** Concurrent requests for the same query share an `asyncio.Lock` per cache key, guaranteeing that only a single upstream provider request is issued while other concurrent callers await and consume the cached result.
- **Safe caching policy:** Only successful searches (`success`) and verified clean zero-result searches (`no_results`) are cached. Transient network errors, rate limits, timeouts, and authorization failures are **never** cached.

### Example success response (`200 OK`)
```json
{
  "status": "success",
  "provider": "google_fact_check_tools",
  "claims": [
    {
      "claim_id": "claim_89445170d105ad36",
      "evidence_id": "fc_8576d1ad62719a86e92b",
      "query": "The city opened 25 clinics in 2025.",
      "claimed_text": "The city opened 25 clinics in 2025.",
      "claimant": "Public Official",
      "claim_date": "2025-06-01",
      "claim_match": "strong_match",
      "publisher": "FactCheck Organization",
      "publisher_site": "factcheck.org",
      "url": "https://factcheck.org/review/123",
      "title": "Claim regarding 2025 city clinics evaluated",
      "review_date": "2025-06-02",
      "textual_rating": "False",
      "rating_category": "false",
      "language": "en"
    }
  ],
  "total_results": 1,
  "queries_performed": 1,
  "cached_queries_count": 0,
  "runtime_ms": 142,
  "warnings": [],
  "claim_verifications": [
    {
      "request_index": 0,
      "claim_id": "claim_89445170d105ad36",
      "original_claim": "The city opened 25 clinics in 2025.",
      "normalized_claim": "The city opened 25 clinics in 2025.",
      "eligibility_status": "ELIGIBLE",
      "eligibility_reason": "Text contains a factual assertion candidate.",
      "verification_status": "COMPLETED",
      "verdict": "CONTRADICTED",
      "publisher_rating_summary": "CONTRADICTED",
      "verdict_semantics": "summary_of_third_party_publisher_ratings_not_truthlens_finding",
      "reasoning_summary": "Relevant third-party fact-check review(s) rate the claim as false or misleading.",
      "supporting_evidence_ids": [],
      "contradicting_evidence_ids": ["fc_8576d1ad62719a86e92b"],
      "conflicting_evidence_ids": [],
      "cache_hit": false,
      "confidence": null,
      "retrieval_error_category": null,
      "limitations": [
        "Verdict summarizes third-party fact-check ratings; it is not an independent TruthLens investigation."
      ]
    }
  ],
  "note": "Fact-check results come from the Google Fact Check Tools API. They reflect third-party reviews, not TruthLens assessments."
}
```

---

## Language identification endpoint (`POST /api/v1/nlp/language`)

The Standalone Language Identification endpoint exposes the local XLM-RoBERTa 20-language sequence classifier (`papluca/xlm-roberta-base-language-detection`) for document-level text language identification and top-$k$ candidate ranking.

### Endpoint signature
- **Method:** `POST`
- **Path:** `/api/v1/nlp/language`
- **Authentication:** Bearer token (`Principal = Depends(get_current_principal)`). Development bypass is permitted only when `DEV_AUTH_BYPASS=true` and `APP_ENV` is `development` or `test`.
- **Rate limiting:** Scoped to `"language"` via fixed-window rate limiter (`RATE_LIMIT_LANGUAGE_PER_MINUTE`, default 60 requests/min).
- **Request model:** `LanguageDetectionRequest`
- **Response model:** `LanguageDetectionResponse`

### Request schema (`LanguageDetectionRequest`)

| Field | Type | Bounds | Description |
|---|---|---|---|
| `text` | string | 1 to 10,000 characters | Text to analyze. Rejects empty and whitespace-only strings with HTTP 422. |
| `top_k` | int (optional) | 1 to 20 (default: 3) | Number of top candidate languages to return ranked by softmax score. |

### Response schema (`LanguageDetectionResponse`)

| Field | Type | Description |
|---|---|---|
| `status` | string | Execution status: `"completed"`, `"insufficient_text"`, `"unavailable"`, or `"failed"`. |
| `outcome` | string | Fine-grained operational outcome: `"success"`, `"undetermined"`, `"insufficient_input"`, `"no_text"`, `"unavailable"`, `"failed"`. |
| `detected_language` | string | Two-letter ISO 639-1 code of identified primary language, or `"unknown"`. |
| `language_code` | string or null | Primary language code if identified confidently; else `null`. |
| `language_name` | string or null | Full English name of identified primary language (e.g. `"English"`, `"Hindi"`); else `null`. |
| `confidence` | float or null | Softmax probability ($0.0 \dots 1.0$) of primary prediction if confident ($\ge 0.70$); else `null`. |
| `top_languages` | array[`LanguageCandidate`] | Top-$k$ candidate languages ranked by softmax score. |
| `top_k_candidates` | array[`LanguageCandidate`] | Alias for `top_languages`. |
| `model_id` | string | Model identifier (`"papluca/xlm-roberta-base-language-detection"`). |
| `model_revision` | string or null | Pinned Git commit hash (`"9865598389ca9d95637462f743f683b51d75b87b"`). |
| `device` | string or null | Execution device (`"cpu"` or `"cuda"`). |
| `runtime_ms` | int | Model forward pass runtime in milliseconds. |
| `processing_time_ms` | int | Total end-to-end processing time in milliseconds. |
| `input_characters` | int | Length of input text in characters. |
| `input_truncated` | bool | `true` if input exceeded maximum characters (4,096) or tokens (256); else `false`. |
| `confidence_semantics` | string | Notice: `"Maximum softmax probability from the model; not calibrated."` |
| `warnings` | array[string] | Operational caveats, truncation notices, or noisy-text warnings. |
| `limitations` | array[string] | Documented architectural boundaries. |
| `unavailable_reason` | string or null | Human-readable explanation when `status` is `"unavailable"`. |

#### Candidate item schema (`LanguageCandidate`)

| Field | Type | Description |
|---|---|---|
| `language_code` | string | ISO 639-1 two-letter code (`"en"`, `"hi"`, `"fr"`, etc.). |
| `language` | string or null | Alias for `language_code`. |
| `language_name` | string | Full English name (`"English"`, `"Hindi"`, `"French"`, etc.). |
| `confidence` | float | Softmax score ($0.0 \dots 1.0$). |

### Supported languages catalog (20 languages)
Arabic (`ar`), Bulgarian (`bg`), German (`de`), Greek (`el`), English (`en`), Spanish (`es`), French (`fr`), Hindi (`hi`), Italian (`it`), Japanese (`ja`), Dutch (`nl`), Polish (`pl`), Portuguese (`pt`), Russian (`ru`), Swahili (`sw`), Thai (`th`), Turkish (`tr`), Urdu (`ur`), Vietnamese (`vi`), Chinese (`zh`).

### Example success response (`200 OK`)
```json
{
  "status": "completed",
  "outcome": "success",
  "detected_language": "en",
  "language_code": "en",
  "language_name": "English",
  "confidence": 0.999841,
  "top_languages": [
    {
      "language_code": "en",
      "language": "en",
      "language_name": "English",
      "confidence": 0.999841
    },
    {
      "language_code": "de",
      "language": "de",
      "language_name": "German",
      "confidence": 0.000085
    },
    {
      "language_code": "fr",
      "language": "fr",
      "language_name": "French",
      "confidence": 0.000032
    }
  ],
  "top_k_candidates": [
    {
      "language_code": "en",
      "language": "en",
      "language_name": "English",
      "confidence": 0.999841
    },
    {
      "language_code": "de",
      "language": "de",
      "language_name": "German",
      "confidence": 0.000085
    },
    {
      "language_code": "fr",
      "language": "fr",
      "language_name": "French",
      "confidence": 0.000032
    }
  ],
  "model_id": "papluca/xlm-roberta-base-language-detection",
  "model_revision": "9865598389ca9d95637462f743f683b51d75b87b",
  "device": "cpu",
  "runtime_ms": 14,
  "processing_time_ms": 15,
  "input_characters": 102,
  "input_truncated": false,
  "confidence_semantics": "Maximum softmax probability from the model; not calibrated.",
  "warnings": [],
  "limitations": [
    "One document-level prediction is returned; mixed-language regions are not separately identified.",
    "Softmax confidence is not calibrated and does not measure truthfulness or authenticity."
  ],
  "unavailable_reason": null
}
```
