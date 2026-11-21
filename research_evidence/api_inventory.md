# TruthLens REST API Inventory

> **Document Status:** Verified Against Actual FastAPI Endpoints & Schemas
> **Base URL:** `http://127.0.0.1:8000/api/v1`
> **Primary Source Code:** `backend/app/api/v1/router.py`, `backend/app/api/v1/endpoints/`

---

## 1. Verified Route Surface

```
/api/v1/
├── POST /analyze                                  # Submit image for asynchronous analysis pipeline
├── GET  /analysis/{analysis_id}                   # Poll analysis lifecycle status and progress
├── GET  /analysis/{analysis_id}/report            # Retrieve final structured analysis report JSON
├── GET  /analysis/{analysis_id}/image             # Retrieve stored original image
├── GET  /analysis/{analysis_id}/heatmap           # Retrieve Grad-CAM heatmap artifact
├── GET  /analysis/{analysis_id}/forensics/ela     # Retrieve Error Level Analysis (ELA) artifact
├── GET  /health                                   # System health, database connection, GPU status
├── GET  /health/models                            # Model availability and cache presence
├── POST /ocr                                      # Explicit HTTP 501 stub (executed in pipeline)
├── POST /fact-check                               # Explicit HTTP 501 stub (executed in pipeline)
├── POST /provenance                               # Explicit HTTP 501 stub (executed in pipeline)
└── GET  /models                                   # Placeholder returning empty model list
```

---

## 2. Endpoint Specifications

### 2.1 Analysis Dispatch & Polling

#### `POST /api/v1/analyze`
- **Description:** Accepts an image upload, validates byte length, MIME type, magic bytes, and Pillow decoding. Creates `Analysis`, `ImageAsset`, and `SystemEvent` records, saves the image with an opaque UUID name, and dispatches the 19-stage pipeline asynchronously via `asyncio.create_task`.
- **Request Format:** `multipart/form-data` with field `file`.
- **Constraints:** Maximum 20 MB, maximum 40,000,000 pixels.
- **Response:** HTTP 201 with `analysis_id`, `status: "queued"`, and metadata.

#### `GET /api/v1/analysis/{analysis_id}`
- **Description:** Returns the current processing lifecycle status, progress count, current step, and timing.
- **Status Enum:** `queued`, `preprocessing`, `analyzing`, `completed`, `failed`, `partial`.

#### `GET /api/v1/analysis/{analysis_id}/report`
- **Description:** Retrieves the completed structured report JSON, including authenticity scores, evidence items, forensic indicators, OCR text, candidate claims, fact-check results, virality heuristics, and limitations.

### 2.2 Artifact Retrieval

#### `GET /api/v1/analysis/{analysis_id}/image`
- **Description:** Streams the uploaded original image within path-containment security bounds.

#### `GET /api/v1/analysis/{analysis_id}/heatmap`
- **Description:** Streams the class-conditioned Grad-CAM heatmap overlay.

#### `GET /api/v1/analysis/{analysis_id}/forensics/ela`
- **Description:** Streams the Error Level Analysis forensic diagnostic image.

### 2.3 Health & Diagnostics

#### `GET /api/v1/health`
- **Description:** Reports database connectivity, system uptime, and GPU device resolution.

#### `GET /api/v1/health/models`
- **Description:** Checks presence and loadability of model weights and caches without full memory allocation.
