# TruthLens REST API Inventory

> **Document Status:** Verified Against FastAPI Endpoints & Schemas  
> **Inspection Date:** 2026-10-20  
> **Base URL:** `http://127.0.0.1:8000/api/v1`  
> **Primary Source Code:** `backend/app/api/v1/endpoints/`, `backend/app/schemas/`

---

## 1. Overview of API Surface

The TruthLens backend provides an asynchronous, OpenAPI-documented REST interface. All endpoints are versioned under the `/api/v1` prefix and implement strict Pydantic v2 schema validation.

```
/api/v1/
├── analyze/
│   └── POST /upload                 # Submit image for 19-stage analysis
├── analysis/
│   ├── GET /{id}/status             # Poll stage execution progress
│   ├── GET /{id}/report             # Retrieve comprehensive report
│   └── GET /history                 # Retrieve paginated past analyses
├── datasets/
│   ├── GET /synthetic/virality      # View simulated virality scenarios
│   ├── GET /synthetic/regional      # View simulated regional scenarios
│   ├── GET /synthetic/impact        # View simulated social impact scenarios
│   └── GET /external/cifake         # View CIFAKE benchmark manifest
└── health
    └── GET /                        # System health and GPU availability
```

---

## 2. Endpoint Specifications

### 2.1 Image Ingestion & Analysis Dispatch

#### `POST /api/v1/analyze/upload`
- **Description:** Accepts an image file, validates magic bytes, computes SHA-256 hash, creates database records, stores raw bytes, and asynchronously dispatches the 19-stage pipeline.
- **Request Format:** `multipart/form-data`
  - `file`: Binary image data (`image/jpeg`, `image/png`, `image/webp`).
  - Max Size: Configured via `settings.MAX_UPLOAD_SIZE_BYTES` (default: 10 MB).
- **Responses:**
  - `200 OK`: Upload accepted and analysis dispatched.
    ```json
    {
      "analysis_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "status": "processing",
      "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "message": "Analysis started successfully"
    }
    ```
  - `400 Bad Request`: Empty file or corrupted image stream.
  - `415 Unsupported Media Type`: Non-allowed MIME type (e.g. `.pdf`, `.exe`, `.gif`).
  - `413 Payload Too Large`: Image exceeds size threshold.

---

### 2.2 Analysis Status & Report Retrieval

#### `GET /api/v1/analysis/{analysis_id}/status`
- **Description:** Real-time polling endpoint for frontend progress bars and status indicators.
- **Path Parameter:** `analysis_id` (UUID string).
- **Response `200 OK`:**
  ```json
  {
    "analysis_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "status": "processing",
    "current_step": "authenticity",
    "completed_steps": ["preprocessing", "metadata", "quality"],
    "total_steps": 19,
    "elapsed_ms": 1420
  }
  ```

#### `GET /api/v1/analysis/{analysis_id}/report`
- **Description:** Returns the complete, auditable multimodal report once pipeline reaches `completed` state.
- **Path Parameter:** `analysis_id` (UUID string).
- **Response `200 OK` Schema (`ReportResponse`):**
  - `analysis_id`: UUID string
  - `timestamp`: ISO-8601 UTC timestamp
  - `image_info`: Dimensions, format, perceptual hashes (pHash, dHash, aHash), SHA-256
  - `authenticity`:
    - `label`: `AUTHENTIC` | `AI_GENERATED` | `UNCERTAIN`
    - `confidence`: Float $[0.0, 1.0]$
    - `probabilities`: `{"real": float, "fake": float}`
    - `model_name`: Model identifier
  - `metadata`: EXIF tags, camera details, editing software flags
  - `forensics`: ELA score, noise variance, JPEG grid consistency
  - `ocr`: Extracted text, detected language script, bounding boxes
  - `claims`: Extracted factual assertions and subject-predicate pairs
  - `fact_checks`: Verified matching debunks from IFCN signatories via Google Fact Check API
  - `provenance`: Historical web appearances and matching source URLs
  - `xai`: Relative URL path to Grad-CAM heatmap visualization
  - `fusion`: Global credibility assessment, final classification, conflict flags
  - `virality`: Risk score $[0, 100]$, `data_status: "DATA_UNAVAILABLE"`, risk factors
  - `regional_spread`: Top high-exposure Indian states, linguistic alignment
  - `social_impact`: Severity tiers across Communal, Electoral, Health, Financial hazards

#### `GET /api/v1/analysis/history`
- **Description:** Returns paginated historical analyses for the dashboard archive.
- **Query Parameters:** `limit` (default: 20), `offset` (default: 0).
- **Response `200 OK`:** List of analysis summary objects with timestamps, thumbnails, and verdicts.

---

### 2.3 Benchmark Datasets & Inspection Endpoints

*(Active when `DEMO_MODE=True`)*

- `GET /api/v1/datasets/synthetic/virality`: Paginated rows from `data/synthetic/virality_scenarios.csv`.
- `GET /api/v1/datasets/synthetic/regional`: Paginated rows from `data/synthetic/regional_spread_scenarios.csv`.
- `GET /api/v1/datasets/synthetic/impact`: Paginated rows from `data/synthetic/social_impact_scenarios.csv`.
- `GET /api/v1/datasets/external/cifake`: Summary statistics and sample records from `data/external/cifake/manifest.json`.

---

### 2.4 System Health & Diagnostics

#### `GET /api/v1/health`
- **Description:** Microservice liveness and dependency probe.
- **Response `200 OK`:**
  ```json
  {
    "status": "healthy",
    "version": "0.1.0",
    "app_env": "development",
    "database_connected": true,
    "gpu_available": true,
    "gpu_device": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "vram_total_mb": 4096,
    "timestamp": "2026-10-20T18:24:00Z"
  }
  ```
