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
| `GET /models` | None | Placeholder empty model list |
| `POST /ocr`, `/fact-check`, `/provenance` | No request schema implemented | `501 NOT_IMPLEMENTED` stubs; not usable standalone endpoints |

## Report response

The report is JSON with `analysis_id`, lifecycle `status`, submitted/generated timestamps, `input`, `image`, `preprocessing`, `authenticity`, `metadata`, `quality`, `forensics`, `objects`, `ocr`, `text_normalized`, `claims`, `fact_checks`, `provenance`, `xai`, `hashes`, `evidence`, `assessment`, `fusion`, `summary`, `explanation`, `virality`, `regional_spread`, `social_impact`, processing duration, `stages`, `module_statuses`, warnings, limitations and disclaimer. Missing modules remain null/unavailable; the `summary` keeps authenticity, claims, provenance, spread and potential impact distinct.

Pipeline stage states currently serialize lowercase: `pending`, `running`, `completed`, `partial`, `failed`, `skipped`, `unavailable`. Analysis lifecycle values include `queued`, `processing`, `completed`, `partial` (derived from stored report), and `failed`. Stage records contain timestamps, runtime, stable stage ID, analysis correlation ID, safe error category/message, warning/limitation arrays, and a report output reference. The pipeline accounts for 20 stages, including upload-time preprocessing. See `system_architecture.md` and `/docs` for current implementation details.
