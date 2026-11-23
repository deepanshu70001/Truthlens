# TruthLens — Package A Post-Cleanup E2E Regression & Verification Report

**Document Identity:** `docs/POST_CLEANUP_E2E_REGRESSION_REPORT.md`  
**Repository:** `D:\Projects\Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**Base HEAD Commit:** `8e7d5511e8a84737260e4d3ff935338feea1eb8b`  
**Execution Date:** 2026-11-23  
**Overall Verdict:** PASSED (FULL VERIFICATION ACROSS WORKFLOWS A & B)

---

## 1. Executive Summary & Audit Scope

Following the codebase cleanup, refactoring, and synthetic demonstration dataset integration, a comprehensive end-to-end regression audit (Package A) was performed across the entire TruthLens application stack.

### Audit Objectives
1. **Workflow A (Synthetic Report Enrichment Enabled):** Verify the end-to-end user journey in a real browser when `TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED=true`, ensuring illustrative content indicators, chronological spread trajectory charts, neutral regional exposure simulations, and report download functions render flawlessly with full disclosures.
2. **Workflow B (Baseline Production Workflow):** Verify the end-to-end user journey in a real browser when `TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED=false`, guaranteeing strict zero-regression against the baseline production workflow (data status `DATA_UNAVAILABLE`, zero synthetic values injected, zero UI layout breakage).
3. **Negative Workflows & Error Semantics:** Verify all 12 negative scenarios (unsupported format, corrupted image, empty file, size limit, OCR non-text, unauthenticated access, etc.).
4. **Model Invariance:** Verify that baseline SigLIP model weights (`models/baselines/siglip_frozen_head.pt`) and evidence fusion policy (`e11.conservative.v1`) remain bit-exact and unaltered.
5. **Full Suite Regression:** Execute full test suites across backend (pytest) and frontend (vitest, oxlint, tsc, vite build).

---

## 2. System & Environment Baseline

All verification tests were executed against local loopback services on the real host machine:

- **Operating System:** Windows 11 Home (x64)
- **Python Runtime:** Python 3.12.10
- **Node Runtime:** Node.js v22.23.5 / npm 10.9.2
- **GPU Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (4,095 MB VRAM, CUDA auto-detected)
- **Local Services:**
  - Backend API: `http://127.0.0.1:8000/` (FastAPI / Uvicorn)
  - Background Worker: Durable lease worker (`backend.app.workers.analysis_worker`)
  - Frontend UI: `http://127.0.0.1:5173/` (Vite 8.3.5 / React 19)
- **Browser Automation:** Playwright Chromium launching system Google Chrome in headless mode (`viewport: 1440x1000`).
- **Database:** Local SQLite (`truthlens.db` via `sqlite+aiosqlite`) with WAL mode and foreign key constraints enabled.

---

## 3. Pipeline Stage Execution Audit (All 21 Stages)

The TruthLens durable orchestrator coordinates 21 modular analysis stages for uploaded image assets. The table below details the execution status of all stages during live E2E processing:

| # | Pipeline Stage | Stage Key | Execution Type | Live E2E Outcome | Runtime | Notes / Evidence |
| :-: | :--- | :--- | :---: | :---: | :-: | :--- |
| 1 | **Image Preprocessing** | `preprocessing` | Synchronous Upload | COMPLETED | ~12 ms | Normalized JPEG created (large, medium, small, thumb). |
| 2 | **Metadata Extraction** | `metadata` | Pipeline Worker | COMPLETED | ~2 ms | Dimensions, format, file size, EXIF absence verified. |
| 3 | **Image Quality Assessment** | `quality` | Pipeline Worker | COMPLETED | ~4 ms | Blur (Laplacian variance 2252.5), brightness (238.4), contrast (21.7). |
| 4 | **Authenticity Detection** | `authenticity` | Pipeline Worker (GPU) | COMPLETED | ~752 ms | SigLIP linear head; raw Real softmax 0.690, Fake 0.310 (inconclusive). |
| 5 | **Classical Forensics** | `forensics` | Pipeline Worker (CPU) | COMPLETED | ~137 ms | 6 signals computed (ELA, noise, JPEG, edges, texture, resampling). |
| 6 | **Object Detection** | `vision` | Pipeline Worker (CPU/GPU) | COMPLETED | ~185 ms | YOLOv8n object detection; bounding boxes normalized. |
| 7 | **Explainable AI (Grad-CAM)** | `xai` | Pipeline Worker (GPU) | COMPLETED | ~210 ms | Class-specific heatmap generated and cached as PNG artifact. |
| 8 | **Optical Character Recognition**| `ocr` | Pipeline Worker | COMPLETED | ~85 ms | Text extracted or confirmed empty; engine metadata logged. |
| 9 | **Language Identification** | `language_identification`| Pipeline Worker | COMPLETED | ~15 ms | XLM-RoBERTa 20-class sequence classifier executed offline. |
| 10| **Text Normalization** | `text_normalized` | Pipeline Worker | COMPLETED | ~1 ms | Unicode normalization, script detection, digit extraction. |
| 11| **Claim Extraction** | `claims` | Pipeline Worker | COMPLETED | ~2 ms | Candidate claim extraction and eligibility filtering. |
| 12| **External Fact-Checking** | `fact_check` | Pipeline Worker | COMPLETED / NO_RESULTS| ~1 ms | Skipped external lookup (no eligible claims / absent API key). |
| 13| **Image Provenance (Web)** | `provenance` | Pipeline Worker | COMPLETED / PARTIAL | ~1 ms | Controlled fallback when external Vision API key absent. |
| 14| **Cryptographic Hashing** | `hashing` | Pipeline Worker | COMPLETED | ~3 ms | SHA-256 and 64-bit dHash perceptual hashing computed. |
| 15| **Evidence Compilation** | `evidence` | Pipeline Worker | COMPLETED | ~1 ms | Evidence items structured under `truthlens.evidence.v1`. |
| 16| **Conservative Evidence Fusion**| `fusion` | Pipeline Worker | COMPLETED | ~3 ms | Policy `e11.conservative.v1`; abstains to `inconclusive`. |
| 17| **Explanation Generation** | `explanation` | Pipeline Worker | COMPLETED | ~1 ms | Template-based markdown explanation with explicit caveats. |
| 18| **Virality Estimation** | `virality` | Pipeline Worker | COMPLETED | ~5 ms | Heuristic correlates or synthetic demo indicators attached. |
| 19| **Regional Spread Estimation** | `regional_spread` | Pipeline Worker | COMPLETED | ~2 ms | 36 regions; neutral exposure counts (demo) or unavailable. |
| 20| **Social Impact Assessment** | `social_impact` | Pipeline Worker | COMPLETED | ~2 ms | Rule-based sensitive category indicators assessed. |
| 21| **Report Assembly** | `report_assembly`| Pipeline Worker | COMPLETED | ~4 ms | Final JSON report assembled, sanitized, and stored. |

---

## 4. Workflow A: Synthetic Report Integration E2E Results

Workflow A executed a live browser submission of `data/test_fixtures/e2e_live_test.jpg` with `TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED=true`.

### Live Browser Evidence
- **Submission:** Job `31ed3a2d-4272-4afd-b4db-165cbc8bc966` submitted via `POST /api/v1/analyze` (HTTP 201).
- **Processing Time:** Total backend pipeline time: **6.8 seconds** (warm pipeline, full 21 stages executed).
- **DOM Verification:**
  - `hasIllustrativeHeader`: **true** (`Illustrative Content Context (Synthetic Demonstration)`)
  - `hasSyntheticDisclaimer`: **true** (`⚠️ SYNTHETIC DEMONSTRATION DATA`)
  - `hasViralityDemo`: **true** (`Content Virality Indicators (Synthetic Demo)`)
  - `hasTrajectorySection`: **true** (`Content Spread Trajectory (Simulated Demonstration)`)
  - `hasRegionalSimulation`: **true** (`India — Regional Exposure Simulation (Synthetic Demo)`)
  - `hasRegionalRankingText`: **false** (Top-10 ranking text and bar chart completely removed)
  - `hasNoRegionalRankingDisclaimer`: **true** (`no regional risk, susceptibility, prevalence, or threat ranking is calculated or displayed`)
  - `hasDownloadButton`: **true** (`Download Report` button visible and active)
  - `hasLeafletMap`: **true** (Full-width Leaflet map container rendered)
  - `circleMarkerCount`: **36** (All 36 Indian states and union territories rendered with uniform neutral fixed radius=7 and sky-blue `#38bdf8` markers)
- **JSON Report Export:** Successfully clicked `Download Report` and saved [`docs/screenshots/downloaded_report_workflow_a.json`](file:///D:/Projects/Truthlens/docs/screenshots/downloaded_report_workflow_a.json). The JSON was verified to contain `content_spread`, `illustrative_context`, `regional_spread.data_status = 'DEMO'`, `top_risk_regions = []`, and full academic disclaimers.
- **Console & Network:** **0 console errors**, **0 network failures** (26 requests logged).

---

## 5. Workflow B: Baseline Production Workflow E2E Results

Workflow B executed a live browser submission of `data/test_fixtures/e2e_live_test.jpg` with `TRUTHLENS_SYNTHETIC_REPORT_DATA_ENABLED=false`.

### Live Browser Evidence
- **Submission:** Job `543517fb-7642-473f-b61a-6f3251a5b44f` submitted via `POST /api/v1/analyze` (HTTP 201).
- **DOM Verification:**
  - `hasIllustrativeHeader`: **false** (No demo callout container rendered)
  - `hasSyntheticDisclaimer`: **false** (No synthetic disclaimers rendered)
  - `hasViralityDemo`: **false** (Standard heuristic virality view rendered)
  - `hasViralityHeuristic`: **true** (`Content Spread Indicators (Heuristic)`)
  - `hasTrajectorySection`: **false** (Trajectory section completely absent)
  - `hasRegionalUnavailable`: **true** (`Regional scores and rankings are unavailable...` notice rendered)
  - `hasRegionalSimulation`: **false** (Regional map hidden; zero simulated counts)
  - `hasDownloadButton`: **true** (`Download Report` button active)
- **JSON Report Export:** Successfully downloaded [`docs/screenshots/downloaded_report_workflow_b.json`](file:///D:/Projects/Truthlens/docs/screenshots/downloaded_report_workflow_b.json). Verified that `content_spread` is `null`, `illustrative_context` is `null`, and `regional_spread.data_status` is `DATA_UNAVAILABLE`.
- **Console & Network:** **0 console errors**, **0 network failures** (32 requests logged).

---

## 6. Negative Workflows & Error Semantics

The dedicated negative workflows test suite ([`scripts/test_negative_workflows.py`](file:///D:/Projects/Truthlens/scripts/test_negative_workflows.py)) verified all 12 required edge-case and failure scenarios:

| Scenario | Input / Test Condition | Expected HTTP Status & Error Code | Live Result |
| :---: | :--- | :--- | :---: |
| 1 | Unsupported file type (`application/pdf`) | `422 Unprocessable Entity` (File type not allowed) | **PASS** |
| 2 | Corrupted image file (invalid magic bytes) | `422 Unprocessable Entity` (`IMAGE_VALIDATION_ERROR`) | **PASS** |
| 3 | Empty image file (0 bytes) | `422 Unprocessable Entity` (`IMAGE_VALIDATION_ERROR`) | **PASS** |
| 4 | Oversized image upload (> 20 MB) | `413 Payload Too Large` (Size limit exceeded) | **PASS** |
| 5 | OCR returns no text | `200 OK` (Stage completes cleanly with `has_text=False`)| **PASS** |
| 6 | Language detector handles non-text gracefully | `200 OK` (Returns `detected: unknown`, `insufficient_input`)| **PASS** |
| 7 | Fact-check provider credentials absent | `200 OK` (Safe fallback, status `unavailable`, clear warning)| **PASS** |
| 8 | Fact-check provider returns no records | `200 OK` (Stage completes with `no_results`) | **PASS** |
| 9 | Optional provenance service unavailable | `200 OK` (Report completes as `partial`) | **PASS** |
| 10| Job processing failure handling | `404 Not Found` for invalid/corrupted analysis ID | **PASS** |
| 11| Partial analysis report status | `200 OK` with `status: partial` when optional stages abstain | **PASS** |
| 12| Unauthorized artifact retrieval | `404 Not Found` / `401 Unauthorized` on missing/unowned assets | **PASS** |

---

## 7. Cryptographic & Model Invariance

TruthLens maintains strict non-negotiable commitments regarding model weights and evidence fusion policy integrity:

### 1. SigLIP Frozen Head Checkpoint Checksum
- **Checkpoint Path:** `models/baselines/siglip_frozen_head.pt`
- **Expected SHA-256:** `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`
- **Computed SHA-256:** `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`
- **Verification Status:** **BIT-EXACT INVARIANCE CONFIRMED**.

### 2. Evidence Fusion Policy
- **Policy Version:** `e11.conservative.v1`
- **Fusion Logic:** Defined in [`backend/app/services/fusion/engine.py`](file:///D:/Projects/Truthlens/backend/app/services/fusion/engine.py).
- **Verification Status:** Verified via unit tests that evidence fusion evaluates only authentic forensic and model evidence; synthetic demo data has zero influence on authenticity classification or decision thresholds.

---

## 8. Full Regression Test Matrix

All test suites across the repository were executed and passed cleanly:

| Test Suite / Tool | Command | Scope | Result | Details |
| :--- | :--- | :--- | :---: | :--- |
| **Backend Pytest** | `python -m pytest backend/tests` | All 35 test modules | **PASS** | **590 passed**, 12 skipped, 0 failed in 69.36s. |
| **Synthetic Integration**| `python -m pytest backend/tests/test_synthetic_data_integration.py` | Dedicated synthetic suite | **PASS** | **11 passed**, 0 failed in 1.01s. |
| **Frontend Vitest** | `npm --prefix frontend run test -- --run` | Unit & component tests | **PASS** | **6 passed**, 0 failed in 4.61s. |
| **Frontend TypeScript**| `npm --prefix frontend run typecheck` | `tsc -b` | **PASS** | **0 type errors**. |
| **Frontend Oxlint** | `npm --prefix frontend run lint` | 7 source files, 116 rules | **PASS** | **0 errors, 0 warnings** in 76ms. |
| **Frontend Production Build** | `npm --prefix frontend run build` | Vite 8.3.5 / Rolldown bundle | **PASS** | Clean build in 669ms. |
| **Git Whitespace Audit**| `git diff --check` | Working copy diff | **PASS** | **Clean**, 0 whitespace or formatting errors. |

---

## 9. Performance & Processing Latency Metrics

| Metric | Workflow A (Synthetic Enabled) | Workflow B (Baseline) | Target / SLA | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Upload & Validation Latency** | 18 ms | 16 ms | < 200 ms | PASS |
| **Synthetic Dataset Caching** | 551 ms (1st load) / < 1 ms (cached) | N/A (disabled) | < 1,000 ms | PASS |
| **SigLIP GPU Inference** | 752 ms | 760 ms | < 2,000 ms | PASS |
| **Classical Forensics (6 probes)**| 137 ms | 134 ms | < 500 ms | PASS |
| **YOLOv8 Detection** | 185 ms | 180 ms | < 500 ms | PASS |
| **Grad-CAM Computation** | 210 ms | 205 ms | < 1,000 ms | PASS |
| **OCR + Multilingual NLP** | 102 ms | 98 ms | < 5,000 ms | PASS |
| **Total Pipeline Processing** | 32.8s (cold) / 6.8s (warm) | 33.1s (cold) / 6.6s (warm) | < 60s | PASS |
| **Frontend Bundle Size (gzipped)**| 270 kB total | 270 kB total | < 500 kB | PASS |

---

## 10. Final Sign-Off & Verification Verdict

### Verification Status: **PASSED (ALL CRITERIA SATISFIED)**

The synthetic demonstration datasets have been successfully integrated into TruthLens without regression to baseline functionality, without altering model weights or fusion logic, and in full compliance with all regional neutrality and academic disclosure mandates.
