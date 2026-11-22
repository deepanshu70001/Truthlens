# TruthLens project roadmap

This roadmap preserves the five-stage sequence supplied for the fresh execution. E0 records the repository state; later milestones are performed only under their task scope. Historical research artifacts, model identity, calibration, inference routing, thresholds, and inconclusive gating remain under their existing governance.

## Main development sequence

| Stage | Milestones | Scope |
|---|---|---|
| 1 | **E0 and E1 — Stabilize the foundation** | Audit and register the actual repository state, then correct pipeline status/reporting, contracts, and foundation issues identified by E0. Preserve existing research history and production baseline. |
| 2 | **E2, E3, E4 and E5 — OCR, language, claims, fact-checking and API** | Complete OCR/runtime handling, language identification and translation, claim extraction and fact-check matching, then provide usable public API contracts. |
| 3 | **E6 and E7 — Forensics, YOLO, Grad-CAM, manipulation detection, provenance and C2PA** | Integrate and validate visual analysis and forensic capabilities, manipulation analysis, reverse-image provenance, and content-credential parsing. Keep evidence limitations explicit. |
| 4 | **E8, E9 and E10 — Durable jobs, security, database lifecycle and frontend E2E** | Replace process-local job execution with durable work, address authentication/authorization and privacy controls, establish schema migration/lifecycle management, and add frontend end-to-end coverage. |
| 5 | **E11, R10, R11, R12 and R13 — Integration, hybrid research, uncertainty, final evaluation and paper** | Integrate the product and research paths, conduct governed hybrid research and uncertainty work, perform final evaluation under the research protocol, and complete the paper. |

## Current milestone progress

As of 2026-11-22, E0, E1, and E2 are recorded complete. E3 language identification is integrated as an optional OCR-text stage and covered by mocked tests, but real inference is unavailable because the configured local PAPLUCA checkpoint is not present. No model weights were downloaded. E3 remains partially implemented until a locally provisioned checkpoint can be verified. E4/E5 claim extraction, fact-check retrieval contracts, and the standalone fact-check endpoint are implemented and mock-tested; live provider operation remains unverified. E6 and E7 are partially implemented. E8, E9 and E10 have implementations and focused evidence, with production database/worker and deployment verification limitations recorded below. E11 and R10–R13 remain unstarted.

| Milestone | Status | Current boundary |
|---|---|---|
| E6 — Advanced forensics, YOLO and Grad-CAM | **PARTIALLY IMPLEMENTED** | Classical indicators pass deterministic image tests. YOLO and Grad-CAM integrations are covered with mocked weights and a toy model, but local YOLO dependencies/weights and the production SigLIP runtime were unavailable. A dedicated manipulation detector remains unimplemented. |
| E7 — Provenance and C2PA | **PARTIALLY IMPLEMENTED** | Original-byte hashing, safe metadata summaries, bounded Web Detection and the standalone endpoint are integrated. C2PA uses an optional official SDK adapter but the SDK and a signed permitted fixture are absent locally; live Google credentials were not exercised. |
| E8 — Durable jobs and background processing | **PARTIALLY IMPLEMENTED** | Database-backed jobs, queue capacity, retries, lease recovery, stable IDs and a separate worker are implemented. SQLite tests simulate worker loss and report-commit recovery; no real multi-process PostgreSQL crash/restart test was run. Delivery is at-least-once, not exactly-once. |
| E9 — Security and production hardening | **PARTIALLY IMPLEMENTED** | Pre-provisioned bearer auth, ownership checks, shared DB rate limits, upload/resource protections, redaction, restrictive config and focused security tests are implemented. Production proxy, PostgreSQL rate-limit and penetration testing remain unverified; user lifecycle is operator-managed. |
| E10 — Database lifecycle and frontend testing | **PARTIALLY IMPLEMENTED** | Alembic fresh/legacy SQLite upgrades preserve reports and recover queued work; schema drift, integrity, deletion and dry-run orphan tests pass. Frontend unit/component and mocked Chromium workflow tests, typecheck and build pass. PostgreSQL migration/restore and production UI integration were not exercised. |

Implementation and exact E6/E7 validation evidence: `docs/e6_e7_forensics_provenance_report.md`. E8–E10 evidence: `docs/e8_e10_production_reliability_report.md` and the linked milestone designs/reports.

## Separate synthetic demo-data workstream

Keep synthetic demo data isolated from production inference. First make the supplied archive available to the coding workspace and register its exact manifest, fields, row counts, and checksums. Later, in a frontend/integration milestone, add a separate demo mode with the visible label **`SYNTHETIC DEMO DATA — NOT A REAL-WORLD PREDICTION`**. Do not train predictive models on these records or connect them to production inference. The demo workstream does not change the five-stage sequence above.

## E0 exit boundary

E0 produces the repository audit, module register, and synthetic data card. Review those documents before starting E1. No application code, model, calibration, threshold, API, or frontend behavior is changed by E0.
