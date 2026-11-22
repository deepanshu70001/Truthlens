# TruthLens project roadmap

This roadmap preserves the five-stage sequence supplied for the fresh execution. E0 records the current state; it does not authorize E1 implementation. Historical research artifacts, model identity, calibration, inference routing, thresholds, and inconclusive gating remain under their existing governance.

## Main development sequence

| Stage | Milestones | Scope |
|---|---|---|
| 1 | **E0 and E1 — Stabilize the foundation** | Audit and register the actual repository state, then correct pipeline status/reporting, contracts, and foundation issues identified by E0. Preserve existing research history and production baseline. |
| 2 | **E2, E3, E4 and E5 — OCR, language, claims, fact-checking and API** | Complete OCR/runtime handling, language identification and translation, claim extraction and fact-check matching, then provide usable public API contracts. |
| 3 | **E6 and E7 — Forensics, YOLO, Grad-CAM, manipulation detection, provenance and C2PA** | Integrate and validate visual analysis and forensic capabilities, manipulation analysis, reverse-image provenance, and content-credential parsing. Keep evidence limitations explicit. |
| 4 | **E8, E9 and E10 — Durable jobs, security, database lifecycle and frontend E2E** | Replace process-local job execution with durable work, address authentication/authorization and privacy controls, establish schema migration/lifecycle management, and add frontend end-to-end coverage. |
| 5 | **E11, R10, R11, R12 and R13 — Integration, hybrid research, uncertainty, final evaluation and paper** | Integrate the product and research paths, conduct governed hybrid research and uncertainty work, perform final evaluation under the research protocol, and complete the paper. |

## Separate synthetic demo-data workstream

Keep synthetic demo data isolated from production inference. First make the supplied archive available to the coding workspace and register its exact manifest, fields, row counts, and checksums. Later, in a frontend/integration milestone, add a separate demo mode with the visible label **`SYNTHETIC DEMO DATA — NOT A REAL-WORLD PREDICTION`**. Do not train predictive models on these records or connect them to production inference. The demo workstream does not change the five-stage sequence above.

## E0 exit boundary

E0 produces the repository audit, module register, and synthetic data card. Review those documents before starting E1. No application code, model, calibration, threshold, API, or frontend behavior is changed by E0.
