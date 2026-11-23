# TruthLens — Authoritative Development Documentation Index

**Document Identity:** `docs/DEVELOPMENT_DOCUMENTATION_INDEX.md`  
**Repository:** `D:\Projects\Truthlens`  
**Branch:** `fix/truthlens-remediation`  
**Last Updated:** 2026-11-23  
**Status:** ACTIVE AUTHORITATIVE DIRECTORY

---

## 1. Overview & Purpose

This index serves as the central directory mapping every engineering, security, infrastructure, and research workstream in TruthLens to its single authoritative source of truth. As TruthLens evolved through multiple engineering stages (E0–E11) and research phases (Phase 1–Phase 10, Experiment 2), earlier reports and draft protocols were produced. 

This document clarifies:
1. **The current authoritative document** for each active domain.
2. **The underlying source files** and primary test suites.
3. **Historical / superseded records** that remain preserved for audit reproducibility.

---

## 2. Engineering & Operational Workstreams

| Domain / Workstream | Authoritative Document | Primary Code Paths | Key Test Suites | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **System Architecture & Data Flow** | [docs/system_architecture.md](file:///D:/Projects/Truthlens/docs/system_architecture.md) | `backend/app/main.py`, `backend/app/services/orchestrator.py` | `backend/tests/test_services.py` | LOCAL RUNTIME VERIFIED |
| **Complete API Reference** | [docs/api_reference.md](file:///D:/Projects/Truthlens/docs/api_reference.md) | `backend/app/api/v1/router.py`, `backend/app/schemas/` | `backend/tests/test_core_api_integration.py` | COMPLETED & VERIFIED |
| **Core Standalone APIs** (Models, OCR, Claims, Fact-Check) | [docs/CORE_API_COMPLETION_REPORT.md](file:///D:/Projects/Truthlens/docs/CORE_API_COMPLETION_REPORT.md) | `backend/app/api/v1/endpoints/{models,ocr,claims,fact_check}.py` | `test_models_endpoint.py`, `test_ocr_endpoint.py`, `test_claims_endpoint.py`, `test_factcheck_endpoint.py` | COMPLETED & VERIFIED |
| **Language Identification API** (`POST /api/v1/nlp/language`) | [docs/LANGUAGE_API_COMPLETION_REPORT.md](file:///D:/Projects/Truthlens/docs/LANGUAGE_API_COMPLETION_REPORT.md) | `backend/app/api/v1/endpoints/language.py`, `backend/app/services/nlp/language_detector.py` | `test_language_endpoint.py`, `test_language_detector.py` | COMPLETED & VERIFIED |
| **Database Lifecycle & Migrations** | [docs/DATABASE_WORKER_RELIABILITY_REPORT.md](file:///D:/Projects/Truthlens/docs/DATABASE_WORKER_RELIABILITY_REPORT.md) | `backend/migrations/`, `backend/app/db/session.py`, `backend/app/models/` | `test_e10_database.py`, `test_postgres_migrations.py`, `test_database_backup_restore.py` | COMPLETED & VERIFIED (SQLite Runtime / Postgres DDL Verified) |
| **Durable Worker & Crash Recovery** | [docs/DATABASE_WORKER_RELIABILITY_REPORT.md](file:///D:/Projects/Truthlens/docs/DATABASE_WORKER_RELIABILITY_REPORT.md) | `backend/app/services/job_queue.py`, `backend/app/workers/analysis_worker.py` | `test_worker_recovery.py`, `test_worker_concurrency.py`, `test_e8_durable_jobs.py` | COMPLETED & VERIFIED (SQLite Runtime / Lease Reclaim Verified) |
| **Security Hardening & Production Audit** | [docs/SECURITY_PRODUCTION_READINESS_REPORT.md](file:///D:/Projects/Truthlens/docs/SECURITY_PRODUCTION_READINESS_REPORT.md) | `backend/app/core/{security,auth,ownership,rate_limit,middleware,logging}.py` | `test_e9_security.py`, `test_security_audit.py` | COMPLETED & VERIFIED (22/22 Audit Scenarios Passed) |
| **Live Browser E2E & Full App Integration** | [docs/LIVE_BROWSER_E2E_REPORT.md](file:///D:/Projects/Truthlens/docs/LIVE_BROWSER_E2E_REPORT.md) | `frontend/src/App.tsx`, `scripts/live_e2e_workflow.js`, `scripts/test_negative_workflows.py` | `vitest run`, `test_negative_workflows.py` | COMPLETED & VERIFIED (PASS / Genuine Chrome Session) |
| **Codebase Cleanup & Optimization** | [docs/CODEBASE_CLEANUP_REPORT.md](file:///D:/Projects/Truthlens/docs/CODEBASE_CLEANUP_REPORT.md) | Whole repository cleanup, deduplication, Deadwood removal | Full Pytest, Vitest, Oxlint, Rollup bundle audit | COMPLETED & VERIFIED (Package 1–4 Audits Clean) |
| **Synthetic Demonstration Data Integration** | [docs/SYNTHETIC_REPORT_INTEGRATION.md](file:///D:/Projects/Truthlens/docs/SYNTHETIC_REPORT_INTEGRATION.md) | `backend/app/services/synthetic/`, `datasets/synthetic_demo/` | `test_synthetic_data_integration.py`, Workflow A Playwright | COMPLETED & VERIFIED (Demonstration Only / Feature-Gated) |
| **Package A Post-Cleanup E2E Regression** | [docs/POST_CLEANUP_E2E_REGRESSION_REPORT.md](file:///D:/Projects/Truthlens/docs/POST_CLEANUP_E2E_REGRESSION_REPORT.md) | Real browser E2E workflows A & B, negative workflows | Playwright Chrome, Negative workflows suite | COMPLETED & VERIFIED (Full Pass Across Workflows A & B) |
| **Deployment & Operations Guide** | [docs/deployment_guide.md](file:///D:/Projects/Truthlens/docs/deployment_guide.md) | `.env.example`, `backend/app/main.py`, `frontend/vite.config.ts` | Manual / CI Deployment | CURRENT OPERATIONAL GUIDE |
| **Database Migration Runbook** | [docs/migration_runbook.md](file:///D:/Projects/Truthlens/docs/migration_runbook.md) | `backend/migrations/versions/` | `test_postgres_migrations.py` | CURRENT OPERATIONAL RUNBOOK |

---

## 3. Multimodal Inference & Visual Forensics Workstreams

| Capability | Authoritative Document | Underlying Implementation | Test & Runtime Evidence |
| :--- | :--- | :--- | :--- |
| **Production Authenticity Detector** | [docs/MODULE_STATUS_REGISTER.md](file:///D:/Projects/Truthlens/docs/MODULE_STATUS_REGISTER.md) (Module 3) | `models/baselines/siglip_frozen_head.pt` (SHA-256 `e39afc5536a0...`), `backend/app/services/authenticity/detector.py` | 0.0 logit diff, invariant weights preserved. Evaluated on GPU. |
| **YOLOv8 Object Detection** | [docs/yolo_verification_report.md](file:///D:/Projects/Truthlens/docs/yolo_verification_report.md) | `models/baselines/yolov8n.pt`, `backend/app/services/vision/detector.py` | Real inference verified via Ultralytics 8.4.216 runtime. |
| **SigLIP Grad-CAM Heatmaps** | [docs/siglip_gradcam_verification_report.md](file:///D:/Projects/Truthlens/docs/siglip_gradcam_verification_report.md) | `backend/app/services/xai/gradcam.py` | Verified on GPU; bit-exact parameter preservation confirmed. |
| **Classical Image Forensics** | [docs/e6_e7_forensics_provenance_report.md](file:///D:/Projects/Truthlens/docs/e6_e7_forensics_provenance_report.md) | `backend/app/services/forensics/analyzer.py` (ELA, noise, JPEG, edges, texture, resampling) | Deterministic image tests pass. Probes remain strictly informational (Experiment 2 rejected directional fusion). |
| **C2PA / Content Credentials** | [docs/c2pa_verification_report.md](file:///D:/Projects/Truthlens/docs/c2pa_verification_report.md) | `backend/app/services/provenance/c2pa_verifier.py`, `c2pa-python 0.39.0` | Real cryptographic verification against official signed fixture `C.jpg` and conformance trust anchors. |
| **OCR (EasyOCR & Tesseract)** | [docs/e2_ocr_completion_report.md](file:///D:/Projects/Truthlens/docs/e2_ocr_completion_report.md) | `backend/app/services/ocr/service.py` | Tesseract 5.4.0 verified with `eng,hin,osd` packs. EasyOCR cache-only runtime verified. |
| **Multimodal Evidence Fusion** | [docs/e11_evidence_fusion_contract.md](file:///D:/Projects/Truthlens/docs/e11_evidence_fusion_contract.md) | `backend/app/services/fusion/engine.py` (Policy `e11.conservative.v1`) | Conservative fusion verified. Directional votes require calibration; absent evidence abstains to `inconclusive`. |

---

## 4. Research & Scientific Governance

| Workstream | Authoritative Document | Governance Status | Empirical Outcome / Decision |
| :--- | :--- | :--- | :--- |
| **Experiment 2: Univariate Forensic Feature Screening** | [research/experiment2_forensic_screening/experiment2_closeout_audit.md](file:///D:/Projects/Truthlens/research/experiment2_forensic_screening/experiment2_closeout_audit.md) | **GATE REJECTION (CLOSED)** | Evaluated 6 forensic indicators across 6,995 training samples under Protocol v1.2.1. 100% format confounding (JPEG vs PNG) identified. All FDR $q \ge 0.292$. Directional forensic fusion formally **REJECTED**. Experiment 3 **NOT RECOMMENDED**. |
| **R10: Preregistered Hybrid Architecture** | [docs/r10_experiment_report.md](file:///D:/Projects/Truthlens/docs/r10_experiment_report.md) | **DEVELOPMENT EVIDENCE ONLY** | Preregistered fit and validation executed; Frontier ROC-AUC delta +0.0631, balanced accuracy trade-off -0.0801. Development evidence only; no production deployment. |
| **Phase 7: Matched Augmentation** | [docs/phase7_evaluation_report.md](file:///D:/Projects/Truthlens/docs/phase7_evaluation_report.md) | **HISTORICAL PHASE EVIDENCE** | In-memory format augmentation eliminated container format shortcuts on in-distribution data. |
| **Master Status Register** | [docs/MODULE_STATUS_REGISTER.md](file:///D:/Projects/Truthlens/docs/MODULE_STATUS_REGISTER.md) | **AUTHORITATIVE LIVING REGISTER** | Current status across all 32 modules and milestones (E0–E11). |
| **Project Roadmap** | [docs/PROJECT_ROADMAP.md](file:///D:/Projects/Truthlens/docs/PROJECT_ROADMAP.md) | **AUTHORITATIVE LIVING ROADMAP** | High-level 5-stage development sequence and milestone boundaries. |
| **Implementation Queue** | [docs/IMPLEMENTATION_QUEUE.md](file:///D:/Projects/Truthlens/docs/IMPLEMENTATION_QUEUE.md) | **AUTHORITATIVE QUEUE** | Detailed implementation specifications for completed priorities 1–8. |

---

## 5. Preserved Historical Reports Directory

The following legacy reports located under `docs/implementation history/` and `docs/` are preserved for historical audit trails:

- `docs/implementation history/phase1_implementation_report.md` through `phase6_implementation_report.md`
- `docs/phase7_evaluation_report.md`, `phase7_training_report.md`
- `docs/e1_failure_handling_report.md`
- `docs/e4_e5_implementation_report.md`
- `docs/e8_e10_production_reliability_report.md`
- `docs/e11_r10_completion_report.md`

> [!IMPORTANT]
> Historical reports document the state of the repository at the time of their execution. They must not be altered to match subsequent implementations. Always refer to Section 2 and Section 3 of this index for current production operational truth.
