# TruthLens Project Evidence Index

This index maps technical claims, architectural structures, empirical findings, and evaluation datasets to verifiable files and locations within the TruthLens repository.

| Finding / Domain | Source Path & Reference | Verified Description & Evidence |
|---|---|---|
| **System Architecture** | `docs/system_architecture.md`<br>`backend/app/services/orchestrator.py` | Complete multi-stage asynchronous processing flow, storage layout, and status-aware error capture. |
| **API Endpoints & Routing** | `backend/app/api/v1/router.py`<br>`backend/app/api/v1/endpoints/analyze.py` | Multipart image upload, lifecycle status polling, report retrieval, and diagnostic artifact serving. |
| **Upload Security Bounds** | `backend/app/config.py`<br>`backend/app/core/security.py` | 20 MB size limit, 40M pixel limit, magic byte validation, and opaque UUID filename generation. |
| **Production Baseline Identity** | `backend/app/services/authenticity/detector.py`<br>`models/baselines/siglip_frozen_head.pt` | Model ID `truthlens/siglip-cifake-frozen-head`, backbone `google/siglip-base-patch16-224`, SHA-256 `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`. |
| **Ingested Datasets ($N=29,640$)** | `docs/phase2_dataset_inventory.md`<br>`experiments/reports/dataset_inventory.json` | 100% SHA-256 verified: Tiny-GenImage ($N=25,006$), Frontier Benchmark 2026-10 ($N=2,976$), WildAIGC-Bench ($N=1,658$). |
| **Leak-Free Split Governance** | `docs/phase2_split_protocol.md`<br>`experiments/create_evaluation_splits.py` | Group-aware stratification isolating prompts and generators into locked test ($N=10,385$), calib ($N=13,000$), val ($N=6,255$), and train ($N=6,995$). |
| **Baseline Benchmark Metrics** | `docs/phase3_evaluation_report.md`<br>`experiments/results/evaluation_summary.json` | Baseline locked test performance: Tiny-GenImage AUC 0.6123, Frontier Benchmark AUC 0.6501, WildAIGC AUC 0.4226. |
| **Shortcut Learning Discovery** | `docs/phase6_evaluation_report.md`<br>`docs/phase7_protocol.md` | Proves container format confounding in Tiny-GenImage (100% JPEG real vs 100% PNG AI; metadata AUC = 1.0000; candidate E3 45.6% probe flips). |
| **Matched In-Memory Augmentation** | `experiments/train_phase7.py`<br>`docs/phase7_training_report.md` | Dynamic in-memory transcoding (JPEG, PNG, WebP) and quality scaling, collapsing metadata AUC from 1.0000 to 0.5028. |
| **Candidate S2 Frozen Identity** | `experiments/reports/phase7_selected_candidate_identity.json` | Model ID `truthlens/siglip-base-v2-s2`, layers 10/11 + head FT, checkpoint SHA-256 `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805`. |
| **Locked Test Evaluation (Phase 7)** | `docs/phase7_evaluation_report.md`<br>`experiments/reports/phase7_locked_test_results.json` | Candidate S2 achieves 0.9423 AUC on Tiny-GenImage (+0.3300 gain), but 0.4234 AUC on Frontier (-0.2267 drop; 95% CI: `[-0.2781, -0.1720]`). |
| **Production Decision** | `experiments/reports/phase7_candidate_decision.json` | `REJECT_CANDIDATE_FOR_PRODUCTION`, `RETAIN_FROZEN_BASELINE_IN_PRODUCTION`. Baseline preserved. |
| **Post-Hoc Calibration** | `experiments/calibrate_phase7_candidate.py`<br>`experiments/reports/phase7_temperature_calibration.json` | Temperature scaling fitted on 13,000 calibration samples ($T=1.2174$ Tiny, $T=10.3621$ Frontier, $T=5.6099$ WildAIGC, $T=1.3187$ Pooled). |
| **Forensic Indicators** | `backend/app/services/forensics/analyzer.py` | ELA, noise variance, JPEG ghosting, texture, and resampling analysis. Marked as informational. |
| **OCR & Text Pipeline** | `backend/app/services/ocr/service.py`<br>`backend/app/services/text/normalizer.py` | EasyOCR primary with Tesseract fallback; token extraction and Unicode normalization. |
| **Status-Aware Fusion Engine** | `backend/app/services/fusion/engine.py` | Implements Phase 6 rules: non-completed evidence cannot vote; model evidence drives direction; forensics informational. |
| **Automated Test Suite** | `backend/tests/` | Full suite: **267 passed, 10 skipped, 0 failed** in 28.60s. |
| **Phase 7 Regression Tests** | `backend/tests/test_phase7.py` | 7 automated tests verifying candidate rejection, baseline preservation, calibration, and augmentations. |
