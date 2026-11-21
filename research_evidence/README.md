# TruthLens — Research Evidence Package

> **Status:** Fully Updated & Traced to Codebase and Empirical Phase 7 Milestones
> **Repository:** `https://github.com/deepanshu70001/Truthlens`
> **Working Branch:** `fix/truthlens-remediation`
> **Project Version:** 0.1.0

---

## 1. Scope & Research Overview

This package collects verifiable research evidence, dataset inventories, model contracts, numerical metrics, and experimental results from the TruthLens repository. It traces the full research lifecycle:
- Ingestion and SHA-256 verification of 3 external image benchmarks ($N = 29,640$ total samples across 16 generator families).
- Group-aware, leak-free dataset split governance ($N = 10,385$ locked test partition; $N = 13,000$ calibration pool).
- Frozen SigLIP baseline evaluation across all benchmarks.
- Discovery and forensic proof of container-format shortcut learning (100% JPEG real vs. 100% PNG AI in Tiny-GenImage training data).
- Development of matched in-memory augmentation (dynamic transcoding and quality scaling), collapsing metadata classification ROC-AUC from 1.0000 to chance level (0.5028).
- Controlled training of 5 model architectures (S1, S2, R1, R2, A1) and locked test evaluation.
- Rigorous candidate governance rejecting candidate S2 for production due to modern generator negative transfer, preserving `truthlens/siglip-cifake-frozen-head` in production.
- Automated test suite confirmation: **267 passed, 10 skipped, 0 failed**.

---

## 2. Package Contents

| # | File | Description | Status |
|---|---|---|---|
| 1 | `README.md` | This file — scope, status, resolution of questions | Updated |
| 2 | `experimental_results.md` | Comprehensive empirical benchmark tables (Baseline, E3, S1, S2, R1, R2, A1) | Updated |
| 3 | `dataset_inventory.md` | Ingested external benchmarks ($N=29,640$), generators, splits, licenses | Updated |
| 4 | `metrics.json` | Machine-readable metrics across all test partitions and calibration parameters | Updated |
| 5 | `model_inventory.md` | Production baseline, research candidates, SHA-256 digests, parameter counts | Updated |
| 6 | `methodology.md` | Matched in-memory augmentation, group-aware splitting, temperature scaling | Updated |
| 7 | `failure_analysis.md` | Container format shortcut failure (E3) and modern generator transfer gap (S2) | Updated |
| 8 | `limitations.md` | Cross-generator gap, open-domain false positive rates, contextual indicators | Updated |
| 9 | `test_results.txt` | Automated test suite execution log (267 passed, 10 skipped) | Updated |
| 10 | `paper_alignment.md` | Paper draft title, abstract, empirical contributions, and IEEE alignment | Updated |
| 11 | `implementation_summary.md` | High-density reference for manuscript preparation | Updated |
| 12 | `project_inventory.md` | Repository structure and component responsibilities | Updated |
| 13 | `reproducibility.md` | Execution scripts, hardware constraints, seeds, and configs | Updated |
| 14 | `research_questions.md` | Empirical resolution of RQ1–RQ5 and future questions | Updated |
| 15 | `experiment_inventory.md` | Full catalog of experimental phases 0 through 7 | Updated |
| 16 | `api_inventory.md` | REST API endpoints, schemas, lifecycle status codes | Verified |
| 17 | `architecture_explanation.md` | Async orchestrator, stage execution, and evidence fusion | Verified |
| 18 | `architecture.mmd` | Mermaid diagram of system architecture | Verified |
| 19 | `implementation_status.csv` | Feature-by-feature verification status | Updated |
| 20 | `figure_index.md` | Index of figures and tables for research paper | Updated |
| 21 | `references.md` | Bibliography and literature lineage | Verified |
| 22 | `plan_vs_current_changes.md` | Audit of design plan evolution across remediation phases | Updated |

---

## 3. Resolution of Prior Unresolved Questions

1. **Has the analysis pipeline been executed end-to-end?**
   *Yes.* The 19-stage pipeline executes asynchronously via `backend/app/services/orchestrator.py` and is verified by automated integration tests (`backend/tests/test_stage3.py`, `test_stage5.py`).
2. **Are real benchmark images available and verified?**
   *Yes.* Three external benchmarks ($N = 29,640$) are ingested under `datasets/external/`, audited with 100% SHA-256 match and zero missing files.
3. **What is the production model status?**
   *Production detector is locked to the frozen SigLIP baseline (`truthlens/siglip-cifake-frozen-head`, SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`).* Candidate S2 (`truthlens/siglip-base-v2-s2`) was rejected for production due to negative transfer on late-2026 generators.
4. **Has confidence calibration been implemented?**
   *Yes.* Post-hoc temperature scaling is implemented in `backend/app/services/authenticity/calibration.py` and fitted across 13,000 held-out calibration samples.
5. **What is the research paper status?**
   *Drafted under `docs/TruthLens_IEEE_research_paper.md`, documenting the empirical shortcut discovery and evaluation governance framework.*
