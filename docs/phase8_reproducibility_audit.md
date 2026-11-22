# TruthLens Phase 8: Full Research Reproducibility and Repository Audit

> **Document Status:** Comprehensive Independent Research & Reproducibility Audit  
> **Working Branch:** `fix/truthlens-remediation`  
> **HEAD Commit:** `9591788769895d7441a8cfa02b7809b9eeddf243`  
> **Audit Date:** 2026-11-22  
> **Audit Scope:** Independent verification of all seven completed project phases (Phases 0–7) prior to future model development.

---

## 1. Executive Summary & Verified Phase Status Matrix

This audit conducts a complete, independent scientific and engineering verification of all completed research phases of TruthLens (Phases 0 through 7). The audit evaluates source code, physical datasets ($N = 29,640$ images), split manifests, saved model checkpoints, raw per-sample prediction CSVs, post-hoc calibration artifacts, and automated regression test suites.

### Verified Phase Status Matrix

| Phase | Title / Scope | Primary Objective | Key Evidence Artifacts | Test Coverage | Verified Status |
|---|---|---|---|---|---|
| **Phase 0** | Baseline Audit & Setup | Verify environment, CI, and baseline dependency lock | `.github/workflows/ci.yml`<br>`baseline-python-packages.txt` | Automated CI workflow | **COMPLETE_VERIFIED** |
| **Phase 1** | Model Identity & Calibration Safety | Model identity contracts, calibration schema, inconclusive gating | `backend/app/services/authenticity/detector.py`<br>`models/baselines/siglip_frozen_head.pt` | `test_calibration.py` | **COMPLETE_VERIFIED** |
| **Phase 2** | External Dataset Ingestion & Split Governance | Ingest 3 external benchmarks ($N=29,640$), SHA-256 audit, leak-free splits | `datasets/external/`<br>`experiments/manifests/`<br>`docs/phase2_dataset_ingestion_report.md` | 100% SHA-256 match, 0 leakage | **COMPLETE_VERIFIED** |
| **Phase 3** | Frozen Baseline Benchmark Evaluation | Benchmark frozen SigLIP baseline on locked test partitions ($N=10,385$) | `experiments/results/phase3/`<br>`docs/phase3_evaluation_report.md` | `test_phase3_evaluation.py` | **COMPLETE_VERIFIED** |
| **Phase 3.5** | Independent Research Audit | Audit Phase 3 baseline evaluation, splits, and seeds | `experiments/reports/phase3_5_audit.json`<br>`docs/phase3_5_research_audit_report.md` | Independent statistical audit | **COMPLETE_VERIFIED** |
| **Phase 4** | Training Dataset Preparation | Construct approved, balanced 6,995-sample manifest from Tiny-GenImage | `experiments/manifests/phase4_training_manifest.csv`<br>`docs/phase4_training_dataset_report.md` | `validate_phase4_training_data.py` | **COMPLETE_VERIFIED** |
| **Phase 5** | Controlled Model Training & Fine-Tuning | Train E1, E2, E3 on approved manifest under 4 GB VRAM budget | `experiments/checkpoints/phase5/`<br>`docs/phase5_training_report.md` | Validation split ($N=3,464$) | **COMPLETE_VERIFIED** |
| **Phase 6** | Calibration, Held-Out Evaluation & Shortcut Discovery | Evaluate E3 on locked test splits, audit compression & shortcuts | `experiments/reports/phase6_*.json`<br>`docs/phase6_evaluation_report.md` | JPEG-90 transcoding probe; metadata RF | **COMPLETE_VERIFIED** |
| **Phase 7** | Shortcut-Resistant Detection via Matched Augmentation | Matched in-memory augmentation defense, train 5 architectures, locked test evaluation | `experiments/checkpoints/phase7/`<br>`experiments/results/phase7/`<br>`docs/phase7_evaluation_report.md` | `test_phase7.py` (7/7 passed), full suite (267 passed) | **COMPLETE_VERIFIED** |

*All seven completed phases are verified with physical artifacts, reproducible scripts, and zero failing tests.*

---

## 2. Repository Baseline Specification

- **Repository:** `deepanshu70001/Truthlens`
- **Working Branch:** `fix/truthlens-remediation`
- **HEAD Commit:** `9591788769895d7441a8cfa02b7809b9eeddf243`
- **Git Status:** Clean working tree (zero untracked or uncommitted changes).
- **Continuous Integration:** `.github/workflows/ci.yml` actively monitors linting, syntax, and automated testing across Python environments.
- **Runtime Environment:**
  - OS: Windows 11 Home 64-bit
  - CPU: Intel Core i5-12500H (12 cores, 16 threads)
  - RAM: 16 GB DDR4
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB dedicated VRAM)
  - Python: 3.12.10
  - PyTorch: 2.7.1
  - Transformers: 4.49.0
- **Physical Dataset Availability on Disk ($N = 29,640$ total):**
  - Tiny-GenImage: 25,006 images (`datasets/external/tiny_genimage/`)
  - Frontier Benchmark 2026-10: 2,976 images (`datasets/external/ai_image_detection_benchmark_2026_10/`)
  - WildAIGC-Bench: 1,658 images (`datasets/external/wildaigc_bench/`)
- **Saved Model Checkpoints on Disk:**
  - Production Baseline Head: `models/baselines/siglip_frozen_head.pt` (SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`)
  - Phase 5 Checkpoints: E1, E2, E3 (`experiments/checkpoints/phase5/`)
  - Phase 7 Checkpoints: S1, S2, R1, R2, A1 (`experiments/checkpoints/phase7/`)

---

## 3. Phase 7 Numerical Reproduction & Discrepancy Report

An independent numerical recomputation was conducted directly from the raw prediction CSV files (`experiments/results/phase7/`) across all 9 dataset/model evaluation pairs using scikit-learn standard evaluation metrics (`accuracy_score`, `balanced_accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `average_precision_score`, `matthews_corrcoef`, `confusion_matrix`).

### 3.1 Numerical Recomputation Master Table

| Dataset Partition | Model | Recomputed Acc | Recomputed BalAcc | Recomputed Prec | Recomputed Rec | Recomputed Spec | Recomputed F1 | Recomputed ROC-AUC | Recomputed PR-AUC | Recomputed MCC | Confusion Matrix (TN, FP, FN, TP) | Match with Recorded JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Tiny-GenImage** ($N=7,000$) | Baseline | 0.5521 | 0.5521 | 0.5356 | 0.7854 | 0.3189 | 0.6369 | 0.6123 | 0.6181 | 0.1179 | (1116, 2384, 751, 2749) | **EXACT MATCH** |
| | **S2 Candidate** | 0.8460 | 0.8460 | 0.9420 | 0.7374 | 0.9546 | 0.8272 | 0.9423 | 0.9489 | 0.7089 | (3341, 159, 919, 2581) | **EXACT MATCH** |
| | **S1 Candidate** | 0.7849 | 0.7849 | 0.8339 | 0.7114 | 0.8583 | 0.7678 | 0.8741 | 0.8773 | 0.5760 | (3004, 496, 1010, 2490) | **EXACT MATCH** |
| **Frontier 2026-10** ($N=2,091$) | Baseline | 0.5801 | 0.6228 | 0.4459 | 0.7689 | 0.4767 | 0.5645 | 0.6501 | 0.4533 | 0.2408 | (644, 707, 171, 569) | **EXACT MATCH** |
| | **S2 Candidate** | 0.5361 | 0.4849 | 0.3328 | 0.3095 | 0.6603 | 0.3207 | 0.4234 | 0.3369 | -0.0308 | (892, 459, 511, 229) | **EXACT MATCH** |
| | **S1 Candidate** | 0.5347 | 0.5568 | 0.4003 | 0.6324 | 0.4811 | 0.4903 | 0.5464 | 0.4000 | 0.1094 | (650, 701, 272, 468) | **EXACT MATCH** |
| **WildAIGC-Bench** ($N=1,294$) | Baseline | 0.4490 | 0.4294 | 0.5940 | 0.4935 | 0.3653 | 0.5391 | 0.4226 | 0.6201 | -0.1350 | (164, 285, 428, 417) | **EXACT MATCH** |
| | **S2 Candidate** | 0.5023 | 0.5589 | 0.7332 | 0.3740 | 0.7439 | 0.4953 | 0.6049 | 0.7356 | 0.1190 | (334, 115, 529, 316) | **EXACT MATCH** |
| | **S1 Candidate** | 0.5811 | 0.6214 | 0.7886 | 0.4899 | 0.7528 | 0.6044 | 0.6628 | 0.7740 | 0.2353 | (338, 111, 431, 414) | **EXACT MATCH** |

### 3.2 Paired Differences & Group-Aware Bootstrap Confidence Intervals

Recomputed paired differences ($\Delta = \text{Candidate S2} - \text{Baseline}$) and 95% group-aware bootstrap confidence intervals ($B = 1,000$ resamples, seed = 42):

| Benchmark Partition | Metric | Point Delta ($\Delta$) | Recomputed Bootstrap Mean | Recomputed 95% Bootstrap CI | Recorded 95% CI in Report | Discrepancy |
|---|---|---|---|---|---|---|
| **Tiny-GenImage** | Balanced Accuracy | +0.2939 | +0.2938 | $[+0.2703, +0.3189]$ | $[+0.2703, +0.3189]$ | **NONE (Exact)** |
| | ROC-AUC | +0.3300 | +0.3287 | $[+0.3034, +0.3554]$ | $[+0.3034, +0.3554]$ | **NONE (Exact)** |
| | F1 Score | +0.1903 | +0.1947 | $[+0.1255, +0.2481]$ | $[+0.1255, +0.2481]$ | **NONE (Exact)** |
| **Frontier Benchmark** | Balanced Accuracy | -0.1379 | -0.1380 | $[-0.1827, -0.0941]$ | $[-0.1827, -0.0941]$ | **NONE (Exact)** |
| | ROC-AUC | -0.2267 | -0.2258 | $[-0.2781, -0.1720]$ | $[-0.2781, -0.1720]$ | **NONE (Exact)** |
| | F1 Score | -0.2438 | -0.2439 | $[-0.3067, -0.1826]$ | $[-0.3067, -0.1826]$ | **NONE (Exact)** |
| **WildAIGC-Bench** | Balanced Accuracy | +0.1295 | +0.1302 | $[+0.0854, +0.1729]$ | $[+0.0854, +0.1729]$ | **NONE (Exact)** |
| | ROC-AUC | +0.1823 | +0.1832 | $[+0.1311, +0.2350]$ | $[+0.1311, +0.2350]$ | **NONE (Exact)** |
| | F1 Score | -0.0438 | -0.0427 | $[-0.1047, +0.0154]$ | $[-0.1047, +0.0154]$ | **NONE (Exact)** |

### 3.3 Metric Conventions and Discrepancy Reconciliation

1. **PR-AUC Metric Convention (Average Precision vs. Trapezoidal):**
   - The authoritative PR-AUC metric serialized in `experiments/reports/phase7_test_evaluation.json` and computed by `experiments/evaluate_phase7_test.py` is `sklearn.metrics.average_precision_score` (step-function interpolation):
     - Tiny-GenImage Baseline: `0.6181` | Tiny S2: `0.9489` | Tiny S1: `0.8773`
     - Frontier Baseline: `0.4533` | Frontier S2: `0.3369` | Frontier S1: `0.4000`
     - WildAIGC Baseline: `0.6201` | WildAIGC S2: `0.7356` | WildAIGC S1: `0.7740`
   - When calculated via trapezoidal integration using `precision_recall_curve` and `sklearn.metrics.auc`, slightly different values are obtained due to linear interpolation between operating points (e.g., Tiny Baseline = `0.6180`, Frontier Baseline = `0.4523`, Frontier S2 = `0.3359`). Both methods differ by $\le 0.0015$, and `average_precision_score` is confirmed as the exact serialized standard.
2. **Matthews Correlation Coefficient (MCC) Verification:**
   - Recomputed values via `sklearn.metrics.matthews_corrcoef` match independent manual calculation from the $2 \times 2$ confusion matrix $(TP \cdot TN - FP \cdot FN)/\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}$ with zero difference ($0.00000000$):
     - Tiny Baseline: `+0.1179` | Tiny S2: `+0.7089` | Tiny S1: `+0.5760`
     - Frontier Baseline: `+0.2408` | Frontier S2: `-0.0308` | Frontier S1: `+0.1094`
     - WildAIGC Baseline: `-0.1350` | WildAIGC S2: `+0.1190` | WildAIGC S1: `+0.2353`
3. **Reconciliation of Narrative Text Typos in Initial Phase 8 Summary:**
   - The initial Phase 8 narrative chat summary quoted artifactual figures (e.g. Tiny Baseline PR-AUC 0.5847, MCC 0.1554) resulting from an uncommitted draft script that inverted the positive class convention to Real (`1 - score`). The underlying repository files `docs/phase8_reproducibility_audit.md` and `phase7_test_evaluation.json` have always recorded the exact, verified values. This amendment formally resolves and supersedes the narrative typo.
4. **Bootstrap Methodology Reconciliation:**
   - The authoritative protocol in `experiments/evaluate_phase7_test.py` uses **Group-Aware Clustered Resampling** ($B=1,000$, seed=42) where generator/prompt clusters are resampled as blocks. This yields honest, wider confidence intervals (e.g., Tiny $\Delta \text{ROC-AUC}$ `[+0.3034, +0.3554]`). An unclustered i.i.d. bootstrap produces artificially narrow intervals (`[+0.3166, +0.3434]`). All authoritative reports are standardized on the group-aware method.
5. **Audit of R1 and R2 Locked-Test Access:**
   - Candidates R1 (`truthlens/resnet50-v2-r1`) and R2 (`truthlens/resnet50-v2-r2`) were evaluated **strictly on validation partitions during training** (R1 Val ROC-AUC: Tiny 0.7828, Frontier 0.5241; R2 Val ROC-AUC: Tiny 0.7421, Frontier 0.5047).
   - Under the pre-registered Phase 7 protocol (`docs/phase7_protocol.md`), only the Primary Candidate S2 and Secondary Candidate S1 were selected and authorized for evaluation on the locked test sets.
   - Zero test prediction files were generated for R1 or R2. Previous mentions of R1/R2 "Test" metrics in `research_evidence/model_inventory.md` were drafting misattributions of validation scores and have been corrected. The locked test partitions remain uncontaminated.
6. **Checkpoint Hash Reconciliation:**
   - All 11 physical checkpoints and heads on disk have been re-verified via SHA-256 byte hashing, reconciling earlier documentation placeholders with exact physical digests. Formal details are documented in [docs/phase8_1_audit_reconciliation_amendment.md](phase8_1_audit_reconciliation_amendment.md).

---

## 4. Training and Checkpoint Audit

All five Phase 7 training runs were audited directly on disk:

| Model ID | Run Directory | Checkpoint Path | Computed Checkpoint SHA-256 | Computed Head SHA-256 | Backbone Revision | Best Epoch | Duration (s) | Peak VRAM | Status |
|---|---|---|---|---|---|---|---|---|---|
| **S1** | `s1_siglip_linear_probe/` | `best_checkpoint.pt` | `2554519107c8ad8de07805e55225e7d5a5ac8f9959aecb47815ac86f6fd92111` | `d7e936a3259f0e7f05a18a426300972870a1071b95b721118cfe779cae343c1e` | `7fd15f0689c79d79...` | 1 | 624.05s | 700.5 MB | **VERIFIED** |
| **S2** | `s2_siglip_partial_ft/` | `best_checkpoint.pt` | `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805` | `df8583ec51b51991723d6162ff59730abea96f5b5859186f63a63b91c1c9f33a` | `7fd15f0689c79d79...` | 4 | 1,752.04s | 839.08 MB | **VERIFIED** |
| **R1** | `r1_resnet50_full_ft/` | `best_checkpoint.pt` | `2887136affa6406e4f09a394f0ce6be7ff7ba874ada21a6ff8538d211859a9cb` | `a241a20f68569b59ec604593d7614969b6f0c236b794ca2bb546f997332cee44` | `torchvision-v1` | 7 | 1,578.33s | 1,121.48 MB | **VERIFIED** |
| **R2** | `r2_resnet50_last_stage/` | `best_checkpoint.pt` | `b721cadc68e7b663adde68c1972475c468e3e34d25fab82d1c59da14222e2086` | `99e24551af5e8f3747dd88914e334aaea1435133a0af92b2817638ce8c8df145` | `torchvision-v1` | 7 | 1,208.21s | 606.86 MB | **VERIFIED** |
| **A1** | `a1_siglip_no_aug_ablation/` | `best_checkpoint.pt` | `872b93ceff594fe6469bde42465ee1763656d65a3cf3a6c88efff1b6d4e8d02d` | `43cc6436a37608360f9211409c611cd4c1e19a39c6d54befecc2185f653496bd` | `7fd15f0689c79d79...` | 1 | 569.74s | 839.08 MB | **VERIFIED** |

- **Ablation Model A1:** Confirmed completed across 6 epochs (best epoch 1), with checkpoint and head hashes matching disk records.
- **Hardware Envelope:** All runs executed strictly within the 4 GB VRAM ceiling (peak VRAM: 1,121.5 MB for R1; 839.1 MB for S2).

---

## 5. Data & Leakage Audit

A comprehensive leakage audit was conducted across all partition manifests:
1. **Training Manifest:** `experiments/manifests/phase4_training_manifest.csv` contains exactly 6,995 samples (3,515 real, 3,480 AI).
2. **Excluded Protocol Records:** In Phase 4 data preparation, from an initial pool of 9,994 candidates, exactly 2,998 samples were excluded by protocol to balance generators, and 1 duplicate sample was excluded. None of these 2,999 excluded samples exist in `phase4_training_manifest.csv`.
3. **Partition Independence:**
   - Overlap between Training and Validation: **0 samples**
   - Overlap between Training and Calibration: **0 samples**
   - Overlap between Training and Locked Test: **0 samples**
   - Overlap between Training and Frontier Benchmark: **0 samples**
   - Overlap between Training and WildAIGC-Bench: **0 samples**
4. **Group Isolation:**
   - Tiny-GenImage group IDs spanning multiple evaluation roles: **0**
   - Frontier Benchmark group IDs spanning multiple evaluation roles: **0**
   - WildAIGC-Bench group IDs spanning multiple evaluation roles: **0**

---

## 6. Model Governance Audit

- **Production Baseline Head:** `models/baselines/siglip_frozen_head.pt`
  - Computed SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`
  - Expected SHA-256: `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`
  - Status: **VERIFIED & UNCHANGED**
- **Production Configuration:** `backend/app/config.py` sets `AUTHENTICITY_BACKEND = "finetuned_siglip"` pointing to `models/baselines/siglip_frozen_head.pt`.
- **Model Identity Contract:** `AuthenticityDetector.model_identity()` emits canonical metadata:
  - `model_id`: `truthlens/siglip-cifake-frozen-head`
  - `backbone_id`: `google/siglip-base-patch16-224`
  - `backbone_revision`: `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`
- **Inconclusive Gating:** Operational label defaults to `"inconclusive"`. Raw model argmax is recorded in `model_prediction`, but application label promotion strictly requires valid temperature scaling and input-domain membership.
- **Candidate Deployment:** Zero candidate models (E3, S1, S2, R1, R2, A1) were promoted to production. The production detector remains strictly on the frozen baseline in adherence to pre-registered evaluation governance.

---

## 7. Documentation Corrections Implemented in Phase 8

| File | Issue Identified | Action Taken | Verification |
|---|---|---|---|
| `experiments/README.md` | Stale title "Phase 6 experiments" and claims that manifest is not approved and metrics are not evaluable | Rewritten to document external benchmarks ($N=29,640$), split manifests, Phase 7 training, calibration, and test commands | Verified |
| `docs/README.md` | Missing index reference to Phase 8 audit | Added Section 3 entry for Phase 8 reproducibility audit | Verified |
| `PROJECT_DOCUMENTATION/PROJECT_EVIDENCE_INDEX.md` | Candidate S2 checkpoint hash recorded as placeholder `90bfe0ea...` | Updated to verified disk SHA-256 `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805` | Verified |
| `research_evidence/metrics.json` | Candidate S2 checkpoint hash recorded as placeholder `90bfe0ea...` | Updated to verified disk SHA-256 `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805` | Verified |
| `research_evidence/model_inventory.md` | Candidate S2 checkpoint hash recorded as placeholder `90bfe0ea...` | Updated to verified disk SHA-256 `620e42f74bb7ae24fb1deeaedf24dbf9147162d3612c8f5171b64e53834f9805` | Verified |

---

## 8. Unresolved Issues & Exact Next Actions

### Unresolved Technical Boundaries (Future Research Scope)
1. **Generative Concept Shift on Modern Generators:** The primary candidate S2 achieved 0.9423 ROC-AUC on Tiny-GenImage (2023 diffusion), but collapsed to 0.4234 ROC-AUC on the Frontier Benchmark (late-2026 generators). Bridging this gap requires multi-generator training data incorporating modern flow-matching (FLUX.1) and diffusion transformer (SD 3.5) architectures.
2. **Elevated False Alarm Rates on Web-Sourced Real Media:** Baseline and candidate models produce high false alarm rates on open-domain photographs ($>30\%$ at default 0.5 threshold). Strict operational thresholds ($t \approx 0.93$) are required to maintain FPR $\le 5\%$.
3. **Contextual Indicators:** Virality and regional reach remain heuristic indicators due to the structural absence of longitudinal dissemination graphs (`DATA_UNAVAILABLE`).

### Exact Next Actions for Phase 9
1. **Preregister Phase 9 Protocol:** Define training objectives for multi-generator contrastive representation learning across modern generator architectures.
2. **Curate Modern Generator Training Partition:** Incorporate flow-matching representations without compromising held-out test isolation.
3. **Calibrated False-Alarm Mitigation:** Formulate cost-sensitive loss functions prioritizing extreme real-image specificity ($p(\text{Real}) \ge 0.95$).

---

## 9. Final Validation & Phase 9 Recommendation

### Final Test Suite Execution
- **Command:** `python -m pytest backend/tests/ -q`
- **Authoritative Test Count:** **267 passed, 10 skipped, 0 failed** in 28.60s.
- **Python Compilation:** `python -m compileall backend experiments scripts -q` passed with 0 errors.
- **Documentation Link Validation:** `python scripts/check_doc_links.py` scanned 48 files and checked 125 internal links: **0 broken links**.
- **Git Diff Hygiene:** `git diff --check` passed cleanly with 0 whitespace errors.

### Final Recommendation
**PHASE 9 IS APPROVED TO PROCEED.**  
The repository has achieved complete numerical reproducibility, verified cryptographic integrity, zero data leakage, and airtight governance contracts.
