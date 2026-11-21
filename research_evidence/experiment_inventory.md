# TruthLens Experiment Inventory

> **Document Status:** Comprehensive Multi-Phase Experimental Program Registry
> **Source Directory:** `experiments/`, `experiments/reports/`, `docs/`

---

## 1. Master Experiment Table (Phases 0–7)

| Experiment ID | Phase | Objective | Status | Dataset & Split | Architecture / Strategy | Primary Findings & Metrics | Evidence Artifact |
|---|---|---|---|---|---|---|---|
| **EXP-P0** | Phase 0 | Baseline Audit & Dependency Setup | COMPLETED | Repo checkout | Environment verification | Initial CI and testing hygiene established | `docs/` |
| **EXP-P1** | Phase 1 | Model Identity & Calibration Safety | COMPLETED | Mock & baseline checkpoints | Production detector schema | Cryptographic model contract registered; SHA-256 `e39afc55...` | `backend/tests/test_calibration.py` |
| **EXP-P2** | Phase 2 | External Dataset Ingestion & Validation | COMPLETED | Tiny-GenImage, Frontier 2026-10, WildAIGC ($N=29,640$) | Cryptographic audit | 100% SHA-256 match, 0 corrupted files, exact duplicate clustering | `docs/phase2_dataset_ingestion_report.md` |
| **EXP-P2-SPLIT** | Phase 2 | Group-Aware Dataset Splitting | COMPLETED | $N=29,640$ images | Disjoint stratification | 6,995 train, 6,255 val, 13,000 calib, 10,385 locked test; zero prompt leakage | `docs/phase2_split_protocol.md` |
| **EXP-P3** | Phase 3 | Frozen Baseline Benchmark Evaluation | COMPLETED | Locked test ($N=10,385$) | `truthlens/siglip-cifake-frozen-head` | Tiny AUC: 0.6123, Frontier AUC: 0.6501, WildAIGC AUC: 0.4226 | `docs/phase3_evaluation_report.md` |
| **EXP-P3.5** | Phase 3.5 | Independent Research & Split Audit | COMPLETED | Split manifests & logs | Statistical verification | Confirmed split isolation and baseline evaluation reproducibility | `docs/phase3_5_research_audit_report.md` |
| **EXP-P4** | Phase 4 | Training Dataset Preparation | COMPLETED | Approved manifest ($N=6,995$) | Data filtering & balancing | 3,515 Real, 3,480 AI verified for training | `docs/phase4_training_dataset_report.md` |
| **EXP-P5** | Phase 5 | Controlled Fine-Tuning | COMPLETED | Approved train ($N=6,995$) | E1 (linear), E2 (balanced), E3 (layer 11 FT) | Candidate E3 selected on validation metrics (Tiny val acc 0.884) | `docs/phase5_training_report.md` |
| **EXP-P6** | Phase 6 | Held-Out Evaluation & Shortcut Discovery | COMPLETED | Locked test ($N=10,385$) | Candidate E3 | Discovered container format shortcut (100% real JPEG vs 100% AI PNG; 45.6% probe flips); Candidate E3 rejected | `docs/phase6_evaluation_report.md` |
| **EXP-P7-AUDIT** | Phase 7 | Pre-Training Shortcut Metadata Audit | COMPLETED | Train & val metadata | Random Forest on metadata | Raw metadata RF AUC = 1.0000; Matched in-memory augmentation RF AUC = 0.5028 | `experiments/audit_phase7_shortcuts.py` |
| **EXP-P7-TRAIN** | Phase 7 | Matched Augmentation Model Training | COMPLETED | Approved train ($N=6,995$) | S1, S2, R1, R2, A1 | Candidate S2 selected as primary candidate (Val score 0.6722, best overall) | `docs/phase7_training_report.md` |
| **EXP-P7-CALIB** | Phase 7 | Temperature Scaling Calibration | COMPLETED | Calibration pool ($N=13,000$) | Candidate S2 logits | Tiny $T=1.2174$, Frontier $T=10.3621$, WildAIGC $T=5.6099$, Pooled $T=1.3187$ | `experiments/reports/phase7_temperature_calibration.json` |
| **EXP-P7-TEST** | Phase 7 | Locked Test Benchmark Evaluation | COMPLETED | Locked test ($N=10,385$) | Candidate S2 vs Frozen Baseline | S2 Tiny AUC: 0.9423 (+0.3300 gain); S2 Frontier AUC: 0.4234 (-0.2267 drop; negative transfer) | `docs/phase7_evaluation_report.md` |
| **EXP-P7-GOV** | Phase 7 | Candidate Governance Decision | COMPLETED | Evaluation registry | Policy decision | Candidate S2 rejected for production; Frozen baseline retained in production | `experiments/reports/phase7_candidate_decision.json` |

---

## 2. Archival Historical Baselines (CIFAKE 1.2k Subset)

| Baseline ID | Architecture | Parameters | Test Acc | Test ROC-AUC | Historical Note |
|---|---|---|---|---|---|
| **ARCH-CNN** | Custom 4-Layer ConvNet | 356,097 (100% train) | 89.50% | 0.9585 | Native $32 \times 32$ pixel resolution; sensitive to local pixel co-occurrence |
| **ARCH-RN18** | ResNet-18 ImageNet Probe | 513 (0.0046% train) | 81.50% | 0.9255 | Upsampled $224 \times 224$ bicubic interpolation smoothed spatial artifacts |
| **ARCH-SIGLIP** | SigLIP Frozen Probe | 769 (0.00083% train) | 87.00% | 0.9613 | Converged in 73.0s using cached vision-language embeddings |
