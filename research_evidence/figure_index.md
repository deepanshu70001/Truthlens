# TruthLens Proposed Research Paper Figure & Table Index

> **Document Status:** Academic Visuals Specification Aligned with Phase 7 Results
> **Target Format:** IEEE Two-Column Format (`docs/TruthLens_IEEE_research_paper.md`)

---

## 1. Master List of Proposed Figures

### Figure 1: TruthLens System Architecture & Multi-Modal Verification Flow
- **Type:** Architectural block diagram and flowchart.
- **Caption:** *Figure 1: End-to-end architecture of TruthLens, depicting the React SPA client layer, FastAPI asynchronous service, 19-stage orchestrator, frozen SigLIP baseline detector, forensic analyzers, OCR and claim extraction, and status-aware evidence fusion.*
- **Placement:** Section III (System Architecture).

### Figure 2: Group-Aware Dataset Splitting & Governance Pipeline
- **Type:** Dataset partition schematic.
- **Caption:** *Figure 2: Cryptographic ingestion and group-aware split protocol across Tiny-GenImage, Frontier Benchmark 2026-10, and WildAIGC-Bench ($N = 29,640$). Disjoint prompt and generator groups prevent evaluation data leakage into locked test ($N = 10,385$) and calibration ($N = 13,000$) pools.*
- **Placement:** Section V (Experimental Setup & Dataset Governance).

### Figure 3: Container Format Shortcut Vulnerability & Matched Augmentation Neutralization
- **Type:** Feature correlation and ROC comparison plot.
- **Caption:** *Figure 3: Pre-training container format confounding. (a) Random Forest on non-pixel container metadata separates raw Tiny-GenImage classes with ROC-AUC = 1.0000. (b) Matched in-memory augmentation (dynamic transcoding and quality scaling) collapses metadata separability to chance level (ROC-AUC = 0.5028).*
- **Placement:** Section V (Dataset Governance & Shortcut Discovery).

### Figure 4: Multi-Benchmark ROC Curves on Locked Test Partitions
- **Type:** Multi-panel ROC curve comparison (Baseline vs. Candidate S2).
- **Caption:** *Figure 4: Locked test ROC curves across three external benchmarks. (a) Tiny-GenImage ($N=7,000$), showing +0.3300 AUC gain for Candidate S2. (b) Frontier Benchmark 2026-10 ($N=2,091$), illustrating catastrophic negative transfer on late-2026 generators (-0.2267 AUC). (c) WildAIGC-Bench ($N=1,294$), demonstrating in-the-wild generalization (+0.1823 AUC).*
- **Placement:** Section VI (Results and Discussion).

### Figure 5: Transcoding Stability & In-Memory Compression Probes
- **Type:** Bar chart comparing prediction flip rates and logit shifts under JPEG-90 re-compression.
- **Caption:** *Figure 5: In-memory JPEG-90 transcoding stability. Candidate E3 (unaugmented fine-tuning) flipped 45.6% of predictions with mean logit shift 6.12. Matched in-memory augmentation (Candidate S2) stabilized predictions to an 11.0% flip rate and 0.793 logit shift.*
- **Placement:** Section VI (Compression Robustness).

### Figure 6: Temperature Scaling Reliability Diagrams
- **Type:** Reliability diagrams (Confidence vs. Accuracy) before and after post-hoc calibration.
- **Caption:** *Figure 6: Expected Calibration Error (ECE) reduction across calibration pools ($N=13,000$), demonstrating how post-hoc temperature scaling softens overconfident out-of-domain predictions.*
- **Placement:** Section VI (Confidence Calibration).

---

## 2. Master List of Proposed Tables

### Table 1: Master Comparative Benchmark Evaluation on Locked Test Partitions ($N = 10,385$)
- **Contents:** Balanced Accuracy, ROC-AUC, Real Accuracy, AI Accuracy, and 95% group-aware bootstrap confidence intervals across Tiny-GenImage, Tiny Matched Aug, Frontier Benchmark 2026-10, and WildAIGC-Bench for Baseline vs. Candidate S2.
- **Placement:** Section VI (Results and Discussion).

### Table 2: Phase 7 Architecture Comparison under 4 GB VRAM Envelope
- **Contents:** S1 (linear probe), S2 (partial FT), R1 (ResNet-50 full FT), R2 (ResNet-50 stage 4 FT), A1 (unaugmented ablation) showing trainable parameters, peak VRAM, training epochs, validation scores, and candidate status.
- **Placement:** Section VI (Ablation and Architecture Search).

### Table 3: Production Governance & Model Retention Decision Matrix
- **Contents:** Evaluation gates, pre-registered pass/fail criteria, empirical delta AUCs, and final retention of the frozen baseline (`truthlens/siglip-cifake-frozen-head`) in production.
- **Placement:** Section VI (Candidate Governance).
