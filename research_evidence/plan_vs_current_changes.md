# Plan vs. Current Implementation Changes

> **Document Status:** Complete Architecture & Research Evolution Audit
> **Source Directory:** `PROJECT_DOCUMENTATION/`, `docs/`, `experiments/`

---

## 1. Executive Summary of System & Research Evolution

TruthLens transitioned from an initial high-level conceptual proposal into an auditable, fail-soft, multi-modal pipeline spanning 19 orchestrated stages, underpinned by an empirical research program across Phases 0 through 7. Crucially, empirical discoveries—specifically GPU memory budgets (4 GB VRAM), container format shortcut learning in standard benchmarks, and the cross-generator generalization gap on modern 2026 generators—forced principled design pivots toward transparent, modular, and evidence-fused assessments governed by pre-registered evaluation safety rules.

---

## 2. Component-by-Component Comparison

| Dimension | Initial Conception / Early Plan | Current Actual Implementation | Rationale for Deviation & Empirical Evidence |
|---|---|---|---|
| **Pipeline Architecture** | Monolithic linear verification pipeline | 19-stage asynchronous fail-soft orchestrator (`backend/app/services/orchestrator.py`) | Individual stage failure (e.g. third-party API timeout, missing EXIF) must not abort the overall forensic evaluation. |
| **Authenticity Detection** | Single uncalibrated fine-tuned binary classifier | Production detector locked to frozen SigLIP baseline (`truthlens/siglip-cifake-frozen-head`, SHA-256 `e39afc5536...`) | Phase 5/6 unaugmented fine-tuning learned container shortcuts (E3 rejected). Phase 7 candidate S2 eliminated shortcuts but suffered negative transfer on modern generators (-0.2267 AUC on Frontier). Baseline retained. |
| **Evaluation Benchmarks** | No external benchmark data; small synthetic tests | Ingested 3 external image benchmarks ($N=29,640$ total samples across 16 generator families and 4 compression tiers) | Rigorous evaluation requires diverse, cryptographically verified datasets with group-aware split governance ($N=10,385$ locked test set). |
| **Data Augmentation Strategy** | Standard Pillow resizing | Matched in-memory augmentation (dynamic JPEG/PNG/WebP transcoding, scale jitter, quality scaling $Q \in [30, 95]$) | Container format audit proved 100% real = JPEG and 100% AI = PNG in Tiny-GenImage. Matched augmentation collapsed metadata RF AUC from 1.0000 to 0.5028. |
| **Confidence Calibration** | Raw uncalibrated softmax scores | Post-hoc temperature scaling (`backend/app/services/authenticity/calibration.py`) | Fitted on 13,000 calibration samples ($T=1.2174$ Tiny, $T=10.3621$ Frontier, $T=5.6099$ WildAIGC, $T=1.3187$ Pooled), smoothing extreme logit overconfidence. |
| **Virality & Spread Estimation** | Predictive deep learning virality model | Transparent heuristic scoring with `DATA_UNAVAILABLE` status explicitly declared for regional spread | Absence of ground-truth dissemination cascades prevents pseudo-scientific prediction claims. |
| **Evidence Fusion Engine** | Unconstrained weighted voting | Status-aware fusion: non-completed evidence cannot vote; model evidence drives direction; forensics informational | Phase 6 regressions proved forensic cues are encoding-sensitive and must not vote directionally without model corroboration. |
| **Automated Testing Suite** | Minimal unit tests | 267 passing tests, 10 skipped, 0 failed across API, security, lifecycle, fusion, calibration, and Phase 7 governance | Provides complete regression protection across the full platform (`backend/tests/`). |

---

## 3. Major Architectural & Governance Pivots

### 3.1 From "Autonomous Truth Detector" to "Multi-Modal Evidence Support"
- TruthLens explicitly disclaims being an absolute truth adjudicator. It outputs calibrated class probabilities, forensic indicators, extracted claims, provenance records, and explicit uncertainty bands ($[0.35, 0.65]$).

### 3.2 Discovery and Mitigation of Container Format Shortcuts
- When Candidate E3 achieved high in-distribution validation accuracy, held-out evaluation revealed that it had learned the JPEG/PNG container format confounding of Tiny-GenImage.
- TruthLens engineered matched in-memory augmentation, demonstrating that models can be immunized against container artifacts without mutating disk assets.

### 3.3 Rejection of Primary Candidate S2 in Favor of Baseline
- Although Candidate S2 achieved a 0.9423 ROC-AUC on Tiny-GenImage (+0.3300 gain over baseline), cross-generator evaluation on late-2026 generators (Frontier Benchmark) revealed negative transfer (-0.2267 AUC).
- In adherence to pre-registered evaluation governance, Candidate S2 was rejected for production, preserving the frozen baseline (`truthlens/siglip-cifake-frozen-head`).
