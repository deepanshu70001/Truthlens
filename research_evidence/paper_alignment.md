# TruthLens Research Paper Alignment & Scope Calibration

> **Document Status:** Strategic Academic Alignment Guide  
> **Inspection Date:** 2026-10-20  
> **Primary References:** `research_paper_draft.md`, `research_evidence/`

---

## 1. Paper Title Recommendations

Choosing an accurate, rigorous title is vital to ensure peer reviewers do not penalize the paper for unverified predictive claims.

### Option A (Recommended — High Academic Precision)
> **TruthLens: A Fail-Soft Multimodal Architecture for AI Image Verification, Forensic Localization, and Sociodemographic Exposure Risk Assessment**
- **Rationale:** Accurately reflects the system's actual contributions: the 19-stage asynchronous fail-soft orchestrator, deep learning detection (SigLIP2) with Grad-CAM, classical forensics (ELA/noise), and India-specific sociodemographic mapping, while avoiding the unprovable claim of "predicting viral cascades."

### Option B (Focus on Systems & Forensics)
> **Beyond Single-Oracle Detectors: Multi-Criteria Evidence Fusion and Regional Risk Analysis for Image-Borne Misinformation**
- **Rationale:** Highlights the core methodological argument that single classifiers are insufficient for modern "cheapfakes" and multimodal memes.

### Option C (Retaining Historical Synopsis Title with Calibrated Subtitle)
> **TruthLens: An AI-Assisted Multimodal System for Image Verification, Visual Forensics, and Sociodemographic Risk Mapping**
- **Rationale:** Preserves continuity with original project registration while scientifically calibrating the sub-scope.

---

## 2. Scope & Objectives Calibration

| Dimension | Initial Synopsis / Conceptual Plan | Recommended Calibrated Paper Framing | Rationale & Codebase Justification |
| :--- | :--- | :--- | :--- |
| **System Role** | "Absolute Truth Detector" | **AI-Assisted Verification & Evidence Fusion Aid** | Explicitly stated across codebase (`docs/IMPLEMENTATION_PLAN.md`, `backend/app/services/fusion/engine.py`); probabilistic assessments with explicit confidence bands. |
| **Virality Estimation** | Supervised deep learning virality prediction | **Multi-Factor Virality Risk Index with Declared `DATA_UNAVAILABLE` Status** | Survey (`docs/VIRALITY_DATA.md`) proved zero legitimate public datasets pair Indian image memes with forwarding telemetry. Epistemic honesty strengthens paper credibility. |
| **Regional Spread** | Predictive spatial contagion model | **Sociodemographic Exposure Mapping Across 36 Indian States/UTs** | Deterministically correlates detected Indic scripts and OCR text with Census 2011 linguistic speakers and TRAI broadband density metrics. |
| **Authenticity Modeling** | Generic fine-tuned classifier | **Empirical Comparative Evaluation: Compact CNN vs. Pretrained ResNet vs. Vision-Language SigLIP Probe** | Grounded in measured experimental data (`results/ml/`): SigLIP achieves 0.9613 ROC-AUC with 769 trainable parameters; CNN achieves 89.5% accuracy. |
| **Orchestration** | Linear script execution | **19-Stage Asynchronous Fail-Soft Pipeline** | A key systems contribution: ensures individual stage failures (e.g. API timeouts) do not abort forensic analysis. |

---

## 3. Recommended Core Academic Contributions

When writing Section I (Introduction) and Section VIII (Conclusion), frame the paper's novel contributions across three distinct pillars:

1. **A Resilient, Multi-Modal Systems Architecture:**
   - Design of an asynchronous, 19-stage orchestrator that unifies visual forensics, deep transformer representations, multilingual OCR, reverse web search, and IFCN fact-checking with strict fail-soft fault tolerance.
2. **Empirical Analysis of Parameter Efficiency in Synthetic Image Detection:**
   - A rigorous benchmark comparing native-resolution convolutional models ($32 \times 32$) with foundation vision-language embeddings (SigLIP-224), proving that frozen multimodal representations achieve higher discrimination (0.9613 ROC-AUC) with $463\times$ fewer trained parameters than trained-from-scratch ConvNets.
3. **Transparent Sociodemographic Risk Modeling Under Ground-Truth Data Scarcity:**
   - A principled methodology for addressing platform encryption constraints: rather than fabricating synthetic virality predictions, TruthLens models demographic vulnerability across 36 Indian administrative regions using public census and telecom telemetry, establishing an ethical standard for misinformation risk tools in the Global South.

---

## 4. Alignment Checklist for Draft Revision

- [x] **Abstract:** Replace claims of "automated virality prediction" with "sociodemographic exposure risk mapping and virality risk indexing."
- [x] **Section III (Architecture):** Incorporate the complete 19-stage DAG from `research_evidence/architecture.mmd`.
- [x] **Section IV (Methodology):** Include the mathematical formulation of the Multi-Criteria Evidence Fusion Engine and layer-targeted Grad-CAM.
- [x] **Section V (Experiments):** Insert the exact numerical metrics from `results/ml/comparison.csv` and discuss the High-Frequency Resolution Paradox.
- [x] **Section VI (Limitations):** Fully disclose the 4 GB VRAM boundary, CIFAKE benchmark subset scale, and closed-messaging data barriers.
