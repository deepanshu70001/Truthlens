# TruthLens Research Questions & Empirical Answers

> **Document Status:** Complete Empirical Resolution of Research Questions
> **Source Artifacts:** `docs/phase7_evaluation_report.md`, `experiments/reports/phase7_locked_test_results.json`

---

## 1. Formal Academic Research Questions & Empirical Findings

### **RQ1: Cross-Generator & Cross-Era Generalization**
> *Can an authenticity detector fine-tuned on early diffusion generator artifacts (2023 era) generalize to contemporary flow-matching, diffusion transformer, and autoregressive architectures (late-2026 era)?*
- **Empirical Resolution:** **Negative.** Fine-tuning Candidate S2 on Tiny-GenImage (2023 diffusion) improved in-distribution test AUC from 0.6123 to 0.9423 (+0.3300), but caused catastrophic negative transfer on the held-out Frontier Benchmark 2026-10 (AUC dropped from 0.6501 to 0.4234, $\Delta = -0.2267$, 95% CI: $[-0.2781, -0.1720]$). Pretrained vision-language features without narrow fine-tuning exhibited superior out-of-domain robustness.

### **RQ2: Container Format Shortcut Vulnerability & Mitigation**
> *How susceptible are image authenticity detectors to non-semantic container format confounding (e.g. JPEG vs. PNG), and can dynamic in-memory augmentation eliminate this vulnerability?*
- **Empirical Resolution:** **Highly Susceptible, Successfully Mitigated.** In unaugmented fine-tuning (Candidate E3), models exploited container differences (100% real = JPEG, 100% AI = PNG; metadata RF AUC = 1.0000), causing 45.6% prediction flips under JPEG-90 probes. Matched in-memory augmentation (dynamic transcoding and quality scaling) collapsed metadata AUC to 0.5028 and reduced probe flips to 11.0% (mean logit shift 0.793).

### **RQ3: Multi-Criteria Evidence Fusion vs. Single Oracle**
> *Does status-aware multi-modal evidence fusion prevent catastrophic misclassifications when individual signals fail or produce contradictory cues?*
- **Empirical Resolution:** **Validated Software Policy.** Phase 6 fusion logic guarantees that unavailable, skipped, or failed stages cannot vote. Classical forensics (proven to be encoding-sensitive) are restricted to informational status, ensuring model evidence alone drives directional authenticity while presenting context to human analysts.

### **RQ4: Post-Hoc Calibration Under Domain Shift**
> *Does temperature scaling fitted on held-out calibration pools effectively mitigate detector overconfidence on out-of-domain distributions?*
- **Empirical Resolution:** **Validated.** Temperature scaling fitted across 13,000 samples ($T = 1.2174$ Tiny, $T = 10.3621$ Frontier, $T = 5.6099$ WildAIGC, $T = 1.3187$ Pooled) successfully softened extreme overconfident logits, reducing calibration error (ECE).

### **RQ5: Social & Regional Predictability from Static Visual Media**
> *Can static visual and textual content indicators reliably predict real-world viral dissemination or regional reach in the absence of forward cascades?*
- **Empirical Resolution:** **Negative / Bound.** Static cues lack correlation with empirical dissemination graphs. TruthLens designates regional spread as `DATA_UNAVAILABLE` to prevent pseudo-scientific reach claims.

---

## 2. Author Decision Registry: Resolution Status

| # | Topic | Status & Resolution in Checkout |
|---|---|---|
| 1 | **Target Publication Venue** | Prepared as IEEE two-column paper draft (`docs/TruthLens_IEEE_research_paper.md`). |
| 2 | **Full 19-Stage Pipeline Run** | Validated end-to-end via automated asynchronous integration tests (`backend/tests/test_stage3.py`, `test_stage5.py`). |
| 3 | **External API Integration Policy** | Decoupled with config-gating and mock/no-credential safe degradation. |
| 4 | **Out-of-Distribution Generative Evaluation** | Comprehensive evaluation on Frontier Benchmark 2026-10 ($N=2,976$ samples across 8 modern generators and 4 compression tiers). |
| 5 | **Production Deployment Decision** | Strictly retains frozen baseline (`truthlens/siglip-cifake-frozen-head`) due to candidate S2 modern generator transfer gap. |
