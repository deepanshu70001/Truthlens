# TruthLens Research Paper Alignment & Scope Calibration

> **Document Status:** Strategic Academic Alignment Guide
> **Target Venue:** IEEE Transactions on Information Forensics and Security / IEEE Access / CVPR Workshops
> **Paper Draft:** `docs/TruthLens_IEEE_research_paper.md`

---

## 1. Paper Title & Positioning

### Recommended Paper Title
**"TruthLens: Shortcut-Resistant AI-Generated Image Detection via Matched Augmentation and Rigorous Cross-Generator Evaluation Governance"**

### Running Abstract Summary
Detecting AI-generated imagery across evolving generative architectures requires models that identify genuine synthesis traces rather than incidental encoding artifacts. In this work, we present TruthLens, an integrated media verification framework, and report on an extensive empirical investigation into dataset shortcuts and cross-generator generalization. Through cryptographic auditing of 29,640 images across 16 generator families, we expose a pervasive container-format shortcut in benchmark subsets (100% JPEG authentic vs. 100% PNG synthetic; metadata ROC-AUC = 1.0000). We demonstrate that standard fine-tuning exploits this shortcut, exhibiting severe prediction flipping (45.6%) under in-memory transcoding. To resolve this, we engineer a matched in-memory augmentation pipeline that dynamically normalizes format and compression distributions, reducing metadata separability to chance (0.5028). In locked test evaluations across 10,385 held-out samples, our primary fine-tuned candidate (`siglip-base-v2-s2`) achieved a 0.9423 ROC-AUC on in-distribution data and demonstrated compression robustness (11.0% flip rate). Crucially, cross-generator evaluation on modern late-2026 generators (Frontier Benchmark) revealed significant negative transfer (ROC-AUC 0.4234 vs. baseline 0.6501, $\Delta = -0.2267$). Under our pre-registered evaluation governance protocol, the candidate was rejected and the frozen baseline retained in production. Our findings demonstrate the critical role of multi-benchmark holdout gates in preventing premature deployment of shortcut-reliant detectors.

---

## 2. Key Academic Contributions

1. **Empirical Shortcut Discovery:** Formal demonstration that popular benchmark subsets (e.g., Tiny-GenImage) suffer from container-format confounding, enabling models to achieve near-perfect metrics without learning image synthesis features.
2. **Matched In-Memory Augmentation Defense:** A principled, dynamic transcoding intervention applied identically across authentic and synthetic classes during training, neutralizing format confounding without corrupting original disk assets.
3. **Cross-Generator Evaluation Governance:** Systematic benchmarking across 16 generator families spanning 2023 diffusion UNets and 2026 flow-matching diffusion transformers, demonstrating that in-distribution gains can mask catastrophic out-of-domain forgetting.
4. **Reproducible Multi-Modal Verification Architecture:** A modular open-source verification platform integrating visual detection, classical forensics, OCR, claim extraction, and status-aware evidence fusion.
