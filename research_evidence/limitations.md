# TruthLens Technical, Scientific, and Ethical Limitations

> **Document Status:** Comprehensive Limitation & Risk Disclosure  
> **Inspection Date:** 2026-10-20  
> **Target Audience:** Academic Reviewers, Ethics Committees & System Integrators

---

## 1. Technical & Engineering Limitations

### 1.1 Local Hardware & Memory Headroom Constraints
- **Constraint:** Primary workstation deployment is restricted to an NVIDIA GeForce RTX 3050 Laptop GPU with 4,096 MB VRAM.
- **Impact:** 
  - Prevents continuous concurrent hosting of state-of-the-art multimodal Vision-Language Models (e.g. Qwen2.5-VL-3B, LLaVA-NeXT) alongside the primary SigLIP2 and YOLOv8 backbones.
  - Requires sequential model loading, inference execution, and explicit VRAM deallocation, introducing latency overheads of 2–5 seconds between stages.

### 1.2 Vulnerability to Lossy Platform Re-compression
- **Constraint:** Social networks (WhatsApp, Telegram, Facebook, X) apply aggressive lossy downsampling and re-quantization upon image transmission.
- **Impact:** High-frequency pixel co-occurrence traces, camera sensor noise profiles (PRNU), and JPEG 8×8 grid boundary discontinuities are severely degraded, reducing the diagnostic reliability of Error Level Analysis (ELA) and classical forensics.

### 1.3 External API Dependencies & Latency Asymmetry
- **Constraint:** Provenance search and authoritative fact-check retrieval depend on third-party cloud services (Google Cloud Vision Web Detection and Google Fact Check Tools API).
- **Impact:** Network partitions, quota rate-limiting, and regional censorship can degrade the pipeline. The fail-soft orchestrator accommodates this by skipping or mocking these stages, but skipped stages reduce evidence corroboration.

### 1.4 Single-Image Scope vs. Multimodal Temporal Media
- **Constraint:** The current implementation processes static raster images (JPEG, PNG, WebP).
- **Impact:** It cannot process synthetic video deepfakes (temporal facial jitter, audio-visual lip-sync inconsistencies) or cloned voice audio, which represent an increasing fraction of modern electoral disinformation.

---

## 2. Scientific & Empirical Limitations

### 2.1 Baseline Dataset Scale and Diversity Constraints
- **Constraint:** Empirical baseline experiments (`results/ml/`) were conducted on a 1,200-image subset of the CIFAKE benchmark (CIFAR-10 real vs. Stable Diffusion v1.4 synthetic).
- **Impact:** While providing a clean, controlled environment for comparing parameter efficiency across ConvNets and transformers, this benchmark does not reflect the diversity of modern diffusion models (e.g., FLUX.1, Midjourney v6, SDXL, DALL-E 3) or cross-generator domain shifts.

### 2.2 Unavailability of Real-World Virality Ground Truth
- **Constraint:** Over 80% of viral visual disinformation in the Global South propagates through peer-to-peer, end-to-end encrypted messaging channels (WhatsApp, Signal, Telegram) where forwarding counts, graph topology, and velocity are strictly unobservable.
- **Scientific Impact:** No legitimate academic dataset exists linking image features to ground-truth forward cascades in India. 
- **Methodological Stance:** To prevent the publication of pseudo-scientific accuracy metrics, the virality and regional spread modules are presented strictly as **demographic exposure heuristics** with an explicit `DATA_UNAVAILABLE` status declaration.

### 2.3 Heuristic Evidence Fusion vs. Learned Graphical Models
- **Constraint:** The Evidence Fusion Engine (`backend/app/services/fusion/engine.py`) employs a multi-criteria weighted scoring algorithm with calibrated static weights.
- **Impact:** While transparent and explainable, static weights cannot capture complex non-linear interactions between modalities that could be modeled by Bayesian belief networks or Dempster-Shafer theory.

---

## 3. Ethical, Societal & Governance Limitations

### 3.1 Automation Bias & Over-Reliance
- **Risk:** End-users (fact-checkers, journalists, civil authorities) may succumb to automation bias, treating probabilistic model outputs as definitive verdicts.
- **Mitigation:** The system interface and generated reports explicitly avoid binary declarations ("This is Fake"), replacing them with evidential confidence intervals, supporting forensic overlays, and prominent disclaimer notices: *"TruthLens is an AI-assisted verification aid, not an absolute truth arbiter."*

### 3.2 False Positive Harm & Chilling Effects
- **Risk:** Low-quality or heavily compressed authentic photographs depicting human rights abuses, citizen protests, or environmental disasters risk being misclassified as synthetic, potentially discrediting legitimate evidentiary documentation.
- **Mitigation:** High decision thresholds for synthetic classification ($p_{\text{fake}} \ge 0.65$) and mandatory corroboration across forensic and provenance checks before issuing an alert.

### 3.3 Adversarial Evasion & Dual-Use Risks
- **Risk:** Malicious actors can utilize open-source authenticity classifiers as discriminators in adversarial training loops (e.g., AdvGAN or diffusion latent refinement) to generate images specifically engineered to evade detection.
- **Mitigation:** Reliance on diverse, orthogonal evidentiary signals—combining cryptographic hashing, reverse web provenance, and factual knowledge graphs—which cannot be bypassed solely by optimizing pixel perturbations.
