# Plan vs. Current Implementation Changes

> **Inspection Date:** 2026-10-20  
> **Repository:** `final year project`  
> **Baseline References:** `docs/IMPLEMENTATION_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/MODELS.md`, `docs/VIRALITY_DATA.md`

---

## 1. Executive Summary of Evolution

TruthLens transitioned from an initial high-level conceptual proposal into an auditable, fail-soft, multi-modal pipeline spanning 19 orchestrated stages. Crucially, engineering realities—such as GPU memory budgets (4 GB VRAM on NVIDIA RTX 3050 Laptop) and the empirical absence of verified ground-truth training datasets for Indian misinformation virality—forced principled design pivots away from "black-box monolithic predictors" toward transparent, modular, and evidence-fused assessments.

---

## 2. Component-by-Component Comparison

| Dimension | Initial Conception / Early Plan | Current Actual Implementation | Rationale for Deviation |
| :--- | :--- | :--- | :--- |
| **Pipeline Architecture** | Monolithic 5–8 step linear verification pipeline | 19-stage asynchronous fail-soft orchestrator (`backend/app/services/orchestrator.py`) | Individual stage failure (e.g. third-party API timeout, missing EXIF) must not abort the overall forensic evaluation. |
| **Authenticity Detection** | Single fine-tuned binary classifier | Multi-model evaluation: Primary SigLIP2 (`prithivMLmods/Deepfake-Real-Class-Siglip2`), comparative baselines (CNN from scratch, ResNet-18, SigLIP-base), with fallback heuristics | Evaluates modern diffusion/transformer artifacts; establishes rigorous experimental baselines; preserves uptime if weights are offline. |
| **Explainable AI (XAI)** | Generic saliency map / black-box explanation | Layer-targeted Grad-CAM on vision transformer / CNN feature maps (`backend/app/services/xai/gradcam.py`) | Provides visual localization of synthesized regions directly to the user dashboard. |
| **Virality Estimation** | Supervised deep learning virality regression model | Transparent multi-factor heuristic scoring engine with `DATA_UNAVAILABLE` status explicitly flagged | Systematic survey (`docs/VIRALITY_DATA.md`) proved zero legitimate public datasets correlate Indian image content with spread velocity. |
| **Regional Spread Assessment** | Predictive geographical diffusion neural network | Demographic and linguistic rule engine mapping content language and OCR entities across 36 Indian States/UTs | Absence of ground-truth state-level forwarding traces; avoids fabricating predictive precision. |
| **Social Impact Risk** | Automated social sentiment classification | Multi-dimensional impact matrix combining authenticity, fact-check ratings, communal/political keywords, and demographic exposure | Misinformation consequences are contextual and policy-sensitive; rule-based auditing provides explainability. |
| **Vision-Language Model (VLM)** | Heavy end-to-end VLM (Qwen2.5-VL / LLaVA) running continuously | Standalone VLM service structure present, but excluded from default 19-stage pipeline | Host hardware constraint: 4 GB VRAM cannot hold a 3B+ parameter VLM alongside SigLIP2 and YOLOv8 without OOM crashes. |
| **Object Detection & Scene Parsing** | Heavy multi-task segmentation | YOLOv8n (`ultralytics`) COCO-pretrained lightweight detector | Inference latency and memory footprint constraints. |
| **OCR & Text Extraction** | Tesseract only | EasyOCR with multilingual Indic language support + PyTesseract fallback | High accuracy across diverse Indian scripts and low-resolution meme overlays. |
| **Data Persistence** | PostgreSQL-only enterprise deployment | Dual mode: PostgreSQL via asyncpg with seamless SQLite fallback via `aiosqlite` | Enables frictionless local development and zero-dependency reproducibility without Docker. |
| **Third-Party Services** | Hard dependencies on Google Vision & Fact Check APIs | Optional, gracefully degrading API connectors with offline mocking and timeout guards | Protects system against quota depletion, network drops, and missing API keys. |

---

## 3. Major Architectural Pivots

### 3.1 From "Truth Detector" to "AI-Assisted Verification & Evidence Fusion"
- **Original Assumption:** The system outputs a binary label: "Real" or "Fake".
- **Current Philosophy:** As formally codified in code comments (`backend/app/services/fusion/engine.py`) and documentation (`docs/IMPLEMENTATION_PLAN.md`), TruthLens explicitly disclaims being an "absolute truth detector." It produces calibrated confidence bands, corroborating forensic evidence, provenance records, and explicit uncertainty indicators.

### 3.2 Honest Handling of Data Scarcity (Virality & Impact)
- **Original Assumption:** Deep learning models would predict virality scores and regional contagion paths.
- **Current Implementation:** Following the data availability investigation (`docs/VIRALITY_DATA.md`), which audited 17 datasets (FakeNewsNet, CREDBANK, MuMiN, BharatFakeNewsKosh, FactDrill, IFND, etc.), it was discovered that no dataset pairs image features with authentic viral trajectory metrics in an Indian socio-linguistic context.
- Rather than training on synthetic nonsense and presenting pseudo-scientific accuracy, TruthLens emits an explicit status flag: `DATA_UNAVAILABLE`, accompanied by transparent, rule-based risk heuristics based on linguistic reach and emotional claim density.

### 3.3 Memory & Resource Constrained Sequential Execution
- **Original Assumption:** Parallel GPU execution of detection, object detection, VLM reasoning, and feature extraction.
- **Current Implementation:** Host environment has 4 GB VRAM (RTX 3050). The pipeline runs sequentially with explicit torch cache clearances (`torch.cuda.empty_cache()`) and lazy model loading.

---

## 4. Unimplemented or Deferred Planned Features

1. **Local Milvus / Vector Search:** Semantic image embedding search was planned for fast duplicate clustering; currently perceptual hashing (pHash/dHash) and Google Reverse Image Search are implemented instead.
2. **Full End-to-End Test Execution Logs:** While unit and integration test suites exist (`backend/tests/`), recorded runtime logs of the complete 19 stages executed on physical hardware are currently absent from repository tracking.
3. **Automated Docker Compose Orchestration:** `docker/` directory exists as a placeholder, but local development relies on direct Python virtual environments.
