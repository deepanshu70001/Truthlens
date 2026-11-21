# TruthLens Failure Modes & Edge Case Analysis

> **Document Status:** Comprehensive Technical Audit  
> **Inspection Date:** 2026-10-20  
> **Primary References:** `backend/app/services/`, `docs/MODELS.md`, `docs/EVALUATION.md`

---

## 1. Taxonomic Classification of Failure Modes

Misinformation verification systems operate in an adversarial and evolving information environment. This document catalogues the principal failure modes, technical blind spots, and vulnerability vectors identified in the current TruthLens codebase.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VULNERABILITY TAXONOMY                       │
├──────────────────────┬───────────────────┬──────────────────────┤
│ Visual & Model Level │ Forensics & Image │ Semantic & Contextual│
│ • Concept drift      │ • JPEG re-encode  │ • Cheapfakes / OOC   │
│ • ViT smoothing      │ • EXIF scrubbing  │ • Script mixing      │
│ • Grad-CAM dilution  │ • Double compress │ • Zero-hit fact-check│
└──────────────────────┴───────────────────┴──────────────────────┘
```

---

## 2. Visual Deep Learning Failure Modes

### 2.1 Generative Concept Drift & Novel Architectures
- **Vulnerability:** Detection backbones (SigLIP2, ResNet, baseline CNN) are trained primarily on historical generative architectures (Stable Diffusion v1.4, StyleGAN, Midjourney v4/v5).
- **Failure Manifestation:** Next-generation latent diffusion models utilizing Flow Matching transformers (e.g., FLUX.1 Schnell/Dev, Stable Diffusion 3, AuraFlow) produce dramatically different high-frequency and structural latent noise signatures. The model often outputs high-confidence `AUTHENTIC` classifications on hyper-realistic modern generations.
- **Evidence in Codebase:** The baseline experiments in `results/ml/` reflect training strictly on CIFAKE (SD v1.4 vs CIFAR-10), which cannot guarantee out-of-distribution generalization to 2026 generative models.

### 2.2 Saliency Map Dilution in Grad-CAM
- **Vulnerability:** Grad-CAM applied to the penultimate layer of a Vision Transformer operates on patch token representations (e.g. $14 \times 14$ grid for $224 \times 224$ images).
- **Failure Manifestation:** Upsampling coarse patch activations via bilinear interpolation often produces diffused, circular heat blobs that encompass valid background objects alongside manipulated areas, providing suboptimal spatial precision for forensic legal standards.

---

## 3. Classical Forensics & Signal Processing Vulnerabilities

### 3.1 Aggressive Social Platform Transcoding (WhatsApp / Telegram Effect)
- **Vulnerability:** Classical forensic probes (ELA, Noise Residuals, JPEG Grid Discontinuities in `backend/app/services/forensics/analyzer.py`) rely on micro-level pixel statistics.
- **Failure Manifestation:** When an image is forwarded across WhatsApp or Twitter, platforms apply heavy lossy re-encoding (e.g. re-saving at JPEG quality 70–80) and downscaling. This uniform re-compression erases high-frequency forensic traces and creates uniform error surfaces, rendering ELA blind to spliced boundaries.

### 3.2 Metadata Stripping
- **Vulnerability:** `extract_metadata` relies on EXIF headers for camera hardware, capture timestamps, and software tags.
- **Failure Manifestation:** Virtually all modern social platforms deliberately scrub EXIF metadata upon upload for user privacy. In real-world operation, Stage 1 returns empty or near-empty metadata structures in >95% of forwarded messaging cases.

---

## 4. Semantic, Multilingual, and Contextual Blind Spots

### 4.1 "Cheapfakes" and Out-of-Context Recycling
- **Definition:** The majority of viral disinformation does **not** involve sophisticated AI generation. It consists of genuine, unmanipulated historical photographs paired with false, provocative captions (e.g., an authentic photograph of a 2017 industrial fire presented as an ongoing attack in 2026).
- **Failure Manifestation:** 
  - Stage 3 (Authenticity) correctly classifies the image as `AUTHENTIC`.
  - Classical Forensics (Stage 4) finds zero manipulation.
  - If Reverse Image Search (Stage 11) fails or returns ambiguous results, the Fusion Engine (Stage 14) risks certifying the meme as harmlessly authentic, failing to catch the deceptive contextual framing.

### 4.2 Vernacular Multilingual OCR and Code-Switching ("Hinglish")
- **Vulnerability:** OCR engine (EasyOCR) must parse colloquial, code-mixed text overlays (e.g. Hindi written in Roman script, or Devanagari mixed with English numerals and slang).
- **Failure Manifestation:**
  - Decorative, non-standard meme fonts and low contrast against complex photographic backgrounds lead to high word error rates (WER).
  - Subsequent stages (Text Normalization, Claim Extraction) receive garbled character strings, causing Stage 10 (Fact Check API) to execute invalid queries.

### 4.3 Fact-Checking API Coverage Asymmetry
- **Vulnerability:** Google Fact Check Tools API coverage is heavily skewed toward mainstream English news claims.
- **Failure Manifestation:** Hyper-local, vernacular claims originating in tier-2/tier-3 Indian districts frequently return zero hits, leading Stage 10 to emit empty results. The system must fall back purely on intrinsic heuristics.

---

## 5. Repository Gaps & Implementation Incompleteness

1. **Empty Heatmap Storage:** The directory `data/heatmaps/` exists but contains 0 files. While `backend/app/services/xai/gradcam.py` contains verified code for generating and saving Grad-CAM heatmaps, no historical output heatmaps are saved in git tracking.
2. **Missing End-to-End Pipeline Execution Logs:** No single system log captures all 19 stages executing sequentially on a physical sample image in the current workspace.
3. **No GPU Docker Environment:** `docker/` is currently an empty directory. Docker Compose configuration for orchestrated production deployment has not been authored.
