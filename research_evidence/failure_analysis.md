# TruthLens Failure Modes & Edge Case Analysis

> **Document Status:** Comprehensive Technical Audit of Model & Pipeline Failure Modes
> **Source Artifacts:** `docs/phase6_evaluation_report.md`, `docs/phase7_evaluation_report.md`, `experiments/reports/phase7_locked_test_results.json`

---

## 1. Container Format Shortcut Failure (Phase 5/6, Candidate E3)

### Mechanism of Failure
In Tiny-GenImage training data, 100% of authentic photographs were stored as JPEG files, whereas 100% of synthetic generations were stored as PNG files. Unaugmented fine-tuning (Candidate E3) learned this spurious container artifact rather than semantic generative features:
- **Metadata Separability:** A Random Forest trained strictly on non-pixel metadata achieved an ROC-AUC of 1.0000.
- **Diagnostic Transcoding Probe:** When real test images were transcoded in-memory to PNG or AI images to JPEG-90, Candidate E3 exhibited a catastrophic **45.6% prediction flip rate**, with an extreme mean logit absolute shift of **6.12**.

### Empirical Resolution
In Phase 7, matched in-memory augmentation (randomized JPEG/PNG/WebP transcoding, scale jitter, and quality scaling applied identically to both classes) eliminated format shortcut exploitation:
- Candidate S2 prediction flip rate on JPEG-90 probe dropped to **11.0%**.
- Mean logit shift dropped from 6.12 to **0.793**, demonstrating compression invariance.

---

## 2. Cross-Generator Architectural Transfer Gap (Phase 7, Candidate S2)

### Mechanism of Failure
Candidate S2 achieved exceptional in-distribution performance on Tiny-GenImage (ROC-AUC 0.9423, +0.3300 gain over baseline). However, when evaluated on the held-out Frontier Benchmark 2026-10 (comprising modern generators: DALL-E 3, FLUX.1 [dev], FLUX.1 [schnell], Midjourney v6, Recraft v3, SD 3 Medium, SD 3.5 Large, SD 3.5 Turbo), performance collapsed:
- **Baseline ROC-AUC:** 0.6501 (Balanced Accuracy: 0.6228)
- **Candidate S2 ROC-AUC:** 0.4234 (Balanced Accuracy: 0.4849)
- **Net Degradation ($\Delta$ AUC):** **-0.2267** (95% CI: $[-0.2781, -0.1720]$)

### Physical & Architectural Cause
1. **Representational Overfitting:** Partial fine-tuning on 2023 diffusion models (SD 1.4/1.5, GLIDE, Midjourney v4) overfitted the SigLIP feature extractor to early UNet deconvolution checkerboard artifacts.
2. **Generative Shift:** Modern late-2026 generators use Diffusion Transformers (DiT), flow-matching objectives, and advanced text-conditioning encoders that eliminate early diffusion noise footprints.
3. **Catastrophic Forgetting:** Fine-tuning destroyed the broad, generalized semantic representations learned by SigLIP during web-scale contrastive pre-training.
4. **Governance Consequence:** Candidate S2 was rejected for production; the frozen baseline was retained.

---

## 3. High Real-Image False Alarm Rate (FAR) on Open Domains

Across all tested models, authentic photographs from diverse web sources trigger elevated false alarms at standard binary decision thresholds ($t = 0.5$):
- **Frontier Benchmark Test Baseline:** Real Accuracy = 40.00% (False Alarm Rate = 60.00%).
- **Frontier Benchmark Test Candidate S2:** Real Accuracy = 66.03% (False Alarm Rate = 33.97%).
- **WildAIGC-Bench Test Baseline:** Real Accuracy = 4.02% (False Alarm Rate = 95.98%).
- **Operational Requirement:** To guarantee a strict operational false alarm rate $\le 5\%$, decision thresholds must be elevated to $t \approx 0.93$, with ambiguous predictions routed to human reviewers.

---

## 4. Classical Forensics Failure Modes

Classical forensic indicators (ELA, noise variance, JPEG ghosting) produce systematic failure modes under social media distribution:
- **Social Platform Transcoding:** Platforms such as WhatsApp, Instagram, and Twitter re-compress uploads, stripping original quantization tables and generating false ELA edge anomalies.
- **Resampling False Positives:** Normal image resizing (e.g., generating website thumbnails) introduces periodic interpolation artifacts that trigger resampling detectors on authentic photographs.
- **Mitigation:** TruthLens restricts forensics strictly to `informational` status in evidence fusion; they cannot cast directional authenticity votes.

---

## 5. OCR & Claim Extraction Edge Cases

- **Stylized & Low-Contrast Text:** Text rendered on complex backgrounds or with non-standard typography causes character omissions in EasyOCR.
- **Context Detachment:** Rule-based claim extraction detects surface entity tokens (dates, names) but cannot determine whether an extract is a verifiable factual assertion or creative hyperbole.
