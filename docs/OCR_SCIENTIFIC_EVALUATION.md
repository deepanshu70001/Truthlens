# TruthLens Scientific OCR Evaluation Benchmark

**Document Version:** `1.0.0`  
**Date:** 2026-11-24  
**Benchmark Suite:** `experiments/evaluate_ocr_scientific.py`  
**Benchmark Report Artifact:** `experiments/reports/ocr_scientific_evaluation.json`  
**Execution Environment:** Windows 11 Enterprise, Python 3.12.10, PyTorch 2.7.1+cu118, NVIDIA RTX 3050 GPU (CUDA 11.8)  
**Tesseract Engine:** Tesseract 5.4.0.20240606 (`C:\Program Files\Tesseract-OCR\tesseract.exe`)  
**EasyOCR Version:** 1.7.2 (CRAFT Text Detector + Deep Text Recognizer)

---

## 1. Executive Summary & Research Context

Prior audits (`docs/MODULE_STATUS_REGISTER.md` row 7 and `docs/ocr_factcheck_audit.md`) identified that while the dual-engine OCR pipeline with opportunistic Tesseract fallback was functionally implemented and passing integration tests, **no independent reference transcript had been adopted for empirical Character Error Rate (CER) and Word Error Rate (WER) quantification**, leaving engine confidence uncalibrated.

Package C resolves this gap by executing an automated empirical evaluation across:
1. **EasyOCR 1.7.2** (Primary GPU-accelerated engine)
2. **PyTesseract 5.4.0** (Tesseract LSTM engine with `eng+hin` language packs)
3. **TruthLens Integrated OCR Routing** (EasyOCR with automatic fallback)

The benchmark assesses **100 distinct image evaluation conditions** spanning 20 representative reference texts across English, Hindi (Devanagari script), and mixed-script code-switched text, subjected to 5 controlled image degradations: Clean, Gaussian Blur, JPEG Compression, Geometric Skew/Rotation, and Low Contrast.

---

## 2. Experimental Methodology & Evaluation Protocol

### 2.1 Benchmark Corpus Composition
The evaluation corpus contains 20 curated reference texts with verified ground truth:
- **English Subcorpus (8 items):** Formal news headlines, judicial rulings, scientific reports, and cyber advisories.
- **Hindi Subcorpus (8 items):** Devanagari script texts rendered with Microsoft `Nirmala.ttc`, covering governmental notices, legal verdicts, and meteorological warnings.
- **Mixed / Code-Switched Subcorpus (4 items):** Bilingual texts combining Latin and Devanagari scripts (e.g., "Official Notice: दिल्ली मेट्रो की नई लाइन कल से शुरू होगी।").

### 2.2 Visual Degradation Slices
For every text sample, five controlled rendering conditions were generated:
1. **Clean Baseline:** Crisp black text rendered at 28pt on high-contrast white background.
2. **Gaussian Blur:** $\sigma = 1.8$ radius Gaussian filtering, simulating focal lens blur and camera motion.
3. **JPEG Compression:** Recompressed at quality $Q = 22$, simulating aggressive social media recompression.
4. **Geometric Skew / Rotation:** Rotated $+6^\circ$ with bicubic interpolation, simulating mobile handheld capture angles.
5. **Low Contrast / Dim Illumination:** Contrast attenuated by $65\%$ ($0.35\times$) with $25\%$ brightness reduction ($0.75\times$), simulating poor lighting.

### 2.3 Mathematical Metrics
- **Character Error Rate (CER):**
  $$\text{CER} = \frac{\text{Levenshtein}(T_{\text{ref}}, T_{\text{hyp}})}{N_{\text{chars}}}$$
- **Word Error Rate (WER):**
  $$\text{WER} = \frac{\text{Levenshtein}_{\text{words}}(W_{\text{ref}}, W_{\text{hyp}})}{N_{\text{words}}}$$
- **Latency:** Wall-clock execution time per image (milliseconds).
- **Confidence-Accuracy Correlation:** Pearson product-moment correlation between reported engine confidence and empirical character accuracy ($1 - \text{CER}$).

---

## 3. Empirical Results

### 3.1 Overall Engine Performance Summary

| Engine | Mean CER | Mean WER | Mean Latency (ms) | Confidence-Accuracy Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: |
| **EasyOCR 1.7.2 (GPU)** | 0.1874 (18.74%) | 0.3282 (32.82%) | 150.88 ms | -0.6113 |
| **PyTesseract 5.4.0 (CPU)** | **0.0670 (6.70%)** | **0.0774 (7.74%)** | 191.72 ms | **+0.7371** |
| **Integrated TruthLens OCR** | 0.1874 (18.74%) | 0.3282 (32.82%) | 136.85 ms | -0.6113 |

> [!IMPORTANT]
> **Key Finding on Engine Routing:**  
> In the current integrated implementation, EasyOCR serves as primary and only falls back to Tesseract when EasyOCR throws an exception or returns empty text blocks. Because EasyOCR returns partial/noisy text rather than throwing an exception or returning zero blocks under degradation, the Tesseract fallback is not automatically triggered on degraded inputs.

---

### 3.2 Performance Across Linguistic Slices

| Language Slice | Metric | EasyOCR | PyTesseract | Integrated |
| :--- | :--- | :---: | :---: | :---: |
| **English (`en`)** | Mean CER | 0.1835 | **0.0513** | 0.1835 |
| | Mean WER | 0.3196 | **0.0578** | 0.3196 |
| | Mean Latency | 195.91 ms | 171.92 ms | 164.78 ms |
| **Hindi (`hi`)** | Mean CER | 0.1874 | **0.0972** | 0.1874 |
| | Mean WER | 0.3262 | **0.1046** | 0.3262 |
| | Mean Latency | 122.09 ms | 204.52 ms | 118.91 ms |
| **Mixed Script (`mix`)** | Mean CER | 0.1951 | **0.0379** | 0.1951 |
| | Mean WER | 0.3491 | **0.0623** | 0.3491 |
| | Mean Latency | 118.38 ms | 205.71 ms | 116.87 ms |

---

### 3.3 Robustness Across Visual Degradation Slices

| Visual Condition | Metric | EasyOCR | PyTesseract | Integrated |
| :--- | :--- | :---: | :---: | :---: |
| **Clean Baseline** | Mean CER | 0.0333 (3.33%) | **0.0029 (0.29%)** | 0.0333 |
| | Mean WER | 0.0901 (9.01%) | **0.0095 (0.95%)** | 0.0901 |
| **Gaussian Blur ($\sigma=1.8$)** | Mean CER | 0.1155 (11.55%) | **0.0009 (0.09%)** | 0.1155 |
| | Mean WER | 0.4486 (44.86%) | **0.0050 (0.50%)** | 0.4486 |
| **JPEG Compression ($Q=22$)** | Mean CER | 0.0301 (3.01%) | **0.0048 (0.48%)** | 0.0301 |
| | Mean WER | 0.1185 (11.85%) | **0.0106 (1.06%)** | 0.1185 |
| **Geometric Skew ($+6^\circ$)** | Mean CER | 0.7259 (72.59%) | **0.3256 (32.56%)** | 0.7259 |
| | Mean WER | 0.8721 (87.21%) | **0.3570 (35.70%)** | 0.8721 |
| **Low Contrast ($0.35\times$)** | Mean CER | 0.0322 (3.22%) | **0.0009 (0.09%)** | 0.0322 |
| | Mean WER | 0.1115 (11.15%) | **0.0050 (0.50%)** | 0.1115 |

---

## 4. In-Depth Failure Mode & Calibration Analysis

### 4.1 Vulnerability to Geometric Skew
Both OCR engines degrade when images exhibit uncorrected rotation or tilt:
- EasyOCR's CRAFT character detector assumes predominantly axis-aligned text lines, fragmenting or skipping rotated word segments ($72.59\%$ CER).
- Tesseract's Leptonica image binarizer and internal page segmentation engine (`PSM 3`) exhibits greater tolerance but still suffers elevated error ($32.56\%$ CER).
- **Engineering Recommendation:** Upstream EXIF rotation normalization (already enforced in TruthLens upload preprocessing) must be complemented by an optional affine deskew step before OCR inference.

### 4.2 Confidence Score Calibration
- **PyTesseract Confidence ($r = +0.7371$):** Tesseract word confidences correlate strongly with empirical accuracy. When Tesseract reports high confidence ($>0.90$), character accuracy exceeds $99\%$.
- **EasyOCR Confidence ($r = -0.6113$):** EasyOCR confidence exhibits negative correlation with accuracy under degradation slices because the CRAFT recognizer assigns high softmax probabilities to erroneous character substitutions when distorted by skew or blur.
- **Scientific Limitation:** EasyOCR raw confidence cannot be used directly as a calibrated probability of textual correctness.

---

## 5. Status Classification

In accordance with the approved status vocabulary:
- **Previous Baseline Status:** `ANNOTATED_DATA_REQUIRED` (Documented in `docs/MODULE_STATUS_REGISTER.md`)
- **Package C Certified Status:** **`IMPLEMENTED_AND_EVALUATED`**  
  *Justification:* The multi-engine OCR pipeline has been rigorously evaluated on a controlled multi-condition, multilingual benchmark with empirical CER, WER, latency, and calibration quantification.
