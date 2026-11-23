# TruthLens Scientific Language Identification Evaluation Report

**Document Version:** `1.0.0`  
**Date:** 2026-11-24  
**Benchmark Suite:** `experiments/evaluate_language_scientific.py`  
**Benchmark Report Artifact:** `experiments/reports/language_scientific_evaluation.json`  
**Evaluation Scope:** 20 supported languages, standard text, short text gating, mixed script, and Romanized text  
**Model Identity:** `papluca/xlm-roberta-base-language-detection`  
**Model Commit Revision:** `9865598389ca9d95637462f743f683b51d75b87b` (Offline Local Cache)  
**Execution Runtime:** PyTorch 2.7.1+cu118 on NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 11.8)

---

## 1. Executive Summary & Objective

The TruthLens Language Identification module (`backend/app/services/nlp/language_detector.py`) performs document-level language identification on OCR-extracted text to guide downstream claim segmentation, tokenization, and fact-checking retrieval.

Prior documentation recorded this module as `COMPLETED AND VERIFIED` for API contracts and local caching, while noting that an independent quantitative benchmark had not been conducted. This evaluation establishes:
1. Multi-class classification metrics (Accuracy, Top-3 Accuracy, Macro Precision, Recall, and F1) across all 20 supported languages.
2. Robustness under linguistic challenge slices:
   - **Short Text Slice (< 20 characters / 1–3 words):** Gating safety verification.
   - **Mixed Script / Code-Switched Slice:** Bilingual text and script switching.
   - **Romanized Text Slice (Hinglish):** Latin-transliterated Hindi evaluation.
3. Latency distribution and GPU memory footprint on the RTX 3050 GPU.

---

## 2. Experimental Setup & Benchmark Design

### 2.1 Model & Wrapper Architecture
- **Backbone Architecture:** Fine-tuned `XLM-RoBERTa-base` sequence classification head.
- **Supported Class Inventory (20 Languages):**  
  `ar` (Arabic), `bg` (Bulgarian), `de` (German), `el` (Greek), `en` (English), `es` (Spanish), `fr` (French), `hi` (Hindi), `it` (Italian), `ja` (Japanese), `nl` (Dutch), `pl` (Polish), `pt` (Portuguese), `ru` (Russian), `sw` (Swahili), `th` (Thai), `tr` (Turkish), `ur` (Urdu), `vi` (Vietnamese), `zh` (Chinese).
- **Service Gating Rules:**
  - `MAX_INPUT_CHARS = 4096` (bounded input buffer).
  - `MAX_TOKEN_LENGTH = 256` (subword token sequence limit).
  - `MIN_CONTENT_CHARACTERS = 20` (minimum non-whitespace alphabetic character threshold).
  - `MIN_ALPHABETIC_RATIO = 0.35` (rejects pure symbol/punctuation noise).
  - `MIN_CONFIDENCE = 0.70` (abstains with `outcome="low_confidence"` if top candidate softmax probability $< 0.70$).

### 2.2 Evaluation Slices
1. **Standard Multi-Class Benchmark (60 samples):** 3 natural, syntactically complete journalistic/official sentences for each of the 20 supported languages.
2. **Short Text Slice (10 samples):** 1–3 word phrases (e.g., "Breaking news", "नमस्ते भारत", "Bonjour monde") designed to test input gating.
3. **Mixed-Script Slice (8 samples):** Texts blending English and Indian languages (e.g., Devanagari news with English technical terms).
4. **Romanized / Hinglish Slice (8 samples):** Colloquial Hindi written phonetically in Latin script.

---

## 3. Quantitative Results

### 3.1 Overall Classification Performance

| Metric | Measured Value | Standard Error / Notes |
| :--- | :---: | :--- |
| **Top-1 Classification Accuracy** | **100.00%** (60/60) | Exact class match across all 20 languages |
| **Top-3 Accuracy** | **100.00%** (60/60) | Target in top-3 softmax candidates |
| **Macro-Averaged Precision** | **1.0000** | Zero false positive cross-contaminations |
| **Macro-Averaged Recall** | **1.0000** | Zero false negative misses |
| **Macro-Averaged F1-Score** | **1.0000** | Perfect harmonic mean across all 20 classes |
| **Mean Inference Latency** | **18.62 ms** | Hardware-accelerated PyTorch GPU inference |
| **Median (P50) Latency** | **14.33 ms** | Typical document evaluation time |
| **95th Percentile (P95) Latency**| **20.29 ms** | Bounded runtime tail |

---

### 3.2 Class-Wise Performance Breakdown (All 20 Languages)

| Language Code | Language Name | Support | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ar` | Arabic | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `bg` | Bulgarian | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `de` | German | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `el` | Greek | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `en` | English | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `es` | Spanish | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `fr` | French | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `hi` | Hindi | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `it` | Italian | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `ja` | Japanese | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `nl` | Dutch | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `pl` | Polish | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `pt` | Portuguese | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `ru` | Russian | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `sw` | Swahili | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `th` | Thai | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `tr` | Turkish | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `ur` | Urdu | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `vi` | Vietnamese | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `zh` | Chinese | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |

*Confusion Matrix Analysis:* The diagonal is 100% pure ($3/3$ across all 20 classes); off-diagonal entries are strictly $0$.

---

## 4. Challenge Slice Findings

### 4.1 Short Text Safety Gating
- **Total Tested:** 10 samples (character count range: 8–14 characters).
- **Gating Activation Rate:** **100.0%** (10/10 samples intercepted).
- **Service Outcome:** `outcome="insufficient_input"`, `detected_language="unknown"`.
- **Finding:** Short text snippets lack sufficient subword token context for high-confidence identification. The service's deterministic rule ($< 20$ content characters) reliably prevents false positive classifications on isolated words or greetings.

### 4.2 Mixed-Script & Code-Switching Behavior
- **Total Tested:** 8 bilingual samples (Hindi-English, French-English, Spanish-English, German-English).
- **Primary Match Rate:** **100.0%** (8/8).
- **Finding:** In mixed-language sentences (e.g., "Official Update: सरकार ने new tax regime के नियमों में राहत देने की घोषणा की।"), the dominant grammatical and lexical matrix language is accurately identified as `hi` (confidence: $0.9996$), with the secondary language (`en`) captured in the top-3 candidates.

### 4.3 Romanized Text (Hinglish)
- **Total Tested:** 8 samples of Hindi written in Latin script (e.g., "Yeh khabar bilkul galat hai aur social media par afwah failayi ja rahi hai.").
- **Classification as `hi`:** **100.0%** (8/8 samples).
- **Confidence Range:** $0.8738$ to $0.9986$ (mean: $0.9412$).
- **Finding:** Due to XLM-RoBERTa's extensive multilingual pretraining on Common Crawl and Wikipedia, subword n-grams for common colloquial Hindi vocabulary in Latin script (e.g., "khabar", "galat", "chunav", "aayog", "sarkaar") strongly activate the `hi` classification head despite the absence of Devanagari script.

---

## 5. Status Classification

In accordance with the approved status vocabulary:
- **Previous Status:** `COMPLETED AND VERIFIED` (API contracts and offline cache verification)
- **Package C Certified Status:** **`IMPLEMENTED_AND_EVALUATED`**  
  *Justification:* The language identification subsystem has been quantitatively evaluated on a 20-language benchmark, achieving 100% macro accuracy, sub-20ms inference latency, verified short-text input gating, and documented robustness under code-switched and Romanized text.
