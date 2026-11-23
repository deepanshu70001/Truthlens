"""Scientific OCR Evaluation Benchmark for TruthLens.

Evaluates EasyOCR, PyTesseract 5.4.0, and Integrated TruthLens OCR routing
across English, Hindi (Devanagari), and Code-switched text, under 5 visual
conditions: Clean, Gaussian Blur, JPEG Compression, Skew/Rotation, and Low Contrast.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Levenshtein distance implementation for CER and WER
def levenshtein_distance(s1: list | str, s2: list | str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def compute_cer(reference: str, hypothesis: str) -> float:
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = levenshtein_distance(ref, hyp)
    return float(dist) / len(ref)

def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist) / len(ref_words)

# Curated reference benchmark samples
BENCHMARK_CORPUS = [
    # English samples
    {"id": "en_01", "lang": "en", "text": "Breaking news: Official verification report confirmed authentic."},
    {"id": "en_02", "lang": "en", "text": "Multimodal forensic analysis detects generative artifacts in real time."},
    {"id": "en_03", "lang": "en", "text": "The election commission announced polling schedules for next month."},
    {"id": "en_04", "lang": "en", "text": "High court orders comprehensive environmental audit of urban infrastructure."},
    {"id": "en_05", "lang": "en", "text": "Renewable solar energy capacity reached record fifty gigawatts in 2026."},
    {"id": "en_06", "lang": "en", "text": "Caution: Do not share unverified viral messages on social messaging apps."},
    {"id": "en_07", "lang": "en", "text": "Medical researchers discovered new therapeutic compounds with ninety percent efficacy."},
    {"id": "en_08", "lang": "en", "text": "Digital signatures verified via cryptographic C2PA content credentials standard."},
    # Hindi samples (Devanagari script)
    {"id": "hi_01", "lang": "hi", "text": "सत्यमेव जयते भारत सरकार का आधिकारिक आदर्श वाक्य है।"},
    {"id": "hi_02", "lang": "hi", "text": "चुनाव आयोग ने मतदान की तारीखों की घोषणा कर दी है।"},
    {"id": "hi_03", "lang": "hi", "text": "सोशल मीडिया पर भ्रामक सूचनाओं से सावधान रहें और पुष्टि करें।"},
    {"id": "hi_04", "lang": "hi", "text": "मौसम विभाग ने अगले चौबीस घंटों में भारी बारिश की चेतावनी दी।"},
    {"id": "hi_05", "lang": "hi", "text": "डिजिटल साक्षरता और तथ्य जांच आज के समय की महत्वपूर्ण आवश्यकता है।"},
    {"id": "hi_06", "lang": "hi", "text": "सर्वोच्च न्यायालय ने ऐतिहासिक निर्णय में पर्यावरण संरक्षण को प्राथमिकता दी।"},
    {"id": "hi_07", "lang": "hi", "text": "वैज्ञानिक अनुसंधान संस्थान ने नई तकनीक का सफल परीक्षण पूरा किया।"},
    {"id": "hi_08", "lang": "hi", "text": "भारतीय रेल ने यात्रियों की सुविधा के लिए विशेष ट्रेन सेवा शुरू की।"},
    # Mixed / Code-switched samples
    {"id": "mix_01", "lang": "mix", "text": "Official Notice: दिल्ली मेट्रो की नई लाइन कल से शुरू होगी।"},
    {"id": "mix_02", "lang": "mix", "text": "Breaking: भारत सरकार ने नया AI Policy Framework जारी किया।"},
    {"id": "mix_03", "lang": "mix", "text": "Fact Check: सोशल मीडिया पर वायरल यह दावा Totally False है।"},
    {"id": "mix_04", "lang": "mix", "text": "Cyber Security Alert: अनजान लिंक पर कभी Click न करें।"},
]

def render_text_image(text: str, lang: str) -> Image.Image:
    """Render text into a clean high-contrast image using appropriate font."""
    font_size = 28
    font_path = "C:\\Windows\\Fonts\\Nirmala.ttc" if (lang in ("hi", "mix") or any(ord(c) > 127 for c in text)) else "C:\\Windows\\Fonts\\arial.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Determine canvas size
    lines = text.split("\n")
    # approximate bbox
    dummy = Image.new("RGB", (1, 1))
    draw_dummy = ImageDraw.Draw(dummy)
    max_w = 0
    total_h = 0
    for l in lines:
        bbox = draw_dummy.textbbox((0, 0), l, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_w = max(max_w, w)
        total_h += h + 16

    img_w = max(600, max_w + 60)
    img_h = max(120, total_h + 50)
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y_offset = 25
    for l in lines:
        draw.text((30, y_offset), l, font=font, fill=(10, 10, 10))
        bbox = draw.textbbox((0, 0), l, font=font)
        y_offset += (bbox[3] - bbox[1]) + 16
    return img

def apply_degradation(img: Image.Image, condition: str) -> Image.Image:
    """Apply specific visual degradation condition."""
    if condition == "clean":
        return img.copy()
    elif condition == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.8))
    elif condition == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=22)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif condition == "skew":
        # Rotate by 6 degrees with white background fill
        return img.rotate(6, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    elif condition == "low_contrast":
        # Reduce contrast and lower brightness
        enhancer = ImageEnhance.Contrast(img)
        low_c = enhancer.enhance(0.35)
        brightener = ImageEnhance.Brightness(low_c)
        return brightener.enhance(0.75)
    raise ValueError(f"Unknown condition: {condition}")

def run_evaluation():
    from backend.app.services.ocr.service import OCRService
    
    print("=" * 70)
    print("TruthLens Scientific OCR Evaluation Benchmark (Package C)")
    print("=" * 70)

    # Initialize OCR Service
    ocr_service = OCRService(languages=["en", "hi"], gpu=True)
    ocr_service.load()
    print(f"OCR Service loaded. Primary Engine: {ocr_service._engine}")
    
    conditions = ["clean", "blur", "jpeg", "skew", "low_contrast"]
    engines = ["EasyOCR", "PyTesseract", "Integrated"]
    
    results: List[Dict[str, Any]] = []
    
    for item in BENCHMARK_CORPUS:
        item_id = item["id"]
        lang = item["lang"]
        ref_text = item["text"]
        base_img = render_text_image(ref_text, lang)
        
        for cond in conditions:
            test_img = apply_degradation(base_img, cond)
            
            # 1. EasyOCR
            easy_start = time.perf_counter()
            easy_text = ""
            easy_conf = 0.0
            try:
                if ocr_service._reader is not None:
                    blocks, _ = ocr_service._run_easyocr(test_img, [])
                    easy_text = " ".join(b.text for b in blocks if b.text).strip()
                    confs = [b.confidence for b in blocks if b.confidence is not None]
                    easy_conf = float(np.mean(confs)) if confs else 0.0
            except Exception as e:
                easy_text = ""
            easy_lat = (time.perf_counter() - easy_start) * 1000.0
            easy_cer = compute_cer(ref_text, easy_text)
            easy_wer = compute_wer(ref_text, easy_text)
            
            # 2. PyTesseract
            tess_start = time.perf_counter()
            tess_text = ""
            tess_conf = 0.0
            try:
                # Force load tesseract if needed
                if ocr_service._tesseract is None:
                    ocr_service._load_tesseract()
                tess_blocks = ocr_service._run_tesseract(test_img, [])
                tess_text = " ".join(b.text for b in tess_blocks if b.text).strip()
                tess_confs = [b.confidence for b in tess_blocks if b.confidence is not None]
                tess_conf = float(np.mean(tess_confs)) if tess_confs else 0.0
            except Exception as e:
                tess_text = ""
            tess_lat = (time.perf_counter() - tess_start) * 1000.0
            tess_cer = compute_cer(ref_text, tess_text)
            tess_wer = compute_wer(ref_text, tess_text)
            
            # 3. Integrated Service extract()
            int_start = time.perf_counter()
            int_res = ocr_service.extract(test_img)
            int_lat = (time.perf_counter() - int_start) * 1000.0
            int_text = int_res.normalized_text or int_res.text or ""
            int_conf = int_res.total_confidence or 0.0
            int_cer = compute_cer(ref_text, int_text)
            int_wer = compute_wer(ref_text, int_text)
            
            record = {
                "id": item_id,
                "language_slice": lang,
                "condition": cond,
                "reference_length_chars": len(ref_text),
                "reference_length_words": len(ref_text.split()),
                "easyocr": {
                    "text": easy_text,
                    "cer": round(easy_cer, 4),
                    "wer": round(easy_wer, 4),
                    "confidence": round(easy_conf, 4),
                    "latency_ms": round(easy_lat, 2)
                },
                "pytesseract": {
                    "text": tess_text,
                    "cer": round(tess_cer, 4),
                    "wer": round(tess_wer, 4),
                    "confidence": round(tess_conf, 4),
                    "latency_ms": round(tess_lat, 2)
                },
                "integrated": {
                    "engine_used": int_res.engine,
                    "text": int_text,
                    "cer": round(int_cer, 4),
                    "wer": round(int_wer, 4),
                    "confidence": round(int_conf, 4),
                    "latency_ms": round(int_lat, 2)
                }
            }
            results.append(record)

    # Compute aggregate statistics
    def agg(engine_name: str, key: str, subset: list) -> float:
        vals = [r[engine_name][key] for r in subset]
        return float(np.mean(vals)) if vals else 0.0

    summary: Dict[str, Any] = {
        "total_benchmark_evaluations": len(results),
        "corpus_composition": {
            "english_items": 8,
            "hindi_items": 8,
            "mixed_items": 4,
            "conditions_per_item": 5
        },
        "overall_by_engine": {},
        "by_language_slice": {},
        "by_condition_slice": {},
        "confidence_accuracy_correlation": {}
    }

    for eng in ["easyocr", "pytesseract", "integrated"]:
        mean_cer = agg(eng, "cer", results)
        mean_wer = agg(eng, "wer", results)
        mean_lat = agg(eng, "latency_ms", results)
        mean_conf = agg(eng, "confidence", results)
        summary["overall_by_engine"][eng] = {
            "mean_cer": round(mean_cer, 4),
            "mean_wer": round(mean_wer, 4),
            "mean_latency_ms": round(mean_lat, 2),
            "mean_confidence": round(mean_conf, 4)
        }

    for lang in ["en", "hi", "mix"]:
        sub = [r for r in results if r["language_slice"] == lang]
        summary["by_language_slice"][lang] = {}
        for eng in ["easyocr", "pytesseract", "integrated"]:
            summary["by_language_slice"][lang][eng] = {
                "mean_cer": round(agg(eng, "cer", sub), 4),
                "mean_wer": round(agg(eng, "wer", sub), 4),
                "mean_latency_ms": round(agg(eng, "latency_ms", sub), 2)
            }

    for cond in conditions:
        sub = [r for r in results if r["condition"] == cond]
        summary["by_condition_slice"][cond] = {}
        for eng in ["easyocr", "pytesseract", "integrated"]:
            summary["by_condition_slice"][cond][eng] = {
                "mean_cer": round(agg(eng, "cer", sub), 4),
                "mean_wer": round(agg(eng, "wer", sub), 4)
            }

    # Correlation between confidence and character accuracy (1 - CER)
    for eng in ["easyocr", "pytesseract", "integrated"]:
        confs = np.array([r[eng]["confidence"] for r in results])
        accs = np.array([max(0.0, 1.0 - r[eng]["cer"]) for r in results])
        if np.std(confs) > 1e-6 and np.std(accs) > 1e-6:
            r_val = float(np.corrcoef(confs, accs)[0, 1])
        else:
            r_val = 0.0
        summary["confidence_accuracy_correlation"][eng] = round(r_val, 4)

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "easyocr_version": "1.7.2",
            "pytesseract_version": "0.3.13",
            "tesseract_version": "5.4.0",
            "languages": ["en", "hi"],
            "total_samples": len(results)
        },
        "summary": summary,
        "detailed_results": results
    }

    out_path = Path("experiments/reports/ocr_scientific_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\nBenchmark Finished! Summary Results:")
    print("-" * 70)
    print(f"{'Engine':<15} | {'Mean CER':<10} | {'Mean WER':<10} | {'Mean Latency':<15} | {'Conf-Acc Corr':<15}")
    print("-" * 70)
    for eng in ["easyocr", "pytesseract", "integrated"]:
        ov = summary["overall_by_engine"][eng]
        corr = summary["confidence_accuracy_correlation"][eng]
        print(f"{eng:<15} | {ov['mean_cer']:<10.4f} | {ov['mean_wer']:<10.4f} | {ov['mean_latency_ms']:<10.1f} ms | {corr:<15.4f}")
    print("-" * 70)
    print(f"Full benchmark artifact written to: {out_path}")

if __name__ == "__main__":
    run_evaluation()
