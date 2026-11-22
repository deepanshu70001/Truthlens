"""Offline synthetic OCR smoke benchmark; never downloads weights or calls APIs.

Run from the repository root with:
    python experiments/ocr_runtime_smoke.py --devices cpu,cuda

All inputs are generated in memory. EasyOCR is explicitly cache-only through
OCRService.load(); Tesseract is exercised only when its executable and each
requested language pack are locally available.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import io
import json
import platform
import shutil
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _font(script: str, size: int) -> ImageFont.FreeTypeFont:
    configured = Path(__import__("os").environ.get("TRUTHLENS_OCR_TEST_FONT", ""))
    if script == "hi":
        candidates = [
            configured,
            Path(r"C:\Windows\Fonts\Nirmala.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-VF.ttf"),
        ]
    else:
        candidates = [configured, Path(r"C:\Windows\Fonts\arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    raise FileNotFoundError(f"No local {script} synthetic-test font found; set TRUTHLENS_OCR_TEST_FONT")


def _text_image(text: str, script: str) -> Image.Image:
    image = Image.new("RGB", (1600, 280), "white")
    draw = ImageDraw.Draw(image)
    font = _font(script, 68)
    bounds = draw.textbbox((0, 0), text, font=font)
    x = max(24, (image.width - (bounds[2] - bounds[0])) // 2)
    y = max(12, (image.height - (bounds[3] - bounds[1])) // 2 - bounds[1])
    draw.text((x, y), text, font=font, fill="black")
    return image


def _synthetic_cases() -> dict[str, dict[str, Any]]:
    english = "THE CITY OPENED 125 NEW SCHOOLS IN 2026"
    hindi = "दिल्ली में 125 नए विद्यालय 2026 में खुले"
    mixed = "दिल्ली में 125 new schools 2026 में खुले"
    english_image = _text_image(english, "en")
    hindi_image = _text_image(hindi, "hi")
    mixed_image = _text_image(mixed, "hi")

    small = hindi_image.resize((400, 70), Image.Resampling.LANCZOS)
    jpeg_buffer = io.BytesIO()
    hindi_image.save(jpeg_buffer, format="JPEG", quality=22)
    jpeg_buffer.seek(0)
    jpeg = Image.open(jpeg_buffer).convert("RGB")

    return {
        "english": {"image": english_image, "reference": english, "format": "PNG", "script": "en"},
        "hindi": {"image": hindi_image, "reference": hindi, "format": "PNG", "script": "hi"},
        "mixed_hindi_english": {"image": mixed_image, "reference": mixed, "format": "PNG", "script": "hi"},
        "blank": {"image": Image.new("RGB", (1600, 280), "white"), "reference": None, "format": "PNG", "script": None},
        "rotated_hindi": {"image": hindi_image.rotate(7, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white"), "reference": hindi, "format": "PNG", "script": "hi"},
        "blurred_hindi": {"image": hindi_image.filter(ImageFilter.GaussianBlur(radius=1.5)), "reference": hindi, "format": "PNG", "script": "hi"},
        "low_resolution_hindi": {"image": small, "reference": hindi, "format": "PNG", "script": "hi"},
        "jpeg_compressed_hindi_q22": {"image": jpeg, "reference": hindi, "format": "JPEG", "script": "hi"},
    }


def _metric_normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _edit_distance(left: Any, right: Any) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _errors(reference: str | None, prediction: str) -> dict[str, float | None]:
    if reference is None:
        return {"cer": None, "wer": None}
    ref = _metric_normalize(reference)
    hyp = _metric_normalize(prediction)
    cer = _edit_distance(ref, hyp) / max(1, len(ref))
    ref_words, hyp_words = ref.split(), hyp.split()
    wer = _edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    return {"cer": cer, "wer": wer}


def _box_bounds(blocks: list[Any], width: int, height: int) -> bool:
    for block in blocks:
        if len(block.bbox) != 4:
            return False
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in block.bbox):
            return False
    return True


def _record(service: Any, case_name: str, case: dict[str, Any], language_list: list[str], engine_name: str, device: str) -> dict[str, Any]:
    image = case["image"]
    started = time.perf_counter()
    result = service.extract(image)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    prediction = result.normalized_text
    return {
        "case": case_name,
        "status": result.status,
        "engine": result.engine or engine_name,
        "device": device,
        "language_configuration": language_list,
        "reference_transcript": case["reference"],
        "predicted_text": prediction,
        "image": {"width": image.width, "height": image.height, "format": case["format"], "mode": image.mode},
        "elapsed_ms_wall": elapsed_ms,
        "engine_reported_runtime_ms": result.runtime_ms,
        "ocr_confidence": result.total_confidence,
        "confidence_semantics": "Arithmetic mean of available engine block confidences; not a calibrated probability.",
        "bounding_boxes": [block.bbox for block in result.blocks],
        "coordinate_bounds_valid": _box_bounds(result.blocks, image.width, image.height),
        "has_text": result.has_text,
        "warnings": result.warnings,
        **_errors(case["reference"], prediction),
    }


def _easyocr_weight_status(model_dir: Path) -> dict[str, bool]:
    return {
        "craft_mlt_25k.pth": (model_dir / "craft_mlt_25k.pth").is_file(),
        "english_g2.pth": (model_dir / "english_g2.pth").is_file(),
        "devanagari.pth": (model_dir / "devanagari.pth").is_file(),
    }


def run(devices: list[str]) -> dict[str, Any]:
    from backend.app.config import settings
    from backend.app.services.ocr.service import OCRService

    cases = _synthetic_cases()
    model_dir = Path(settings.OCR_MODEL_DIR)
    torch_available = importlib.util.find_spec("torch") is not None
    cuda_available = False
    cuda_device = None
    if torch_available:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)

    tesseract_path = shutil.which("tesseract")
    tesseract_status: dict[str, Any] = {
        "pytesseract_package": importlib.util.find_spec("pytesseract") is not None,
        "executable_path": tesseract_path,
        "version": None,
        "available_language_packs": [],
        "configurations": {},
    }
    if tesseract_path and tesseract_status["pytesseract_package"]:
        try:
            import pytesseract
            tesseract_status["version"] = str(pytesseract.get_tesseract_version())
            tesseract_status["available_language_packs"] = pytesseract.get_languages(config="")
        except Exception as exc:
            tesseract_status["readiness_error"] = type(exc).__name__

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "easyocr_version": _version("easyocr"),
            "pytesseract_version": _version("pytesseract"),
            "torch_version": _version("torch"),
            "cuda_available": cuda_available,
            "cuda_device": cuda_device,
            "model_cache_dir": str(model_dir),
        },
        "easyocr": {
            "package_available": importlib.util.find_spec("easyocr") is not None,
            "weights_available": _easyocr_weight_status(model_dir),
            "application_configured_languages": [code.strip() for code in settings.OCR_LANGUAGES.split(",") if code.strip()],
            "download_policy": "Reader(download_enabled=False); local weights only",
            "language_configurations": {},
        },
        "tesseract": tesseract_status,
        "synthetic_fixtures": list(cases),
        "fixture_recipe": {
            "base_dimensions_px": [1600, 280],
            "english_reference": "THE CITY OPENED 125 NEW SCHOOLS IN 2026",
            "hindi_reference": "दिल्ली में 125 नए विद्यालय 2026 में खुले",
            "mixed_reference": "दिल्ली में 125 new schools 2026 में खुले",
            "rotated_degrees": 7,
            "blur_gaussian_radius": 1.5,
            "low_resolution_dimensions_px": [400, 70],
            "jpeg_quality": 22,
            "generation": "Synthetic PIL images created in memory; no source image or dataset fixture is read.",
        },
        "results": [],
        "metric_method": "NFC + casefold + whitespace collapse; CER uses Unicode code points; WER uses whitespace-delimited tokens; punctuation is retained.",
    }

    for device in devices:
        if device == "cuda" and not cuda_available:
            report["easyocr"]["language_configurations"][f"cuda"] = {"status": "unavailable", "reason": "CUDA is not available."}
            continue
        for language_list in (["en"], ["hi"], ["en", "hi"]):
            config_name = "+".join(language_list)
            service = OCRService(languages=language_list, gpu=(device == "cuda"))
            load_started = time.perf_counter()
            try:
                service.load()
                load_ms = round((time.perf_counter() - load_started) * 1000, 3)
            except Exception as exc:
                load_ms = round((time.perf_counter() - load_started) * 1000, 3)
                report["easyocr"]["language_configurations"][f"{device}:{config_name}"] = {
                    "status": "initialization_error", "load_ms": load_ms, "error_type": type(exc).__name__,
                }
                continue

            is_easyocr = service._reader is not None
            conf_record: dict[str, Any] = {
                "status": "ready" if is_easyocr else "unavailable",
                "engine": "EasyOCR" if is_easyocr else None,
                "load_ms": load_ms,
                "warnings": list(service._load_warnings),
                "device_requested": device,
                "device_used": str(getattr(service._reader, "device", "unknown")) if is_easyocr else None,
                "download_enabled": False,
            }
            if not is_easyocr and config_name == "en" and not report["easyocr"]["weights_available"]["english_g2.pth"]:
                conf_record["reason"] = "Local English recognition weight english_g2.pth is missing; automatic downloads are disabled."
            report["easyocr"]["language_configurations"][f"{device}:{config_name}"] = conf_record
            if is_easyocr:
                for case_name, case in cases.items():
                    report["results"].append(_record(service, case_name, case, language_list, "EasyOCR", device))
            service.unload()

    # Tesseract is exercised as an independent engine only for configurations
    # whose executable and complete language packs passed the adapter checks.
    for language_list in (["en"], ["hi"], ["en", "hi"]):
        config_name = "+".join(language_list)
        service = OCRService(languages=language_list, gpu=False)
        service._load_tesseract()
        ready = service._tesseract is not None
        key = config_name
        report["tesseract"]["configurations"][key] = {
            "status": "ready" if ready else "unavailable",
            "engine": service._engine if ready else None,
            "warnings": list(service._load_warnings),
        }
        if ready:
            for case_name, case in cases.items():
                report["results"].append(_record(service, case_name, case, language_list, "PyTesseract", "cpu"))
        service.unload()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cpu", help="Comma-separated EasyOCR devices to try: cpu,cuda")
    parser.add_argument("--output", default="experiments/reports/ocr_runtime_smoke.json")
    args = parser.parse_args()
    requested_devices = [device.strip().lower() for device in args.devices.split(",") if device.strip()]
    invalid = set(requested_devices) - {"cpu", "cuda"}
    if invalid:
        parser.error(f"Unsupported device(s): {', '.join(sorted(invalid))}")
    report = run(requested_devices)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "easyocr": report["easyocr"]["language_configurations"],
        "tesseract": report["tesseract"],
        "inference_records": len(report["results"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
