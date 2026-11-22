# OCR Phase 1: Runtime Readiness and Real Inference Verification

**Date:** 2026-11-22

**Branch:** `fix/truthlens-remediation`

**Scope:** Cache-only OCR readiness and synthetic-image inference. No training, calibration, locked-test, private-user, or external-dataset images were used. No model weights were downloaded. No external API calls were made.

## Summary

EasyOCR 1.7.2 performs real OCR from the local cache in this environment on both CPU and CUDA. The application’s configured `en,hi` reader initializes on both devices and recognized English, Hindi Devanagari, and mixed Hindi-English synthetic inputs. Standalone English-only EasyOCR initialization is unavailable because `english_g2.pth` is absent. This distinction matters: the combined `en+hi` configuration is the one actually verified for the app configuration; the standalone English-only configuration was not verified.

`pytesseract` 0.3.13 is installed, but the Tesseract executable is absent from PATH. Tesseract version, language packs, and inference could not be verified. It is unavailable as a fallback in the current environment.

OCR loads are now cache-only: `OCRService._load()` passes `download_enabled=False` to EasyOCR. Missing local weights cause initialization to fail and the service follows its existing Tesseract fallback path. Tesseract readiness now checks its executable and every configured language pack. Tesseract boxes are clipped to the supplied EXIF-corrected image dimensions and zero-area boxes are discarded.

## Runtime environment and readiness

| Check | Result |
|---|---|
| Operating system | Windows 11, build 26300 |
| Python | 3.12.10 |
| EasyOCR | 1.7.2 installed |
| pytesseract | 0.3.13 installed |
| PyTorch | 2.7.1+cu118 |
| CUDA | Available; NVIDIA GeForce RTX 3050 Laptop GPU |
| EasyOCR CRAFT detector `craft_mlt_25k.pth` | Present locally |
| EasyOCR English-only `english_g2.pth` | Missing locally |
| EasyOCR Hindi `devanagari.pth` | Present locally |
| EasyOCR `hi` reader | Initialized and inferred on CPU and CUDA |
| EasyOCR `en,hi` reader | Initialized and inferred on CPU and CUDA |
| EasyOCR `en` reader | Unavailable on CPU and CUDA; missing local English weight; no download attempted |
| Tesseract executable/version | Not found / unavailable |
| Tesseract `eng` / `hin` packs | Not inspectable because executable is absent |

CUDA was available, so both CPU and CUDA were exercised directly. Both worked for `hi` and `en,hi`; no GPU-to-CPU fallback was triggered or tested. The service does not currently retry on CPU after a CUDA initialization or inference error. If EasyOCR initialization fails, it proceeds to Tesseract; if EasyOCR inference raises, it attempts Tesseract. Empty successful EasyOCR output remains a completed no-text result and does not trigger fallback.

## Synthetic benchmark

The reproducible benchmark is `experiments/ocr_runtime_smoke.py`. It creates eight images in memory using known transcripts: English, Hindi, mixed Hindi-English, blank, 7-degree rotated Hindi, Gaussian-blurred Hindi (radius 1.5), low-resolution Hindi (400×70), and JPEG Q22 Hindi. Base images are 1600×280. Text is rendered with the local Arial and Nirmala UI fonts; no fixture image is opened. The JPEG is encoded and decoded in memory.

The run produced **32 real EasyOCR inference records**: eight cases for each of `hi` and `en,hi` on CPU and CUDA. The engine’s `runtime_ms`, wall time around `extract()`, raw/normalized output, reference, confidence, boxes, and coordinate-validity result are stored per record in [`ocr_runtime_smoke.json`](../experiments/reports/ocr_runtime_smoke.json). The standalone `en` configuration produced no inference records because initialization correctly stopped at the missing local weight.

Metric normalization is Unicode NFC, case-folding, and whitespace collapse; punctuation and numeral script are retained. CER uses character-level Levenshtein distance; WER uses token-level Levenshtein distance over whitespace-separated words. These are synthetic smoke-test measurements, not population-level OCR accuracy estimates.

### Representative CPU results (`en,hi`)

| Case | Reference transcript | OCR output | CER | WER | `extract()` wall time |
|---|---|---|---:|---:|---:|
| English | `THE CITY OPENED 125 NEW SCHOOLS IN 2026` | `THE CITY OPENED १२५ NEW SCHOOLS IN २०२६` | 0.1795 | 0.2500 | 1411.9 ms |
| Hindi | `दिल्ली में 125 नए विद्यालय 2026 में खुले` | `दल्ली में १२५ नए वदियालय २०२६ में खुले` | 0.2500 | 0.5000 | 1091.4 ms |
| Mixed | `दिल्ली में 125 new schools 2026 में खुले` | `दल़्ली में १२५ new schools २०२६ में खुले` | 0.2250 | 0.3750 | 1263.2 ms |
| Blank | No transcript | Empty output; `has_text=false` | N/A | N/A | 756.9 ms |
| Rotated Hindi | Hindi reference above | `में खुले में नए वदियालय 2026 1२५ दल्ली` | 0.5000 | 0.7500 | 1452.7 ms |
| Blurred Hindi | Hindi reference above | `दल्िली में १२५ नए वदियालय २०२६ में खुले` | 0.2750 | 0.5000 | 1112.9 ms |
| Low-resolution Hindi | Hindi reference above | `ल्ली में १२५ नए वदियालय २०२६ में` | 0.4000 | 0.6250 | 369.0 ms |
| JPEG Q22 Hindi | Hindi reference above | `दल्ली में १२५ नए वदियालय २०२६ में खुले` | 0.2500 | 0.5000 | 1154.0 ms |

The CUDA combined-config outputs had the same CER/WER for these cases except the blurred Hindi CER was 0.3000 rather than 0.2750. CUDA `extract()` wall times ranged from 56.5 ms (low resolution) to 220.3 ms (English) for this run; CPU times varied from 369.0 ms to 1452.7 ms. These timings exclude reader initialization and are specific to this local hardware/run. The JSON preserves the full CPU and CUDA measurements.

The output has substantive errors: Hindi spelling/diacritics are not exact, Latin digits become Devanagari digits, and rotation/low resolution reduce accuracy. Mixed text preserved the English words in the combined configuration. Therefore this demonstrates that inference executes and recognizes meaningful text; it does not establish production-grade multilingual accuracy.

The local Pillow build did not support passing a script language to the text renderer (it raised a missing-libraqm error). The benchmark therefore rendered with the installed fonts through Pillow’s default layout. Hindi synthetic typography may not match fully shaped production text; treat the Hindi CER/WER as smoke-test values only. The generated image definitions and every observed transcript remain available for reproduction in the script and JSON.

## Confidence and bounding boxes

Confidence is taken from EasyOCR’s per-block confidence and `OCRResult.total_confidence` is the arithmetic mean of blocks that have a confidence value. This is not a calibrated probability. For the successful CPU combined-config examples, mean confidence was 0.6085 for English, 0.5847 for Hindi, 0.5784 for mixed text, and 0.2165 for blurred Hindi. The blank image had no blocks and `total_confidence=null`.

All returned boxes in the 32-record smoke output passed coordinate-bound checks against the image actually passed to OCR: 1600×280 for standard fixtures and 400×70 for the low-resolution fixture. EasyOCR’s existing invalid/out-of-bounds/non-finite box filter is covered by offline tests. Tesseract `xywh` boxes are now clipped and zero-area results discarded; Tesseract could not be run against a real image here.

The orchestrator applies EXIF transpose then RGB conversion before OCR. OCR itself does no resize or crop, so coordinates are in that EXIF-corrected original image frame. The synthetic benchmark does not contain EXIF orientation metadata; its dimensions are the actual in-memory dimensions supplied to the engine.

## Tesseract installation and verification on this Windows machine

No system package was installed or PATH changed during this task. To enable the fallback:

1. Install the 64-bit Windows Tesseract distribution (UB Mannheim build) and include the English and Hindi language data during setup. Confirm `eng.traineddata` and `hin.traineddata` are in the installation’s `tessdata` directory (normally `C:\Program Files\Tesseract-OCR\tessdata`).
2. Add the executable directory (normally `C:\Program Files\Tesseract-OCR`) to the Windows user `PATH`, then open a new PowerShell session.
3. Verify locally:

   ```powershell
   $env:Path = "C:\Program Files\Tesseract-OCR;$env:Path" # current session check
   Get-Command tesseract
   tesseract --version
   tesseract --list-langs
   ```

   The language list must contain both `eng` and `hin`. The backend’s existing `OCR_LANGUAGES=en,hi` configuration maps to those names and the service will now report unavailable if either is missing. After installation, rerun `python experiments/ocr_runtime_smoke.py --devices cpu,cuda` to perform actual separate `eng`, `hin`, and combined Tesseract inference.

## Failure-path tests and commands

Offline regression command:

```text
python -m pytest backend/tests/test_ocr_runtime.py backend/tests/test_phase2.py -q
23 passed, 2 skipped in 2.76s
```

The two skips are the opt-in real-model test cases; ordinary test runs remain cache/network independent. The suite covers cache-only reader initialization, missing package/weights, reader initialization failure, missing Tesseract executable/language pack, Tesseract inference failure, clipped/out-of-bounds Tesseract boxes, malformed/out-of-bounds/non-finite EasyOCR results, empty successful output with unchanged no-fallback behavior, and unavailable engines.

Opt-in real OCR command (English and Hindi parameters):

```text
$env:TRUTHLENS_REAL_OCR_TESTS='1'; python -m pytest backend/tests/test_ocr_runtime.py::test_real_easyocr_inference_is_cache_only -q
1 passed, 1 skipped in 9.83s
```

Hindi real inference passed with local weights. English-only skipped because `english_g2.pth` is absent; downloads are disabled. Pytest reported the existing pytest-asyncio fixture-loop deprecation and an EasyOCR/Pillow mode deprecation.

Full-backend tests were not run. The backend suite includes Phase 6/7 tests that read historical evaluation and prediction artifacts, which this task explicitly prohibits inspecting. No OCR runtime test loaded any training, calibration, locked-test, or benchmark data.

Smoke command:

```text
python experiments/ocr_runtime_smoke.py --devices cpu,cuda
```

It completed successfully and wrote `experiments/reports/ocr_runtime_smoke.json` with 32 inference records. The run emitted the expected EasyOCR missing-English-weight message with downloads disabled. Tesseract status was unavailable because the executable was not found.

## Files changed and remaining limits

- `backend/app/services/ocr/service.py`: prevent implicit EasyOCR downloads, verify Tesseract executable/language packs, and bound Tesseract coordinates.
- `backend/tests/test_ocr_runtime.py`: add offline readiness/failure/coordinate tests and opt-in real inference tests using generated text images.
- `experiments/ocr_runtime_smoke.py`: add reproducible cache-only synthetic benchmark and CER/WER reporting.
- `experiments/reports/ocr_runtime_smoke.json`: record current runtime readiness and all synthetic inference outputs.
- `docs/deployment_guide.md`: document cache-only EasyOCR setup.
- `docs/ocr_factcheck_audit.md`: note the subsequent runtime verification.

Tesseract remains a manual system dependency. Standalone `/api/v1/ocr` and `/api/v1/fact-check` behavior, language-model integration, claim extraction, fact-checking behavior, and detector/model training were outside this milestone and were not changed.
