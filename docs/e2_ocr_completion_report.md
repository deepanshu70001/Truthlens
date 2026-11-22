# E2 OCR Completion and Reliability Report

## Scope and baseline

E2 completes the existing integrated OCR path and its failure handling. Work was performed on branch `fix/truthlens-remediation`, starting at `6e188e0`. The worktree was clean at the start of E2. The ignored local `models/ocr/` directory and its weights were preserved. No commit or push was made.

Changes are limited to OCR service/configuration, the OCR orchestration timeout, OCR and API regression tests, and the E0 module status register. No production SigLIP checkpoint, calibration, identity contract, inference routing, fusion policy, or research artifact was modified. E3 language identification and E5 standalone OCR endpoint work were not started.

## Real-runtime verification pass

On 2026-11-22, verification began on branch `fix/truthlens-remediation` at HEAD `1ba0d92159f4261e750b1069823f378c2c7bcdb7`; `git status --short` was clean. The pass changed only this report, the module status register, and the roadmap. Existing E11/R10 state and ignored local assets were left untouched. No datasets or models were downloaded, and production SigLIP identity, weights, calibration, routing, and fusion policy were not changed.

## Implementation

- EasyOCR remains the cache-only primary engine (`download_enabled=False`). A Tesseract fallback is attempted when EasyOCR raises or completes without usable text; fallback output replaces the primary result rather than being merged with it.
- Empty readable output is a successful `completed` result with `has_text=false` and a user-safe `no_text_reason`. OCR inference failure is `failed`; absence of all usable engines is `unavailable`. Malformed region data can yield `partial` while preserving usable text.
- Image decoding, format, dimensions, OCR-local RGB conversion, region geometry, and confidence values are checked. Boxes are clipped to the input coordinate frame; invalid confidence does not discard otherwise usable text. Engine confidence remains uncalibrated.
- The orchestrator bounds the OCR wait using `OCR_TIMEOUT_SECONDS` (default 60 seconds), reports timeout as a failed E1 stage with `STAGE_TIMEOUT`, and keeps exception details out of public results. A timed-out worker thread cannot be forcibly stopped by Python and may continue until native inference returns.
- OCR text continues through the existing normalization pipeline. The existing OCR-derived script estimate remains a heuristic; no generic language-identification capability was added.

## Runtime evidence

The configured environment has EasyOCR 1.7.2, PyTesseract 0.3.13, and Torch 2.7.1+cu118 importable. The app's EasyOCR model-storage directory is `D:\Projects\Truthlens\models\ocr`, selected by `OCR_MODEL_DIR`. At the E3 inspection, it contained the detector file `craft_mlt_25k.pth` and the recognition file `devanagari.pth`; `english_g2.pth` and `latin_g2.pth` were absent there. EasyOCR 1.7.2's reader selection logic selects recognition file `devanagari.pth` for both `en,hi` and `hi`, while `craft_mlt_25k.pth` is the separate CRAFT text-detection model. The E2 smoke record did not log the resolved recognition pathname, so that exact historical file consumption could not be independently confirmed; the filename above is confirmed from the recorded language configuration, current configured storage, and installed EasyOCR 1.7.2 selection logic. The default user cache `C:\Users\DELL\.EasyOCR\model` also exists and contains `english_g2.pth`, but the app passes the explicit `OCR_MODEL_DIR` and does not select from that directory.

Cache-only EasyOCR inference was run on `backend/tests/fixtures/test_photo.jpg` (640 x 480):

| Configured languages | Result | Detected regions | Mean engine confidence |
|---|---|---:|---:|
| `en,hi` | completed, readable text | 7 | ~0.033327 |
| `hi` | completed, readable text | 8 | ~0.002894 |

Both real runs completed without warnings. The fixture has no ground-truth transcript, so these runs establish runtime readiness only, not transcription accuracy. Confidence values are not calibrated probabilities.

### Initial English Tesseract verification before Hindi pack installation (2026-11-22)

The Windows executable is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`, but the install directory is not in the process's default `PATH`. For verification, that directory was prepended to the test process `PATH`; pytesseract 0.3.13's configured command remained `tesseract`, which resolved to `C:\Program Files\Tesseract-OCR\tesseract.EXE`. Tesseract reported version `5.4.0.20240606`. At this initial check, installed language packs were `eng` and `osd`; English fallback was exercised with an `en` service configuration. The `en,hi` Tesseract configuration was unavailable before the Hindi pack was installed.

| Fixture identifier | Use | Actual Tesseract result |
|---|---|---|
| `synthetic-runtime-en-001` (generated in memory, 1300 x 160 RGB; Arial 54; runtime-only text) | Independent pytesseract inference and real-service fallback after a simulated EasyOCR exception or empty result | Independent output: `TRUTHLENS OCR RUNTIME CHECK 2026` (5 word regions, 227.5 ms). Service output in both fallback cases: `TRUTHLENS\nOCR\nRUNTIME\nCHECK\n2026` (5 regions). Exception fallback: `completed`, engine `PyTesseract`, 234 ms, per-region confidence `[0.91, 0.96, 0.96, 0.96, 0.96]`. Empty-output fallback: `completed`, engine `PyTesseract`, 181 ms. |
| `backend/tests/fixtures/test_photo.jpg` (640 x 480 JPEG) | Real Tesseract empty-text behavior after simulated EasyOCR empty result | Empty output, 0 regions, 271 ms, mean confidence `null`, status `completed`, engine `PyTesseract`, `has_text=false`, outcome `no_text`. |

On the synthetic runtime fixture, fallback text replaces the primary result; a separate one-region simulated EasyOCR result returned as `PRIMARY ONLY`, engine `EasyOCR`, with no Tesseract merge. Tesseract word confidence is emitted on a 0–1 scale after conversion from its 0–100 output. These values are engine estimates, not calibrated confidence probabilities. No accuracy evaluation was undertaken: the existing fixture generator's construction string was not adopted as an independent transcript for this pass, so no CER/WER metrics are reported. The synthetic fixture is an execution check, not an accuracy benchmark.

### Hindi Tesseract completion (2026-11-22)

Installed the official `hin.traineddata` from [`tesseract-ocr/tessdata_fast`](https://github.com/tesseract-ocr/tessdata_fast/blob/4e7c9ce934584e30654178261986ca1a03ffcfe8/hin.traineddata), pinned to upstream commit `4e7c9ce934584e30654178261986ca1a03ffcfe8` (last change to this file). The raw download URL was `https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4e7c9ce934584e30654178261986ca1a03ffcfe8/hin.traineddata`. Installed file: `C:\Program Files\Tesseract-OCR\tessdata\hin.traineddata`, 1,122,751 bytes, SHA-256 `4c73ffc59d497c186b19d1e90f5d721d678ea6b2e277b719bee4e2af12271825`. Existing `eng.traineddata` (4,113,088 bytes) and `osd.traineddata` (10,562,727 bytes) were left untouched. Their current SHA-256 values are `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` and `9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff`, respectively.

With `C:\Program Files\Tesseract-OCR` prepended to the process `PATH`, `tesseract.exe --version` reported `5.4.0.20240606`; `tesseract.exe --list-langs` reported `eng`, `hin`, and `osd` from `C:\Program Files\Tesseract-OCR\tessdata`.

The approved runtime fixture was the existing in-memory `hindi` case from `experiments/ocr_runtime_smoke.py` (1600 x 280, Nirmala font). It was used for inference only; no CER/WER was calculated.

| Check | Exact output and result |
|---|---|
| Independent pytesseract (`lang='hin'`, `image_to_string`) | `दिल्ली में 425 नए विद्यालय 2026 में खुले` followed by a newline; 8 non-empty word regions; 338.2 ms for the paired `image_to_string` and `image_to_data` calls. |
| TruthLens service (`OCRService(languages=['en','hi'])`, simulated EasyOCR empty result, real Tesseract fallback) | `दिल्ली\nमें\n125\nनए\nविद्यालय\n2026\nमें\nखुले`; 8 regions; `completed`, engine `PyTesseract`, `has_text=true`, 199 ms. Per-region confidence: `[0.67, 0.96, 0.94, 0.96, 0.96, 0.96, 0.96, 0.96]`. |
| Missing-`hin` behavior | A temporary tessdata folder containing only copies of `eng` and `osd` produced available languages `['eng', 'osd']`. With EasyOCR unavailable and `en,hi` requested, the service returned `unavailable`, `tesseract_loaded=false`, and warning `Tesseract fallback unavailable (RuntimeError: Tesseract language packs unavailable: hin).` The installed Hindi file was not removed or altered. |

The independent `hin` invocation and the combined `en,hi` service invocation returned different numeral readings (`425` and `125`). A second `en,hi` service run with a simulated EasyOCR inference exception returned the same eight-region Tesseract text, `completed`, engine `PyTesseract`, in 845 ms; its warnings identify the primary exception and fallback. Both fallback triggers are therefore exercised with the bilingual configuration. Outputs are recorded as observed, without treating either as ground truth or calculating accuracy. The missing-pack path explicitly identifies `hin`; the service does not claim bilingual Tesseract availability when that file is absent.

Fallback is invoked when EasyOCR throws during inference or produces no usable text blocks. Valid EasyOCR text returns immediately without fallback. Tesseract no-text after an EasyOCR no-text result stays `completed` with outcome `no_text`; inference errors are `failed` if no successful primary result exists; missing executable or required language packs make Tesseract `unavailable`. Malformed EasyOCR regions are discarded with warnings and can yield `partial` when usable text remains. The stage-contract tests confirm execution status and outcome are separate fields: readable completion is `completed`/`success`, an empty result is `completed`/`no_text`, and `partial`, `failed`, and `unavailable` remain execution statuses. The stage-timeout test verifies `failed` with `STAGE_TIMEOUT`; the thread-based timeout does not forcibly terminate native OCR work.

## Validation

- Earlier focused E2/OCR/E1 regression command: `python -m pytest backend/tests/test_ocr_runtime.py backend/tests/test_phase2.py::test_ocr_preserves_text_and_structured_confidence backend/tests/test_phase2.py::test_ocr_empty_invalid_unavailable_and_fallback backend/tests/test_api.py::test_ocr_text_is_passed_to_text_normalization backend/tests/test_api.py::test_upload_to_partial_report_and_image_retrieval backend/tests/test_stage_contract.py backend/tests/test_phase1.py -q` - **58 passed, 2 skipped** (historical result).
- Initial English Tesseract verification focused command (with `C:\Program Files\Tesseract-OCR` prepended to `PATH`): `python -m pytest backend/tests/test_ocr_runtime.py backend/tests/test_phase2.py::test_ocr_preserves_text_and_structured_confidence backend/tests/test_phase2.py::test_ocr_empty_invalid_unavailable_and_fallback backend/tests/test_api.py::test_ocr_text_is_passed_to_text_normalization backend/tests/test_api.py::test_upload_to_partial_report_and_image_retrieval backend/tests/test_stage_contract.py -q` - **45 passed, 0 failed, 2 skipped** in 10.61 seconds, before `hin` installation.
- Post-install focused OCR/API/stage command (with `C:\Program Files\Tesseract-OCR` prepended to `PATH`): `python -m pytest backend/tests/test_ocr_runtime.py backend/tests/test_phase2.py::test_ocr_preserves_text_and_structured_confidence backend/tests/test_phase2.py::test_ocr_empty_invalid_unavailable_and_fallback backend/tests/test_api.py::test_ocr_text_is_passed_to_text_normalization backend/tests/test_api.py::test_upload_to_partial_report_and_image_retrieval backend/tests/test_stage_contract.py -q` - **45 passed, 0 failed, 2 skipped** in 6.17 seconds. The two skips are opt-in cached real EasyOCR English and Hindi tests (`TRUTHLENS_REAL_OCR_TESTS=1` was not set); no EasyOCR weights were downloaded.
- The focused suite covers invalid/corrupt images, malformed EasyOCR regions, malformed/missing Tesseract confidence and box handling, unavailable executable/language packs, simulated inference errors, fallback/no-text behavior, and the orchestration timeout contract. Its tests verify Tesseract failure redaction and stage status/outcome mapping. The real runtime checks separately exercised actual English inference and both fallback triggers.
- Safe backend suite: `python -m pytest backend/tests -q --ignore=backend/tests/test_phase2_dataset_splits.py --ignore=backend/tests/test_phase3_evaluation.py --ignore=backend/tests/test_phase4_and_phase3_5.py --ignore=backend/tests/test_phase5_training.py --ignore=backend/tests/test_phase6_evaluation.py --ignore=backend/tests/test_phase7.py` - **262 passed, 12 skipped, 9 warnings**. The skips include opt-in OCR/runtime cases and existing optional tests. Warnings were the pytest-asyncio fixture-loop-scope deprecation and Pillow `mode` deprecations emitted through EasyOCR.
- Protected evaluation/research test files were intentionally excluded: the E2 repository constraints prohibit access to locked-test and calibration research artifacts. Therefore the entire backend suite was not run.
- `python -m compileall -q backend/app/services/ocr/service.py backend/app/services/orchestrator.py backend/app/config.py` — passed.
- `git diff --check` — passed (no whitespace errors).

## Verification boundary and remaining work

The English and Hindi Tesseract runtimes, the combined `en,hi` service fallback, and explicit unavailable behavior when `hin` is absent are verified. The executable directory still must be supplied on `PATH`; the repository does not configure a machine-specific installation path. No CER/WER accuracy claim is made; the fixture generator's construction string was not used as an independent reference. A process-isolated worker would be needed for hard cancellation of native OCR work. E5 owns the standalone `/ocr` endpoint; E3 owns dedicated language identification. E2's integrated OCR and bilingual Tesseract runtime scope is complete and verified; the standalone endpoint and OCR accuracy validation remain outside this verification.
