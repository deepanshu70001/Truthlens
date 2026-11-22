# E3 PAPLUCA Model Verification

## Verification snapshot

- Date: 2026-11-22
- Repository: `D:/Projects/Truthlens`
- Branch: `fix/truthlens-remediation`
- HEAD: `1ba0d92159f4261e750b1069823f378c2c7bcdb7`
- Starting worktree: existing changes in `backend/tests/test_language_detector.py`, `docs/MODULE_STATUS_REGISTER.md`, `docs/PROJECT_ROADMAP.md`, `docs/e3_language_identification_report.md`, plus the prior E2 report modification and untracked `docs/e3_model_verification.md`. All were preserved and updated only where requested.
- Protected research/evaluation partitions were not accessed. No SigLIP, calibration, authenticity, fusion, or inference-routing files were changed. No commit or push was made.

## Official identity and pinned revision

- Repository ID: `papluca/xlm-roberta-base-language-detection`
- Current immutable revision resolved from the official Hugging Face model API: `9865598389ca9d95637462f743f683b51d75b87b`
- Model card: [official PAPLUCA model page](https://huggingface.co/papluca/xlm-roberta-base-language-detection)
- Task: text classification / sequence language identification.
- Model card architecture description: XLM-RoBERTa with a classification head over the pooled output. Local configuration resolves to `XLMRobertaForSequenceClassification` with `model_type=xlm-roberta`.
- License: MIT. Intended use is document/text language classification. The model card lists 20 supported language codes; its training/evaluation statements are not treated as TruthLens evaluation evidence.

## Cache provisioning and file integrity

Configured cache root: `D:\Projects\Truthlens\models\cache`.

Resolved snapshot: `D:\Projects\Truthlens\models\cache\models--papluca--xlm-roberta-base-language-detection\snapshots\9865598389ca9d95637462f743f683b51d75b87b`.

The download was pinned to the immutable revision and restricted to the six files needed by Transformers' tokenizer and PyTorch classifier. Duplicate `pytorch_model.bin`, TensorFlow weights, README, and other repository files were not downloaded. This Windows account cannot create cache symlinks (`WinError 1314`); the six snapshot files are regular files, and `refs/main` points to the pinned revision. The service subsequently resolved and loaded this cache offline.

Provisioning call (the initial cache operation fetched the files but exited while creating a symlink for `special_tokens_map.json`):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="papluca/xlm-roberta-base-language-detection",
    revision="9865598389ca9d95637462f743f683b51d75b87b",
    cache_dir=r"D:\Projects\Truthlens\models\cache",
    allow_patterns=[
        "config.json", "model.safetensors", "sentencepiece.bpe.model",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    ],
    local_files_only=False,
)
```

The existing downloaded 239-byte `blobs/2ea7ad0e45a9d1d1591782ba7e29a703d0758831` was copied to the matching pinned snapshot filename `special_tokens_map.json`; `refs/main` was set to the immutable commit hash. No existing file was overwritten.

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `config.json` | 1,417 | `c853c935230721240f2fc06acb72fb086cecea3cec99b62d843ac4678b7daba9` |
| `model.safetensors` | 1,112,264,584 | `a835d6e8ed50ef6b3c180db87446a83ee5ac437e981c932c8e1e239aacbe08b7` |
| `sentencepiece.bpe.model` | 5,069,051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `special_tokens_map.json` | 239 | `378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d` |
| `tokenizer.json` | 9,081,351 | `2a0d7366dd7780ea36cc42431dd74cd79289b783ab01acd33013fcc96865a8e9` |
| `tokenizer_config.json` | 502 | `0606f1a769dcea369f11010642c030cbd0a0a545ac21b11f2f86e9ae975dac3f` |

The `model.safetensors` and `sentencepiece.bpe.model` SHA-256 values match the official repository's LFS object IDs at the pinned revision. No unrelated model cache was inspected or changed.

## Offline loading and verified configuration

Installed runtime versions: Python 3.12.10, PyTorch 2.7.1+cu118, Transformers 4.57.6, Tokenizers 0.22.2, Hugging Face Hub 0.36.0, Safetensors 0.7.0.

For verification, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1` were set. Python socket connections were replaced with a function that raises, preventing network use by the verification process. `LanguageDetector.load(device='cpu')` succeeded in 5,664 ms. The detector uses `local_files_only=True` for both tokenizer and model loads.

| Property | Offline local result |
|---|---|
| Model class | `XLMRobertaForSequenceClassification` |
| Tokenizer class | `XLMRobertaTokenizerFast` |
| Model type / architecture | `xlm-roberta` / `XLMRobertaForSequenceClassification` |
| Resolved model revision | `9865598389ca9d95637462f743f683b51d75b87b` |
| Output classes | 20 |
| `config.max_position_embeddings` | 514 |
| `tokenizer.model_max_length` | 512 |
| Safetensors integrity | SHA-256 matches official LFS object ID; model loads successfully |

The checkpoint mapping was read from its local config and checked for 20 unique labels, exact set equality with the model card's supported language codes, and inverse consistency between `id2label` and `label2id`. No class order was supplied or reordered by the implementation.

`id2label` by class index:

| Index | Code | Index | Code |
|---:|:---:|---:|:---:|
| 0 | `ja` | 10 | `el` |
| 1 | `nl` | 11 | `ur` |
| 2 | `ar` | 12 | `bg` |
| 3 | `pl` | 13 | `en` |
| 4 | `de` | 14 | `fr` |
| 5 | `it` | 15 | `zh` |
| 6 | `pt` | 16 | `ru` |
| 7 | `tr` | 17 | `th` |
| 8 | `es` | 18 | `sw` |
| 9 | `hi` | 19 | `vi` |

Verified inverse `label2id`: `{"ar":2,"bg":12,"de":4,"el":10,"en":13,"es":8,"fr":14,"hi":9,"it":5,"ja":0,"nl":1,"pl":3,"pt":6,"ru":16,"sw":18,"th":17,"tr":7,"ur":11,"vi":19,"zh":15}`.

## Actual English and Hindi inference

The inputs below were independently written ordinary-language sentences and were run through the actual `LanguageDetector` service. They are runtime probes only; two examples do not establish accuracy.

**English input:** `The neighborhood library will host free workshops throughout the summer, and residents can register at the front desk.`

Result: `completed/success`, code `en`, name `English`, confidence `0.974530`, runtime **273 ms**. Top three: `en` 0.974530, `ur` 0.003368, `nl` 0.002381.

**Hindi input:** `कल सुबह शहर के छोटे बाज़ार में किसानों ने ताज़ी सब्ज़ियाँ बेचीं और ग्राहकों से बातचीत की।`

Result: `completed/success`, code `hi`, name `Hindi`, confidence `0.994774`, runtime **77 ms**. Top three: `hi` 0.994774, `ur` 0.001222, `zh` 0.000341.

Both results reported model ID `papluca/xlm-roberta-base-language-detection` and revision `9865598389ca9d95637462f743f683b51d75b87b`. Confidence is the maximum softmax score, not a calibrated probability. No accuracy, CER, or WER claim is made.

## Abstention and failure checks

| Case | Result |
|---|---|
| Empty text | `completed/no_text`; null code/confidence; model not loaded |
| Whitespace-only text | `completed/no_text`; null code/confidence; model not loaded |
| Very short `Hi!` | `completed/insufficient_input`; null code/confidence; model not loaded |
| Ambiguous text `hello hola bonjour hallo ciao salut hello` | Actual model returned `completed/undetermined`, null code, confidence 0.633436 (< 0.70); top three `hi` 0.633436, `sw` 0.198837, `ur` 0.045614 |
| Missing checkpoint | With `MODEL_CACHE_DIR` pointed to an empty temporary directory, result was `unavailable/unavailable`, language `unknown`, null code/confidence/revision, safe generic warning |
| Invalid mapping | Deterministic tests reject `None`, empty mappings, malformed indices, empty labels, and non-string labels |
| Low-confidence abstention | Mocked logits test confirms below-threshold output is `undetermined`, with `unknown` language and bounded raw score |

The missing-checkpoint result confirms no implicit English default. The actual ambiguous-text output confirms the current threshold abstains for this low-confidence input; it does not establish a general ambiguity-detection guarantee.

## Pipeline and safety audit

- The optional language stage runs on normalized OCR text after text normalization and follows the E1 `StageResult` status/outcome contract, with a configured 30-second timeout.
- API coverage confirms analysis and text/image stages continue when E3 is unavailable. Only a successful language result is passed to claim metadata and fact-check language filtering.
- The E3 result is not passed to authenticity scoring or evidence fusion. No SigLIP, calibration, authenticity, fusion, or routing files were modified.
- The frontend was not changed; no frontend checks were needed.

## Regression validation

The first focused run after adding the timeout test exited 1 with **123 passed, 1 failed, 2 skipped**: the test expected a detector-level failed outcome. Inspection confirmed timeout is represented by the stage (`status=failed`, `outcome=null`, `STAGE_TIMEOUT`), so the assertion was corrected to the stage contract. The final run below is the passing run.
Focused E3/E1/E2/E4-E5/E11 command:

```text
python -m pytest backend/tests/test_language_detector.py backend/tests/test_ocr_runtime.py backend/tests/test_phase2.py::test_ocr_preserves_text_and_structured_confidence backend/tests/test_phase2.py::test_ocr_empty_invalid_unavailable_and_fallback backend/tests/test_api.py::test_ocr_text_is_passed_to_text_normalization backend/tests/test_api.py::test_unavailable_language_model_does_not_block_image_or_text_stages backend/tests/test_api.py::test_upload_to_partial_report_and_image_retrieval backend/tests/test_stage_contract.py backend/tests/test_phase1.py backend/tests/test_phase3.py backend/tests/test_e4_e5_contract.py backend/tests/test_e11_fusion.py -q
```

Exit 0: **124 passed, 0 failed, 2 skipped** in 11.62 seconds. The skips are opt-in real OCR tests.

Established safe backend command:

```text
python -m pytest backend/tests -q --ignore=backend/tests/test_phase2_dataset_splits.py --ignore=backend/tests/test_phase3_evaluation.py --ignore=backend/tests/test_phase4_and_phase3_5.py --ignore=backend/tests/test_phase5_training.py --ignore=backend/tests/test_phase6_evaluation.py --ignore=backend/tests/test_phase7.py
```

Exit 0: **359 passed, 0 failed, 12 skipped, 6 warnings** in 25.17 seconds. Warnings were Python 3.12 `aiosqlite` datetime-adapter deprecations. The six protected files were not run.

Compile check: `python -m compileall -q backend/app/services/nlp/language_detector.py backend/app/services/orchestrator.py backend/app/services/reporting/builder.py backend/app/config.py backend/tests/test_language_detector.py` — exit 0. `git diff --check` — exit 0.

## E3 status

**COMPLETE AND VERIFIED** for the requested integration/runtime scope: immutable revision pinned in the configured local cache, offline tokenizer/classifier load, checkpoint-derived 20-class mapping verified, actual English and Hindi inference succeeded, and regression suites passed. Language confidence remains uncalibrated and this verification is not an accuracy evaluation.
