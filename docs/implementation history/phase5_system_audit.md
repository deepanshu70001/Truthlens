# Phase 5 system audit

Audit performed 2026-11-17 against the repository checkout. Status reflects code inspection and the test commands recorded in `phase5_e2e_test_report.md`; a module file by itself was not counted as integration.

| Component | Implemented | Tested | Integrated | Known issues | Required action |
|---|---|---|---|---|---|
| React/Vite upload and report UI | Yes | TypeScript/build only in this environment | Yes | Browser-level E2E not run; refresh restore is newly added | Run browser E2E on a normal local runtime |
| Upload API and validation | Yes | Unit/security tests; API fixture blocked here | Yes | no auth/rate limiter; retention has no deletion endpoint | Local/demo use only; add deployment perimeter controls |
| Preprocessing | Yes | Unit tests; API run blocked in this sandbox | Runs at upload and is now represented as the first persisted pipeline stage from a SystemEvent | Failure is nonfatal; output includes compact safe summary only | Verify event-backed status in full API E2E |
| Metadata | Yes | Unit tests | Yes | EXIF is partial and privacy-sensitive | Keep redaction and report limitations |
| Quality analysis | Yes | Unit tests | Yes | heuristic image quality signals | Keep signals descriptive |
| Authenticity detector | Yes, lazy SigLIP2 model | Synthetic-image CPU smoke inference completed; does not assess accuracy | Yes | model cache present (~743 MB); Transformers emitted BOS/EOS config warnings; scores uncalibrated | Validate model separately on licensed dataset |
| Classical forensics | Yes | Unit tests | Yes | indicators are not manipulation probabilities | Preserve explanations and caveats |
| OCR | Yes, EasyOCR/Tesseract fallback | Unit tests; end-to-end API run blocked | Yes | runtime engine and language packs depend on host; no OCR means downstream skip | Install configured OCR language packs |
| Translation | No translation provider is implemented | No | No | Text normalizer only cleans text; no translated field | Explicitly unavailable; add only with an approved provider/model |
| Object detection | Yes, Ultralytics YOLO | Unit tests/mock; actual weights not confirmed in `models/` | Yes | may require checkpoint download; can return unavailable | Pre-provision model in demo environment |
| VLM contextual analysis | Placeholder service | No meaningful model test | No | explicitly deferred in `services/vlm/service.py` | Not claimed as available |
| Explainability | Grad-CAM implemented | Unit tests exist; not run with actual model in this environment | Conditional on authenticity model | heatmaps depend on compatible model internals | Exercise with configured model and verify artifact retrieval |
| Text normalization/language | Yes | Unit tests | Yes | no translation | Language estimation is OCR/script-derived |
| Claim extraction/classification | Yes | Phase 3 tests | Yes | rule-based eligibility, not factual verdict | Keep eligibility separate from fact-check verdict |
| Fact-check retrieval | Google Fact Check API integration | Unit/mocked tests | Conditional on eligible claims and credential | unavailable without key/network; not a comprehensive fact-check | Configure key only for consented demo |
| Provenance/reverse image | Google Vision Web Detection | Unit/mocked tests | Yes | sends uploaded image to Google when configured; unavailable without credential | Disclose and obtain consent/configuration |
| C2PA/content credentials | Not implemented | No | Reports explicit unavailable status | No parser or validation | Keep unavailable |
| Hashing | SHA-256 and perceptual hash | Unit tests | Upload plus pipeline | perceptual hash is similarity hint only | None |
| Evidence/fusion/explanation | Yes | Phase 3/4 tests | Yes | fusion is not ground-truth truth score | Explain score semantics and limitations |
| Virality indicators | Phase 4 heuristic | Phase 4 tests | Yes | no historical model/data | Do not present as reach prediction |
| Regional context | Explicit unavailable scores for 36 units | Phase 4 tests | Yes | no valid regional dataset installed | Maintain null scores until data exists |
| Social impact cues | Exact-token category cues | Phase 4 tests | Yes | no outcome severity or audience model | Keep outputs as cues only |
| Unified report | Yes | builder tests | Yes | report assembly failure fallback currently retains status/stage summary, not every module result | Track as reliability limitation |
| Database/storage | SQLAlchemy async; SQLite default and PostgreSQL driver config | API tests blocked by local pytest loop startup | Yes | files/results have no retention cleanup/delete workflow; schema uses `create_all`, not migrations | Set retention and use migrations before deployment |
| Health/models API | Health checks and dependency/cache/model-load readiness route; `/models` remains a placeholder | API test blocked here | Yes | readiness checks do not run model inference | Document distinction; exercise endpoint under normal API test runtime |
| Docker/deployment | No Docker files and Docker CLI unavailable | Not tested | Native run paths only | No containerized deployment artifact | Follow native deployment guide; Docker remains optional future work |

## Integration classification

Working as connected stages: upload/preprocess, metadata, quality, authenticity, forensics, OCR, object detection, conditional XAI, text normalization, claim extraction, fact-check lookup, provenance lookup, hashing, evidence collection/fusion, explanation, virality, regional, social-impact, and report assembly. Optional dependencies and credentials intentionally yield unavailable/failed/skipped states.

Incomplete/unavailable: translation, VLM context inference, C2PA parsing, authenticated multi-user access, deletion/retention, durable job queue, actual regional/virality/social outcome models, registry-backed `/models`, and legally verified diverse demonstration corpus. The report is assembled once; there is no duplicate report system.

The existing `DEMO_MODE` flag did not generate synthetic sample data; it only changed the virality status label while returning the same heuristic calculation. That relabeling was removed. No synthetic analysis result is substituted when a model or external service fails. The application has no verified presentation-sample dataset; see the test report.
