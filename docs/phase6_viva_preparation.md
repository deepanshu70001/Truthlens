# TruthLens panel/viva preparation

## A. Project explanation

**Problem:** A single image classifier cannot establish whether a visual claim is accurate, where an image came from, or what context is missing.

**Solution:** TruthLens is a local research prototype that combines a configured image-authenticity signal, classical forensic indicators, OCR, optional object detection and Grad-CAM, rule-based claim extraction, optional third-party evidence lookup, and heuristic context cues in one traceable report.

**Architecture:** React/TypeScript UI → FastAPI upload/API → SQLAlchemy record and local image storage → in-process orchestrator → independent analysis stages → evidence collection/fusion → structured report and artifact endpoints → UI. Optional dependencies/credentials produce explicit unavailable states.

**Technology:** Python 3.11, FastAPI, PyTorch/Transformers, Pillow/OpenCV, optional EasyOCR/Tesseract/Ultralytics/providers, SQLAlchemy async and SQLite default; React, TypeScript, Vite and Tailwind frontend.

**Honest status:** Engineering integration and software behavior have been tested. No approved labeled dataset exists, so detector accuracy, calibration, robustness, multilingual quality, fusion benefit, or human usability have not been established.

## B. Technical questions and defensible answers

1. **What is SigLIP2?** A vision-language encoder family. TruthLens uses a separate configured binary fake/real classifier checkpoint built on the SigLIP2 architecture; the encoder paper's results do not transfer as TruthLens detector accuracy.
2. **How does AI-image detection work here?** The loaded classifier returns fake/real softmax scores and code applies an operational uncertain band. The score is uncalibrated and can fail under distribution shift.
3. **Does it detect manipulation?** No separate manipulation class is implemented; manipulation score is null. Classical cues are indicators, not a reliable editing classifier.
4. **What do forensic indicators show?** Encoding/noise/edge/resampling anomalies that depend on image format and history. They are not probabilities or proof.
5. **What does OCR do?** Attempts text extraction through an available configured engine; default languages are English/Hindi. Engine availability is environment-dependent; no CER/WER benchmark was completed.
6. **What is YOLO used for?** Optional object detection/context. It does not verify factual claims, and weights/runtime were not benchmarked in Phase 6.
7. **What is Grad-CAM?** A class-conditioned attribution visualization. It highlights influential regions for the model; it is not a manipulation mask or proof.
8. **How does evidence fusion work?** Weighted rule aggregation with conflict handling and uncertainty. Phase 6 restricts directional voting to completed/partial model evidence. Current forensics and other contexts are informational; no predictive gain is measured.
9. **How are claims checked?** Rule-based candidate extraction is followed, when configured, by provider review lookup and traceable rating matches. A provider rating is not an independent investigation by TruthLens.
10. **What is provenance?** Optional web-detection evidence from a configured service. It does not prove the first source or original date. C2PA parsing is unavailable.
11. **What is virality estimation?** Hand-weighted content cues only. There is no trained spread predictor or observed outcome dataset.
12. **What does regional analysis say?** Currently regional numeric scores/ranks are unavailable. The system must not claim any state or community is more susceptible.
13. **Can it predict social harm?** No. It flags potential content categories using heuristic tokens; it has no observed harm outcome or validated severity model.
14. **What happens when services fail?** Stage status and limitations are reported; missing/failed stages should not count as evidence votes. This behavior is covered by Phase 6 fusion regression tests.
15. **Why distinguish authenticity from factual accuracy?** An authentic photograph can accompany a false caption, and an AI image can illustrate a true claim. They are separate questions and report domains.
16. **Why is uncertainty important?** Scores overlap and sources can be missing or conflict; a forced binary verdict would overstate evidence.
17. **What are the main security limits?** No authentication/ownership, retention deletion workflow, durable job queue, or rate limiter. Do not expose the local prototype publicly.
18. **How was it evaluated?** Backend software tests and controlled code scenarios. A dataset audit found no labeled, licensed, independent corpus; the detector evaluation therefore returned NOT_EVALUABLE.
19. **What is the main contribution?** An integrated, traceable research prototype and explicit evaluation/reporting protocol. Do not claim a novel detector or measured accuracy gain.
20. **What would validate it next?** Permitted source-labeled data; independent event/generator splits; fixed evaluation protocol; held-out metrics and confidence intervals; OCR/claim annotation; explanation tests; paired ablation; and consented usability study.

## C. 5–8 minute live demonstration script

**Preflight (before panel):** Use only an image with documented source, permission/license and expected label. Create a separate demo storage/database; clear browser file input; verify no prior selection. Do not use the repository's undocumented `test_data` images as truth-labeled examples. If no approved example is available, demonstrate upload using an explicitly generated synthetic graphic for workflow only and state it has no ground-truth detector label. Turn off optional external integrations unless panel consent and data flow are clear.

1. **0:00–0:45 — Scope:** Explain the difference between image authenticity, factual claim accuracy and social context. State caveat that this prototype is not a definitive detector.
2. **0:45–1:30 — Upload:** Select the documented sample deliberately; show filename, preview and analysis action. Point to local processing and any external integration disclosure.
3. **1:30–2:30 — Pipeline:** Show progress and explain that completed/unavailable status is different from model confidence. Highlight authenticity label and uncalibrated-score caveat.
4. **2:30–3:15 — Heatmap/forensics:** Show Grad-CAM only if generated; explain it attributes model influence, not manipulation. Show forensic indicators as encoding-dependent cues.
5. **3:15–4:15 — Text and claim:** If OCR text exists, show extracted text and claim eligibility. If none, show an honest no-text/unavailable state. Explain translation is absent.
6. **4:15–5:00 — Evidence:** Show claim-level provider evidence only if configured and available; distinguish no result from false. Trace the source/URL and review date.
7. **5:00–6:00 — Social indicators:** Show virality as heuristic cues, regional scores as DATA_UNAVAILABLE, and social-impact categories as potential cues only.
8. **6:00–7:00 — Report and limits:** Show module statuses, summary domains, and limitations. Close with what data/evaluation is still needed before scientific efficacy claims.

**Fallback:** If model load, OCR, provider or heatmap is unavailable, demonstrate the truthful status and explain why; never substitute an invented successful result.

## D. Likely research questions

- **What happens when model and forensic cues disagree?** Current forensics are informational and cannot outvote the model. A future validated directional policy requires paired labeled evidence and correlation-aware design.
- **Are model scores probabilities?** No, softmax scores are uncalibrated.
- **What dataset supports the results?** No approved model evaluation dataset is present; therefore efficacy metrics are not reported.
- **Does a regional score represent spread?** There is no score/rank currently; language/connectivity context would not mean susceptibility.
- **Did you conduct a user study?** No; a protocol is prepared, but there are no participants/results.
- **Can you submit the paper now?** It is an honest technical prototype draft. A performance/efficacy paper needs the missing dataset, experiments and verified author details.

## E. Limitations to disclose

No independent detector metrics/calibration, no suitable licensed labeled image data, duplicate test artifacts, no OCR CER/WER, no human reviewed claim/retrieval corpus, no Grad-CAM localization study, no empirical fusion gain, no observed virality/regional/harm data, no usability study, optional model/provider availability, no authenticated shared deployment, and no retention/deletion flow. Author and institution metadata require confirmation from team records.
