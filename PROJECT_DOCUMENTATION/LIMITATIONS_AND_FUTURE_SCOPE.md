# TruthLens Limitations and Future Scope

## 1. Purpose

This document separates constraints demonstrated by the current implementation and evidence from possible future improvements. Items labeled future are not claims about existing functionality.

## 2. Research and data limitations

- The approved local authenticity evaluation is `NOT_EVALUABLE` with zero eligible examples.
- Existing test images lack documented source rights, labels, provenance and independent split metadata; duplicate copies and derivatives further limit use.
- The upstream CIFAKE subset metrics are not reproduced from a reviewed manifest in this checkout and do not demonstrate current real-world generalization.
- No independent evaluation establishes accuracy, threshold quality, calibration, robustness, fairness, or benefit from fusion.
- No ground-truth OCR corpus or measured CER/WER exists; no gold claims set or reviewed fact-check retrieval relevance set exists.
- No Grad-CAM localization study or human interpretation study exists.
- Virality/content spread/regional spread/social impact lack validated observed outcomes. Current virality and topic outputs are heuristics/demo cues; regional prediction is unavailable.
- No user study or human-subject results support usability, trust or decision-improvement claims.

## 3. Technical and operational limitations

- Model/OCR/YOLO/XAI availability depends on optional packages, weights, model cache, compatible internals and hardware.
- The SigLIP head's scores are raw and uncalibrated; distribution shift can produce wrong tags. The model has no separate manipulation class.
- Classical forensics are encoding-sensitive indicators, not proof of editing.
- OCR and rule-based claim extraction can miss or misread content, especially outside supported language/script and image conditions.
- Fact-check and provenance lookups depend on external connectivity/credentials; results may be stale, irrelevant or unavailable.
- The analysis task executes in-process and can be interrupted by server restart; no durable queue/retry orchestration exists.
- SQLite is convenient for local use; scalable concurrency and production operations have not been established.
- No migration scripts were found; table creation uses `create_all` according to deployment docs.
- No authentication or authorization, account ownership, retention/deletion workflow, or configured public deployment hardening is present.
- Uploaded originals, derivatives and reports persist in local storage. Optional providers can receive image bytes or OCR-derived claim text.
- No Dockerfile/Compose setup is documented; no representative system load, resilience or resource study exists.

## 4. Scientific and ethical limitations

- A real image may carry a false caption; a generated image may accompany a true claim. Authenticity and claim truth must be reported separately.
- Exact-token harm-topic cues do not detect actual harm or reliably infer intent/context.
- Region or language cues must not be used to stereotype communities or claim regional susceptibility.
- False accusations and false reassurance can affect people. The app should remain decision support and should not be used for high-impact moderation, legal or safety decisions without independent safeguards and validation.
- Optional external API calls create disclosure, consent, licensing and provider-terms considerations.
- No fairness/bias-free or privacy-compliance certification is claimed.

## 5. Future scope: data and predictive models

### Authenticity and manipulation

- Acquire properly licensed, diverse real/AI images and separately labeled edited/manipulated images.
- Record capture/generator/source/event metadata and group related images before splitting.
- Reproduce the exact checkpoint result, then run independent generator/source holdouts and transformations.
- Select thresholds/calibration only on validation data and report per-class metrics and abstention coverage.

### OCR, claims and external evidence

- Build licensed multilingual OCR benchmarks and report CER/WER by script and image quality.
- Annotate claim spans/types and evaluate extraction separately from fact-check retrieval.
- Human-review retrieval relevance, matched claim identity and rating attribution; measure provider unavailability.

### Virality, content spread and regional spread

- Collect ethically and legally permissible, time-stamped propagation observations with defined platform, event, region and outcome units.
- Define content-spread and virality labels from actual observations (for example, time-bounded reach/rank only when those data are available and permitted).
- Build a regional dataset with defensible geographic labels, missingness documentation and privacy safeguards.
- Train and validate predictive models only after dataset definition; use temporal/source/event holdouts and compare against simple baselines.
- Until then, retain explicit `DATA_UNAVAILABLE` or heuristic labels and do not display fabricated predictions or scores as observed spread.

### Explainability, fusion and user study

- Test Grad-CAM faithfulness with parameter randomization, transformations and region masks.
- Evaluate evidence-fusion ablations on paired labeled records; ensure missing/unavailable evidence cannot vote.
- Conduct consented usability experiments on whether the report helps users distinguish image authenticity, claim truth and source context without increasing overconfidence.

## 6. Future scope: deployment and privacy

- Add authentication, authorization and analysis ownership if shared use is intended.
- Add explicit image deletion and retention controls, storage quotas and operator documentation.
- Move long-running tasks to a durable queue with idempotent retries and recovery.
- Add schema migrations, deployment/container definitions, observability and backup/restore procedures.
- Configure rate limits at application/reverse-proxy level; review MIME/decompression/resource abuse.
- Use TLS and managed secrets for any remote deployment; review provider data flow and obtain consent.
- Conduct systematic security, load, fault-injection and cross-browser/UI accessibility testing.
- Do not expose the unauthenticated local prototype directly to the public internet.

## 7. Priority roadmap

| Priority | Work item | Exit evidence |
|---|---|---|
| 1 | Confirm authors, guide, institution and demonstration rights | Verified project metadata and source/permission records |
| 2 | Curate authenticity corpus with labels and grouped splits | Reviewed manifest with licenses, provenance, label guide and adjudication |
| 3 | Reproduce model and evaluate independent test split | Frozen protocol, checkpoint digest, metrics, uncertainty and failure analysis |
| 4 | Evaluate OCR, claims, fact-check retrieval and XAI separately | Annotated gold sets and task-specific metrics/reviewer checks |
| 5 | Validate or remove unsupported social indicators | Real outcome data with temporal/geographic protocol, or explicit unavailable state |
| 6 | Harden multi-user deployment and data lifecycle | Auth/ownership, deletion/retention, durable jobs, migrations, deployment and security tests |
| 7 | Run controlled usability research | Ethics/consent materials, protocol, participant results and limitations |

## 8. Claims safe to make today

- TruthLens integrates a local image-upload workflow with multiple analysis stages and a structured, status-aware report.
- It includes software components for authenticity inference, image heuristics, OCR, rule-based claim extraction, optional provider lookups and heuristic context indicators.
- Backend tests and frontend build results have been recorded in repository reports.
- Current approved authenticity evaluation has no eligible examples, so detector effectiveness is not established.
- Spread models and datasets are future scope; current indicators do not establish actual propagation or regional impact.

Avoid claims that the model is accurate in the wild, calibrated, state-of-the-art, unbiased, detects all manipulation, verifies truth, predicts viral spread, improves decisions, or is ready for public deployment.
