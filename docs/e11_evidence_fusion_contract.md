# E11 Evidence Fusion Contract

**Policy ID:** `e11.conservative.v1`  
**Output schema:** `truthlens.fusion.v1`  
**Status:** Implemented; validation evidence and remaining limits are recorded in `docs/e11_r10_completion_report.md`.

## Purpose and boundary

E11 integrates existing E1–E10 outputs into one conservative report assessment. Execution status remains separate from the assessment category. A completed stage can have an empty outcome such as `no_text` or `no_results`; those outcomes are not failures and do not imply authenticity, generation, or manipulation.

The final assessment category is distinct from the primary detector's raw output. The pre-existing `overall_score` and `confidence` fields remain serialized for older API consumers, but both are legacy uncalibrated fusion margins. They are not probabilities. New consumers should use `category` and `uncertainty`; the frontend intentionally does not render the legacy numbers.

## Evidence eligibility

| Module or evidence | Required execution and validity conditions | E11 treatment |
|---|---|---|
| Primary authenticity detector | `model_evidence`, source `authenticity_detector`, stage `authenticity`, canonical completed/partial status, matching analysis and stage correlation, matching model identity, valid evidence schema, calibrated probability in `[0,1]`, `validated_temperature_scaling`, `validated_for_input_domain`, and direction consistent with the application label | The only current directional vote. It may be excluded if any condition fails. |
| OCR and language identification | Successful execution and valid output; `no_text` remains a successful empty result | Informational context only. |
| Claim extraction and third-party fact-checks | Successful or partial provider response, with publisher and claim attribution preserved; `no_results` is successful empty output | Informational context only. A publisher's rating is not a TruthLens truth label. |
| Classical image forensics | Completed signal with a valid status and measurement | Informational heuristic only; scores are not probabilities or manipulation votes. |
| YOLO/object detection | Real stage result with canonical status | Scene context only. Object presence or absence is not authenticity evidence. |
| EXIF/metadata | Valid extraction result | Informational context only. Missing metadata is not evidence of manipulation. |
| C2PA/content credentials | Actual verifier result with signature, content binding, and signer trust represented independently | Informational provenance only. Signature/content-binding results do not establish signer trust or event truth. |
| Web provenance | Actual provider result and attributed matches; `no_results` is successful empty output | Informational context only. Matches do not prove authenticity. |
| Grad-CAM/XAI | Actual generated attribution result | Informational model explanation only; not a manipulation signal. Local file paths are not included in the evidence payload. |
| Hashes and other evidence types | No directional validation contract is registered in E11 | Informational or excluded; cannot vote. |

## Status and outcome handling

Evidence must have a registered type, non-empty source/stage/evidence identities, valid strength/direction, a mapping-shaped data field, valid provenance, and a canonical execution status. The fusion stage also checks the matching `StageResult` when supplied by the orchestrator. Evidence associated with another analysis or a different stage correlation is excluded.

| Condition | Fusion behavior |
|---|---|
| `completed` with usable evidence | Evaluate against the explicit source and schema rules above. |
| `completed` with `no_text` or `no_results` | Retain as informational; do not vote. |
| `partial` | Keep only valid evidence from successful work; expose the partial stage in the execution summary. |
| `failed`, `unavailable`, `skipped`, `pending`, or `running` | Exclude from scoring and retain the reason/status for traceability. |
| `not_applicable` outcome | Exclude from scoring; it is not an execution error or a negative finding. |
| Invalid or malformed evidence | Exclude with a stable reason code; do not fail the report or vote. |
| Duplicate evidence | The same stable ID or canonical fingerprint is processed once. Repeated same-direction output from the same module/stage is excluded and cannot increase its score. Opposing valid classifier directions are surfaced as conflict and produce an inconclusive assessment under the existing policy. |

Missing, failed, skipped, unavailable, and empty evidence are represented separately in `missing_evidence` and `stage_execution_summary`. Absence is never converted into a negative vote.

## Assessment policy

The E1–E10 fusion implementation already defined strength multipliers `low=0.1`, `medium=0.3`, `high=0.6`, a determination threshold of `0.15`, and a conflict threshold of `0.10`. The source contained no empirical justification for these constants. E11 records that limitation and preserves the constants and existing score formulas. E11 adds eligibility gates and result explanations; it introduces no new voting source, weight, threshold, or calibration.

The current production integration produces at most one normal primary-detector vote. Third-party reviews, OCR, language, metadata, forensics, YOLO, C2PA, web provenance, and XAI remain outside the directional score. If no eligible vote exists, the category is `inconclusive`. Fused uncertainty is always marked `unquantified`; `assessment_confidence` is nullable and remains `null`.

## Versioned result and compatibility

The report's `fusion` object contains `schema_version`, `policy_version`, `category`, explicit uncertainty semantics, considered and excluded evidence references, supporting/contradictory/conflicting/informational findings, missing evidence, a stage execution summary, limitations, warnings, a UTC timestamp, and correlation ID. The existing `label`, `overall_score`, `confidence`, `reasoning`, `evidence_summary`, `conflicting`, `limitations`, and `runtime_ms` fields remain available.

The top-level report still includes both `assessment` and `fusion`; they reference the same assessment payload. The report builder supplies a versioned safe fallback when fusion itself fails. Older reports without E11 fields remain readable and are identified by the frontend as historical reports with unavailable detailed dispositions.

The explanation generator only summarizes completed or partial evidence and does not promote raw classifier output or failed module results into key findings. The frontend separates final evidence assessment from the primary detector output and shows informational, excluded, missing, conflicting, and incomplete-stage details without presenting the legacy score as a percentage.

## Non-goals

E11 does not modify the production checkpoint, detector identity, preprocessing, inference routing, calibration artifact, or production decision constants. R10 remains an isolated research track and cannot affect this policy or production inference.
