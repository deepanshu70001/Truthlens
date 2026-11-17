# Evidence-fusion ablation study

## Scope

The requested A–E predictive ablation requires labeled examples and measured outcomes. None are available. Instead, `experiments/fusion_controlled_scenarios.py` runs six deterministic constructed scenarios through the actual fusion function. It is a code-path experiment and **not** an accuracy/ablation benchmark.

## Measured rule-path outputs

The actual artifact is `experiments/results/fusion_controlled_scenarios.json`. Model-only synthetic-direction evidence yields `likely_ai_generated`. Adding completed forensic informational evidence, OCR informational evidence, and provenance informational evidence leaves that output unchanged. A failed model item with an opposing direction is excluded. Forensic-only evidence yields `inconclusive` because the engine only allows model evidence to vote directionally; the current forensic builder emits informational directions.

## Finding and implementation correction

Code inspection found that the fusion loop previously counted the direction of evidence items regardless of `status`. That could allow a failed/unavailable/skipped directional item to vote. Phase 6 now excludes any evidence whose status is not `completed` or `partial`; a regression test covers unavailable, failed and skipped evidence. Controlled checks also confirm that current forensic/OCR/provenance item types are informational and do not alter authenticity direction, while duplicate items are suppressed.

## Limits and next experiment

These scenario outputs verify policy behavior, not whether the model-only or fused verdict is correct. There is no labeled paired set for A–E metrics, no correlation study of forensic cues, and no empirical weight justification. The current architecture does not actually add directional value from forensics/OCR/provenance to authenticity classification; those channels are reported separately as contextual evidence. A future predictive ablation needs the same independent group-split examples, human-reviewed authenticity labels, frozen model thresholds, explicit missing-stage handling, and paired group-bootstrap intervals. Do not imply a measured gain.
