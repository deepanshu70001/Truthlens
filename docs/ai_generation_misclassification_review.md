# AI-Generated Image Misclassification Review

## Finding

The reported source image and its verified label/provenance were not available in the approved repository evaluation manifest. The checked-in test images have no reviewed labels or rights records, and the project's earlier dataset audit found duplicates. They were not used to estimate accuracy or as a regression target. The specific visual/domain feature that caused the reported image's false negative therefore cannot be established from the available evidence.

The code path did contain a concrete reliability defect that could present such a failure too confidently:

1. `AuthenticityDetector` applied a two-class softmax and set `confidence = max(fake_score, real_score)`.
2. That uncalibrated value selected high evidence strength, so a single classifier could dominate the final authenticity assessment.
3. The UI formatted confidence and raw class scores as percentages. A value such as 0.999 could therefore read like 99.9% real-world certainty despite no calibration or target-domain accuracy evidence.

This explains how the software could overstate the model's output; it does not prove why the checkpoint assigned the particular uploaded image a high real-class score. Domain shift to an unseen generator is a plausible model-side explanation, but is not demonstrated for that image.

## Checkpoint and input path audit

- The locally cached checkpoint configuration identifies class index `0` as `Fake` and `1` as `Real`; the runtime maps by configured class names, not an assumed index order. The checkpoint class mapping is not reversed in this cache.
- Its processor configuration specifies 224×224 resize, RGB-oriented image processing, rescaling by 1/255, and normalization with mean/std 0.5. Inference uses the checkpoint processor and now converts inputs to RGB explicitly before calling it.
- The detector does not expose a separate manipulation classifier. `manipulation_score` remains unavailable.
- The model metadata and local files do not establish a TruthLens-verified dataset composition, class balance, labeling quality, or cross-generator test performance. A model card or checkpoint name is not independent validation.
- The orchestration path stores detector output, converts it to evidence, fuses it, and assembles the report. The bug was not a fallback setting the result to authentic: the direct score and downstream weighting/display semantics permitted the overconfident interpretation.

## Changes made

- Added explicit raw-logit, raw-softmax, model-argmax, calibration-status, domain-validation-status, and calibrated-probability fields.
- With no accepted calibration artifact, the detector now records the raw argmax separately and sets its application classification to `inconclusive`. Raw softmax scores do not become a confidence field or directional fusion vote.
- The fusion engine ignores uncalibrated directional model votes. Failed, unavailable, and skipped stages remain non-voting. Contradictory calibrated model outputs produce an inconclusive assessment.
- The report only sets a decision band when a calibrated probability and application-level calibrated classification are present.
- The UI renders raw outputs as decimal scores with an adjacent warning that they are not probabilities or accuracy estimates. It does not call the raw score a confidence percentage.
- Added opt-in binary temperature scaling, validation-only threshold selection under class-wise error caps, and NLL/Brier/ECE diagnostics. Runtime loading rejects artifacts without a reviewed-validation flag, sufficient examples in both classes, manifest fingerprint, diagnostics, and exact checkpoint revision.
- Added deterministic code-path tests for reversed class indices, explicit RGB conversion, raw-score semantics, uncalibrated abstention, calibration artifacts, temperature fitting, and conflicting evidence.

## Evaluation and limitations

No independently labeled, licensed, generator-diverse dataset was available to run before/after metrics. No accuracy, precision, recall, F1, false-negative rate, ROC-AUC, ECE, or Brier score for the production detector is claimed. Synthetic fixtures verify software behavior only. Calibration is not active by default because no reviewed artifact exists. Temperature scaling can adjust confidence calibration; it cannot repair a detector that learned dataset-specific artifacts or guarantee generalization to diffusion systems, new image generators, portraits, landscapes, objects, event-style images, compression, crops, or reuploads.

The exact reported image still needs to be supplied as a regression example with confirmed AI-generation provenance and permission to retain/use it. It must be held out from calibration and used only in an independent evaluation split. Until those data are available and tested, the underlying model false-negative rate remains unknown and the image-level failure must be considered unresolved.

## Reproduction

Run software checks from the repository root:

```powershell
python -m pytest backend/tests/test_phase6.py backend/tests/test_authenticity_calibration.py -q
python -m pytest backend/tests -q
python -m compileall -q experiments backend/app
cd frontend
& 'C:\Program Files\WindowsApps\OpenAI.Codex_26.1111.2228.0_x64__2p2nqsd0c76g0\app\resources\cua_node\bin\node.exe' node_modules\typescript\bin\tsc --noEmit -p tsconfig.app.json
```

For empirical calibration/evaluation, first create a reviewed manifest using the schema in `experiments/manifests/authenticity.example.csv`, with independent train, validation, and test splits and documented source/rights/group/generator fields. Then follow `experiments/README.md`. Do not use validation-fit metrics as held-out performance results.
