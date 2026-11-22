# R10 Blockers and Limitations

**Status:** Preregistered; exact archive preflight not yet executed at time of draft.

## Preflight blockers

The R10 runner is permitted to read only `experiments/reports/phase9_feature_vectors.npz` and the frozen head checkpoint. It must stop before fitting if any split role, count, class distribution, group field, feature width, finite-value check, or checkpoint digest differs from the preregistered contract. The mixed Phase 2 manifests are not a fallback source and must not be opened to repair a mismatch.

## Known research limits

- The approved training set has a documented source/encoding confound: all authentic training images are JPEG while AI training images are PNG. Group isolation cannot remove this dataset-level confound.
- Frontier validation has only 297 samples and small generator groups; its confidence interval can be wide and subgroup estimates are exploratory.
- WildAIGC labels are contextual and are not objective provenance labels; they are diagnostic only.
- The Phase 9 archive contains precomputed features. This experiment tests a fixed residual fit over those features, not live end-to-end inference latency or robustness to unseen preprocessing.
- A validation improvement is not final held-out evaluation, calibration, OOD validation, causal attribution, or production approval.
- OCR, language, publisher ratings, YOLO, C2PA, and XAI are excluded from the feature set for the reasons in `docs/r10_hybrid_architecture.md`.

## Required later work

Any future use beyond this fixed R10 study requires new authorization and a fresh protocol. Locked final evaluation remains reserved for the later R13 milestone. Calibration and uncertainty work remain outside E11 and R10.
