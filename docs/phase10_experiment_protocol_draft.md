# Phase 10 Draft: Low-Capacity Visual–Forensic Fusion

**Status:** Draft for review; no Phase 10 training is authorized by this document.
**Research-only:** Any candidate remains separate from production and the existing model identity/calibration contracts.

## Hypothesis

A small, independently normalized frequency branch may add stable information to SigLIP-base for some acquisition and generator conditions. The Phase 9 evidence is mixed: the grouped-training paired AUC change was +0.0007 (95% CI [-0.0019, +0.0075]); Tiny validation improved +0.0083 [0.0048, +0.0121]; Frontier validation improved +0.0231 [-0.0268, +0.0711]. The experiment tests whether a low-capacity fusion branch retains a validation benefit across seeds and matched perturbations. It does not assume that frequency statistics are universal indicators of AI generation.

## Architecture

- Visual branch: the unchanged production SigLIP-base patch16-224 encoder and frozen 769-parameter baseline head, using pinned revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.
- Forensic branch: the deterministic 34-feature Phase 9 extractor, standardized using training-fold statistics only; MLP `34 → 64 → 32` with GELU and dropout 0.2.
- Fusion: branch output maps to one residual logit, initialized to zero contribution; `z_final = z_production + α × z_forensic` with α initialized to zero and learned with regularization. The visual logit remains the anchor.
- The primary phase freezes the entire visual encoder and baseline head; only the forensic MLP and α are trained. A separate secondary ablation may unfreeze SigLIP's final two transformer blocks after the primary comparison is complete.
- Count and report all trainable parameters. Keep the existing production baseline checkpoint, calibration schema, identity contract, and inconclusive gating unchanged.

## Data and partitions

- Training: only the existing 6,995 approved training samples for this experiment (3,515 authentic and 3,480 AI). No calibration labels enter model fitting.
- Validation and early stopping: existing Tiny-GenImage validation (n=5,958) and Frontier 2026-10 validation (n=297), using the pre-registered 50/50 mean balanced-accuracy score. WildAIGC validation (n=181) is a contextual-label diagnostic only and is excluded from selection and early stopping.
- Calibration: reserved exclusively for calibration fitting after a model is frozen. It cannot be used to choose architecture, epoch, threshold, augmentation, or features.
- Locked test: historical evidence only under the existing governance. Do not open, recompute, or use for tuning. A later authorized evaluation requires a frozen candidate identity and protocol.
- Preserve source-group isolation in all training CV diagnostics. No new datasets are downloaded in this protocol draft.

## Preprocessing and augmentation

Use the Phase 7 canonical decode and normalization contracts. Apply the same random shorter-side resize (224–340), 224 center/random crop, and in-memory JPEG/WebP/PNG transcoding probabilities to both labels. Include blur with the existing Phase 7 probability. Add pre-registered resize probes at 0.5× and 2× and recompression at JPEG Q=90/75/50 for evaluation diagnostics; never modify source files. Fit feature standardization only on the current training fold.

## Training recipe

- Loss: binary cross-entropy with logits and class weights computed from the training fold. No label smoothing in the primary experiment.
- Optimizer: AdamW; forensic branch learning rate `1e-3`, weight decay `1e-2`. In the separate unfreezing ablation, final two SigLIP blocks use `1e-5`.
- Batch size: 16 with gradient accumulation 2; mixed precision only if numerically stable on available hardware.
- Epoch cap: 12; patience 3 on the pre-registered composite validation score. Select the checkpoint with maximum score; tie-break on Frontier validation ROC-AUC.
- Seeds: 42, 123, and 2026. Record Python, NumPy, framework seeds, deterministic flags, config hash, manifest hashes, checkpoint hashes, runtime, and peak VRAM for every run.
- Do not use validation labels for fitting the encoder or frequency standardizer. Validation is limited to diagnostics, early stopping, and the pre-specified comparison.

## Metrics and comparisons

Primary: balanced accuracy, ROC-AUC, FPR, FNR, and paired differences versus the exact frozen production baseline head under identical validation preprocessing and perturbations. A SigLIP embedding-only probe remains a secondary comparison. Report accuracy, precision, recall, average precision, confusion matrices, calibration/reliability diagnostics, and 95% source-group bootstrap intervals as secondary measures. Report each generator/source group with sample count and uncertainty; flag groups below 100 examples as exploratory. Keep WildAIGC separately labeled contextual evidence.

Required ablations: SigLIP-only; frequency-only; fusion with the branch randomly permuted within training fold; FFT-only; DCT-only; residual-only; edge/resampling-only; frozen visual branch versus Phase B partial unfreezing; matched augmentation on/off only as a clearly separated control.

## Acceptance and rejection criteria

The research hypothesis is supported only if all of the following hold:

1. Across the three seeds, the mean paired Frontier validation ROC-AUC improvement over the frozen production baseline is positive, and the paired source-group bootstrap 95% interval for the pooled improvement excludes zero.
2. Tiny validation balanced accuracy does not fall by more than 0.01 on average versus the frozen production baseline, and Tiny validation FPR does not increase by more than 0.02 at the fixed 0.5 diagnostic threshold.
3. The gain persists under the pre-registered JPEG and resize probes and is not reproduced by the permuted-frequency control.
4. No single small Frontier generator group is used alone to claim success; subgroup results include counts and uncertainty.

Reject or revise the hypothesis if the primary Frontier delta is non-positive, its interval includes zero, the effect disappears under matched transformations, or the permuted control performs similarly. Even a validation pass does not authorize production promotion or establish external validity. Require a later independently sourced, provenance-verified benchmark with held-out generator families and a separately frozen evaluation protocol before any deployment discussion.

## Hardware and runtime

Phase 9 feature extraction peaked at approximately 709 MiB allocated GPU memory on the local RTX 3050 Laptop GPU. The Phase 10 budget is 4 GiB VRAM; start with batch size 16 and stop on out-of-memory rather than silently changing the protocol. Estimate and record actual encoder, training, inference, and peak memory for every seed. Do not begin this training until Phase 9 findings and this protocol have been reviewed.
