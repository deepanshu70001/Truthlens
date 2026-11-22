# TruthLens Phase 9: Research Findings

## Scope and governance

Phase 9 used the approved training set (6,995 images) and validation partitions only: Tiny-GenImage n=5,958, Frontier Benchmark 2026-10 n=297, and WildAIGC-Bench n=181. Calibration roles were not used for fitting probes. Locked test images and prediction files were not opened or analyzed. No dataset or production model was modified. The baseline and Phase 7 S1, S2, R1, R2, and A1 checkpoint/head digests were verified; the frozen-head production identity remains `truthlens/siglip-cifake-frozen-head`. WildAIGC retains its contextual-label caveat.

## Evidence summary

| Representation | Grouped training CV ROC-AUC (95% CI) | Tiny validation ROC-AUC | Frontier validation ROC-AUC (95% CI) |
|---|---:|---:|---:|
| SigLIP-base frozen embeddings | 0.948 [0.941, 0.956] | 0.899 | 0.604 [0.474, 0.738] |
| ResNet-50 ImageNet V2 | 0.659 [0.634, 0.685] | 0.636 | 0.453 |
| Frequency features | 0.895 [0.830, 0.919] | 0.762 | 0.623 [0.480, 0.754] |
| SigLIP plus frequency | 0.949 [0.943, 0.960] | 0.907 | 0.627 |
| ResNet-50 plus frequency | 0.760 [0.736, 0.780] | 0.723 | 0.344 |
| Production frozen SigLIP head | — | 0.602 | 0.750 [0.635, 0.847] |

The probes are diagnostics fit on approved training features and labels, not production models. The small Frontier partition produces wide uncertainty, and generator-level Frontier validation groups contain only 4 or 8 AI images each.

## Answers to the Phase 9 questions

1. **Is poor Frontier performance consistent with a representation gap?** The Phase 9 validation evidence does not support a simple absence of useful pretrained visual signal. The exact frozen production SigLIP head reaches 0.750 AUC on Frontier validation [0.635, 0.847], although its FPR at 0.5 is 0.404. A logistic probe trained only on Tiny data reaches 0.604 [0.474, 0.738], while the partially fine-tuned S2 checkpoint reaches 0.477 [0.343, 0.628]. This suggests that training and decision-boundary changes can reduce Frontier ranking quality; it does not prove why. The benchmark changes source, prompt/style mix, resolution, compression, and label context together, and n=297 leaves broad uncertainty.

2. **Do unseen generators cluster differently from seen generators?** The SigLIP centroid distances suggest that Frontier AI validation groups are farther from the approved-training AI centroid (8.26–13.34) than Tiny validation generator groups (2.57–2.99). Frontier category counts are only 4–8, so this is a hypothesis-generating pattern, not a reliable per-generator conclusion. PCA plots are not used as quantitative proof.

3. **Do frequency features add information beyond SigLIP?** The result is mixed. On grouped training predictions, adding all 34 features changes AUC by +0.0007, with paired 95% group-bootstrap interval [-0.0019, +0.0075]. On Tiny validation the paired AUC delta is +0.0083 [0.0048, 0.0121]. On Frontier validation it is +0.0231 [-0.0268, +0.0711]. The small Frontier sample does not establish a stable gain beyond SigLIP. A bounded hybrid experiment is reasonable, but the current evidence is not enough to select or promote one.

4. **Are forensic measurements stable under JPEG and resizing?** Most features retain high rank correlation under the tested JPEG qualities (mean featurewise correlation 0.963–0.970 on 400 training images). Stability is weaker for downscaling to 50% and applying the canonical resize (0.851); some individual high-frequency/residual statistics change substantially. DCT AC sparsity is especially unstable. These are limited transformation probes, not evidence of invariance to every platform pipeline.

5. **Does the existing data support a hybrid architecture experiment?** It supports a pre-registered, validation-diagnostic experiment. Frequency-only grouped CV is strong on the training distribution, and fusion improves Tiny validation modestly. The Frontier delta is positive but uncertain, while ResNet-plus-frequency lowers Frontier AUC. Any next experiment should use a separate low-capacity branch, grouped validation comparisons, and no locked-test iteration.

6. **What is the most defensible next model hypothesis?** Preserve the frozen production SigLIP encoder and head as the visual logit, then add a low-capacity, separately normalized frequency branch as a zero-initialized residual logit. This tests complementary evidence while limiting disruption to the baseline's Frontier ranking. Keep encoder unfreezing out of the primary run; consider it only as a separately preregistered ablation. See the [Phase 10 draft](phase10_experiment_protocol_draft.md).

7. **What data is needed for stronger external validity?** Obtain rights-cleared, provenance-verified real and generated images across multiple current generator families, model versions, prompt styles, and languages. Include matched source/acquisition pipelines and repeated JPEG, social-media resizing, screenshots, and post-processing on both classes. Provide hundreds to thousands of independent prompts/items per generator and an independently held-out generator family. WildAIGC's contextual labels should remain separately analyzed or be supplemented with verified provenance labels.

## Limitations

The training labels are strongly associated with acquisition metadata: authentic is 100% JPEG and AI is 100% PNG in the approved manifest. The probe uses pixel embeddings/features only, but those pixels carry source and processing history. Group isolation controls source groups in cross-validation; it cannot remove every dataset-level confound. Frontier validation is small and heterogeneous. UMAP was unavailable. PCA geometry is descriptive. No causal claim is made about a frequency cue, generator family, or representation failure.

Full details are in the [generator analysis](phase9_generator_analysis.md), [feature-space report](phase9_feature_space_report.md), [forensic report](phase9_forensic_feature_report.md), and machine-readable artifacts under `experiments/reports/phase9_*`.

## Runtime and resource use

The first full end-to-end analysis took 658.5 seconds wall time. Feature extraction accounted for about 499.8 seconds: SigLIP 190.8 seconds, ResNet-50 129.8 seconds, and frequency features 179.2 seconds. The final cached analysis refresh took 127.8 seconds, including 55.1 seconds for grouped CV and univariate probes and 24.9 seconds for validation diagnostics. Measured encoder throughput was about 70.4 SigLIP samples/second and 103.5 ResNet samples/second. Exact validation inference over all three validation roles took 61–78 seconds per Phase 7 candidate. Peak allocated GPU memory was 708.9 MiB. These timings describe this RTX 3050 Laptop GPU and local storage; they are not production service latency estimates.

## Validation run record

Final validation completed: focused Phase 9 tests **3 passed, 0 failed, 0 skipped**; full backend suite **267 passed, 0 failed, 10 skipped**; Python compilation passed; documentation links passed (148 links checked); and `git diff --check` passed. The backend suite emitted 13 existing library warnings (scikit-learn single-class metrics and Pillow mode deprecations).
