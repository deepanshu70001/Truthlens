# Explainability evaluation

## Implemented behavior

The application conditionally computes Grad-CAM for the configured SigLIP image classifier, targeting the predicted class. It maps patch tokens to a grid, writes a raw heatmap and overlay, records target class/model and provides retrieval URLs. The UI labels highlighted areas as model-attribution cues, not proof of manipulation. See `backend/app/services/xai/gradcam.py`.

## Checks and results

| Check | Result | What it establishes |
|---|---|---|
| API route/path confinement and artifact response | Existing API tests pass | Stored artifacts can be served through tested routes when present |
| Unit/schema behavior | Existing Phase 2/5 tests pass | Software contracts and failure state handling, not explanation faithfulness |
| Actual checkpoint heatmap and image-to-heatmap visual pair | Not verified in Phase 5/6 | No actual UI correctness conclusion |
| Localization/pointing game/IoU/deletion-insertion | Not evaluable | No masks or benchmark reference regions |
| Parameter randomization sanity checks | Not conducted | No attribution-method faithfulness result |
| Explanation stability across transforms | Not evaluated | No repeated labeled image protocol |
| Human interpretability study | Not conducted | No participants or consented protocol |

No heatmap from the accidental Phase 5 browser upload is used as a research case. No screenshot is treated as scientific validation.

## Reproducible future protocol

Freeze checkpoint and target class; verify the displayed image hash matches the API image hash; verify target label agrees with prediction record; regenerate uncached maps and compare dimensions/analysis ID. For sanity, progressively randomize model layers and quantify map similarity while checking predictions; use fixed transforms and report map stability with limitations. For quantitative localization, obtain independent masks with annotation protocol and report pointing-game and overlap only if the mask target corresponds to the attribution question. Run deletion/insertion with a justified baseline and disclose perturbation artifacts. For human evaluation, recruit consenting reviewers and assess clarity, relevance, misleadingness, and uncertainty; report participant count and instrument only after actual completion.

Grad-CAM is an attribution visualization, not evidence that a highlighted region is manipulated or that the model is right.
