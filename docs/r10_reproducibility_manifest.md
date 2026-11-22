# R10 Reproducibility Manifest

**Protocol ID:** `truthlens-r10-siglip-frequency-residual-v1`  
**Run status:** `EXECUTED` under the preregistered protocol.

## Baseline identity check

| Item | Expected | Read-only verification |
|---|---|---|
| Production head identity | `truthlens/siglip-cifake-frozen-head` | Fine-tuned production code assigns this ID. |
| Backbone | `google/siglip-base-patch16-224` | Configured in the production detector. |
| Backbone revision | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Exact local cache `refs/main` value; no model was loaded. |
| Head checkpoint | `models/baselines/siglip_frozen_head.pt` | SHA-256 `e39afc5536a0de44dd07fd33fa911bbd58d8799b85e680eb08d532222d974538`; matches the supplied baseline. |

## Run record

| Field | Value |
|---|---|
| Repository branch | `fix/truthlens-remediation` |
| Base commit before E11/R10 | `93b551c` |
| Working tree | E11 changes are uncommitted; no commit or push. |
| Python / libraries | Python 3.12.10; NumPy 2.2.6; SciPy L-BFGS-B; CPU. |
| Hardware | RTX 3050 Laptop GPU, 4 GB VRAM, per task brief; this candidate uses CPU optimization over precomputed vectors. |
| Allowed feature archive | `experiments/reports/phase9_feature_vectors.npz`; SHA-256 `a1b9d3530f9555ba4b67cc4a5d2b3793ebac7804bb150eb0a2856da4efb00ed9`. |
| Training/evaluation | `python research/r10_e11/run_experiment.py`; training 6,995; validation Tiny 5,958, Frontier 297, WildAIGC diagnostic 181. Frontier AUC delta +0.063123, paired group-bootstrap 95% CI [+0.017259, +0.118160]. |
| Candidate coefficients | `research/r10_e11/candidate_coefficients.json`; SHA-256 `d6c88fd365a4ca1c609ea135db69f7534c254512f1730a19be2a81b35d06c4ed`. |
| Runtime | 7.037 s optimizer; fixed CPU run. |
| Protected partition access | None intended. Combined role manifests and calibration/test/reserve artifacts are not in the allowlist. |

No model weights or dataset files have been modified. No large model or dataset download is permitted by this protocol.
