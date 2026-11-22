# R10 Hybrid Experiment Report

**Status:** `EXECUTED`; the preregistered primary hypothesis met its fixed acceptance checks.  
**Protocol:** `truthlens-r10-siglip-frequency-residual-v1` (`docs/r10_research_protocol.md`).

The single fixed 35-parameter frequency residual was fit on 6,995 approved training rows. No feature selection, validation tuning, or image inference was performed. Exact input hashes and partition counts matched the preregistration. The run used only the Phase 9 feature archive and frozen production linear head; protected partitions and combined split manifests were not accessed.

Command: `python research/r10_e11/run_experiment.py` (exit 0; CPU, Python 3.12.10, NumPy 2.2.6; optimizer runtime 7.037 s). Archive SHA-256: `a1b9d3530f9555ba4b67cc4a5d2b3793ebac7804bb150eb0a2856da4efb00ed9`. Training: 6,995 (3,515 authentic, 3,480 AI-generated); Tiny validation: 5,958; Frontier validation: 297 / 75 source groups; WildAIGC diagnostic: 181.

Frontier ROC-AUC increased from 0.7497 to 0.8128 (delta +0.0631); paired source-group bootstrap (2,000 replicates, seed 2026) mean delta +0.0626, 95% CI [+0.0173, +0.1182]. Fixed checks passed: positive delta, interval excludes zero, Tiny balanced accuracy drop no worse than 0.01, and Frontier FPR increase no greater than 0.02. However, at the fixed zero-logit threshold Frontier balanced accuracy fell 0.0801 and FNR rose 0.4712 while FPR fell 0.3109. This is a meaningful operating-point tradeoff; AUC evidence does not establish a production decision policy. Tiny balanced accuracy improved +0.0633. WildAIGC was diagnostic only.

The missing-frequency fallback exactly matched the frozen baseline. Candidate coefficients are stored at `research/r10_e11/candidate_coefficients.json` (SHA-256 `d6c88fd365a4ca1c609ea135db69f7534c254512f1730a19be2a81b35d06c4ed`); aggregate metrics at `research/r10_e11/result.json`. Outputs contain no row-level predictions or identifiers. These are development validation results, not locked held-out evaluation, calibration, or production approval.
