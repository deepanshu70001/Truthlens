# Phase 6 social analytics evaluation

| Module | Actual method/status | Evidence class | Available evaluation | Finding |
|---|---|---|---|---|
| Virality | Hand-weighted keyword/content cues; no learned predictor or observed propagation target | Heuristic | Unit/contract tests and prior synthetic timing microbenchmark | Does not estimate validated reach, probability, or time-to-viral. Predictive accuracy unavailable. |
| India regional context | Canonical 36 administrative units; risk scores/ranks remain null without regional data; context fields are not validated outcomes | DATA_UNAVAILABLE | Contract checks for null scores and output coverage | No region can be ranked or described as more susceptible. Language fit/connectivity context would not establish spread or susceptibility. |
| Social impact | Exact-token categories identify potential content themes; no observed harm outcome or severity model | Heuristic | Rule-path tests only | Output is a content cue, not measured or predicted societal harm. No human-reviewed neutrality study. |

## Consistency checks and status

Existing Phase 4 tests check data-unavailable output contracts, stable administrative units, and category cue behavior. No source dataset is installed for virality, regional dissemination, or realized social impact. Thus there are no MAE/RMSE/rank metrics, regional outcome estimates, precision/recall of harm alerts, or human-review results. No synthetic scenario outputs are represented as real observations.

## Risks and future evaluation

Use permitted longitudinal engagement/cascade data with a declared prediction horizon for virality. Use a region mapping reference with explicit boundary/version and observed outcomes for regional analysis; never infer susceptibility from identity or demographic proxies. For social-impact cues, predefine a neutral category taxonomy, include politically sensitive and non-harmful controls, blind independent reviewers, measure inter-rater agreement and false alarms, and keep content opinion separate from factual claims. Report all findings as heuristic, observational, human-reviewed, or predictive; never blur the categories.
