# Virality methodology

TruthLens reports transparent **content spread indicators**, not an estimate of actual sharing, reach, virality probability, or future outcome. There is no locally installed labeled historical dataset suitable for image-level calibration in India; therefore no trained model, calibration, or predictive claim is made.

The existing heuristic feature names and weights are retained for compatibility with the application. For each available feature, the displayed aggregate is the weighted mean over available feature weights only. Missing OCR, claim extraction, normalized text, or vision fields are excluded from the denominator rather than imputed as zero. The result is null with `risk_level=unknown` when no feature can be assessed. Feature values are observable cue flags/count-based indicators; their weights are not learned or validated. Risk bands must be read as heuristic indicator bands, never likelihoods.

OCR language, urgency/emotional terms, topic terms, extracted claims/numbers, people, and object count are incomplete proxies and can be wrong. Keyword matching can miss inflections, context, negation, irony, and non-English content. Pixel authenticity does not imply virality. Historical prediction status remains `DATA_UNAVAILABLE`, including when heuristic features are available. Demo outputs must remain explicitly marked DEMO.
