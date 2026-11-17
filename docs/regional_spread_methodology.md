# Regional spread methodology

Regional spread prediction is disabled. The service returns the canonical 28 states and 8 union territories with null score, confidence, language compatibility, and connectivity fields, and an empty ranking. This is an availability contract, not an assertion that the regions have equivalent exposure.

The repository has no installed, traceable, licensed dataset connecting content and language to observed dissemination by state/UT. Census language tables are demographic context, not vulnerability or spread outcomes. TRAI service areas do not map cleanly to state/UT boundaries. Neither is used to score regions. OCR language is passed only as descriptive content metadata when present; it is never used to rank audiences. Maps/charts must render only records with actual non-null scores.

Potential future sources: [Census C-16 (2011)](https://censusindia.gov.in/nada/index.php/catalog/10191/study-description) and [TRAI performance reports](https://trai.gov.in/release-publication/reports/performance-indicators-reports). Both require acquisition, provenance/version records, geography alignment, license review, and separate validation before use.
