# TruthLens responsible-AI assessment

Qualitative risk review dated 2026-11-17. Likelihood is not statistically estimated; terms indicate plausible exposure in the intended demo context. This is not a legal opinion, bias audit or security certification.

| Risk | Likelihood | Potential impact | Mitigation in current system | Residual risk |
|---|---|---|---|---|
| Genuine image falsely labeled AI-generated/manipulated | Medium; rate unknown | Reputational harm, false accusation, distrust | Inconclusive band, uncalibrated-score caveat, no separate manipulation score, human review message | High until representative validation and threshold study; model must not make consequential decisions |
| AI-generated image missed | Medium; rate unknown | False reassurance or misinformation persistence | Score and limitations displayed; evidence channels separated | High under new generators, recompression and distribution shift |
| User over-relies on model/fusion score | Medium | Treats signal as proof | UI and API caveats state probabilistic/uncalibrated nature | Medium/high; usability testing absent; cautious wording must remain visible |
| Incomplete/stale fact-check evidence interpreted as truth | Medium | Incorrect public claim adjudication | Provider matches and retrieval status are traceable; no-results/unavailable distinguished | High; provider coverage/rating quality not comprehensively evaluated |
| OCR or script estimate mistranslates content | Medium | Wrong query/verdict, unequal language experience | Raw/normalized text and script estimate are surfaced; translation is absent | High for low-quality, mixed-script, unsupported languages; no CER/WER study |
| Regional context becomes stereotyping or susceptibility ranking | Low in current output, potentially high if misrepresented | Stigmatization of regions/language communities | Regional scores/ranks null when data unavailable; methodology disclaims prediction | Medium; UI/presentation must not infer susceptibility from demographic proxies |
| Political/sensitive content over-flagged as harm | Medium; false-alarm rate unknown | Chilling effect, perceived partisan bias, alarm | Exact-token categories are presented as potential cues, not measured harm | High; taxonomy and neutrality not human-reviewed |
| Personal information in image/OCR exposed | Medium | Privacy loss | Local storage paths omitted from API; OCR/image content not deliberately logged; CORS constrained | High for shared deployments: no auth, deletion/TTL, or retention consent flow |
| External provider receives user data when enabled | Conditional | Data disclosure beyond local system | Integrations require configured credentials; docs disclose Google Vision image and Fact Check query transmission | Medium/high; obtain informed consent and review provider terms before use |
| Metadata/location leaks | Low/medium | Exposes capture location or device details | GPS redaction in reports and privacy notes | Residual logs/storage and future API changes need review |
| Heatmap misinterpreted as a manipulation mask | Medium | Unsupported accusation | Attribution caveat in service/UI | Medium; no localization or human interpretability study |

No bias-free or fair-model claim is made. The sample set is unlabeled and tiny, and no demographic or political subgroup assessment has been conducted. Before public or panel deployment, use consented documented test materials, remove private metadata, explain local/external data flows, provide retention/deletion operations, and include a human review pathway. Do not use this prototype for eligibility, enforcement, or high-impact decisions.
