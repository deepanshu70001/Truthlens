# Multilingual evaluation

## Verified implementation scope

OCR is configured for English (`en`) and Hindi (`hi`) by default, with EasyOCR preferred and Tesseract fallback when installed with mapped language packs. The returned `language` value is estimated from Unicode script counts (Devanagari → `hi`, Latin → `en`, and several other script tags), not a trained language-identification result. Text normalization applies Unicode NFC and whitespace collapsing. Claim extraction is rule based with English and a limited Hindi cue set. Translation is not implemented.

## Evaluation status

No licensed multilingual image-text dataset or ground-truth transcripts exist in the repository. The Phase 5 runtime had neither EasyOCR nor Tesseract, so actual OCR output was unavailable. No CER, WER, exact recovery, language-ID accuracy, translation metric, or per-language claim extraction score is reported. Tests exercise software contracts, not language quality.

| Language/script | Current status | Quantitative result |
|---|---|---|
| English / Latin | Configured OCR; runtime engine not installed during Phase 5 checks | Not evaluable |
| Hindi / Devanagari | Configured OCR; script estimate available in code; no labeled benchmark | Not evaluable |
| Hinglish / mixed Hindi-English | No dedicated mixed-language model or annotated test set | Not evaluated |
| Other scripts in script estimator | Recognition is not promised by default OCR configuration; no benchmark | Unsupported/unverified; not evaluated |
| Translation | No translation module | Not implemented |

## Future protocol

For each explicitly supported language, construct licensed images with human transcripts and a documented orthographic normalization guide. Keep scripts/language pairs separate. Stratify by small text, rotation, compression, decorative fonts, mixed scripts and resolution. Record OCR engine/version/language packs and failures. Report character and word error rate, exact text recovery, language/script confusion and claim-span metrics separately by language. Use double annotation and adjudication for ambiguous transcripts; consent and privacy review are required where source images include people or sensitive text. Do not aggregate an unsupported language into an overall score.
