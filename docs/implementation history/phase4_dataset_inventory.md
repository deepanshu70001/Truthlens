# Phase 4 dataset inventory

Inventory date: 2026-11-17. A repository scan found no local virality scenarios, engagement/cascade observations, regional dissemination outcomes, or measured social-impact outcomes. Existing `data/` content consists of individual analysis uploads and derived artifacts; these are not population datasets and are not used for training or calibration. No Phase 4 model is trained.

| Dataset/source | Class | Local status | Target and coverage | Rights/readiness | Use |
|---|---|---|---|---|---|
| Repository analysis uploads and processed artifacts | USER_DATA | Present, case-level | No validated engagement, region, or impact outcome labels | Not a research corpus; user inputs remain case data | No training/calibration |
| FakeNewsNet | RESEARCH_DATASET | Not present | English news and Twitter engagement/context, principally US; content may require rehydration | Complete distribution constrained by Twitter/news rights; candidate only | Not used |
| PHEME | RESEARCH_DATASET | Not present | Event-limited English Twitter rumour conversations and veracity annotations | Tweet rights remain with platform users; no Indian regional/outcome fit | Not used |
| MuMiN | RESEARCH_DATASET | Not present | Multilingual fact-check/social graph research dataset | CC BY-NC terms; noncommercial restrictions; no Indian state outcome calibration | Not used |
| Indian Fake News Dataset (IFND) | RESEARCH_DATASET | Not present | India-related text/image fact-check classification; not spread or impact outcomes | Paper describes dataset; bundled media reuse rights require separate confirmation | Not used |
| Census C-16 mother-tongue tables (2011) | OFFICIAL_REFERENCE | No source table installed | Language counts by geography; demographic context only | Official reference, dated; cannot establish dissemination | Not used for scores |
| TRAI Performance Indicator reports | OFFICIAL_REFERENCE | No source table installed | Telecom service-area subscriber context, geography not identical to state/UT | Official reports; service-area boundaries/denominators need careful alignment | Not used for scores |
| Indian/Brazil WhatsApp fact-checked image studies; PoliWAM | RESTRICTED_OR_UNCONFIRMED | Not present | Selected fact-checked images or political WhatsApp communications; not representative spread outcomes | Content privacy, consent, redistribution, and license need confirmation; sensitive context | Not used |

Readiness: virality outcome model **UNAVAILABLE**; state/UT dissemination model **UNAVAILABLE**; measured social-impact model **UNAVAILABLE**. No file with 500+ labeled scenarios per category or any other accepted train/evaluation split was found. Existing rule-based indicators are not trained models and are not evaluated as predictions. User uploads are not pooled into a dataset.

Sources: [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet), [PHEME downloads](https://www.pheme.eu/software-downloads/), [MuMiN](https://mumin-dataset.github.io/), [IFND paper](https://link.springer.com/article/10.1007/s40747-021-00552-1), [Census C-16](https://censusindia.gov.in/nada/index.php/catalog/10191/study-description), [TRAI reports](https://trai.gov.in/release-publication/reports/performance-indicators-reports), [PoliWAM paper](https://aclanthology.org/2021.wnut-1.15/).
