# TruthLens — Virality & Social Impact Data Availability Assessment

> **Date**: 2026-10-02
> **Status**: Investigation complete — most modules lack sufficient training data for real-world predictions.

> **Phase 4/5 update (2026-11-17):** This historical investigation is not an inventory of datasets installed in the current checkout. No Census/TRAI tables are bundled or used for region scores; language-region estimates were removed. The current virality implementation has no synthetic `DEMO_MODE` data path. It returns only labeled content indicators, with historical prediction `DATA_UNAVAILABLE`. See [`phase4_dataset_inventory.md`](phase4_dataset_inventory.md), [`regional_spread_methodology.md`](regional_spread_methodology.md), and [`virality_methodology.md`](virality_methodology.md).

---

## 1. Purpose

This document records every data source investigated for the virality estimation, regional spread, and social impact modules. Each source is assessed for legitimacy, availability, and suitability.

**Critical Rule**: If a module cannot be trained or validated on real data, it MUST return `data_unavailable` status and NEVER present predictions as reliable.

---

## 2. Investigated Data Sources

### 2.1 Misinformation Detection & Classification Datasets

| # | Name | URL | License | Variables | Sample Size | Population | Time Period | Geographic Coverage | Limitations |
|---|------|-----|---------|-----------|-------------|------------|-------------|---------------------|-------------|
| 1 | **FakeNewsNet** | github.com/KaiDMML/FakeNewsNet | MIT | News content, social context, engagement (retweets, replies), temporal data, user profiles | ~23k articles | PolitiFact + GossipCop users | 2015–2019 | USA (English) | US-only; engagement counts may be stale; Twitter/X API changes limit reproducibility |
| 2 | **CREDBANK** | github.com/compsocial/CREDBANK | Research use | Tweets, credibility annotations (5-point scale), event clusters | ~60M tweets across ~1,049 events | Twitter users | Oct 2014 – Feb 2015 | Global (English-dominant) | Old data; Twitter-specific; no image-level engagement; credibility is crowd-annotated |
| 3 | **MuMiN** | github.com/MUMIN-dataset | CC BY-NC-SA 4.0 | 21M+ tweets, 41 languages, claim-tweet graph, temporal metadata | 21M tweets, 12k claims | Twitter users | 2020–2021 | Global (41 languages) | Twitter-specific; no image-level virality; requires Twitter API hydration |
| 4 | **BharatFakeNewsKosh** | kaggle.com | CC BY-NC 4.0 | News articles, 9 Indian languages, category labels, source metadata | ~26k articles | Indian news consumers | 2018–2023 | India (9 languages) | Text-only; no engagement/spread metrics; no image-level data |
| 5 | **FactDrill** | IIIT-Delhi/Precog | Research use | 22k+ fact-checked posts, 13 Indian languages, domain labels, multimedia | ~22k items | Indian social media | 2018–2022 | India (13 languages) | No engagement counts; fact-check labels only; limited image metadata |
| 6 | **IFND (Indian Fake News Dataset)** | Zenodo/ResearchGate | Research use | Text + images, event labels, timestamps | ~10k articles | Indian news events | 2013–2021 | India | No virality/engagement metrics; primarily classification labels |
| 7 | **ISOT Fake News Dataset** | uvic.ca | Research use | Article text, title, date, subject | ~44k articles | US news | 2016–2017 | USA | Text-only; no engagement data; old |
| 8 | **Gen AI Misinfo Dataset** | Kaggle | CC BY 4.0 | Content, engagement (likes, shares, comments), quality indicators | ~500 records | Synthetic + curated | 2024 | Mixed | Very small (500); partially synthetic; not representative |

### 2.2 Engagement & Virality Datasets

| # | Name | URL | License | Variables | Sample Size | Limitations |
|---|------|-----|---------|-----------|-------------|-------------|
| 9 | **Twitter Viral Tweets** | Various Kaggle sources | Varies | Retweet counts, favorite counts, text, hashtags, timestamps | 10k–100k tweets | Twitter-specific; no images; API-dependent; platform bias |
| 10 | **Reddit Submissions** | pushshift.io / academic | Research use | Upvotes, comments, post type, subreddit, timestamp | Millions | Reddit-specific; not India-focused; different engagement dynamics |
| 11 | **WhatsApp Election 2019 (ICWSM)** | MIT Media Lab | Research use | Fact-checked images, WhatsApp group metadata, timestamps | ~1,000 images | Small; India-specific but WhatsApp-only; no direct engagement counts |

### 2.3 Indian Regional & Demographic Data

| # | Name | URL | License | Variables | Sample Size | Limitations |
|---|------|-----|---------|-----------|-------------|-------------|
| 12 | **Census of India 2011** | censusindia.gov.in | Public/Gov | State literacy rates, population, age distribution, language distribution | Full census | No internet/social media data; 15 years old |
| 13 | **TRAI Telecom Reports** | trai.gov.in | Public/Gov | Internet subscribers by state, broadband/narrowband split, urban/rural | Quarterly reports | Subscriber counts, not usage behavior; no content engagement data |
| 14 | **NFHS-5 (National Family Health Survey)** | rchiips.org | Public/Gov | Media exposure, phone ownership, internet use by state/gender/age | ~636k households | Survey-based; no social media engagement specifics |
| 15 | **India Internet Report (IAMAI)** | iamai.in | Proprietary (summary public) | Internet users by state, urban/rural, language preference | Survey-based | Proprietary full data; only summaries public; no engagement metrics |

### 2.4 Language & Linguistic Data

| # | Name | URL | License | Variables | Sample Size | Limitations |
|---|------|-----|---------|-----------|-------------|-------------|
| 16 | **Census 2011 Language Data** | censusindia.gov.in | Public/Gov | Speakers per language, bilingual rates, state-wise distribution | Full census | Static snapshot; no digital behavior correlation |
| 17 | **IndicNLP / AI4Bharat** | ai4bharat.org | Various open | NLP resources for Indian languages, corpora, models | Varies | NLP tools, not engagement/spread data |

---

## 3. Assessment Summary

### What IS available
- **Fake news classification** labels (real/fake) for Indian content — YES (BharatFakeNewsKosh, FactDrill, IFND)
- **Language distribution** across Indian states — YES (Census 2011)
- **Internet subscriber counts** by state — YES (TRAI reports)
- **Literacy rates** by state — YES (Census 2011)
- **Fact-checked content** in Indian languages — YES (FactDrill)

### What is NOT available
- ⚠️ **Image-level engagement/virality metrics** for Indian content — NO
- ⚠️ **Regional spread patterns** (which states shared what) — NO
- ⚠️ **WhatsApp forwarding behavior** at scale — NO (encrypted, no public API)
- ⚠️ **Causal relationship** between image features and virality — NO
- ⚠️ **Social impact measurements** (communal tension, health outcomes from misinfo) — NO
- ⚠️ **Calibrated virality prediction models** trained on Indian data — NO

---

## 4. Honest Conclusion

### Virality Estimation
**Status: `DATA_UNAVAILABLE` for trained prediction models**

No publicly available dataset provides sufficient image-level engagement data with Indian context to train a reliable virality prediction model. The available datasets (FakeNewsNet, CREDBANK) are US/Twitter-centric and do not generalize to Indian platforms (WhatsApp, ShareChat, Koo). We can compute **heuristic feature scores** based on known correlates of virality (emotional content, faces, urgency language, etc.) but these are NOT validated predictions.

### Regional Spread Estimation
**Status: `DATA_UNAVAILABLE` for trained prediction models**

While state-level internet penetration (TRAI) and language distribution (Census) data exist, there is NO dataset mapping content characteristics to actual regional spread patterns in India. We can estimate **language compatibility** (does the content language match state-level language speakers) but this is a heuristic, not a trained model.

### Social Impact Assessment
**Status: `DATA_UNAVAILABLE` for trained prediction models**

No publicly available dataset provides measured social impact outcomes (communal incidents, health behavior changes, financial losses) linked to specific misinformation images. We can assess **content characteristics** (sensitivity categories, emotional tone) using rule-based analysis, but these are informational indicators, not impact predictions.

---

## 5. TruthLens Approach

Given the data reality, TruthLens implements:

1. **Heuristic feature extraction** — compute features known from literature to correlate with virality
2. **Rule-based risk indicators** — identify content characteristics associated with social harm
3. **Language-region mapping** — use Census data to estimate language reach by state
4. **Transparent labeling** — ALL outputs carry `data_status: "DATA_UNAVAILABLE"` or `"DEMO"`
5. **DEMO_MODE** — optional synthetic demonstration data, always clearly labeled
6. **Future-ready architecture** — plug in real models when training data becomes available

---

## 6. References

1. Shu et al. (2020). "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information." *Big Data* 8(3).
2. Mitra & Gilbert (2015). "CREDBANK: A Large-Scale Social Media Corpus With Associated Credibility Annotations." *ICWSM*.
3. Nielsen & McConville (2022). "MuMiN: A Large-Scale Multilingual Multimodal Fact-Checked Misinformation Social Network Dataset." *SIGIR*.
4. Singhal et al. (2022). "FactDrill: A Data Repository of Fact-Checked Social Media Content." *AAAI ICWSM*.
5. Census of India (2011). Office of the Registrar General & Census Commissioner, India.
6. TRAI (2024). "Indian Telecom Services Performance Indicator Reports." Telecom Regulatory Authority of India.
7. Reis et al. (2020). "Can WhatsApp Counter Misinformation by Limiting Message Forwarding?" *ICWSM*.
