"""Scientific Claim Extraction Evaluation Benchmark for TruthLens.

Evaluates deterministic syntactic claim extraction across English and Hindi texts:
1. Verifiable claim detection (Precision, Recall, F1 for ELIGIBLE vs INELIGIBLE)
2. Fine-grained statement classification (FACTUAL, QUOTE, OPINION, QUESTION, SLOGAN, AMBIGUOUS)
3. Span boundary alignment (Exact match and token-level Jaccard overlap)
4. Key entity, number, and date preservation
5. Offset indexing integrity (source_start / source_end character alignment)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from backend.app.services.claims.extractor import extract_claims

# Curated evaluation dataset with ground truth annotations
CLAIMS_BENCHMARK = [
    # Document 1 (English News Statement with Factual Claims & Opinions)
    {
        "doc_id": "doc_en_01",
        "lang": "en",
        "text": "The Ministry of Health confirmed 1,450 new cases in Delhi on October 15, 2026. In my opinion, this is the worst healthcare crisis ever. Did the authorities provide adequate medical supplies? Officials said that twenty hospitals received emergency funding.",
        "annotations": [
            {
                "span": "The Ministry of Health confirmed 1,450 new cases in Delhi on October 15, 2026.",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": ["1,450", "15", "2026"],
                "dates": ["October 15, 2026"],
                "locations": ["Delhi"]
            },
            {
                "span": "In my opinion, this is the worst healthcare crisis ever.",
                "classification": "OPINION",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "Did the authorities provide adequate medical supplies?",
                "classification": "QUESTION",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "Officials said that twenty hospitals received emergency funding.",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            }
        ]
    },
    # Document 2 (English Attributed Quote and Slogan)
    {
        "doc_id": "doc_en_02",
        "lang": "en",
        "text": "\"The inflation rate dropped by 2.5% in September 2026,\" announced the Reserve Bank of India in Mumbai. #BoycottForeignGoods! Support local businesses today.",
        "annotations": [
            {
                "span": "\"The inflation rate dropped by 2.5% in September 2026,\" announced the Reserve Bank of India in Mumbai.",
                "classification": "QUOTE",
                "eligibility": "ELIGIBLE",
                "numbers": ["2.5%", "2026"],
                "dates": ["September 2026"],
                "locations": ["Mumbai", "India"]
            },
            {
                "span": "#BoycottForeignGoods!",
                "classification": "SLOGAN",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "Support local businesses today.",
                "classification": "SLOGAN",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            }
        ]
    },
    # Document 3 (English Statistical & Event Reporting)
    {
        "doc_id": "doc_en_03",
        "lang": "en",
        "text": "Solar power generation reached 50 gigawatts across Gujarat and Rajasthan. I think solar energy is the most beautiful source of power. A severe earthquake occurred near Pune yesterday. Are citizens prepared for disasters?",
        "annotations": [
            {
                "span": "Solar power generation reached 50 gigawatts across Gujarat and Rajasthan.",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": ["50"],
                "dates": [],
                "locations": ["Gujarat", "Rajasthan"]
            },
            {
                "span": "I think solar energy is the most beautiful source of power.",
                "classification": "OPINION",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "A severe earthquake occurred near Pune yesterday.",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": ["Pune"]
            },
            {
                "span": "Are citizens prepared for disasters?",
                "classification": "QUESTION",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            }
        ]
    },
    # Document 4 (Hindi News Statement: Factual Claims, Opinion, Question)
    {
        "doc_id": "doc_hi_01",
        "lang": "hi",
        "text": "चुनाव आयोग ने 12 नवंबर 2026 को मतदान की घोषणा की। मेरे विचार में यह सबसे अच्छा निर्णय है। क्या सभी दलों ने इसका स्वागत किया? भारतीय रेल ने 500 विशेष ट्रेनों का संचालन शुरू किया।",
        "annotations": [
            {
                "span": "चुनाव आयोग ने 12 नवंबर 2026 को मतदान की घोषणा की।",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": ["12", "2026"],
                "dates": [],
                "locations": []
            },
            {
                "span": "मेरे विचार में यह सबसे अच्छा निर्णय है।",
                "classification": "OPINION",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "क्या सभी दलों ने इसका स्वागत किया?",
                "classification": "QUESTION",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "भारतीय रेल ने 500 विशेष ट्रेनों का संचालन शुरू किया।",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": ["500"],
                "dates": [],
                "locations": []
            }
        ]
    },
    # Document 5 (Hindi Attributed Statements and Fragmentary Warnings)
    {
        "doc_id": "doc_hi_02",
        "lang": "hi",
        "text": "मौसम विभाग ने चेतावनी दी कि बिहार और असम में भारी बारिश हुई। ब्रेकिंग! लोगों को घर में रहना चाहिए। पुलिस ने 5 संदिग्धों को गिरफ्तार किया।",
        "annotations": [
            {
                "span": "मौसम विभाग ने चेतावनी दी कि बिहार और असम में भारी बारिश हुई।",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": ["Bihar", "Assam"]
            },
            {
                "span": "ब्रेकिंग!",
                "classification": "AMBIGUOUS",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "लोगों को घर में रहना चाहिए।",
                "classification": "OPINION",
                "eligibility": "INELIGIBLE",
                "numbers": [],
                "dates": [],
                "locations": []
            },
            {
                "span": "पुलिस ने 5 संदिग्धों को गिरफ्तार किया।",
                "classification": "FACTUAL",
                "eligibility": "ELIGIBLE",
                "numbers": ["5"],
                "dates": [],
                "locations": []
            }
        ]
    }
]

def token_jaccard(s1: str, s2: str) -> float:
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def run_evaluation():
    print("=" * 70)
    print("TruthLens Scientific Claim Extraction Evaluation (Package C)")
    print("=" * 70)

    total_gold_propositions = 0
    total_extracted_claims = 0
    
    eligibility_gold: List[str] = []
    eligibility_pred: List[str] = []
    classification_gold: List[str] = []
    classification_pred: List[str] = []
    
    exact_span_matches = 0
    jaccard_scores: List[float] = []
    offset_alignment_valid = 0
    total_spans_checked = 0
    
    # Entity / Number / Date preservation tracking
    gold_numbers_count = 0
    recovered_numbers_count = 0
    gold_dates_count = 0
    recovered_dates_count = 0
    gold_locations_count = 0
    recovered_locations_count = 0

    latencies: List[float] = []
    doc_results: List[Dict[str, Any]] = []

    for doc in CLAIMS_BENCHMARK:
        raw_text = doc["text"]
        annotations = doc["annotations"]
        total_gold_propositions += len(annotations)

        t0 = time.perf_counter()
        res = extract_claims(raw_text, min_confidence=0.0, max_claims=20)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        claims = res.claims
        total_extracted_claims += len(claims)

        doc_eval = {
            "doc_id": doc["doc_id"],
            "lang": doc["lang"],
            "raw_text_length": len(raw_text),
            "claims_extracted_count": len(claims),
            "gold_propositions_count": len(annotations),
            "extracted_claims": [c.to_dict() for c in claims],
            "pairings": []
        }

        # Check offset indexing integrity on all extracted claims
        for c in claims:
            total_spans_checked += 1
            if c.source_start is not None and c.source_end is not None:
                sub = raw_text[c.source_start:c.source_end]
                if sub == c.original_text:
                    offset_alignment_valid += 1

        # Match each gold annotation with best matching extracted claim
        for anno in annotations:
            g_span = anno["span"]
            g_elig = anno["eligibility"]
            g_class = anno["classification"]
            eligibility_gold.append(g_elig)
            classification_gold.append(g_class)

            for num in anno["numbers"]:
                gold_numbers_count += 1
            for d in anno["dates"]:
                gold_dates_count += 1
            for loc in anno["locations"]:
                gold_locations_count += 1

            best_claim = None
            best_jaccard = 0.0

            for c in claims:
                j_score = token_jaccard(g_span, c.original_text or c.claim)
                if j_score > best_jaccard:
                    best_jaccard = j_score
                    best_claim = c

            if best_claim is not None and best_jaccard >= 0.5:
                # Matched extracted claim
                pred_elig = best_claim.verification_eligibility
                pred_class = best_claim.classification
                eligibility_pred.append(pred_elig)
                classification_pred.append(pred_class)
                jaccard_scores.append(best_jaccard)

                if (best_claim.source_start is not None and 
                    raw_text.find(g_span) == best_claim.source_start and 
                    len(g_span) == (best_claim.source_end - best_claim.source_start)):
                    exact_span_matches += 1

                # Check numbers preservation
                for num in anno["numbers"]:
                    clean_num = num.replace("%", "").replace(",", "")
                    if any(clean_num in n.replace("%", "").replace(",", "") for n in best_claim.numbers):
                        recovered_numbers_count += 1

                # Check dates preservation
                for d in anno["dates"]:
                    if any(d in extracted_d for extracted_d in best_claim.dates):
                        recovered_dates_count += 1

                # Check locations preservation
                for loc in anno["locations"]:
                    if any(loc.lower() in l.lower() for l in best_claim.locations):
                        recovered_locations_count += 1

                doc_eval["pairings"].append({
                    "gold_span": g_span,
                    "extracted_span": best_claim.claim,
                    "jaccard_similarity": round(best_jaccard, 4),
                    "gold_eligibility": g_elig,
                    "pred_eligibility": pred_elig,
                    "gold_classification": g_class,
                    "pred_classification": pred_class
                })
            else:
                # False negative (unextracted)
                eligibility_pred.append("INELIGIBLE")
                classification_pred.append("AMBIGUOUS")
                jaccard_scores.append(0.0)
                doc_eval["pairings"].append({
                    "gold_span": g_span,
                    "extracted_span": None,
                    "jaccard_similarity": 0.0,
                    "gold_eligibility": g_elig,
                    "pred_eligibility": "INELIGIBLE",
                    "gold_classification": g_class,
                    "pred_classification": "AMBIGUOUS"
                })

        doc_results.append(doc_eval)

    # 1. Verification Eligibility Metrics (ELIGIBLE as positive class)
    tp = sum(1 for g, p in zip(eligibility_gold, eligibility_pred) if g == "ELIGIBLE" and p == "ELIGIBLE")
    fp = sum(1 for g, p in zip(eligibility_gold, eligibility_pred) if g == "INELIGIBLE" and p == "ELIGIBLE")
    fn = sum(1 for g, p in zip(eligibility_gold, eligibility_pred) if g == "ELIGIBLE" and p == "INELIGIBLE")
    tn = sum(1 for g, p in zip(eligibility_gold, eligibility_pred) if g == "INELIGIBLE" and p == "INELIGIBLE")

    prec_elig = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_elig = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_elig = (2 * prec_elig * rec_elig) / (prec_elig + rec_elig) if (prec_elig + rec_elig) > 0 else 0.0
    acc_elig = (tp + tn) / len(eligibility_gold) if eligibility_gold else 0.0

    # 2. Multi-class Confusion Matrix
    unique_classes = sorted(list(set(classification_gold + classification_pred)))
    conf_matrix = defaultdict(lambda: defaultdict(int))
    for g, p in zip(classification_gold, classification_pred):
        conf_matrix[g][p] += 1

    # 3. Span Alignment Metrics
    exact_match_rate = exact_span_matches / total_gold_propositions if total_gold_propositions > 0 else 0.0
    mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    offset_accuracy = offset_alignment_valid / total_spans_checked if total_spans_checked > 0 else 0.0

    # 4. Entity / Number / Date Recall
    number_recall = recovered_numbers_count / gold_numbers_count if gold_numbers_count > 0 else 1.0
    date_recall = recovered_dates_count / gold_dates_count if gold_dates_count > 0 else 1.0
    location_recall = recovered_locations_count / gold_locations_count if gold_locations_count > 0 else 1.0

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "extraction_method": "conservative_rules_v2",
            "total_documents": len(CLAIMS_BENCHMARK),
            "total_gold_propositions": total_gold_propositions,
            "total_extracted_claims": total_extracted_claims
        },
        "eligibility_detection_metrics": {
            "accuracy": round(acc_elig, 4),
            "precision": round(prec_elig, 4),
            "recall": round(rec_elig, 4),
            "f1_score": round(f1_elig, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        },
        "span_boundary_alignment": {
            "exact_match_rate": round(exact_match_rate, 4),
            "mean_token_jaccard": round(mean_jaccard, 4),
            "offset_alignment_integrity": round(offset_accuracy, 4),
            "total_spans_checked": total_spans_checked,
            "offset_alignment_valid": offset_alignment_valid
        },
        "information_preservation": {
            "number_recall": round(number_recall, 4),
            "numbers_gold_count": gold_numbers_count,
            "numbers_recovered_count": recovered_numbers_count,
            "date_recall": round(date_recall, 4),
            "dates_gold_count": gold_dates_count,
            "dates_recovered_count": recovered_dates_count,
            "location_recall": round(location_recall, 4),
            "locations_gold_count": gold_locations_count,
            "locations_recovered_count": recovered_locations_count
        },
        "latency_metrics": {
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2)
        },
        "confusion_matrix": {g: dict(preds) for g, preds in conf_matrix.items()},
        "document_evaluations": doc_results
    }

    out_path = Path("experiments/reports/claim_extraction_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nClaim Extraction Evaluation Summary:")
    print("-" * 65)
    print(f"Eligibility Accuracy:       {acc_elig * 100:.2f}%")
    print(f"Eligibility Precision:      {prec_elig:.4f}")
    print(f"Eligibility Recall:         {rec_elig:.4f}")
    print(f"Eligibility F1-Score:       {f1_elig:.4f}")
    print("-" * 65)
    print(f"Exact Span Match Rate:      {exact_match_rate * 100:.2f}%")
    print(f"Mean Token Jaccard:         {mean_jaccard:.4f}")
    print(f"Offset Index Integrity:     {offset_accuracy * 100:.2f}% ({offset_alignment_valid}/{total_spans_checked})")
    print("-" * 65)
    print(f"Number Preservation Recall: {number_recall * 100:.2f}% ({recovered_numbers_count}/{gold_numbers_count})")
    print(f"Date Preservation Recall:   {date_recall * 100:.2f}% ({recovered_dates_count}/{gold_dates_count})")
    print(f"Location Preservation:      {location_recall * 100:.2f}% ({recovered_locations_count}/{gold_locations_count})")
    print(f"Mean Extraction Latency:    {np.mean(latencies):.2f} ms")
    print("-" * 65)
    print(f"Artifact written to: {out_path}")

if __name__ == "__main__":
    run_evaluation()
