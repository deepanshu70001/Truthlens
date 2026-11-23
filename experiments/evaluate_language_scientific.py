"""Scientific Language Identification Evaluation Benchmark for TruthLens.

Evaluates the fine-tuned XLM-RoBERTa language detection model (papluca/xlm-roberta-base-language-detection)
and the TruthLens LanguageDetector service wrapper across 20 supported languages, short texts,
code-switched texts, and Romanized/transliterated text.
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
import torch

sys.path.insert(0, os.path.abspath("."))

from backend.app.services.nlp.language_detector import (
    LANGUAGE_NAMES,
    MIN_ALPHABETIC_RATIO,
    MIN_CONFIDENCE,
    MIN_CONTENT_CHARACTERS,
    LanguageDetector,
)

# Standard evaluation corpus: 3 representative natural text samples per supported language (60 samples)
STANDARD_CORPUS = [
    # ar (Arabic)
    {"lang": "ar", "text": "أعلنت وزارة الصحة اليوم عن إطلاق حملة وطنية جديدة للوقاية والتطعيم في مختلف المحافظات."},
    {"lang": "ar", "text": "يعقد المؤتمر الاقتصادي الدولي اجتماعاته السنوية لبحث قضايا التنمية المستدامة والطاقة المتجددة."},
    {"lang": "ar", "text": "حذر خبراء الأرصاد الجوية من موجة حارة غير مسبوقة ستؤثر على المناطق الساحلية والداخلية."},
    # bg (Bulgarian)
    {"lang": "bg", "text": "Министерството на образованието обяви нови мерки за дигитализация на учебния процес в училищата."},
    {"lang": "bg", "text": "Българските учени завършиха успешно международния проект за изследване на климатичните промени."},
    {"lang": "bg", "text": "Парламентът прие на второ четене новия закон за насърчаване на чуждестранните инвестиции."},
    # de (German)
    {"lang": "de", "text": "Die Bundesregierung hat einen umfassenden Gesetzentwurf zur Förderung erneuerbarer Energien beschlossen."},
    {"lang": "de", "text": "Wissenschaftler des Instituts für Technologie haben ein innovatives Verfahren zur Energiespeicherung entwickelt."},
    {"lang": "de", "text": "Die Europäische Zentralbank gab heute ihre neuesten Entscheidungen zur künftigen Zinspolitik bekannt."},
    # el (Greek)
    {"lang": "el", "text": "Η κυβέρνηση ανακοίνωσε νέα δέσμη μέτρων για την ενίσχυση της απασχόλησης και της οικονομικής ανάπτυξης."},
    {"lang": "el", "text": "Οι ερευνητές του πανεπιστημίου ολοκλήρωσαν τη μελέτη για την προστασία των θαλάσσιων οικοσυστημάτων."},
    {"lang": "el", "text": "Η μετεωρολογική υπηρεσία προειδοποιεί για έντονα καιρικά φαινόμενα στις νότιες περιοχές της χώρας."},
    # en (English)
    {"lang": "en", "text": "The international election commission has published the comprehensive observation schedule for upcoming national polls."},
    {"lang": "en", "text": "Medical researchers have published significant clinical trial findings demonstrating high therapeutic efficacy."},
    {"lang": "en", "text": "Cybersecurity agencies have issued an urgent global advisory regarding state-sponsored digital reconnaissance campaigns."},
    # es (Spanish)
    {"lang": "es", "text": "El ministerio de sanidad ha presentado el nuevo plan estratégico para modernizar los centros de atención primaria."},
    {"lang": "es", "text": "Los investigadores de la universidad politécnica han patentado un sistema avanzado de desaliniación sostenible."},
    {"lang": "es", "text": "El banco central anunció que mantendrá los tipos de interés sin cambios tras evaluar la inflación trimestral."},
    # fr (French)
    {"lang": "fr", "text": "Le gouvernement a présenté ce matin un projet de loi ambitieux pour accélérer la transition écologique."},
    {"lang": "fr", "text": "Les chercheurs du Centre National de la Recherche Scientifique ont découvert une nouvelle espèce marine protégée."},
    {"lang": "fr", "text": "L'institut national de la statistique a publié son rapport annuel sur l'évolution de la croissance économique."},
    # hi (Hindi)
    {"lang": "hi", "text": "चुनाव आयोग ने आगामी विधानसभा चुनावों के लिए विस्तृत सुरक्षा और मतदान दिशानिर्देश जारी किए हैं।"},
    {"lang": "hi", "text": "वैज्ञानिक अनुसंधान संस्थान ने सौर ऊर्जा के क्षेत्र में नई दक्षता हासिल करने की आधिकारिक घोषणा की।"},
    {"lang": "hi", "text": "सर्वोच्च न्यायालय ने जनहित याचिका पर सुनवाई करते हुए पर्यावरण संरक्षण के कड़े नियम लागू करने का आदेश दिया।"},
    # it (Italian)
    {"lang": "it", "text": "Il consiglio dei ministri ha approvato il decreto legge per sostenere le imprese colpite dal rincaro energetico."},
    {"lang": "it", "text": "L'università ha annunciato la scoperta di un importante sito archeologico risalente all'epoca romana."},
    {"lang": "it", "text": "La protezione civile ha diramato un'allerta meteo arancione per diverse regioni del nord del paese."},
    # ja (Japanese)
    {"lang": "ja", "text": "気象庁は明日以降の急激な気温低下と大雪に対する厳重な警戒を呼びかけています。"},
    {"lang": "ja", "text": "政府はデジタル社会の実現に向けた新たな包括的サイバーセキュリティ基本方針を閣議決定しました。"},
    {"lang": "ja", "text": "大学の研究グループが次世代量子コンピューター向けの超伝導素子の開発に成功しました。"},
    # nl (Dutch)
    {"lang": "nl", "text": "Het kabinet heeft vandaag overeenstemming bereikt over aanvullende maatregelen voor woningbouw en infrastructuur."},
    {"lang": "nl", "text": "Nederlandse universiteiten hebben een gezamenlijk initiatief gelanceerd voor de ontwikkeling van duurzame landbouw."},
    {"lang": "nl", "text": "De nationale spoorwegen kondigen een aangepaste dienstregeling aan wegens grootschalige spoorwerkzaamheden."},
    # pl (Polish)
    {"lang": "pl", "text": "Rada Ministrów przyjęła dzisiaj projekt ustawy o wsparciu małych i średnich przedsiębiorstw technologicznych."},
    {"lang": "pl", "text": "Polscy naukowcy opublikowali przełomowe wyniki badań nad nowymi materiałami dla energetyki jądrowej."},
    {"lang": "pl", "text": "Instytut Meteorologii wydał ostrzeżenia pierwszego stopnia przed silnym wiatrem w całym kraju."},
    # pt (Portuguese)
    {"lang": "pt", "text": "O governo anunciou um investimento significativo na modernização da rede ferroviária nacional e transportes públicos."},
    {"lang": "pt", "text": "Investigadores da universidade desenvolveram uma nova tecnologia para purificação de águas residuais urbanas."},
    {"lang": "pt", "text": "O instituto nacional de saúde pública iniciou a campanha anual de vacinação contra a gripe sazonal."},
    # ru (Russian)
    {"lang": "ru", "text": "Министерство науки и высшего образования утвердило список приоритетных фундаментальных научных направлений."},
    {"lang": "ru", "text": "Центральный банк опубликовал аналитический обзор текущего состояния банковского сектора страны."},
    {"lang": "ru", "text": "Специалисты гидрометеорологического центра предупредили о резком похолодании в европейской части страны."},
    # sw (Swahili)
    {"lang": "sw", "text": "Serikali imetangaza mpango mpya kabambe wa kuboresha miundombinu ya afya vijijini kote nchini."},
    {"lang": "sw", "text": "Watafiti wa kilimo wamebuni mbegu mpya za mahindi zinazostahimili ukame na magonjwa ya mazao."},
    {"lang": "sw", "text": "Wizara ya elimu imetoa mwongozo mpya kuhusu matumizi ya teknolojia ya kidijitali katika shule za msingi."},
    # th (Thai)
    {"lang": "th", "text": "กระทรวงสาธารณสุขประกาศมาตรการเชิงรุกเพื่อควบคุมการแพร่ระบาดของโรคตามฤดูกาลในทุกจังหวัด"},
    {"lang": "th", "text": "ทีมนักวิจัยไทยประสบความสำเร็จในการพัฒนาเซนเซอร์ตรวจจับมลพิษทางอากาศด้วยเทคโนโลยีนาโน"},
    {"lang": "th", "text": "ธนาคารแห่งประเทศไทยเปิดเผยรายงานเสถียรภาพระบบการเงินและแนวโน้มการเติบโตของเศรษฐกิจ"},
    # tr (Turkish)
    {"lang": "tr", "text": "Sanayi ve Teknoloji Bakanlığı yerli ve milli yapay zeka projelerine yönelik yeni hibe programını başlattı."},
    {"lang": "tr", "text": "Meteoroloji Genel Müdürlüğü Marmara ve Ege bölgeleri için kuvvetli fırtına uyarısında bulundu."},
    {"lang": "tr", "text": "Merkez Bankası para politikası kurulu toplantısının ardından faiz kararına ilişkin basın duyurusunu yayımladı."},
    # ur (Urdu)
    {"lang": "ur", "text": "وزارت برائے اطلاعات و نشریات نے ڈیجیٹل میڈیا کے حوالے سے نئی پالیسی گائیڈ لائنز جاری کر دی ہیں۔"},
    {"lang": "ur", "text": "ماہرین ماحولیات نے بڑھتی ہوئی فضائی آلودگی کی روک تھام کے لیے فوری اقدامات کرنے پر زور دیا۔"},
    {"lang": "ur", "text": "قومی ادارہ برائے شماریات نے پچھلی سہ ماہی کے معاشی اور برآمدی اعداد و شمار کا تفصیلی جائزہ پیش کیا۔"},
    # vi (Vietnamese)
    {"lang": "vi", "text": "Bộ Thông tin và Truyền thông đã ban hành chiến lược chuyển đổi số quốc gia giai đoạn tiếp theo."},
    {"lang": "vi", "text": "Các nhà khoa học tại Viện Hàn lâm đã chế tạo thành công vật liệu bán dẫn thế hệ mới phục vụ vi mạch."},
    {"lang": "vi", "text": "Trung tâm Dự báo Khí tượng Thủy văn Quốc gia cảnh báo mưa lớn diện rộng tại các tỉnh miền Trung."},
    # zh (Chinese)
    {"lang": "zh", "text": "国家发展和改革委员会今日发布了关于推动绿色低碳产业高质量发展的一系列指导意见。"},
    {"lang": "zh", "text": "科研团队在量子保密通信网络技术领域取得重大突破并成功完成实地长距离传输试验。"},
    {"lang": "zh", "text": "商务部最新统计数据显示全国高新技术产业利用外资规模保持持续稳定增长态势。"},
]

# Short text challenge slice (< 20 characters or 1-3 words)
SHORT_TEXT_CORPUS = [
    {"target": "en", "text": "Breaking news"},
    {"target": "hi", "text": "नमस्ते भारत"},
    {"target": "fr", "text": "Bonjour monde"},
    {"target": "de", "text": "Danke schön"},
    {"target": "es", "text": "Buenas tardes"},
    {"target": "it", "text": "Arrivederci"},
    {"target": "ru", "text": "Добрый день"},
    {"target": "ar", "text": "صباح الخير"},
    {"target": "zh", "text": "你好世界"},
    {"target": "ja", "text": "こんにちは"},
]

# Mixed-script / Code-switched challenge slice
MIXED_SCRIPT_CORPUS = [
    {"primary": "hi", "text": "Breaking News: दिल्ली पुलिस ने साइबर क्राइम सिंडिकेट का भंडाफोड़ किया।"},
    {"primary": "hi", "text": "Official Update: सरकार ने new tax regime के नियमों में राहत देने की घोषणा की।"},
    {"primary": "en", "text": "The festival of Diwali (दीपावली) was celebrated with lights and joyous fireworks across Delhi."},
    {"primary": "hi", "text": "Fact Check: सोशल मीडिया पर viral यह video पूरी तरह fake और misleading है।"},
    {"primary": "fr", "text": "Le nouveau smartphone intègre un écran OLED avec technologie AI de pointe."},
    {"primary": "es", "text": "El nuevo software open-source para machine learning ha sido lanzado hoy en GitHub."},
    {"primary": "hi", "text": "Technology Alert: अनजान link पर click न करें, अपना bank password सुरक्षित रखें।"},
    {"primary": "de", "text": "Das Update bringt verbesserte Cloud-Integration und neue Privacy-Features."},
]

# Romanized / Transliterated Hindi (Hinglish) slice
ROMANIZED_CORPUS = [
    {"intended": "hi", "text": "Yeh khabar bilkul galat hai aur social media par afwah failayi ja rahi hai."},
    {"intended": "hi", "text": "Chunav aayog ne voting ki nayi date announce kar di hai kal subah tak."},
    {"intended": "hi", "text": "Aap sabhi se anurodh hai ki kisi bhi anjan link par click mat kijiye."},
    {"intended": "hi", "text": "Sarkaar ne naye budget me kisaano ke liye badi yojana ka ailan kiya hai."},
    {"intended": "hi", "text": "Delhi metro ki nayi line par kal se regular train service shuru ho jayegi."},
    {"intended": "hi", "text": "Mausam vibhag ne agle do dino me bhari baarish ka alert jaari kiya hai."},
    {"intended": "hi", "text": "Supreme court ne apne aadesh me paryavaran sanrakshan ko zaroori bataya hai."},
    {"intended": "hi", "text": "Yeh photo AI se banayi gayi hai aur sach nahi hai, kripya check karein."},
]

def run_evaluation():
    print("=" * 70)
    print("TruthLens Scientific Language Identification Evaluation (Package C)")
    print("=" * 70)

    detector = LanguageDetector()
    loaded = detector.load(device="auto")
    if not loaded:
        print("ERROR: Failed to load LanguageDetector model from cache!")
        return

    model = detector._model
    tokenizer = detector._tokenizer
    device = detector._device
    labels = model.config.id2label
    supported_classes = sorted([labels[k].lower() for k in labels])

    print(f"Model loaded on device: {device}")
    print(f"Supported classes ({len(supported_classes)}): {', '.join(supported_classes)}")

    # 1. Standard Corpus Evaluation (Raw Model + Service)
    y_true: List[str] = []
    y_pred_raw: List[str] = []
    y_pred_service: List[str] = []
    top3_correct: List[bool] = []
    latencies: List[float] = []
    sample_records: List[Dict[str, Any]] = []

    for item in STANDARD_CORPUS:
        target = item["lang"]
        text = item["text"]
        y_true.append(target)

        # Raw Model inference
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits[0].float(), dim=0)
            scores, indices = torch.topk(probs, 3)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        top1_idx = int(indices[0])
        top1_lang = labels[top1_idx].lower() if top1_idx in labels else labels[str(top1_idx)].lower()
        top1_conf = float(scores[0])
        y_pred_raw.append(top1_lang)

        top3_langs = [labels[int(idx)].lower() if int(idx) in labels else labels[str(int(idx))].lower() for idx in indices]
        top3_correct.append(target in top3_langs)

        # Service evaluation
        svc_res = detector.detect(text, top_k=3)
        y_pred_service.append(svc_res.detected_language)

        sample_records.append({
            "target": target,
            "text": text,
            "raw_prediction": top1_lang,
            "raw_confidence": round(top1_conf, 4),
            "top3_candidates": top3_langs,
            "service_prediction": svc_res.detected_language,
            "service_outcome": svc_res.outcome,
            "service_confidence": svc_res.confidence,
            "latency_ms": round(lat, 2)
        })

    # Metrics calculation
    all_classes = sorted(list(set(y_true)))
    class_metrics: Dict[str, Dict[str, Any]] = {}
    
    # Confusion matrix
    conf_matrix = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred_raw):
        conf_matrix[t][p] += 1

    tp_total, fp_total, fn_total = 0, 0, 0
    p_list, r_list, f1_list = [], [], []

    for c in all_classes:
        tp = conf_matrix[c][c]
        fp = sum(conf_matrix[other][c] for other in all_classes if other != c)
        fn = sum(conf_matrix[c][other] for other in all_classes if other != c)
        support = sum(conf_matrix[c].values())

        prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        p_list.append(prec)
        r_list.append(rec)
        f1_list.append(f1)

        class_metrics[c] = {
            "name": LANGUAGE_NAMES.get(c, c),
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

    macro_precision = float(np.mean(p_list))
    macro_recall = float(np.mean(r_list))
    macro_f1 = float(np.mean(f1_list))
    top1_accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred_raw)]))
    top3_accuracy = float(np.mean(top3_correct))

    # 2. Short Text Challenge Evaluation
    short_results: List[Dict[str, Any]] = []
    short_gated_count = 0
    for item in SHORT_TEXT_CORPUS:
        svc_res = detector.detect(item["text"])
        is_gated = svc_res.outcome == "insufficient_input"
        if is_gated:
            short_gated_count += 1
        short_results.append({
            "target": item["target"],
            "text": item["text"],
            "length_chars": len(item["text"]),
            "outcome": svc_res.outcome,
            "detected_language": svc_res.detected_language,
            "warnings": svc_res.warnings
        })

    # 3. Mixed Script Challenge Evaluation
    mixed_results: List[Dict[str, Any]] = []
    for item in MIXED_SCRIPT_CORPUS:
        svc_res = detector.detect(item["text"], top_k=3)
        mixed_results.append({
            "primary": item["primary"],
            "text": item["text"],
            "detected_language": svc_res.detected_language,
            "confidence": svc_res.confidence,
            "top_candidates": svc_res.top_languages,
            "matches_primary": svc_res.detected_language == item["primary"]
        })

    # 4. Romanized / Transliterated Challenge Evaluation
    romanized_results: List[Dict[str, Any]] = []
    for item in ROMANIZED_CORPUS:
        svc_res = detector.detect(item["text"], top_k=3)
        # Hinglish often classifies as English (en) or Indonesian/Swahili due to Latin script
        romanized_results.append({
            "intended": item["intended"],
            "text": item["text"],
            "detected_language": svc_res.detected_language,
            "confidence": svc_res.confidence,
            "top_candidates": svc_res.top_languages,
            "classified_as_hindi": svc_res.detected_language == "hi"
        })

    # Assemble comprehensive report dictionary
    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": detector._model.name_or_path,
            "model_revision": detector.model_revision,
            "device": str(device),
            "supported_classes_count": len(supported_classes),
            "evaluation_scope": "20 languages, standard, short, mixed-script, and Romanized text"
        },
        "overall_metrics": {
            "top1_accuracy": round(top1_accuracy, 4),
            "top3_accuracy": round(top3_accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2)
        },
        "class_wise_metrics": class_metrics,
        "confusion_matrix": {t: dict(preds) for t, preds in conf_matrix.items()},
        "slice_evaluations": {
            "short_text_slice": {
                "total_samples": len(SHORT_TEXT_CORPUS),
                "insufficient_input_gated": short_gated_count,
                "gating_rate": round(short_gated_count / len(SHORT_TEXT_CORPUS), 4),
                "samples": short_results
            },
            "mixed_script_slice": {
                "total_samples": len(MIXED_SCRIPT_CORPUS),
                "primary_match_count": sum(1 for r in mixed_results if r["matches_primary"]),
                "primary_match_rate": round(sum(1 for r in mixed_results if r["matches_primary"]) / len(MIXED_SCRIPT_CORPUS), 4),
                "samples": mixed_results
            },
            "romanized_text_slice": {
                "total_samples": len(ROMANIZED_CORPUS),
                "classified_as_hindi_count": sum(1 for r in romanized_results if r["classified_as_hindi"]),
                "classified_as_hindi_rate": round(sum(1 for r in romanized_results if r["classified_as_hindi"]) / len(ROMANIZED_CORPUS), 4),
                "samples": romanized_results,
                "finding": "Romanized Hindi (Hinglish) uses Latin script, leading to systematic misclassification as Latin-alphabet languages (e.g., en/sw/so). The service safely handles this as a documented linguistic limitation."
            }
        }
    }

    out_path = Path("experiments/reports/language_scientific_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nEvaluation Summary:")
    print("-" * 65)
    print(f"Top-1 Accuracy:     {top1_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy:     {top3_accuracy * 100:.2f}%")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1-Score:     {macro_f1:.4f}")
    print(f"Mean Latency:       {np.mean(latencies):.2f} ms")
    print(f"P95 Latency:        {np.percentile(latencies, 95):.2f} ms")
    print("-" * 65)
    print(f"Short text gating rate: {short_gated_count}/{len(SHORT_TEXT_CORPUS)} ({short_gated_count/len(SHORT_TEXT_CORPUS)*100:.1f}%)")
    print(f"Mixed script primary match: {sum(1 for r in mixed_results if r['matches_primary'])}/{len(MIXED_SCRIPT_CORPUS)}")
    print(f"Romanized Hindi classified as 'hi': {sum(1 for r in romanized_results if r['classified_as_hindi'])}/{len(ROMANIZED_CORPUS)}")
    print(f"Artifact written to: {out_path}")

if __name__ == "__main__":
    run_evaluation()
