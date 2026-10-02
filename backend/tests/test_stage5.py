"""
TruthLens — Stage 5 tests: Virality, Regional Spread, Social Impact.

Tests cover:
- Virality estimation (data_unavailable, demo mode, features)
- Regional spread (language compatibility, all states, demo mode)
- Social impact (sensitivity categories, severity, demo mode)
- Report integration (data status labels)
- All to_dict serialization
"""

import io
import json
import os
from unittest.mock import patch

import pytest
from PIL import Image


# ── Virality Engine ──────────────────────────────────────────────

class TestViralityEngine:
    def test_basic_estimation(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality()
        assert result.data_status == "DATA_UNAVAILABLE"
        assert 0.0 <= result.virality_score <= 1.0
        assert result.risk_level in ("low", "moderate", "high", "unknown")

    def test_data_unavailable_by_default(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality()
        assert result.data_status == "DATA_UNAVAILABLE"
        d = result.to_dict()
        assert "disclaimer" in d
        assert "HEURISTIC" in d["disclaimer"]

    def test_demo_mode(self):
        from backend.app.services.virality.engine import estimate_virality

        with patch.dict(os.environ, {"DEMO_MODE": "true"}):
            result = estimate_virality()
        assert result.data_status == "DEMO"
        d = result.to_dict()
        assert "DEMO DATA" in d["disclaimer"]
        assert any("DEMO DATA" in w for w in result.warnings)

    def test_urgency_features(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality(
            ocr_text="BREAKING NEWS: urgent alert now! Share immediately!",
        )
        urgency_feat = [f for f in result.features if f.name == "urgency_language"]
        assert len(urgency_feat) == 1
        assert urgency_feat[0].present is True
        assert urgency_feat[0].value > 0

    def test_emotional_features(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality(
            ocr_text="This is outrageous! A shocking scandal exposed today!",
        )
        emotional_feat = [f for f in result.features if f.name == "emotional_language"]
        assert len(emotional_feat) == 1
        assert emotional_feat[0].present is True

    def test_health_topic_detection(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality(
            ocr_text="New covid vaccine cure discovered. This treatment works against the virus.",
        )
        health_feat = [f for f in result.features if f.name == "health_topic"]
        assert len(health_feat) == 1
        assert health_feat[0].present is True

    def test_political_topic_detection(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality(
            ocr_text="Election fraud by the government party exposed. Vote rigged in Parliament.",
        )
        political_feat = [f for f in result.features if f.name == "political_topic"]
        assert len(political_feat) == 1
        assert political_feat[0].present is True

    def test_no_text_low_score(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality()
        assert result.virality_score == 0.0
        assert result.risk_level == "low"

    def test_claims_detected(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality(
            claims=[{"claim": "500 people died", "claim_type": "statistical"}],
        )
        claims_feat = [f for f in result.features if f.name == "has_claims"]
        assert len(claims_feat) == 1
        assert claims_feat[0].present is True

    def test_image_features_with_people(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality(
            vision_result={"has_people": True, "object_count": 5},
        )
        face_feat = [f for f in result.features if f.name == "has_faces"]
        assert face_feat[0].present is True
        obj_feat = [f for f in result.features if f.name == "multiple_objects"]
        assert obj_feat[0].present is True

    def test_to_dict_serializable(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality(
            ocr_text="Breaking news: 1000 deaths reported today!",
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        assert json_str
        assert "methodology" in d
        assert "features" in d

    def test_methodology_always_present(self):
        from backend.app.services.virality.engine import estimate_virality

        result = estimate_virality()
        assert result.methodology
        assert "heuristic" in result.methodology.lower() or "HEURISTIC" in result.methodology


# ── Regional Spread Engine ───────────────────────────────────────

class TestRegionalSpread:
    def test_basic_estimation(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread(content_language="hi")
        assert result.data_status == "DATA_UNAVAILABLE"
        assert len(result.regions) > 0

    def test_data_unavailable_by_default(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread()
        assert result.data_status == "DATA_UNAVAILABLE"
        d = result.to_dict()
        assert "disclaimer" in d
        assert "ESTIMATE" in d["disclaimer"]

    def test_demo_mode(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        with patch.dict(os.environ, {"DEMO_MODE": "true"}):
            result = estimate_regional_spread()
        assert result.data_status == "DEMO"
        d = result.to_dict()
        assert "DEMO DATA" in d["disclaimer"]

    def test_hindi_content_highest_in_hindi_states(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread(content_language="hi")
        # Hindi-speaking states should have higher risk scores
        hindi_states = {"Uttar Pradesh", "Bihar", "Rajasthan", "Madhya Pradesh"}
        top_regions = result.regions[:10]
        top_names = {r.region for r in top_regions}
        # At least some Hindi states should be in top 10
        assert len(hindi_states & top_names) >= 2

    def test_tamil_content_highest_in_tamil_states(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread(content_language="ta")
        # Tamil Nadu and Puducherry should rank high
        top_5 = [r.region for r in result.regions[:5]]
        assert "Tamil Nadu" in top_5 or "Puducherry" in top_5

    def test_all_indian_states_covered(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread()
        # Should include major states
        regions = {r.region for r in result.regions}
        major_states = {"Uttar Pradesh", "Maharashtra", "Tamil Nadu", "Kerala", "West Bengal"}
        assert major_states.issubset(regions)

    def test_risk_score_range(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread(content_language="hi")
        for region in result.regions:
            assert 0.0 <= region.risk_score <= 1.0
            assert 0.0 <= region.confidence <= 1.0
            assert 0.0 <= region.language_compatibility <= 1.0

    def test_confidence_is_low(self):
        """Confidence should be low since this is heuristic."""
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread()
        for region in result.regions:
            assert region.confidence <= 0.3  # Low confidence for heuristic

    def test_evidence_populated(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread(content_language="hi")
        for region in result.regions:
            assert len(region.evidence) > 0  # At least internet penetration info

    def test_to_dict_serializable(self):
        from backend.app.services.regional.engine import estimate_regional_spread

        result = estimate_regional_spread(content_language="en")
        d = result.to_dict()
        json_str = json.dumps(d)
        assert json_str
        assert "regions" in d
        assert "methodology" in d


# ── Social Impact Engine ─────────────────────────────────────────

class TestSocialImpact:
    def test_basic_assessment(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact()
        assert result.data_status == "DATA_UNAVAILABLE"
        assert 0.0 <= result.overall_risk <= 1.0

    def test_data_unavailable_by_default(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact()
        assert result.data_status == "DATA_UNAVAILABLE"
        d = result.to_dict()
        assert "disclaimer" in d
        assert "RULE-BASED" in d["disclaimer"]

    def test_demo_mode(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        with patch.dict(os.environ, {"DEMO_MODE": "true"}):
            result = assess_social_impact()
        assert result.data_status == "DEMO"
        d = result.to_dict()
        assert "DEMO DATA" in d["disclaimer"]

    def test_communal_content_detected(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(
            ocr_text="The mob attacked a temple. Communal riot near the mosque. Kill them all.",
        )
        communal_cat = [c for c in result.categories if c.category == "communal_harmony"]
        assert len(communal_cat) == 1
        assert communal_cat[0].risk_score > 0

    def test_health_content_detected(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(
            ocr_text="This miracle cure treats cancer and diabetes. Take this medicine instead of vaccine.",
        )
        health_cat = [c for c in result.categories if c.category == "public_health"]
        assert len(health_cat) == 1
        assert health_cat[0].risk_score > 0

    def test_financial_content_detected(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(
            ocr_text="Investment scheme scam! Your bank account and money are at risk. UPI fraud alert.",
        )
        financial_cat = [c for c in result.categories if c.category == "financial_safety"]
        assert len(financial_cat) == 1
        assert financial_cat[0].risk_score > 0

    def test_no_text_low_risk(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact()
        assert result.overall_risk == 0.0
        assert result.risk_level == "low"

    def test_misinformation_severity_fake(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(
            authenticity_label="likely_ai_generated",
            factcheck_status="success",
            factcheck_rating="False",
        )
        assert result.misinformation_severity == "high"

    def test_misinformation_severity_authentic(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(
            authenticity_label="likely_authentic",
        )
        assert result.misinformation_severity == "low"

    def test_audience_scale_hindi(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(content_language="hi")
        assert result.estimated_audience_scale == "broad"

    def test_audience_scale_tamil(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(content_language="ta")
        assert result.estimated_audience_scale == "moderate"

    def test_urgency_level(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(
            ocr_text="Breaking alert! Emergency now! Act immediately!",
        )
        assert result.urgency_level in ("moderate", "high")

    def test_five_categories_present(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact()
        assert len(result.categories) == 5
        cat_names = {c.category for c in result.categories}
        assert "communal_harmony" in cat_names
        assert "public_health" in cat_names
        assert "political_stability" in cat_names
        assert "financial_safety" in cat_names
        assert "child_safety" in cat_names

    def test_to_dict_serializable(self):
        from backend.app.services.social_impact.engine import assess_social_impact

        result = assess_social_impact(
            ocr_text="Vaccine scam fraud in hospital. Breaking news alert!",
            authenticity_label="likely_manipulated",
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        assert json_str
        assert "categories" in d
        assert "methodology" in d


# ── Report Integration ───────────────────────────────────────────

class TestReportIntegration:
    def test_report_includes_virality(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test",
            status="completed",
            virality={"data_status": "DATA_UNAVAILABLE", "virality_score": 0.5},
        )
        assert report["virality"] is not None
        assert report["virality"]["data_status"] == "DATA_UNAVAILABLE"

    def test_report_includes_regional(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test",
            status="completed",
            regional_spread={"data_status": "DATA_UNAVAILABLE", "regions": []},
        )
        assert report["regional_spread"] is not None

    def test_report_includes_social_impact(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test",
            status="completed",
            social_impact={"data_status": "DATA_UNAVAILABLE", "overall_risk": 0.0},
        )
        assert report["social_impact"] is not None

    def test_report_data_status_in_limitations(self):
        """Heuristic modules should be called out in limitations."""
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test",
            status="completed",
            virality={"data_status": "DATA_UNAVAILABLE", "virality_score": 0.5},
            regional_spread={"data_status": "DATA_UNAVAILABLE"},
            social_impact={"data_status": "DATA_UNAVAILABLE"},
        )
        limitations_text = " ".join(report["limitations"])
        assert "heuristic" in limitations_text.lower() or "DATA_UNAVAILABLE" in limitations_text

    def test_report_differentiates_real_vs_demo_vs_unavailable(self):
        """Report must clearly label each module's data status."""
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test",
            status="completed",
            assessment={"label": "likely_authentic", "limitations": []},
            virality={"data_status": "DEMO", "virality_score": 0.5},
            regional_spread={"data_status": "DATA_UNAVAILABLE"},
        )

        # Assessment is a real model result — no data_status
        assert report["assessment"]["label"] == "likely_authentic"
        # Virality is DEMO
        assert report["virality"]["data_status"] == "DEMO"
        # Regional is DATA_UNAVAILABLE
        assert report["regional_spread"]["data_status"] == "DATA_UNAVAILABLE"

    def test_full_report_serializable(self):
        from backend.app.services.reporting.builder import build_report

        report = build_report(
            analysis_id="test-full",
            status="completed",
            image_info={"width": 640, "height": 480},
            assessment={"label": "likely_authentic", "limitations": []},
            virality={"data_status": "DATA_UNAVAILABLE", "virality_score": 0.3},
            regional_spread={"data_status": "DATA_UNAVAILABLE", "regions": []},
            social_impact={"data_status": "DATA_UNAVAILABLE", "overall_risk": 0.0},
        )
        json_str = json.dumps(report)
        assert json_str
        assert report["disclaimer"]
