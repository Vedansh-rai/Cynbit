"""
Unit tests for end-to-end inference engine and batch triage.
"""

import pytest
import pandas as pd
from src.inference import RevioInferenceEngine


@pytest.fixture(scope="module")
def engine():
    return RevioInferenceEngine(model_dir="models")


class TestInferenceEngine:
    def test_engine_ready(self, engine):
        assert engine.is_ready() is True

    def test_analyze_positive_review(self, engine):
        review = "The food quality was exceptional, bursting with authentic fresh flavors. The staff was polite and helpful."
        res = engine.analyze_single_review(review, business_name="Test Cafe")
        assert res["success"] is True
        assert res["sentiment"]["label"] == "positive"
        assert res["sentiment"]["confidence"] > 0.6
        assert res["risk_assessment"]["risk_tier"] == "LOW"
        assert "ai_response" in res
        assert res["latency_ms"] > 0

    def test_analyze_critical_crisis_review(self, engine):
        review = (
            "EMERGENCY HEALTH HAZARD! Four of us got violent food poisoning and were admitted to Fortis Hospital. "
            "Our lawyer is filing an FIR and consumer court notice tomorrow morning."
        )
        res = engine.analyze_single_review(review)
        assert res["success"] is True
        assert res["sentiment"]["label"] == "negative"
        assert res["risk_assessment"]["risk_tier"] == "CRITICAL"
        assert res["ai_response"]["requires_human_review"] is True
        assert res["ai_response"]["status"] == "READY_FOR_HUMAN_APPROVAL"

    def test_invalid_input_handling(self, engine):
        res = engine.analyze_single_review("")
        assert res["success"] is False
        assert "empty or blank" in res["error"]

        res_short = engine.analyze_single_review("abc")
        assert res_short["success"] is False
        assert "too short" in res_short["error"]

    def test_batch_analysis(self, engine):
        df = pd.DataFrame({
            "id": [1, 2],
            "review_text": [
                "The staff was incredibly polite, attentive, and helpful throughout our visit.",
                "The meal was completely bland, cold, and tasted like it was reheated from yesterday."
            ],
            "location": ["Jaipur Central", "Jaipur Malviya Nagar"]
        })
        results_df = engine.analyze_batch(df)
        assert len(results_df) == 2
        assert "sentiment" in results_df.columns
        assert "risk_tier" in results_df.columns
        assert results_df.iloc[0]["sentiment"] == "positive"
        assert results_df.iloc[1]["sentiment"] == "negative"

