"""
Unit tests for model architectures, baseline heuristics, and persistence.
"""

import os
import pytest
from src.model import (
    BaselineSentimentHeuristic,
    RevioSentimentPipeline,
    RevioAspectModelSuite,
    RevioCrisisRiskClassifier,
    save_models,
    load_models
)


class TestBaselineHeuristic:
    def test_positive_lexicon(self):
        baseline = BaselineSentimentHeuristic()
        res = baseline.predict_one("The food was delicious and staff was amazingly friendly!")
        assert res["prediction"] == "positive"
        assert res["confidence"] > 0.5

    def test_negative_lexicon(self):
        baseline = BaselineSentimentHeuristic()
        res = baseline.predict_one("Horrible terrible disaster and awful service.")
        assert res["prediction"] == "negative"
        assert res["confidence"] > 0.5

    def test_negation_handling(self):
        baseline = BaselineSentimentHeuristic()
        res = baseline.predict_one("The food was not good at all.")
        assert res["prediction"] == "negative"


class TestRevioModels:
    def test_sentiment_pipeline_training_and_proba(self):
        texts = [
            "Outstanding service and delicious wonderful food!",
            "Great experience loved the meal and helpful staff.",
            "Worst place ever, dirty and rude staff horrible.",
            "Terrible disaster unacceptable quality and cold food.",
            "Standard average visit, nothing special routine.",
            "Decent place adequate experience ordinary."
        ]
        labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
        pipe = RevioSentimentPipeline()
        pipe.fit(texts, labels)

        preds = pipe.predict(["Wonderful meal and delicious food"])
        assert preds[0] == "positive"

        probs = pipe.predict_proba(["Terrible disaster place"])
        assert probs.shape == (1, 3)

    def test_crisis_model_risk_scoring(self):
        texts = [
            "Hospital emergency food poisoning bacterial contamination",
            "Violent vomiting salmonella hospitalization emergency",
            "Great coffee and fast lovely wifi",
            "Superb dining nice day pleasant ambiance"
        ]
        flags = [1, 1, 0, 0]
        crisis = RevioCrisisRiskClassifier()
        crisis.fit(texts, flags)

        res_crisis = crisis.predict_risk("Hospitalized with acute food poisoning last night!")
        assert res_crisis["risk_tier"] in ["HIGH", "CRITICAL"]
        assert res_crisis["risk_score"] >= 0.70

        res_low = crisis.predict_risk("Pleasant ambiance and lovely coffee.", sentiment_pred="positive")
        assert res_low["risk_tier"] == "LOW"

    def test_model_serialization(self, tmp_path):
        texts = ["good food", "delicious meal", "bad food", "horrible meal"]
        labels = ["positive", "positive", "negative", "negative"]
        pipe = RevioSentimentPipeline().fit(texts, labels)
        aspects = RevioAspectModelSuite().fit(texts, {"customer_service": labels})
        crisis = RevioCrisisRiskClassifier().fit(["food poisoning", "hospital emergency", "nice day", "good food"], [1, 1, 0, 0])

        save_models(pipe, aspects, crisis, save_dir=str(tmp_path))
        p_loaded, a_loaded, c_loaded = load_models(str(tmp_path))

        assert p_loaded.predict(["good food"])[0] == "positive"
        assert c_loaded.is_fitted is True
