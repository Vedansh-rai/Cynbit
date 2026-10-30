"""
Inference Pipeline and Orchestration Engine for Revio AI.

Coordinates input validation, preprocessing, sentiment classification,
aspect extraction, risk scoring, explainability, and response generation.
"""

from typing import Dict, Any, List, Optional
import time
import pandas as pd
import numpy as np

from src.preprocessing import validate_review_input, clean_text, extract_linguistic_features
from src.model import load_models, RevioSentimentPipeline, RevioAspectModelSuite, RevioCrisisRiskClassifier
from src.response_engine import LocalGuardrailedResponseGenerator, ExternalLLMResponseGenerator


class RevioInferenceEngine:
    """
    Main entrypoint for running inference on customer reviews.
    """
    def __init__(
        self,
        model_dir: str = "models",
        use_external_llm: bool = False,
        openai_api_key: Optional[str] = None
    ):
        self.model_dir = model_dir
        self.sentiment_model: Optional[RevioSentimentPipeline] = None
        self.aspect_suite: Optional[RevioAspectModelSuite] = None
        self.crisis_model: Optional[RevioCrisisRiskClassifier] = None

        self._load_all_models()

        if use_external_llm and openai_api_key:
            self.response_generator = ExternalLLMResponseGenerator(api_key=openai_api_key)
        else:
            self.response_generator = LocalGuardrailedResponseGenerator()

    def _load_all_models(self):
        """Loads models if present on disk."""
        try:
            self.sentiment_model, self.aspect_suite, self.crisis_model = load_models(self.model_dir)
            self.models_loaded = True
        except FileNotFoundError:
            self.models_loaded = False

    def is_ready(self) -> bool:
        return self.models_loaded and (self.sentiment_model is not None)

    def analyze_single_review(
        self,
        review_text: str,
        tone: str = "Empathetic & Resolution-Oriented",
        business_name: str = "Cynbit Partner Store",
        location_id: str = "Jaipur Central",
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-aspect triage and response drafting for a single review.
        """
        start_time = time.perf_counter()

        # Step 1: Input Validation
        is_valid, error_msg = validate_review_input(review_text)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        if not self.is_ready():
            return {
                "success": False,
                "error": "Models are not trained or loaded. Please run 'python scripts/train.py' first.",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        cleaned = clean_text(review_text)
        ling_feats = extract_linguistic_features(review_text)

        # Step 2: Overall Sentiment Prediction & Probabilities
        classes = self.sentiment_model.get_classes()
        probs = self.sentiment_model.predict_proba([cleaned])[0]
        pred_idx = int(np.argmax(probs))
        sentiment_pred = str(classes[pred_idx])
        confidence = float(probs[pred_idx])
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

        # Step 3: Multi-Aspect Extraction
        aspect_results = self.aspect_suite.predict_aspects(cleaned)

        # Step 4: Crisis & Urgency Risk Scoring
        risk_info = self.crisis_model.predict_risk(review_text, sentiment_pred=sentiment_pred)

        # Step 5: Draft AI Response (if requested)
        response_data = None
        if generate_response:
            response_data = self.response_generator.generate(
                review_text=review_text,
                overall_sentiment=sentiment_pred,
                aspect_results=aspect_results,
                risk_info=risk_info,
                tone=tone,
                business_name=business_name
            )

        latency = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "success": True,
            "raw_review": review_text,
            "cleaned_review": cleaned,
            "location_id": location_id,
            "business_name": business_name,
            "sentiment": {
                "label": sentiment_pred,
                "confidence": round(confidence, 3),
                "probabilities": prob_dict
            },
            "aspects": aspect_results,
            "risk_assessment": risk_info,
            "linguistic_profile": ling_feats,
            "ai_response": response_data,
            "latency_ms": latency
        }

    def analyze_batch(
        self,
        reviews_df: pd.DataFrame,
        text_column: str = "review_text",
        generate_responses: bool = False
    ) -> pd.DataFrame:
        """
        Processes a collection of reviews, adding structured analytical columns.
        """
        if text_column not in reviews_df.columns:
            raise ValueError(f"Column '{text_column}' not found in provided DataFrame.")

        results = []
        for _, row in reviews_df.iterrows():
            text = str(row[text_column])
            loc = row.get("location", "Main Branch")
            biz = row.get("business_name", "Partner Brand")

            analysis = self.analyze_single_review(
                text,
                business_name=biz,
                location_id=loc,
                generate_response=generate_responses
            )

            if analysis["success"]:
                record = {
                    "review_id": row.get("id", len(results) + 1),
                    "review_text": text,
                    "location": loc,
                    "sentiment": str(analysis["sentiment"]["label"]),
                    "confidence": analysis["sentiment"]["confidence"],
                    "risk_tier": str(analysis["risk_assessment"]["risk_tier"]),
                    "risk_score": analysis["risk_assessment"]["risk_score"],
                    "active_triggers": "; ".join(analysis["risk_assessment"]["active_triggers"]),
                    "recommended_action": analysis["risk_assessment"]["recommended_action"],
                    "service_sentiment": str(analysis["aspects"]["customer_service"]["sentiment"]),
                    "quality_sentiment": str(analysis["aspects"]["product_quality"]["sentiment"]),
                    "pricing_sentiment": str(analysis["aspects"]["pricing_value"]["sentiment"]),
                    "speed_sentiment": str(analysis["aspects"]["speed_efficiency"]["sentiment"]),
                    "cleanliness_sentiment": str(analysis["aspects"]["cleanliness_ambiance"]["sentiment"]),
                    "billing_sentiment": str(analysis["aspects"]["billing_transactions"]["sentiment"]),
                    "latency_ms": analysis["latency_ms"]
                }
            else:
                record = {
                    "review_id": row.get("id", len(results) + 1),
                    "review_text": text,
                    "location": loc,
                    "sentiment": "invalid",
                    "confidence": 0.0,
                    "risk_tier": "ERROR",
                    "risk_score": 0.0,
                    "active_triggers": analysis["error"],
                    "recommended_action": "Invalid input formatting",
                    "service_sentiment": "none",
                    "quality_sentiment": "none",
                    "pricing_sentiment": "none",
                    "speed_sentiment": "none",
                    "cleanliness_sentiment": "none",
                    "billing_sentiment": "none",
                    "latency_ms": analysis["latency_ms"]
                }
            results.append(record)

        return pd.DataFrame(results)

