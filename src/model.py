"""
Model Architecture and Training Pipelines for Revio AI Engine.

Implements:
1. Overall Sentiment Pipeline (TF-IDF + Calibrated Logistic Regression)
2. Aspect-Specific Multi-Task Classifier (TF-IDF + Binary/Multi-class classifiers per aspect)
3. Urgency & Crisis Risk Estimator (Hybrid Statistical ML + Punctuation/Linguistic signals)
4. Heuristic Baseline Model (for comparative benchmark in Phase 5)
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.preprocessing import (
    ASPECT_CATEGORIES,
    clean_text,
    extract_linguistic_features,
    detect_crisis_triggers
)

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class BaselineSentimentHeuristic:
    """
    A transparent, rule-based lexicon baseline for benchmark comparison.
    Uses positive/negative word matching and negation awareness.
    """
    def __init__(self):
        self.pos_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", "loved", "best",
            "delicious", "fast", "friendly", "helpful", "clean", "fresh", "superb", "awesome",
            "perfect", "prompt", "polite", "courteous", "reasonable", "affordable", "impressed"
        }
        self.neg_words = {
            "bad", "terrible", "horrible", "awful", "worst", "poor", "slow", "rude", "cold",
            "dirty", "stale", "expensive", "overpriced", "disaster", "disappointed", "unacceptable",
            "hate", "scam", "waste", "filthy", "arrogant", "broken", "unprofessional", "sick"
        }
        self.negations = {"not", "never", "no", "hardly", "barely", "neither", "nor"}

    def predict_one(self, text: str) -> Dict[str, Any]:
        words = clean_text(text).lower().split()
        score = 0
        negate = False

        for i, word in enumerate(words):
            if word in self.negations:
                negate = True
                continue

            delta = 0
            if word in self.pos_words:
                delta = -1 if negate else 1
            elif word in self.neg_words:
                delta = 1 if negate else -1

            score += delta
            # Negation decays after 2 words
            if negate and i > 0 and words[i - 1] not in self.negations:
                negate = False

        if score > 0:
            pred = "positive"
            conf = min(0.5 + score * 0.1, 0.95)
        elif score < 0:
            pred = "negative"
            conf = min(0.5 + abs(score) * 0.1, 0.95)
        else:
            pred = "neutral"
            conf = 0.50

        return {
            "prediction": pred,
            "confidence": round(conf, 3),
            "raw_score": score
        }

    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t)["prediction"] for t in texts]


class RevioSentimentPipeline:
    """
    Production-grade text classification pipeline for overall review sentiment.
    Uses sublinear TF-IDF word & character n-grams with a Calibrated Logistic Regression classifier.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                min_df=1,
                strip_accents="unicode"
            )),
            ("clf", LogisticRegression(
                C=1.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            ))
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        cleaned = [clean_text(t) for t in texts]
        self.pipeline.fit(cleaned, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        return self.pipeline.predict(cleaned)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        return self.pipeline.predict_proba(cleaned)

    def get_classes(self) -> List[str]:
        return list(self.pipeline.named_steps["clf"].classes_)


class RevioAspectModelSuite:
    """
    Multi-aspect sentiment suite for fine-grained operational diagnosis.
    Contains specialized binary/multiclass pipelines for each aspect.
    """
    def __init__(self, aspects: Optional[List[str]] = None):
        self.aspects = aspects or ASPECT_CATEGORIES
        self.aspect_pipelines: Dict[str, Pipeline] = {}
        for asp in self.aspects:
            self.aspect_pipelines[asp] = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                    sublinear_tf=True,
                    min_df=1
                )),
                ("clf", LogisticRegression(
                    C=1.2,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                ))
            ])
        self.is_fitted = False

    def fit(self, texts: List[str], aspect_labels: Dict[str, List[str]]):
        cleaned = [clean_text(t) for t in texts]
        for asp in self.aspects:
            if asp in aspect_labels:
                self.aspect_pipelines[asp].fit(cleaned, aspect_labels[asp])
        self.is_fitted = True
        return self

    def predict_aspects(self, text: str) -> Dict[str, Dict[str, Any]]:
        cleaned = clean_text(text)
        results = {}
        for asp in self.aspects:
            pipe = self.aspect_pipelines[asp]
            pred = pipe.predict([cleaned])[0]
            probs = pipe.predict_proba([cleaned])[0]
            classes = list(pipe.named_steps["clf"].classes_)
            conf = float(np.max(probs))

            # If the model predicts 'not_mentioned', present cleanly
            is_present = (pred != "not_mentioned")
            sentiment = pred if is_present else "none"

            results[asp] = {
                "present": is_present,
                "sentiment": sentiment,
                "confidence": round(conf, 3),
                "probabilities": {c: round(float(p), 3) for c, p in zip(classes, probs)}
            }
        return results


class RevioCrisisRiskClassifier:
    """
    Evaluates brand and compliance risk, predicting an urgency score from 0.0 to 1.0.
    Fuses statistical text classification with linguistic urgency indicators and trigger terms.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=4000,
                sublinear_tf=True,
                min_df=1
            )),
            ("clf", LogisticRegression(
                C=2.0,
                class_weight={0: 1.0, 1: 3.0}, # Emphasize high recall on crisis instances
                max_iter=1000,
                random_state=42
            ))
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], crisis_flags: List[int]):
        cleaned = [clean_text(t) for t in texts]
        self.pipeline.fit(cleaned, crisis_flags)
        self.is_fitted = True
        return self

    def predict_risk(self, text: str, sentiment_pred: str = "neutral") -> Dict[str, Any]:
        cleaned = clean_text(text)
        ling_feats = extract_linguistic_features(text)
        triggers = detect_crisis_triggers(text)

        # Statistical ML model probability of crisis
        if self.is_fitted:
            ml_proba = float(self.pipeline.predict_proba([cleaned])[0][1])
        else:
            ml_proba = 0.2

        # Punctuation / uppercase urgency boost
        ling_boost = 0.0
        if ling_feats["all_caps_ratio"] > 0.20:
            ling_boost += 0.12
        if ling_feats["exclamation_count"] >= 3:
            ling_boost += 0.08
        if ling_feats["has_repeated_punctuation"]:
            ling_boost += 0.05

        # Explicit trigger keywords boost (critical safety nets)
        trigger_boost = 0.0
        active_triggers = []
        for cat, hits in triggers.items():
            active_triggers.append(f"{cat}: {', '.join(hits)}")
            if cat in ["legal_threat", "health_safety", "discrimination_harassment"]:
                trigger_boost += 0.35
            elif cat in ["fraud_billing"]:
                trigger_boost += 0.25
            elif cat in ["severe_churn"]:
                trigger_boost += 0.15

        # Fuse components
        raw_risk = ml_proba * 0.55 + trigger_boost * 0.35 + ling_boost * 0.10
        if "legal_threat" in triggers or "health_safety" in triggers:
            raw_risk = max(raw_risk, 0.85)
        elif "discrimination_harassment" in triggers or "fraud_billing" in triggers:
            raw_risk = max(raw_risk, 0.75)
        elif "severe_churn" in triggers:
            raw_risk = max(raw_risk, 0.55)

        if sentiment_pred == "positive" and not triggers:
            raw_risk *= 0.25 # Dampen risk for overwhelmingly positive reviews without safety triggers

        calibrated_risk = float(np.clip(raw_risk, 0.0, 1.0))

        # Assign risk tier based on calibrated risk score
        if calibrated_risk >= 0.70:
            tier = "CRITICAL"
            action = "Immediate Escalation: Notify Legal/Executive Ops and Branch Director within 1 hour."
        elif calibrated_risk >= 0.45:
            tier = "HIGH"
            action = "Priority Review: Flag for General Manager contact within 4 hours."
        elif calibrated_risk >= 0.25:
            tier = "MEDIUM"
            action = "Standard Triage: Assign to Customer Support team for 24-hour turnaround."
        else:
            tier = "LOW"
            action = "Routine Feedback: Standard automated or semi-automated engagement."

        return {
            "risk_score": round(calibrated_risk, 3),
            "risk_tier": tier,
            "recommended_action": action,
            "active_triggers": active_triggers,
            "ml_probability": round(ml_proba, 3),
            "linguistic_indicators": ling_feats
        }


def save_models(
    sentiment_model: RevioSentimentPipeline,
    aspect_suite: RevioAspectModelSuite,
    crisis_model: RevioCrisisRiskClassifier,
    save_dir: str = "models"
):
    """Serializes all trained models to disk."""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(sentiment_model, os.path.join(save_dir, "sentiment_pipeline.joblib"))
    joblib.dump(aspect_suite, os.path.join(save_dir, "aspect_suite.joblib"))
    joblib.dump(crisis_model, os.path.join(save_dir, "crisis_classifier.joblib"))


def load_models(model_dir: str = "models") -> Tuple[RevioSentimentPipeline, RevioAspectModelSuite, RevioCrisisRiskClassifier]:
    """Loads serialized models from disk."""
    sent_path = os.path.join(model_dir, "sentiment_pipeline.joblib")
    asp_path = os.path.join(model_dir, "aspect_suite.joblib")
    crisis_path = os.path.join(model_dir, "crisis_classifier.joblib")

    if not (os.path.exists(sent_path) and os.path.exists(asp_path) and os.path.exists(crisis_path)):
        raise FileNotFoundError(f"Model artifacts not found in '{model_dir}'. Run training first.")

    sentiment_model = joblib.load(sent_path)
    aspect_suite = joblib.load(asp_path)
    crisis_model = joblib.load(crisis_path)
    return sentiment_model, aspect_suite, crisis_model
