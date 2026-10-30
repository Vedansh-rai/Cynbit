"""
Model Evaluation and Benchmark Comparison Suite for Revio AI Engine.

Implements rigorous empirical comparison between baseline and proposed ML models,
confusion matrix calculation, per-aspect F1 breakdown, and failure case analysis.
"""

from typing import Dict, Any, List, Tuple
import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.dummy import DummyClassifier

from src.model import (
    BaselineSentimentHeuristic,
    RevioSentimentPipeline,
    RevioAspectModelSuite,
    RevioCrisisRiskClassifier
)
from src.preprocessing import clean_text, ASPECT_CATEGORIES


def evaluate_sentiment_models(
    test_df: pd.DataFrame,
    sentiment_model: RevioSentimentPipeline,
    text_col: str = "review_text",
    label_col: str = "sentiment"
) -> Dict[str, Any]:
    """
    Compares the Baseline Heuristic model against the Revio ML Sentiment Pipeline.
    """
    texts = test_df[text_col].tolist()
    y_true = test_df[label_col].tolist()

    # --- 0. Evaluate Dummy Classifier (true statistical floor) ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(texts, y_true)  # fit needed but ignored; DummyClassifier uses label distribution only
    t0 = time.perf_counter()
    y_pred_dummy = dummy.predict(texts)
    dummy_latency_ms = ((time.perf_counter() - t0) / len(texts)) * 1000

    dummy_acc = accuracy_score(y_true, y_pred_dummy)
    dummy_f1_macro = f1_score(y_true, y_pred_dummy, average="macro", zero_division=0)
    dummy_f1_weighted = f1_score(y_true, y_pred_dummy, average="weighted", zero_division=0)
    dummy_prec_macro = precision_score(y_true, y_pred_dummy, average="macro", zero_division=0)
    dummy_rec_macro = recall_score(y_true, y_pred_dummy, average="macro", zero_division=0)

    # --- 1. Evaluate Baseline Heuristic ---
    baseline = BaselineSentimentHeuristic()
    t0 = time.perf_counter()
    y_pred_base = baseline.predict(texts)
    base_latency_ms = ((time.perf_counter() - t0) / len(texts)) * 1000

    base_acc = accuracy_score(y_true, y_pred_base)
    base_f1_macro = f1_score(y_true, y_pred_base, average="macro", zero_division=0)
    base_f1_weighted = f1_score(y_true, y_pred_base, average="weighted", zero_division=0)
    base_prec_macro = precision_score(y_true, y_pred_base, average="macro", zero_division=0)
    base_rec_macro = recall_score(y_true, y_pred_base, average="macro", zero_division=0)

    # --- 2. Evaluate Revio ML Pipeline ---
    t0 = time.perf_counter()
    y_pred_ml = sentiment_model.predict(texts)
    ml_latency_ms = ((time.perf_counter() - t0) / len(texts)) * 1000

    ml_acc = accuracy_score(y_true, y_pred_ml)
    ml_f1_macro = f1_score(y_true, y_pred_ml, average="macro", zero_division=0)
    ml_f1_weighted = f1_score(y_true, y_pred_ml, average="weighted", zero_division=0)
    ml_prec_macro = precision_score(y_true, y_pred_ml, average="macro", zero_division=0)
    ml_rec_macro = recall_score(y_true, y_pred_ml, average="macro", zero_division=0)

    # Confusion Matrix for ML Model
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_true, y_pred_ml, labels=labels)

    # Error analysis: identify misclassified samples
    errors = []
    for i, (text, true_lbl, pred_lbl) in enumerate(zip(texts, y_true, y_pred_ml)):
        if true_lbl != pred_lbl and len(errors) < 8:
            errors.append({
                "index": i,
                "text": text[:120] + "..." if len(text) > 120 else text,
                "true_label": true_lbl,
                "predicted_label": pred_lbl
            })

    comparison_table = [
        {
            "Metric": "Accuracy",
            "Dummy Baseline": f"{dummy_acc * 100:.2f}%",
            "Lexicon Baseline": f"{base_acc * 100:.2f}%",
            "Revio ML Pipeline": f"{ml_acc * 100:.2f}%",
            "Lift vs Dummy": f"{(ml_acc - dummy_acc) * 100:+.2f}%"
        },
        {
            "Metric": "Macro F1-Score",
            "Dummy Baseline": f"{dummy_f1_macro * 100:.2f}%",
            "Lexicon Baseline": f"{base_f1_macro * 100:.2f}%",
            "Revio ML Pipeline": f"{ml_f1_macro * 100:.2f}%",
            "Lift vs Dummy": f"{(ml_f1_macro - dummy_f1_macro) * 100:+.2f}%"
        },
        {
            "Metric": "Weighted F1-Score",
            "Dummy Baseline": f"{dummy_f1_weighted * 100:.2f}%",
            "Lexicon Baseline": f"{base_f1_weighted * 100:.2f}%",
            "Revio ML Pipeline": f"{ml_f1_weighted * 100:.2f}%",
            "Lift vs Dummy": f"{(ml_f1_weighted - dummy_f1_weighted) * 100:+.2f}%"
        },
        {
            "Metric": "Macro Precision",
            "Dummy Baseline": f"{dummy_prec_macro * 100:.2f}%",
            "Lexicon Baseline": f"{base_prec_macro * 100:.2f}%",
            "Revio ML Pipeline": f"{ml_prec_macro * 100:.2f}%",
            "Lift vs Dummy": f"{(ml_prec_macro - dummy_prec_macro) * 100:+.2f}%"
        },
        {
            "Metric": "Macro Recall",
            "Dummy Baseline": f"{dummy_rec_macro * 100:.2f}%",
            "Lexicon Baseline": f"{base_rec_macro * 100:.2f}%",
            "Revio ML Pipeline": f"{ml_rec_macro * 100:.2f}%",
            "Lift vs Dummy": f"{(ml_rec_macro - dummy_rec_macro) * 100:+.2f}%"
        },
        {
            "Metric": "Inference Latency (per review)",
            "Dummy Baseline": f"{dummy_latency_ms:.3f} ms",
            "Lexicon Baseline": f"{base_latency_ms:.3f} ms",
            "Revio ML Pipeline": f"{ml_latency_ms:.3f} ms",
            "Lift vs Dummy": f"{ml_latency_ms - dummy_latency_ms:+.3f} ms"
        }
    ]

    return {
        "dummy_baseline": {
            "accuracy": round(dummy_acc, 4),
            "macro_f1": round(dummy_f1_macro, 4),
            "weighted_f1": round(dummy_f1_weighted, 4),
            "latency_ms": round(dummy_latency_ms, 3)
        },
        "baseline": {
            "accuracy": round(base_acc, 4),
            "macro_f1": round(base_f1_macro, 4),
            "weighted_f1": round(base_f1_weighted, 4),
            "latency_ms": round(base_latency_ms, 3)
        },
        "ml_pipeline": {
            "accuracy": round(ml_acc, 4),
            "macro_f1": round(ml_f1_macro, 4),
            "weighted_f1": round(ml_f1_weighted, 4),
            "latency_ms": round(ml_latency_ms, 3),
            "classification_report": classification_report(y_true, y_pred_ml, output_dict=True, zero_division=0)
        },
        "comparison_table": comparison_table,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist()
        },
        "sample_errors": errors
    }


def evaluate_crisis_model(
    test_df: pd.DataFrame,
    crisis_model: RevioCrisisRiskClassifier,
    text_col: str = "review_text",
    crisis_col: str = "is_crisis"
) -> Dict[str, Any]:
    """
    Evaluates the crisis and escalation risk classifier on safety-critical reviews.
    """
    texts = test_df[text_col].tolist()
    y_true = test_df[crisis_col].astype(int).tolist()

    predictions = []
    scores = []
    for t in texts:
        res = crisis_model.predict_risk(t)
        # Classify as crisis if risk tier is HIGH or CRITICAL (risk_score >= 0.45)
        is_pred_crisis = 1 if res["risk_tier"] in ["HIGH", "CRITICAL"] else 0
        predictions.append(is_pred_crisis)
        scores.append(res["risk_score"])

    prec = precision_score(y_true, predictions, zero_division=0)
    rec = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    acc = accuracy_score(y_true, predictions)

    return {
        "crisis_accuracy": round(acc, 4),
        "crisis_precision": round(prec, 4),
        "crisis_recall": round(rec, 4),
        "crisis_f1": round(f1, 4),
        "total_test_samples": len(y_true),
        "actual_crisis_count": int(sum(y_true)),
        "predicted_crisis_count": int(sum(predictions))
    }


def evaluate_aspect_suite(
    test_df: pd.DataFrame,
    aspect_suite: RevioAspectModelSuite,
    text_col: str = "review_text"
) -> Dict[str, Any]:
    """
    Evaluates classification performance across all 6 operational dimensions.
    """
    aspect_scores = {}
    for asp in ASPECT_CATEGORIES:
        col = f"aspect_{asp}"
        if col not in test_df.columns:
            continue
        y_true = test_df[col].tolist()
        texts = test_df[text_col].tolist()

        pipe = aspect_suite.aspect_pipelines[asp]
        cleaned = [clean_text(t) for t in texts]
        y_pred = pipe.predict(cleaned)

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        aspect_scores[asp] = {
            "macro_f1": round(f1, 4),
            "accuracy": round(acc, 4)
        }
    return aspect_scores

