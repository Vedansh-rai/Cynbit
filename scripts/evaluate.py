"""
Evaluation and Benchmark Script for Revio AI Engine.

Runs comprehensive validation on the held-out test set (data/test_reviews.csv),
comparing the Baseline Heuristic with Revio's trained ML models.
Outputs exact experimental metrics, confusion matrices, and error analysis.
"""

import os
import sys
import json
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
# Formatting handled via pandas DataFrame strings

from src.model import load_models
from src.evaluation import (
    evaluate_sentiment_models,
    evaluate_crisis_model,
    evaluate_aspect_suite
)
from src.utils import get_logger

logger = get_logger("evaluate")


def run_evaluation(
    test_path: str = "data/test_reviews.csv",
    model_dir: str = "models",
    output_json: str = "reports/evaluation_results.json"
):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset '{test_path}' not found. Run scripts/generate_data.py first.")

    logger.info(f"Loading held-out test data from '{test_path}'...")
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(test_df)} test samples.")

    logger.info(f"Loading trained models from '{model_dir}'...")
    sentiment_model, aspect_suite, crisis_model = load_models(model_dir)

    print("\n" + "=" * 70)
    print("      REVIO AI ENGINE — EXPERIMENTAL EVALUATION REPORT")
    print("=" * 70)

    # 1. Sentiment Pipeline vs Baseline Heuristic
    logger.info("Evaluating Overall Sentiment Pipeline against Lexicon Baseline...")
    sent_results = evaluate_sentiment_models(test_df, sentiment_model)

    print("\n[BENCHMARK 1] OVERALL SENTIMENT: BASELINE VS REVIO ML PIPELINE")
    print("-" * 70)
    comp_df = pd.DataFrame(sent_results["comparison_table"])
    print(comp_df.to_string(index=False))

    print("\n[CONFUSION MATRIX] Revio Sentiment Pipeline (Test Set)")
    print(f"Labels: {sent_results['confusion_matrix']['labels']}")
    for row, lbl in zip(sent_results['confusion_matrix']['matrix'], sent_results['confusion_matrix']['labels']):
        print(f"True {lbl:8s}: {row}")

    # 2. Crisis & Escalation Risk Model
    logger.info("Evaluating Crisis & Escalation Risk Classifier...")
    crisis_results = evaluate_crisis_model(test_df, crisis_model)

    print("\n[BENCHMARK 2] CRISIS & ESCALATION DETECTION (SAFETY-CRITICAL)")
    print("-" * 70)
    print(f"Total Test Samples:        {crisis_results['total_test_samples']}")
    print(f"Ground Truth Crisis Count: {crisis_results['actual_crisis_count']}")
    print(f"Predicted Crisis Count:    {crisis_results['predicted_crisis_count']}")
    print(f"Crisis Detection Recall:   {crisis_results['crisis_recall'] * 100:.2f}% (Crucial Safety Metric)")
    print(f"Crisis Precision:          {crisis_results['crisis_precision'] * 100:.2f}%")
    print(f"Crisis F1-Score:           {crisis_results['crisis_f1'] * 100:.2f}%")
    print(f"Overall Accuracy:          {crisis_results['crisis_accuracy'] * 100:.2f}%")

    # 3. Multi-Aspect Model Suite
    logger.info("Evaluating Multi-Aspect Models across 6 Dimensions...")
    aspect_results = evaluate_aspect_suite(test_df, aspect_suite)

    print("\n[BENCHMARK 3] MULTI-ASPECT SENTIMENT ANALYSIS (ABSA)")
    print("-" * 70)
    aspect_rows = []
    for asp, metrics in aspect_results.items():
        aspect_rows.append({
            "Aspect Category": asp.replace("_", " ").title(),
            "Macro F1-Score": f"{metrics['macro_f1'] * 100:.2f}%",
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%"
        })
    print(pd.DataFrame(aspect_rows).to_string(index=False))

    # 4. Error Analysis
    print("\n[ERROR ANALYSIS] Sample Misclassifications on Test Set")
    print("-" * 70)
    for err in sent_results["sample_errors"][:4]:
        print(f"Review: \"{err['text']}\"")
        print(f" -> True: {err['true_label']} | Predicted: {err['predicted_label']}\n")

    # Save structured results to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    full_output = {
        "test_dataset_size": len(test_df),
        "sentiment_benchmark": sent_results,
        "crisis_benchmark": crisis_results,
        "aspect_benchmark": aspect_results
    }
    with open(output_json, "w") as f:
        json.dump(full_output, f, indent=2)

    logger.info(f"Evaluation metrics successfully saved to '{output_json}'.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Revio AI models on test set.")
    parser.add_argument("--test_data", type=str, default="data/test_reviews.csv", help="Path to test CSV.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory containing trained models.")
    parser.add_argument("--output_json", type=str, default="reports/evaluation_results.json", help="Path to output JSON.")
    args = parser.parse_args()

    run_evaluation(test_path=args.test_data, model_dir=args.model_dir, output_json=args.output_json)


if __name__ == "__main__":
    main()
