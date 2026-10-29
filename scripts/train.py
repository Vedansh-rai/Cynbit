"""
Model Training Script for Revio AI Engine.

Trains:
1. RevioSentimentPipeline (Overall Sentiment Classifier)
2. RevioAspectModelSuite (Multi-Aspect Classifiers across 6 operational dimensions)
3. RevioCrisisRiskClassifier (Crisis & Escalation Risk Estimator)

Saves artifacts to models/ directory.
"""

import os
import sys
import time
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.metrics import classification_report

from src.model import (
    RevioSentimentPipeline,
    RevioAspectModelSuite,
    RevioCrisisRiskClassifier,
    save_models
)
from src.preprocessing import ASPECT_CATEGORIES
from src.utils import get_logger

logger = get_logger("train")


def train(data_path: str = "data/train_reviews.csv", model_dir: str = "models"):
    """Loads training data, fits all pipelines, and serializes artifacts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset '{data_path}' not found. Run scripts/generate_data.py first.")

    start_time = time.perf_counter()
    logger.info(f"Loading training data from '{data_path}'...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} training examples.")

    texts = df["review_text"].astype(str).tolist()
    sentiments = df["sentiment"].tolist()
    crisis_flags = df["is_crisis"].astype(int).tolist()

    # 1. Train Overall Sentiment Pipeline
    logger.info("Training Overall Sentiment Pipeline (TF-IDF + Calibrated Logistic Regression)...")
    sentiment_model = RevioSentimentPipeline()
    sentiment_model.fit(texts, sentiments)
    train_sent_preds = sentiment_model.predict(texts)
    logger.info("Sentiment Model Training Complete.")

    # 2. Train Multi-Aspect Model Suite
    logger.info("Training Multi-Aspect Model Suite across 6 operational dimensions...")
    aspect_labels = {}
    for asp in ASPECT_CATEGORIES:
        col = f"aspect_{asp}"
        if col in df.columns:
            aspect_labels[asp] = df[col].tolist()
        else:
            logger.warning(f"Aspect column '{col}' not found in dataframe.")

    aspect_suite = RevioAspectModelSuite(aspects=ASPECT_CATEGORIES)
    aspect_suite.fit(texts, aspect_labels)
    logger.info("Aspect Suite Training Complete.")

    # 3. Train Crisis & Urgency Risk Classifier
    logger.info("Training Crisis & Urgency Risk Classifier...")
    crisis_model = RevioCrisisRiskClassifier()
    crisis_model.fit(texts, crisis_flags)
    logger.info("Crisis Risk Model Training Complete.")

    # 4. Serialize Model Artifacts
    logger.info(f"Serializing all models to '{model_dir}'...")
    save_models(sentiment_model, aspect_suite, crisis_model, save_dir=model_dir)

    total_time = time.perf_counter() - start_time
    logger.info(f"Model training and serialization completed successfully in {total_time:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Train Revio AI ML pipelines.")
    parser.add_argument("--data", type=str, default="data/train_reviews.csv", help="Path to training CSV.")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models.")
    args = parser.parse_args()

    train(data_path=args.data, model_dir=args.output_dir)


if __name__ == "__main__":
    main()
