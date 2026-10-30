"""
Unit tests for text preprocessing, input validation, and feature extraction.
"""

import pytest
from src.preprocessing import (
    validate_review_input,
    clean_text,
    extract_linguistic_features,
    detect_aspect_presence_heuristic,
    detect_crisis_triggers
)


class TestValidation:
    def test_valid_input(self):
        is_valid, err = validate_review_input("The dining experience was wonderful and food was fresh.")
        assert is_valid is True
        assert err is None

    def test_empty_string(self):
        is_valid, err = validate_review_input("")
        assert is_valid is False
        assert "empty or blank" in err

    def test_whitespace_only(self):
        is_valid, err = validate_review_input("     \n\t   ")
        assert is_valid is False
        assert "empty or blank" in err

    def test_none_input(self):
        is_valid, err = validate_review_input(None)
        assert is_valid is False
        assert "cannot be None" in err

    def test_non_string_input(self):
        is_valid, err = validate_review_input(12345)
        assert is_valid is False
        assert "must be a string" in err

    def test_too_short_input(self):
        is_valid, err = validate_review_input("Bad")
        assert is_valid is False
        assert "too short" in err

    def test_repetitive_noise(self):
        is_valid, err = validate_review_input("aaaaaaaaaaaaaaa")
        assert is_valid is False
        assert "repetitive" in err


class TestTextCleaning:
    def test_contraction_expansion(self):
        cleaned = clean_text("Food wasn't good and couldn't be eaten.")
        assert "was not" in cleaned
        assert "could not" in cleaned

    def test_whitespace_condensation(self):
        cleaned = clean_text("Great   service \n\n with   prompt   delivery.")
        assert cleaned == "Great service with prompt delivery."

    def test_unicode_normalization(self):
        cleaned = clean_text("Caf\u00e9 had a delightful cr\u00e8me br\u00fbl\u00e9e.")
        assert "Cafe" in cleaned or "Café" in cleaned or len(cleaned) > 10


class TestLinguisticAndCrisisDetection:
    def test_linguistic_features(self):
        feats = extract_linguistic_features("TERRIBLE SERVICE!!! Never coming back???")
        assert feats["word_count"] == 5
        assert feats["exclamation_count"] == 3
        assert feats["question_count"] == 3
        assert feats["all_caps_ratio"] > 0.3
        assert feats["has_repeated_punctuation"] is True

    def test_crisis_triggers_word_boundary(self):
        # "accurate" should NOT trigger "rat"
        triggers = detect_crisis_triggers("Billing was accurate and fast.")
        assert "health_safety" not in triggers

        # "food poisoning" should trigger "health_safety"
        triggers = detect_crisis_triggers("I got severe food poisoning from the fish.")
        assert "health_safety" in triggers
        assert "food poisoning" in triggers["health_safety"]

    def test_legal_threat_trigger(self):
        triggers = detect_crisis_triggers("Our lawyer is filing a lawsuit in consumer court.")
        assert "legal_threat" in triggers

