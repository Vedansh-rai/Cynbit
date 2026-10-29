"""
Text Preprocessing and Input Validation Module for Revio AI Engine.

Handles text sanitization, normalization, negation preservation,
linguistic feature extraction, and defensive input validation.
"""

from typing import Tuple, List, Dict, Any, Optional
import re
import unicodedata

# Operational Aspects tracked by Revio
ASPECT_CATEGORIES = [
    "customer_service",
    "product_quality",
    "pricing_value",
    "speed_efficiency",
    "cleanliness_ambiance",
    "billing_transactions"
]

# Lexical dictionaries for rule-based aspect association
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "customer_service": [
        "staff", "waiter", "waitress", "server", "manager", "receptionist",
        "employee", "attitude", "rude", "polite", "friendly", "helpful",
        "courteous", "unprofessional", "host", "agent", "support", "treated"
    ],
    "product_quality": [
        "food", "dish", "meal", "taste", "flavor", "delicious", "fresh",
        "stale", "cold", "cooked", "ingredient", "bland", "overcooked",
        "product", "item", "material", "quality", "durable", "broken", "defective"
    ],
    "pricing_value": [
        "price", "cost", "expensive", "cheap", "worth", "value", "overpriced",
        "bill", "rate", "money", "rip-off", "charge", "affordable", "deal", "steep"
    ],
    "speed_efficiency": [
        "slow", "fast", "quick", "delay", "waited", "waiting", "time", "hour",
        "minute", "prompt", "speed", "late", "rush", "queue", "line", "forever"
    ],
    "cleanliness_ambiance": [
        "clean", "dirty", "filthy", "hygiene", "smell", "odor", "decor",
        "atmosphere", "ambiance", "noise", "loud", "music", "lighting", "restroom",
        "bathroom", "table", "cozy", "crowded", "environment"
    ],
    "billing_transactions": [
        "card", "credit", "debit", "receipt", "overcharge", "refund", "charge",
        "payment", "checkout", "pos", "invoice", "double-billed", "tip", "transaction"
    ]
}

# High-risk crisis terms indicating legal, health, fraud, or discrimination risks
CRISIS_INDICATORS: Dict[str, List[str]] = {
    "health_safety": [
        "food poisoning", "poisoned", "hospital", "vomiting", "sick", "nausea",
        "diarrhea", "infestation", "cockroach", "rat", "bug in food", "glass in food",
        "allergic reaction", "severe injury", "expired"
    ],
    "legal_threat": [
        "lawyer", "attorney", "sue", "lawsuit", "legal action", "court",
        "consumer protection", "consumer court", "reporting to police", "police report"
    ],
    "fraud_billing": [
        "credit card stolen", "stolen card", "fraud", "unauthorized charge",
        "stole my money", "scam", "scammed", "double charged deliberately"
    ],
    "discrimination_harassment": [
        "racist", "racism", "harassment", "harassed", "discriminated",
        "assault", "abused", "sexual harassment", "slur"
    ],
    "severe_churn": [
        "never coming back", "boycott", "tell everyone to avoid", "worst experience of my life",
        "warn everybody", "disaster"
    ]
}


def validate_review_input(text: Any, min_length: int = 5, max_length: int = 4000) -> Tuple[bool, Optional[str]]:
    """
    Defensively validates review input before processing.

    Returns:
        (is_valid, error_message): Tuple indicating validation status and reason if invalid.
    """
    if text is None:
        return False, "Review text cannot be None."
    if not isinstance(text, str):
        return False, f"Review text must be a string, got {type(text).__name__}."

    stripped = text.strip()
    if not stripped:
        return False, "Review text cannot be empty or blank whitespace."

    if len(stripped) < min_length:
        return False, f"Review text too short ({len(stripped)} chars). Must be at least {min_length} characters."

    if len(stripped) > max_length:
        return False, f"Review text exceeds maximum allowed length of {max_length} characters."

    # Check for non-informative gibberish (e.g. "asdfghjk" or single repeating char)
    if len(set(stripped.lower())) <= 2 and len(stripped) > 8:
        return False, "Input appears to be repetitive character noise without semantic meaning."

    return True, None


def clean_text(text: str) -> str:
    """
    Standardizes review text:
    - Normalizes Unicode characters.
    - Preserves negation structures ('not', 'never', 'n\\'t').
    - Removes control characters and excess whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode (NFKD)
    normalized = unicodedata.normalize("NFKD", text)

    # Standardize common contractions for negation preservation
    normalized = re.sub(r"\bcan't\b", "can not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bwon't\b", "will not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdon't\b", "do not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdidn't\b", "did not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bisn't\b", "is not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\baren't\b", "are not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bwasn't\b", "was not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bweren't\b", "were not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bhaven't\b", "have not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bhasn't\b", "has not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bhadn't\b", "had not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcouldn't\b", "could not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bwouldn't\b", "would not", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bshouldn't\b", "should not", normalized, flags=re.IGNORECASE)

    # Replace newlines/tabs with space
    cleaned = re.sub(r"[\r\n\t]+", " ", normalized)

    # Remove non-printable characters
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch)[0] != "C")

    # Condense multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """
    Extracts high-level stylistic and punctuation features useful for urgency and emotion scoring.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "all_caps_ratio": 0.0,
            "has_repeated_punctuation": False,
        }

    words = text.split()
    word_count = len(words)
    exclamations = text.count("!")
    questions = text.count("?")

    # Count uppercase words (excluding single letters like 'I')
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_ratio = len(caps_words) / max(word_count, 1)

    has_repeated = bool(re.search(r"(!{2,}|\?{2,}|\.{3,})", text))

    return {
        "char_count": len(text),
        "word_count": word_count,
        "exclamation_count": exclamations,
        "question_count": questions,
        "all_caps_ratio": round(caps_ratio, 4),
        "has_repeated_punctuation": has_repeated,
    }


def detect_aspect_presence_heuristic(text: str) -> Dict[str, bool]:
    """
    Heuristic / lexicon-based aspect detector for baseline comparison.
    """
    lower = text.lower()
    results = {}
    for aspect, kws in ASPECT_KEYWORDS.items():
        found = any(re.search(r"\b" + re.escape(kw) + r"\b", lower) for kw in kws)
        results[aspect] = found
    return results


def detect_crisis_triggers(text: str) -> Dict[str, List[str]]:
    """
    Scans text for critical escalation phrases across risk dimensions using word boundaries.
    Returns matched categories and terms.
    """
    lower = text.lower()
    matched = {}
    for category, terms in CRISIS_INDICATORS.items():
        hits = []
        for term in terms:
            # Multi-word phrase or single word with boundaries
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, lower):
                hits.append(term)
        if hits:
            matched[category] = hits
    return matched
