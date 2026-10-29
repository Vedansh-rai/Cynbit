"""
Utility functions, logging configuration, and preset review fixtures for Revio AI.
"""

from typing import Dict, Any, List
import logging
import os
import json

def get_logger(name: str = "revio_ai") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def format_risk_badge(risk_tier: str) -> str:
    """Returns markdown badge representation for a risk level."""
    badges = {
        "CRITICAL": "🔴 **CRITICAL RISK (IMMEDIATE ACTION)**",
        "HIGH": "🟠 **HIGH RISK (MANAGER ESCALATION)**",
        "MEDIUM": "🟡 **MEDIUM RISK (SUPPORT QUEUE)**",
        "LOW": "🟢 **LOW RISK (ROUTINE)**"
    }
    return badges.get(risk_tier, f"⚪ {risk_tier}")


# Curated review presets for live testing and demonstration
DEMO_PRESETS: List[Dict[str, Any]] = [
    {
        "title": "Severe Food Poisoning & Hospitalization (Critical Crisis)",
        "location": "Jaipur - Malviya Nagar",
        "business": "Spice & Savor Bistro",
        "industry": "Dining & Hospitality",
        "text": (
            "EMERGENCY HEALTH HAZARD!! Three of us ate the seafood platter last night and woke up with severe vomiting and violent chills. "
            "My sister had to be admitted to Fortis Hospital for acute food poisoning! The doctor confirmed bacterial contamination. "
            "When we called the restaurant, the manager was arrogant and hung up on us. We are filing a formal police complaint "
            "and reporting this to the municipal food safety authority tomorrow morning. NEVER EAT HERE!"
        )
    },
    {
        "title": "Unauthorized Credit Card Double Charge & Scam Accusation",
        "location": "Jaipur - Tonk Road",
        "business": "Urban Trend Boutique",
        "industry": "Retail & Fashion",
        "text": (
            "Total fraud and scam! I made a purchase of Rs. 4,200 yesterday and discovered your cashier swiped my card twice, charging me Rs. 8,400. "
            "When I returned with my bank statement, your cashier claimed it wasn't their problem and refused a refund. "
            "I have already called my bank to dispute the unauthorized charge and if my money is not returned within 24 hours, "
            "I am filing an FIR for credit card fraud with the cyber crime cell."
        )
    },
    {
        "title": "Mixed Experience: Superb Food but Terrible Billing & Delay",
        "location": "Jaipur - C-Scheme",
        "business": "The Grand Haveli",
        "industry": "Dining & Hospitality",
        "text": (
            "The culinary team deserves applause—the butter chicken and garlic naan were absolutely outstanding and rich in flavor. "
            "However, the operational execution was frustrating. We waited 45 minutes just to get our water glasses refilled, "
            "and when the bill finally arrived, they had charged us for two desserts we never ordered. "
            "Great kitchen, but the floor staff and checkout system need urgent retraining."
        )
    },
    {
        "title": "Glowing 5-Star Experience: Outstanding Service & Ambiance",
        "location": "Jaipur - Mansarovar",
        "business": "Aura Wellness Spa",
        "industry": "Wellness & Healthcare",
        "text": (
            "An extraordinary wellness retreat! From the moment you walk through the doors, the calming aromatherapy and serene music "
            "set a relaxing tone. Dr. Neha and therapist Rahul were exceptionally courteous, attentive, and knowledgeable. "
            "The pricing is very reasonable considering the five-star facility. Will definitely be a regular visitor here!"
        )
    },
    {
        "title": "Rude Reception Staff & Unacceptable Waiting Time",
        "location": "Jaipur - Pratap Nagar",
        "business": "Pratap Care Diagnostics",
        "industry": "Healthcare & Clinics",
        "text": (
            "Had an appointment scheduled for 10:00 AM for blood work. I arrived 15 minutes early, yet the receptionist didn't even acknowledge me "
            "for 20 minutes while chatting on her personal phone. I finally got called in at 11:40 AM! No explanation, no apology for the 100-minute delay. "
            "The lab technician was competent, but the front desk attitude is utterly dismissive and unprofessional."
        )
    },
    {
        "title": "SaaS Platform Bug & Unresponsive Technical Support",
        "location": "Bengaluru Tech Hub",
        "business": "CloudSync Enterprise",
        "industry": "B2B SaaS",
        "text": (
            "We have been paying $450/month for CloudSync, but the latest release broke our automated webhooks and halted our production billing. "
            "Submitted 3 urgent support tickets over 48 hours without a single human response—just generic automated bots. "
            "If your team cannot restore our API sync by end of business today, we are cancelling our annual contract and switching to your competitor."
        )
    }
]
