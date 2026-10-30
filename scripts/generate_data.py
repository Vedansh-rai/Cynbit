"""
Reproducible Synthetic Data Generator for Revio AI Engine.

Generates realistic, domain-specific multi-aspect customer reviews
across hospitality, healthcare, retail, SaaS, and property management.
Splits data into train (80%) and test (20%) sets with fixed random seed.
"""

import os
import random
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Generation vocabulary and sentence templates
LOCATIONS = [
    "Jaipur Central", "Jaipur - Malviya Nagar", "Jaipur - Mansarovar",
    "Delhi NCR - CyberCity", "Mumbai South", "Bengaluru Tech Park", "Pune Hinjawadi"
]

INDUSTRIES = [
    "Dining & Hospitality", "Retail & Fashion", "Healthcare & Clinics", "B2B SaaS", "Real Estate & Rentals"
]

BUSINESS_NAMES = {
    "Dining & Hospitality": ["Spice & Savor Bistro", "The Grand Haveli", "Royal Dine & Lounge", "Urban Tadka"],
    "Retail & Fashion": ["Urban Trend Boutique", "EliteVogue Apparel", "Luxe Lifestyle Store", "Aura Footwear"],
    "Healthcare & Clinics": ["Pratap Care Diagnostics", "Aura Wellness Spa", "CarePlus Polyclinic", "WizHealth Dental"],
    "B2B SaaS": ["CloudSync Enterprise", "TenantEase Cloud", "Revio Suite", "DataFlow Systems"],
    "Real Estate & Rentals": ["Skyline Residency", "Green Meadows Estates", "Urban Haven Living", "Heritage Apartments"]
}

# Snippets by Aspect and Sentiment
SERVICE_POS = [
    "The staff was incredibly polite, attentive, and helpful throughout our visit.",
    "Our server was warm, courteous, and anticipated every request with a smile.",
    "The manager personally checked in on us to ensure everything was satisfactory.",
    "Customer support responded within minutes and resolved our issue with great expertise."
]
SERVICE_NEG = [
    "The staff was shockingly rude, dismissive, and utterly unprofessional.",
    "Our waiter rolled his eyes when we asked for a modification and ignored us thereafter.",
    "The front desk receptionist was cold, arrogant, and acted like we were an inconvenience.",
    "Customer service was completely unhelpful and showed zero empathy."
]

QUALITY_POS = [
    "The food quality was exceptional, bursting with authentic fresh flavors.",
    "The product is remarkably durable, beautifully designed, and works flawlessly.",
    "The medical care was first-rate and executed with rigorous clinical standards.",
    "The software platform is lightning-fast, rock-solid, and completely bug-free."
]
QUALITY_NEG = [
    "The meal was completely bland, cold, and tasted like it was reheated from yesterday.",
    "The product broke down after two days of light use; shoddy build quality.",
    "The software crashed repeatedly and corrupted our saved project data.",
    "The diagnostics equipment looked outdated and gave inconsistent results."
]

PRICING_POS = [
    "Prices are very reasonable and provide exceptional value for money.",
    "Considering the premium standard, the rates are surprisingly affordable.",
    "Great transparent pricing with no hidden surcharges or surprise fees.",
    "The monthly subscription is worth every single dollar for the features offered."
]
PRICING_NEG = [
    "Extremely overpriced for the tiny portions and mediocre service delivered.",
    "Felt like a total rip-off with exorbitant markups on basic necessities.",
    "They added hidden service charges and mandatory fees that were never disclosed upfront.",
    "The renewal fee doubled without prior warning; terrible cost-to-value proposition."
]

SPEED_POS = [
    "Service was remarkably prompt; our order arrived in under 10 minutes.",
    "Fast turnaround time and zero waiting at the checkout counter.",
    "The team addressed our urgent ticket within twenty minutes.",
    "Quick, efficient workflow that saved us hours of operational hassle."
]
SPEED_NEG = [
    "We endured an excruciating 50-minute delay before anyone even took our order.",
    "Everything moved at a snail's pace with endless queues and zero organization.",
    "Waited over an hour past our confirmed appointment time with no explanation.",
    "Ticket resolution took over four days for a critical system outage."
]

CLEANLINESS_POS = [
    "The premises were spotlessly clean, hygienic, and wonderfully maintained.",
    "The ambiance was inviting, relaxing, and aesthetically designed.",
    "Immaculate environment with pristine seating and spotless restrooms.",
    "The UI is sleek, modern, uncluttered, and a delight to navigate."
]
CLEANLINESS_NEG = [
    "The place was filthy with sticky tables and foul odors near the seating area.",
    "Restrooms were disgusting, lacked basic hygiene supplies, and clearly had not been cleaned all day.",
    "The ambiance was ruined by deafening noise, poor lighting, and dirty floors.",
    "The interface was chaotic, cluttered, and confusing to navigate."
]

BILLING_POS = [
    "Billing was transparent, accurate, and checkout was completed in seconds.",
    "The invoice was itemized clearly and our loyalty discount was applied seamlessly.",
    "Refund was processed immediately back to my original payment card without fuss.",
    "Clean payment portal with effortless instant digital receipts."
]
BILLING_NEG = [
    "They double-charged my debit card and refused to acknowledge their POS glitch.",
    "The final bill contained mysterious extra items and erroneous surcharge fees.",
    "I have been waiting three weeks for a promised refund that has not arrived.",
    "Unauthorized transaction appeared on my statement after paying at this terminal."
]

CRISIS_TEMPLATES = [
    {
        "category": "health_safety",
        "templates": [
            "SEVERE HEALTH HAZARD!! Four of us got violent food poisoning with vomiting and fever after dining here. My child was hospitalized at Fortis last night with severe dehydration. Doctor confirmed salmonella! We are filing an urgent complaint with the Food Safety Authority.",
            "Found a dead cockroach inside my soup bowl!! When I showed the manager, he laughed it off and offered a 10% coupon. Repulsive hygiene violation! I took photos and am submitting a report to the municipal health inspector today.",
            "Allergic reaction emergency! I explicitly informed the kitchen about my severe peanut allergy three times, yet they served a sauce loaded with peanut oil. My throat closed and we had to rush to the ER. Gross negligence!"
        ]
    },
    {
        "category": "legal_threat",
        "templates": [
            "We have retained legal counsel and our lawyer is serving a formal legal notice for breach of contract and deceptive trade practices tomorrow. See you in Consumer Court.",
            "Unlawful eviction threat and harassment by the building management. Our attorney is filing an injunction and civil damages lawsuit this Friday. Expect court summons.",
            "Blatant defamation and violation of consumer rights. I have documented everything and am initiating legal action with our legal team."
        ]
    },
    {
        "category": "fraud_billing",
        "templates": [
            "FRAUD ALERT! Your cashier took my credit card behind the counter and swiped Rs. 24,000 for an unauthorized transaction. This is outright theft and cybercrime. I have blocked my card and registered an FIR with the police.",
            "Scam company! They automatically renewed a subscription we canceled three months ago and billed $1,200 without permission. Bank is disputing this fraudulent charge and fraud investigators are notified.",
            "Double-billed Rs. 15,000 and the accounts manager hung up on me when I demanded a reversal. Pure extortion and criminal fraud!"
        ]
    },
    {
        "category": "discrimination_harassment",
        "templates": [
            "Horrific experience with overt racial discrimination by the floor supervisor. He hurled derogatory slurs and told my family people like us were not welcome. We are reporting this hate incident to the authorities and human rights commission.",
            "Female staff member was verbally harassed and physically intimidated by the shift manager. This is an unsafe workplace and customer environment. HR was notified and a police report is underway.",
            "Blatant discrimination based on appearance. The bouncer made degrading comments and refused entry despite confirmed reservations. Utterly abhorrent bigotry."
        ]
    },
    {
        "category": "severe_churn",
        "templates": [
            "WORST DISASTER OF MY ENTIRE LIFE. Absolutely ruined our anniversary evening. I am cancelling our corporate membership, demanding an immediate full refund, and telling everyone in our network to boycott this place.",
            "A complete dumpster fire. You just lost our enterprise contract of $85k/year. We are migrating our entire team to your rival this afternoon. Good riddance.",
            "I have never encountered such gross incompetence. We will never set foot in any of your branches again and will warn everyone online."
        ]
    }
]

NEUTRAL_SNIPPETS = [
    "Overall an average experience, nothing standout but nothing terrible.",
    "The visit was adequate. Met basic expectations but didn't leave a strong impression.",
    "A standard, routine experience. Some things were fine, others could be improved.",
    "Decent enough for a quick stop, though prices are slightly high for what you receive.",
    "It gets the job done, but don't expect anything extraordinary."
]


def generate_synthetic_review(idx: int, target_sentiment: str, force_crisis: bool = False) -> dict:
    """
    Assembles a coherent, realistic review combining relevant aspect sentences.
    """
    industry = random.choice(INDUSTRIES)
    biz = random.choice(BUSINESS_NAMES[industry])
    loc = random.choice(LOCATIONS)

    aspect_ratings = {
        "aspect_customer_service": "not_mentioned",
        "aspect_product_quality": "not_mentioned",
        "aspect_pricing_value": "not_mentioned",
        "aspect_speed_efficiency": "not_mentioned",
        "aspect_cleanliness_ambiance": "not_mentioned",
        "aspect_billing_transactions": "not_mentioned"
    }

    body_sentences = []
    is_crisis = 1 if force_crisis else 0
    crisis_cat = "none"

    if force_crisis:
        target_sentiment = "negative"
        crisis_obj = random.choice(CRISIS_TEMPLATES)
        crisis_cat = crisis_obj["category"]
        body_sentences.append(random.choice(crisis_obj["templates"]))
        # Add supplementary operational context
        if crisis_cat == "health_safety":
            aspect_ratings["aspect_product_quality"] = "negative"
            aspect_ratings["aspect_cleanliness_ambiance"] = "negative"
        elif crisis_cat == "fraud_billing":
            aspect_ratings["aspect_billing_transactions"] = "negative"
        elif crisis_cat in ["legal_threat", "discrimination_harassment"]:
            aspect_ratings["aspect_customer_service"] = "negative"
        elif crisis_cat == "severe_churn":
            aspect_ratings["aspect_customer_service"] = "negative"
            aspect_ratings["aspect_speed_efficiency"] = "negative"

    elif target_sentiment == "positive":
        # Pick 2 to 4 positive aspects
        selected_aspects = random.sample([
            ("aspect_customer_service", SERVICE_POS, "positive"),
            ("aspect_product_quality", QUALITY_POS, "positive"),
            ("aspect_pricing_value", PRICING_POS, "positive"),
            ("aspect_speed_efficiency", SPEED_POS, "positive"),
            ("aspect_cleanliness_ambiance", CLEANLINESS_POS, "positive"),
            ("aspect_billing_transactions", BILLING_POS, "positive")
        ], k=random.randint(2, 4))

        for asp_name, pool, sent in selected_aspects:
            body_sentences.append(random.choice(pool))
            aspect_ratings[asp_name] = sent

    elif target_sentiment == "negative":
        # Pick 2 to 4 negative aspects (sometimes with 1 positive concession)
        selected_aspects = random.sample([
            ("aspect_customer_service", SERVICE_NEG, "negative"),
            ("aspect_product_quality", QUALITY_NEG, "negative"),
            ("aspect_pricing_value", PRICING_NEG, "negative"),
            ("aspect_speed_efficiency", SPEED_NEG, "negative"),
            ("aspect_cleanliness_ambiance", CLEANLINESS_NEG, "negative"),
            ("aspect_billing_transactions", BILLING_NEG, "negative")
        ], k=random.randint(2, 3))

        for asp_name, pool, sent in selected_aspects:
            body_sentences.append(random.choice(pool))
            aspect_ratings[asp_name] = sent

        # Concession: e.g. "Food was good but staff was rude"
        if random.random() < 0.35:
            concession_pool = random.choice([
                ("aspect_product_quality", QUALITY_POS, "positive"),
                ("aspect_cleanliness_ambiance", CLEANLINESS_POS, "positive")
            ])
            if aspect_ratings[concession_pool[0]] == "not_mentioned":
                body_sentences.insert(0, random.choice(concession_pool[1]))
                aspect_ratings[concession_pool[0]] = "positive"

    else: # Neutral
        body_sentences.append(random.choice(NEUTRAL_SNIPPETS))
        # Add 1 mild positive and 1 mild negative
        if random.random() < 0.6:
            body_sentences.append(random.choice(QUALITY_POS))
            aspect_ratings["aspect_product_quality"] = "positive"
            body_sentences.append(random.choice(SPEED_NEG))
            aspect_ratings["aspect_speed_efficiency"] = "negative"

    review_text = " ".join(body_sentences)

    return {
        "id": idx,
        "review_text": review_text,
        "sentiment": target_sentiment,
        "is_crisis": is_crisis,
        "crisis_category": crisis_cat,
        "industry": industry,
        "business_name": biz,
        "location": loc,
        **aspect_ratings
    }


def generate_dataset(num_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """Generates balanced synthetic review dataset."""
    random.seed(random_seed)
    records = []

    # Target distribution:
    # 40% Positive, 35% Negative (including 15% Crisis), 25% Neutral
    n_positive = int(num_samples * 0.40)
    n_neutral = int(num_samples * 0.25)
    n_crisis = int(num_samples * 0.15)
    n_negative_std = num_samples - (n_positive + n_neutral + n_crisis)

    curr_id = 1
    for _ in range(n_positive):
        records.append(generate_synthetic_review(curr_id, "positive", force_crisis=False))
        curr_id += 1

    for _ in range(n_neutral):
        records.append(generate_synthetic_review(curr_id, "neutral", force_crisis=False))
        curr_id += 1

    for _ in range(n_negative_std):
        records.append(generate_synthetic_review(curr_id, "negative", force_crisis=False))
        curr_id += 1

    for _ in range(n_crisis):
        records.append(generate_synthetic_review(curr_id, "negative", force_crisis=True))
        curr_id += 1

    random.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = i + 1

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer review dataset for Revio.")
    parser.add_argument("--samples", type=int, default=1200, help="Total number of reviews to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory path.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {args.samples} realistic customer reviews (seed={args.seed})...")
    df = generate_dataset(num_samples=args.samples, random_seed=args.seed)

    # Stratified 80/20 train/test split based on sentiment and crisis flag
    strat_col = df["sentiment"] + "_" + df["is_crisis"].astype(str)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=args.seed, stratify=strat_col)

    train_path = os.path.join(args.output_dir, "train_reviews.csv")
    test_path = os.path.join(args.output_dir, "test_reviews.csv")
    full_path = os.path.join(args.output_dir, "all_reviews.csv")

    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Successfully saved:")
    print(f" - Full dataset: {full_path} ({len(df)} rows)")
    print(f" - Train set:    {train_path} ({len(train_df)} rows)")
    print(f" - Test set:     {test_path} ({len(test_df)} rows)")
    print(f"\nClass balance in Train:")
    print(train_df["sentiment"].value_counts(normalize=True).round(3))
    print(f"\nCrisis ratio in Train: {train_df['is_crisis'].mean():.2%}")


if __name__ == "__main__":
    main()

