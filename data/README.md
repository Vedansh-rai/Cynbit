# Revio Dataset Documentation & Production Data Strategy

## 1. Overview and Purpose
This directory contains the dataset used to train, validate, and benchmark the **Revio AI Multi-Aspect Sentiment Intelligence & Crisis Escalation Engine** for Cynbit Technologies.

In enterprise reputation management, customer feedback spans multiple operational dimensions (service, quality, price, speed, ambiance, billing) and occasionally contains severe legal, health, or fraud hazards. Because client proprietary review logs and private CRM records are confidential and unavailable, this proof of concept utilizes a **rigorously structured, reproducible synthetic dataset** generated via `scripts/generate_data.py`.

> [!IMPORTANT]
> **Data Transparency Notice**:
> The datasets in this directory (`all_reviews.csv`, `train_reviews.csv`, `test_reviews.csv`) are synthetic. They demonstrate functional capability, algorithmic stability, and evaluation methodology. Performance metrics derived from this dataset demonstrate the model's architectural validity but do not represent actual historical business performance metrics of Cynbit Technologies or its clients.

---

## 2. Generation Assumptions & Methodology
The data generation pipeline (`scripts/generate_data.py`) was constructed using the following domain-informed specifications:

1. **Multi-Domain Representation**:
   Samples span five industry verticals aligned with Cynbit's documented client portfolio:
   - **Dining & Hospitality** (e.g., Spice & Savor Bistro, The Grand Haveli)
   - **Retail & Fashion** (e.g., Urban Trend Boutique, EliteVogue Apparel)
   - **Healthcare & Clinics** (e.g., Pratap Care Diagnostics, CarePlus Polyclinic)
   - **B2B SaaS** (e.g., CloudSync Enterprise, Revio Suite, TenantEase Cloud)
   - **Real Estate & Property Management** (e.g., Skyline Residency, Green Meadows Estates)

2. **Stratification & Class Balance**:
   - **Total Records**: 1,200
   - **Sentiment Distribution**: 40% Positive, 35% Negative, 25% Neutral
   - **Crisis Escalation Ratio**: 15% of all samples (180 total records) are designated safety/compliance crises.
   - **Train/Test Split**: 80% Train (960 samples) and 20% Test (240 samples), stratified jointly on `sentiment` and `is_crisis`.
   - **Random Seed**: Fixed at `seed=42` for 100% deterministic reproducibility.

3. **Multi-Aspect Coverage**:
   Reviews simulate authentic human expression where multiple aspects are mentioned concurrently:
   - `aspect_customer_service`: Staff demeanor, courtesy, attentiveness, competence.
   - `aspect_product_quality`: Core offering quality, freshness, defects, reliability.
   - `aspect_pricing_value`: Cost-to-value perception, affordability, transparency.
   - `aspect_speed_efficiency`: Wait times, delivery speed, response latency.
   - `aspect_cleanliness_ambiance`: Physical hygiene, atmosphere, digital UI/UX.
   - `aspect_billing_transactions`: Invoice accuracy, POS swipes, refunds, charges.

4. **Crisis and Escalation Categories**:
   - `health_safety`: Severe food poisoning, bacterial contamination, hospitalizations, allergen negligence, pest infestations.
   - `legal_threat`: Retention of legal counsel, court notices, consumer forum filings.
   - `fraud_billing`: Credit card double swipes, unauthorized charges, scam accusations.
   - `discrimination_harassment`: Verbal abuse, discrimination, workplace safety violations.
   - `severe_churn`: Immediate contract cancellation, high ARR enterprise boycotts.

---

## 3. Schema Definition
Each CSV file contains the following schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique review identifier. |
| `review_text` | String | Raw text of customer review. |
| `sentiment` | String | Ground-truth overall sentiment (`positive`, `neutral`, `negative`). |
| `is_crisis` | Integer (0/1) | Binary indicator if review requires immediate emergency escalation. |
| `crisis_category` | String | Specific hazard classification (`none`, `health_safety`, `legal_threat`, `fraud_billing`, `discrimination_harassment`, `severe_churn`). |
| `industry` | String | Domain sector. |
| `business_name` | String | Entity name. |
| `location` | String | Physical branch or digital region (e.g., Jaipur Central, Bengaluru Tech Park). |
| `aspect_customer_service` | String | Aspect sentiment (`positive`, `negative`, `neutral`, `not_mentioned`). |
| `aspect_product_quality` | String | Aspect sentiment (`positive`, `negative`, `neutral`, `not_mentioned`). |
| `aspect_pricing_value` | String | Aspect sentiment (`positive`, `negative`, `neutral`, `not_mentioned`). |
| `aspect_speed_efficiency` | String | Aspect sentiment (`positive`, `negative`, `neutral`, `not_mentioned`). |
| `aspect_cleanliness_ambiance` | String | Aspect sentiment (`positive`, `negative`, `neutral`, `not_mentioned`). |
| `aspect_billing_transactions` | String | Aspect sentiment (`positive`, `negative`, `neutral`, `not_mentioned`). |

---

## 4. Production Data Migration Strategy
When deploying Revio for live enterprise clients, synthetic data will be substituted through the following staged ingestion pipeline:

```
[Google My Business / Yelp / Trustpilot API Webhooks]
                          │
                          ▼
            [Raw Data Ingestion & Sanitization]
      (PII Masking, Unicode Normalization, Spam Filter)
                          │
                          ▼
           [Active Learning & Weak Supervision]
     (Snorkel / LLM-assisted Silver Labeling for ABSA)
                          │
                          ▼
          [Human-in-the-Loop Gold Annotation]
      (Domain Managers validate low-confidence reviews)
                          │
                          ▼
         [Automated Retraining & Model Registry]
         (Continuous Model Evaluation & Deployment)
```

### Key Production Steps:
1. **API Webhook Ingestion**: Ingest live feeds via Google My Business API, Yelp Fusion API, and Trustpilot Webhooks.
2. **PII Redaction**: Run regex and Named Entity Recognition (NER) to redact credit card numbers, phone numbers, and customer names before model persistence.
3. **Active Learning Loop**: Route low-confidence model inferences (< 0.65 probability) to branch managers for confirmation, continuously expanding the labeled gold-standard corpus.
4. **Data Versioning**: Store raw and labeled snapshots in DVC (Data Version Control) with automated drift detection.

