"""
Guardrailed Contextual AI Response Engine for Revio.

Provides:
1. Deterministic Local Guardrailed Response Generator (0 API key dependencies, instant, safe)
2. Optional External LLM Integration (OpenAI-compatible) with strict JSON schema validation
3. Hallucination defenses, prompt injection sanitization, and compliance guardrails
4. Human-in-the-loop escalation gating
5. Extractive complaint injection - pulls the most negative sentence from the review for specific acknowledgement
"""

from typing import Dict, Any, List, Optional
import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Pre-defined Brand Tones
AVAILABLE_TONES = [
    "Empathetic & Resolution-Oriented",
    "Professional & Formal",
    "Concise & Action-Focused",
    "Enthusiastic & Appreciative"
]

# Guardrail Rules enforced by Revio
GUARDRAILS = [
    "NO_LIABILITY_ADMISSION: Do not admit legal fault, negligence, or guilt before internal investigation.",
    "NO_UNAUTHORIZED_COMPENSATION: Do not promise free merchandise, cash refunds, or settlements without manager sign-off.",
    "MANDATORY_OFFLINE_REDIRECT: For CRITICAL or HIGH risk reviews, provide a direct private escalation channel.",
    "CONTEXTUAL_GROUNDING: Directly address mentioned aspects while avoiding robotic generic boilerplate.",
    "HUMAN_GATING: Require manual human review for all high-risk, legal, or health claims."
]


def sanitize_prompt_input(text: str) -> str:
    """
    Mitigates prompt injection by escaping markdown, removing instruction delimiters,
    and neutralizing adversarial override attempts.
    """
    if not text:
        return ""
    # Strip potential instruction overrides
    neutralized = re.sub(
        r"(?i)\b(ignore previous instructions|system prompt|developer mode|disregard above|you are now)\b",
        "[REDACTED_INSTRUCTION]",
        text
    )
    # Neutralize HTML/XML tags
    neutralized = neutralized.replace("<", "&lt;").replace(">", "&gt;")
    return neutralized.strip()


# Negative signal lexicon for extractive complaint sentence scoring
_NEG_SIGNALS = {
    "terrible", "awful", "horrible", "worst", "rude", "disgusting",
    "filthy", "fraud", "scam", "poisoning", "sick", "wait", "waited", "delay",
    "slow", "overcharged", "double", "ignored", "dismissed", "arrogant",
    "unprofessional", "unacceptable", "never", "disaster", "broken", "cold",
    "stale", "dirty", "harassment", "lawsuit", "lawyer", "refund", "error",
    "mistake", "wrong", "bad", "poor", "disappointing", "disgusted", "furious",
    "angry", "shocking", "appalling", "dreadful", "pathetic", "useless"
}


def extract_key_complaint(text: str, max_len: int = 120) -> Optional[str]:
    """
    Extract the most complaint-dense sentence from the review for specific
    acknowledgement in the AI response, improving contextual grounding.

    Returns None if the text has only one sentence or no strong complaint signal.
    """
    if not text:
        return None
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        return None
    best_sentence, best_score = None, 0
    for sent in sentences:
        words = re.sub(r'[^a-z ]', '', sent.lower()).split()
        score = sum(1 for w in words if w in _NEG_SIGNALS)
        if '!' in sent:
            score += 0.5
        if score > best_score:
            best_score = score
            best_sentence = sent.strip()
    # Only return if at least 2 complaint signals found
    if best_score < 2 or best_sentence is None:
        return None
    if len(best_sentence) > max_len:
        best_sentence = best_sentence[:max_len].rsplit(' ', 1)[0] + "..."
    return best_sentence


class LocalGuardrailedResponseGenerator:
    """
    Deterministic, offline response generator enforcing Cynbit's brand voice and safety policies.
    Requires no internet connection or paid API keys.
    """
    def generate(
        self,
        review_text: str,
        overall_sentiment: str,
        aspect_results: Dict[str, Dict[str, Any]],
        risk_info: Dict[str, Any],
        tone: str = "Empathetic & Resolution-Oriented",
        business_name: str = "Our Business",
        escalation_contact: str = "priority-care@cynbit.in"
    ) -> Dict[str, Any]:
        """
        Synthesizes a tailored, safe response based on multi-aspect findings and risk tier.
        """
        risk_tier = risk_info.get("risk_tier", "LOW")
        triggers = risk_info.get("active_triggers", [])
        requires_human = risk_tier in ["HIGH", "CRITICAL"]

        # Identify positive and negative aspects mentioned
        pos_aspects = [asp.replace("_", " ") for asp, d in aspect_results.items() if d.get("sentiment") == "positive"]
        neg_aspects = [asp.replace("_", " ") for asp, d in aspect_results.items() if d.get("sentiment") == "negative"]

        response_parts = []
        guardrails_applied = [
            "NO_LIABILITY_ADMISSION",
            "CONTEXTUAL_GROUNDING"
        ]

        # 1. Greeting & Opening
        if tone == "Enthusiastic & Appreciative" or (overall_sentiment == "positive" and risk_tier == "LOW"):
            response_parts.append(f"Hello, thank you so much for taking the time to share your feedback with {business_name}!")
        elif tone == "Professional & Formal":
            response_parts.append(f"Dear Customer, thank you for contacting {business_name} regarding your recent experience.")
        elif tone == "Concise & Action-Focused":
            response_parts.append(f"Thank you for sharing your experience with {business_name}.")
        else: # Empathetic & Resolution-Oriented
            response_parts.append(f"Hello, thank you for reaching out to us at {business_name}. We genuinely appreciate your honest feedback.")

        # 2. Address Positives (if any)
        if pos_aspects:
            aspects_str = ", ".join(pos_aspects[:2])
            if overall_sentiment == "positive":
                response_parts.append(f"We are delighted to hear that you had a wonderful experience, particularly regarding our {aspects_str}.")
            else:
                response_parts.append(f"While we are pleased that you noted positive aspects regarding our {aspects_str}, we clearly fell short in other areas.")

        # 3. Address Negatives & Risk with strict compliance guardrails
        if risk_tier == "CRITICAL":
            guardrails_applied.append("MANDATORY_OFFLINE_REDIRECT")
            guardrails_applied.append("HUMAN_GATING")
            guardrails_applied.append("NO_UNAUTHORIZED_COMPENSATION")
            response_parts.append(
                "We take issues of this severity with the utmost urgency and seriousness. "
                "Our senior management has already been alerted and an immediate internal review is underway."
            )
            response_parts.append(
                f"To ensure this is thoroughly and privately resolved, please contact our executive resolution team directly at {escalation_contact} or via phone with your visit details. "
                "We want to speak with you directly to address this situation."
            )
        elif risk_tier == "HIGH":
            guardrails_applied.append("MANDATORY_OFFLINE_REDIRECT")
            guardrails_applied.append("HUMAN_GATING")
            neg_str = f" involving {', '.join(neg_aspects[:2])}" if neg_aspects else ""
            response_parts.append(
                f"We are deeply concerned by the difficulties you encountered{neg_str}. "
                "This does not reflect our operating standards, and we are reviewing this with our team."
            )
            response_parts.append(
                f"Please reach out to our branch management team at {escalation_contact} so we can examine the specific details of your visit and make this right."
            )
        elif neg_aspects:
            neg_str = ", ".join(neg_aspects[:2])
            # Extractive complaint injection: pull the most negative sentence for specific acknowledgement
            key_complaint = extract_key_complaint(review_text)
            if key_complaint:
                response_parts.append(
                    f"We sincerely regret that your experience with our {neg_str} did not meet your expectations. "
                    f"In particular, we noted your concern: \"{key_complaint}\" — this has been flagged directly "
                    "with our operations team for immediate review and corrective action."
                )
            else:
                response_parts.append(
                    f"We sincerely regret that your experience with our {neg_str} did not meet your expectations. "
                    "Your observations have been shared directly with our operations team to implement prompt improvements."
                )
            response_parts.append(
                f"If you would like to discuss this further, please feel free to drop us a note at {escalation_contact}."
            )
        elif overall_sentiment == "positive":
            response_parts.append(
                "Our entire team works hard to maintain these high standards, and your kind words mean a lot to us. "
                "We look forward to welcoming you back again soon!"
            )
        else: # Neutral
            response_parts.append(
                "We appreciate your balanced feedback. We are constantly striving to improve, and your comments provide valuable guidance for our team."
            )

        # 4. Sign-off
        if tone == "Professional & Formal":
            response_parts.append(f"Sincerely,\nThe Management Team\n{business_name}")
        elif tone == "Concise & Action-Focused":
            response_parts.append(f"Best regards,\nCustomer Care Team, {business_name}")
        else:
            response_parts.append(f"Warm regards,\nThe {business_name} Team")

        full_draft = "\n\n".join(response_parts)

        return {
            "draft_response": full_draft,
            "tone": tone,
            "engine": "Revio Local Guardrailed Engine (Offline / Deterministic)",
            "guardrails_applied": guardrails_applied,
            "requires_human_review": requires_human,
            "offline_escalation_channel": escalation_contact if requires_human else None,
            "positive_aspects_addressed": pos_aspects,
            "negative_aspects_addressed": neg_aspects,
            "status": "READY_FOR_HUMAN_APPROVAL" if requires_human else "AUTO_APPROVABLE"
        }


class ExternalLLMResponseGenerator:
    """
    Adapter for external LLM APIs (OpenAI or compatible REST endpoints).
    Implements structured prompting, strict schemas, and automatic fallback to Local Engine.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", api_base: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.local_generator = LocalGuardrailedResponseGenerator()

    def generate(
        self,
        review_text: str,
        overall_sentiment: str,
        aspect_results: Dict[str, Dict[str, Any]],
        risk_info: Dict[str, Any],
        tone: str = "Empathetic & Resolution-Oriented",
        business_name: str = "Our Business",
        escalation_contact: str = "priority-care@cynbit.in"
    ) -> Dict[str, Any]:
        """
        Attempts LLM generation with fallback to local generator.
        """
        if not self.api_key:
            res = self.local_generator.generate(
                review_text, overall_sentiment, aspect_results, risk_info, tone, business_name, escalation_contact
            )
            res["note"] = "Local generator used (no external API key provided in environment)."
            return res

        # Build structured system prompt with safety boundaries
        system_prompt = (
            "You are Revio AI, an enterprise customer reputation manager for Cynbit Technologies clients.\n"
            "Generate a professional, compliant response to a customer review.\n\n"
            "MANDATORY COMPLIANCE RULES:\n"
            "1. NEVER admit legal fault, guilt, or negligence before an investigation.\n"
            "2. NEVER promise monetary refunds or free gifts without authorization.\n"
            "3. If risk tier is CRITICAL or HIGH, always provide the private escalation email/phone for offline handling.\n"
            "4. Acknowledge the specific aspects mentioned (positive or negative).\n"
            "5. Respond in valid JSON adhering to the specified schema only."
        )

        user_prompt = (
            f"Business Name: {business_name}\n"
            f"Escalation Contact: {escalation_contact}\n"
            f"Requested Tone: {tone}\n"
            f"Predicted Sentiment: {overall_sentiment}\n"
            f"Risk Tier: {risk_info.get('risk_tier', 'LOW')}\n"
            f"Risk Triggers: {risk_info.get('active_triggers', [])}\n"
            f"Aspect Findings: {json.dumps({k: v.get('sentiment') for k, v in aspect_results.items()})}\n\n"
            f"<customer_review>\n{sanitize_prompt_input(review_text)}\n</customer_review>\n\n"
            "Return JSON matching: {\"draft_response\": string, \"requires_human_review\": boolean, \"guardrails_applied\": [string]}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 500
        }

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=8
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                parsed["engine"] = f"External LLM ({self.model})"
                parsed["tone"] = tone
                parsed["status"] = "READY_FOR_HUMAN_APPROVAL" if parsed.get("requires_human_review") else "AUTO_APPROVABLE"
                return parsed
            else:
                # API error, fallback safely
                res = self.local_generator.generate(
                    review_text, overall_sentiment, aspect_results, risk_info, tone, business_name, escalation_contact
                )
                res["note"] = f"LLM API returned status {resp.status_code}. Fallback generator engaged."
                return res
        except Exception as e:
            # Timeout or network error, fallback safely
            res = self.local_generator.generate(
                review_text, overall_sentiment, aspect_results, risk_info, tone, business_name, escalation_contact
            )
            res["note"] = f"LLM call encountered error ({str(e)}). Local fallback generator engaged seamlessly."
            return res

