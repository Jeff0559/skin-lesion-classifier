"""
src/nlp/explainer.py - Natural language explanation generation via Claude API
ZHAW AI-Applications Project

Generates patient-friendly explanations of model predictions with DISCLAIMER.
"""
import json
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import NLP_CONFIG, ANTHROPIC_API_KEY, APP_CONFIG, HAM10000_CLASSES, CLASS_NAMES


EXPLANATION_SYSTEM_PROMPT = """You are a medical AI assistant explaining skin lesion analysis results.

CRITICAL RULES:
1. ALWAYS start with: "WICHTIG: Dies ist eine KI-Analyse zu Bildungszwecken, KEINE medizinische Diagnose."
2. Explain the AI model findings in simple, clear language (German or English based on user input)
3. NEVER make a definitive medical diagnosis
4. ALWAYS recommend: "Bitte konsultieren Sie einen Dermatologen."
5. Mention top-2 possibilities with their probabilities
6. Explain what ABCD criteria (Asymmetry, Border, Color, Diameter) mean
7. Keep response under 250 words
8. Be empathetic, not alarmist
"""


class LesionExplainer:
    """
    Generates human-readable explanations using Claude API.
    Always includes mandatory medical disclaimer.
    """

    def __init__(
        self,
        api_key: str = ANTHROPIC_API_KEY,
        model: str = NLP_CONFIG["anthropic_model"],
    ):
        import anthropic
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def explain(
        self,
        cv_result: Dict,
        nlp_features: Optional[Dict] = None,
        ml_result: Optional[Dict] = None,
        symptom_text: str = "",
        language: str = "de",
    ) -> str:
        """
        Generate explanation for combined model output.

        Args:
            cv_result:     CV model prediction dict
            nlp_features:  extracted NLP features (optional)
            ml_result:     ML ensemble result (optional)
            symptom_text:  original user symptom text
            language:      "de" for German, "en" for English

        Returns:
            Explanation string with mandatory disclaimer
        """
        top_k = cv_result.get("top_k", [])[:3]
        top_k_str = "\n".join(
            [f"  - {name} ({cls}): {prob:.1%}" for cls, name, prob in top_k]
        )

        nlp_str = ""
        if nlp_features and isinstance(nlp_features, dict):
            relevant = {k: v for k, v in nlp_features.items()
                        if v not in (-1, "unknown", "", None)}
            if relevant:
                nlp_str = "Extracted symptoms: " + ", ".join(
                    [f"{k}={v}" for k, v in list(relevant.items())[:5]]
                )

        ml_str = ""
        if ml_result and "label_name" in ml_result:
            ml_str = (f"Ensemble model: {ml_result['label_name']} "
                      f"(risk score: {ml_result.get('risk_score', 0):.2f})")

        prompt = f"""Analyse der Hautläsion:

Bildanalyse (ResNet50) - Top Diagnosen:
{top_k_str}

{nlp_str}
{ml_str}
{"Symptombeschreibung: " + symptom_text if symptom_text else ""}

Sprache der Erklärung: {"Deutsch" if language == "de" else "English"}

Bitte erkläre diese Ergebnisse verständlich für den Patienten.
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=EXPLANATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        explanation = message.content[0].text
        # Ensure disclaimer is always present
        if APP_CONFIG["disclaimer"] not in explanation:
            explanation = f"{APP_CONFIG['disclaimer']}\n\n{explanation}"
        return explanation

    def explain_abcd(
        self,
        asymmetry: float,
        border: float,
        color: str,
        diameter_mm: float,
    ) -> str:
        """Generate ABCD rule explanation."""
        score = 0
        if asymmetry > 0:
            score += 1
        if border > 0:
            score += 1
        if color in ("mixed", "black"):
            score += 1
        if diameter_mm > 6:
            score += 1

        risk = ["Low", "Low-Medium", "Medium", "High-Medium", "High"][min(score, 4)]

        return (
            f"ABCD-Analyse:\n"
            f"  Asymmetrie: {'Ja' if asymmetry > 0 else 'Nein'}\n"
            f"  Rand (Border): {'Unregelmässig' if border > 0 else 'Regelmässig'}\n"
            f"  Farbe: {color}\n"
            f"  Durchmesser: {diameter_mm}mm (Risiko ab 6mm)\n"
            f"  ABCD-Score: {score}/4 -> {risk} Risk\n"
            f"\n{APP_CONFIG['disclaimer']}"
        )


def get_fallback_explanation(cv_result: Dict, disclaimer: str = "") -> str:
    """Fallback explanation without API (for when ANTHROPIC_API_KEY is not set)."""
    top_k   = cv_result.get("top_k", [])[:2]
    top_str = " and ".join([f"{name} ({prob:.0%})" for _, name, prob in top_k])
    return (
        f"{disclaimer}\n\n"
        f"The AI image analysis suggests: {top_str}.\n"
        f"Confidence: {cv_result.get('confidence', 0):.1%}\n\n"
        f"Please consult a dermatologist for proper diagnosis. "
        f"This is an educational tool only."
    )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not ANTHROPIC_API_KEY:
        print("[!] No ANTHROPIC_API_KEY in .env - showing fallback explanation")
        dummy_cv = {
            "label": "nv", "label_name": "Melanocytic nevi",
            "confidence": 0.72,
            "top_k": [("nv","Melanocytic nevi",0.72), ("mel","Melanoma",0.18), ("bkl","Benign keratosis",0.06)],
        }
        print(get_fallback_explanation(dummy_cv, APP_CONFIG["disclaimer"]))
    else:
        explainer = LesionExplainer()
        dummy_cv  = {
            "label": "mel", "label_name": "Melanoma",
            "confidence": 0.68,
            "top_k": [("mel","Melanoma",0.68), ("nv","Melanocytic nevi",0.22), ("bkl","Benign keratosis",0.06)],
        }
        explanation = explainer.explain(
            cv_result=dummy_cv,
            symptom_text="Dark growing mole on back",
        )
        print(explanation)
