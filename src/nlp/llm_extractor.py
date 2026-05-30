"""
src/nlp/llm_extractor.py - Approach B: Claude API structured symptom extraction
ZHAW AI-Applications Project
"""
import json
import re
from pathlib import Path
import sys
from typing import Dict, Optional, List
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import NLP_CONFIG, ANTHROPIC_API_KEY


# ─── Extraction schema ─────────────────────────────────────────────────────────
EXTRACTION_SCHEMA = {
    "duration_days": {
        "type": "number",
        "description": "Duration of lesion in days (estimate if vague: 'weeks'=14, 'months'=60, 'years'=365)",
        "default": -1
    },
    "color": {
        "type": "string",
        "description": "Primary color(s) of lesion",
        "options": ["black", "brown", "pink", "red", "white", "blue", "mixed", "unknown"],
        "default": "unknown"
    },
    "size_mm": {
        "type": "number",
        "description": "Estimated size in mm (-1 if not mentioned)",
        "default": -1
    },
    "pain_level": {
        "type": "number",
        "description": "Pain level 0-10 (0=no pain, -1=not mentioned)",
        "default": -1
    },
    "localization": {
        "type": "string",
        "description": "Body location",
        "options": ["face", "scalp", "neck", "chest", "back", "abdomen", "upper extremity",
                    "lower extremity", "hand", "foot", "genital", "oral mucosa", "acral", "unknown"],
        "default": "unknown"
    },
    "itching": {
        "type": "number",
        "description": "Itching: 1=yes, 0=no, -1=not mentioned",
        "default": -1
    },
    "bleeding": {
        "type": "number",
        "description": "Bleeding: 1=yes, 0=no, -1=not mentioned",
        "default": -1
    },
    "change_rate": {
        "type": "string",
        "description": "How fast is it changing?",
        "options": ["stable", "slow", "fast", "sudden", "unknown"],
        "default": "unknown"
    },
    "asymmetry": {
        "type": "number",
        "description": "Asymmetric shape: 1=yes, 0=no, -1=not mentioned",
        "default": -1
    },
    "border_irregularity": {
        "type": "number",
        "description": "Irregular borders: 1=yes, 0=no, -1=not mentioned",
        "default": -1
    },
}

SYSTEM_PROMPT = """You are a medical NLP assistant helping extract structured information from skin lesion symptom descriptions.

Extract ONLY the following fields from the patient's description. Return a valid JSON object with EXACTLY these keys.
Do not add medical diagnoses or recommendations. If information is not mentioned, use the default value.

Required fields and their defaults:
- duration_days: number (-1 if unknown)
- color: one of [black, brown, pink, red, white, blue, mixed, unknown]
- size_mm: number (-1 if unknown)
- pain_level: 0-10 (-1 if not mentioned)
- localization: one of [face, scalp, neck, chest, back, abdomen, upper extremity, lower extremity, hand, foot, genital, oral mucosa, acral, unknown]
- itching: 1/0/-1
- bleeding: 1/0/-1
- change_rate: one of [stable, slow, fast, sudden, unknown]
- asymmetry: 1/0/-1
- border_irregularity: 1/0/-1

Return ONLY valid JSON, no explanation."""

USER_TEMPLATE = """Extract structured features from this skin lesion description:

"{text}"

Return JSON only."""


class ClaudeSymptomExtractor:
    """
    Approach B: Use Claude API for structured feature extraction from
    free-text symptom descriptions.

    Compares to Approach A (sentence transformers) in terms of:
    - Extraction quality
    - Downstream ML performance
    - Cost and latency
    """

    def __init__(
        self,
        api_key: str = ANTHROPIC_API_KEY,
        model: str = NLP_CONFIG["anthropic_model"],
        max_tokens: int = NLP_CONFIG["max_tokens"],
    ):
        import anthropic
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to .env file."
            )
        self.client     = anthropic.Anthropic(api_key=api_key)
        self.model      = model
        self.max_tokens = max_tokens
        self._cache     = {}

    def extract(self, text: str, use_cache: bool = True) -> Dict:
        """
        Extract structured symptom features using Claude API.

        Args:
            text: Free-text symptom description
            use_cache: Cache results to avoid duplicate API calls

        Returns:
            dict with extracted features (see EXTRACTION_SCHEMA)
        """
        if use_cache and text in self._cache:
            return self._cache[text]

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_TEMPLATE.format(text=text)}
            ],
        )

        response_text = message.content[0].text.strip()

        # Parse JSON (handle markdown code blocks)
        json_match = re.search(r"\{[\s\S]+\}", response_text)
        if json_match:
            try:
                features = json.loads(json_match.group())
            except json.JSONDecodeError:
                features = self._get_defaults()
        else:
            features = self._get_defaults()

        # Fill defaults for missing keys
        for key, schema in EXTRACTION_SCHEMA.items():
            if key not in features:
                features[key] = schema["default"]

        if use_cache:
            self._cache[text] = features

        return features

    def extract_batch(
        self,
        texts: List[str],
        delay: float = 0.1,
    ) -> List[Dict]:
        """Extract features from a list of texts with rate limiting."""
        results = []
        for i, text in enumerate(texts):
            result = self.extract(text)
            results.append(result)
            if delay > 0 and i < len(texts) - 1:
                time.sleep(delay)
        return results

    def _get_defaults(self) -> Dict:
        """Return default values for all fields."""
        return {k: v["default"] for k, v in EXTRACTION_SCHEMA.items()}

    def to_feature_vector(self, features: Dict) -> list:
        """
        Convert extracted features dict to numeric vector for ML.
        Categorical features are encoded numerically.
        """
        color_map = {"black": 0, "brown": 1, "pink": 2, "red": 3,
                     "white": 4, "blue": 5, "mixed": 6, "unknown": -1}
        loc_map   = {"face": 0, "scalp": 1, "neck": 2, "chest": 3,
                     "back": 4, "abdomen": 5, "upper extremity": 6,
                     "lower extremity": 7, "hand": 8, "foot": 9,
                     "genital": 10, "oral mucosa": 11, "acral": 12, "unknown": -1}
        rate_map  = {"stable": 0, "slow": 1, "fast": 2, "sudden": 3, "unknown": -1}

        return [
            float(features.get("duration_days", -1)),
            float(color_map.get(features.get("color", "unknown"), -1)),
            float(features.get("size_mm", -1)),
            float(features.get("pain_level", -1)),
            float(loc_map.get(features.get("localization", "unknown"), -1)),
            float(features.get("itching", -1)),
            float(features.get("bleeding", -1)),
            float(rate_map.get(features.get("change_rate", "unknown"), -1)),
            float(features.get("asymmetry", -1)),
            float(features.get("border_irregularity", -1)),
        ]

    def generate_explanation(
        self,
        symptom_text: str,
        cv_prediction: Dict,
        ml_prediction: Dict,
        disclaimer: str = "",
    ) -> str:
        """
        Generate plain-language explanation of the combined prediction
        using Claude API.

        NOTE: Always includes disclaimer - this is NOT medical advice.
        """
        context = f"""
Patient symptom description: {symptom_text}

Computer Vision model prediction: {cv_prediction.get('label_name', 'Unknown')} 
(confidence: {cv_prediction.get('confidence', 0):.1%})

ML ensemble prediction: {ml_prediction.get('label_name', 'Unknown')}
(risk score: {ml_prediction.get('risk_score', 0):.2f})

Top CV probabilities:
{json.dumps(cv_prediction.get('probabilities', {}), indent=2)}
""".strip()

        system_msg = """You are a medical AI assistant explaining skin lesion analysis results to a non-medical user.

CRITICAL RULES:
1. Always add a disclaimer that this is NOT medical advice
2. Explain the results in simple language
3. NEVER make a definitive diagnosis  
4. Always recommend seeing a dermatologist
5. Keep response under 200 words
6. Be empathetic and clear"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system_msg,
            messages=[{"role": "user", "content": context}],
        )
        explanation = message.content[0].text
        if disclaimer:
            explanation = f"{disclaimer}\n\n{explanation}"
        return explanation


if __name__ == "__main__":
    extractor = ClaudeSymptomExtractor()
    test_text = "I have a dark irregular mole on my back that has been growing for the past 3 months. It has uneven edges and multiple shades of brown and black. It does not hurt but sometimes itches."
    features  = extractor.extract(test_text)
    print("Extracted features:")
    for k, v in features.items():
        print(f"  {k:25s}: {v}")
    vec = extractor.to_feature_vector(features)
    print(f"\nFeature vector ({len(vec)}d): {vec}")
