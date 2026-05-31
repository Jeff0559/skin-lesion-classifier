"""
src/pipeline.py - End-to-end inference pipeline
ZHAW AI-Applications Project

Flow:
    Image + Symptom Text + Metadata
    -> CV: ResNet50 -> 7 class probs + confidence
    -> NLP: Symptom extraction -> structured features
    -> ML: Ensemble -> risk score + final prediction
    -> LLM: Claude API -> explanation with DISCLAIMER
"""
from pathlib import Path
from typing import Optional, Union, Dict
import numpy as np
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CV_CONFIG, NLP_CONFIG, ML_CONFIG, APP_CONFIG,
    CLASS_NAMES, HAM10000_CLASSES, ANTHROPIC_API_KEY
)


class SkinLesionPipeline:
    """
    Unified inference pipeline combining CV + NLP + ML.

    Usage:
        pipeline = SkinLesionPipeline()
        result   = pipeline.predict(
            image="path/to/image.jpg",
            symptom_text="Dark irregular mole on my back",
            metadata={"age": 45, "sex": "male", "localization": "back"}
        )
    """

    def __init__(
        self,
        cv_model_path: str = CV_CONFIG["best_model"],
        ml_model_path: Optional[str] = None,
        device: str = "cpu",
        use_llm: bool = True,
    ):
        self.device   = device
        self.use_llm  = use_llm
        self._cv      = None
        self._nlp     = None
        self._llm     = None
        self._ml      = None

        # Lazy-load models to avoid memory issues
        self._cv_path = cv_model_path
        self._ml_path = ml_model_path

    def _load_cv(self):
        if self._cv is None:
            from src.cv.inference import SkinLesionInference
            self._cv = SkinLesionInference(self._cv_path, device=self.device)

    def _load_nlp(self):
        if self._nlp is None:
            from src.nlp.embeddings import SymptomEmbedder
            self._nlp = SymptomEmbedder()

    def _load_llm(self):
        if self._llm is None and self.use_llm and ANTHROPIC_API_KEY:
            from src.nlp.llm_extractor import ClaudeSymptomExtractor
            self._llm = ClaudeSymptomExtractor()

    def _load_ml(self):
        if self._ml is None and self._ml_path and Path(self._ml_path).exists():
            import joblib
            self._ml = joblib.load(self._ml_path)

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        symptom_text: str = "",
        metadata: Optional[Dict] = None,
        generate_explanation: bool = True,
    ) -> Dict:
        """
        Full pipeline prediction.

        Args:
            image:               PIL Image or path
            symptom_text:        free-text symptom description
            metadata:            dict with age, sex, localization
            generate_explanation: whether to call Claude for explanation

        Returns:
            {
              "cv":          CV prediction dict
              "nlp":         NLP features dict
              "ml":          ML prediction dict
              "explanation": LLM explanation string
              "disclaimer":  disclaimer text
              "final_label": final predicted class
              "risk_score":  float 0-1
            }
        """
        result = {"disclaimer": APP_CONFIG["disclaimer"]}

        # ── Step 1: CV Inference ───────────────────────────────────────────────
        self._load_cv()
        cv_result = self._cv.predict(image, return_features=True)
        result["cv"] = cv_result

        # ── Step 2: NLP Feature Extraction ────────────────────────────────────
        nlp_features = {}
        if symptom_text:
            if self.use_llm and ANTHROPIC_API_KEY:
                self._load_llm()
                nlp_features = self._llm.extract(symptom_text)
                nlp_vector   = self._llm.to_feature_vector(nlp_features)
            else:
                self._load_nlp()
                sim = self._nlp.similarity_to_classes(symptom_text)
                nlp_features = {"class_similarities": sim}
                nlp_vector   = [sim.get(c, 0.0) for c in CLASS_NAMES] + [0.0] * 3
        else:
            nlp_vector = [0.0] * 10
        result["nlp"] = nlp_features

        # ── Step 3: ML Ensemble ───────────────────────────────────────────────
        cv_probs   = np.array([cv_result["probabilities"][c] for c in CLASS_NAMES])
        meta       = metadata or {}
        meta_vec   = [
            float(meta.get("age", 0)),
            float({"male": 1, "female": 0}.get(meta.get("sex", ""), -1)),
            float(meta.get("localization_enc", -1)),
            float(meta.get("dx_type_enc", -1)),
        ]
        feature_vec = np.concatenate([cv_probs, meta_vec, nlp_vector[:10]]).reshape(1, -1)

        self._load_ml()
        if self._ml is not None:
            ml_probs  = self._ml.predict_proba(feature_vec)[0]
            ml_pred   = int(self._ml.predict(feature_vec)[0])
            risk_score = float(np.max(ml_probs))
            result["ml"] = {
                "label":      CLASS_NAMES[ml_pred],
                "label_name": HAM10000_CLASSES[CLASS_NAMES[ml_pred]],
                "risk_score": risk_score,
                "probabilities": {CLASS_NAMES[i]: float(ml_probs[i]) for i in range(len(CLASS_NAMES))},
            }
            result["final_label"] = CLASS_NAMES[ml_pred]
            result["risk_score"]  = risk_score
        else:
            # Fallback to CV prediction
            result["ml"] = {}
            result["final_label"] = cv_result["label"]
            result["risk_score"]  = cv_result["confidence"]

        # ── Step 4: LLM Explanation ───────────────────────────────────────────
        if generate_explanation and self.use_llm and ANTHROPIC_API_KEY and symptom_text:
            self._load_llm()
            explanation = self._llm.generate_explanation(
                symptom_text=symptom_text,
                cv_prediction=cv_result,
                ml_prediction=result.get("ml", {}),
                disclaimer=APP_CONFIG["disclaimer"],
            )
            result["explanation"] = explanation
        else:
            result["explanation"] = (
                f"{APP_CONFIG['disclaimer']}\n\n"
                f"CV Model: {HAM10000_CLASSES[cv_result['label']]} "
                f"(confidence: {cv_result['confidence']:.1%})\n"
                f"Top diagnosis: {cv_result['top_k'][0][1]}"
            )

        return result


if __name__ == "__main__":
    print("Testing pipeline with dummy data...")
    import os
    from dotenv import load_dotenv
    load_dotenv()

    pipe = SkinLesionPipeline(use_llm=bool(ANTHROPIC_API_KEY))
    print("Pipeline created. Load CV model and run predict() with a real image.")
    print(f"ANTHROPIC_API_KEY set: {bool(ANTHROPIC_API_KEY)}")
