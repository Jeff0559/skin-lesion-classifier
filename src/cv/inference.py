"""
src/cv/inference.py - Inference pipeline for trained ResNet50 model
ZHAW AI-Applications Project
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from typing import Union, Tuple, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CV_CONFIG, CLASS_NAMES, HAM10000_CLASSES, DEVICE
from src.cv.model import SkinLesionResNet50, load_model
from src.cv.preprocessing import get_inference_transforms


class SkinLesionInference:
    """
    High-level inference interface for the skin lesion classifier.

    Usage:
        predictor = SkinLesionInference("models/resnet50_best.pth")
        result = predictor.predict("path/to/image.jpg")
    """

    def __init__(
        self,
        model_path: str = CV_CONFIG["best_model"],
        device: str = DEVICE,
        top_k: int = 3,
    ):
        self.device    = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.top_k     = top_k
        self.transform = get_inference_transforms()
        self.model     = load_model(model_path, device=self.device)
        print(f"[+] Model loaded from {model_path} on {self.device}")

    def preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Load and preprocess a single image."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        return_features: bool = False,
    ) -> Dict:
        """
        Predict skin lesion class for a single image.

        Returns:
            {
                "label":       predicted class name (str)
                "label_name":  human-readable name (str)
                "confidence":  confidence score (float)
                "probabilities": {class_name: prob} for all 7 classes
                "top_k":       [(class, prob)] sorted by prob
                "features":    512-dim np.ndarray (if return_features=True)
            }
        """
        tensor = self.preprocess(image)
        probs  = self.model.get_probabilities(tensor).cpu().numpy()[0]

        if return_features:
            features = self.model.get_feature_vector(tensor).cpu().numpy()[0]
        else:
            features = None

        pred_idx   = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]

        top_k_indices = np.argsort(probs)[::-1][:self.top_k]
        top_k_results = [
            (CLASS_NAMES[i], HAM10000_CLASSES[CLASS_NAMES[i]], float(probs[i]))
            for i in top_k_indices
        ]

        result = {
            "label":         pred_label,
            "label_name":    HAM10000_CLASSES[pred_label],
            "confidence":    float(probs[pred_idx]),
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            },
            "top_k": top_k_results,
        }
        if return_features:
            result["features"] = features

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        return_features: bool = False,
    ) -> List[Dict]:
        """Predict for a batch of images."""
        tensors = torch.cat([self.preprocess(img) for img in images], dim=0)
        probs   = self.model.get_probabilities(tensors).cpu().numpy()

        results = []
        for i, p in enumerate(probs):
            pred_idx = int(np.argmax(p))
            pred_lbl = CLASS_NAMES[pred_idx]
            results.append({
                "label":      pred_lbl,
                "label_name": HAM10000_CLASSES[pred_lbl],
                "confidence": float(p[pred_idx]),
                "probabilities": {CLASS_NAMES[j]: float(p[j]) for j in range(len(CLASS_NAMES))},
            })
        return results

    def get_cv_features(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (probabilities_7d, features_512d) for use in ML block.
        """
        tensor   = self.preprocess(image)
        probs    = self.model.get_probabilities(tensor).cpu().numpy()[0]
        features = self.model.get_feature_vector(tensor).cpu().numpy()[0]
        return probs, features


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.cv.inference <image_path>")
        sys.exit(1)

    predictor = SkinLesionInference()
    result    = predictor.predict(sys.argv[1])
    print(f"Predicted: {result['label_name']} ({result['label']})")
    print(f"Confidence: {result['confidence']:.3f}")
    print("Top-3 predictions:")
    for cls, name, prob in result["top_k"]:
        print(f"  {name:30s} ({cls:5s}): {prob:.3f}")
