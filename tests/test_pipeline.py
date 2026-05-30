"""
tests/test_pipeline.py - Unit tests for Skin Lesion Classifier
ZHAW AI-Applications Project

Run: pytest tests/ -v --tb=short
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Config tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_class_names_length(self):
        from src.config import CLASS_NAMES, NUM_CLASSES
        assert len(CLASS_NAMES) == 7, "HAM10000 has 7 classes"
        assert NUM_CLASSES == 7

    def test_class_names_valid(self):
        from src.config import CLASS_NAMES
        expected = {"mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"}
        assert set(CLASS_NAMES) == expected

    def test_ham10000_classes_mapping(self):
        from src.config import HAM10000_CLASSES, CLASS_NAMES
        for cls in CLASS_NAMES:
            assert cls in HAM10000_CLASSES
            assert len(HAM10000_CLASSES[cls]) > 0

    def test_cv_config_keys(self):
        from src.config import CV_CONFIG
        required = {"model_name", "pretrained", "image_size", "batch_size",
                    "num_epochs", "lr", "num_classes" if "num_classes" in CV_CONFIG else "model_path"}
        for key in ["batch_size", "lr", "image_size", "num_epochs"]:
            assert key in CV_CONFIG, f"Missing key: {key}"


# ── CV Model tests ───────────────────────────────────────────────────────────

class TestCVModel:
    def test_model_instantiation(self):
        from src.cv.model import SkinLesionResNet50
        model = SkinLesionResNet50()
        assert model is not None

    def test_forward_shape(self):
        import torch
        from src.cv.model import SkinLesionResNet50
        from src.config import NUM_CLASSES
        model = SkinLesionResNet50()
        dummy = torch.randn(2, 3, 224, 224)
        out   = model(dummy)
        assert out.shape == (2, NUM_CLASSES)

    def test_probabilities_sum_to_one(self):
        import torch
        from src.cv.model import SkinLesionResNet50
        model = SkinLesionResNet50()
        dummy = torch.randn(4, 3, 224, 224)
        probs = model.get_probabilities(dummy)
        sums  = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_feature_vector_shape(self):
        import torch
        from src.cv.model import SkinLesionResNet50
        model = SkinLesionResNet50()
        dummy = torch.randn(2, 3, 224, 224)
        feat  = model.get_feature_vector(dummy)
        assert feat.shape[0] == 2
        assert feat.shape[1] == 512

    def test_parameter_count(self):
        from src.cv.model import SkinLesionResNet50, count_parameters
        model  = SkinLesionResNet50()
        params = count_parameters(model)
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["trainable"] < params["total"], "Some layers should be frozen"


# ── Preprocessing tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_train_transforms(self):
        from src.cv.preprocessing import get_train_transforms
        import torch
        from PIL import Image
        import numpy as np
        transform = get_train_transforms()
        img    = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_val_transforms(self):
        from src.cv.preprocessing import get_val_transforms
        import torch
        from PIL import Image
        import numpy as np
        transform = get_val_transforms()
        img    = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)


# ── NLP tests ────────────────────────────────────────────────────────────────

class TestNLP:
    def test_symptom_embedder(self):
        from src.nlp.embeddings import SymptomEmbedder
        embedder = SymptomEmbedder()
        emb = embedder.encode("Dark mole on back")
        assert emb.shape[0] == 384  # all-MiniLM-L6-v2 dim

    def test_class_similarities(self):
        from src.nlp.embeddings import SymptomEmbedder
        from src.config import CLASS_NAMES
        embedder = SymptomEmbedder()
        sims = embedder.similarity_to_classes("Dark irregular growing mole")
        assert set(sims.keys()) == set(CLASS_NAMES)
        for v in sims.values():
            assert -1.0 <= v <= 1.0

    def test_feature_vector_shape(self):
        from src.nlp.embeddings import SymptomEmbedder
        embedder = SymptomEmbedder()
        vec = embedder.get_feature_vector("Itchy red bump")
        assert vec.shape[0] == 384 + 7  # embedding + 7 class sims

    def test_synthetic_data_generation(self):
        from src.nlp.embeddings import generate_synthetic_symptom_data
        from src.config import CLASS_NAMES
        df = generate_synthetic_symptom_data(n=50)
        assert len(df) == 50
        assert "symptom_text" in df.columns
        assert "true_class" in df.columns
        assert set(df["true_class"].unique()).issubset(set(CLASS_NAMES))

    def test_llm_extractor_defaults(self):
        from src.nlp.llm_extractor import ClaudeSymptomExtractor
        extractor = ClaudeSymptomExtractor.__new__(ClaudeSymptomExtractor)
        extractor._cache = {}
        defaults = extractor._get_defaults()
        assert "duration_days" in defaults
        assert defaults["duration_days"] == -1

    def test_feature_vector_conversion(self):
        from src.nlp.llm_extractor import ClaudeSymptomExtractor
        extractor = ClaudeSymptomExtractor.__new__(ClaudeSymptomExtractor)
        extractor._cache = {}
        features  = extractor._get_defaults()
        features["color"] = "brown"
        features["localization"] = "back"
        vec = extractor.to_feature_vector(features)
        assert len(vec) == 10
        assert all(isinstance(v, float) for v in vec)


# ── ML Features tests ────────────────────────────────────────────────────────

class TestMLFeatures:
    def setup_method(self):
        from src.config import CLASS_NAMES
        N = 50
        self.meta = pd.DataFrame({
            "age":              np.random.uniform(20, 80, N),
            "sex_enc":          np.random.randint(-1, 2, N),
            "localization_enc": np.random.randint(0, 15, N),
            "dx_type_enc":      np.random.randint(0, 4, N),
        })
        self.cv_probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)), N)
        self.nlp      = np.random.randn(N, 10)
        self.N        = N

    def test_feature_matrix_cv_only(self):
        from src.ml.features import build_feature_matrix
        X, names = build_feature_matrix(self.meta, self.cv_probs, self.nlp, "cv_only")
        assert X.shape == (self.N, 7)
        assert len(names) == 7

    def test_feature_matrix_metadata_only(self):
        from src.ml.features import build_feature_matrix
        X, names = build_feature_matrix(self.meta, self.cv_probs, self.nlp, "metadata_only")
        assert X.shape == (self.N, 4)

    def test_feature_matrix_all(self):
        from src.ml.features import build_feature_matrix
        X, names = build_feature_matrix(self.meta, self.cv_probs, self.nlp, "all_features")
        assert X.shape == (self.N, 7 + 4 + 10)
        assert len(names) == X.shape[1]

    def test_invalid_group_raises(self):
        from src.ml.features import build_feature_matrix
        with pytest.raises(ValueError):
            build_feature_matrix(self.meta, self.cv_probs, None, "invalid_group")


# ── Integration smoke test ───────────────────────────────────────────────────

class TestIntegration:
    def test_cv_inference_pipeline(self):
        """End-to-end CV inference with random image (no model weights needed)."""
        import torch
        from src.cv.model import SkinLesionResNet50
        from src.cv.preprocessing import get_val_transforms
        from src.config import CLASS_NAMES, HAM10000_CLASSES
        from PIL import Image
        import numpy as np

        model     = SkinLesionResNet50()
        transform = get_val_transforms()
        img       = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor    = transform(img).unsqueeze(0)
        probs     = model.get_probabilities(tensor).detach().numpy()[0]

        assert len(probs) == len(CLASS_NAMES)
        assert abs(probs.sum() - 1.0) < 1e-5
        pred_idx = int(np.argmax(probs))
        assert CLASS_NAMES[pred_idx] in HAM10000_CLASSES
