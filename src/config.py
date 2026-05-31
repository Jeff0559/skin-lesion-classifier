"""
src/config.py - Central configuration for Skin Lesion Classifier
ZHAW AI-Applications Project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR    = DATA_DIR / "processed"
MODELS_DIR  = ROOT_DIR / "models"
LOGS_DIR    = ROOT_DIR / "logs"

for d in [RAW_DIR, INTERIM_DIR, PROC_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── HAM10000 ─────────────────────────────────────────────────────────────────
HAM10000_CLASSES = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular lesions",
}
CLASS_NAMES = list(HAM10000_CLASSES.keys())
NUM_CLASSES = len(CLASS_NAMES)

# ─── Computer Vision ──────────────────────────────────────────────────────────
CV_CONFIG = {
    "model_name":     "resnet50",
    "pretrained":     True,
    "image_size":     224,
    "batch_size":     32,
    "num_epochs":     20,
    "lr":             1e-4,
    "weight_decay":   1e-4,
    "dropout":        0.4,
    "num_workers":    4,
    "pin_memory":     True,
    "val_split":      0.15,
    "test_split":     0.15,
    "random_state":   42,
    "model_path":     str(MODELS_DIR / "resnet50_ham10000.pth"),
    "best_model":     str(MODELS_DIR / "resnet50_best.pth"),
}

# ─── NLP ──────────────────────────────────────────────────────────────────────
NLP_CONFIG = {
    "sentence_model":   "all-MiniLM-L6-v2",
    "embeddings_path":  str(PROC_DIR / "symptom_embeddings.npy"),
    "anthropic_model":  "claude-sonnet-4-20250514",
    "max_tokens":       512,
    "features": [
        "duration_days", "color", "size_mm",
        "pain_level", "localization", "itching",
        "bleeding", "change_rate",
    ],
}

# ─── ML Ensemble ─────────────────────────────────────────────────────────────
ML_CONFIG = {
    "random_state": 42,
    "cv_folds":     5,
    "test_size":    0.2,
    "models": {
        "logistic_regression": {
            "C": 1.0, "max_iter": 1000, "multi_class": "multinomial"
        },
        "random_forest": {
            "n_estimators": 200, "max_depth": 10,
            "min_samples_split": 5, "n_jobs": -1
        },
        "xgboost": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "eval_metric": "mlogloss", "use_label_encoder": False
        },
    },
    "feature_groups": {
        "cv_only":       ["cv_prob_0","cv_prob_1","cv_prob_2","cv_prob_3","cv_prob_4","cv_prob_5","cv_prob_6"],
        "metadata_only": ["age","sex_enc","localization_enc","dx_type_enc"],
        "nlp_only":      ["duration_days","color_enc","size_mm","pain_level","itching","bleeding","change_rate"],
        "cv_meta":       None,  # cv_only + metadata_only (populated at runtime)
        "all_features":  None,  # all groups combined
    },
}

# ─── App ──────────────────────────────────────────────────────────────────────
APP_CONFIG = {
    "host":       os.getenv("APP_HOST", "0.0.0.0"),
    "port":       int(os.getenv("APP_PORT", 7860)),
    "share":      False,
    "disclaimer": (
        "\u26a0\ufe0f DISCLAIMER: Dieses Tool ist zu Bildungszwecken. "
        "Es ersetzt KEINE medizinische Diagnose. "
        "Bei Hautver\u00e4nderungen immer einen Dermatologen aufsuchen."
    ),
}

# ─── API Keys (from .env) ─────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
KAGGLE_USERNAME   = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY        = os.getenv("KAGGLE_KEY", "")

DEVICE = os.getenv("DEVICE", "cuda")
