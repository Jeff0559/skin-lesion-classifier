"""
src/ml/train.py - ML ensemble training: LR vs RF vs XGBoost with ablation study
ZHAW AI-Applications Project

Usage:
    python -m src.ml.train

Performs:
    1. Load features (CV probs + metadata + NLP)
    2. Train Logistic Regression, Random Forest, XGBoost
    3. Ablation study across feature groups
    4. Compute ROC-AUC + F1 metrics
    5. Save best model
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, accuracy_score
)
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import ML_CONFIG, CLASS_NAMES, NUM_CLASSES, MODELS_DIR, LOGS_DIR, PROC_DIR
from src.ml.features import (
    load_processed_data, build_feature_matrix,
    get_cv_probs_from_checkpoint, get_feature_names_for_group
)


def get_models() -> Dict:
    """Return configured model instances."""
    cfg = ML_CONFIG["models"]
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**cfg["logistic_regression"])),
        ]),
        "random_forest": RandomForestClassifier(**cfg["random_forest"]),
    }
    try:
        from xgboost import XGBClassifier
        xgb_cfg = {k: v for k, v in cfg["xgboost"].items() if k != "use_label_encoder"}
        models["xgboost"] = XGBClassifier(**xgb_cfg, verbosity=0)
    except ImportError:
        print("[!] XGBoost not installed, skipping.")
    return models


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict:
    """Fit model and evaluate on test set."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        proba = model.decision_function(X_test)
    else:
        proba = label_binarize(preds, classes=list(range(NUM_CLASSES)))

    # Binarize labels for OvR ROC-AUC
    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

    try:
        roc_auc = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = float("nan")

    f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))
    f1_per   = f1_score(y_test, preds, average=None, zero_division=0).tolist()
    acc      = float(accuracy_score(y_test, preds))

    report = {
        "model":      model_name,
        "accuracy":   acc,
        "f1_macro":   f1_macro,
        "f1_per_class": {CLASS_NAMES[i]: f1_per[i] for i in range(NUM_CLASSES)},
        "roc_auc":    roc_auc,
        "report":     classification_report(
            y_test, preds, target_names=CLASS_NAMES, zero_division=0
        ),
    }

    print(f"  {model_name:25s} | acc={acc:.3f} | f1={f1_macro:.3f} | auc={roc_auc:.3f}")
    return report


def ablation_study(
    X_groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Ablation study: evaluate best model (XGBoost) on each feature group.
    Returns DataFrame with results per feature group.
    """
    print("\n[+] Ablation Study")
    print("-" * 60)

    rows = []
    try:
        from xgboost import XGBClassifier
        xgb_cfg = {k: v for k, v in ML_CONFIG["models"]["xgboost"].items() if k != "use_label_encoder"}
        ModelClass = lambda: XGBClassifier(**xgb_cfg, verbosity=0)
    except ImportError:
        ModelClass = lambda: RandomForestClassifier(**ML_CONFIG["models"]["random_forest"])

    for group_name, (X_tr, X_te) in X_groups.items():
        model = ModelClass()
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        f1    = float(f1_score(y_test, preds, average="macro", zero_division=0))
        acc   = float(accuracy_score(y_test, preds))

        proba = model.predict_proba(X_te)
        y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
        try:
            auc = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")

        print(f"  {group_name:20s} | dim={X_tr.shape[1]:4d} | acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
        rows.append({"feature_group": group_name, "dim": X_tr.shape[1], "accuracy": acc, "f1_macro": f1, "roc_auc": auc})

    return pd.DataFrame(rows)


def train_and_evaluate(
    cv_model_path: str = None,
    device: str = "cpu",
):
    """Main training pipeline."""
    print("=" * 60)
    print("  ML Ensemble Training")
    print("=" * 60)

    # ── Load processed data ───────────────────────────────────────────────────
    train_df = load_processed_data("train")
    test_df  = load_processed_data("test")
    y_train  = train_df["label"].values
    y_test   = test_df["label"].values
    print(f"[+] Train: {len(train_df)} | Test: {len(test_df)}")

    # ── Get CV probabilities ──────────────────────────────────────────────────
    if cv_model_path and Path(cv_model_path).exists():
        print(f"[+] Extracting CV probs from {cv_model_path}")
        cv_probs_train = get_cv_probs_from_checkpoint(train_df, cv_model_path, device)
        cv_probs_test  = get_cv_probs_from_checkpoint(test_df, cv_model_path, device)
    else:
        print("[!] No CV model found, using random probs (train first with src.cv.train)")
        cv_probs_train = np.random.dirichlet(np.ones(NUM_CLASSES), len(train_df))
        cv_probs_test  = np.random.dirichlet(np.ones(NUM_CLASSES), len(test_df))

    # ── Synthetic NLP features ────────────────────────────────────────────────
    # In production, run NLP extraction on real symptom texts
    # Here we use zeros as placeholder - replace with real NLP outputs
    nlp_train = np.zeros((len(train_df), 10))
    nlp_test  = np.zeros((len(test_df), 10))

    # ── Build feature groups for ablation ────────────────────────────────────
    feature_groups = {}
    for group in ("cv_only", "metadata_only", "cv_meta", "all_features"):
        X_tr, _ = build_feature_matrix(train_df, cv_probs_train, nlp_train, feature_group=group)
        X_te, _ = build_feature_matrix(test_df,  cv_probs_test,  nlp_test,  feature_group=group)
        feature_groups[group] = (X_tr, X_te)

    # ── Ablation Study ────────────────────────────────────────────────────────
    ablation_df = ablation_study(feature_groups, y_train, y_test)

    # ── Full feature set: compare all models ──────────────────────────────────
    X_train_full, feat_names = build_feature_matrix(
        train_df, cv_probs_train, nlp_train, feature_group="all_features"
    )
    X_test_full, _ = build_feature_matrix(
        test_df, cv_probs_test, nlp_test, feature_group="all_features"
    )

    print("\n[+] Model Comparison (all features)")
    print("-" * 60)
    models  = get_models()
    reports = []
    for name, model in models.items():
        rep = evaluate_model(model, X_train_full, y_train, X_test_full, y_test, name)
        reports.append(rep)

    # ── Save best model ───────────────────────────────────────────────────────
    best_report = max(reports, key=lambda r: r["roc_auc"] if not (r["roc_auc"] != r["roc_auc"]) else r["f1_macro"])
    best_name   = best_report["model"]
    best_model  = models[best_name]
    best_path   = MODELS_DIR / f"ml_{best_name}_best.pkl"
    joblib.dump(best_model, best_path)
    print(f"\n[+] Best model: {best_name} (F1={best_report["f1_macro"]:.3f}, AUC={best_report["roc_auc"]:.3f})")
    print(f"[+] Saved to {best_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model_reports":  reports,
        "ablation_study": ablation_df.to_dict(orient="records"),
        "best_model":     best_name,
        "feature_names":  feat_names,
    }
    out_path = LOGS_DIR / "ml_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[+] Results saved to {out_path}")

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    for r in reports:
        print(f"  {r["model"]:25s} | F1={r["f1_macro"]:.3f} | AUC={r["roc_auc"]:.3f}")

    return results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    cv_path = os.getenv("CV_MODEL_PATH")
    train_and_evaluate(cv_model_path=cv_path)
