"""
src/ml/shap_analysis.py - SHAP feature importance analysis
ZHAW AI-Applications Project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASS_NAMES, LOGS_DIR


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list,
    model_type: str = "tree",
) -> object:
    """
    Compute SHAP values for a trained model.

    Args:
        model:         trained sklearn/xgboost model
        X:             feature matrix (N, D)
        feature_names: list of feature names
        model_type:    "tree" for RF/XGBoost, "linear" for LR

    Returns:
        shap_values object
    """
    import shap
    if model_type == "tree":
        # Handle sklearn Pipeline
        if hasattr(model, "steps"):
            clf = model.named_steps["clf"]
            X_transformed = model.named_steps["scaler"].transform(X)
        else:
            clf = model
            X_transformed = X
        explainer = shap.TreeExplainer(clf)
    else:
        if hasattr(model, "steps"):
            clf = model.named_steps["clf"]
            X_transformed = model.named_steps["scaler"].transform(X)
        else:
            clf = model
            X_transformed = X
        explainer = shap.LinearExplainer(clf, X_transformed)

    shap_values = explainer.shap_values(X_transformed)
    return shap_values, explainer, X_transformed


def plot_shap_summary(
    shap_values,
    X: np.ndarray,
    feature_names: list,
    class_idx: int = 0,
    save_path: Optional[str] = None,
    max_display: int = 20,
):
    """Plot SHAP summary (beeswarm) for a specific class."""
    import shap
    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    else:
        sv = shap_values

    shap.summary_plot(
        sv, X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Feature Importance: {CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else class_idx}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_shap_bar(
    shap_values,
    feature_names: list,
    save_path: Optional[str] = None,
):
    """Plot mean absolute SHAP values (global importance)."""
    import shap
    if isinstance(shap_values, list):
        # Multi-class: average across classes
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature"], df["mean_abs_shap"], color="#ff6b6b")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    return df


def run_shap_analysis(
    model,
    X: np.ndarray,
    feature_names: list,
    model_type: str = "tree",
    save_dir: Optional[Path] = None,
) -> dict:
    """Full SHAP analysis pipeline."""
    save_dir = save_dir or LOGS_DIR

    print("[+] Computing SHAP values...")
    shap_values, explainer, X_tr = compute_shap_values(
        model, X, feature_names, model_type
    )

    print("[+] Plotting global importance...")
    bar_df = plot_shap_bar(
        shap_values,
        feature_names,
        save_path=str(save_dir / "shap_global.png"),
    )

    print("[+] Plotting per-class summaries...")
    if isinstance(shap_values, list):
        for i in range(min(len(shap_values), len(CLASS_NAMES))):
            plot_shap_summary(
                shap_values, X_tr, feature_names, class_idx=i,
                save_path=str(save_dir / f"shap_{CLASS_NAMES[i]}.png"),
            )
    else:
        plot_shap_summary(
            shap_values, X_tr, feature_names,
            save_path=str(save_dir / "shap_summary.png"),
        )

    return {"top_features": bar_df.to_dict(orient="records")}
