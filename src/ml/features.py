"""
src/ml/features.py - Feature engineering for ML ensemble block
ZHAW AI-Applications Project

Combines:
  1. CV features: 7 class probabilities from ResNet50
  2. NLP features: 10 structured symptoms OR 391d embeddings
  3. Metadata: age, sex, localization, dx_type from HAM10000
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASS_NAMES, ML_CONFIG, PROC_DIR


METADATA_FEATURES = ["age", "sex_enc", "localization_enc", "dx_type_enc"]
CV_PROB_FEATURES  = [f"cv_prob_{i}" for i in range(len(CLASS_NAMES))]
NLP_FEATURES      = [
    "duration_days", "color_enc", "size_mm", "pain_level",
    "localization_nlp_enc", "itching", "bleeding", "change_rate_enc",
    "asymmetry", "border_irregularity",
]


def build_feature_matrix(
    metadata_df: pd.DataFrame,
    cv_probs: np.ndarray,
    nlp_features: Optional[np.ndarray] = None,
    feature_group: str = "all_features",
) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble feature matrix for ML ensemble.

    Args:
        metadata_df:   DataFrame with metadata columns
        cv_probs:      (N, 7) array of ResNet50 class probabilities
        nlp_features:  (N, K) array of NLP features (optional)
        feature_group: which feature subset to use (ablation study)

    Returns:
        X:            (N, D) feature matrix
        feature_names: list of feature names
    """
    parts       = []
    feat_names  = []

    use_cv   = feature_group in ("cv_only", "cv_meta", "all_features")
    use_meta = feature_group in ("metadata_only", "cv_meta", "all_features")
    use_nlp  = feature_group in ("nlp_only", "all_features") and nlp_features is not None

    if use_cv:
        parts.append(cv_probs)
        feat_names.extend(CV_PROB_FEATURES)

    if use_meta:
        meta = metadata_df[METADATA_FEATURES].fillna(-1).values.astype(float)
        parts.append(meta)
        feat_names.extend(METADATA_FEATURES)

    if use_nlp:
        parts.append(nlp_features)
        feat_names.extend([f"nlp_{i}" for i in range(nlp_features.shape[1])])

    if not parts:
        raise ValueError(f"No features selected for group: {feature_group}")

    X = np.hstack(parts)
    return X, feat_names


def load_processed_data(
    split: str = "train",
    proc_dir: Path = PROC_DIR,
) -> pd.DataFrame:
    """Load processed CSV split."""
    path = proc_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            f"Run: python -m src.data.prepare_ham10000"
        )
    return pd.read_csv(path)


def get_cv_probs_from_checkpoint(
    df: pd.DataFrame,
    model_path: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run ResNet50 inference on all images in df and return probability matrix.
    Only run once and cache results.
    """
    cache_path = Path(model_path).parent / f"cv_probs_{Path(model_path).stem}.npy"
    if cache_path.exists():
        print(f"[+] Loading cached CV probs from {cache_path}")
        return np.load(cache_path)

    from src.cv.model import load_model
    from src.cv.preprocessing import get_val_transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torch

    class SimpleDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths     = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    model  = load_model(model_path, device=device)
    model.eval()
    ds     = SimpleDataset(df["image_path"].tolist(), get_val_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = model.get_probabilities(batch).cpu().numpy()
            all_probs.append(probs)

    probs_matrix = np.vstack(all_probs)
    np.save(cache_path, probs_matrix)
    print(f"[+] Cached CV probs to {cache_path}")
    return probs_matrix


def get_feature_names_for_group(group: str, nlp_dim: int = 10) -> List[str]:
    """Return expected feature names for a given group."""
    names = []
    if group in ("cv_only", "cv_meta", "all_features"):
        names.extend(CV_PROB_FEATURES)
    if group in ("metadata_only", "cv_meta", "all_features"):
        names.extend(METADATA_FEATURES)
    if group in ("nlp_only", "all_features"):
        names.extend([f"nlp_{i}" for i in range(nlp_dim)])
    return names


if __name__ == "__main__":
    # Quick test with dummy data
    N = 100
    meta = pd.DataFrame({
        "age":              np.random.uniform(20, 80, N),
        "sex_enc":          np.random.randint(-1, 2, N),
        "localization_enc": np.random.randint(0, 15, N),
        "dx_type_enc":      np.random.randint(0, 4, N),
    })
    cv   = np.random.dirichlet(np.ones(7), N)
    nlp  = np.random.randn(N, 10)
    for group in ("cv_only", "metadata_only", "nlp_only", "cv_meta", "all_features"):
        X, names = build_feature_matrix(meta, cv, nlp, feature_group=group)
        print(f"  {group:20s}: X.shape={X.shape}, features={names[:3]}...")
