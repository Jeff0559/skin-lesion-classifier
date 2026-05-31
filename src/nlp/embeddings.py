"""
src/nlp/embeddings.py - Approach A: Sentence-Transformer symptom embeddings
ZHAW AI-Applications Project
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Optional, Union
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import NLP_CONFIG, PROC_DIR


# ─── Symptom description templates for HAM10000 classes ──────────────────────
SYMPTOM_TEMPLATES = {
    "mel": [
        "Dark irregular mole that has been growing for months, asymmetric borders, multiple colors",
        "Black and brown lesion with uneven edges, recently changed color and grew larger",
        "Suspicious dark spot on skin, irregular shape, has been changing over past year",
    ],
    "nv": [
        "Regular round mole, uniform brown color, stable for years, smooth borders",
        "Common brown flat mole, symmetric, no changes noticed, no pain",
        "Small uniform brown spot, soft edges, present since childhood",
    ],
    "bcc": [
        "Pearly bump on face that bleeds occasionally, appears translucent, slow growing",
        "Non-healing sore on sun-exposed area with rolled edges, shiny surface",
        "Small pink growth on nose with visible blood vessels, bleeds when touched",
    ],
    "akiec": [
        "Rough scaly patch on face, reddish, slightly itchy, appears after sun exposure",
        "Crusty lesion on hands with dry scaly texture, present for several months",
        "Flat red scaly area on sun-damaged skin, rough texture, occasionally bleeds",
    ],
    "bkl": [
        "Waxy stuck-on appearance, tan to brown, rough surface, keratosis",
        "Benign warty growth, multiple colors, well-defined borders, present for years",
        "Rough bumpy patch, looks like its stuck on the skin, light to dark brown",
    ],
    "df": [
        "Firm small bump on leg, dimples inward when squeezed, slightly itchy",
        "Hard nodule on skin, pinches inward with lateral pressure, brownish",
        "Tough benign skin growth, presses inward, appears on lower legs",
    ],
    "vasc": [
        "Bright red or purple spot, flat, vascular origin, bleeds easily",
        "Small cherry-red dome on skin, vascular lesion, appeared suddenly",
        "Purple vascular mark on skin, soft, compressible, appears on trunk",
    ],
}


class SymptomEmbedder:
    """
    Approach A: Encode free-text symptom descriptions using Sentence-Transformers.

    The embedding captures semantic meaning of symptom descriptions and can be
    used as features for the ML ensemble block.
    """

    def __init__(self, model_name: str = NLP_CONFIG["sentence_model"]):
        from sentence_transformers import SentenceTransformer
        print(f"[+] Loading sentence transformer: {model_name}")
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embed_dim  = self.model.get_sentence_embedding_dimension()
        print(f"[+] Embedding dimension: {self.embed_dim}")

        # Precompute class prototype embeddings
        self.class_embeddings = self._compute_class_embeddings()

    def _compute_class_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute mean embedding for each class from templates."""
        class_embs = {}
        for cls, templates in SYMPTOM_TEMPLATES.items():
            embs = self.model.encode(templates, normalize_embeddings=True)
            class_embs[cls] = embs.mean(axis=0)
        return class_embs

    def encode(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode one or multiple symptom descriptions."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        embeddings = self.model.encode(texts, normalize_embeddings=normalize)
        return embeddings if len(texts) > 1 else embeddings[0]

    def similarity_to_classes(self, text: str) -> Dict[str, float]:
        """
        Compute cosine similarity of user text to each class prototype.

        Returns:
            {class_name: similarity_score} for all 7 classes
        """
        text_emb = self.encode(text, normalize=True)
        sims = {}
        for cls, cls_emb in self.class_embeddings.items():
            sim = float(np.dot(text_emb, cls_emb))
            sims[cls] = sim
        return sims

    def get_feature_vector(self, text: str) -> np.ndarray:
        """
        Return combined feature vector for ML block:
        [embedding(384d) + class_similarities(7d)] = 391d
        """
        emb  = self.encode(text, normalize=True)
        sims = np.array([self.similarity_to_classes(text)[cls] for cls in sorted(self.class_embeddings.keys())])
        return np.concatenate([emb, sims])

    def predict_class(self, text: str) -> Dict:
        """Predict most likely class from symptom description alone."""
        sims     = self.similarity_to_classes(text)
        pred_cls = max(sims, key=sims.get)
        return {
            "predicted_class":  pred_cls,
            "confidence":       sims[pred_cls],
            "all_similarities": sims,
        }

    def precompute_dataset_embeddings(
        self,
        descriptions: List[str],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Precompute and optionally save embeddings for a list of descriptions."""
        print(f"[+] Computing {len(descriptions)} embeddings...")
        embeddings = self.model.encode(
            descriptions,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
        if save_path:
            np.save(save_path, embeddings)
            print(f"[+] Saved embeddings to {save_path}")
        return embeddings


def generate_synthetic_symptom_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic symptom descriptions for testing (no real patient data).
    Each sample is mapped to a HAM10000 class with realistic noise.
    """
    from src.config import CLASS_NAMES
    np.random.seed(seed)
    records = []
    for i in range(n):
        cls = np.random.choice(CLASS_NAMES)
        tmpl_idx = np.random.randint(0, len(SYMPTOM_TEMPLATES[cls]))
        base_text = SYMPTOM_TEMPLATES[cls][tmpl_idx]
        # Add noise by appending random duration/size info
        noise = np.random.choice([
            "Present for 2 weeks.", "Appeared 6 months ago.",
            "Patient noticed it last year.", "Recent change in past month.",
            "Longstanding lesion.", "",
        ])
        records.append({
            "symptom_text": f"{base_text} {noise}".strip(),
            "true_class":   cls,
            "sample_id":    i,
        })
    return pd.DataFrame(records)


DX_LABEL = {
    "mel":   "melanoma",
    "nv":    "nevus",
    "bcc":   "basal cell carcinoma",
    "akiec": "actinic keratosis",
    "bkl":   "benign keratosis",
    "df":    "dermatofibroma",
    "vasc":  "vascular lesion",
}


def build_text(row: pd.Series) -> str:
    dx   = DX_LABEL.get(row["dx"], row["dx"])
    loc  = str(row["localization"]).replace("_", " ") if pd.notna(row["localization"]) else "unknown location"
    age  = f"age {int(row['age'])}" if pd.notna(row["age"]) else "unknown age"
    sex  = str(row["sex"]) if pd.notna(row["sex"]) else "unknown"
    return f"{dx} on {loc}, {age}, {sex}"


if __name__ == "__main__":
    from src.config import PROC_DIR, MODELS_DIR

    dfs = []
    for split in ("train", "val", "test"):
        csv = PROC_DIR / f"{split}.csv"
        if csv.exists():
            dfs.append(pd.read_csv(csv))
    df = pd.concat(dfs, ignore_index=True)
    print(f"[+] Loaded {len(df)} real HAM10000 samples")

    df["text"] = df.apply(build_text, axis=1)
    print(f"[+] Example: {df['text'].iloc[0]}")

    embedder   = SymptomEmbedder()
    embeddings = embedder.precompute_dataset_embeddings(df["text"].tolist(), save_path=None)

    out = MODELS_DIR / "embeddings.npz"
    np.savez(str(out), embeddings=embeddings, labels=df["dx"].values, texts=df["text"].values)
    import os
    print(f"[+] Saved {embeddings.shape} → {round(os.path.getsize(out)/1024/1024, 1)} MB  →  {out}")

    print("\n[+] Query test:")
    result = embedder.predict_class("dark irregular spot growing for 6 months")
    sims   = sorted(result["all_similarities"].items(), key=lambda x: x[1], reverse=True)
    for cls, sim in sims:
        print(f"  {cls:8s} {sim:.4f}")
    print(f"  => {result['predicted_class']} (conf={result['confidence']:.4f})")
