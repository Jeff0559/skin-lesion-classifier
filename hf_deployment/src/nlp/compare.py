"""
src/nlp/compare.py - Compare Approach A (Sentence-Transformers) vs Approach B (Claude API)
ZHAW AI-Applications Project - Rubrik: NLP Approach Comparison (Pflicht)
"""
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASS_NAMES, LOGS_DIR
from src.nlp.embeddings import SymptomEmbedder, generate_synthetic_symptom_data
from src.nlp.llm_extractor import ClaudeSymptomExtractor


class NLPApproachComparator:
    """
    Side-by-side comparison of NLP Approach A vs B.

    Approach A: Sentence-Transformers (all-MiniLM-L6-v2)
      - Dense semantic embeddings (384d + 7d class similarities = 391d)
      - Fast, local, no API cost, no interpretability

    Approach B: Claude API (claude-sonnet-4-20250514)
      - Structured feature extraction (10 interpretable features)
      - Requires API key, has latency and cost
      - Interpretable, human-readable output
    """

    def __init__(self, use_llm: bool = True):
        print("[+] Loading Approach A: Sentence-Transformer embedder...")
        self.embedder = SymptomEmbedder()
        if use_llm:
            print("[+] Loading Approach B: Claude LLM extractor...")
            try:
                self.extractor = ClaudeSymptomExtractor()
                self.use_llm = True
            except ValueError as e:
                print(f"[!] Claude API not available: {e}")
                self.extractor = None
                self.use_llm = False
        else:
            self.extractor = None
            self.use_llm = False

    def get_approach_a_features(self, texts: List[str]) -> np.ndarray:
        """Approach A: concatenate embedding + class similarities."""
        return np.array([self.embedder.get_feature_vector(t) for t in texts])

    def get_approach_b_features(self, texts: List[str]) -> np.ndarray:
        """Approach B: structured LLM extraction -> numeric vector."""
        features = []
        for text in texts:
            if self.extractor:
                extracted = self.extractor.extract(text)
                vec = self.extractor.to_feature_vector(extracted)
            else:
                vec = [-1.0] * 10  # mock defaults
            features.append(vec)
        return np.array(features)

    def evaluate_approach(self, X, y, approach_name, n_splits=5):
        """Evaluate a feature set via cross-validated logistic regression."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial")),
        ])
        accs = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
        f1s  = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
        results = {
            "approach": approach_name,
            "feature_dim": int(X.shape[1]),
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std()),
        }
        print(f"  {approach_name:40s} | dim={X.shape[1]:4d} | acc={accs.mean():.3f} | f1={f1s.mean():.3f}")
        return results

    def run_comparison(self, n_samples=500, save_results=True):
        """Full comparison: generate data, extract features, evaluate, report."""
        print("\n" + "="*70)
        print("  NLP Approach Comparison: A (Sentence-Transformer) vs B (Claude)")
        print("="*70)

        df     = generate_synthetic_symptom_data(n=n_samples)
        texts  = df["symptom_text"].tolist()
        labels = df["true_class"].map({k: i for i, k in enumerate(CLASS_NAMES)}).values

        results = []

        # Approach A
        print("\n[Approach A] Sentence-Transformer Embeddings")
        t0  = time.time()
        X_a = self.get_approach_a_features(texts)
        ta  = time.time() - t0
        res = self.evaluate_approach(X_a, labels, "Sentence-Transformer (391d)")
        res["latency_ms_per_sample"] = (ta / len(texts)) * 1000
        res["api_cost_usd"] = 0.0
        results.append(res)

        # Approach B
        print("\n[Approach B] Claude API Structured Extraction")
        sample = texts[:50] if not self.use_llm else texts
        t0  = time.time()
        X_b = self.get_approach_b_features(sample)
        tb  = time.time() - t0
        lbl_b = labels[:len(sample)]
        res = self.evaluate_approach(X_b, lbl_b, "Claude API (10 structured features)")
        res["latency_ms_per_sample"] = (tb / max(1, len(sample))) * 1000
        res["api_cost_usd"] = len(sample) * (150 * 3 + 50 * 15) / 1_000_000
        results.append(res)

        # Winner analysis
        print("\nWINNER ANALYSIS:")
        a_f1 = results[0]["f1_mean"]
        b_f1 = results[1]["f1_mean"]
        print(f"  F1 Approach A: {a_f1:.3f}")
        print(f"  F1 Approach B: {b_f1:.3f}")
        winner = "A (Sentence-Transformer)" if a_f1 >= b_f1 else "B (Claude)"
        print(f"  Better classification F1: {winner}")
        print(f"  Approach B advantage: interpretable features, no black box")
        print(f"  Approach A advantage: faster ({results[0]["latency_ms_per_sample"]:.1f} vs {results[1]["latency_ms_per_sample"]:.1f} ms/sample), free")

        df_out = pd.DataFrame(results)
        if save_results:
            out = LOGS_DIR / "nlp_comparison.json"
            df_out.to_json(out, orient="records", indent=2)
            print(f"\n[+] Saved to {out}")
        return df_out


if __name__ == "__main__":
    c = NLPApproachComparator(use_llm=True)
    c.run_comparison(n_samples=200)
