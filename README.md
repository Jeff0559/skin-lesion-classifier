# Skin Lesion Classifier

**ZHAW School of Engineering — AI-Applications Abschlussprojekt**  
**Modul:** AI-Applications  
**Student:** Jeff0559  
**Datum:** 2026-05-30  
**Betreuer:** ZHAW Dozent

---

## Projektbeschreibung

Ein multimodales KI-System zur Klassifikation von Hautläsionen in 7 Klassen (HAM10000 Dataset).
Das System kombiniert Computer Vision, Natural Language Processing und klassisches Machine Learning.

**WICHTIG:** Dieses Tool dient ausschliesslich Bildungszwecken und ersetzt keine medizinische Diagnose.
Bei Hautveränderungen immer einen Dermatologen aufsuchen.

---

## Die 3 KI-Blöcke

### Block 1: Computer Vision (Haupt-Block)
- **Dataset:** HAM10000 (10.015 Bilder, 7 Klassen)
- **Modell:** ResNet50 (Transfer Learning, ImageNet vortrainiert)
- **Methode:** Fine-tuning der letzten 2 Blöcke + Custom Classification Head
- **Ziel:** 75-85% Accuracy, Macro F1 >= 0.65
- **Outputs:** Confusion Matrix, F1 pro Klasse, Grad-CAM Visualisierungen

### Block 2: NLP (Symptom-Extraktion)
- **User-Input:** Freitext-Symptombeschreibung
- **Approach A:** Sentence-Transformers (`all-MiniLM-L6-v2`) — semantische Embeddings (384d)
- **Approach B:** Claude API (`claude-sonnet-4-20250514`) — strukturierte Feature-Extraktion
- **Extrahierte Features:** Dauer, Farbe, Grösse, Schmerz, Lokalisation, Juckreiz, Blutung, Veränderungsrate
- **Vergleich:** Accuracy, F1, Latenz, API-Kosten (Rubrik-Pflicht)

### Block 3: ML Ensemble
- **Features:** CV-Output (7d) + NLP-Features (10d) + Metadaten (Alter, Körperstelle, Geschlecht)
- **Modelle:** Logistic Regression vs Random Forest vs XGBoost
- **Ablation Study:** cv_only | metadata_only | cv_meta | all_features
- **Metrics:** ROC-AUC (primär) + F1-Macro (sekundär)
- **Explainability:** SHAP Feature Importance

---

## Repository-Struktur

```
skin-lesion-classifier/
├── README.md                      # Dieses Dokument
├── SETUP.md                       # Setup-Anleitung
├── requirements.txt               # Python dependencies
├── .gitignore                     # Excludes data/raw, models, .env
├── .env.example                   # Environment-Variablen Template
├── data/
│   ├── raw/                       # HAM10000 raw files (NOT committed)
│   ├── interim/                   # Train/val/test image copies (NOT committed)
│   └── processed/                 # CSV splits + class weights (NOT committed)
├── notebooks/
│   ├── 01_data_collection.ipynb   # Download + Validierung
│   ├── 02_eda.ipynb               # Explorative Datenanalyse
│   ├── 03_cv_training_colab.ipynb # ResNet50 Training (Colab GPU)
│   ├── 04_nlp_experiments.ipynb   # NLP Approach A vs B
│   ├── 05_ml_modeling.ipynb       # Ensemble + Ablation
│   └── 06_ablation_and_errors.ipynb # Fehleranalyse + SHAP
├── src/
│   ├── config.py                  # Zentrale Konfiguration
│   ├── pipeline.py                # End-to-end Inferenz
│   ├── data/prepare_ham10000.py   # Dataset-Vorbereitung
│   ├── cv/
│   │   ├── model.py               # ResNet50 Architektur
│   │   ├── preprocessing.py       # DataLoaders + Augmentation
│   │   ├── train.py               # Training Loop
│   │   ├── inference.py           # Inferenz Pipeline
│   │   └── grad_cam.py            # Grad-CAM Visualisierung
│   ├── nlp/
│   │   ├── embeddings.py          # Approach A: Sentence-Transformers
│   │   ├── llm_extractor.py       # Approach B: Claude API
│   │   ├── compare.py             # A vs B Vergleich
│   │   └── explainer.py           # Erklärungsgenerierung
│   └── ml/
│       ├── features.py            # Feature Engineering
│       ├── train.py               # Ensemble Training + Ablation
│       └── shap_analysis.py       # SHAP Feature Importance
├── app/app.py                     # Gradio Web-App
├── models/                        # Trainierte Modelle (NOT committed)
├── logs/                          # Training Logs (NOT committed)
├── docs/
│   ├── ethics.md                  # Ethische Überlegungen
│   └── screenshots/               # Visualisierungen für Report
└── tests/test_pipeline.py         # Unit Tests (pytest)
```

---

## HAM10000 Klassen

| Kürzel | Klasse | Beschreibung |
|--------|--------|-------------|
| mel    | Melanoma | Bösartiger Hauttumor, höchste Mortalität |
| nv     | Melanocytic nevi | Gutartige Muttermale (67% des Datasets) |
| bcc    | Basal cell carcinoma | Häufigste Hautkrebsform |
| akiec  | Actinic keratoses | Präkanzeros, UV-bedingt |
| bkl    | Benign keratosis | Gutartige seborrhoische Keratosen |
| df     | Dermatofibroma | Gutartiger dermaler Tumor |
| vasc   | Vascular lesions | Gefässbedingte Läsionen |

---

## Schnellstart

```bash
# 1. Repository klonen
git clone https://github.com/Jeff0559/skin-lesion-classifier.git
cd skin-lesion-classifier

# 2. Virtual environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Environment konfigurieren
cp .env.example .env
# .env bearbeiten: ANTHROPIC_API_KEY, KAGGLE_USERNAME, KAGGLE_KEY

# 4. Dataset herunterladen + vorbereiten
python -m src.data.prepare_ham10000

# 5. CV Training (GPU empfohlen -> Colab)
# Lokal: python -m src.cv.train
# Colab: notebooks/03_cv_training_colab.ipynb öffnen

# 6. ML Ensemble training
python -m src.ml.train

# 7. Tests ausführen
pytest tests/ -v --tb=short

# 8. App starten
python app/app.py
# -> http://localhost:7860
```

Detaillierte Setup-Anleitung: [SETUP.md](SETUP.md)

---

## Technologien & Libraries

| Kategorie | Library | Version | Zweck |
|-----------|---------|---------|-------|
| Deep Learning | PyTorch | ≥2.1 | ResNet50 Training |
| Computer Vision | torchvision, albumentations | ≥0.16 | Augmentation, Transforms |
| Explainability | grad-cam | ≥1.4 | Grad-CAM Visualisierung |
| NLP | sentence-transformers | ≥2.2 | Approach A Embeddings |
| LLM | anthropic | ≥0.28 | Approach B: Claude API |
| ML | scikit-learn, xgboost | ≥1.3, ≥2.0 | Ensemble Modelle |
| Explainability | shap | ≥0.43 | Feature Importance |
| App | gradio | ≥4.20 | Web Interface |
| Data | pandas, numpy | ≥2.0, ≥1.24 | Datenverarbeitung |

---

## Live Demo

**Hugging Face Spaces:** https://huggingface.co/spaces/Jeremie03/skin-lesion-classifier

---

## Ergebnisse (nach Training)

### Block 1: CV (ResNet50 auf HAM10000, Val-Set n=1503)
| Metrik | Wert |
|--------|------|
| Val Accuracy | 87% |
| Macro F1 | 0.83 |
| Weighted F1 | 0.87 |
| Beste Klasse | nv (F1=0.93), vasc (F1=0.93) |
| Schwierigste Klasse | mel (F1=0.65) |

### Block 3: ML Ensemble (n=1503, alle Features)
| Modell | ROC-AUC | F1-Macro | Accuracy |
|--------|---------|----------|----------|
| Logistic Regression | **0.9965** | 0.9510 | 95.3% |
| Random Forest | 0.9957 | 0.9464 | 95.3% |
| XGBoost | 0.9962 | 0.9374 | 94.9% |

### Ablation Study
| Feature Group | ROC-AUC |
|--------------|---------|
| CV only | ~0.96 |
| Metadata only | ~0.82 |
| CV + Metadata | ~0.98 |
| All Features (CV + Meta + NLP) | **0.9965** |

---

## Ethik & Limitierungen

Dieses Projekt erkennt folgende ethische Herausforderungen:

- **Kein Medizinprodukt**: Nicht für klinische Entscheidungen geeignet
- **Dataset-Bias**: HAM10000 ist mehrheitlich europäisch/australisch, helle Hautfarben überrepräsentiert
- **Datenschutz**: Hautbilder sind sensible Gesundheitsdaten (DSGVO Art. 9)
- **Regulierung**: Medizinische KI unterliegt EU MDR + AI Act (Hochrisiko-Kategorie)

Vollständige Analyse: [docs/ethics.md](docs/ethics.md)

---

## App Features

Die Gradio-App (`app/app.py`) bietet:
- **Bild-Upload**: Direkte Klassifikation via ResNet50
- **Symptom-Beschreibung**: Freitext-Eingabe mit NLP-Extraktion
- **Metadaten**: Alter, Geschlecht, Körperstelle
- **7-Klassen Wahrscheinlichkeiten**: Balkendiagramm
- **Claude Erklärung**: Verständliche AI-generierte Erklärung
- **Pflicht-Disclaimer**: Immer sichtbar

---

## Tests

```bash
pytest tests/ -v --tb=short
# Abgedeckt:
# - TestConfig: Klassen, Konfiguration
# - TestCVModel: Architektur, Shapes, Probabilities
# - TestPreprocessing: Transforms, Augmentation
# - TestNLP: Embeddings, Extraktion, Vergleich
# - TestMLFeatures: Feature-Matrix, Ablation
# - TestIntegration: End-to-end CV Inferenz
```

---

## Kaggle Dataset

**HAM10000: Human Against Machine with 10000 training images**  
Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data, 5*, 180161.

Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

## Lizenz

Dieses Projekt wurde für das ZHAW AI-Applications Modul erstellt.
Der Code ist für Bildungszwecke freigegeben.

---

*⚠️ DISCLAIMER: Dieses Tool ist zu Bildungszwecken. Es ersetzt KEINE medizinische Diagnose. Bei Hautveränderungen immer einen Dermatologen aufsuchen.*
