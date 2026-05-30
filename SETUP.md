# Setup Guide

## 1. Clone & Environment

```bash
git clone https://github.com/Jeff0559/skin-lesion-classifier.git
cd skin-lesion-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Environment Variables

```bash
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY=sk-ant-...
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_api_key
```

Get your Kaggle API key at: https://www.kaggle.com/settings/account

## 3. Download Dataset

**Option A: Automatic (requires Kaggle API key in .env)**
```bash
python -m src.data.prepare_ham10000
```

**Option B: Manual**
1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click Download
3. Extract to `data/raw/`
4. Run: `python -m src.data.prepare_ham10000`

**Option C: Colab (recommended for GPU)**
- Open `notebooks/03_cv_training_colab.ipynb` in Google Colab
- The notebook handles dataset download automatically

## 4. Train Models

### Block 1: Computer Vision (GPU recommended)

**Local (if you have GPU):**
```bash
python -m src.cv.train
# Expected: 75-85% accuracy after 20 epochs
# Best model saved to: models/resnet50_best.pth
```

**Google Colab (recommended):**
```
Open notebooks/03_cv_training_colab.ipynb
Runtime -> Change runtime type -> GPU (T4)
Run all cells
Download models/resnet50_best.pth
```

### Block 2: NLP Experiments
```bash
# Run NLP comparison (Approach A vs B)
python -m src.nlp.compare

# Or run notebook:
jupyter notebook notebooks/04_nlp_experiments.ipynb
```

### Block 3: ML Ensemble
```bash
# Requires trained CV model
export CV_MODEL_PATH=models/resnet50_best.pth
python -m src.ml.train

# Or run notebook:
jupyter notebook notebooks/05_ml_modeling.ipynb
```

## 5. Run Tests

```bash
# Run all tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 6. Launch App

```bash
python app/app.py
# Open: http://localhost:7860
```

## 7. Run Notebooks in Order

```
01_data_collection.ipynb    # Download + validate data
02_eda.ipynb                # Exploratory analysis
03_cv_training_colab.ipynb  # Train ResNet50 (Colab/GPU)
04_nlp_experiments.ipynb    # NLP Approach A vs B
05_ml_modeling.ipynb        # Ensemble + Ablation
06_ablation_and_errors.ipynb # Error analysis + SHAP
```

## 8. Expected Results

| Model | Metric | Expected |
|-------|--------|---------|
| ResNet50 (CV) | Accuracy | 75-85% |
| ResNet50 (CV) | Macro F1 | 0.65-0.75 |
| XGBoost (ML) | ROC-AUC | 0.90-0.95 |
| XGBoost (ML) | Macro F1 | 0.70-0.80 |

## 9. Directory Structure After Setup

```
data/
  raw/           <- HAM10000 raw files (not committed)
  interim/       <- Train/val/test image copies (not committed)
  processed/     <- CSV splits + class weights (not committed)
models/
  resnet50_best.pth     <- Trained CV model
  ml_xgboost_best.pkl   <- Trained ML model
logs/
  cv_results.json
  ml_results.json
  nlp_comparison.json
```

## Troubleshooting

**Kaggle download fails:**
- Ensure KAGGLE_USERNAME and KAGGLE_KEY are set in `.env`
- Or manually place `kaggle.json` in `~/.kaggle/`

**CUDA out of memory:**
- Reduce `batch_size` in `src/config.py` (try 16 or 8)
- Use Google Colab with T4 GPU for free

**anthropic module not found:**
- `pip install anthropic>=0.28.0`

**Gradio app crashes:**
- Ensure CV model exists at `models/resnet50_best.pth`
- Check logs in terminal for stack trace
