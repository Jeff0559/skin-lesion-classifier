"""
src/data/prepare_ham10000.py
Download, validate, split and save HAM10000 dataset.

Usage:
    python -m src.data.prepare_ham10000

Requirements:
    - Set KAGGLE_USERNAME and KAGGLE_KEY in .env
    - Or place kaggle.json in ~/.kaggle/
"""
import os
import sys
import zipfile
import shutil
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RAW_DIR, INTERIM_DIR, PROC_DIR,
    CLASS_NAMES, HAM10000_CLASSES, CV_CONFIG
)

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
METADATA_FILE  = "HAM10000_metadata.csv"
IMAGE_DIRS     = ["ham10000_images_part_1", "ham10000_images_part_2"]


def setup_kaggle_credentials():
    """Configure Kaggle API from .env."""
    username = os.getenv("KAGGLE_USERNAME")
    key      = os.getenv("KAGGLE_KEY")
    if username and key:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        cred_file  = kaggle_dir / "kaggle.json"
        cred_file.write_text(json.dumps({"username": username, "key": key}))
        cred_file.chmod(0o600)
        print(f"[+] Kaggle credentials saved to {cred_file}")
    else:
        print("[!] KAGGLE_USERNAME or KAGGLE_KEY not set in .env")
        print("    Ensure ~/.kaggle/kaggle.json exists manually.")


def download_dataset():
    """Download HAM10000 via kaggle CLI."""
    import subprocess
    print(f"[+] Downloading {KAGGLE_DATASET}...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(RAW_DIR),
        "--unzip"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Kaggle download failed:\n{result.stderr}")
        print("    Please download manually from:")
        print(f"    https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"    and extract to: {RAW_DIR}")
        sys.exit(1)
    print("[+] Download complete.")


def collect_images() -> dict:
    """Collect all image paths from both part directories."""
    img_paths = {}
    for d in IMAGE_DIRS:
        img_dir = RAW_DIR / d
        if not img_dir.exists():
            continue
        for fp in img_dir.glob("*.jpg"):
            img_paths[fp.stem] = fp
    print(f"[+] Found {len(img_paths)} images total.")
    return img_paths


def load_metadata() -> pd.DataFrame:
    """Load and validate HAM10000 metadata CSV."""
    meta_path = RAW_DIR / METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    df = pd.read_csv(meta_path)
    print(f"[+] Loaded metadata: {len(df)} rows, cols={list(df.columns)}")
    return df


def preprocess_metadata(df: pd.DataFrame, img_paths: dict) -> pd.DataFrame:
    """Filter, encode, and augment metadata."""
    # Only keep rows with existing images
    df = df[df["image_id"].isin(img_paths)].copy()
    df["image_path"] = df["image_id"].map(img_paths).apply(str)

    # Encode categorical features
    df["sex_enc"]          = df["sex"].map({"male": 1, "female": 0, "unknown": -1}).fillna(-1)
    df["label"]            = df["dx"].map({k: i for i, k in enumerate(CLASS_NAMES)})
    df["label_name"]       = df["dx"]
    df["dx_type_enc"]      = df["dx_type"].astype("category").cat.codes
    df["localization_enc"] = df["localization"].astype("category").cat.codes
    df["age"]              = df["age"].fillna(df["age"].median())

    print(f"[+] Preprocessed: {len(df)} valid samples")
    print(df["dx"].value_counts().to_string())
    return df


def split_dataset(df: pd.DataFrame):
    """Stratified train/val/test split."""
    val_split  = CV_CONFIG["val_split"]
    test_split = CV_CONFIG["test_split"]
    seed       = CV_CONFIG["random_state"]

    train_val, test = train_test_split(
        df, test_size=test_split, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_split / (1 - test_split),
        stratify=train_val["label"],
        random_state=seed,
    )
    print(f"[+] Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def copy_images_to_interim(df: pd.DataFrame, split_name: str):
    """Copy images to interim/{split_name}/{class_name}/ directories."""
    target_dir = INTERIM_DIR / split_name
    copied = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        cls_dir = target_dir / row["label_name"]
        cls_dir.mkdir(parents=True, exist_ok=True)
        dst = cls_dir / Path(row["image_path"]).name
        if not dst.exists():
            shutil.copy2(row["image_path"], dst)
        copied += 1
    print(f"[+] {split_name}: {copied} images copied to {target_dir}")


def save_processed_splits(train, val, test):
    """Save CSV splits to data/processed/."""
    train.to_csv(PROC_DIR / "train.csv", index=False)
    val.to_csv(PROC_DIR / "val.csv", index=False)
    test.to_csv(PROC_DIR / "test.csv", index=False)

    # Save class weights for imbalanced training
    counts = train["label"].value_counts().sort_index()
    total  = len(train)
    weights = {i: total / (len(CLASS_NAMES) * c) for i, c in counts.items()}
    weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["weight"])
    weights_df.to_csv(PROC_DIR / "class_weights.csv")
    print(f"[+] Saved splits to {PROC_DIR}")
    print(f"    Class weights: {weights}")


def validate_images(df: pd.DataFrame, split_name: str, n_sample: int = 50):
    """Spot-check images for corruption."""
    sample = df.sample(min(n_sample, len(df)), random_state=42)
    corrupted = []
    for _, row in sample.iterrows():
        try:
            img = Image.open(row["image_path"])
            img.verify()
        except Exception as e:
            corrupted.append((row["image_id"], str(e)))
    if corrupted:
        print(f"[!] {split_name}: {len(corrupted)} corrupted images found!")
        for img_id, err in corrupted[:5]:
            print(f"    {img_id}: {err}")
    else:
        print(f"[+] {split_name}: all sampled images valid.")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("  HAM10000 Dataset Preparation")
    print("=" * 60)

    # Step 1: Credentials
    setup_kaggle_credentials()

    # Step 2: Download (skip if already present)
    metadata_path = RAW_DIR / METADATA_FILE
    if not metadata_path.exists():
        download_dataset()
    else:
        print(f"[+] Dataset already exists at {RAW_DIR}, skipping download.")

    # Step 3: Load data
    img_paths = collect_images()
    df        = load_metadata()
    df        = preprocess_metadata(df, img_paths)

    # Step 4: Validate
    validate_images(df, "full_dataset")

    # Step 5: Split
    train, val, test = split_dataset(df)

    # Step 6: Copy to interim
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        copy_images_to_interim(split_df, split_name)

    # Step 7: Save processed
    save_processed_splits(train, val, test)

    print("\n" + "=" * 60)
    print("  Preparation complete!")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
