"""
src/cv/preprocessing.py - Data augmentation and dataset classes for HAM10000
ZHAW AI-Applications Project
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CV_CONFIG, PROC_DIR, NUM_CLASSES


# ─── ImageNet stats ───────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = CV_CONFIG["image_size"]


def get_train_transforms():
    """Aggressive augmentation for training to combat overfitting."""
    return T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """Minimal transforms for validation/test."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms():
    """TTA-compatible transforms for inference."""
    return get_val_transforms()


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Args:
        csv_path:   Path to split CSV (train/val/test.csv)
        transform:  torchvision transform pipeline
        return_meta: whether to return metadata dict alongside image
    """

    def __init__(
        self,
        csv_path: str,
        transform=None,
        return_meta: bool = False,
    ):
        self.df         = pd.read_csv(csv_path)
        self.transform  = transform or get_val_transforms()
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row    = self.df.iloc[idx]
        img    = Image.open(row["image_path"]).convert("RGB")
        img    = self.transform(img)
        label  = int(row["label"])

        if self.return_meta:
            meta = {
                "image_id":        row["image_id"],
                "age":             float(row.get("age", 0)),
                "sex_enc":         float(row.get("sex_enc", -1)),
                "localization_enc": float(row.get("localization_enc", -1)),
                "dx_type_enc":     float(row.get("dx_type_enc", -1)),
            }
            return img, label, meta

        return img, label


def get_dataloaders(
    proc_dir: Path = PROC_DIR,
    batch_size: int = CV_CONFIG["batch_size"],
    num_workers: int = CV_CONFIG["num_workers"],
):
    """Create train/val/test DataLoaders."""
    datasets = {
        "train": HAM10000Dataset(proc_dir / "train.csv", transform=get_train_transforms()),
        "val":   HAM10000Dataset(proc_dir / "val.csv",   transform=get_val_transforms()),
        "test":  HAM10000Dataset(proc_dir / "test.csv",  transform=get_val_transforms()),
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=CV_CONFIG["pin_memory"],
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }
    return loaders


def compute_class_weights(csv_path: str) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float
    )
    return weights


if __name__ == "__main__":
    loaders = get_dataloaders()
    for split, loader in loaders.items():
        batch = next(iter(loader))
        imgs, labels = batch[0], batch[1]
        print(f"{split:5}: {len(loader.dataset):5} samples | "
              f"batch {imgs.shape} | labels {labels.shape}")
