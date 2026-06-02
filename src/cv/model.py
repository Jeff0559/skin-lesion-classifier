"""
src/cv/model.py - ResNet50 fine-tuning for HAM10000 7-class classification
ZHAW AI-Applications Project
"""
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CV_CONFIG, NUM_CLASSES


class SkinLesionResNet50(nn.Module):
    """
    ResNet50 fine-tuned for 7-class skin lesion classification.

    Architecture:
        - Frozen base: ResNet50 (ImageNet pretrained)
        - Unfrozen last 2 blocks (layer3, layer4) for fine-tuning
        - Custom head: FC(2048 -> 512) + BN + ReLU + Dropout + FC(512 -> 7)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = CV_CONFIG["dropout"],
        pretrained: bool = CV_CONFIG["pretrained"],
        freeze_until: str = "layer2",
    ):
        super().__init__()
        # Load backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Freeze layers up to freeze_until
        freeze = True
        for name, param in backbone.named_parameters():
            if freeze_until in name:
                freeze = False
            param.requires_grad = not freeze

        # Remove original FC
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Custom classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-dim feature vector before final FC (for ML block)."""
        features = self.backbone(x)
        flat = nn.Flatten()(features)
        for layer in list(self.head.children())[:-1]:  # all but last Linear
            flat = layer(flat)
        return flat

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (7-dim vector per sample)."""
        prev_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        self.train(prev_training)
        return torch.softmax(logits, dim=-1)


class _ColabResNet50(nn.Module):
    """Matches checkpoint saved in Colab: flat ResNet keys + fc Sequential head (2048→num_classes)."""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc      = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, num_classes))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def get_probabilities(self, x):
        return torch.softmax(self.forward(x), dim=-1)

    def get_feature_vector(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Load a trained model from checkpoint, handling both local and Colab-saved formats."""
    state = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    # Detect Colab format: keys start with conv1/layer* instead of backbone.*
    if any(k.startswith("conv1") or k.startswith("layer1") for k in state.keys()):
        model = _ColabResNet50(num_classes=num_classes)
    else:
        model = SkinLesionResNet50(num_classes=num_classes)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


if __name__ == "__main__":
    model = SkinLesionResNet50()
    params = count_parameters(model)
    print(f"Model: ResNet50 -> {NUM_CLASSES} classes")
    print(f"Parameters: {params}")
    # Sanity check forward pass
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    probs = model.get_probabilities(dummy)
    feat  = model.get_feature_vector(dummy)
    print(f"Output shape:    {out.shape}")
    print(f"Probs shape:     {probs.shape}")
    print(f"Features shape:  {feat.shape}")
    print("All checks passed.")
