"""
src/cv/grad_cam.py - Grad-CAM visualization for model explainability
ZHAW AI-Applications Project
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from typing import Union, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASS_NAMES, HAM10000_CLASSES, CV_CONFIG
from src.cv.model import SkinLesionResNet50, load_model
from src.cv.preprocessing import get_inference_transforms


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet50.

    Selve-Zeiler et al., 2016 / Selvaraju et al., 2017

    Usage:
        gradcam = GradCAM(model, target_layer="layer4")
        heatmap = gradcam(image, class_idx=None)  # None = predicted class
    """

    def __init__(self, model: SkinLesionResNet50, target_layer: str = "layer4"):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        # Find target layer in backbone
        layer = None
        for name, module in self.model.backbone.named_modules():
            if self.target_layer in name and isinstance(module, torch.nn.Sequential):
                layer = module
        if layer is None:
            # Fallback: last conv layer in backbone
            for name, module in self.model.backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    layer = module
        if layer is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")
        self._layer = layer

        def save_gradient(grad):
            self.gradients = grad.detach()

        def save_activation(module, inp, out):
            self.activations = out.detach()
            out.register_hook(save_gradient)

        self._layer.register_forward_hook(save_activation)

    def __call__(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap.

        Returns:
            heatmap: (H, W) array in [0, 1]
            class_idx: predicted or specified class
            confidence: probability of that class
        """
        transform = get_inference_transforms()
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            img_pil = None
            tensor  = image.unsqueeze(0) if image.dim() == 3 else image
        else:
            img_pil = image.convert("RGB")

        if img_pil is not None:
            tensor = transform(img_pil).unsqueeze(0)

        device = next(self.model.parameters()).device
        tensor = tensor.to(device).requires_grad_(True)

        self.model.eval()
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=-1)

        if class_idx is None:
            class_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, class_idx].item())

        # Backprop on target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # (C,)
        activations  = self.activations[0]                  # (C, H, W)

        # Weight activations
        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx, confidence


def overlay_heatmap(
    image: Union[str, Path, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    if isinstance(image, (str, Path)):
        img = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image

    H, W = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.float32(img) * (1 - alpha) + np.float32(heatmap_colored) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def visualize_grad_cam(
    image: Union[str, Path, Image.Image],
    model: SkinLesionResNet50,
    class_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    """Full Grad-CAM visualization pipeline."""
    if isinstance(image, (str, Path)):
        img_pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image).convert("RGB")
    else:
        img_pil = image.convert("RGB")

    gradcam    = GradCAM(model, target_layer="layer4")
    heatmap, pred_idx, conf = gradcam(img_pil, class_idx=class_idx)
    overlay    = overlay_heatmap(img_pil, heatmap)

    pred_label = CLASS_NAMES[pred_idx]
    pred_name  = HAM10000_CLASSES[pred_label]

    if show or save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_pil)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\n{pred_name} ({conf:.2%})")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    return overlay
