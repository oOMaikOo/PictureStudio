"""
Grad-CAM: Gradient-weighted Class Activation Mapping.
Shows which image regions drove the model's prediction.

Supports: ResNet-18/50, MobileNetV2, EfficientNet-B0, SimpleCNN.
"""
from typing import Optional, Tuple
from utils.logging_utils import get_logger

log = get_logger()

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image as PILImage
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Layer selection per architecture
# ---------------------------------------------------------------------------

def get_target_layer(model, model_type: str):
    """
    Return the last convolutional layer suited for Grad-CAM.

    Selects a model-type-specific layer by name for known architectures
    (ResNet-18/50, MobileNetV2, EfficientNet-B0, SimpleCNN). Falls back
    to scanning all modules for the last ``nn.Conv2d`` when the
    architecture is not recognised.
    """
    mt = (model_type or "").lower()
    try:
        if mt in ("resnet18", "resnet50"):
            return model.layer4[-1]
        elif mt == "mobilenet_v2":
            return model.features[-1]
        elif mt == "efficientnet_b0":
            return model.features[-1]
        elif mt == "simple_cnn":
            # SimpleCNN.features: last real conv before AdaptiveAvgPool
            import torch.nn as nn
            layers = list(model.features.children())
            for layer in reversed(layers):
                if isinstance(layer, nn.Conv2d):
                    return layer
    except Exception:
        pass
    # Fallback: find last Conv2d in the whole model
    import torch.nn as nn
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


# ---------------------------------------------------------------------------
# GradCAM engine
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Computes a Grad-CAM heatmap for a given image and class.
    Usage:
        gcam = GradCAM(model, target_layer)
        cam  = gcam.compute(input_tensor, class_idx)
        gcam.remove()
    """

    def __init__(self, model, target_layer):
        """Register forward and backward hooks on *target_layer*."""
        self._model  = model
        self._feats  = None
        self._grads  = None
        self._fwd    = target_layer.register_forward_hook(self._hook_fwd)
        self._bwd    = target_layer.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, module, inp, out):
        """Store the forward feature maps produced by the target layer."""
        self._feats = out.detach()

    def _hook_bwd(self, module, grad_in, grad_out):
        """Store the gradients flowing back through the target layer."""
        self._grads = grad_out[0].detach()

    def compute(self, input_tensor, class_idx: Optional[int] = None):
        """
        Returns a 2-D numpy array in [0, 1] (h × w), same size as input_tensor.
        """
        self._model.eval()
        # Need gradients – do NOT use torch.no_grad()
        self._model.zero_grad()
        output = self._model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1))
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims
        weights = self._grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._feats).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam * 0.0
        return cam

    def remove(self):
        """Deregister both hooks. Always call this after ``compute()``."""
        self._fwd.remove()
        self._bwd.remove()


# ---------------------------------------------------------------------------
# Colormap (jet) without matplotlib dependency
# ---------------------------------------------------------------------------

def _jet_colormap(value: float) -> Tuple[int, int, int]:
    """Map [0,1] → RGB using the jet colormap."""
    v = max(0.0, min(1.0, value))
    if v < 0.125:
        r, g, b = 0, 0, 0.5 + 4 * v
    elif v < 0.375:
        r, g, b = 0, 4 * (v - 0.125), 1
    elif v < 0.625:
        r, g, b = 4 * (v - 0.375), 1, 1 - 4 * (v - 0.375)
    elif v < 0.875:
        r, g, b = 1, 1 - 4 * (v - 0.625), 0
    else:
        r, g, b = 1 - 4 * (v - 0.875), 0, 0
    return (int(r * 255), int(g * 255), int(b * 255))


def cam_to_heatmap_pil(cam_array) -> "PILImage.Image":
    """
    Convert a 2-D [0,1] CAM array to a jet-coloured PIL RGBA image.

    The alpha channel is proportional to activation strength so that
    low-activation regions remain nearly transparent when composited.
    """
    h, w = cam_array.shape
    rgba = []
    for row in range(h):
        for col in range(w):
            v = float(cam_array[row, col])
            r, g, b = _jet_colormap(v)
            alpha = int(v * 200)          # transparent where not activated
            rgba.extend([r, g, b, alpha])
    img = PILImage.frombytes("RGBA", (w, h), bytes(rgba))
    return img


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

_TRANSFORM = None


def _get_transform(image_size: int):
    """Return (and cache) the inference pre-processing transform for *image_size*."""
    global _TRANSFORM
    if _TRANSFORM is None or _TRANSFORM.__image_size != image_size:
        _TRANSFORM = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _TRANSFORM.__image_size = image_size
    return _TRANSFORM


def compute_gradcam_overlay(
    model,
    model_type: str,
    image_path: str,
    class_idx: Optional[int],
    image_size: int = 224,
    alpha: float = 0.5,
    roi: Optional[dict] = None,
) -> Tuple["PILImage.Image", "PILImage.Image"]:
    """
    Compute Grad-CAM for image_path and return:
        (original_pil, overlay_pil)
    Both are RGB PIL images at the analysis region's resolution.

    If *roi* is provided (dict with x, y, w, h keys in pixel coords) the image
    is cropped to that region before the CAM is computed.  The model was trained
    on ROI crops, so passing the matching ROI here ensures the activation map is
    meaningful.
    """
    if not HAS_TORCH or not HAS_PIL:
        raise RuntimeError("PyTorch und Pillow sind erforderlich.")

    original = PILImage.open(image_path).convert("RGB")

    # Crop to the labeling ROI so the model sees the same input it was trained on
    if roi is not None:
        x = int(roi.get("x", 0)); y = int(roi.get("y", 0))
        w = int(roi.get("w", original.width)); h = int(roi.get("h", original.height))
        x2 = min(original.width, x + w); y2 = min(original.height, y + h)
        if x2 > x and y2 > y:
            original = original.crop((x, y, x2, y2))

    orig_w, orig_h = original.size

    transform = _get_transform(image_size)
    inp = transform(original).unsqueeze(0)

    # Move input to same device as model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None
    if device is not None:
        inp = inp.to(device)

    target_layer = get_target_layer(model, model_type)
    if target_layer is None:
        raise RuntimeError("Kein geeigneter Conv-Layer für Grad-CAM gefunden.")

    gcam = GradCAM(model, target_layer)
    try:
        cam = gcam.compute(inp, class_idx)
    finally:
        gcam.remove()

    # Resize CAM to original image dimensions
    cam_pil = PILImage.fromarray((cam * 255).astype("uint8"), mode="L")
    cam_pil = cam_pil.resize((orig_w, orig_h), PILImage.BILINEAR)
    cam_arr = [p / 255.0 for p in cam_pil.getdata()]

    heatmap = PILImage.new("RGBA", (orig_w, orig_h))
    rgba_data = []
    for v in cam_arr:
        r, g, b = _jet_colormap(v)
        rgba_data.append((r, g, b, int(v * 220)))
    heatmap.putdata(rgba_data)

    # Composite heatmap over original
    orig_rgba = original.convert("RGBA")
    overlay = PILImage.alpha_composite(orig_rgba, heatmap).convert("RGB")

    return original, overlay
