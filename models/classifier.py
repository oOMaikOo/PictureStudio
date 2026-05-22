from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from utils.logging_utils import get_logger

log = get_logger()


def create_model(model_type: str, num_classes: int, pretrained: bool = True):
    """Return a PyTorch model adapted for num_classes outputs."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch ist nicht installiert.")

    model_type = model_type.lower()

    weights_arg = "DEFAULT" if pretrained else None

    if model_type == "resnet18":
        model = models.resnet18(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "resnet50":
        model = models.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights_arg)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights_arg)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == "convnext_tiny":
        model = models.convnext_tiny(weights=weights_arg)
        # ConvNeXt head: Sequential(LayerNorm, Flatten, Linear)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_type == "simple_cnn":
        model = SimpleCNN(num_classes)

    else:
        raise ValueError(f"Unbekannte Modellarchitektur: {model_type}")

    log.info("Modell '%s' erstellt mit %d Klassen (pretrained=%s)", model_type, num_classes, pretrained)
    return model


class SimpleCNN(nn.Module if HAS_TORCH else object):
    """Lightweight baseline CNN – useful when no internet for pretrained weights."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def save_checkpoint(model, path: str, metadata: dict = None) -> None:
    import torch
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    log.info("Checkpoint gespeichert: %s", path)


def load_checkpoint(model, path: str):
    import torch
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    return payload.get("metadata", {})


def get_available_models():
    return [
        "resnet18", "resnet50", "mobilenet_v2",
        "efficientnet_b0", "efficientnet_b3",
        "convnext_tiny",
        "simple_cnn",
    ]
