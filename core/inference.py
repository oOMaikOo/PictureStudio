"""
Enhanced inference: top-k predictions, confidence filtering, ROI template support.
"""
import os
from typing import List, Dict, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from utils.config import IMAGE_FORMATS
from utils.logging_utils import get_logger

log = get_logger()


class Inferencer:
    """Load a model checkpoint and classify images or ROI crops."""

    def __init__(self):
        self.model = None
        self.class_names: List[str] = []
        self.image_size: int = 224
        self.model_path: str = ""
        self.model_type: str = ""
        self.device = None
        self.transform = None
        self.metadata: Dict = {}

    # ------------------------------------------------------------------ setup

    def load_model(self, model_path: str) -> Dict:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch ist nicht installiert.")
        from models.classifier import create_model, load_checkpoint

        raw = torch.load(model_path, map_location="cpu", weights_only=False)
        meta = raw.get("metadata", {})
        self.class_names = meta.get("class_names", [])
        self.model_type = meta.get("model_type", "resnet18")
        self.image_size = meta.get("image_size", 224)
        self.metadata = meta

        if not self.class_names:
            raise ValueError("Checkpoint enthält keine Klasseninformationen.")

        self.model = create_model(self.model_type, len(self.class_names), pretrained=False)
        load_checkpoint(self.model, model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model_path = model_path
        log.info("Modell geladen: %s | Klassen: %s", model_path, self.class_names)
        return meta

    # ------------------------------------------------------------------ prediction

    def predict_image(
        self,
        image_path: str,
        roi: Optional[Dict] = None,
        top_k: int = 3,
    ) -> Dict:
        """
        Returns dict with keys:
          predicted_label, confidence, top_k (list of {label, prob}), all_probs
        """
        if not self.is_ready():
            raise RuntimeError("Kein Modell geladen.")

        image = Image.open(image_path).convert("RGB")
        if roi is not None:
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            image = image.crop((max(0, x), max(0, y), x + w, y + h))

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().tolist()

        k = min(top_k, len(self.class_names))
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top = [{"label": self.class_names[i], "prob": round(p, 4)} for i, p in indexed[:k]]

        return {
            "predicted_label": top[0]["label"],
            "confidence": round(top[0]["prob"], 4),
            "top_k": top,
            "all_probs": {cls: round(p, 4) for cls, p in zip(self.class_names, probs)},
        }

    def predict_folder(
        self,
        folder_path: str,
        roi_templates: List[Dict] = None,
        top_k: int = 3,
        progress_callback=None,
    ) -> List[Dict]:
        """Classify all images in a folder. Optional ROI templates applied per image."""
        from datetime import datetime

        files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_FORMATS
        ])
        results = []

        for i, fname in enumerate(files):
            img_path = os.path.join(folder_path, fname)
            roi = None
            if roi_templates:
                roi = roi_templates[0].get("roi") if roi_templates else None

            try:
                pred = self.predict_image(img_path, roi=roi, top_k=top_k)
                results.append({
                    "filename": fname,
                    "path": img_path,
                    "predicted_label": pred["predicted_label"],
                    "confidence": pred["confidence"],
                    "top_k": pred["top_k"],
                    "all_probs": pred["all_probs"],
                    "model_path": self.model_path,
                    "model_type": self.model_type,
                    "timestamp": datetime.now().isoformat(),
                    "low_confidence": pred["confidence"] < 0.70,
                    "error": None,
                })
            except Exception as exc:
                log.warning("Fehler bei %s: %s", fname, exc)
                results.append({
                    "filename": fname,
                    "path": img_path,
                    "predicted_label": "ERROR",
                    "confidence": 0.0,
                    "top_k": [],
                    "all_probs": {},
                    "model_path": self.model_path,
                    "model_type": self.model_type,
                    "timestamp": "",
                    "low_confidence": True,
                    "error": str(exc),
                })

            if progress_callback:
                progress_callback(i + 1, len(files))

        return results

    def filter_results(
        self,
        results: List[Dict],
        min_confidence: float = 0.0,
        label_filter: str = "",
        only_low_confidence: bool = False,
    ) -> List[Dict]:
        out = results
        if label_filter:
            out = [r for r in out if r.get("predicted_label") == label_filter]
        if min_confidence > 0:
            out = [r for r in out if r.get("confidence", 0) >= min_confidence]
        if only_low_confidence:
            out = [r for r in out if r.get("low_confidence", False)]
        return out

    def is_ready(self) -> bool:
        return self.model is not None
