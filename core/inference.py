"""
Enhanced inference: top-k predictions, confidence filtering, ROI template support.
"""
import hashlib
import json
import os
from typing import List, Dict, Optional, Tuple

from utils.config import IMAGE_FORMATS
from utils.logging_utils import get_logger

log = get_logger("ImageLabelingStudio.inference")

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


class Inferencer:
    """
    Loads a trained .pth checkpoint and exposes prediction methods.

    Used by InferencePage, BatchInferencePage, and the REST API server.
    Supports single-image prediction, ROI-cropped prediction, folder batch
    classification, and optional Test-Time Augmentation (TTA).
    """

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

    @staticmethod
    def verify_checksum(model_path: str) -> tuple:
        """Return (ok, message). Skips check if no .sha256 sidecar exists."""
        sidecar = model_path + ".sha256"
        if not os.path.exists(sidecar):
            return True, "Keine Prüfsumme vorhanden (älteres Modell)"
        try:
            stored = json.load(open(sidecar, encoding="utf-8"))["sha256"]
            actual = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
            if actual == stored:
                return True, f"SHA256 OK: {actual[:16]}…"
            return False, (
                f"PRÜFSUMME UNGÜLTIG!\n"
                f"Erwartet: {stored[:16]}…\nGefunden: {actual[:16]}…"
            )
        except Exception as exc:
            return False, f"Prüfsummen-Fehler: {exc}"

    def load_model(self, model_path: str) -> Dict:
        """
        Load a checkpoint from *model_path*.

        Verifies the SHA256 checksum sidecar (if present) before loading.
        Reads class_names, model_type, and image_size from the checkpoint
        metadata, creates the matching architecture, and moves it to the
        best available device. Returns the metadata dict.
        Raises ValueError if the checkpoint has no class information.
        Raises RuntimeError if the checksum does not match.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch ist nicht installiert.")
        from models.classifier import create_model, load_checkpoint

        ok, msg = self.verify_checksum(model_path)
        if not ok:
            raise RuntimeError(f"Integritätsprüfung fehlgeschlagen:\n{msg}")
        log.debug("Checksumme: %s", msg)

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
        tta_passes: int = 1,
    ) -> Dict:
        """
        Returns dict with keys:
          predicted_label, confidence, top_k (list of {label, prob}), all_probs

        tta_passes > 1 activates Test-Time Augmentation: the image is run through
        multiple randomly augmented versions and probabilities are averaged.
        """
        if not self.is_ready():
            raise RuntimeError("Kein Modell geladen.")

        image = Image.open(image_path).convert("RGB")
        if roi is not None:
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            image = image.crop((max(0, x), max(0, y), x + w, y + h))

        if tta_passes > 1:
            # Build a randomized TTA transform (in addition to the base transform)
            tta_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensors = [self.transform(image).unsqueeze(0)]
            for _ in range(tta_passes - 1):
                tensors.append(tta_tf(image).unsqueeze(0))
            batch = torch.cat(tensors, dim=0).to(self.device)
            with torch.no_grad():
                probs_all = F.softmax(self.model(batch), dim=1).cpu()
            probs = probs_all.mean(dim=0).tolist()
        else:
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
            "tta_passes": tta_passes,
        }

    def classify_single(self, image_path: str, top_k: int = 3) -> Dict:
        """Alias for the REST API. Same as predict_image without ROI."""
        result = self.predict_image(image_path, top_k=top_k)
        result["low_confidence"] = result["confidence"] < 0.70
        result["path"] = image_path
        return result

    def predict_folder(
        self,
        folder_path: str,
        roi_templates: List[Dict] = None,
        top_k: int = 3,
        progress_callback=None,
        tta_passes: int = 1,
        recursive: bool = False,
    ) -> List[Dict]:
        """Classify all images in a folder (and optionally all subfolders)."""
        from datetime import datetime

        if recursive:
            file_pairs: List[tuple] = []
            for root, _dirs, files in os.walk(folder_path):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in IMAGE_FORMATS:
                        full = os.path.join(root, f)
                        try:
                            rel_dir = os.path.relpath(root, folder_path)
                            display = f if rel_dir == "." else f"{rel_dir}/{f}"
                        except ValueError:
                            display = f
                        file_pairs.append((display, full))
            file_pairs.sort(key=lambda x: x[0])
        else:
            file_pairs = [
                (f, os.path.join(folder_path, f))
                for f in sorted(os.listdir(folder_path))
                if os.path.splitext(f)[1].lower() in IMAGE_FORMATS
            ]

        results = []
        roi = roi_templates[0].get("roi") if roi_templates else None

        for i, (display_name, img_path) in enumerate(file_pairs):
            try:
                pred = self.predict_image(img_path, roi=roi, top_k=top_k, tta_passes=tta_passes)
                results.append({
                    "filename": display_name,
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
                log.warning("Fehler bei %s: %s", display_name, exc)
                results.append({
                    "filename": display_name,
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
                progress_callback(i + 1, len(file_pairs))

        return results

    def filter_results(
        self,
        results: List[Dict],
        min_confidence: float = 0.0,
        label_filter: str = "",
        only_low_confidence: bool = False,
    ) -> List[Dict]:
        """
        Filter a list of result dicts by label and/or confidence.

        Parameters
        ----------
        results            : As returned by predict_folder().
        min_confidence     : Exclude results with confidence below this value.
        label_filter       : When non-empty, keep only results with this predicted label.
        only_low_confidence: When True, keep only results marked as low_confidence.
        """
        out = results
        if label_filter:
            out = [r for r in out if r.get("predicted_label") == label_filter]
        if min_confidence > 0:
            out = [r for r in out if r.get("confidence", 0) >= min_confidence]
        if only_low_confidence:
            out = [r for r in out if r.get("low_confidence", False)]
        return out

    def is_ready(self) -> bool:
        """Return True when a model has been successfully loaded."""
        return self.model is not None
