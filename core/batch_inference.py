"""
Batch inference worker: classifies a list of images through a loaded PyTorch model.
Called from a QThread; reports progress via callbacks.
"""
import os
from typing import List, Dict, Callable, Optional

try:
    import torch
    from torchvision import transforms
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BatchInferenceWorker:
    """Runs classification on a list of image paths. Thread-safe, cancellable."""

    def __init__(
        self,
        model,
        class_names: List[str],
        image_size: int = 224,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ):
        self.model = model
        self.class_names = class_names
        self.image_size = image_size
        self.progress_cb = progress_cb
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self, image_paths: List[str]) -> List[Dict]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch nicht verfügbar.")

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self.model.eval()
        results: List[Dict] = []
        total = len(image_paths)

        with torch.no_grad():
            for i, path in enumerate(image_paths):
                if self._cancelled:
                    break
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(device)
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1)[0].cpu().tolist()
                    pred_idx = int(max(range(len(probs)), key=lambda j: probs[j]))
                    pred_label = (
                        self.class_names[pred_idx]
                        if pred_idx < len(self.class_names)
                        else str(pred_idx)
                    )
                    confidence = probs[pred_idx]
                    results.append({
                        "path": path,
                        "filename": os.path.basename(path),
                        "predicted": pred_label,
                        "confidence": confidence,
                        "probabilities": {
                            self.class_names[j]: probs[j]
                            for j in range(min(len(self.class_names), len(probs)))
                        },
                        "error": None,
                    })
                except Exception as exc:
                    results.append({
                        "path": path,
                        "filename": os.path.basename(path),
                        "predicted": "FEHLER",
                        "confidence": 0.0,
                        "probabilities": {},
                        "error": str(exc),
                    })

                if self.progress_cb:
                    self.progress_cb(i + 1, total)

        return results
