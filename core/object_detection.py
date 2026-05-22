"""
Object Detection: YOLOv8 wrapper for training and inference.

Optional dependency: ultralytics (pip install ultralytics)
"""
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Callable

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import torch
    from torch.utils.data import DataLoader  # noqa: F401 (keep torch import guard)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from utils.logging_utils import get_logger

log = get_logger()


def has_ultralytics() -> bool:
    return HAS_ULTRALYTICS


# ------------------------------------------------------------------ detector


class ObjectDetector:
    """YOLOv8-based object detector — thin wrapper around ultralytics."""

    MODEL_SIZES = {
        "yolov8n": "Nano  — sehr schnell, ~3 M Parameter",
        "yolov8s": "Small — schnell, ~11 M Parameter",
        "yolov8m": "Medium — ausgewogen, ~26 M Parameter",
        "yolov8l": "Large — langsam, sehr genau, ~44 M Parameter",
    }

    def __init__(self):
        self._model = None
        self.model_path: Optional[str] = None
        self.class_names: List[str] = []

    def is_ready(self) -> bool:
        return self._model is not None

    # ---------------------------------------------------------------- load / save

    def load(self, path: str) -> "ObjectDetector":
        if not HAS_ULTRALYTICS:
            raise RuntimeError("ultralytics nicht installiert. Bitte: pip install ultralytics")
        self._model = YOLO(path)
        self.model_path = path
        self.class_names = list(self._model.names.values())
        log.info("Detektionsmodell geladen: %s | Klassen: %s", path, self.class_names)
        return self

    # ---------------------------------------------------------------- inference

    def predict_image(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> List[Dict]:
        """
        Returns list of dicts:
          label, confidence, x1, y1, x2, y2, x, y, w, h
        """
        if not self.is_ready():
            raise RuntimeError("Kein Detektionsmodell geladen.")
        results = self._model(image_path, conf=conf, iou=iou, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0]
                coords = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
                x1, y1, x2, y2 = [float(v) for v in coords]
                cls_val  = box.cls[0] if hasattr(box.cls, "__getitem__") else box.cls
                conf_val = box.conf[0] if hasattr(box.conf, "__getitem__") else box.conf
                detections.append({
                    "label":      self.class_names[int(cls_val)],
                    "confidence": round(float(conf_val), 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x": x1,  "y": y1,
                    "w": x2 - x1, "h": y2 - y1,
                })
        return sorted(detections, key=lambda d: d["confidence"], reverse=True)

    def predict_folder(
        self,
        folder_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        progress_callback: Optional[Callable] = None,
        recursive: bool = False,
    ) -> List[Dict]:
        """Classify all images in folder; returns flat list with filename added."""
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        if recursive:
            pairs = []
            for root, _, files in os.walk(folder_path):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                        full = os.path.join(root, f)
                        rel = os.path.relpath(root, folder_path)
                        pairs.append((f if rel == "." else f"{rel}/{f}", full))
        else:
            pairs = [
                (f, os.path.join(folder_path, f))
                for f in sorted(os.listdir(folder_path))
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]

        results = []
        for i, (display, path) in enumerate(pairs):
            try:
                dets = self.predict_image(path, conf=conf, iou=iou)
                results.append({
                    "filename":   display,
                    "path":       path,
                    "detections": dets,
                    "n_objects":  len(dets),
                    "error":      None,
                })
            except Exception as exc:
                results.append({
                    "filename": display, "path": path,
                    "detections": [], "n_objects": 0,
                    "error": str(exc),
                })
            if progress_callback:
                progress_callback(i + 1, len(pairs))
        return results

    # ---------------------------------------------------------------- export

    def export_onnx(self, output_path: str) -> str:
        if not self.is_ready():
            raise RuntimeError("Kein Modell geladen.")
        export_path = self._model.export(format="onnx")
        shutil.copy(export_path, output_path)
        return output_path


# ------------------------------------------------------------------ QThread wrappers

try:
    from PySide6.QtCore import QThread, Signal as _Signal

    class DetectionTrainingThread(QThread):
        """Run YOLO training in a background thread."""

        progress = _Signal(int, int, float, float)  # epoch, total, box_loss, cls_loss
        log_line = _Signal(str)
        finished = _Signal(str)   # best model path
        error    = _Signal(str)

        def __init__(
            self,
            data_yaml: str,
            model_size: str = "yolov8n",
            epochs: int = 50,
            imgsz: int = 640,
            batch: int = 16,
            device: str = "auto",
            project_dir: str = "",
            parent=None,
        ):
            super().__init__(parent)
            self.data_yaml  = data_yaml
            self.model_size = model_size
            self.epochs     = epochs
            self.imgsz      = imgsz
            self.batch      = batch
            self.device     = device
            self.project_dir = project_dir
            self._stop_flag = False

        def stop(self):
            self._stop_flag = True

        def run(self):
            if not HAS_ULTRALYTICS:
                self.error.emit("ultralytics nicht installiert. Bitte: pip install ultralytics")
                return
            try:
                model = YOLO(f"{self.model_size}.pt")

                epoch_counter = [0]

                def on_epoch_end(trainer):
                    epoch_counter[0] += 1
                    ep   = epoch_counter[0]
                    tot  = self.epochs
                    loss = trainer.loss.item() if hasattr(trainer.loss, "item") else float(trainer.loss)
                    metrics = getattr(trainer, "metrics", {})
                    box_loss = float(metrics.get("train/box_loss", loss))
                    cls_loss = float(metrics.get("train/cls_loss", 0.0))
                    self.progress.emit(ep, tot, box_loss, cls_loss)
                    self.log_line.emit(
                        f"[Epoch {ep:>3}/{tot}]  box_loss={box_loss:.4f}  cls_loss={cls_loss:.4f}"
                    )
                    if self._stop_flag:
                        trainer.epoch = trainer.epochs  # force stop

                model.add_callback("on_train_epoch_end", on_epoch_end)

                device = self.device
                if device == "auto":
                    if HAS_TORCH:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            device = "0"
                        elif _torch.backends.mps.is_available():
                            device = "mps"
                        else:
                            device = "cpu"
                    else:
                        device = "cpu"

                save_dir = os.path.join(self.project_dir, "detection_runs") if self.project_dir else None
                result = model.train(
                    data=self.data_yaml,
                    epochs=self.epochs,
                    imgsz=self.imgsz,
                    batch=self.batch,
                    device=device,
                    project=save_dir,
                    verbose=False,
                )

                best = str(result.save_dir / "weights" / "best.pt") if result else ""
                if not best or not os.path.exists(best):
                    # Fallback: search for best.pt
                    for root, _, files in os.walk(save_dir or "."):
                        if "best.pt" in files:
                            best = os.path.join(root, "best.pt")
                            break
                self.finished.emit(best)

            except Exception as exc:
                self.error.emit(str(exc))

    class DetectionInferenceThread(QThread):
        """Run folder detection in a background thread."""

        progress = _Signal(int, int)
        finished = _Signal(list)
        error    = _Signal(str)

        def __init__(self, detector: ObjectDetector, folder: str,
                     conf: float = 0.25, recursive: bool = False, parent=None):
            super().__init__(parent)
            self.detector  = detector
            self.folder    = folder
            self.conf      = conf
            self.recursive = recursive

        def run(self):
            try:
                results = self.detector.predict_folder(
                    self.folder,
                    conf=self.conf,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                    recursive=self.recursive,
                )
                self.finished.emit(results)
            except Exception as exc:
                self.error.emit(str(exc))

except ImportError:
    pass  # PySide6 not available (e.g., in test environments without Qt)
