"""
Pre-Labeling: run a trained classification model on unlabeled project images
and suggest labels with confidence scores.

No extra dependencies — reuses core/inference.py (torch + torchvision).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from utils.logging_utils import get_logger

log = get_logger()


class PreLabeler:
    """
    Wraps an Inferencer to suggest labels for unlabeled images.

    Usage:
        pl = PreLabeler()
        pl.load_model(path)
        suggestions = pl.suggest(project, confidence_threshold=0.75)
    """

    def __init__(self):
        self._inferencer = None
        self.model_path: str = ""
        self.class_names: List[str] = []

    def is_ready(self) -> bool:
        return self._inferencer is not None and self._inferencer.is_ready()

    def load_model(self, path: str) -> Dict:
        from core.inference import Inferencer
        inf = Inferencer()
        meta = inf.load_model(path)
        self._inferencer = inf
        self.model_path  = path
        self.class_names = inf.class_names
        return meta

    def suggest(
        self,
        image_paths: List[str],
        project_labels: List[str],
        confidence_threshold: float = 0.75,
        roi: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict]:
        """
        Run inference on *image_paths* and return suggestion dicts.

        Each dict: {path, label, confidence, skip}
          skip=True  → confidence below threshold (suggestion held back)
          skip=False → label + confidence above threshold

        Only labels that exist in *project_labels* are accepted; images whose
        top-1 label is unknown to the project are marked skip=True.
        """
        if not self.is_ready():
            raise RuntimeError("Kein Modell geladen.")

        known = set(project_labels)
        results = []
        for i, path in enumerate(image_paths):
            try:
                pred = self._inferencer.predict_image(path, roi=roi, top_k=1)
                label = pred["predicted_label"]
                conf  = pred["confidence"]
                skip  = conf < confidence_threshold or label not in known
                results.append({
                    "path":       path,
                    "label":      label,
                    "confidence": round(conf, 4),
                    "skip":       skip,
                    "error":      None,
                })
            except Exception as exc:
                results.append({
                    "path": path, "label": "", "confidence": 0.0,
                    "skip": True, "error": str(exc),
                })
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
        return results


# ------------------------------------------------------------------ QThread

try:
    from PySide6.QtCore import QThread, Signal as _Signal

    class PreLabelingThread(QThread):
        """Run pre-labeling in a background thread."""

        progress  = _Signal(int, int)
        finished  = _Signal(list)   # list of suggestion dicts
        error     = _Signal(str)

        def __init__(
            self,
            pre_labeler: PreLabeler,
            image_paths: List[str],
            project_labels: List[str],
            confidence_threshold: float = 0.75,
            roi: Optional[Dict] = None,
            parent=None,
        ):
            super().__init__(parent)
            self.pre_labeler          = pre_labeler
            self.image_paths          = image_paths
            self.project_labels       = project_labels
            self.confidence_threshold = confidence_threshold
            self.roi                  = roi

        def run(self):
            try:
                results = self.pre_labeler.suggest(
                    self.image_paths,
                    self.project_labels,
                    confidence_threshold=self.confidence_threshold,
                    roi=self.roi,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                )
                self.finished.emit(results)
            except Exception as exc:
                self.error.emit(str(exc))

except ImportError:
    pass
