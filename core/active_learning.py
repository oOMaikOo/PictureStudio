"""
Active Learning: score unlabeled images by model uncertainty, fill the AL queue.

Uses the trained Inferencer to run inference on unlabeled images and returns
the most uncertain candidates (lowest confidence) for human review.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from utils.logging_utils import get_logger

log = get_logger()

try:
    from PySide6.QtCore import QThread, Signal as _Signal
    HAS_QT = True
except ImportError:
    HAS_QT = False


class ActiveLearningSampler:
    """
    Runs inference on a list of image paths and returns the most uncertain ones.

    Uncertainty is measured as 1 - max_confidence. Images below
    `confidence_threshold` are considered uncertain; the result is sorted from
    most uncertain to least uncertain and capped at `n_samples`.
    """

    def run(
        self,
        model_path: str,
        image_paths: List[str],
        confidence_threshold: float = 0.70,
        n_samples: int = 50,
        roi_template: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict]:
        """
        Returns a list of dicts sorted by uncertainty (most uncertain first):
          {path, predicted_label, confidence, uncertainty}
        """
        from core.inference import Inferencer

        inf = Inferencer()
        inf.load_model(model_path)

        results = []
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            try:
                pred = inf.predict_image(path, roi=roi_template, top_k=1)
                conf = pred["confidence"]
                results.append({
                    "path":            path,
                    "predicted_label": pred["predicted_label"],
                    "confidence":      round(conf, 4),
                    "uncertainty":     round(1.0 - conf, 4),
                })
            except Exception as exc:
                log.warning("AL-Scan: überspringe %s — %s", path, exc)
            if progress_callback:
                progress_callback(i + 1, total)

        uncertain = [r for r in results if r["confidence"] < confidence_threshold]
        uncertain.sort(key=lambda r: r["confidence"])
        return uncertain[:n_samples]


if HAS_QT:
    class ActiveLearningThread(QThread):
        progress = _Signal(int, int)
        finished = _Signal(list)
        error    = _Signal(str)

        def __init__(
            self,
            model_path: str,
            image_paths: List[str],
            confidence_threshold: float = 0.70,
            n_samples: int = 50,
            roi_template: Optional[Dict] = None,
            parent=None,
        ):
            super().__init__(parent)
            self.model_path           = model_path
            self.image_paths          = image_paths
            self.confidence_threshold = confidence_threshold
            self.n_samples            = n_samples
            self.roi_template         = roi_template
            self._stop                = False

        def request_stop(self) -> None:
            self._stop = True

        def run(self) -> None:
            try:
                sampler = ActiveLearningSampler()

                def _progress(c: int, t: int) -> None:
                    if not self._stop:
                        self.progress.emit(c, t)

                results = sampler.run(
                    self.model_path,
                    self.image_paths,
                    self.confidence_threshold,
                    self.n_samples,
                    self.roi_template,
                    progress_callback=_progress,
                )
                if not self._stop:
                    self.finished.emit(results)
            except Exception as exc:
                self.error.emit(str(exc))
