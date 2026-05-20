from __future__ import annotations
import logging
import os
import random
from typing import Optional, Callable

log = logging.getLogger(__name__)

_PIL_AVAILABLE = False
try:
    from PIL import Image as _PILImage, ImageEnhance as _PILEnhance, ImageFilter as _PILFilter
    _PIL_AVAILABLE = True
except ImportError:
    pass


class AugmentationPipeline:
    """Erstellt augmentierte Kopien gelabelter Bilder für Daten-Erweiterung."""

    DEFAULT_CONFIG = {
        "rotation": True,
        "flip_h": True,
        "flip_v": False,
        "brightness": True,
        "contrast": True,
        "blur": False,
        "noise": False,
        "copies_per_image": 3,
    }

    def __init__(self, config: dict | None = None) -> None:
        self._config = {**self.DEFAULT_CONFIG, **(config or {})}

    def augment_image(self, image_path: str, output_dir: str) -> list[str]:
        """Erstellt N augmentierte Kopien. Gibt Liste der neuen Pfade zurück."""
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow ist nicht installiert.")
        n = int(self._config.get("copies_per_image", 3))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        suffix = os.path.splitext(image_path)[1] or ".png"
        os.makedirs(output_dir, exist_ok=True)
        results = []
        with _PILImage.open(image_path) as orig:
            orig = orig.convert("RGB")
            for i in range(n):
                img = orig.copy()
                if self._config.get("rotation"):
                    angle = random.uniform(-15, 15)
                    img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
                if self._config.get("flip_h") and random.random() > 0.5:
                    img = img.transpose(_PILImage.FLIP_LEFT_RIGHT)
                if self._config.get("flip_v") and random.random() > 0.5:
                    img = img.transpose(_PILImage.FLIP_TOP_BOTTOM)
                if self._config.get("brightness"):
                    factor = random.uniform(0.8, 1.2)
                    img = _PILEnhance.Brightness(img).enhance(factor)
                if self._config.get("contrast"):
                    factor = random.uniform(0.8, 1.2)
                    img = _PILEnhance.Contrast(img).enhance(factor)
                if self._config.get("blur"):
                    img = img.filter(_PILFilter.GaussianBlur(radius=1.0))
                out_path = os.path.join(output_dir, f"{stem}_aug{i}{suffix}")
                img.save(out_path)
                results.append(out_path)
        return results

    def run(
        self,
        labeled_paths: list[str],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, str]:
        """
        Augmentiert alle labeled_paths.
        Returns {neue_path: original_label} für alle erfolgreich erstellten Kopien.
        """
        result: dict[str, str] = {}
        total = len(labeled_paths)
        for i, (path, label) in enumerate(labeled_paths):
            if progress_callback:
                progress_callback(i, total)
            try:
                new_paths = self.augment_image(path, output_dir)
                for np_ in new_paths:
                    result[np_] = label
            except Exception as exc:
                log.warning("Augmentierung fehlgeschlagen für %s: %s", path, exc)
        if progress_callback:
            progress_callback(total, total)
        return result


class AugmentationWorker:
    """Plain worker für Augmentierungspipeline."""

    def __init__(
        self,
        image_label_pairs: list[tuple[str, str]],
        output_dir: str,
        config: dict | None = None,
    ) -> None:
        self._pairs = list(image_label_pairs)
        self._output_dir = output_dir
        self._pipeline = AugmentationPipeline(config)
        self._progress_cb = None

    def set_progress_callback(self, cb: Callable[[int, int], None]) -> None:
        self._progress_cb = cb

    def run(self) -> dict:
        """Returns {"generated": [str], "label_map": {str: str}, "skipped": int}"""
        label_map: dict[str, str] = {}
        skipped = 0
        total = len(self._pairs)
        generated: list[str] = []
        for i, (path, label) in enumerate(self._pairs):
            if self._progress_cb:
                self._progress_cb(i, total)
            try:
                new_paths = self._pipeline.augment_image(path, self._output_dir)
                for np_ in new_paths:
                    label_map[np_] = label
                    generated.append(np_)
            except Exception as exc:
                log.warning("Augmentierung übersprungen: %s — %s", path, exc)
                skipped += 1
        if self._progress_cb:
            self._progress_cb(total, total)
        return {"generated": generated, "label_map": label_map, "skipped": skipped}


def _make_augmentation_thread():
    from PySide6.QtCore import QThread, Signal as _Signal

    class _AT(QThread):
        progress = _Signal(int, int)
        finished = _Signal(dict)
        error    = _Signal(str)

        def __init__(self, image_label_pairs, output_dir, config=None, parent=None):
            super().__init__(parent)
            self._worker = AugmentationWorker(image_label_pairs, output_dir, config)
            self._worker.set_progress_callback(self._on_progress)

        def _on_progress(self, current, total):
            self.progress.emit(current, total)

        def run(self):
            try:
                result = self._worker.run()
                self.finished.emit(result)
            except Exception as exc:
                self.error.emit(str(exc))

    return _AT

try:
    _AugQThread = _make_augmentation_thread()

    class AugmentationThread(_AugQThread):  # type: ignore[no-redef]
        """QThread für AugmentationWorker."""

except Exception:
    class AugmentationThread:  # type: ignore[no-redef]
        def __init__(self, image_label_pairs, output_dir, config=None, parent=None):
            self._worker = AugmentationWorker(image_label_pairs, output_dir, config)
        def run(self):
            return self._worker.run()
