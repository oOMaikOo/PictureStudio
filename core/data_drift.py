"""
Data Drift Detection: compare production images to training distribution.

No heavy dependencies required — only numpy, Pillow (already required), and
optionally cv2 for edge density. scipy is used for KS test if available.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from PySide6.QtCore import QThread, Signal as _Signal
    HAS_QT = True
except ImportError:
    HAS_QT = False

from utils.logging_utils import get_logger

log = get_logger()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Feature vector layout (24 values):
#   0-2:  mean R, G, B  (normalized 0-1)
#   3-5:  std  R, G, B
#   6:    laplacian variance (sharpness / 10 000, clamped)
#   7:    edge density  (mean Canny / 255, or 0 if no cv2)
#   8-23: grayscale histogram 16 bins (normalized)
FEAT_DIM = 24


def _extract_features(path: str) -> np.ndarray:
    """Return a 26-dim feature vector for one image. Raises on failure."""
    img = Image.open(path).convert("RGB").resize((128, 128), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (128,128,3)

    means = arr.mean(axis=(0, 1))   # 3
    stds  = arr.std(axis=(0, 1))    # 3

    gray = arr.mean(axis=2)  # (128,128)

    # Sharpness via Laplacian
    if HAS_CV2:
        gray_u8 = (gray * 255).astype(np.uint8)
        lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var()) / 10_000.0
        edges   = cv2.Canny(gray_u8, 50, 150).astype(np.float32) / 255.0
        edge_density = float(edges.mean())
    else:
        # Approximate Laplacian via numpy
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        lap_var = float((dx ** 2).mean() + (dy ** 2).mean())
        edge_density = 0.0

    lap_var = min(lap_var, 10.0)  # clamp to [0, 10]

    hist, _ = np.histogram(gray.ravel(), bins=16, range=(0.0, 1.0), density=True)
    hist = hist / (hist.sum() + 1e-9)  # normalize to sum=1

    feat = np.concatenate([means, stds, [lap_var, edge_density], hist])
    return feat.astype(np.float32)


class DriftDetector:
    """Builds a baseline from training images and scores new images for drift."""

    VERSION = 1

    def __init__(self):
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std:  Optional[np.ndarray] = None
        self._baseline_features: Optional[np.ndarray] = None  # (N, FEAT_DIM)
        self.n_baseline: int = 0
        self.baseline_paths: List[str] = []

    def is_ready(self) -> bool:
        return self._baseline_mean is not None

    # ---------------------------------------------------------------- baseline

    def build_baseline(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        """Extract features from training images and store distribution stats."""
        features = []
        errors = 0
        for i, p in enumerate(image_paths):
            try:
                features.append(_extract_features(p))
            except Exception as exc:
                log.warning("Drift baseline: skip %s — %s", p, exc)
                errors += 1
            if progress_callback:
                progress_callback(i + 1, len(image_paths))

        if not features:
            raise ValueError("Keine gültigen Bilder für die Baseline gefunden.")

        mat = np.stack(features)  # (N, FEAT_DIM)
        self._baseline_features = mat
        self._baseline_mean = mat.mean(axis=0)
        self._baseline_std  = mat.std(axis=0) + 1e-6   # avoid /0
        self.n_baseline = len(features)
        self.baseline_paths = image_paths
        log.info("Drift-Baseline: %d Bilder, %d Fehler", len(features), errors)
        return {"n_ok": len(features), "n_errors": errors}

    # ---------------------------------------------------------------- scoring

    def score_image(self, path: str) -> Tuple[float, Dict]:
        """
        Returns (drift_score, details).
        drift_score is the max z-score across features (higher = more drift).
        """
        if not self.is_ready():
            raise RuntimeError("Keine Baseline geladen.")
        feat = _extract_features(path)
        z = np.abs((feat - self._baseline_mean) / self._baseline_std)
        drift_score = float(z.max())
        details = {
            "z_max":         drift_score,
            "z_color_mean":  float(z[:3].mean()),
            "z_color_std":   float(z[3:6].mean()),
            "z_sharpness":   float(z[6]),
            "z_edges":       float(z[7]),
            "z_histogram":   float(z[8:].mean()),
        }
        return drift_score, details

    def score_batch(
        self,
        folder: str,
        recursive: bool = False,
        threshold: float = 3.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict]:
        """Score all images in folder. Returns list of result dicts."""
        if recursive:
            pairs = []
            for root, _, files in os.walk(folder):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                        full = os.path.join(root, f)
                        rel  = os.path.relpath(root, folder)
                        pairs.append((f if rel == "." else f"{rel}/{f}", full))
        else:
            pairs = [
                (f, os.path.join(folder, f))
                for f in sorted(os.listdir(folder))
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]

        results = []
        for i, (display, path) in enumerate(pairs):
            try:
                score, details = self.score_image(path)
                results.append({
                    "filename": display,
                    "path":     path,
                    "score":    round(score, 3),
                    "drifted":  score > threshold,
                    "details":  details,
                    "error":    None,
                })
            except Exception as exc:
                results.append({
                    "filename": display, "path": path,
                    "score": -1.0, "drifted": False,
                    "details": {}, "error": str(exc),
                })
            if progress_callback:
                progress_callback(i + 1, len(pairs))
        return results

    # ---------------------------------------------------------------- ks test

    def ks_test(self, image_paths: List[str]) -> Dict[str, float]:
        """
        Run KS test (requires scipy) comparing production feature distribution
        to baseline. Returns p-values per feature group.
        """
        if not HAS_SCIPY:
            return {}
        if not self.is_ready():
            return {}
        prod_feats = []
        for p in image_paths:
            try:
                prod_feats.append(_extract_features(p))
            except Exception:
                pass
        if not prod_feats:
            return {}
        prod = np.stack(prod_feats)
        base = self._baseline_features
        out = {}
        groups = {
            "color_mean": slice(0, 3),
            "color_std":  slice(3, 6),
            "sharpness":  slice(6, 7),
            "edges":      slice(7, 8),
            "histogram":  slice(8, 24),
        }
        for name, sl in groups.items():
            p_vals = []
            for dim in range(sl.start, sl.stop):
                _, p = _scipy_stats.ks_2samp(base[:, dim], prod[:, dim])
                p_vals.append(p)
            out[name] = float(np.min(p_vals))
        return out

    # ---------------------------------------------------------------- persist

    def save(self, path: str) -> None:
        data = {
            "version":           self.VERSION,
            "n_baseline":        self.n_baseline,
            "baseline_paths":    self.baseline_paths,
            "baseline_mean":     self._baseline_mean.tolist() if self._baseline_mean is not None else None,
            "baseline_std":      self._baseline_std.tolist()  if self._baseline_std  is not None else None,
            "baseline_features": self._baseline_features.tolist() if self._baseline_features is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        log.info("Drift-Baseline gespeichert: %s", path)

    def load(self, path: str) -> "DriftDetector":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.n_baseline     = data.get("n_baseline", 0)
        self.baseline_paths = data.get("baseline_paths", [])
        m = data.get("baseline_mean")
        s = data.get("baseline_std")
        bf = data.get("baseline_features")
        self._baseline_mean     = np.array(m,  dtype=np.float32) if m  else None
        self._baseline_std      = np.array(s,  dtype=np.float32) if s  else None
        self._baseline_features = np.array(bf, dtype=np.float32) if bf else None
        log.info("Drift-Baseline geladen: %s (%d Bilder)", path, self.n_baseline)
        return self


# ------------------------------------------------------------------ QThread wrappers

if HAS_QT:
    class DriftBaselineThread(QThread):
        progress = _Signal(int, int)
        finished = _Signal(dict)   # {"n_ok": int, "n_errors": int}
        error    = _Signal(str)

        def __init__(self, detector: DriftDetector, image_paths: List[str], parent=None):
            super().__init__(parent)
            self.detector     = detector
            self.image_paths  = image_paths

        def run(self):
            try:
                stats = self.detector.build_baseline(
                    self.image_paths,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                )
                self.finished.emit(stats)
            except Exception as exc:
                self.error.emit(str(exc))

    class DriftScoringThread(QThread):
        progress = _Signal(int, int)
        finished = _Signal(list)
        error    = _Signal(str)

        def __init__(
            self,
            detector: DriftDetector,
            folder: str,
            recursive: bool = False,
            threshold: float = 3.0,
            parent=None,
        ):
            super().__init__(parent)
            self.detector  = detector
            self.folder    = folder
            self.recursive = recursive
            self.threshold = threshold

        def run(self):
            try:
                results = self.detector.score_batch(
                    self.folder,
                    recursive=self.recursive,
                    threshold=self.threshold,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                )
                self.finished.emit(results)
            except Exception as exc:
                self.error.emit(str(exc))
