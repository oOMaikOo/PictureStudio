"""
Anomalie-Clustering: Gruppiert Alarm-Bilder anhand visueller Ähnlichkeit.

Feature-Extraktion: Jedes Bild wird auf 64×64 skaliert, zu einem 1D-Vektor
abgeflacht und normalisiert (L2).  Clustering: sklearn KMeans (Fallback:
alle Bilder in Cluster 0 wenn sklearn fehlt).

Öffentliche API
---------------
AnomalyClustering.fit(image_paths, n_clusters=5) -> dict
    {cluster_id: [path, ...]}

AnomalyClustering.get_representative(cluster_id) -> str
    Pfad des Bildes, das dem Cluster-Zentroid am nächsten liegt.

AnomalyClustering.to_dataframe() -> list[dict]
    [{path, cluster, is_representative}, ...]

AnomalyClustering.export_csv(output_path)
    Schreibt CSV mit Kopfzeile path,cluster,is_representative.

ClusteringWorker  — Plain class, run() wird von ClusteringThread aufgerufen.
ClusteringThread  — QThread-Wrapper mit Signalen progress, finished, error.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("ImageLabelingStudio.anomaly_clustering")

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_THUMB_SIZE = 64   # resize target for feature extraction


def _load_image_flat(path: str) -> Optional[np.ndarray]:
    """Load image at *path*, resize to 64×64, flatten to 1D float32 vector.

    Returns None if the image cannot be read.
    """
    arr: Optional[np.ndarray] = None

    if _HAS_CV2:
        raw = _cv2.imread(path)
        if raw is not None:
            raw = _cv2.resize(raw, (_THUMB_SIZE, _THUMB_SIZE))
            arr = raw.astype(np.float32)
    elif _HAS_PIL:
        try:
            img = _PILImage.open(path).convert("RGB")
            img = img.resize((_THUMB_SIZE, _THUMB_SIZE))
            arr = np.array(img, dtype=np.float32)
        except Exception as exc:
            log.debug("Bild konnte nicht geladen werden (%s): %s", path, exc)

    if arr is None:
        return None

    flat = arr.flatten()
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat /= norm
    return flat


# ---------------------------------------------------------------------------
# Core clustering class
# ---------------------------------------------------------------------------

class AnomalyClustering:
    """Groups images by visual similarity using KMeans on flattened pixel vectors."""

    def __init__(self) -> None:
        self._clusters: Dict[int, List[str]] = {}        # cluster_id → [path, ...]
        self._representatives: Dict[int, str] = {}       # cluster_id → path
        self._path_cluster: Dict[str, int] = {}          # path → cluster_id
        self._centroids: Optional[np.ndarray] = None     # shape (n_clusters, n_features)
        self._features: Optional[np.ndarray] = None      # shape (n_images, n_features)
        self._fitted_paths: List[str] = []

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        image_paths: List[str],
        n_clusters: int = 5,
        progress_callback=None,
    ) -> Dict[int, List[str]]:
        """Cluster *image_paths* into *n_clusters* groups.

        Parameters
        ----------
        image_paths:
            Absolute paths to images.
        n_clusters:
            Number of clusters (clamped to len(image_paths)).
        progress_callback:
            Optional callable(current: int, total: int) for progress reporting.

        Returns
        -------
        dict
            {cluster_id: [path, ...]}  — empty dict if image_paths is empty.
        """
        self._clusters.clear()
        self._representatives.clear()
        self._path_cluster.clear()
        self._centroids = None
        self._features = None
        self._fitted_paths = []

        if not image_paths:
            return {}

        # --- Feature extraction ---
        features: List[np.ndarray] = []
        valid_paths: List[str] = []
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i, total)
            feat = _load_image_flat(path)
            if feat is not None:
                features.append(feat)
                valid_paths.append(path)

        if not features:
            return {}

        X = np.vstack(features)
        self._features = X
        self._fitted_paths = valid_paths

        # Clamp n_clusters to number of valid images
        k = max(1, min(n_clusters, len(valid_paths)))

        # --- Clustering ---
        if _HAS_SKLEARN and k > 1:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            self._centroids = km.cluster_centers_
        else:
            # Fallback: all images go to cluster 0
            labels = np.zeros(len(valid_paths), dtype=int)
            self._centroids = X.mean(axis=0, keepdims=True)

        # Build cluster → paths mapping
        for path, label_id in zip(valid_paths, labels):
            cid = int(label_id)
            self._clusters.setdefault(cid, []).append(path)
            self._path_cluster[path] = cid

        # Find representative (closest to centroid) for each cluster
        for cid, paths in self._clusters.items():
            centroid = self._centroids[cid]
            best_path = paths[0]
            best_dist = float("inf")
            for path in paths:
                feat = _load_image_flat(path)
                if feat is None:
                    continue
                dist = float(np.linalg.norm(feat - centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
            self._representatives[cid] = best_path

        if progress_callback:
            progress_callback(total, total)

        return dict(self._clusters)

    # ------------------------------------------------------------------ query

    def get_representative(self, cluster_id: int) -> str:
        """Return the path of the image closest to *cluster_id*'s centroid."""
        return self._representatives.get(cluster_id, "")

    def to_dataframe(self) -> List[dict]:
        """Return a list of dicts with keys: path, cluster, is_representative."""
        rows: List[dict] = []
        for cid, paths in self._clusters.items():
            rep = self._representatives.get(cid, "")
            for path in paths:
                rows.append(
                    {
                        "path": path,
                        "cluster": cid,
                        "is_representative": path == rep,
                    }
                )
        return rows

    def export_csv(self, output_path: str) -> None:
        """Write clustering results as CSV to *output_path*.

        Columns: path, cluster, is_representative
        """
        rows = self.to_dataframe()
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "cluster", "is_representative"])
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------ properties

    @property
    def clusters(self) -> Dict[int, List[str]]:
        """The current cluster mapping {cluster_id: [path, ...]}."""
        return dict(self._clusters)

    @property
    def n_clusters(self) -> int:
        """Number of clusters in the most recent fit."""
        return len(self._clusters)


# ---------------------------------------------------------------------------
# Worker / Thread (same pattern as TrainingWorker / TrainingThread)
# ---------------------------------------------------------------------------

class ClusteringWorker:
    """Plain worker (not a QThread). Runs inside ClusteringThread."""

    def __init__(self, image_paths: List[str], n_clusters: int = 5) -> None:
        self._paths = list(image_paths)
        self._n_clusters = n_clusters
        self._clustering: Optional[AnomalyClustering] = None
        self._progress_cb = None

    def set_progress_callback(self, cb) -> None:
        """Attach a callable(current, total) progress callback."""
        self._progress_cb = cb

    def run(self) -> dict:
        """Perform clustering and return the cluster dict."""
        ac = AnomalyClustering()
        result = ac.fit(
            self._paths,
            n_clusters=self._n_clusters,
            progress_callback=self._progress_cb,
        )
        self._clustering = ac
        return result

    @property
    def clustering(self) -> Optional[AnomalyClustering]:
        """Access the fitted AnomalyClustering object after run() completes."""
        return self._clustering


class ClusteringThread:
    """QThread wrapper — imported lazily to allow non-Qt unit tests."""

    # Signals defined at class body when PySide6 is available
    _qt_imported = False

    def __new__(cls, *args, **kwargs):
        # Dynamically create a real QThread subclass on first instantiation
        if not cls._qt_imported:
            cls._make_qt_class()
        return object.__new__(_ClusteringQThread)

    @classmethod
    def _make_qt_class(cls) -> None:
        cls._qt_imported = True


def _make_clustering_thread():
    """Build and return the concrete QThread subclass for clustering."""
    from PySide6.QtCore import QThread, Signal as _Signal

    class _CT(QThread):
        progress = _Signal(int, int)   # current, total
        finished = _Signal(dict)       # {cluster_id: [path, ...]}
        error    = _Signal(str)

        def __init__(self, image_paths: List[str], n_clusters: int = 5, parent=None):
            super().__init__(parent)
            self._worker = ClusteringWorker(image_paths, n_clusters)
            self._worker.set_progress_callback(self._on_progress)

        def _on_progress(self, current: int, total: int) -> None:
            self.progress.emit(current, total)

        def run(self) -> None:
            try:
                result = self._worker.run()
                self.finished.emit(result)
            except Exception as exc:
                self.error.emit(str(exc))

        @property
        def clustering(self) -> Optional[AnomalyClustering]:
            return self._worker.clustering

    return _CT


# Build concrete class at module import time if Qt is available
try:
    _ClusteringQThread = _make_clustering_thread()

    class ClusteringThread(_ClusteringQThread):  # type: ignore[no-redef]
        """QThread that runs AnomalyClustering.fit() in the background.

        Signals
        -------
        progress(int, int)
            Emitted with (current_image, total_images) during feature extraction.
        finished(dict)
            Emitted with {cluster_id: [path, ...]} when clustering is complete.
        error(str)
            Emitted if an unhandled exception occurs.
        """

except Exception:
    # Qt not available (e.g. headless test environment) — stub
    class ClusteringThread:  # type: ignore[no-redef]
        """Stub ClusteringThread for environments without PySide6."""

        def __init__(self, image_paths, n_clusters=5, parent=None):
            self._worker = ClusteringWorker(image_paths, n_clusters)

        def run(self):
            return self._worker.run()

        @property
        def clustering(self):
            return self._worker.clustering
