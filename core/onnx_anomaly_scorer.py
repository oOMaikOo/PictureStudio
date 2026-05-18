"""
ONNX-based anomaly scorer — runs the exported autoencoder without PyTorch.
Requires onnxruntime. Falls back gracefully if not available.
"""
import os
import json
import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

_IMG = 128   # must match the value used during export


class OnnxAnomalyScorer:
    """
    Scores frames using an exported ONNX autoencoder model.
    Provides the same interface as AnomalyDetector for drop-in use in monitor.py.

    Metadata sidecar: looks for <model_path>.meta.json next to the .onnx file.
    """

    def __init__(self, onnx_path: str, threshold: float = 0.001,
                 metadata: dict | None = None):
        if not HAS_ORT:
            raise RuntimeError(
                "onnxruntime is not installed. pip install onnxruntime"
            )
        self._onnx_path = onnx_path
        self._threshold: float = max(1e-6, float(threshold))
        self._metadata: dict = dict(metadata) if metadata else {}
        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name

    @classmethod
    def from_path(cls, onnx_path: str) -> "OnnxAnomalyScorer":
        """Load scorer from .onnx file, reading threshold from .meta.json sidecar."""
        threshold = 0.001
        metadata: dict = {}
        meta_path = onnx_path + ".meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                threshold = float(data.get("threshold", threshold))
                metadata = dict(data.get("metadata", {}))
            except Exception:
                pass
        return cls(onnx_path, threshold=threshold, metadata=metadata)

    @property
    def threshold(self) -> float:
        """MSE value above which a frame is classified as an anomaly."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set a custom threshold; clipped to a minimum of 1e-6."""
        self._threshold = max(1e-6, float(value))

    @property
    def metadata(self) -> dict:
        """Read-only copy of the model metadata dict."""
        return dict(self._metadata)

    def score(self, frame: np.ndarray) -> float:
        """Return MSE reconstruction error for frame (BGR uint8)."""
        inp = self._preprocess(frame)
        (out,) = self._session.run(None, {self._input_name: inp})
        return float(np.mean((out - inp) ** 2))

    def is_anomaly(self, frame: np.ndarray) -> tuple[float, bool]:
        """Return (score, score > threshold)."""
        s = self.score(frame)
        return s, s > self._threshold

    def score_detailed(
        self, frame: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, tuple | None]:
        """
        Same API as AnomalyDetector.score_detailed().
        Returns (score, reconstruction_bgr, overlay_bgr, anomaly_bbox).
        """
        inp = self._preprocess(frame)   # (1,3,128,128) float32
        (out,) = self._session.run(None, {self._input_name: inp})

        score = float(np.mean((out - inp) ** 2))

        # Reconstruction → BGR uint8
        rec_np = out.squeeze(0).transpose(1, 2, 0)   # HWC, [0,1]
        rec_np = (rec_np * 255).clip(0, 255).astype(np.uint8)
        rec_bgr = cv2.cvtColor(rec_np, cv2.COLOR_RGB2BGR)

        # Heatmap: channel-averaged MSE per pixel (128×128)
        diff = ((out - inp) ** 2).mean(axis=1).squeeze(0)   # (128,128)
        peak = float(diff.max())
        if peak > 0:
            diff_u8 = ((diff / peak) * 255).clip(0, 255).astype(np.uint8)
        else:
            diff_u8 = np.zeros_like(diff, dtype=np.uint8)

        heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
        h, w = frame.shape[:2]
        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(frame, 0.55, heatmap_full, 0.45, 0)

        # Bounding box around the hottest anomaly region (top 15% of pixels)
        anomaly_bbox: tuple | None = None
        if peak > 0:
            thresh_val = float(np.percentile(diff, 85))
            mask = ((diff > thresh_val) * 255).astype(np.uint8)
            mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(
                mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                bx, by, bw, bh = cv2.boundingRect(
                    max(contours, key=cv2.contourArea)
                )
                anomaly_bbox = (bx, by, bw, bh)
                color = (0, 0, 255) if score > self._threshold else (0, 200, 255)
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
                label = "ANOMALIE" if score > self._threshold else "Hotspot"
                cv2.putText(
                    overlay, label, (bx, max(by - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
                )

        return score, rec_bgr, overlay, anomaly_bbox

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 → float32 (1,3,128,128) NCHW numpy array, values in [0,1]."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (_IMG, _IMG), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]  # NCHW
