"""
Tests for ONNX export (AnomalyDetector.export_onnx_with_meta)
and ONNX inference (OnnxAnomalyScorer).

Skipped entirely when onnxruntime is not installed.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from core.onnx_anomaly_scorer import HAS_ORT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trained_detector(n_frames=30, epochs=3):
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        noise = rng.integers(-5, 6, (240, 320, 3), dtype=np.int16)
        base = np.full((240, 320, 3), 128, dtype=np.int16)
        frame = np.clip(base + noise, 0, 255).astype(np.uint8)
        det.collect_frame(frame)
    det.train(epochs=epochs)
    return det


def _grey_frame(h=240, w=320) -> np.ndarray:
    """Return a near-grey BGR frame similar to the training distribution."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _red_frame(h=240, w=320) -> np.ndarray:
    """Return a pure-red BGR frame — far from the grey training distribution."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 2] = 255   # B=0, G=0, R=255 in BGR
    return frame


# ---------------------------------------------------------------------------
# Test class — all tests skipped when onnxruntime is absent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
class TestOnnxExport:

    # -- export_onnx_with_meta ----------------------------------------------

    def test_export_creates_onnx_file(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)
        assert os.path.isfile(onnx_path)
        assert os.path.getsize(onnx_path) > 0

    def test_export_creates_meta_json(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)
        meta_path = onnx_path + ".meta.json"
        assert os.path.isfile(meta_path)

    def test_meta_json_has_threshold_key(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)
        with open(onnx_path + ".meta.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "threshold" in data

    def test_meta_json_has_metadata_key(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)
        with open(onnx_path + ".meta.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data

    def test_export_returns_onnx_path(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        result = det.export_onnx_with_meta(onnx_path)
        assert result == onnx_path

    # -- OnnxAnomalyScorer.from_path ----------------------------------------

    def test_from_path_loads_threshold_from_meta(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        assert scorer.threshold == pytest.approx(det.threshold, rel=1e-5)

    def test_from_path_without_meta_uses_default_threshold(self, tmp_path):
        """If .meta.json is absent, scorer should use the default 0.001."""
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx(onnx_path)   # no meta

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        assert scorer.threshold == pytest.approx(0.001, rel=1e-5)

    # -- score() ------------------------------------------------------------

    def test_score_returns_float(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        result = scorer.score(_grey_frame())
        assert isinstance(result, float)

    def test_score_is_non_negative(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        assert scorer.score(_grey_frame()) >= 0.0

    # -- is_anomaly() -------------------------------------------------------

    def test_is_anomaly_returns_float_and_bool(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        result = scorer.is_anomaly(_grey_frame())
        assert isinstance(result, tuple) and len(result) == 2
        score, flag = result
        assert isinstance(score, float)
        assert isinstance(flag, bool)

    # -- score_detailed() ---------------------------------------------------

    def test_score_detailed_returns_tuple_of_four(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        result = scorer.score_detailed(_grey_frame())
        assert isinstance(result, tuple) and len(result) == 4

    def test_score_detailed_overlay_same_size_as_input(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        frame = _grey_frame(h=480, w=640)
        _, _, overlay, _ = scorer.score_detailed(frame)
        assert overlay.shape[:2] == frame.shape[:2]

    def test_score_detailed_reconstruction_is_ndarray(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        _, rec, overlay, _ = scorer.score_detailed(_grey_frame())
        assert isinstance(rec, np.ndarray)
        assert isinstance(overlay, np.ndarray)

    # -- anomaly detection quality ------------------------------------------

    def test_anomalous_frame_scores_higher_than_normal(self, tmp_path):
        """
        Train on more epochs for a decisive separation.
        The red frame (far from grey training data) should score higher.
        """
        det = _make_trained_detector(n_frames=60, epochs=10)
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)

        normal_score = scorer.score(_grey_frame())
        anomaly_score = scorer.score(_red_frame())
        assert anomaly_score > normal_score, (
            f"Expected anomaly ({anomaly_score:.5f}) > normal ({normal_score:.5f})"
        )

    # -- threshold property -------------------------------------------------

    def test_threshold_setter_works(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        scorer.threshold = 0.05
        assert scorer.threshold == pytest.approx(0.05)

    def test_threshold_clamped_at_minimum(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        scorer.threshold = -5.0
        assert scorer.threshold >= 1e-6

    # -- metadata property --------------------------------------------------

    def test_metadata_returns_dict(self, tmp_path):
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        meta = scorer.metadata
        assert isinstance(meta, dict)

    def test_metadata_is_copy(self, tmp_path):
        """Mutating the returned dict should not affect the scorer's internal state."""
        det = _make_trained_detector()
        onnx_path = str(tmp_path / "model.onnx")
        det.export_onnx_with_meta(onnx_path)

        from core.onnx_anomaly_scorer import OnnxAnomalyScorer
        scorer = OnnxAnomalyScorer.from_path(onnx_path)
        meta1 = scorer.metadata
        meta1["_injected"] = True
        meta2 = scorer.metadata
        assert "_injected" not in meta2
