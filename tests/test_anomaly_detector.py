"""
Unit tests for core/anomaly_detector.py

Tests cover: collection, training, scoring, threshold logic,
save/load round-trip, and the key property that normal frames score
lower than visually different (anomalous) frames after training.

These tests run on CPU and need no GPU.  Training uses only 3 epochs
so the suite finishes in a few seconds.
"""
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(r: int, g: int, b: int, h: int = 240, w: int = 320) -> np.ndarray:
    """Return a solid-colour BGR numpy frame (uint8)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = b
    frame[:, :, 1] = g
    frame[:, :, 2] = r
    return frame


def _random_frame(seed: int = 0, h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_trained_detector(n_frames: int = 30, epochs: int = 3):
    """Return an AnomalyDetector trained on near-identical grey frames."""
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        # Slight noise around grey (128, 128, 128) so training set is consistent
        noise = rng.integers(-5, 6, (240, 320, 3), dtype=np.int16)
        base = np.full((240, 320, 3), 128, dtype=np.int16)
        frame = np.clip(base + noise, 0, 255).astype(np.uint8)
        det.collect_frame(frame)
    det.train(epochs=epochs)
    return det


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class TestCollection:
    def test_initial_count_zero(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        assert det.n_collected() == 0

    def test_collect_increments_count(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        for i in range(5):
            det.collect_frame(_solid_frame(128, 128, 128))
        assert det.n_collected() == 5

    def test_clear_resets_count(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        det.collect_frame(_solid_frame(0, 0, 0))
        det.collect_frame(_solid_frame(255, 255, 255))
        det.clear_frames()
        assert det.n_collected() == 0

    def test_collect_accepts_various_resolutions(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        det.collect_frame(_solid_frame(100, 100, 100, h=480, w=640))
        det.collect_frame(_solid_frame(100, 100, 100, h=1080, w=1920))
        det.collect_frame(_solid_frame(100, 100, 100, h=64, w=64))
        assert det.n_collected() == 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_raises_without_frames(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        with pytest.raises(ValueError, match="Keine Frames"):
            det.train(epochs=1)

    def test_train_sets_trained_flag(self):
        det = _make_trained_detector()
        assert det.trained is True

    def test_train_returns_positive_threshold(self):
        det = _make_trained_detector()
        assert det.threshold > 0.0

    def test_progress_callback_called(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        for _ in range(10):
            det.collect_frame(_solid_frame(128, 128, 128))
        calls = []
        det.train(epochs=3, progress_cb=lambda e, t, l: calls.append((e, t, l)))
        assert len(calls) == 3
        # epochs should be 1-indexed and go up to 3
        assert calls[0][0] == 1
        assert calls[-1][0] == 3
        # losses should be finite floats
        for _, _, loss in calls:
            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_train_with_single_frame(self):
        """Edge-case: batch_size > n_frames should not crash."""
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        det.collect_frame(_solid_frame(200, 100, 50))
        det.train(epochs=2)
        assert det.trained


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_returns_zero_before_training(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        assert det.score(_solid_frame(128, 128, 128)) == 0.0

    def test_score_returns_float_after_training(self):
        det = _make_trained_detector()
        score = det.score(_solid_frame(128, 128, 128))
        assert isinstance(score, float)
        assert score >= 0.0

    def test_is_anomaly_returns_tuple(self):
        det = _make_trained_detector()
        result = det.is_anomaly(_solid_frame(128, 128, 128))
        assert isinstance(result, tuple) and len(result) == 2
        score, flag = result
        assert isinstance(score, float)
        assert isinstance(flag, bool)

    def test_normal_frame_scores_lower_than_anomaly(self):
        """
        A frame similar to the training distribution should reconstruct
        better (lower MSE) than a very different frame.

        The detector trains on near-grey frames (r≈g≈b≈128).
        A pure red frame (0, 0, 255 BGR) is far from the training distribution.
        After 3 epochs the autoencoder may not yet perfectly separate them,
        so we use 10 epochs and 60 frames for a more decisive test.
        """
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        rng = np.random.default_rng(0)
        for _ in range(60):
            noise = rng.integers(-5, 6, (240, 320, 3), dtype=np.int16)
            frame = np.clip(np.full((240, 320, 3), 128, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            det.collect_frame(frame)
        det.train(epochs=10)

        # Normal: grey frame similar to training data
        normal_score = det.score(_solid_frame(128, 128, 128))
        # Anomaly: pure red — completely outside training distribution
        anomaly_score = det.score(_solid_frame(255, 0, 0))

        assert anomaly_score > normal_score, (
            f"Expected anomaly ({anomaly_score:.5f}) > normal ({normal_score:.5f})"
        )

    def test_score_is_deterministic(self):
        """Same frame should always produce the same score."""
        det = _make_trained_detector()
        frame = _solid_frame(100, 150, 200)
        scores = [det.score(frame) for _ in range(5)]
        assert all(abs(s - scores[0]) < 1e-9 for s in scores)


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_default_threshold(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        assert det.threshold == pytest.approx(0.02)

    def test_set_threshold(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        det.threshold = 0.05
        assert det.threshold == pytest.approx(0.05)

    def test_threshold_clamped_at_min(self):
        from core.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        det.threshold = -99.0
        assert det.threshold >= 1e-7

    def test_auto_threshold_after_training(self):
        """Trained threshold should be > 0 and reasonably small for consistent frames."""
        det = _make_trained_detector()
        assert 0 < det.threshold < 1.0

    def test_is_anomaly_respects_threshold(self):
        """Manually lower threshold so any non-zero score triggers alarm."""
        det = _make_trained_detector()
        det.threshold = 1e-9  # absurdly low → almost every frame is anomaly
        _, flag = det.is_anomaly(_random_frame())
        assert flag is True

        det.threshold = 1.0  # absurdly high → nothing is anomaly
        _, flag = det.is_anomaly(_random_frame())
        assert flag is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        det = _make_trained_detector()
        path = str(tmp_path / "ae_model.pth")
        det.save(path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_load_restores_trained_flag(self, tmp_path):
        det = _make_trained_detector()
        path = str(tmp_path / "ae_model.pth")
        det.save(path)

        from core.anomaly_detector import AnomalyDetector
        det2 = AnomalyDetector()
        assert not det2.trained
        det2.load(path)
        assert det2.trained

    def test_load_restores_threshold(self, tmp_path):
        det = _make_trained_detector()
        path = str(tmp_path / "ae_model.pth")
        det.save(path)

        from core.anomaly_detector import AnomalyDetector
        det2 = AnomalyDetector()
        det2.load(path)
        assert det2.threshold == pytest.approx(det.threshold, rel=1e-5)

    def test_load_restores_scoring_behaviour(self, tmp_path):
        """Scores from a re-loaded model must match the original."""
        det = _make_trained_detector()
        frame = _solid_frame(128, 128, 128)
        score_before = det.score(frame)

        path = str(tmp_path / "ae_model.pth")
        det.save(path)

        from core.anomaly_detector import AnomalyDetector
        det2 = AnomalyDetector()
        det2.load(path)
        score_after = det2.score(frame)

        assert score_after == pytest.approx(score_before, rel=1e-5)

    def test_load_bad_file_raises(self, tmp_path):
        from core.anomaly_detector import AnomalyDetector
        bad = tmp_path / "bad.pth"
        bad.write_bytes(b"not a valid checkpoint")
        det = AnomalyDetector()
        with pytest.raises(Exception):
            det.load(str(bad))
