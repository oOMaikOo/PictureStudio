"""Tests for core/active_learning.py — ActiveLearningSampler."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.active_learning import ActiveLearningSampler


def _make_pred(label: str, conf: float) -> dict:
    return {
        "predicted_label": label,
        "confidence": conf,
        "top_k": [{"label": label, "prob": conf}],
        "all_probs": {label: conf},
    }


def _make_paths(n: int) -> list:
    return [f"/fake/img_{i}.jpg" for i in range(n)]


class TestActiveLearningsampler:

    def _run_with_mock(self, preds: list, threshold=0.70, n_samples=50, paths=None):
        """Helper: patch Inferencer.predict_image and run sampler."""
        if paths is None:
            paths = [f"/fake/img_{i}.jpg" for i in range(len(preds))]
        mock_inf = MagicMock()
        mock_inf.predict_image.side_effect = preds

        sampler = ActiveLearningSampler()
        with patch("core.inference.Inferencer", return_value=mock_inf):
            return sampler.run(
                model_path="/fake/model.pth",
                image_paths=paths,
                confidence_threshold=threshold,
                n_samples=n_samples,
            )

    def test_returns_only_uncertain(self):
        preds = [
            _make_pred("cat", 0.95),  # confident → excluded
            _make_pred("dog", 0.40),  # uncertain → included
            _make_pred("cat", 0.65),  # uncertain → included
        ]
        results = self._run_with_mock(preds, threshold=0.70)
        assert len(results) == 2
        for r in results:
            assert r["confidence"] < 0.70

    def test_sorted_by_ascending_confidence(self):
        preds = [
            _make_pred("cat", 0.60),
            _make_pred("dog", 0.30),
            _make_pred("cat", 0.50),
        ]
        results = self._run_with_mock(preds, threshold=0.70)
        confs = [r["confidence"] for r in results]
        assert confs == sorted(confs)

    def test_n_samples_cap(self):
        preds = [_make_pred("cat", 0.10) for _ in range(20)]
        results = self._run_with_mock(preds, threshold=0.70, n_samples=5)
        assert len(results) == 5

    def test_empty_if_all_confident(self):
        preds = [_make_pred("cat", 0.95), _make_pred("dog", 0.99)]
        results = self._run_with_mock(preds, threshold=0.70)
        assert results == []

    def test_result_fields(self):
        preds = [_make_pred("cat", 0.45)]
        results = self._run_with_mock(preds, threshold=0.70)
        assert len(results) == 1
        r = results[0]
        assert "path" in r
        assert "predicted_label" in r
        assert "confidence" in r
        assert "uncertainty" in r
        assert abs(r["uncertainty"] - (1 - r["confidence"])) < 1e-4

    def test_skips_on_exception(self):
        mock_inf = MagicMock()
        mock_inf.predict_image.side_effect = [
            Exception("read error"),
            _make_pred("dog", 0.30),
        ]
        sampler = ActiveLearningSampler()
        with patch("core.inference.Inferencer", return_value=mock_inf):
            results = sampler.run(
                model_path="/fake/model.pth",
                image_paths=["/bad.jpg", "/ok.jpg"],
                confidence_threshold=0.70,
            )
        assert len(results) == 1
        assert results[0]["confidence"] == 0.30

    def test_progress_callback_called(self):
        preds = [_make_pred("cat", 0.50), _make_pred("dog", 0.60)]
        calls = []
        mock_inf = MagicMock()
        mock_inf.predict_image.side_effect = preds
        sampler = ActiveLearningSampler()
        with patch("core.inference.Inferencer", return_value=mock_inf):
            sampler.run(
                model_path="/fake/model.pth",
                image_paths=["/a.jpg", "/b.jpg"],
                confidence_threshold=0.70,
                progress_callback=lambda c, t: calls.append((c, t)),
            )
        assert calls == [(1, 2), (2, 2)]

    def test_empty_image_list(self):
        mock_inf = MagicMock()
        sampler = ActiveLearningSampler()
        with patch("core.inference.Inferencer", return_value=mock_inf):
            results = sampler.run(
                model_path="/fake/model.pth",
                image_paths=[],
                confidence_threshold=0.70,
            )
        assert results == []

    def test_threshold_boundary(self):
        """Images exactly at threshold are excluded (strict <)."""
        preds = [_make_pred("cat", 0.70)]
        results = self._run_with_mock(preds, threshold=0.70)
        assert results == []

    def test_roi_template_forwarded(self):
        roi = {"x": 10, "y": 10, "w": 50, "h": 50}
        mock_inf = MagicMock()
        mock_inf.predict_image.return_value = _make_pred("cat", 0.40)
        sampler = ActiveLearningSampler()
        with patch("core.inference.Inferencer", return_value=mock_inf):
            sampler.run(
                model_path="/fake/model.pth",
                image_paths=["/img.jpg"],
                confidence_threshold=0.70,
                roi_template=roi,
            )
        call_kwargs = mock_inf.predict_image.call_args
        assert call_kwargs[1]["roi"] == roi or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == roi
        )
