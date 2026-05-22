"""
Tests for core/data_drift.py — DriftDetector baseline, scoring, persist.
All tests run without scipy or cv2 installed (graceful fallback).
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


# ------------------------------------------------------------------ helpers

def _make_tiny_png(path: str, color=(128, 128, 128)):
    from PIL import Image
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path, "PNG")


def _make_image_folder(tmp: str, n: int = 5, color=(128, 128, 128)):
    paths = []
    for i in range(n):
        p = os.path.join(tmp, f"img_{i:03d}.png")
        _make_tiny_png(p, color)
        paths.append(p)
    return paths


# ------------------------------------------------------------------ unit tests

class TestExtractFeatures:
    def test_returns_correct_dim(self):
        from core.data_drift import _extract_features, FEAT_DIM
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.png")
            _make_tiny_png(p)
            feat = _extract_features(p)
        assert feat.shape == (FEAT_DIM,)

    def test_raises_on_missing_file(self):
        from core.data_drift import _extract_features
        with pytest.raises(Exception):
            _extract_features("/nonexistent/path/img.png")

    def test_different_colors_give_different_features(self):
        from core.data_drift import _extract_features
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "white.png")
            p2 = os.path.join(td, "black.png")
            _make_tiny_png(p1, color=(255, 255, 255))
            _make_tiny_png(p2, color=(0, 0, 0))
            f1 = _extract_features(p1)
            f2 = _extract_features(p2)
        assert not np.allclose(f1, f2), "White and black images must differ"


class TestDriftDetector:
    def test_not_ready_on_init(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        assert not d.is_ready()

    def test_build_baseline(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=5)
            stats = d.build_baseline(paths)
        assert d.is_ready()
        assert d.n_baseline == 5
        assert stats["n_ok"] == 5
        assert stats["n_errors"] == 0

    def test_build_baseline_skips_bad_files(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=3)
            bad = os.path.join(td, "bad.png")
            open(bad, "wb").close()   # empty file → PIL will fail
            stats = d.build_baseline(paths + [bad])
        assert stats["n_ok"] == 3
        assert stats["n_errors"] >= 1

    def test_build_baseline_raises_with_no_valid_images(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad.png")
            open(bad, "wb").close()
            with pytest.raises(ValueError, match="Keine gültigen"):
                d.build_baseline([bad])

    def test_score_image_raises_without_baseline(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with pytest.raises(RuntimeError, match="Keine Baseline"):
            d.score_image("dummy.png")

    def test_score_image_returns_score_and_details(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=5)
            d.build_baseline(paths)
            score, details = d.score_image(paths[0])
        assert isinstance(score, float)
        assert score >= 0.0
        assert "z_max" in details
        assert "z_color_mean" in details

    def test_score_in_distribution_image_lower_than_ood(self):
        """In-distribution image has lower drift score than very different image."""
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            # Baseline: grey images
            baseline = _make_image_folder(td, n=8, color=(128, 128, 128))
            d.build_baseline(baseline)

            # In-dist: same grey
            in_dist = os.path.join(td, "in_dist.png")
            _make_tiny_png(in_dist, color=(130, 130, 130))

            # OOD: very bright/white
            ood = os.path.join(td, "ood.png")
            _make_tiny_png(ood, color=(255, 255, 255))

            score_in, _ = d.score_image(in_dist)
            score_ood, _ = d.score_image(ood)
        assert score_ood > score_in

    def test_score_batch_returns_list(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=4)
            d.build_baseline(paths)
            results = d.score_batch(td, threshold=3.0)
        assert len(results) == 4
        for r in results:
            assert "filename" in r
            assert "score" in r
            assert "drifted" in r
            assert r["error"] is None

    def test_score_batch_flags_drifted(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            # Baseline: grey
            baseline = _make_image_folder(td, n=6, color=(128, 128, 128))
            d.build_baseline(baseline)

            # Production folder: mix of grey and very white
            prod_dir = os.path.join(td, "prod")
            os.makedirs(prod_dir)
            for i in range(3):
                _make_tiny_png(os.path.join(prod_dir, f"grey_{i}.png"), color=(128, 128, 128))
            for i in range(3):
                _make_tiny_png(os.path.join(prod_dir, f"white_{i}.png"), color=(255, 255, 255))

            results = d.score_batch(prod_dir, threshold=1.0)  # low threshold to catch whites
        scores = {r["filename"]: r["score"] for r in results}
        # White images should have higher scores than grey
        grey_scores  = [scores[k] for k in scores if "grey"  in k]
        white_scores = [scores[k] for k in scores if "white" in k]
        assert np.mean(white_scores) > np.mean(grey_scores)

    def test_score_batch_recursive(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=3)
            d.build_baseline(paths)

            prod = os.path.join(td, "prod")
            sub  = os.path.join(prod, "sub")
            os.makedirs(sub)
            _make_tiny_png(os.path.join(prod, "a.png"))
            _make_tiny_png(os.path.join(sub,  "b.png"))

            results_flat = d.score_batch(prod, recursive=False)
            results_rec  = d.score_batch(prod, recursive=True)
        assert len(results_flat) == 1
        assert len(results_rec)  == 2

    def test_progress_callback_called(self):
        from core.data_drift import DriftDetector
        calls = []
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=3)
            d.build_baseline(paths, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_save_and_load_roundtrip(self):
        from core.data_drift import DriftDetector
        d1 = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=4)
            d1.build_baseline(paths)
            save_path = os.path.join(td, "baseline.json")
            d1.save(save_path)

            d2 = DriftDetector()
            d2.load(save_path)

        assert d2.is_ready()
        assert d2.n_baseline == 4
        np.testing.assert_allclose(d1._baseline_mean, d2._baseline_mean, atol=1e-5)
        np.testing.assert_allclose(d1._baseline_std,  d2._baseline_std,  atol=1e-5)

    def test_save_creates_valid_json(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=3)
            d.build_baseline(paths)
            save_path = os.path.join(td, "bl.json")
            d.save(save_path)
            with open(save_path) as f:
                data = json.load(f)
        assert "baseline_mean" in data
        assert "n_baseline" in data
        assert data["n_baseline"] == 3

    def test_ks_test_returns_empty_without_scipy(self, monkeypatch):
        from core import data_drift as mod
        monkeypatch.setattr(mod, "HAS_SCIPY", False)
        d = mod.DriftDetector()
        with tempfile.TemporaryDirectory() as td:
            paths = _make_image_folder(td, n=3)
            d.build_baseline(paths)
            result = d.ks_test(paths)
        assert result == {}

    def test_ks_test_returns_dict_without_baseline(self):
        from core.data_drift import DriftDetector
        d = DriftDetector()
        assert d.ks_test([]) == {}
