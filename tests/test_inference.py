"""
Tests for core/inference.py — Inferencer.

Mocks the model so tests run on CPU without real checkpoints.
Tests cover: load_model errors, predict_image (happy path, ROI, TTA,
missing file, not-ready), classify_single, filter_results, predict_folder
(empty, recursive, roi_templates, error handling, progress callback).
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

from core.inference import Inferencer


# ------------------------------------------------------------------ helpers

def _make_model(n_classes: int = 2, image_size: int = 64):
    """Tiny model that handles 4D input (B, C, H, W) from the Inferencer pipeline."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * image_size * image_size, n_classes),
    )
    model.eval()
    return model


def _make_image(path: str, color=(128, 64, 200), size=(80, 60)):
    img = Image.fromarray(
        np.full((*size, 3), color, dtype=np.uint8), "RGB"
    )
    img.save(path)
    return path


def _loaded_inferencer(n_classes: int = 2, class_names=None, image_size: int = 64):
    """Return an Inferencer with a fake loaded model."""
    if class_names is None:
        class_names = [f"cls{i}" for i in range(n_classes)]
    inf = Inferencer()
    inf.model = _make_model(n_classes, image_size)
    inf.class_names = class_names
    inf.image_size = image_size
    inf.model_path = "/fake/model.pth"
    inf.model_type = "simple_cnn"
    inf.device = torch.device("cpu")
    from torchvision import transforms
    inf.transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return inf


# ================================================================== load_model

class TestLoadModel:
    def test_raises_without_torch(self, tmp_path):
        inf = Inferencer()
        with patch("core.inference.HAS_TORCH", False):
            with pytest.raises(RuntimeError, match="PyTorch"):
                inf.load_model(str(tmp_path / "model.pth"))

    def test_raises_on_missing_file(self):
        inf = Inferencer()
        with pytest.raises(Exception):
            inf.load_model("/nonexistent/model.pth")

    def test_raises_on_no_class_names(self, tmp_path):
        """Checkpoint without class_names metadata should raise ValueError."""
        model = _make_model(2)
        path = str(tmp_path / "model.pth")
        torch.save({"metadata": {}, "model_state_dict": model.state_dict()}, path)
        inf = Inferencer()
        with patch("core.inference.HAS_TORCH", True):
            with pytest.raises(ValueError, match="Klasseninformation"):
                inf.load_model(path)

    def test_is_ready_false_before_load(self):
        assert not Inferencer().is_ready()


# ================================================================== predict_image

class TestPredictImage:
    def test_raises_when_not_ready(self, tmp_path):
        inf = Inferencer()
        img = _make_image(str(tmp_path / "x.png"))
        with pytest.raises(RuntimeError, match="Modell"):
            inf.predict_image(img)

    def test_returns_required_keys(self, tmp_path):
        inf = _loaded_inferencer(2, ["cat", "dog"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.predict_image(img)
        for key in ("predicted_label", "confidence", "top_k", "all_probs", "tta_passes"):
            assert key in result

    def test_predicted_label_is_class_name(self, tmp_path):
        inf = _loaded_inferencer(2, ["cat", "dog"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.predict_image(img)
        assert result["predicted_label"] in ("cat", "dog")

    def test_confidence_in_range(self, tmp_path):
        inf = _loaded_inferencer(3, ["a", "b", "c"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.predict_image(img)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top_k_length_capped(self, tmp_path):
        inf = _loaded_inferencer(3, ["a", "b", "c"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.predict_image(img, top_k=2)
        assert len(result["top_k"]) == 2

    def test_all_probs_sum_to_one(self, tmp_path):
        inf = _loaded_inferencer(3, ["a", "b", "c"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.predict_image(img)
        total = sum(result["all_probs"].values())
        assert abs(total - 1.0) < 0.01

    def test_roi_crop_applied(self, tmp_path):
        """predict_image with a ROI should not crash and still return valid output."""
        inf = _loaded_inferencer(2, ["cat", "dog"])
        img = _make_image(str(tmp_path / "x.png"), size=(100, 100))
        roi = {"x": 10, "y": 10, "w": 50, "h": 50}
        result = inf.predict_image(img, roi=roi)
        assert result["predicted_label"] in ("cat", "dog")

    def test_roi_clamped_to_zero(self, tmp_path):
        """Negative ROI coords should be clamped (no crash)."""
        inf = _loaded_inferencer(2, ["cat", "dog"])
        img = _make_image(str(tmp_path / "x.png"), size=(100, 100))
        roi = {"x": -20, "y": -10, "w": 50, "h": 50}
        result = inf.predict_image(img, roi=roi)
        assert "predicted_label" in result

    def test_tta_passes_recorded(self, tmp_path):
        inf = _loaded_inferencer(2, ["cat", "dog"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.predict_image(img, tta_passes=3)
        assert result["tta_passes"] == 3

    def test_missing_image_raises(self, tmp_path):
        inf = _loaded_inferencer(2, ["cat", "dog"])
        with pytest.raises(Exception):
            inf.predict_image(str(tmp_path / "nonexistent.png"))


# ================================================================== classify_single

class TestClassifySingle:
    def test_adds_low_confidence_flag(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.classify_single(img)
        assert "low_confidence" in result
        assert isinstance(result["low_confidence"], bool)

    def test_adds_path_field(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        img = _make_image(str(tmp_path / "x.png"))
        result = inf.classify_single(img)
        assert result["path"] == img


# ================================================================== filter_results

class TestFilterResults:
    def _make_results(self):
        return [
            {"predicted_label": "cat", "confidence": 0.95, "low_confidence": False},
            {"predicted_label": "dog", "confidence": 0.55, "low_confidence": True},
            {"predicted_label": "cat", "confidence": 0.40, "low_confidence": True},
            {"predicted_label": "bird", "confidence": 0.80, "low_confidence": False},
        ]

    def test_no_filter_returns_all(self):
        inf = _loaded_inferencer()
        results = self._make_results()
        assert inf.filter_results(results) == results

    def test_label_filter(self):
        inf = _loaded_inferencer()
        out = inf.filter_results(self._make_results(), label_filter="cat")
        assert all(r["predicted_label"] == "cat" for r in out)
        assert len(out) == 2

    def test_min_confidence_filter(self):
        inf = _loaded_inferencer()
        out = inf.filter_results(self._make_results(), min_confidence=0.80)
        assert all(r["confidence"] >= 0.80 for r in out)
        assert len(out) == 2

    def test_only_low_confidence_filter(self):
        inf = _loaded_inferencer()
        out = inf.filter_results(self._make_results(), only_low_confidence=True)
        assert all(r["low_confidence"] for r in out)
        assert len(out) == 2

    def test_combined_label_and_confidence(self):
        inf = _loaded_inferencer()
        out = inf.filter_results(
            self._make_results(), label_filter="cat", min_confidence=0.90
        )
        assert len(out) == 1
        assert out[0]["confidence"] == 0.95

    def test_empty_results_returns_empty(self):
        inf = _loaded_inferencer()
        assert inf.filter_results([]) == []

    def test_min_confidence_zero_no_effect(self):
        inf = _loaded_inferencer()
        results = self._make_results()
        assert inf.filter_results(results, min_confidence=0.0) == results


# ================================================================== predict_folder

class TestPredictFolder:
    def test_empty_folder_returns_empty(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        results = inf.predict_folder(str(tmp_path))
        assert results == []

    def test_classifies_images_in_folder(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        for i in range(3):
            _make_image(str(tmp_path / f"img_{i}.png"))
        results = inf.predict_folder(str(tmp_path))
        assert len(results) == 3
        for r in results:
            assert r["predicted_label"] in ("a", "b")
            assert r["error"] is None

    def test_result_has_required_keys(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        _make_image(str(tmp_path / "x.png"))
        r = inf.predict_folder(str(tmp_path))[0]
        for key in ("filename", "path", "predicted_label", "confidence",
                    "top_k", "all_probs", "low_confidence", "error", "timestamp"):
            assert key in r

    def test_broken_image_gets_error_entry(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")
        results = inf.predict_folder(str(tmp_path))
        assert len(results) == 1
        assert results[0]["error"] is not None
        assert results[0]["predicted_label"] == "ERROR"

    def test_progress_callback(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        for i in range(4):
            _make_image(str(tmp_path / f"img_{i}.png"))
        calls = []
        inf.predict_folder(str(tmp_path), progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) == 4
        assert calls[-1] == (4, 4)

    def test_roi_templates_none(self, tmp_path):
        """roi_templates=None should not crash (was the IndexError bug)."""
        inf = _loaded_inferencer(2, ["a", "b"])
        _make_image(str(tmp_path / "x.png"))
        results = inf.predict_folder(str(tmp_path), roi_templates=None)
        assert len(results) == 1
        assert results[0]["error"] is None

    def test_roi_templates_empty_list(self, tmp_path):
        """roi_templates=[] must NOT crash (was IndexError at index [0])."""
        inf = _loaded_inferencer(2, ["a", "b"])
        _make_image(str(tmp_path / "x.png"))
        results = inf.predict_folder(str(tmp_path), roi_templates=[])
        assert len(results) == 1
        assert results[0]["error"] is None

    def test_roi_templates_with_roi(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        _make_image(str(tmp_path / "x.png"), size=(100, 100))
        roi_templates = [{"roi": {"x": 5, "y": 5, "w": 40, "h": 40}}]
        results = inf.predict_folder(str(tmp_path), roi_templates=roi_templates)
        assert results[0]["error"] is None

    def test_recursive_subfolder(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        inf = _loaded_inferencer(2, ["a", "b"])
        _make_image(str(tmp_path / "root.png"))
        _make_image(str(sub / "deep.png"))
        results = inf.predict_folder(str(tmp_path), recursive=True)
        assert len(results) == 2

    def test_non_recursive_ignores_subfolders(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        inf = _loaded_inferencer(2, ["a", "b"])
        _make_image(str(tmp_path / "root.png"))
        _make_image(str(sub / "deep.png"))
        results = inf.predict_folder(str(tmp_path), recursive=False)
        assert len(results) == 1

    def test_low_confidence_flag_set(self, tmp_path):
        inf = _loaded_inferencer(2, ["a", "b"])
        _make_image(str(tmp_path / "x.png"))
        results = inf.predict_folder(str(tmp_path))
        assert isinstance(results[0]["low_confidence"], bool)
