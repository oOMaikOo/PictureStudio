"""
Tests for core/batch_inference.py — BatchInferenceWorker.

Covers: happy path, cancellation, broken images, progress callback,
class_names bounds, empty list, model with no parameters.
"""
import os
import sys
import tempfile

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

from core.batch_inference import BatchInferenceWorker


# ------------------------------------------------------------------ helpers

def _model(n_classes: int = 2, image_size: int = 32) -> nn.Module:
    """Tiny flat model compatible with 3×image_size×image_size input."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * image_size * image_size, n_classes),
    )
    model.eval()
    return model


def _make_image(path: str, size=(32, 32)):
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return path


# ================================================================== happy path

class TestBatchInferenceWorkerHappy:
    def test_classifies_all_images(self, tmp_path):
        paths = [_make_image(str(tmp_path / f"img_{i}.png")) for i in range(5)]
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        results = worker.run(paths)
        assert len(results) == 5

    def test_result_keys(self, tmp_path):
        p = _make_image(str(tmp_path / "x.png"))
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        r = worker.run([p])[0]
        for key in ("path", "filename", "predicted", "confidence", "probabilities", "error"):
            assert key in r

    def test_predicted_is_class_name(self, tmp_path):
        p = _make_image(str(tmp_path / "x.png"))
        worker = BatchInferenceWorker(
            model=_model(), class_names=["cat", "dog"], image_size=32
        )
        r = worker.run([p])[0]
        assert r["predicted"] in ("cat", "dog")

    def test_confidence_in_range(self, tmp_path):
        p = _make_image(str(tmp_path / "x.png"))
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        r = worker.run([p])[0]
        assert 0.0 <= r["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, tmp_path):
        p = _make_image(str(tmp_path / "x.png"))
        worker = BatchInferenceWorker(
            model=_model(3), class_names=["a", "b", "c"], image_size=32
        )
        r = worker.run([p])[0]
        total = sum(r["probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_error_is_none_on_success(self, tmp_path):
        p = _make_image(str(tmp_path / "x.png"))
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        r = worker.run([p])[0]
        assert r["error"] is None

    def test_filename_is_basename(self, tmp_path):
        p = _make_image(str(tmp_path / "photo.png"))
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        r = worker.run([p])[0]
        assert r["filename"] == "photo.png"

    def test_empty_list_returns_empty(self):
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        assert worker.run([]) == []


# ================================================================== broken images

class TestBatchInferenceWorkerErrors:
    def test_broken_image_gets_error_entry(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        results = worker.run([str(bad)])
        assert len(results) == 1
        assert results[0]["error"] is not None
        assert results[0]["confidence"] == 0.0

    def test_missing_file_gets_error_entry(self, tmp_path):
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        results = worker.run([str(tmp_path / "nonexistent.png")])
        assert results[0]["error"] is not None

    def test_good_images_after_bad_still_processed(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"garbage")
        good = _make_image(str(tmp_path / "good.png"))
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        results = worker.run([str(bad), good])
        assert len(results) == 2
        assert results[0]["error"] is not None
        assert results[1]["error"] is None


# ================================================================== cancellation

class TestBatchInferenceWorkerCancel:
    def test_cancel_stops_early(self, tmp_path):
        paths = [_make_image(str(tmp_path / f"img_{i}.png")) for i in range(10)]
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        processed = []

        def progress(current, total):
            processed.append(current)
            if current >= 3:
                worker.cancel()

        worker.progress_cb = progress
        results = worker.run(paths)
        # Should stop before processing all 10
        assert len(results) < 10

    def test_cancel_before_run_returns_empty(self, tmp_path):
        paths = [_make_image(str(tmp_path / f"img_{i}.png")) for i in range(5)]
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        worker.cancel()
        results = worker.run(paths)
        assert results == []

    def test_cancel_flag_is_false_initially(self):
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        assert not worker._cancelled

    def test_cancel_sets_flag(self):
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32
        )
        worker.cancel()
        assert worker._cancelled


# ================================================================== progress callback

class TestBatchInferenceWorkerProgress:
    def test_progress_callback_called_for_each_image(self, tmp_path):
        paths = [_make_image(str(tmp_path / f"img_{i}.png")) for i in range(4)]
        calls = []
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32,
            progress_cb=lambda c, t: calls.append((c, t)),
        )
        worker.run(paths)
        assert len(calls) == 4
        assert calls[0] == (1, 4)
        assert calls[-1] == (4, 4)

    def test_no_progress_callback_is_fine(self, tmp_path):
        p = _make_image(str(tmp_path / "x.png"))
        worker = BatchInferenceWorker(
            model=_model(), class_names=["a", "b"], image_size=32,
            progress_cb=None,
        )
        results = worker.run([p])
        assert len(results) == 1


# ================================================================== edge cases

class TestBatchInferenceWorkerEdgeCases:
    def test_model_with_no_parameters_uses_cpu(self, tmp_path):
        """Model with no parameters (StopIteration path) should not crash."""
        p = _make_image(str(tmp_path / "x.png"))
        # nn.Identity has no parameters → triggers the StopIteration cpu fallback
        # The forward pass may or may not error; we just verify no unhandled exception
        empty_model = nn.Identity()
        worker = BatchInferenceWorker(
            model=empty_model, class_names=["a", "b"], image_size=32
        )
        # Should not raise — either succeeds or error is caught per-image
        results = worker.run([p])
        assert len(results) == 1

    def test_single_class(self, tmp_path):
        p = _make_image(str(tmp_path / "x.png"))
        worker = BatchInferenceWorker(
            model=_model(1), class_names=["only"], image_size=32
        )
        r = worker.run([p])[0]
        assert r["predicted"] == "only"

    def test_class_names_bounds_guard(self, tmp_path):
        """pred_idx < len(class_names) guard: extra logit should use str(idx)."""
        p = _make_image(str(tmp_path / "x.png"))
        # Model outputs 3 logits but we only give 2 class names
        worker = BatchInferenceWorker(
            model=_model(3), class_names=["a", "b"], image_size=32
        )
        r = worker.run([p])[0]
        # Should not raise; predicted is either a class name or str(idx)
        assert isinstance(r["predicted"], str)
