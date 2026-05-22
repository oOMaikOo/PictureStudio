"""
Tests for core/pre_labeling.py — PreLabeler without a real model (mocked).
"""
from __future__ import annotations

import os
import tempfile
import types

import pytest


# ------------------------------------------------------------------ helpers

class _FakeInferencer:
    """Minimal stand-in that returns configurable predictions."""

    def __init__(self, label="cat", conf=0.90):
        self.class_names = ["cat", "dog"]
        self._label = label
        self._conf  = conf

    def is_ready(self):
        return True

    def predict_image(self, path, roi=None, top_k=1, **kw):
        return {
            "predicted_label": self._label,
            "confidence":      self._conf,
            "top_k":           [{"label": self._label, "prob": self._conf}],
        }


# ------------------------------------------------------------------ PreLabeler

class TestPreLabeler:
    def test_not_ready_on_init(self):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        assert not pl.is_ready()

    def test_suggest_raises_without_model(self):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        with pytest.raises(RuntimeError, match="Kein Modell"):
            pl.suggest(["x.png"], ["cat"])

    def test_suggest_returns_results(self, monkeypatch):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        pl._inferencer = _FakeInferencer(label="cat", conf=0.92)
        pl.class_names = ["cat", "dog"]

        with tempfile.TemporaryDirectory() as td:
            paths = [os.path.join(td, f"img_{i}.png") for i in range(3)]
            for p in paths:
                open(p, "wb").close()  # files don't need to be real — inferencer is mocked

            results = pl.suggest(paths, ["cat", "dog"], confidence_threshold=0.80)

        assert len(results) == 3
        for r in results:
            assert r["label"] == "cat"
            assert r["confidence"] == pytest.approx(0.92, abs=0.01)
            assert not r["skip"]
            assert r["error"] is None

    def test_suggest_skips_low_confidence(self, monkeypatch):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        pl._inferencer = _FakeInferencer(label="cat", conf=0.50)
        pl.class_names = ["cat", "dog"]

        results = pl.suggest(["x.png"], ["cat", "dog"], confidence_threshold=0.75)
        assert len(results) == 1
        assert results[0]["skip"]

    def test_suggest_skips_unknown_label(self, monkeypatch):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        pl._inferencer = _FakeInferencer(label="bird", conf=0.95)  # not in project
        pl.class_names = ["bird"]

        results = pl.suggest(["x.png"], ["cat", "dog"], confidence_threshold=0.75)
        assert results[0]["skip"]   # label not in project_labels

    def test_suggest_handles_inference_error(self):
        from core.pre_labeling import PreLabeler

        class _ErrorInferencer(_FakeInferencer):
            def predict_image(self, path, **kw):
                raise RuntimeError("Fake inference error")

        pl = PreLabeler()
        pl._inferencer = _ErrorInferencer()
        pl.class_names = ["cat"]

        results = pl.suggest(["x.png"], ["cat"])
        assert results[0]["skip"]
        assert results[0]["error"] is not None

    def test_suggest_progress_callback(self):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        pl._inferencer = _FakeInferencer()
        pl.class_names = ["cat", "dog"]

        calls = []
        pl.suggest(
            ["a.png", "b.png", "c.png"],
            ["cat", "dog"],
            progress_callback=lambda c, t: calls.append((c, t)),
        )
        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_suggest_all_accepted_above_threshold(self):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        pl._inferencer = _FakeInferencer(label="dog", conf=0.85)
        pl.class_names = ["cat", "dog"]

        results = pl.suggest(
            ["x.png", "y.png"],
            ["cat", "dog"],
            confidence_threshold=0.80,
        )
        accepted = [r for r in results if not r["skip"]]
        assert len(accepted) == 2
        assert all(r["label"] == "dog" for r in accepted)

    def test_suggest_empty_paths(self):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        pl._inferencer = _FakeInferencer()
        pl.class_names = ["cat"]
        results = pl.suggest([], ["cat"])
        assert results == []

    def test_confidence_rounded(self):
        from core.pre_labeling import PreLabeler
        pl = PreLabeler()
        pl._inferencer = _FakeInferencer(label="cat", conf=0.123456789)
        pl.class_names = ["cat"]
        results = pl.suggest(["x.png"], ["cat"], confidence_threshold=0.0)
        assert len(str(results[0]["confidence"]).split(".")[-1]) <= 4


# ------------------------------------------------------------------ BulkSetImageLabelCommand with label_map

class TestBulkLabelCommandWithMap:
    """Verify that BulkSetImageLabelCommand supports per-image label_map."""

    def test_label_map_redo(self, monkeypatch):
        from gui.labeling_commands import BulkSetImageLabelCommand

        applied = {}

        class _FakePage:
            def _do_set_image_label(self, path, label):
                applied[path] = label

        page = _FakePage()
        old  = {"a.png": "", "b.png": "cat"}
        lmap = {"a.png": "dog", "b.png": "cat"}

        cmd = BulkSetImageLabelCommand(page, ["a.png", "b.png"], "", old, label_map=lmap)
        cmd.redo()
        assert applied == {"a.png": "dog", "b.png": "cat"}

    def test_label_map_undo(self, monkeypatch):
        from gui.labeling_commands import BulkSetImageLabelCommand

        applied = {}

        class _FakePage:
            def _do_set_image_label(self, path, label):
                applied[path] = label

        page = _FakePage()
        old  = {"a.png": "bird", "b.png": ""}
        lmap = {"a.png": "dog", "b.png": "cat"}

        cmd = BulkSetImageLabelCommand(page, ["a.png", "b.png"], "", old, label_map=lmap)
        cmd.redo()
        cmd.undo()
        assert applied == {"a.png": "bird", "b.png": ""}

    def test_without_label_map_uses_new_label(self):
        from gui.labeling_commands import BulkSetImageLabelCommand

        applied = {}

        class _FakePage:
            def _do_set_image_label(self, path, label):
                applied[path] = label

        page = _FakePage()
        old  = {"x.png": "cat"}
        cmd  = BulkSetImageLabelCommand(page, ["x.png"], "dog", old)
        cmd.redo()
        assert applied["x.png"] == "dog"
