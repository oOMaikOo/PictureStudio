"""Tests für core/augmentation_pipeline.py"""
from __future__ import annotations
import os
import pytest

PIL = pytest.importorskip("PIL", reason="Pillow not installed")
from PIL import Image as PILImage

from core.augmentation_pipeline import AugmentationPipeline, AugmentationWorker


def _make_img(tmp_path, name="test.png"):
    p = str(tmp_path / name)
    PILImage.new("RGB", (64, 64), color=(100, 150, 200)).save(p)
    return p


def test_augmentation_pipeline_default_config():
    pipeline = AugmentationPipeline()
    assert pipeline._config["copies_per_image"] == 3
    assert pipeline._config["rotation"] is True


def test_augment_image_creates_files(tmp_path):
    img_path = _make_img(tmp_path)
    pipeline = AugmentationPipeline({"copies_per_image": 2})
    out = pipeline.augment_image(img_path, str(tmp_path / "aug"))
    assert len(out) == 2
    for p in out:
        assert os.path.isfile(p)


def test_augment_image_returns_correct_count(tmp_path):
    img_path = _make_img(tmp_path)
    for n in [1, 3, 5]:
        pipeline = AugmentationPipeline({"copies_per_image": n})
        out = pipeline.augment_image(img_path, str(tmp_path / f"aug{n}"))
        assert len(out) == n


def test_augmentation_worker_run_returns_dict(tmp_path):
    img_path = _make_img(tmp_path)
    pairs = [(img_path, "normal")]
    worker = AugmentationWorker(pairs, str(tmp_path / "out"), {"copies_per_image": 2})
    result = worker.run()
    assert isinstance(result, dict)
    assert "generated" in result
    assert "label_map" in result
    assert "skipped" in result


def test_run_returns_label_map(tmp_path):
    img_path = _make_img(tmp_path)
    pairs = [(img_path, "defekt")]
    worker = AugmentationWorker(pairs, str(tmp_path / "out"), {"copies_per_image": 2})
    result = worker.run()
    for path, lbl in result["label_map"].items():
        assert lbl == "defekt"


def test_run_handles_missing_file_gracefully(tmp_path):
    pairs = [("/nonexistent/img.png", "label")]
    worker = AugmentationWorker(pairs, str(tmp_path / "out"), {})
    result = worker.run()
    assert result["skipped"] == 1
    assert result["generated"] == []
