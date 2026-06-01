"""
Tests for core/detection_dataset.py — prepare_yolo_dataset
"""
import os

import pytest

PIL = pytest.importorskip("PIL", reason="Pillow nicht installiert")

from PIL import Image as PILImage
from core.detection_dataset import prepare_yolo_dataset
from core.project import Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project_with_rois(tmp_path, n_images=6):
    """
    Create a Project whose images are real tiny PNG files with labeled ROIs.
    Returns the project.
    """
    import pathlib
    pathlib.Path(tmp_path).mkdir(parents=True, exist_ok=True)

    p = Project()
    p.add_label("gut", "#2ECC71")
    p.add_label("schlecht", "#E74C3C")

    labels = list(p.labels.keys())  # ["gut", "schlecht"]

    for i in range(n_images):
        img_path = str(tmp_path / f"img_{i:03d}.png")
        PILImage.new("RGB", (64, 64), color=(i * 20, 100, 50)).save(img_path)
        p.add_image(img_path)
        lbl = labels[i % 2]
        p.add_roi(img_path, {
            "id": f"roi{i}",
            "type": "rect",
            "x": 5.0, "y": 5.0, "w": 20.0, "h": 20.0,
            "label": lbl,
            "color": "#ffffff",
        })

    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_project_raises(tmp_path):
    """A project with no labeled ROIs raises ValueError."""
    p = Project()
    p.add_label("gut", "#ffffff")
    # No images, no ROIs

    with pytest.raises(ValueError):
        prepare_yolo_dataset(p, str(tmp_path / "out"))


def test_project_no_labels_raises(tmp_path):
    """A project with no labels at all raises ValueError."""
    p = Project()
    # One image but no labels defined
    img_path = str(tmp_path / "x.png")
    PILImage.new("RGB", (32, 32)).save(img_path)
    p.add_image(img_path)

    with pytest.raises(ValueError):
        prepare_yolo_dataset(p, str(tmp_path / "out"))


def test_creates_expected_directories(tmp_path):
    """prepare_yolo_dataset creates images/train, images/val, labels/train, labels/val."""
    p = _make_project_with_rois(tmp_path / "imgs", n_images=6)
    out_dir = str(tmp_path / "dataset")

    prepare_yolo_dataset(p, out_dir, train_split=0.8, seed=42)

    for split in ("train", "val"):
        assert os.path.isdir(os.path.join(out_dir, "images", split))
        assert os.path.isdir(os.path.join(out_dir, "labels", split))


def test_data_yaml_created(tmp_path):
    """data.yaml is written at the root of output_dir."""
    p = _make_project_with_rois(tmp_path / "imgs")
    out_dir = str(tmp_path / "dataset")
    yaml_path, _ = prepare_yolo_dataset(p, out_dir)

    assert os.path.isfile(yaml_path)
    assert yaml_path == os.path.join(out_dir, "data.yaml")


def test_data_yaml_contents(tmp_path):
    """data.yaml contains nc, names, train, val, and path entries."""
    p = _make_project_with_rois(tmp_path / "imgs")
    out_dir = str(tmp_path / "dataset")
    yaml_path, stats = prepare_yolo_dataset(p, out_dir)

    with open(yaml_path, encoding="utf-8") as fh:
        content = fh.read()

    assert f"nc: {stats['n_classes']}" in content
    assert "train:" in content
    assert "val:" in content
    assert "path:" in content
    for cls_name in stats["class_names"]:
        assert cls_name in content


def test_label_file_yolo_format(tmp_path):
    """Each label file line has 5 space-separated floats in [0, 1]."""
    p = _make_project_with_rois(tmp_path / "imgs", n_images=4)
    out_dir = str(tmp_path / "dataset")
    prepare_yolo_dataset(p, out_dir, seed=42)

    label_files = []
    for split in ("train", "val"):
        lbl_dir = os.path.join(out_dir, "labels", split)
        for fname in os.listdir(lbl_dir):
            if fname.endswith(".txt"):
                label_files.append(os.path.join(lbl_dir, fname))

    assert label_files, "No label files were written"

    for lbl_file in label_files:
        with open(lbl_file) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Expected 5 columns, got {len(parts)} in '{line}'"
            cls_idx = int(parts[0])
            assert cls_idx >= 0
            coords = [float(p) for p in parts[1:]]
            for val in coords:
                assert 0.0 <= val <= 1.0, f"Coordinate {val} out of [0,1]"


def test_train_val_split_honored(tmp_path):
    """With 10 images and 80% split, roughly 8 go to train."""
    p = _make_project_with_rois(tmp_path / "imgs", n_images=10)
    out_dir = str(tmp_path / "dataset")
    _, stats = prepare_yolo_dataset(p, out_dir, train_split=0.8, seed=0)

    assert stats["n_train"] == 8
    assert stats["n_val"] == 2


def test_stats_dict_keys(tmp_path):
    """prepare_yolo_dataset returns a stats dict with all expected keys."""
    p = _make_project_with_rois(tmp_path / "imgs", n_images=4)
    out_dir = str(tmp_path / "dataset")
    _, stats = prepare_yolo_dataset(p, out_dir)

    for key in ("n_images", "n_train", "n_val", "n_classes", "class_names", "n_annotations"):
        assert key in stats, f"Key '{key}' missing from stats"
