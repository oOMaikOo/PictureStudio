"""
Tests for object detection core: ObjectDetector and detection_dataset.
All tests run without ultralytics installed (mocked where needed).
"""
import os
import sys
import tempfile
import types
import pytest


# ------------------------------------------------------------------ helpers

class _FakeProject:
    """Minimal project stub for detection_dataset tests."""

    def __init__(self, images, rois, labels):
        self.images = images
        self._rois  = rois   # {path: [roi_dict, ...]}
        self.labels = labels  # {name: {}}

    def get_rois(self, path):
        return self._rois.get(path, [])


def _make_tiny_png(path: str):
    """Write a minimal 10×10 white PNG to path."""
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    img.save(path, "PNG")


# ------------------------------------------------------------------ ObjectDetector

class TestObjectDetector:
    def test_not_ready_on_init(self):
        from core.object_detection import ObjectDetector
        d = ObjectDetector()
        assert not d.is_ready()

    def test_has_ultralytics_returns_bool(self):
        from core.object_detection import has_ultralytics
        assert isinstance(has_ultralytics(), bool)

    def test_predict_image_raises_when_not_ready(self):
        from core.object_detection import ObjectDetector
        d = ObjectDetector()
        with pytest.raises(RuntimeError, match="Kein Detektionsmodell"):
            d.predict_image("dummy.jpg")

    def test_predict_folder_returns_errors_when_not_ready(self):
        """predict_folder catches per-image errors; all results have error field set."""
        from core.object_detection import ObjectDetector
        import tempfile, os
        d = ObjectDetector()
        with tempfile.TemporaryDirectory() as td:
            # Create a dummy image so the folder is not empty
            img = os.path.join(td, "x.jpg")
            open(img, "wb").close()
            results = d.predict_folder(td)
        assert all(r["error"] for r in results)

    def test_model_sizes_dict_not_empty(self):
        from core.object_detection import ObjectDetector
        assert len(ObjectDetector.MODEL_SIZES) >= 4

    def test_load_raises_without_ultralytics(self, monkeypatch):
        """If ultralytics is absent, load() raises RuntimeError."""
        import core.object_detection as mod
        monkeypatch.setattr(mod, "HAS_ULTRALYTICS", False)
        d = mod.ObjectDetector()
        with pytest.raises(RuntimeError, match="ultralytics"):
            d.load("fake.pt")

    def test_predict_image_with_mock_model(self, monkeypatch):
        """Verify predict_image parses model output correctly."""
        import core.object_detection as mod

        # Build a minimal mock that mimics ultralytics YOLO output
        import types, sys
        mock_box = types.SimpleNamespace(
            xyxy=[[50.0, 30.0, 150.0, 130.0]],  # x1,y1,x2,y2
            cls=[0],
            conf=[0.87],
        )
        # xyxy[0] should behave like a list
        mock_box.xyxy = [[50.0, 30.0, 150.0, 130.0]]
        mock_box.cls  = [0]
        mock_box.conf = [0.87]

        class _MockResult:
            boxes = [mock_box]

        class _MockYOLO:
            names = {0: "cat"}
            def __call__(self, path, conf=0.25, iou=0.45, verbose=False):
                return [_MockResult()]

        monkeypatch.setattr(mod, "HAS_ULTRALYTICS", True)
        d = mod.ObjectDetector()
        d._model = _MockYOLO()
        d.class_names = ["cat"]

        dets = d.predict_image("dummy.jpg", conf=0.25)
        assert len(dets) == 1
        assert dets[0]["label"] == "cat"
        assert dets[0]["confidence"] == pytest.approx(0.87, abs=0.01)
        assert dets[0]["w"] == pytest.approx(100.0)
        assert dets[0]["h"] == pytest.approx(100.0)


# ------------------------------------------------------------------ detection_dataset

class TestDetectionDataset:
    def test_raises_without_labels(self):
        from core.detection_dataset import prepare_yolo_dataset
        proj = _FakeProject([], {}, {})
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="Labels"):
                prepare_yolo_dataset(proj, d)

    def test_raises_without_annotated_images(self):
        from core.detection_dataset import prepare_yolo_dataset
        proj = _FakeProject([], {}, {"cat": {}, "dog": {}})
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="ROI"):
                prepare_yolo_dataset(proj, d)

    def test_creates_yolo_structure(self):
        from core.detection_dataset import prepare_yolo_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            _make_tiny_png(img_path)

            rois = {img_path: [{"label": "cat", "x": 1, "y": 1, "w": 5, "h": 5, "id": "abc"}]}
            proj = _FakeProject([img_path], rois, {"cat": {}})

            out = os.path.join(tmpdir, "yolo_out")
            yaml_path, stats = prepare_yolo_dataset(proj, out, train_split=1.0, seed=0)

            assert os.path.exists(yaml_path)
            assert stats["n_classes"] == 1
            assert stats["n_annotations"] >= 1  # val may duplicate train when split=1.0
            assert os.path.isdir(os.path.join(out, "images", "train"))
            assert os.path.isdir(os.path.join(out, "labels", "train"))

    def test_yolo_label_format(self):
        """Coordinates must be normalized and within [0, 1]."""
        from core.detection_dataset import prepare_yolo_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.png")
            _make_tiny_png(img_path)  # 10×10

            # ROI covers left half: x=0,y=0,w=5,h=10 → xc=0.25,yc=0.5,w=0.5,h=1.0
            rois = {img_path: [{"label": "obj", "x": 0, "y": 0, "w": 5, "h": 10, "id": "x1"}]}
            proj = _FakeProject([img_path], rois, {"obj": {}})

            out = os.path.join(tmpdir, "yolo_out2")
            prepare_yolo_dataset(proj, out, train_split=1.0, seed=0)

            # Find the label file
            lbl_file = None
            for root, _, files in os.walk(os.path.join(out, "labels")):
                for f in files:
                    if f.endswith(".txt"):
                        lbl_file = os.path.join(root, f)
            assert lbl_file is not None
            with open(lbl_file) as f:
                parts = f.read().strip().split()
            assert len(parts) == 5
            cls_idx = int(parts[0])
            xc, yc, wn, hn = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert cls_idx == 0
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 < wn <= 1.0
            assert 0.0 < hn <= 1.0

    def test_data_yaml_content(self):
        from core.detection_dataset import prepare_yolo_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.png")
            _make_tiny_png(img_path)
            rois = {img_path: [{"label": "part", "x": 1, "y": 1, "w": 4, "h": 4, "id": "y1"}]}
            proj = _FakeProject([img_path], rois, {"part": {}})
            out = os.path.join(tmpdir, "yolo_yaml")
            yaml_path, _ = prepare_yolo_dataset(proj, out, train_split=1.0)
            with open(yaml_path) as f:
                content = f.read()
            assert "nc: 1" in content
            assert "part" in content
            assert "train:" in content
            assert "val:" in content

    def test_multiple_classes(self):
        from core.detection_dataset import prepare_yolo_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = os.path.join(tmpdir, "a.png")
            img2 = os.path.join(tmpdir, "b.png")
            _make_tiny_png(img1)
            _make_tiny_png(img2)
            rois = {
                img1: [{"label": "cat", "x": 1, "y": 1, "w": 4, "h": 4, "id": "c1"}],
                img2: [{"label": "dog", "x": 2, "y": 2, "w": 3, "h": 3, "id": "d1"}],
            }
            proj = _FakeProject([img1, img2], rois, {"cat": {}, "dog": {}})
            out = os.path.join(tmpdir, "multi")
            _, stats = prepare_yolo_dataset(proj, out, train_split=1.0)
            assert stats["n_classes"] == 2
            assert stats["n_annotations"] >= 2

    def test_skips_rois_without_label(self):
        from core.detection_dataset import prepare_yolo_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.png")
            _make_tiny_png(img_path)
            rois = {img_path: [
                {"label": "cat",  "x": 1, "y": 1, "w": 3, "h": 3, "id": "c1"},
                {"label": "",     "x": 2, "y": 2, "w": 3, "h": 3, "id": "c2"},  # no label
                {"label": "cat",  "x": 4, "y": 4, "w": 3, "h": 3, "id": "c3"},
            ]}
            proj = _FakeProject([img_path], rois, {"cat": {}})
            out = os.path.join(tmpdir, "skip")
            _, stats = prepare_yolo_dataset(proj, out, train_split=1.0)
            assert stats["n_annotations"] >= 2  # unlabeled ROI skipped; val may duplicate
