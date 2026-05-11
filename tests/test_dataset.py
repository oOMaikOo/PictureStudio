"""
Unit tests for core.dataset — analysis, splits, duplicate detection, exports.
"""
import csv
import json
import os

import pytest


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_analyze_returns_dict(self, sample_project):
        from core.dataset import analyze_dataset
        result = analyze_dataset(sample_project)
        assert isinstance(result, dict)

    def test_analyze_label_counts(self, sample_project):
        from core.dataset import analyze_dataset
        result = analyze_dataset(sample_project)
        dist = result.get("label_counts", {})
        assert dist.get("gut", 0) == 5
        assert dist.get("schlecht", 0) == 5
        assert dist.get("neutral", 0) == 5

    def test_analyze_missing_files(self, sample_project):
        from core.dataset import analyze_dataset
        result = analyze_dataset(sample_project)
        # All fake paths → all missing
        missing = result.get("missing_files", [])
        assert len(missing) == 15

    def test_analyze_warnings_key_present(self, sample_project):
        from core.dataset import analyze_dataset
        result = analyze_dataset(sample_project)
        assert "warnings" in result

    def test_analyze_total_and_labeled(self, sample_project):
        from core.dataset import analyze_dataset
        result = analyze_dataset(sample_project)
        assert result.get("total") == 15
        assert result.get("labeled") == 15
        assert result.get("unlabeled") == 0

    def test_analyze_with_real_images(self, sample_images):
        from core.dataset import analyze_dataset
        from core.project import Project

        paths, tmp_dir = sample_images
        p = Project()
        for fname, lbl in paths:
            p.add_image(fname)
            p.add_label(lbl, "#FFFFFF")
            p.set_image_label(fname, lbl)

        result = analyze_dataset(p)
        assert result.get("missing_files", []) == []


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_returns_three_lists(self, sample_project):
        from core.dataset import create_stratified_split
        train, val, test = create_stratified_split(sample_project, val_ratio=0.2, test_ratio=0.1)
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(test, list)

    def test_split_no_overlap(self, sample_project):
        from core.dataset import create_stratified_split
        train, val, test = create_stratified_split(sample_project, val_ratio=0.2, test_ratio=0.1)
        all_paths = train + val + test
        assert len(all_paths) == len(set(all_paths))

    def test_split_total_equals_labeled(self, sample_project):
        from core.dataset import create_stratified_split
        train, val, test = create_stratified_split(sample_project, val_ratio=0.2, test_ratio=0.1)
        labeled = [p for p, l in sample_project.image_labels.items() if l]
        assert len(train) + len(val) + len(test) == len(labeled)

    def test_split_val_ratio(self, sample_project):
        from core.dataset import create_stratified_split
        train, val, test = create_stratified_split(sample_project, val_ratio=0.2, test_ratio=0.0)
        # val should be roughly 20% — allow ±2 due to stratification rounding
        assert 1 <= len(val) <= 5

    def test_split_paths_are_strings(self, sample_project):
        from core.dataset import create_stratified_split
        train, val, test = create_stratified_split(sample_project)
        for p in train + val + test:
            assert isinstance(p, str)

    def test_split_reproducible(self, sample_project):
        from core.dataset import create_stratified_split
        t1, v1, _ = create_stratified_split(sample_project, seed=7)
        t2, v2, _ = create_stratified_split(sample_project, seed=7)
        assert sorted(t1) == sorted(t2)
        assert sorted(v1) == sorted(v2)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_no_duplicates_in_unique_set(self, sample_images):
        from core.dataset import analyze_dataset
        from core.project import Project

        paths, _ = sample_images
        p = Project()
        for fname, lbl in paths:
            p.add_image(fname)
            p.add_label(lbl, "#FFFFFF")
            p.set_image_label(fname, lbl)

        result = analyze_dataset(p)
        assert result.get("duplicates", []) == []

    def test_duplicate_detected(self, tmp_dir):
        from core.dataset import analyze_dataset
        from core.project import Project

        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow nicht installiert")

        img = PILImage.new("RGB", (32, 32), color=(100, 100, 100))
        p1 = os.path.join(tmp_dir, "dup_a.png")
        p2 = os.path.join(tmp_dir, "dup_b.png")
        img.save(p1)
        img.save(p2)

        p = Project()
        p.add_label("gut", "#2ECC71")
        for path in (p1, p2):
            p.add_image(path)
            p.set_image_label(path, "gut")

        result = analyze_dataset(p)
        dups = result.get("duplicates", [])
        assert len(dups) >= 1


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_csv_export_creates_file(self, sample_project, tmp_dir):
        from core.dataset import export_csv
        out = os.path.join(tmp_dir, "out.csv")
        export_csv(sample_project, out)
        assert os.path.exists(out)

    def test_csv_has_header(self, sample_project, tmp_dir):
        from core.dataset import export_csv
        out = os.path.join(tmp_dir, "out.csv")
        export_csv(sample_project, out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        assert header is not None
        # CSV uses image_path or image_label columns
        assert any("path" in h.lower() or "image" in h.lower() for h in header)

    def test_csv_row_count(self, sample_project, tmp_dir):
        from core.dataset import export_csv
        out = os.path.join(tmp_dir, "out.csv")
        export_csv(sample_project, out)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 15


# ---------------------------------------------------------------------------
# COCO export
# ---------------------------------------------------------------------------

class TestCOCOExport:
    def test_coco_export_creates_file(self, sample_images, tmp_dir):
        """COCO export skips images that don't exist; use real images."""
        from core.dataset import export_coco
        from core.project import Project

        paths, _ = sample_images
        p = Project()
        for fname, lbl in paths:
            p.add_image(fname)
            p.add_label(lbl, "#FFFFFF")
            p.set_image_label(fname, lbl)
            p.add_roi(fname, {
                "id": "r1", "type": "rect",
                "x": 0.0, "y": 0.0, "w": 20.0, "h": 20.0,
                "label": lbl, "color": "#fff",
            })

        out = os.path.join(tmp_dir, "coco.json")
        export_coco(p, out)
        assert os.path.exists(out)

    def test_coco_structure(self, sample_images, tmp_dir):
        from core.dataset import export_coco
        from core.project import Project

        paths, _ = sample_images
        p = Project()
        for fname, lbl in paths:
            p.add_image(fname)
            p.add_label(lbl, "#FFFFFF")
            p.set_image_label(fname, lbl)

        out = os.path.join(tmp_dir, "coco.json")
        export_coco(p, out)
        with open(out) as f:
            data = json.load(f)
        for key in ("images", "annotations", "categories"):
            assert key in data

    def test_coco_categories_match_labels(self, sample_images, tmp_dir):
        from core.dataset import export_coco
        from core.project import Project

        paths, _ = sample_images
        p = Project()
        label_set = set()
        for fname, lbl in paths:
            p.add_image(fname)
            p.add_label(lbl, "#FFFFFF")
            p.set_image_label(fname, lbl)
            label_set.add(lbl)

        out = os.path.join(tmp_dir, "coco2.json")
        export_coco(p, out)
        with open(out) as f:
            data = json.load(f)
        cat_names = {c["name"] for c in data["categories"]}
        assert label_set == cat_names


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------

class TestYOLOExport:
    def test_yolo_export_creates_classes_txt(self, sample_images, tmp_dir):
        from core.dataset import export_yolo
        from core.project import Project

        paths, _ = sample_images
        p = Project()
        for fname, lbl in paths:
            p.add_image(fname)
            p.add_label(lbl, "#FFFFFF")
            p.set_image_label(fname, lbl)
            p.add_roi(fname, {
                "id": "r1", "type": "rect",
                "x": 0.0, "y": 0.0, "w": 20.0, "h": 20.0,
                "label": lbl, "color": "#fff",
            })

        out_dir = os.path.join(tmp_dir, "yolo")
        export_yolo(p, out_dir)
        assert os.path.exists(os.path.join(out_dir, "classes.txt"))

    def test_yolo_classes_content(self, sample_images, tmp_dir):
        from core.dataset import export_yolo
        from core.project import Project

        paths, _ = sample_images
        p = Project()
        label_set = set()
        for fname, lbl in paths:
            p.add_image(fname)
            p.add_label(lbl, "#FFFFFF")
            p.set_image_label(fname, lbl)
            label_set.add(lbl)

        out_dir = os.path.join(tmp_dir, "yolo2")
        export_yolo(p, out_dir)
        with open(os.path.join(out_dir, "classes.txt")) as f:
            classes = [l.strip() for l in f if l.strip()]
        assert label_set == set(classes)
