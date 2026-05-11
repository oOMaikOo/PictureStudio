"""
Unit tests for ROI data structures — storage, retrieval, deletion, templates.
"""
import os

import pytest


# ---------------------------------------------------------------------------
# ROI CRUD via Project
# ---------------------------------------------------------------------------

class TestROICRUD:
    def test_add_rect_roi(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        roi = {"id": "r1", "type": "rect", "x": 0.0, "y": 0.0, "w": 100.0, "h": 50.0,
               "label": "gut", "color": "#2ECC71"}
        sample_project.add_roi(path, roi)
        ids = [r["id"] for r in sample_project.rois[path]]
        assert "r1" in ids

    def test_add_ellipse_roi(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_001.jpg")
        roi = {"id": "e1", "type": "ellipse", "x": 10.0, "y": 10.0, "w": 60.0, "h": 40.0,
               "label": "schlecht", "color": "#E74C3C"}
        sample_project.add_roi(path, roi)
        ids = [r["id"] for r in sample_project.rois[path]]
        assert "e1" in ids

    def test_add_polygon_roi(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_002.jpg")
        roi = {"id": "p1", "type": "polygon",
               "points": [[10, 10], [50, 10], [30, 50]],
               "label": "gut", "color": "#2ECC71"}
        sample_project.add_roi(path, roi)
        ids = [r["id"] for r in sample_project.rois[path]]
        assert "p1" in ids

    def test_remove_roi_by_id(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        roi_id = sample_project.rois[path][0]["id"]
        count_before = len(sample_project.rois[path])
        sample_project.remove_roi(path, roi_id)
        assert len(sample_project.rois[path]) == count_before - 1

    def test_remove_nonexistent_roi_no_crash(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.remove_roi(path, "does_not_exist")  # should not raise

    def test_update_roi(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        roi_id = sample_project.rois[path][0]["id"]
        updated = dict(sample_project.rois[path][0])
        updated["x"] = 99.0
        sample_project.update_roi(path, roi_id, updated)
        found = next(r for r in sample_project.rois[path] if r["id"] == roi_id)
        assert found["x"] == pytest.approx(99.0)

    def test_rois_per_image_independent(self, sample_project, tmp_dir):
        path0 = os.path.join(tmp_dir, "img_000.jpg")
        path1 = os.path.join(tmp_dir, "img_001.jpg")
        count0 = len(sample_project.rois.get(path0, []))
        count1 = len(sample_project.rois.get(path1, []))
        sample_project.remove_roi(path0, sample_project.rois[path0][0]["id"])
        assert len(sample_project.rois.get(path0, [])) == count0 - 1
        assert len(sample_project.rois.get(path1, [])) == count1  # unchanged

    def test_get_rois_returns_list(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        rois = sample_project.get_rois(path)
        assert isinstance(rois, list)

    def test_get_rois_empty_for_unknown_path(self, sample_project):
        rois = sample_project.get_rois("/nonexistent/path.jpg")
        assert rois == []


# ---------------------------------------------------------------------------
# ROI serialization round-trip
# ---------------------------------------------------------------------------

class TestROISerialisation:
    def test_rect_survives_save_load(self, sample_project, tmp_dir):
        from core.project import Project
        path = os.path.join(tmp_dir, "img_000.jpg")
        roi = {"id": "serial_r", "type": "rect",
               "x": 5.5, "y": 10.5, "w": 80.0, "h": 40.0,
               "label": "gut", "color": "#2ECC71"}
        sample_project.add_roi(path, roi)
        save_path = os.path.join(tmp_dir, "roi_test.json")
        sample_project.save(save_path)

        p2 = Project.load(save_path)
        matches = [r for r in p2.rois.get(path, []) if r["id"] == "serial_r"]
        assert len(matches) == 1
        assert matches[0]["x"] == pytest.approx(5.5)
        assert matches[0]["type"] == "rect"

    def test_polygon_points_survive_save_load(self, sample_project, tmp_dir):
        from core.project import Project
        path = os.path.join(tmp_dir, "img_000.jpg")
        points = [[1, 2], [3, 4], [5, 6]]
        roi = {"id": "serial_p", "type": "polygon",
               "points": points, "label": "schlecht", "color": "#E74C3C"}
        sample_project.add_roi(path, roi)
        save_path = os.path.join(tmp_dir, "poly_test.json")
        sample_project.save(save_path)

        p2 = Project.load(save_path)
        matches = [r for r in p2.rois.get(path, []) if r["id"] == "serial_p"]
        assert len(matches) == 1
        assert matches[0]["points"] == points

    def test_all_roi_types_preserved(self, tmp_dir):
        from core.project import Project
        p = Project()
        p.add_label("gut", "#2ECC71")
        img_path = os.path.join(tmp_dir, "multi.jpg")
        p.add_image(img_path)
        p.set_image_label(img_path, "gut")

        rois = [
            {"id": "r1", "type": "rect", "x": 0, "y": 0, "w": 10, "h": 10,
             "label": "gut", "color": "#fff"},
            {"id": "e1", "type": "ellipse", "x": 0, "y": 0, "w": 10, "h": 10,
             "label": "gut", "color": "#fff"},
            {"id": "p1", "type": "polygon", "points": [[0, 0], [10, 0], [5, 10]],
             "label": "gut", "color": "#fff"},
        ]
        for roi in rois:
            p.add_roi(img_path, roi)

        save_path = os.path.join(tmp_dir, "types.json")
        p.save(save_path)
        p2 = Project.load(save_path)

        saved_types = {r["type"] for r in p2.rois.get(img_path, [])}
        assert "rect" in saved_types
        assert "ellipse" in saved_types
        assert "polygon" in saved_types


# ---------------------------------------------------------------------------
# ROI validation
# ---------------------------------------------------------------------------

class TestROIValidation:
    def test_roi_requires_id(self, sample_project):
        for roi_list in sample_project.rois.values():
            for roi in roi_list:
                assert "id" in roi

    def test_roi_requires_type(self, sample_project):
        for roi_list in sample_project.rois.values():
            for roi in roi_list:
                assert "type" in roi
                assert roi["type"] in ("rect", "ellipse", "polygon")

    def test_roi_rect_has_geometry(self, sample_project):
        for roi_list in sample_project.rois.values():
            for roi in roi_list:
                if roi["type"] == "rect":
                    for field in ("x", "y", "w", "h"):
                        assert field in roi

    def test_roi_label_valid(self, sample_project):
        for roi_list in sample_project.rois.values():
            for roi in roi_list:
                assert roi.get("label") in sample_project.labels


# ---------------------------------------------------------------------------
# ROI templates
# ---------------------------------------------------------------------------

class TestROITemplates:
    def test_add_roi_template(self, tmp_dir):
        from core.project import Project
        p = Project()
        p.add_label("gut", "#2ECC71")
        template_roi = {"type": "rect", "x": 25.0, "y": 25.0, "w": 50.0,
                        "h": 50.0, "label": "gut", "color": "#2ECC71"}
        p.add_roi_template("center_crop", template_roi)
        templates = p.get_roi_templates()
        assert any(t["name"] == "center_crop" for t in templates)

    def test_apply_roi_template(self, tmp_dir):
        from core.project import Project
        p = Project()
        p.add_label("gut", "#2ECC71")
        template_roi = {"type": "rect", "x": 0.0, "y": 0.0, "w": 100.0,
                        "h": 100.0, "label": "gut", "color": "#2ECC71"}
        p.add_roi_template("full_frame", template_roi)

        img = os.path.join(tmp_dir, "tpl.jpg")
        p.add_image(img)
        p.set_image_label(img, "gut")
        count = p.apply_roi_template("full_frame", [img])
        assert count == 1
        assert len(p.rois.get(img, [])) == 1

    def test_apply_nonexistent_template_returns_zero(self, tmp_dir):
        from core.project import Project
        p = Project()
        p.add_label("gut", "#2ECC71")
        img = os.path.join(tmp_dir, "x.jpg")
        p.add_image(img)
        count = p.apply_roi_template("no_such_template", [img])
        assert count == 0

    def test_applied_roi_has_unique_id(self, tmp_dir):
        from core.project import Project
        p = Project()
        p.add_label("gut", "#2ECC71")
        template_roi = {"type": "rect", "x": 0.0, "y": 0.0, "w": 50.0,
                        "h": 50.0, "label": "gut", "color": "#2ECC71"}
        p.add_roi_template("t1", template_roi)

        imgs = [os.path.join(tmp_dir, f"img{i}.jpg") for i in range(3)]
        for img in imgs:
            p.add_image(img)
        p.apply_roi_template("t1", imgs)

        all_ids = [r["id"] for img in imgs for r in p.rois.get(img, [])]
        assert len(all_ids) == len(set(all_ids))  # all unique
