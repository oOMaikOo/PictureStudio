"""
Unit tests for core.project — save/load, labels, ROIs, backup, validation.
"""
import json
import os

import pytest


# ---------------------------------------------------------------------------
# Label management
# ---------------------------------------------------------------------------

class TestLabels:
    def test_add_labels(self, sample_project):
        assert "gut" in sample_project.labels
        assert "schlecht" in sample_project.labels
        assert "neutral" in sample_project.labels

    def test_add_duplicate_label_ignored(self, sample_project):
        count_before = len(sample_project.labels)
        sample_project.add_label("gut", "#000000")
        assert len(sample_project.labels) == count_before

    def test_remove_label(self, sample_project):
        sample_project.remove_label("neutral")
        assert "neutral" not in sample_project.labels

    def test_rename_label(self, sample_project):
        sample_project.rename_label("gut", "prima")
        assert "prima" in sample_project.labels
        assert "gut" not in sample_project.labels
        for lbl in sample_project.image_labels.values():
            assert lbl != "gut"

    def test_label_colors(self, sample_project):
        assert sample_project.labels["gut"]["color"] == "#2ECC71"
        assert sample_project.labels["schlecht"]["color"] == "#E74C3C"

    def test_get_label_color(self, sample_project):
        assert sample_project.get_label_color("neutral") == "#3498DB"


# ---------------------------------------------------------------------------
# Image management
# ---------------------------------------------------------------------------

class TestImages:
    def test_image_count(self, sample_project):
        assert len(sample_project.images) == 15

    def test_image_labels_set(self, sample_project):
        labeled = sum(1 for v in sample_project.image_labels.values() if v)
        assert labeled == 15

    def test_set_and_get_image_label(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_image_label(path, "schlecht")
        assert sample_project.get_image_label(path) == "schlecht"

    def test_add_image_dedup(self, sample_project, tmp_dir):
        count_before = len(sample_project.images)
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.add_image(path)
        assert len(sample_project.images) == count_before

    def test_remove_image(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.remove_image(path)
        assert path not in sample_project.images
        assert path not in sample_project.image_labels
        assert path not in sample_project.rois

    def test_get_unlabeled_images_empty(self, sample_project):
        assert sample_project.get_unlabeled_images() == []

    def test_get_images_by_label(self, sample_project):
        gut_images = sample_project.get_images_by_label("gut")
        assert len(gut_images) == 5


# ---------------------------------------------------------------------------
# ROI management
# ---------------------------------------------------------------------------

class TestROIs:
    def test_roi_count(self, sample_project):
        total = sum(len(v) for v in sample_project.rois.values())
        assert total == 6

    def test_add_roi(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        before = len(sample_project.rois.get(path, []))
        sample_project.add_roi(path, {
            "id": "new_roi", "type": "rect",
            "x": 5.0, "y": 5.0, "w": 20.0, "h": 20.0,
            "label": "gut", "color": "#2ECC71",
        })
        assert len(sample_project.rois[path]) == before + 1

    def test_remove_roi(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        roi_id = sample_project.rois[path][0]["id"]
        count_before = len(sample_project.rois[path])
        sample_project.remove_roi(path, roi_id)
        assert len(sample_project.rois[path]) == count_before - 1

    def test_remove_nonexistent_roi_no_crash(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.remove_roi(path, "does_not_exist")  # should not raise

    def test_roi_label_preserved(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        roi = sample_project.rois[path][0]
        assert roi["label"] in sample_project.labels

    def test_roi_has_required_fields(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        roi = sample_project.rois[path][0]
        for field in ("id", "type", "x", "y", "w", "h", "label"):
            assert field in roi

    def test_get_roi_count(self, sample_project):
        assert sample_project.get_roi_count() == 6


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "saved.json")
        sample_project.save(path)
        assert os.path.exists(path)

    def test_save_is_valid_json(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "saved.json")
        sample_project.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "config" in data
        assert "images" in data
        assert "labels" in data

    def test_load_roundtrip(self, sample_project, tmp_dir):
        from core.project import Project
        path = os.path.join(tmp_dir, "rt.json")
        sample_project.save(path)
        p2 = Project.load(path)
        assert p2.config.name == sample_project.config.name
        assert len(p2.images) == len(sample_project.images)
        assert len(p2.labels) == len(sample_project.labels)

    def test_load_preserves_rois(self, sample_project, tmp_dir):
        from core.project import Project
        path = os.path.join(tmp_dir, "roi_rt.json")
        sample_project.save(path)
        p2 = Project.load(path)
        total = sum(len(v) for v in p2.rois.values())
        assert total == 6

    def test_load_preserves_labels(self, sample_project, tmp_dir):
        from core.project import Project
        path = os.path.join(tmp_dir, "lbl_rt.json")
        sample_project.save(path)
        p2 = Project.load(path)
        assert p2.image_labels == sample_project.image_labels

    def test_atomic_save_no_tmp_leftover(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "atomic.json")
        sample_project.save(path)
        assert not os.path.exists(path + ".tmp")

    def test_load_labels_are_dict(self, sample_project, tmp_dir):
        from core.project import Project
        path = os.path.join(tmp_dir, "lbl_dict.json")
        sample_project.save(path)
        p2 = Project.load(path)
        assert isinstance(p2.labels, dict)
        assert "gut" in p2.labels
        assert "color" in p2.labels["gut"]

    def test_load_missing_file_raises_file_not_found(self, tmp_dir):
        from core.project import Project
        with pytest.raises(FileNotFoundError, match="nicht gefunden"):
            Project.load(os.path.join(tmp_dir, "does_not_exist.json"))

    def test_load_corrupt_json_raises_value_error(self, tmp_dir):
        from core.project import Project
        bad = os.path.join(tmp_dir, "corrupt.json")
        with open(bad, "w") as f:
            f.write("{not valid json{{")
        with pytest.raises(ValueError, match="beschädigt"):
            Project.load(bad)

    def test_load_empty_file_raises_value_error(self, tmp_dir):
        from core.project import Project
        empty = os.path.join(tmp_dir, "empty.json")
        with open(empty, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match="beschädigt"):
            Project.load(empty)


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

class TestBackup:
    def test_backup_creates_file(self, sample_project, tmp_dir):
        backup_path = sample_project.create_backup(tmp_dir)
        assert backup_path and os.path.exists(backup_path)

    def test_backup_is_valid_json(self, sample_project, tmp_dir):
        backup_path = sample_project.create_backup(tmp_dir)
        with open(backup_path) as f:
            data = json.load(f)
        assert "config" in data


# ---------------------------------------------------------------------------
# Dashboard data
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_dashboard_returns_dict(self, sample_project):
        data = sample_project.get_dashboard_data()
        assert isinstance(data, dict)

    def test_dashboard_counts(self, sample_project):
        data = sample_project.get_dashboard_data()
        assert data.get("total_images") == 15
        assert data.get("labeled_images") == 15
        assert data.get("total_rois") == 6

    def test_dashboard_label_counts(self, sample_project):
        data = sample_project.get_dashboard_data()
        dist = data.get("label_counts", {})
        assert "gut" in dist
        assert "schlecht" in dist
        assert "neutral" in dist

    def test_dashboard_total_labels(self, sample_project):
        data = sample_project.get_dashboard_data()
        assert data.get("total_labels") == 3


# ---------------------------------------------------------------------------
# Validate image files
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validate_missing_files(self, sample_project):
        result = sample_project.validate_image_files()
        missing = result.get("missing", [])
        assert len(missing) == 15

    def test_validate_returns_dict(self, sample_project):
        result = sample_project.validate_image_files()
        assert isinstance(result, dict)
        assert "missing" in result
        assert "ok" in result


# ---------------------------------------------------------------------------
# QA label flags (Point 14)
# ---------------------------------------------------------------------------

class TestQAFlags:
    def test_default_no_flags(self, sample_project):
        for path in sample_project.images:
            assert not sample_project.is_label_uncertain(path)
            assert sample_project.get_label_flag(path) == {}

    def test_set_uncertain(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=True)
        assert sample_project.is_label_uncertain(path)

    def test_set_uncertain_with_comment(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_001.jpg")
        sample_project.set_label_flag(path, uncertain=True, comment="zu dunkel")
        flag = sample_project.get_label_flag(path)
        assert flag["uncertain"] is True
        assert flag["comment"] == "zu dunkel"

    def test_set_not_uncertain_clears_entry(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=True)
        sample_project.set_label_flag(path, uncertain=False)
        assert not sample_project.is_label_uncertain(path)
        assert path not in sample_project.image_label_flags

    def test_set_flag_comment_only_stored(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=False, comment="Notiz")
        assert path in sample_project.image_label_flags
        assert not sample_project.is_label_uncertain(path)

    def test_clear_label_flag(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=True, comment="prüfen")
        sample_project.clear_label_flag(path)
        assert not sample_project.is_label_uncertain(path)
        assert path not in sample_project.image_label_flags

    def test_clear_nonexistent_flag_no_crash(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.clear_label_flag(path)  # should not raise

    def test_get_uncertain_images(self, sample_project, tmp_dir):
        p0 = os.path.join(tmp_dir, "img_000.jpg")
        p1 = os.path.join(tmp_dir, "img_001.jpg")
        p2 = os.path.join(tmp_dir, "img_002.jpg")
        sample_project.set_label_flag(p0, uncertain=True)
        sample_project.set_label_flag(p1, uncertain=True)
        sample_project.set_label_flag(p2, uncertain=False)
        uncertain = sample_project.get_uncertain_images()
        assert p0 in uncertain
        assert p1 in uncertain
        assert p2 not in uncertain
        assert len(uncertain) == 2

    def test_get_uncertain_images_empty(self, sample_project):
        assert sample_project.get_uncertain_images() == []

    def test_get_uncertain_images_order_matches_project(self, sample_project, tmp_dir):
        # uncertain images should appear in the same order as project.images
        p5 = os.path.join(tmp_dir, "img_005.jpg")
        p3 = os.path.join(tmp_dir, "img_003.jpg")
        sample_project.set_label_flag(p5, uncertain=True)
        sample_project.set_label_flag(p3, uncertain=True)
        uncertain = sample_project.get_uncertain_images()
        img_order = [p for p in sample_project.images if p in {p3, p5}]
        assert uncertain == img_order

    def test_remove_image_clears_flag(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=True, comment="weg")
        sample_project.remove_image(path)
        assert path not in sample_project.image_label_flags

    def test_flags_survive_save_load_roundtrip(self, sample_project, tmp_dir):
        from core.project import Project
        p0 = os.path.join(tmp_dir, "img_000.jpg")
        p1 = os.path.join(tmp_dir, "img_001.jpg")
        sample_project.set_label_flag(p0, uncertain=True, comment="Grenzfall")
        sample_project.set_label_flag(p1, uncertain=True)
        save_path = os.path.join(tmp_dir, "qa_rt.json")
        sample_project.save(save_path)
        p2 = Project.load(save_path)
        assert p2.is_label_uncertain(p0)
        assert p2.get_label_flag(p0)["comment"] == "Grenzfall"
        assert p2.is_label_uncertain(p1)
        assert len(p2.get_uncertain_images()) == 2

    def test_flags_not_saved_when_all_clear(self, sample_project, tmp_dir):
        import json as _json
        save_path = os.path.join(tmp_dir, "no_flags.json")
        sample_project.save(save_path)
        with open(save_path) as f:
            data = _json.load(f)
        assert data.get("image_label_flags", {}) == {}

    def test_flags_key_present_in_saved_json(self, sample_project, tmp_dir):
        import json as _json
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=True)
        save_path = os.path.join(tmp_dir, "with_flags.json")
        sample_project.save(save_path)
        with open(save_path) as f:
            data = _json.load(f)
        assert "image_label_flags" in data
        assert path in data["image_label_flags"]

    def test_multiple_flags_independent(self, sample_project, tmp_dir):
        paths = [os.path.join(tmp_dir, f"img_{i:03d}.jpg") for i in range(5)]
        for i, p in enumerate(paths):
            sample_project.set_label_flag(p, uncertain=True, comment=f"c{i}")
        # Clearing one doesn't affect others
        sample_project.clear_label_flag(paths[2])
        uncertain = sample_project.get_uncertain_images()
        assert paths[2] not in uncertain
        assert all(p in uncertain for p in paths if p != paths[2])

    def test_overwrite_flag_updates_comment(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=True, comment="alt")
        sample_project.set_label_flag(path, uncertain=True, comment="neu")
        assert sample_project.get_label_flag(path)["comment"] == "neu"

    def test_set_uncertain_false_no_comment_removes_entry(self, sample_project, tmp_dir):
        path = os.path.join(tmp_dir, "img_000.jpg")
        sample_project.set_label_flag(path, uncertain=True, comment="x")
        sample_project.set_label_flag(path, uncertain=False, comment="")
        assert path not in sample_project.image_label_flags


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

class TestCrashRecovery:
    """Tests for Project.check_tmp_recovery — covers all branch conditions."""

    def test_no_tmp_file_returns_false(self, tmp_dir):
        from core.project import Project
        proj_path = os.path.join(tmp_dir, "proj.json")
        available, tmp_path = Project.check_tmp_recovery(proj_path)
        assert not available
        assert tmp_path == proj_path + ".tmp"

    def test_newer_tmp_is_detected(self, tmp_dir):
        import time
        from core.project import Project
        proj_path = os.path.join(tmp_dir, "proj.json")
        # Write main file first
        with open(proj_path, "w") as f:
            f.write("{}")
        time.sleep(0.05)
        # Write tmp file slightly later with valid JSON
        tmp_path = proj_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"format_version": 1, "config": {}, "images": []}, f)
        available, returned_tmp = Project.check_tmp_recovery(proj_path)
        assert available
        assert returned_tmp == tmp_path

    def test_older_tmp_is_ignored(self, tmp_dir):
        import time
        from core.project import Project
        proj_path = os.path.join(tmp_dir, "proj.json")
        tmp_path = proj_path + ".tmp"
        # Write tmp first, then main — tmp is older
        with open(tmp_path, "w") as f:
            json.dump({"format_version": 1}, f)
        time.sleep(0.05)
        with open(proj_path, "w") as f:
            f.write("{}")
        available, _ = Project.check_tmp_recovery(proj_path)
        assert not available

    def test_invalid_json_tmp_is_ignored(self, tmp_dir):
        import time
        from core.project import Project
        proj_path = os.path.join(tmp_dir, "proj.json")
        with open(proj_path, "w") as f:
            f.write("{}")
        time.sleep(0.05)
        tmp_path = proj_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write("NOT VALID JSON {{{")
        available, _ = Project.check_tmp_recovery(proj_path)
        assert not available

    def test_tmp_without_main_file_is_detected(self, tmp_dir):
        from core.project import Project
        proj_path = os.path.join(tmp_dir, "proj_new.json")
        tmp_path = proj_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"format_version": 1, "config": {}, "images": []}, f)
        # No main file exists yet
        available, returned_tmp = Project.check_tmp_recovery(proj_path)
        assert available
        assert returned_tmp == tmp_path
