"""
Tests for core/project.py — relocate_images(), migrate_to_multi_label(),
migrate_to_single_label(), and related label-management methods.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.project import Project


# ------------------------------------------------------------------ helpers

def _project_with_images(paths, labels=None):
    """Return a bare Project with *paths* added and optional single-label mapping."""
    p = Project()
    for path in paths:
        p.images.append(path)
    p.add_label("cat", "#E74C3C")
    p.add_label("dog", "#3498DB")
    if labels:
        for path, lbl in labels.items():
            p.image_labels[path] = lbl
    return p


# ================================================================== relocate_images

class TestRelocateImages:
    def test_returns_count_of_relocated(self):
        proj = _project_with_images(
            ["/old/a.png", "/old/b.png", "/other/c.png"],
            {"/old/a.png": "cat", "/old/b.png": "dog"},
        )
        count = proj.relocate_images("/old", "/new")
        assert count == 2

    def test_paths_updated_in_images_list(self):
        proj = _project_with_images(["/old/a.png", "/old/b.png"])
        proj.relocate_images("/old", "/new")
        assert "/new/a.png" in proj.images
        assert "/new/b.png" in proj.images
        assert "/old/a.png" not in proj.images

    def test_image_labels_keys_updated(self):
        proj = _project_with_images(
            ["/old/a.png"], {"/old/a.png": "cat"}
        )
        proj.relocate_images("/old", "/new")
        assert proj.image_labels.get("/new/a.png") == "cat"
        assert "/old/a.png" not in proj.image_labels

    def test_rois_keys_updated(self):
        proj = _project_with_images(["/old/a.png"])
        proj.rois["/old/a.png"] = [{"x": 1, "y": 1, "w": 10, "h": 10}]
        proj.relocate_images("/old", "/new")
        assert "/new/a.png" in proj.rois
        assert "/old/a.png" not in proj.rois

    def test_non_matching_paths_untouched(self):
        proj = _project_with_images(["/other/x.png"], {"/other/x.png": "dog"})
        proj.relocate_images("/old", "/new")
        assert proj.images == ["/other/x.png"]
        assert proj.image_labels["/other/x.png"] == "dog"

    def test_no_match_returns_zero(self):
        proj = _project_with_images(["/data/img.png"])
        assert proj.relocate_images("/nowhere", "/new") == 0

    def test_empty_project_returns_zero(self):
        proj = Project()
        assert proj.relocate_images("/old", "/new") == 0

    def test_partial_prefix_not_matched(self):
        proj = _project_with_images(["/oldstuff/img.png"])
        count = proj.relocate_images("/old/", "/new/")
        assert count == 0  # "/oldstuff/" does not start with "/old/"
        assert proj.images == ["/oldstuff/img.png"]


# ================================================================== migrate_to_multi_label

class TestMigrateToMultiLabel:
    def test_returns_count_of_migrated(self):
        proj = _project_with_images(
            ["/a.png", "/b.png", "/c.png"],
            {"/a.png": "cat", "/b.png": "dog"},
        )
        count = proj.migrate_to_multi_label()
        assert count == 2

    def test_sets_multi_label_flag(self):
        proj = _project_with_images(["/a.png"], {"/a.png": "cat"})
        proj.migrate_to_multi_label()
        assert proj.config.multi_label is True

    def test_copies_labels_to_multi_labels(self):
        proj = _project_with_images(
            ["/a.png", "/b.png"],
            {"/a.png": "cat", "/b.png": "dog"},
        )
        proj.migrate_to_multi_label()
        assert proj.image_multi_labels["/a.png"] == ["cat"]
        assert proj.image_multi_labels["/b.png"] == ["dog"]

    def test_does_not_duplicate_existing_multi_label(self):
        proj = _project_with_images(["/a.png"], {"/a.png": "cat"})
        proj.image_multi_labels["/a.png"] = ["cat"]  # already there
        count = proj.migrate_to_multi_label()
        assert count == 0
        assert proj.image_multi_labels["/a.png"] == ["cat"]

    def test_preserves_existing_multi_labels(self):
        proj = _project_with_images(["/a.png"], {"/a.png": "cat"})
        proj.image_multi_labels["/a.png"] = ["dog"]
        proj.migrate_to_multi_label()
        assert "cat" in proj.image_multi_labels["/a.png"]
        assert "dog" in proj.image_multi_labels["/a.png"]

    def test_skips_images_with_empty_label(self):
        proj = _project_with_images(["/a.png"])  # no label
        count = proj.migrate_to_multi_label()
        assert count == 0
        assert "/a.png" not in proj.image_multi_labels


# ================================================================== migrate_to_single_label

class TestMigrateToSingleLabel:
    def test_returns_count_of_migrated(self):
        proj = Project()
        proj.images = ["/a.png", "/b.png", "/c.png"]
        proj.image_multi_labels = {"/a.png": ["cat"], "/b.png": ["dog", "cat"]}
        count = proj.migrate_to_single_label()
        assert count == 2

    def test_clears_multi_label_flag(self):
        proj = Project()
        proj.config.multi_label = True
        proj.image_multi_labels = {"/a.png": ["cat"]}
        proj.migrate_to_single_label()
        assert proj.config.multi_label is False

    def test_takes_first_multi_label(self):
        proj = Project()
        proj.image_multi_labels = {"/a.png": ["cat", "dog"]}
        proj.migrate_to_single_label()
        assert proj.image_labels["/a.png"] == "cat"

    def test_skips_images_with_empty_multi_labels(self):
        proj = Project()
        proj.image_multi_labels = {"/a.png": []}
        count = proj.migrate_to_single_label()
        assert count == 0
        assert "/a.png" not in proj.image_labels

    def test_roundtrip_migrate(self):
        """migrate_to_multi → migrate_to_single recovers original label."""
        proj = _project_with_images(
            ["/a.png", "/b.png"],
            {"/a.png": "cat", "/b.png": "dog"},
        )
        proj.migrate_to_multi_label()
        proj.migrate_to_single_label()
        assert proj.image_labels["/a.png"] == "cat"
        assert proj.image_labels["/b.png"] == "dog"


# ================================================================== remove_label cascades

class TestRemoveLabel:
    def test_removes_from_labels_dict(self):
        proj = Project()
        proj.add_label("cat")
        proj.remove_label("cat")
        assert "cat" not in proj.labels

    def test_removes_from_image_labels(self):
        proj = _project_with_images(["/a.png"], {"/a.png": "cat"})
        proj.remove_label("cat")
        assert "/a.png" not in proj.image_labels

    def test_removes_from_image_multi_labels(self):
        proj = Project()
        proj.add_label("cat")
        proj.add_label("dog")
        proj.image_multi_labels["/a.png"] = ["cat", "dog"]
        proj.remove_label("cat")
        assert proj.image_multi_labels["/a.png"] == ["dog"]

    def test_clears_roi_label(self):
        proj = Project()
        proj.add_label("cat")
        proj.rois["/a.png"] = [{"label": "cat", "x": 0, "y": 0, "w": 10, "h": 10}]
        proj.remove_label("cat")
        assert proj.rois["/a.png"][0]["label"] == ""

    def test_noop_on_nonexistent_label(self):
        proj = Project()
        proj.remove_label("ghost")   # should not raise


# ================================================================== get_unlabeled_images

class TestGetUnlabeledImages:
    def test_single_label_mode(self):
        proj = _project_with_images(
            ["/a.png", "/b.png"],
            {"/a.png": "cat"},
        )
        unlabeled = proj.get_unlabeled_images()
        assert "/b.png" in unlabeled
        assert "/a.png" not in unlabeled

    def test_multi_label_mode(self):
        proj = Project()
        proj.config.multi_label = True
        proj.images = ["/a.png", "/b.png"]
        proj.image_multi_labels["/a.png"] = ["cat"]
        unlabeled = proj.get_unlabeled_images()
        assert "/b.png" in unlabeled
        assert "/a.png" not in unlabeled

    def test_all_unlabeled(self):
        proj = _project_with_images(["/a.png", "/b.png"])
        assert len(proj.get_unlabeled_images()) == 2

    def test_all_labeled(self):
        proj = _project_with_images(
            ["/a.png"], {"/a.png": "cat"}
        )
        assert proj.get_unlabeled_images() == []
