"""
Tests for core/labeling.py — LabelManager.

Covers: no-project guard, get_labels, get_label_names, get_color,
next_available_color, add_label (auto-color, duplicate guard),
remove_label (not-found guard, delegates to Project).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.labeling import LabelManager
from core.project import Project
from utils.config import DEFAULT_COLORS


# ------------------------------------------------------------------ helpers

def _manager_with_labels(*names):
    """Return a (LabelManager, Project) pair pre-populated with the given label names."""
    proj = Project()
    for i, name in enumerate(names):
        proj.add_label(name, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    return LabelManager(proj), proj


# ================================================================== no-project guard

class TestNoProject:
    def test_get_labels_returns_empty(self):
        assert LabelManager(None).get_labels() == {}

    def test_get_label_names_returns_empty(self):
        assert LabelManager(None).get_label_names() == []

    def test_get_color_returns_default(self):
        assert LabelManager(None).get_color("anything") == DEFAULT_COLORS[0]

    def test_next_available_color_returns_first_default(self):
        assert LabelManager(None).next_available_color() == DEFAULT_COLORS[0]

    def test_add_label_returns_false(self):
        assert LabelManager(None).add_label("x") is False

    def test_remove_label_returns_false(self):
        assert LabelManager(None).remove_label("x") is False


# ================================================================== get_labels / get_label_names

class TestGetLabels:
    def test_returns_full_dict(self):
        mgr, proj = _manager_with_labels("cat", "dog")
        labels = mgr.get_labels()
        assert "cat" in labels
        assert "dog" in labels

    def test_returns_empty_on_empty_project(self):
        mgr = LabelManager(Project())
        assert mgr.get_labels() == {}

    def test_get_label_names_list(self):
        mgr, _ = _manager_with_labels("a", "b", "c")
        names = mgr.get_label_names()
        assert set(names) == {"a", "b", "c"}

    def test_get_label_names_order_stable(self):
        mgr, _ = _manager_with_labels("x", "y")
        names = mgr.get_label_names()
        assert names == ["x", "y"]


# ================================================================== get_color

class TestGetColor:
    def test_returns_assigned_color(self):
        mgr, _ = _manager_with_labels("cat")
        expected = DEFAULT_COLORS[0]
        assert mgr.get_color("cat") == expected

    def test_unknown_label_returns_default(self):
        mgr, _ = _manager_with_labels("cat")
        assert mgr.get_color("ghost") == DEFAULT_COLORS[0]


# ================================================================== next_available_color

class TestNextAvailableColor:
    def test_first_color_when_no_labels(self):
        mgr = LabelManager(Project())
        assert mgr.next_available_color() == DEFAULT_COLORS[0]

    def test_skips_used_colors(self):
        proj = Project()
        proj.add_label("a", DEFAULT_COLORS[0])
        mgr = LabelManager(proj)
        assert mgr.next_available_color() == DEFAULT_COLORS[1]

    def test_all_colors_used_wraps_to_first(self):
        proj = Project()
        for i, c in enumerate(DEFAULT_COLORS):
            proj.add_label(f"lbl_{i}", c)
        mgr = LabelManager(proj)
        # All colors taken → falls back to first
        assert mgr.next_available_color() == DEFAULT_COLORS[0]


# ================================================================== add_label

class TestAddLabel:
    def test_returns_true_on_new_label(self):
        mgr = LabelManager(Project())
        assert mgr.add_label("cat") is True

    def test_label_appears_in_project(self):
        proj = Project()
        mgr = LabelManager(proj)
        mgr.add_label("cat")
        assert "cat" in proj.labels

    def test_returns_false_on_duplicate(self):
        mgr, _ = _manager_with_labels("cat")
        assert mgr.add_label("cat") is False

    def test_auto_color_assigned(self):
        proj = Project()
        mgr = LabelManager(proj)
        mgr.add_label("cat")
        assert proj.labels["cat"]["color"] in DEFAULT_COLORS

    def test_explicit_color_used(self):
        proj = Project()
        mgr = LabelManager(proj)
        mgr.add_label("cat", color="#AABBCC")
        assert proj.labels["cat"]["color"] == "#AABBCC"

    def test_description_stored(self):
        proj = Project()
        mgr = LabelManager(proj)
        mgr.add_label("cat", description="felid")
        assert proj.labels["cat"]["description"] == "felid"

    def test_sequential_colors_auto_incremented(self):
        proj = Project()
        mgr = LabelManager(proj)
        mgr.add_label("a")
        mgr.add_label("b")
        colors = [proj.labels["a"]["color"], proj.labels["b"]["color"]]
        # Both colors should be from the defaults and distinct
        assert colors[0] != colors[1]
        assert colors[0] in DEFAULT_COLORS
        assert colors[1] in DEFAULT_COLORS


# ================================================================== remove_label

class TestRemoveLabel:
    def test_returns_true_on_existing(self):
        mgr, _ = _manager_with_labels("cat")
        assert mgr.remove_label("cat") is True

    def test_label_gone_from_project(self):
        proj = Project()
        mgr = LabelManager(proj)
        mgr.add_label("cat")
        mgr.remove_label("cat")
        assert "cat" not in proj.labels

    def test_returns_false_on_missing(self):
        mgr = LabelManager(Project())
        assert mgr.remove_label("ghost") is False

    def test_cascades_to_image_labels(self):
        proj = Project()
        proj.add_label("cat")
        proj.images.append("/img.png")
        proj.image_labels["/img.png"] = "cat"
        mgr = LabelManager(proj)
        mgr.remove_label("cat")
        assert "/img.png" not in proj.image_labels

    def test_cascades_to_roi_labels(self):
        proj = Project()
        proj.add_label("cat")
        proj.rois["/img.png"] = [{"label": "cat"}]
        mgr = LabelManager(proj)
        mgr.remove_label("cat")
        assert proj.rois["/img.png"][0]["label"] == ""
