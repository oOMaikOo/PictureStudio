"""
Tests for core/project_index.py — ProjectSearchIndex
"""
import os
import threading

import pytest

from core.project_index import ProjectSearchIndex
from core.project import Project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path, n_images=5, labels=None):
    """Create a minimal in-memory Project (not saved to disk)."""
    p = Project()
    if labels is None:
        labels = ["gut", "schlecht"]
    for lbl in labels:
        p.add_label(lbl, "#ffffff")
    for i in range(n_images):
        path = str(tmp_path / f"img_{i:03d}.jpg")
        p.images.append(path)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rebuild_empty_project(tmp_path):
    """Rebuild with an empty project; all query methods return empty results."""
    p = Project()  # no images, no labels
    idx = ProjectSearchIndex()
    idx.rebuild(p)

    assert idx.get_label_counts() == {}
    assert idx.get_unlabeled() == []
    assert idx.get_images_by_label("gut") == []
    assert idx.get_labeled_count() == 0
    assert idx.is_ready is True


def test_get_label_counts(tmp_path):
    """Label counts must match the labels assigned in the project."""
    p = _make_project(tmp_path, n_images=6, labels=["gut", "schlecht", "neutral"])
    for i, path in enumerate(p.images):
        lbl = p.labels_list()[i % 3] if hasattr(p, "labels_list") else list(p.labels.keys())[i % 3]
        p.image_labels[path] = lbl

    idx = ProjectSearchIndex()
    idx.rebuild(p)

    counts = idx.get_label_counts()
    # 6 images split evenly across 3 labels → 2 each
    assert sum(counts.values()) == 6
    for lbl in p.labels:
        assert counts.get(lbl) == 2


def test_get_images_by_label(tmp_path):
    """get_images_by_label returns exactly the paths with that primary label."""
    p = _make_project(tmp_path, n_images=9, labels=["gut", "schlecht"])
    # Assign 4 "gut", 3 "schlecht", leave 2 unlabeled
    for i, path in enumerate(p.images):
        if i < 4:
            p.image_labels[path] = "gut"
        elif i < 7:
            p.image_labels[path] = "schlecht"
        # else: unlabeled

    idx = ProjectSearchIndex()
    idx.rebuild(p)

    gut_paths = idx.get_images_by_label("gut")
    schlecht_paths = idx.get_images_by_label("schlecht")
    assert len(gut_paths) == 4
    assert len(schlecht_paths) == 3
    assert all(p.images[i] in gut_paths for i in range(4))


def test_get_unlabeled(tmp_path):
    """get_unlabeled returns images that have no label assigned."""
    p = _make_project(tmp_path, n_images=8, labels=["gut", "schlecht"])
    # Label only the first 5
    for i, path in enumerate(p.images):
        if i < 5:
            p.image_labels[path] = "gut"

    idx = ProjectSearchIndex()
    idx.rebuild(p)

    unlabeled = idx.get_unlabeled()
    assert len(unlabeled) == 3
    # The unlabeled paths should be the last 3
    for path in p.images[5:]:
        assert path in unlabeled


def test_rebuild_updates_index(tmp_path):
    """Calling rebuild a second time with different data replaces the old index."""
    p = _make_project(tmp_path, n_images=4, labels=["gut"])
    for path in p.images:
        p.image_labels[path] = "gut"

    idx = ProjectSearchIndex()
    idx.rebuild(p)
    assert idx.get_labeled_count() == 4

    # Build a new project with different data and rebuild
    p2 = _make_project(tmp_path / "b", n_images=2, labels=["schlecht"])
    # No labels assigned → labeled_count should be 0

    idx.rebuild(p2)
    assert idx.get_labeled_count() == 0
    assert idx.get_unlabeled() == p2.images


def test_invalidate_makes_queries_return_empty(tmp_path):
    """After invalidate(), queries return empty results until rebuild is called again."""
    p = _make_project(tmp_path, n_images=3, labels=["gut"])
    for path in p.images:
        p.image_labels[path] = "gut"

    idx = ProjectSearchIndex()
    idx.rebuild(p)
    assert idx.get_labeled_count() == 3
    assert idx.is_ready is True

    idx.invalidate()
    assert idx.is_ready is False
    assert idx.get_label_counts() == {}
    assert idx.get_unlabeled() == []
