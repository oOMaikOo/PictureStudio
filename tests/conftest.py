"""
pytest fixtures for Image Labeling Studio tests.
"""
import json
import os
import sys
import tempfile

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_project(tmp_dir):
    """Create a minimal Project with labels, images, ROIs."""
    from core.project import Project

    p = Project()
    p.config.name = "TestProjekt"
    p.config.created_at = "2025-01-01T00:00:00"

    p.add_label("gut",     "#2ECC71")
    p.add_label("schlecht","#E74C3C")
    p.add_label("neutral", "#3498DB")

    # Fake image paths (files don't need to exist for most unit tests)
    for i in range(15):
        lbl = ["gut", "schlecht", "neutral"][i % 3]
        path = os.path.join(tmp_dir, f"img_{i:03d}.jpg")
        p.add_image(path)
        p.set_image_label(path, lbl)

    # Add some ROIs
    for i in range(6):
        path = os.path.join(tmp_dir, f"img_{i:03d}.jpg")
        lbl = ["gut", "schlecht"][i % 2]
        p.add_roi(path, {
            "id": f"roi{i}", "type": "rect",
            "x": 10.0, "y": 10.0, "w": 50.0, "h": 50.0,
            "label": lbl, "color": "#E74C3C",
        })

    project_path = os.path.join(tmp_dir, "test_project.json")
    p.save(project_path)
    return p


@pytest.fixture
def sample_images(tmp_dir):
    """Create real tiny PNG images for dataset tests."""
    paths = []
    try:
        from PIL import Image as PILImage
        for i in range(12):
            lbl = ["gut", "schlecht", "neutral"][i % 3]
            fname = os.path.join(tmp_dir, f"{lbl}_{i}.png")
            img = PILImage.new("RGB", (64, 64), color=(i * 20, i * 10, 100))
            img.save(fname)
            paths.append((fname, lbl))
    except ImportError:
        pytest.skip("Pillow nicht installiert")
    return paths, tmp_dir
