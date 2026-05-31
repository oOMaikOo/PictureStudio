"""Smoke tests for CameraPage."""
import pytest
from unittest.mock import patch


@pytest.fixture
def page(qtbot):
    with patch("gui.pages.camera_page.list_usb_cameras", return_value=[]):
        from gui.pages.camera_page import CameraPage
        p = CameraPage()
        qtbot.addWidget(p)
        yield p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page._project is sample_project


def test_set_project_none_does_not_crash(page):
    page.set_project(None)


def test_key_attributes_exist(page):
    assert hasattr(page, "_detector")
    assert hasattr(page, "_model_lbl")
    assert hasattr(page, "_scoring_btn")


def test_detector_is_none_initially(page):
    assert page._detector is None


def test_model_path_is_none_initially(page):
    assert page._model_path is None
