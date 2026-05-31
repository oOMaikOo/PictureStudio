"""Smoke tests for ObjectDetectionPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.object_detection_page import ObjectDetectionPage
    p = ObjectDetectionPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page.project is sample_project


def test_set_project_none_does_not_crash(page):
    page.set_project(None)


def test_key_attributes_exist(page):
    assert hasattr(page, "_detector")
    assert hasattr(page, "_prepare_btn")
    assert hasattr(page, "_ds_info_label")


def test_results_empty_initially(page):
    assert page._all_results == []
