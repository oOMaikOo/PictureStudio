"""Smoke tests for AnomalyTrainingPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.anomaly_training_page import AnomalyTrainingPage
    p = AnomalyTrainingPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)


def test_set_project_none_does_not_crash(page):
    page.set_project(None)


def test_camera_page_reference_is_none_by_default(page):
    assert page._camera_page is None


def test_set_camera_page(page):
    class FakeCameraPage:
        _model_path = None
        _detector = None
    fake = FakeCameraPage()
    page.set_camera_page(fake)
    assert page._camera_page is fake


def test_model_status_label_exists(page):
    assert hasattr(page, "_model_status_lbl")
