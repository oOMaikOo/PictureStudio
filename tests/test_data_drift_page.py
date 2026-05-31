"""Smoke tests for DataDriftPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.data_drift_page import DataDriftPage
    p = DataDriftPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page.project is sample_project


def test_set_project_none_does_not_crash(page):
    page.set_project(None)


def test_drift_detector_exists(page):
    assert hasattr(page, "_detector")
    assert page._detector is not None


def test_results_empty_initially(page):
    assert page._results == []
