"""Smoke tests for ExportPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.export_page import ExportPage
    p = ExportPage()
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
    assert hasattr(page, "count_label")
    assert hasattr(page, "file_label")


def test_results_empty_initially(page):
    assert page._results == []


def test_set_project_with_no_results_shows_no_results(page, sample_project):
    # sample_project has no inference_results
    page.set_project(sample_project)
    assert page._results == []
