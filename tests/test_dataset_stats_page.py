"""Smoke tests for DatasetStatsPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.dataset_stats_page import DatasetStatsPage
    p = DatasetStatsPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page.project is sample_project


def test_set_project_none_does_not_crash(page):
    page.set_project(None)


def test_refresh_with_no_project_does_not_crash(page):
    page.project = None
    page.refresh()


def test_key_attributes_exist(page):
    assert hasattr(page, "_lbl_layout")
    assert hasattr(page, "_dup_list")
