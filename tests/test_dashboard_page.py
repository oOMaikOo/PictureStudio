"""Smoke tests for DashboardPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.dashboard_page import DashboardPage
    p = DashboardPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page.project is sample_project


def test_set_project_none_does_not_crash(page):
    page.set_project(None)


def test_stat_cards_exist(page):
    assert hasattr(page, "_cards")
    assert "total_images" in page._cards
    assert "labeled_images" in page._cards


def test_signals_exist(page):
    assert hasattr(page, "open_project_requested")
    assert hasattr(page, "new_project_requested")


def test_no_project_widget_exists(page):
    assert hasattr(page, "_no_project_widget")
