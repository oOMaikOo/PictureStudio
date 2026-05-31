"""Smoke tests for LabelingPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.labeling_page import LabelingPage
    p = LabelingPage()
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
    assert hasattr(page, "search_edit")
    assert hasattr(page, "sort_combo")
    assert hasattr(page, "_undo_stack")


def test_signals_exist(page):
    assert hasattr(page, "project_changed")
    assert hasattr(page, "al_retrain_requested")


def test_current_image_empty_initially(page):
    assert page._current_image == ""
