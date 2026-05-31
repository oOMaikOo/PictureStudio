"""Smoke tests for InferencePage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.inference_page import InferencePage
    p = InferencePage()
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
    assert hasattr(page, "inferencer")
    assert hasattr(page, "_all_results")
    assert hasattr(page, "_filtered")


def test_all_results_empty_initially(page):
    assert page._all_results == []


def test_signals_exist(page):
    assert hasattr(page, "al_queue_updated")
    assert hasattr(page, "labels_applied")
