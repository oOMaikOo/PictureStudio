"""Smoke tests for BatchInferencePage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.batch_inference_page import BatchInferencePage
    p = BatchInferencePage()
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
    assert hasattr(page, "_run_btn")
    assert hasattr(page, "_model_combo")
    assert hasattr(page, "_table")


def test_no_model_loaded_initially(page):
    assert page._model is None


def test_results_empty_initially(page):
    assert page._results == []
