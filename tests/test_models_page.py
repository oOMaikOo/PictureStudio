"""Smoke tests for ModelsPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.models_page import ModelsPage
    p = ModelsPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page.project is sample_project


def test_set_project_none_does_not_crash(page):
    # ModelsPage.set_project calls _init_manager which checks if project is truthy
    page.set_project(None)


def test_key_attributes_exist(page):
    assert hasattr(page, "_tabs")
    assert hasattr(page, "model_loaded")


def test_manager_is_none_initially(page):
    assert page._manager is None


def test_manager_created_after_set_project(page, sample_project):
    page.set_project(sample_project)
    assert page._manager is not None
