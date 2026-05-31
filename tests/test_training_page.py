"""Smoke tests for TrainingPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.training_page import TrainingPage
    p = TrainingPage()
    # _hpt_thread is set lazily; initialise here to avoid closeEvent AttributeError
    p._hpt_thread = None
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    page.set_project(sample_project)
    assert page.project is sample_project


def test_key_attributes_exist(page):
    assert hasattr(page, "save_dir_label")
    assert hasattr(page, "_al_scan_btn")
    assert hasattr(page, "_history")


def test_signals_exist(page):
    assert hasattr(page, "training_finished")
    assert hasattr(page, "al_queue_updated")


def test_no_thread_initially(page):
    assert page._thread is None


def test_set_project_updates_save_dir_label(page, sample_project):
    page.set_project(sample_project)
    label_text = page.save_dir_label.text()
    assert "models" in label_text
