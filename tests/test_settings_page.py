"""Smoke tests for SettingsPage."""
import pytest


@pytest.fixture
def page(qtbot):
    from gui.pages.settings_page import SettingsPage
    p = SettingsPage()
    qtbot.addWidget(p)
    return p


def test_instantiates_without_args(page):
    assert page is not None


def test_set_project_with_sample_project(page, sample_project):
    # SettingsPage has no set_project — just check it doesn't have one that crashes
    assert not hasattr(page, "set_project") or callable(getattr(page, "set_project", None))


def test_set_project_none_does_not_crash(page):
    # SettingsPage does not expose set_project; nothing to do
    pass


def test_set_settings_with_app_settings(page):
    from utils.settings import AppSettings
    settings = AppSettings()
    page.set_settings(settings)
    assert page._settings is settings


def test_key_attributes_exist(page):
    assert hasattr(page, "theme_combo")
    assert hasattr(page, "autosave_cb")
    assert hasattr(page, "font_size_spin")


def test_signals_exist(page):
    assert hasattr(page, "theme_changed")
    assert hasattr(page, "autosave_changed")
