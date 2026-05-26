"""Tests for gui/pages/data_page.py"""
from __future__ import annotations
import pytest
from gui.pages.data_page import DataPage


def test_analyze_btn_disabled_without_project(qtbot):
    """_analyze_btn must be disabled when no project is loaded."""
    page = DataPage()
    qtbot.addWidget(page)
    assert not page._analyze_btn.isEnabled()


def test_export_btns_disabled_without_project(qtbot):
    """All export buttons must be disabled when no project is loaded."""
    page = DataPage()
    qtbot.addWidget(page)
    assert len(page._export_btns) == 3
    for btn in page._export_btns:
        assert not btn.isEnabled()


def test_buttons_enabled_after_set_project(qtbot, sample_project):
    """After set_project(), analyze and export buttons must become enabled."""
    page = DataPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)
    assert page._analyze_btn.isEnabled()
    for btn in page._export_btns:
        assert btn.isEnabled()


def test_buttons_disabled_after_project_cleared(qtbot, sample_project):
    """set_project(None) must disable analyze/export buttons again."""
    page = DataPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)
    page.set_project(None)
    assert not page._analyze_btn.isEnabled()
    for btn in page._export_btns:
        assert not btn.isEnabled()
