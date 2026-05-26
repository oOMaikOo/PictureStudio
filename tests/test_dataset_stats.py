"""Tests für gui/pages/dataset_stats_page.py"""
from __future__ import annotations
import pytest
from gui.pages.dataset_stats_page import DatasetStatsPage


def test_instantiates_without_args(qtbot):
    page = DatasetStatsPage()
    qtbot.addWidget(page)


def test_set_project_with_sample_project(qtbot, sample_project):
    page = DatasetStatsPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)


def test_set_project_none_does_not_crash(qtbot):
    page = DatasetStatsPage()
    qtbot.addWidget(page)
    page.set_project(None)


def test_has_refresh_method(qtbot):
    page = DatasetStatsPage()
    qtbot.addWidget(page)
    assert callable(page.refresh)


def test_refresh_without_project_does_not_crash(qtbot):
    page = DatasetStatsPage()
    qtbot.addWidget(page)
    page.refresh()


def test_refresh_with_project_does_not_crash(qtbot, sample_project):
    page = DatasetStatsPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)
    page.refresh()


def test_refresh_survives_broken_project(qtbot):
    """refresh() must not propagate exceptions from a malformed project."""

    class BrokenProject:
        @property
        def images(self):
            raise RuntimeError("intentional test error")

    page = DatasetStatsPage()
    qtbot.addWidget(page)
    page.project = BrokenProject()
    page.refresh()  # must not raise
