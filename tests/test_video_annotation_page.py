"""Tests für gui/pages/video_annotation_page.py"""
from __future__ import annotations
import pytest
from gui.pages.video_annotation_page import VideoAnnotationPage


def test_instantiates_without_args(qtbot):
    page = VideoAnnotationPage()
    qtbot.addWidget(page)


def test_set_project_does_not_crash(qtbot, sample_project):
    page = VideoAnnotationPage()
    qtbot.addWidget(page)
    page.set_project(sample_project)


def test_set_project_none_does_not_crash(qtbot):
    page = VideoAnnotationPage()
    qtbot.addWidget(page)
    page.set_project(None)


def test_has_load_video_button(qtbot):
    from PySide6.QtWidgets import QAbstractButton
    page = VideoAnnotationPage()
    qtbot.addWidget(page)
    btns = [b for b in page.findChildren(QAbstractButton) if "Video" in b.text()]
    assert btns, "Kein 'Video laden' Button gefunden"


def test_frame_paths_empty_initially(qtbot):
    page = VideoAnnotationPage()
    qtbot.addWidget(page)
    assert page._frame_paths == []


def test_frame_labels_empty_initially(qtbot):
    page = VideoAnnotationPage()
    qtbot.addWidget(page)
    assert page._frame_labels == {}
