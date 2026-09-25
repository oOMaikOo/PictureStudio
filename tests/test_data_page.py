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


# ── Threaded exports (BACKLOG T) ─────────────────────────────────────────────

def _project_with_images(tmp_path):
    from core.project import Project
    from PIL import Image
    p = Project()
    p.project_path = str(tmp_path / "p.json")
    p.add_label("gut", "#3FB950")
    for i in range(3):
        f = tmp_path / f"img{i}.png"
        Image.new("RGB", (16, 16)).save(f)
        p.images.append(str(f))
        p.image_labels[str(f)] = "gut"
    return p


def test_export_runs_off_the_ui_thread(qtbot, tmp_path, monkeypatch):
    """The export function must not execute on the GUI thread."""
    import threading
    from gui.pages.data_page import DataPage
    from PySide6.QtWidgets import QMessageBox

    page = DataPage()
    qtbot.addWidget(page)
    page.set_project(_project_with_images(tmp_path))

    seen = {}

    def fake_export(project, target):
        seen["thread"] = threading.current_thread().name
        seen["target"] = target

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    page._start_export(fake_export, str(tmp_path / "out.json"), "data.msg.csv_saved",
                       path=str(tmp_path / "out.json"))
    thread = page._export_thread
    qtbot.waitUntil(lambda: page._export_thread is None, timeout=5000)

    assert seen["thread"] != threading.main_thread().name
    assert thread.isFinished()


def test_export_error_is_surfaced(qtbot, tmp_path, monkeypatch):
    """A failing export shows a critical box, not a success box."""
    from gui.pages.data_page import DataPage
    from PySide6.QtWidgets import QMessageBox

    page = DataPage()
    qtbot.addWidget(page)
    page.set_project(_project_with_images(tmp_path))

    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append("critical"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append("information"))

    def boom(project, target):
        raise OSError("disk full")

    page._start_export(boom, str(tmp_path / "x.json"), "data.msg.csv_saved",
                       path=str(tmp_path / "x.json"))
    qtbot.waitUntil(lambda: page._export_thread is None, timeout=5000)

    assert shown == ["critical"]


def test_export_buttons_locked_during_export(qtbot, tmp_path, monkeypatch):
    """Export buttons are disabled while an export runs and re-enabled after."""
    import time
    from gui.pages.data_page import DataPage
    from PySide6.QtWidgets import QMessageBox

    page = DataPage()
    qtbot.addWidget(page)
    page.set_project(_project_with_images(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    def slow_export(project, target):
        time.sleep(0.3)

    page._start_export(slow_export, str(tmp_path / "s.json"), "data.msg.csv_saved",
                       path=str(tmp_path / "s.json"))
    assert all(not b.isEnabled() for b in page._export_btns)
    qtbot.waitUntil(lambda: page._export_thread is None, timeout=5000)
    assert all(b.isEnabled() for b in page._export_btns)
