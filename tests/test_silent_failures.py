"""Tests for failures that used to disappear into bare `except: pass`.

Each test drives the error path and asserts that the failure is either logged
or surfaced to the user — never both swallowed and reported as success.
"""
import logging
import os
import tempfile

import pytest


def test_i18n_reports_argument_mismatch(caplog):
    """A translation whose placeholders don't match the call must warn, not hide."""
    from utils.i18n import init_i18n, tr
    init_i18n("de")
    with caplog.at_level(logging.WARNING, logger="utils.i18n"):
        out = tr("videoanno.frame_progress", wrong_arg=1)
    assert "{cur}" in out                      # falls back to the raw text
    assert any("frame_progress" in r.getMessage() for r in caplog.records)


def test_audit_reports_unreadable_log(caplog):
    from core.audit import AuditTrail
    d = tempfile.mkdtemp()
    trail = AuditTrail(d, "test")
    trail.log("test", "entity", {})
    os.chmod(trail._file, 0o000)
    try:
        with caplog.at_level(logging.WARNING):
            assert trail.get_entries(10) == []
        assert any("nicht lesbar" in r.getMessage() for r in caplog.records)
    finally:
        os.chmod(trail._file, 0o644)


def test_video_annotation_reports_failed_save(qtbot, monkeypatch, tmp_path):
    """A failing project.save() must not be reported as a successful transfer."""
    from PySide6.QtWidgets import QMessageBox
    from gui.pages.video_annotation_page import VideoAnnotationPage
    from core.project import Project
    from PIL import Image

    project = Project()
    project.project_path = str(tmp_path / "p.json")
    project.add_label("gut", "#3FB950")

    frame = tmp_path / "frame0.png"
    Image.new("RGB", (16, 16)).save(frame)

    page = VideoAnnotationPage()
    qtbot.addWidget(page)
    page.set_project(project)
    page._frame_paths = [str(frame)]
    page._frame_labels = {0: "gut"}

    def _boom(*_a, **_kw):
        raise OSError("disk full")
    monkeypatch.setattr(project, "save", _boom)

    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *a, **k: shown.append(("critical", a)))
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown.append(("information", a)))

    page._save_to_project()

    kinds = [k for k, _ in shown]
    assert "critical" in kinds, "failed save must be reported"
    assert "information" not in kinds, "must not also claim success"
