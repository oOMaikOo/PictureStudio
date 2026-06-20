"""Tests for live video classification (Inferencer.predict_frame + the page)."""
import numpy as np
import pytest


def test_predict_frame_raises_without_model():
    from core.inference import Inferencer
    inf = Inferencer()
    with pytest.raises(RuntimeError):
        inf.predict_frame(np.zeros((64, 64, 3), dtype=np.uint8))


def test_live_page_constructs(qtbot):
    from gui.pages.live_classification_page import LiveClassificationPage
    page = LiveClassificationPage()
    qtbot.addWidget(page)
    assert page._inferencer is not None
    assert page._camera_thread is None


def test_capture_without_project_warns(qtbot, monkeypatch):
    from gui.pages.live_classification_page import LiveClassificationPage
    from PySide6.QtWidgets import QMessageBox
    page = LiveClassificationPage()
    qtbot.addWidget(page)
    calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: calls.append(a))
    page._capture_to_project()           # no project set
    assert calls, "expected a warning when capturing without a project"


def test_capture_without_frame_warns(qtbot, monkeypatch, tmp_path):
    from gui.pages.live_classification_page import LiveClassificationPage
    from PySide6.QtWidgets import QMessageBox

    class _Cfg:
        image_dir = str(tmp_path)

    class _Proj:
        config = _Cfg()
        project_path = str(tmp_path / "p.json")

    page = LiveClassificationPage()
    qtbot.addWidget(page)
    page.set_project(_Proj())
    calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: calls.append(a))
    page._capture_to_project()           # project set but no frame captured yet
    assert calls, "expected a warning when no frame is available"


def test_overlay_no_result_returns_frame(qtbot):
    from gui.pages.live_classification_page import LiveClassificationPage
    page = LiveClassificationPage()
    qtbot.addWidget(page)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    out = page._overlay(frame, None)
    assert out.shape == frame.shape


def test_roi_disabled_returns_none(qtbot):
    from gui.pages.live_classification_page import LiveClassificationPage
    page = LiveClassificationPage()
    qtbot.addWidget(page)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert page._roi_px(frame) is None


def test_roi_enabled_computes_pixels(qtbot):
    from gui.pages.live_classification_page import LiveClassificationPage
    page = LiveClassificationPage()
    qtbot.addWidget(page)
    page._roi_cb.setChecked(True)  # defaults 10/10/90/90
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert page._roi_px(frame) == (20, 10, 180, 90)


def test_confidence_threshold(qtbot):
    from gui.pages.live_classification_page import LiveClassificationPage
    page = LiveClassificationPage()
    qtbot.addWidget(page)
    page._minconf_spin.setValue(50)
    assert page._confident({"confidence": 0.9}) is True
    assert page._confident({"confidence": 0.3}) is False
    assert page._confident(None) is False
