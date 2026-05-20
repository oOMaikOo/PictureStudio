"""Tests für Kamera-Einstellungen und Vorverarbeitungsfilter."""
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch

# ── apply_cam_props ──────────────────────────────────────────────────────────

def test_apply_cam_props_calls_set_for_known_props():
    from core.camera import apply_cam_props
    cap = MagicMock()
    apply_cam_props(cap, {"brightness": 10, "contrast": 50})
    assert cap.set.call_count == 2

def test_apply_cam_props_ignores_unknown_props():
    from core.camera import apply_cam_props
    cap = MagicMock()
    apply_cam_props(cap, {"unknown_prop": 99})
    cap.set.assert_not_called()

# ── apply_frame_filter ───────────────────────────────────────────────────────

@pytest.fixture
def bgr_frame():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

def test_filter_none_returns_original(bgr_frame):
    from core.camera import apply_frame_filter
    result = apply_frame_filter(bgr_frame, "none")
    assert np.array_equal(result, bgr_frame)

def test_filter_grayscale_returns_bgr(bgr_frame):
    from core.camera import apply_frame_filter
    result = apply_frame_filter(bgr_frame, "grayscale")
    assert result.shape == bgr_frame.shape

def test_filter_canny_returns_bgr(bgr_frame):
    from core.camera import apply_frame_filter
    result = apply_frame_filter(bgr_frame, "canny")
    assert result.shape == bgr_frame.shape

def test_filter_sobel_returns_bgr(bgr_frame):
    from core.camera import apply_frame_filter
    result = apply_frame_filter(bgr_frame, "sobel")
    assert result.shape == bgr_frame.shape

def test_filter_laplacian_returns_bgr(bgr_frame):
    from core.camera import apply_frame_filter
    result = apply_frame_filter(bgr_frame, "laplacian")
    assert result.shape == bgr_frame.shape

def test_filter_unknown_returns_original(bgr_frame):
    from core.camera import apply_frame_filter
    result = apply_frame_filter(bgr_frame, "does_not_exist")
    assert np.array_equal(result, bgr_frame)

# ── CameraFrameThread.set_cam_props ─────────────────────────────────────────

def test_set_cam_props_updates_pending():
    from core.camera import CameraFrameThread
    t = CameraFrameThread(source=0, fps=15.0)
    t.set_cam_props({"brightness": 20})
    with t._props_lock:
        assert t._pending_props.get("brightness") == 20

# ── CameraPage UI ────────────────────────────────────────────────────────────

def test_camera_page_filter_combo_has_five_options(qtbot):
    from gui.pages.camera_page import CameraPage
    page = CameraPage()
    qtbot.addWidget(page)
    assert page._filter_combo.count() == 5

def test_camera_page_cam_settings_group_exists(qtbot):
    from gui.pages.camera_page import CameraPage
    page = CameraPage()
    qtbot.addWidget(page)
    assert hasattr(page, "_cam_settings_grp")
    assert hasattr(page, "_brightness_sl")
    assert hasattr(page, "_contrast_sl")
