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
    assert page._cam_settings.filter_combo.count() == 5

def test_camera_page_cam_settings_group_exists(qtbot):
    from gui.pages.camera_page import CameraPage
    page = CameraPage()
    qtbot.addWidget(page)
    assert page._cam_settings.settings_group is not None
    assert set(page._cam_settings.cam_props()) == {
        "brightness", "contrast", "saturation", "sharpness", "exposure"}


# ── CameraSettingsGroup (shared by CameraPage and CameraCaptureDialog) ───────

def test_settings_group_starts_from_given_props(qtbot):
    from gui.widgets.camera_settings_group import CameraSettingsGroup
    w = CameraSettingsGroup(cam_props={"brightness": 12}, filter_name="canny")
    qtbot.addWidget(w)
    assert w.cam_props()["brightness"] == 12
    assert w.cam_props()["contrast"] == 0        # unspecified → default
    assert w.filter_name() == "canny"

def test_settings_group_unknown_filter_falls_back_to_none(qtbot):
    from gui.widgets.camera_settings_group import CameraSettingsGroup
    w = CameraSettingsGroup(filter_name="does_not_exist")
    qtbot.addWidget(w)
    assert w.filter_name() == "none"

def test_settings_group_slider_emits_prop_changed(qtbot):
    from gui.widgets.camera_settings_group import CameraSettingsGroup
    w = CameraSettingsGroup()
    qtbot.addWidget(w)
    seen = []
    w.prop_changed.connect(lambda p, v: seen.append((p, v)))
    w._sliders["contrast"].setValue(42)
    assert seen == [("contrast", 42)]

def test_settings_group_reset_restores_defaults_once(qtbot):
    """Reset emits a single props_reset, not one signal per slider."""
    from gui.widgets.camera_settings_group import CameraSettingsGroup
    w = CameraSettingsGroup(cam_props={"brightness": 30, "exposure": -2})
    qtbot.addWidget(w)
    per_prop, resets = [], []
    w.prop_changed.connect(lambda p, v: per_prop.append(p))
    w.props_reset.connect(resets.append)
    returned = w.reset()
    assert returned == w.defaults() == w.cam_props()
    assert returned["brightness"] == 0 and returned["exposure"] == -6
    assert resets == [returned]
    assert per_prop == []

def test_settings_group_reset_updates_value_labels(qtbot):
    """The number beside each slider must follow a reset, not go stale."""
    from gui.widgets.camera_settings_group import CameraSettingsGroup
    w = CameraSettingsGroup(cam_props={"brightness": 30})
    qtbot.addWidget(w)
    assert w._value_labels["brightness"].text() == "30"
    w.reset()
    assert w._value_labels["brightness"].text() == "0"

def test_settings_group_filter_change_emits_key(qtbot):
    from gui.widgets.camera_settings_group import CameraSettingsGroup
    w = CameraSettingsGroup()
    qtbot.addWidget(w)
    seen = []
    w.filter_changed.connect(seen.append)
    w.filter_combo.setCurrentIndex(w.filter_combo.findData("sobel"))
    assert seen == ["sobel"]
