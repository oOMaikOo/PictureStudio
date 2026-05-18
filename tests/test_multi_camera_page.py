"""
Tests for gui/pages/multi_camera_page.py (Feature 4: Multi-Kamera-Monitoring).

The module is imported via pytest.importorskip so these tests are skipped
gracefully until the parallel agent finishes creating the file.
"""
import pytest

multi_cam = pytest.importorskip(
    "gui.pages.multi_camera_page",
    reason="multi_camera_page not yet available",
)

MultiCameraPage = multi_cam.MultiCameraPage
_ConfigDialog   = multi_cam._ConfigDialog
_ChannelWidget  = multi_cam._ChannelWidget


# ---------------------------------------------------------------------------
# MultiCameraPage
# ---------------------------------------------------------------------------

def test_multi_camera_page_instantiates(qtbot):
    """MultiCameraPage should construct without raising."""
    page = MultiCameraPage()
    qtbot.addWidget(page)


def test_multi_camera_page_has_four_channels(qtbot):
    """MultiCameraPage should expose exactly 4 channel widgets after init."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    channels = page.findChildren(_ChannelWidget)
    assert len(channels) == 4


def test_set_notifier_none_does_not_crash(qtbot):
    """set_notifier(None) must not raise."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page.set_notifier(None)


def test_set_rest_server_none_does_not_crash(qtbot):
    """set_rest_server(None) must not raise."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page.set_rest_server(None)


def test_alle_starten_button_exists(qtbot):
    """Page must contain an 'Alle starten' button."""
    from PySide6.QtWidgets import QAbstractButton
    page = MultiCameraPage()
    qtbot.addWidget(page)
    buttons = [b for b in page.findChildren(QAbstractButton)
               if "Alle starten" in b.text()]
    assert buttons, "No 'Alle starten' button found"


def test_alle_stoppen_button_exists(qtbot):
    """Page must contain an 'Alle stoppen' button."""
    from PySide6.QtWidgets import QAbstractButton
    page = MultiCameraPage()
    qtbot.addWidget(page)
    buttons = [b for b in page.findChildren(QAbstractButton)
               if "Alle stoppen" in b.text()]
    assert buttons, "No 'Alle stoppen' button found"


# ---------------------------------------------------------------------------
# _ConfigDialog
# ---------------------------------------------------------------------------

def test_config_dialog_instantiates(qtbot):
    """_ConfigDialog should construct without raising."""
    dlg = _ConfigDialog(0, 0, "", [])
    qtbot.addWidget(dlg)


def test_config_dialog_selected_camera_idx_returns_int(qtbot):
    """_ConfigDialog.selected_camera_idx must return an int."""
    dlg = _ConfigDialog(0, 0, "", [])
    qtbot.addWidget(dlg)
    assert isinstance(dlg.selected_camera_idx, int)


def test_config_dialog_selected_model_path_returns_str(qtbot):
    """_ConfigDialog.selected_model_path must return a str."""
    dlg = _ConfigDialog(0, 0, "", [])
    qtbot.addWidget(dlg)
    assert isinstance(dlg.selected_model_path, str)


# ---------------------------------------------------------------------------
# _ChannelWidget
# ---------------------------------------------------------------------------

def test_channel_widget_instantiates(qtbot):
    """_ChannelWidget should construct with channel_idx=0 without raising."""
    widget = _ChannelWidget(channel_idx=0)
    qtbot.addWidget(widget)


def test_channel_widget_set_status_does_not_crash(qtbot):
    """_ChannelWidget.set_status() must not raise."""
    widget = _ChannelWidget(channel_idx=0)
    qtbot.addWidget(widget)
    widget.set_status("OK", "#27AE60")


def test_channel_widget_set_running_true_does_not_crash(qtbot):
    """_ChannelWidget.set_running(True) must not raise."""
    widget = _ChannelWidget(channel_idx=0)
    qtbot.addWidget(widget)
    widget.set_running(True)


def test_channel_widget_set_running_false_does_not_crash(qtbot):
    """_ChannelWidget.set_running(False) must not raise."""
    widget = _ChannelWidget(channel_idx=0)
    qtbot.addWidget(widget)
    widget.set_running(False)
