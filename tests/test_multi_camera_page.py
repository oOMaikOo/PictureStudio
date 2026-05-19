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


def test_multi_camera_page_has_two_channels_by_default(qtbot):
    """MultiCameraPage should default to 2 channel widgets."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    assert len(page._widgets) == 2


def test_multi_camera_page_has_channel_count_spinbox(qtbot):
    """MultiCameraPage should have a QSpinBox for choosing camera count."""
    from PySide6.QtWidgets import QSpinBox
    page = MultiCameraPage()
    qtbot.addWidget(page)
    spins = page.findChildren(QSpinBox)
    assert spins, "No QSpinBox found for camera count"
    assert spins[0].value() == 2


def test_apply_channel_count_grows_channels(qtbot):
    """_apply_channel_count() should add channels when count increases."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(4)
    assert len(page._channels) == 4
    assert len(page._widgets) == 4


def test_apply_channel_count_shrinks_channels(qtbot):
    """_apply_channel_count() should remove channels when count decreases."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(4)
    page._apply_channel_count(1)
    assert len(page._channels) == 1
    assert len(page._widgets) == 1


def test_apply_channel_count_preserves_max_9(qtbot):
    """Camera count can go up to 9."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(9)
    assert len(page._channels) == 9
    assert len(page._widgets) == 9


def test_page_nav_hidden_for_four_or_fewer_channels(qtbot):
    """Page navigation should be hidden when channel count <= 4."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(4)
    assert page._page_nav.isHidden()


def test_page_nav_visible_for_five_channels(qtbot):
    """Page navigation should not be hidden when channel count > 4."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(5)
    assert not page._page_nav.isHidden()


def test_page_prev_button_disabled_on_first_page(qtbot):
    """Prev button is disabled on the first page."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(9)
    page._current_page = 0
    page._refresh_page_view()
    assert not page._prev_btn.isEnabled()


def test_page_next_button_enabled_when_more_pages_exist(qtbot):
    """Next button is enabled when there are more pages."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(9)   # 3 pages of 4
    assert page._next_btn.isEnabled()


def test_page_navigation_advances_page(qtbot):
    """_on_page_next() increments current page."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(9)
    assert page._current_page == 0
    page._on_page_next()
    assert page._current_page == 1


def test_page_navigation_retreats_page(qtbot):
    """_on_page_prev() decrements current page."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(9)
    page._on_page_next()
    page._on_page_prev()
    assert page._current_page == 0


def test_page_label_shows_correct_page_info(qtbot):
    """Page label text matches current page / total pages."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(9)   # 3 pages
    assert "1 / 3" in page._page_label.text()
    page._on_page_next()
    assert "2 / 3" in page._page_label.text()


def test_grid_shows_only_current_page_widgets(qtbot):
    """Grid layout should contain at most 4 widgets at once."""
    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._apply_channel_count(9)
    count_in_grid = page._grid_layout.count()
    assert count_in_grid <= 4


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
