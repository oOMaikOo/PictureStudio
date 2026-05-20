"""
Tests für _RemoteTrainDialog und Hilfsklassen in gui/pages/fleet_page.py.

Alle Tests sind leichtgewichtig (kein Netzwerk, kein PyTorch-Training).
"""
from __future__ import annotations

import pytest

from gui.pages.fleet_page import (
    FleetPage,
    _RemoteTrainDialog,
    _FrameCollectThread,
    _LocalTrainThread,
)


# ── Fixture: dummy device dict ────────────────────────────────────────────────

@pytest.fixture
def dummy_device() -> dict:
    return {
        "name": "Test-Gerät",
        "url": "http://localhost:9999",
        "api_key": "",
    }


# ── 1. _RemoteTrainDialog instantiates without error ─────────────────────────

def test_remote_train_dialog_instantiates(qtbot, dummy_device):
    """Dialog must open without raising exceptions."""
    dlg = _RemoteTrainDialog(dummy_device)
    qtbot.addWidget(dlg)
    assert dlg is not None


# ── 2. Dialog has three tabs ──────────────────────────────────────────────────

def test_remote_train_dialog_has_three_tabs(qtbot, dummy_device):
    """The dialog's QTabWidget must contain exactly three tabs."""
    from PySide6.QtWidgets import QTabWidget
    dlg = _RemoteTrainDialog(dummy_device)
    qtbot.addWidget(dlg)
    tab_widgets = dlg.findChildren(QTabWidget)
    assert tab_widgets, "Kein QTabWidget gefunden"
    tab = tab_widgets[0]
    assert tab.count() == 3, f"Erwartet 3 Tabs, gefunden {tab.count()}"
    labels = [tab.tabText(i) for i in range(tab.count())]
    # At least one tab should relate to channels and one to deploy
    assert any("Kan" in lbl for lbl in labels), f"Kein Kanal-Tab: {labels}"
    assert any("Deploy" in lbl for lbl in labels), f"Kein Deploy-Tab: {labels}"


# ── 3. _FrameCollectThread instantiates ──────────────────────────────────────

def test_frame_collect_thread_instantiates():
    """_FrameCollectThread must be creatable without starting it."""
    from unittest.mock import MagicMock
    detector_mock = MagicMock()
    thread = _FrameCollectThread(
        device_url="http://localhost:9999",
        channel_id=0,
        detector=detector_mock,
    )
    assert thread is not None
    # Make sure the thread is not running
    assert not thread.isRunning()


# ── 4. _LocalTrainThread instantiates ────────────────────────────────────────

def test_local_train_thread_instantiates():
    """_LocalTrainThread must be creatable without starting it."""
    from unittest.mock import MagicMock
    detector_mock = MagicMock()
    thread = _LocalTrainThread(detector=detector_mock, epochs=10)
    assert thread is not None
    assert not thread.isRunning()


# ── 5. FleetPage has a "Training" button after rebuild ───────────────────────

def test_fleet_page_has_training_button(qtbot):
    """After _rebuild_table() with a dummy device, a 'Training' button must exist."""
    from PySide6.QtWidgets import QAbstractButton
    page = FleetPage()
    qtbot.addWidget(page)

    # Inject a dummy device and rebuild
    page._devices = [
        {"name": "Kamera Test", "url": "http://localhost:9999", "api_key": ""}
    ]
    page._rebuild_table()

    buttons = page.findChildren(QAbstractButton)
    btn_texts = [b.text() for b in buttons]
    training_btns = [t for t in btn_texts if "Training" in t]
    assert training_btns, (
        f"Kein Button mit 'Training' gefunden. Verfügbare Buttons: {btn_texts}"
    )
