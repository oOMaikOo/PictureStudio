"""
Test scenarios for the product reshape (update banner, review hub,
Studio/Monitor mode, Advanced fold, anomaly train vs live, Windows smoke helpers).
"""
import io
import json
from unittest.mock import MagicMock, patch

import pytest


# ── U1: updater / What's new ─────────────────────────────────────────────────

def test_parse_and_compare_versions():
    from utils.updater import parse_version, is_newer
    assert parse_version("v2.5.3") == (2, 5, 3)
    assert is_newer("2.6.0", "2.5.2")
    assert not is_newer("2.5.2", "2.5.2")
    assert not is_newer("2.5.1", "2.5.2")


def test_changelog_whats_new_prefers_unreleased():
    from utils.updater import changelog_whats_new
    text = (
        "## Unreleased\n\n- Feature A\n\n"
        "## 2.5.2\n\n- Old stuff\n"
    )
    out = changelog_whats_new(text, "2.5.2")
    assert "Feature A" in out
    assert "Old stuff" not in out


def test_fetch_latest_release_tag_mocked():
    from utils.updater import fetch_latest_release_tag

    payload = json.dumps({"tag_name": "v2.9.0"}).encode()

    def fake_open(req, timeout=5):
        return io.BytesIO(payload)

    tag = fetch_latest_release_tag(opener=fake_open)
    assert tag == "v2.9.0"


def test_update_banner_shows(qtbot):
    from gui.widgets.update_banner import UpdateBanner
    b = UpdateBanner()
    qtbot.addWidget(b)
    b.show_update("2.5.2", "v2.6.0", "hello")
    assert not b.isHidden()
    assert "2.6.0" in b._msg.text()


# ── U2: review queue ─────────────────────────────────────────────────────────

def test_review_page_lists_uncertain_and_al(qtbot, sample_project):
    from gui.pages.review_queue_page import ReviewQueuePage

    page = ReviewQueuePage()
    qtbot.addWidget(page)
    p = sample_project
    path = p.images[0]
    p.set_label_flag(path, uncertain=True, comment="check me")
    p.add_to_al_queue(p.images[1], "defect", 0.42)
    page.set_project(p)
    assert page._uncertain_list.count() == 1
    assert page._al_list.count() == 1

    page.set_batch_low_confidence([
        {"path": p.images[2], "confidence": 0.3, "predicted": "ok"},
    ])
    assert page._batch_list.count() == 1


# ── D1: product line settings + sidebar ──────────────────────────────────────

def test_product_line_settings_roundtrip():
    from utils.settings import AppSettings
    s = AppSettings()
    s.set_product_line("monitor")
    assert s.get_product_line() == "monitor"
    s.set_product_line("studio")
    assert s.get_product_line() == "studio"
    s.mark_product_line_chosen()
    assert s.is_product_line_chosen()


def test_sidebar_studio_vs_monitor(qtbot):
    from gui.sidebar import Sidebar
    from gui.page_index import Page

    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.set_ui_mode("expert")
    sb.set_product_line("studio")
    idxs = {i for _, i in sb._buttons}
    assert Page.LABELING in idxs
    assert Page.CAMERA not in idxs

    sb.set_product_line("monitor")
    idxs = {i for _, i in sb._buttons}
    assert Page.CAMERA in idxs
    assert Page.ANOMALY_TRAINING in idxs
    assert Page.LABELING not in idxs


# ── D2: advanced fold ────────────────────────────────────────────────────────

def test_advanced_hides_edge_and_ssh(qtbot):
    from gui.pages.models_page import ModelsPage
    from gui.pages.training_page import TrainingPage

    mp = ModelsPage()
    qtbot.addWidget(mp)
    assert mp._advanced_panel.isHidden()
    mp.set_advanced_visible(True)
    assert not mp._advanced_panel.isHidden()

    tp = TrainingPage()
    qtbot.addWidget(tp)
    assert tp.ssh_box.isHidden()
    assert tp.hpt_btn.isHidden()
    tp.set_advanced_visible(True)
    assert not tp.ssh_box.isHidden()
    assert not tp.hpt_btn.isHidden()


# ── D3: camera navigates to anomaly training ─────────────────────────────────

def test_camera_goto_anomaly_training(qtbot):
    from gui.pages.camera_page import CameraPage
    from gui.page_index import Page

    page = CameraPage()
    qtbot.addWidget(page)
    mw = MagicMock()
    # window() returns the Qt top-level; patch method on instance
    page.window = MagicMock(return_value=mw)
    page._goto_anomaly_training()
    mw._switch_page.assert_called_once_with(Page.ANOMALY_TRAINING)


# ── U3: camera smoke (no real device) ────────────────────────────────────────

def test_camera_list_smoke_mocked():
    with patch("core.camera.list_usb_cameras", return_value=[(0, "FakeCam")]):
        from core.camera import list_usb_cameras
        cams = list_usb_cameras()
    assert cams == [(0, "FakeCam")]


def test_windows_installer_script_exists():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "build_windows.bat").is_file()
    assert (root / "installer" / "windows" / "PictureStudio.iss").is_file()


def test_main_window_has_review_and_banner(qtbot):
    from utils.settings import AppSettings
    AppSettings().mark_product_line_chosen()
    from gui.main_window import MainWindow
    from gui.page_index import Page

    win = MainWindow()
    qtbot.addWidget(win)
    assert type(win.stack.widget(Page.REVIEW)).__name__ == "ReviewQueuePage"
    assert hasattr(win, "_update_banner")
