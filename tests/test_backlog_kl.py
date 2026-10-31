"""Tests for BACKLOG K (confidence histogram) and L (multi-cam alarm export)."""
import csv
import os
from unittest.mock import patch


# ── K: confidence histogram ───────────────────────────────────────────────────

def test_confidence_histogram_bins_empty():
    from gui.pages.batch_inference_page import confidence_histogram_bins

    assert confidence_histogram_bins([]) == [0] * 10


def test_confidence_histogram_bins_ten_percent_buckets():
    from gui.pages.batch_inference_page import confidence_histogram_bins

    confs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    assert confidence_histogram_bins(confs) == [1] * 10


def test_confidence_histogram_bins_clamps_one():
    from gui.pages.batch_inference_page import confidence_histogram_bins

    # 1.0 must land in the last (90–100%) bucket, not overflow
    bins = confidence_histogram_bins([1.0, 0.0, 0.99])
    assert bins[0] == 1
    assert bins[9] == 2
    assert sum(bins) == 3


def test_confidence_histogram_widget_on_page(qtbot):
    from gui.pages.batch_inference_page import BatchInferencePage

    page = BatchInferencePage()
    qtbot.addWidget(page)
    assert hasattr(page, "_conf_hist")
    page._conf_hist.set_bins([0, 0, 1, 2, 3, 2, 1, 0, 0, 5])
    assert page._conf_hist._bins[9] == 5


def test_show_results_updates_histogram(qtbot):
    from gui.pages.batch_inference_page import BatchInferencePage

    page = BatchInferencePage()
    qtbot.addWidget(page)
    page._results = [
        {"filename": "a.jpg", "path": "/a.jpg", "predicted": "ok",
         "confidence": 0.92, "error": None},
        {"filename": "b.jpg", "path": "/b.jpg", "predicted": "ok",
         "confidence": 0.41, "error": None},
        {"filename": "c.jpg", "path": "/c.jpg", "predicted": "?",
         "confidence": 0.0, "error": "boom"},
    ]
    page._show_results(page._results)
    # errors excluded; 0.92 → bin 9, 0.41 → bin 4
    assert page._conf_hist._bins[9] == 1
    assert page._conf_hist._bins[4] == 1
    assert sum(page._conf_hist._bins) == 2


# ── L: multi-cam alarm protocol ───────────────────────────────────────────────

def test_multi_cam_export_button_exists(qtbot):
    from gui.pages.multi_camera_page import MultiCameraPage

    page = MultiCameraPage()
    qtbot.addWidget(page)
    assert hasattr(page, "_export_log_btn")
    assert not page._export_log_btn.isEnabled()
    assert page._export_log_btn.text()  # translated label present


def test_write_alarm_event_creates_csv(qtbot, tmp_path):
    from gui.pages.multi_camera_page import MultiCameraPage

    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._alarm_output_dir = str(tmp_path)
    page._alarm_log_path = str(tmp_path / "multi_cam_events.csv")

    page._write_alarm_event(0, 0.87, 0.5, "mc_ch1_test.jpg")
    page._write_alarm_event(1, 0.91, 0.5, "mc_ch2_test.jpg")

    assert len(page._alarm_events) == 2
    assert os.path.isfile(page._alarm_log_path)
    with open(page._alarm_log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["channel"] == "1"
    assert rows[1]["channel"] == "2"
    assert rows[0]["frame_file"] == "mc_ch1_test.jpg"


def test_export_alarm_log_copies_csv(qtbot, tmp_path):
    from gui.pages.multi_camera_page import MultiCameraPage

    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._alarm_output_dir = str(tmp_path / "src")
    os.makedirs(page._alarm_output_dir, exist_ok=True)
    page._alarm_log_path = str(tmp_path / "src" / "multi_cam_events.csv")
    page._write_alarm_event(0, 0.8, 0.4, "frame.jpg")

    dest = str(tmp_path / "exported.csv")
    with patch(
        "gui.pages.multi_camera_page.QFileDialog.getSaveFileName",
        return_value=(dest, "CSV (*.csv)"),
    ), patch("gui.pages.multi_camera_page.QMessageBox.information"):
        page._export_alarm_log()

    assert os.path.isfile(dest)
    with open(dest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["score"] == "0.800000"


def test_export_alarm_log_empty_shows_info(qtbot):
    from gui.pages.multi_camera_page import MultiCameraPage

    page = MultiCameraPage()
    qtbot.addWidget(page)
    page._alarm_events = []
    page._alarm_log_path = "/nonexistent/path.csv"

    with patch("gui.pages.multi_camera_page.QMessageBox.information") as info:
        page._export_alarm_log()
    info.assert_called_once()
