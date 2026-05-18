"""
Unit tests for monitor.py — standalone anomaly monitor client.

Tests cover: build_parser() defaults and required-argument enforcement,
find_camera_index() fallback behaviour, apply_roi() cropping logic,
draw_hud() and composite_overlay() output shapes, and save_alarm()
file/CSV writing.

Camera hardware and the main run_monitor loop are NOT exercised; only
the pure helper functions are tested.
"""
import csv
import os
import sys

import numpy as np
import pytest

# Ensure project root is on the path so monitor.py can be imported directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_frame(h: int = 240, w: int = 320, color=(100, 150, 200)) -> np.ndarray:
    """Create a solid BGR uint8 frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = color[0]
    frame[:, :, 1] = color[1]
    frame[:, :, 2] = color[2]
    return frame


def _read_csv_rows(path: str):
    """Return all data rows (excluding header) from a CSV file."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_model_is_required_exits_with_error_if_missing(self):
        from monitor import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_model_argument_accepted(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "path/to/model.pt"])
        assert args.model == "path/to/model.pt"

    def test_camera_accepts_integer(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt", "--camera", "2"])
        assert args.camera == 2

    def test_threshold_accepts_float(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt", "--threshold", "0.0035"])
        assert abs(args.threshold - 0.0035) < 1e-9

    def test_headless_is_boolean_flag(self):
        from monitor import build_parser
        parser = build_parser()
        args_off = parser.parse_args(["--model", "m.pt"])
        args_on  = parser.parse_args(["--model", "m.pt", "--headless"])
        assert args_off.headless is False
        assert args_on.headless is True

    def test_default_fps_is_15(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt"])
        assert abs(args.fps - 15.0) < 1e-9

    def test_default_cooldown_is_10(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt"])
        assert abs(args.cooldown - 10.0) < 1e-9

    def test_default_output_is_monitor_logs(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt"])
        assert args.output == "monitor_logs"

    def test_default_headless_is_false(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt"])
        assert args.headless is False

    def test_default_camera_is_none(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt"])
        assert args.camera is None

    def test_default_threshold_is_none(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt"])
        assert args.threshold is None

    def test_fps_accepts_float(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--model", "m.pt", "--fps", "7.5"])
        assert abs(args.fps - 7.5) < 1e-9


# ---------------------------------------------------------------------------
# find_camera_index
# ---------------------------------------------------------------------------

class TestFindCameraIndex:
    def test_returns_zero_for_none(self):
        from monitor import find_camera_index
        assert find_camera_index(None) == 0

    def test_returns_zero_for_empty_string(self):
        from monitor import find_camera_index
        assert find_camera_index("") == 0

    def test_returns_zero_for_unrecognised_source(self):
        """When the named camera is not found, must fall back to 0."""
        from monitor import find_camera_index
        # A camera name that certainly doesn't exist
        result = find_camera_index("NonExistentCam_XYZ_99")
        assert result == 0


# ---------------------------------------------------------------------------
# apply_roi
# ---------------------------------------------------------------------------

class TestApplyRoi:
    def test_none_roi_returns_frame_unchanged(self):
        from monitor import apply_roi
        frame = _bgr_frame()
        result = apply_roi(frame, None)
        assert result is frame

    def test_empty_roi_list_returns_frame_unchanged(self):
        from monitor import apply_roi
        frame = _bgr_frame()
        result = apply_roi(frame, [])
        assert result is frame

    def test_full_frame_roi_returns_full_size(self):
        from monitor import apply_roi
        frame = _bgr_frame(240, 320)
        result = apply_roi(frame, [0.0, 0.0, 1.0, 1.0])
        assert result.shape == (240, 320, 3)

    def test_center_50_percent_roi_crops_correctly(self):
        from monitor import apply_roi
        frame = _bgr_frame(240, 320)
        # roi [0.25, 0.25, 0.75, 0.75] → crop to centre 50%
        result = apply_roi(frame, [0.25, 0.25, 0.75, 0.75])
        # Expected: x1=80, x2=240, y1=60, y2=180 → shape (120, 160, 3)
        assert result.shape == (120, 160, 3)

    def test_roi_resulting_in_tiny_region_returns_frame_unchanged(self):
        """ROI that produces < 4 px width or height returns original frame."""
        from monitor import apply_roi
        frame = _bgr_frame(240, 320)
        # Width: (0.505 - 0.500) * 320 = 1.6 px → less than 4
        result = apply_roi(frame, [0.500, 0.0, 0.505, 1.0])
        assert result is frame

    def test_roi_that_is_too_thin_in_height_returns_frame_unchanged(self):
        from monitor import apply_roi
        frame = _bgr_frame(240, 320)
        # Height: (0.502 - 0.5) * 240 = 0.48 px → less than 4
        result = apply_roi(frame, [0.0, 0.500, 1.0, 0.502])
        assert result is frame

    def test_apply_roi_preserves_pixel_data(self):
        """Cropped pixels must match the original frame at the same coordinates."""
        from monitor import apply_roi
        frame = np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3)
        roi = [0.25, 0.25, 0.75, 0.75]
        result = apply_roi(frame, roi)
        # y1=60, y2=180, x1=80, x2=240
        expected = frame[60:180, 80:240]
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# draw_hud
# ---------------------------------------------------------------------------

class TestDrawHud:
    def test_returns_ndarray(self):
        from monitor import draw_hud
        frame = _bgr_frame()
        out = draw_hud(frame, 0.05, 0.10, False, 0, None)
        assert isinstance(out, np.ndarray)

    def test_output_shape_matches_input(self):
        from monitor import draw_hud
        frame = _bgr_frame(480, 640)
        out = draw_hud(frame, 0.05, 0.10, False, 0, None)
        assert out.shape == frame.shape

    def test_does_not_modify_input_frame(self):
        from monitor import draw_hud
        frame = _bgr_frame()
        original = frame.copy()
        draw_hud(frame, 0.05, 0.10, False, 0, None)
        np.testing.assert_array_equal(frame, original)

    def test_is_anomaly_true_does_not_crash(self):
        from monitor import draw_hud
        frame = _bgr_frame()
        out = draw_hud(frame, 0.95, 0.10, True, 5, None)
        assert out.shape == frame.shape

    def test_roi_not_none_does_not_crash(self):
        from monitor import draw_hud
        frame = _bgr_frame(240, 320)
        out = draw_hud(frame, 0.05, 0.10, False, 0, [0.1, 0.1, 0.9, 0.9])
        assert out.shape == frame.shape

    def test_roi_none_does_not_crash(self):
        from monitor import draw_hud
        frame = _bgr_frame()
        out = draw_hud(frame, 0.05, 0.10, False, 0, None)
        assert out.shape == frame.shape

    def test_zero_threshold_does_not_crash(self):
        from monitor import draw_hud
        frame = _bgr_frame()
        out = draw_hud(frame, 0.0, 0.0, False, 0, None)
        assert out.shape == frame.shape


# ---------------------------------------------------------------------------
# composite_overlay
# ---------------------------------------------------------------------------

class TestCompositeOverlay:
    def test_returns_ndarray(self):
        from monitor import composite_overlay
        full = _bgr_frame(240, 320)
        overlay = _bgr_frame(120, 160)
        roi = [0.25, 0.25, 0.75, 0.75]
        out = composite_overlay(full, overlay, roi)
        assert isinstance(out, np.ndarray)

    def test_output_same_size_as_full_frame(self):
        from monitor import composite_overlay
        full = _bgr_frame(240, 320)
        overlay = _bgr_frame(120, 160)
        roi = [0.25, 0.25, 0.75, 0.75]
        out = composite_overlay(full, overlay, roi)
        assert out.shape == full.shape

    def test_does_not_modify_full_frame(self):
        from monitor import composite_overlay
        full = _bgr_frame(240, 320)
        original = full.copy()
        overlay = _bgr_frame(60, 80)
        composite_overlay(full, overlay, [0.0, 0.0, 0.25, 0.25])
        np.testing.assert_array_equal(full, original)

    def test_full_roi_pastes_overlay_across_whole_frame(self):
        from monitor import composite_overlay
        full = _bgr_frame(240, 320, color=(0, 0, 0))
        overlay = _bgr_frame(240, 320, color=(255, 255, 255))
        out = composite_overlay(full, overlay, [0.0, 0.0, 1.0, 1.0])
        # All pixels should be white (or very close, due to resize)
        assert out.mean() > 200


# ---------------------------------------------------------------------------
# save_alarm
# ---------------------------------------------------------------------------

class TestSaveAlarm:
    def test_save_alarm_writes_jpeg_file(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        fname = save_alarm(frame, 0.95, 0.5, out_dir, log_path, 1)
        assert fname != ""
        assert os.path.isfile(os.path.join(out_dir, fname))

    def test_save_alarm_returns_filename_string(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        result = save_alarm(frame, 0.95, 0.5, out_dir, log_path, 1)
        assert isinstance(result, str)
        assert result.endswith(".jpg")

    def test_save_alarm_creates_csv_with_header_on_first_call(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        assert not os.path.exists(log_path)
        save_alarm(frame, 0.95, 0.5, out_dir, log_path, 1)
        assert os.path.isfile(log_path)

        with open(log_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "timestamp_utc" in header
        assert "score" in header

    def test_save_alarm_writes_one_data_row(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        save_alarm(frame, 0.95, 0.5, out_dir, log_path, 1)
        rows = _read_csv_rows(log_path)
        assert len(rows) == 1

    def test_save_alarm_second_call_appends_no_second_header(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        save_alarm(frame, 0.95, 0.5, out_dir, log_path, 1)
        save_alarm(frame, 0.85, 0.5, out_dir, log_path, 2)

        # DictReader uses first line as header; there must be exactly 2 data rows
        rows = _read_csv_rows(log_path)
        assert len(rows) == 2

    def test_save_alarm_score_stored_in_csv(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        save_alarm(frame, 0.123456, 0.5, out_dir, log_path, 1)
        rows = _read_csv_rows(log_path)
        assert abs(float(rows[0]["score"]) - 0.123456) < 1e-4

    def test_save_alarm_score_pct_computed_correctly(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        score, threshold = 0.15, 0.10
        expected_pct = int(score / threshold * 100)   # matches implementation exactly
        save_alarm(frame, score, threshold, out_dir, log_path, 1)
        rows = _read_csv_rows(log_path)
        assert int(rows[0]["score_pct"]) == expected_pct

    def test_save_alarm_filename_in_csv_row(self, tmp_path):
        from monitor import save_alarm
        out_dir = str(tmp_path / "alarms")
        os.makedirs(out_dir, exist_ok=True)
        log_path = str(tmp_path / "events.csv")
        frame = _bgr_frame()

        fname = save_alarm(frame, 0.95, 0.5, out_dir, log_path, 1)
        rows = _read_csv_rows(log_path)
        assert rows[0]["frame_file"] == fname
