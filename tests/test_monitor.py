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
    def test_model_is_optional_in_parser(self):
        # --model is now optional at argparse level; main() enforces it when --setup absent
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.model is None

    def test_setup_flag_makes_model_unnecessary(self):
        from monitor import build_parser
        parser = build_parser()
        args = parser.parse_args(["--setup"])
        assert args.setup is True
        assert args.model is None

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


# ---------------------------------------------------------------------------
# _CameraThread — video detection and construction (no hardware required)
# ---------------------------------------------------------------------------

class TestCameraThreadVideoDetection:
    """Tests for _CameraThread._detect_video() — pure logic, no camera access."""

    def test_integer_source_is_not_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video(0) is False
        assert _CameraThread._detect_video(2) is False

    def test_mp4_extension_detected_as_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("/path/to/video.mp4") is True

    def test_avi_extension_detected_as_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("recording.AVI") is True

    def test_mov_extension_detected_as_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("clip.mov") is True

    def test_mkv_extension_detected_as_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("film.mkv") is True

    def test_rtsp_url_is_not_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("rtsp://admin:pass@192.168.1.100/stream") is False

    def test_http_url_is_not_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("http://camera.local/video.cgi") is False

    def test_no_extension_is_not_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("rtsp://cam.local/stream1") is False

    def test_unknown_extension_is_not_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("/path/to/file.xyz") is False

    def test_webm_extension_detected_as_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("recording.webm") is True

    def test_ts_extension_detected_as_video(self):
        from monitor import _CameraThread
        assert _CameraThread._detect_video("/recordings/stream.ts") is True

    def test_thread_is_daemon(self):
        from monitor import _CameraThread
        t = _CameraThread(0, 15.0, lambda f: None)
        assert t.daemon is True

    def test_is_video_flag_set_correctly_for_int(self):
        from monitor import _CameraThread
        t = _CameraThread(0, 15.0, lambda f: None)
        assert t._is_video is False

    def test_is_video_flag_set_correctly_for_mp4(self):
        from monitor import _CameraThread
        t = _CameraThread("/path/video.mp4", 15.0, lambda f: None)
        assert t._is_video is True

    def test_is_video_flag_set_correctly_for_rtsp(self):
        from monitor import _CameraThread
        t = _CameraThread("rtsp://cam/stream", 15.0, lambda f: None)
        assert t._is_video is False

    def test_stop_clears_running_flag(self):
        from monitor import _CameraThread
        t = _CameraThread(0, 15.0, lambda f: None)
        t._running = True
        t.stop()
        assert t._running is False


# ---------------------------------------------------------------------------
# _MonitorState — thread-safe shared state
# ---------------------------------------------------------------------------

class TestMonitorState:
    def test_initial_score_is_zero(self):
        from monitor import _MonitorState
        s = _MonitorState()
        assert s.score == 0.0

    def test_initial_event_count_is_zero(self):
        from monitor import _MonitorState
        s = _MonitorState()
        assert s.event_count == 0

    def test_initial_score_buffer_empty(self):
        from monitor import _MonitorState
        s = _MonitorState()
        assert s.score_buffer == []

    def test_initial_latest_alarm_empty(self):
        from monitor import _MonitorState
        s = _MonitorState()
        assert s.latest_alarm == {}

    def test_push_score_updates_score(self):
        from monitor import _MonitorState
        s = _MonitorState()
        s.push_score(0.42, 0.10)
        assert abs(s.score - 0.42) < 1e-9

    def test_push_score_sets_is_alarm_true_when_above_threshold(self):
        from monitor import _MonitorState
        s = _MonitorState()
        s.push_score(0.50, 0.10)
        assert s.is_alarm is True

    def test_push_score_sets_is_alarm_false_when_below_threshold(self):
        from monitor import _MonitorState
        s = _MonitorState()
        s.push_score(0.05, 0.10)
        assert s.is_alarm is False

    def test_push_score_appends_to_buffer(self):
        from monitor import _MonitorState
        s = _MonitorState()
        s.push_score(0.1, 0.5)
        s.push_score(0.2, 0.5)
        assert len(s.score_buffer) == 2

    def test_push_score_buffer_capped_at_500(self):
        from monitor import _MonitorState
        s = _MonitorState()
        for i in range(600):
            s.push_score(float(i) / 1000.0, 0.5)
        assert len(s.score_buffer) <= 500

    def test_push_score_entry_has_expected_keys(self):
        from monitor import _MonitorState
        s = _MonitorState()
        s.push_score(0.123, 0.5)
        entry = s.score_buffer[-1]
        assert "ts" in entry
        assert "score" in entry
        assert "threshold" in entry
        assert "alarm" in entry

    def test_push_alarm_updates_latest_alarm(self):
        from monitor import _MonitorState
        s = _MonitorState()
        s.push_alarm(0.9, 0.5, "alarm_001.jpg")
        assert s.latest_alarm["frame_filename"] == "alarm_001.jpg"
        assert abs(s.latest_alarm["score"] - 0.9) < 1e-6

    def test_push_alarm_increments_event_count(self):
        from monitor import _MonitorState
        s = _MonitorState()
        s.push_alarm(0.9, 0.5, "f.jpg")
        s.push_alarm(0.8, 0.5, "g.jpg")
        assert s.event_count == 2

    def test_snapshot_returns_dict(self):
        from monitor import _MonitorState
        s = _MonitorState()
        snap = s.snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_contains_expected_keys(self):
        from monitor import _MonitorState
        s = _MonitorState()
        snap = s.snapshot()
        for key in ("model_name", "threshold", "score", "is_alarm",
                    "event_count", "cam_status", "uptime_s", "score_count"):
            assert key in snap, f"Missing key: {key}"

    def test_snapshot_uptime_is_non_negative(self):
        from monitor import _MonitorState
        import time
        s = _MonitorState()
        time.sleep(0.01)
        assert s.snapshot()["uptime_s"] >= 0


# ---------------------------------------------------------------------------
# build_parser — new arguments added in v1.1.0
# ---------------------------------------------------------------------------

class TestBuildParserNewArgs:
    def test_url_argument_accepted(self):
        from monitor import build_parser
        p = build_parser()
        args = p.parse_args(["--model", "m.pt", "--url", "rtsp://cam/stream"])
        assert args.url == "rtsp://cam/stream"

    def test_url_and_camera_are_mutually_exclusive(self):
        from monitor import build_parser
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--model", "m.pt", "--camera", "0", "--url", "rtsp://cam"])

    def test_default_url_is_none(self):
        from monitor import build_parser
        p = build_parser()
        args = p.parse_args(["--model", "m.pt"])
        assert args.url is None

    def test_reconnect_delay_default_is_5(self):
        from monitor import build_parser
        p = build_parser()
        args = p.parse_args(["--model", "m.pt"])
        assert abs(args.reconnect_delay - 5.0) < 1e-9

    def test_reconnect_delay_custom_value(self):
        from monitor import build_parser
        p = build_parser()
        args = p.parse_args(["--model", "m.pt", "--reconnect-delay", "10.0"])
        assert abs(args.reconnect_delay - 10.0) < 1e-9

    def test_mqtt_host_default_is_empty(self):
        from monitor import build_parser
        p = build_parser()
        assert p.parse_args(["--model", "m.pt"]).mqtt_host == ""

    def test_mqtt_port_default_is_1883(self):
        from monitor import build_parser
        p = build_parser()
        assert p.parse_args(["--model", "m.pt"]).mqtt_port == 1883

    def test_mqtt_topic_default(self):
        from monitor import build_parser
        p = build_parser()
        assert p.parse_args(["--model", "m.pt"]).mqtt_topic == "picture_studio/monitor"

    def test_mqtt_user_default_is_empty(self):
        from monitor import build_parser
        p = build_parser()
        assert p.parse_args(["--model", "m.pt"]).mqtt_user == ""

    def test_mqtt_pass_default_is_empty(self):
        from monitor import build_parser
        p = build_parser()
        assert p.parse_args(["--model", "m.pt"]).mqtt_pass == ""

    def test_api_port_default_is_zero(self):
        from monitor import build_parser
        p = build_parser()
        assert p.parse_args(["--model", "m.pt"]).api_port == 0

    def test_api_port_custom_value(self):
        from monitor import build_parser
        p = build_parser()
        args = p.parse_args(["--model", "m.pt", "--api-port", "8766"])
        assert args.api_port == 8766

    def test_api_key_default_is_empty(self):
        from monitor import build_parser
        p = build_parser()
        assert p.parse_args(["--model", "m.pt"]).api_key == ""

    def test_api_key_custom_value(self):
        from monitor import build_parser
        p = build_parser()
        args = p.parse_args(["--model", "m.pt", "--api-key", "secret123"])
        assert args.api_key == "secret123"


# ---------------------------------------------------------------------------
# _MonitorApiServer + _MonitorHandler — live HTTP tests
# ---------------------------------------------------------------------------

import json as _json
import socket as _socket
import urllib.error as _urlerr
import urllib.request as _urlreq


def _free_port() -> int:
    with _socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _http_get(url: str, headers: dict | None = None, timeout: float = 5.0):
    req = _urlreq.Request(url, headers=headers or {})
    try:
        with _urlreq.urlopen(req, timeout=timeout) as r:
            return r.status, _json.loads(r.read())
    except _urlerr.HTTPError as e:
        return e.code, _json.loads(e.read())


@pytest.fixture(scope="module")
def monitor_server(tmp_path_factory):
    """Start a _MonitorApiServer on a free port; tear it down after the module."""
    from monitor import _MonitorApiServer, _MonitorState
    state = _MonitorState()
    state.model_name = "test_model.pth"
    state.threshold = 0.05
    state.cam_status = "Verbunden"
    state.output_dir = str(tmp_path_factory.mktemp("api_frames"))

    port = _free_port()
    srv = _MonitorApiServer(port, state)
    srv.start()
    import time; time.sleep(0.1)
    yield port, state, srv
    srv.stop()


@pytest.fixture(scope="module")
def auth_server(tmp_path_factory):
    """Server with API key set."""
    from monitor import _MonitorApiServer, _MonitorState, _MonitorHandler
    state = _MonitorState()
    state.api_key = "monitor-secret-key"
    state.output_dir = str(tmp_path_factory.mktemp("auth_frames"))

    port = _free_port()
    _MonitorHandler.state = state
    srv = _MonitorApiServer(port, state)
    srv.start()
    import time; time.sleep(0.1)
    yield port, state, srv
    srv.stop()


class TestMonitorApiServer:
    def test_status_endpoint_returns_200(self, monitor_server):
        port, state, _ = monitor_server
        status, body = _http_get(f"http://localhost:{port}/api/status")
        assert status == 200

    def test_status_has_model_name(self, monitor_server):
        port, state, _ = monitor_server
        _, body = _http_get(f"http://localhost:{port}/api/status")
        assert body["model_name"] == "test_model.pth"

    def test_status_has_threshold(self, monitor_server):
        port, state, _ = monitor_server
        _, body = _http_get(f"http://localhost:{port}/api/status")
        assert abs(body["threshold"] - 0.05) < 1e-9

    def test_scores_endpoint_returns_200(self, monitor_server):
        port, state, _ = monitor_server
        status, _ = _http_get(f"http://localhost:{port}/api/scores")
        assert status == 200

    def test_scores_endpoint_has_scores_key(self, monitor_server):
        port, state, _ = monitor_server
        _, body = _http_get(f"http://localhost:{port}/api/scores")
        assert "scores" in body

    def test_latest_alarm_endpoint_returns_200(self, monitor_server):
        port, state, _ = monitor_server
        status, _ = _http_get(f"http://localhost:{port}/api/latest_alarm")
        assert status == 200

    def test_scores_reflect_pushed_data(self, monitor_server):
        port, state, _ = monitor_server
        state.push_score(0.123, 0.05)
        _, body = _http_get(f"http://localhost:{port}/api/scores")
        assert body["count"] >= 1

    def test_latest_alarm_reflects_pushed_alarm(self, monitor_server):
        port, state, _ = monitor_server
        state.push_alarm(0.9, 0.05, "alarm_test.jpg")
        _, body = _http_get(f"http://localhost:{port}/api/latest_alarm")
        assert body["frame_filename"] == "alarm_test.jpg"

    def test_unknown_endpoint_returns_404(self, monitor_server):
        port, _, _ = monitor_server
        status, _ = _http_get(f"http://localhost:{port}/api/does_not_exist")
        assert status == 404

    def test_frame_invalid_path_returns_400(self, monitor_server):
        port, _, _ = monitor_server
        status, _ = _http_get(f"http://localhost:{port}/api/frame/../etc/passwd")
        assert status in (400, 404)

    def test_dashboard_returns_html(self, monitor_server):
        port, _, _ = monitor_server
        req = _urlreq.Request(f"http://localhost:{port}/dashboard")
        with _urlreq.urlopen(req, timeout=5) as r:
            ct = r.headers.get("Content-Type", "")
            body = r.read().decode()
        assert "text/html" in ct
        assert "PictureStudio Monitor" in body


class TestMonitorApiAuth:
    def test_status_is_public_no_key_needed(self, auth_server):
        port, _, _ = auth_server
        status, _ = _http_get(f"http://localhost:{port}/api/status")
        assert status == 200

    def test_dashboard_is_public_no_key_needed(self, auth_server):
        port, _, _ = auth_server
        req = _urlreq.Request(f"http://localhost:{port}/dashboard")
        with _urlreq.urlopen(req, timeout=5) as r:
            assert r.status == 200

    def test_scores_requires_key_returns_401_without_it(self, auth_server):
        port, _, _ = auth_server
        status, _ = _http_get(f"http://localhost:{port}/api/scores")
        assert status == 401

    def test_latest_alarm_requires_key_returns_401_without_it(self, auth_server):
        port, _, _ = auth_server
        status, _ = _http_get(f"http://localhost:{port}/api/latest_alarm")
        assert status == 401

    def test_correct_key_grants_access_to_scores(self, auth_server):
        port, _, _ = auth_server
        status, _ = _http_get(
            f"http://localhost:{port}/api/scores",
            headers={"X-Api-Key": "monitor-secret-key"},
        )
        assert status == 200

    def test_wrong_key_returns_401(self, auth_server):
        port, _, _ = auth_server
        status, _ = _http_get(
            f"http://localhost:{port}/api/scores",
            headers={"X-Api-Key": "wrong-key"},
        )
        assert status == 401

    def test_bearer_token_auth_works(self, auth_server):
        port, _, _ = auth_server
        status, _ = _http_get(
            f"http://localhost:{port}/api/scores",
            headers={"Authorization": "Bearer monitor-secret-key"},
        )
        assert status == 200

    def test_401_body_has_error_field(self, auth_server):
        port, _, _ = auth_server
        _, body = _http_get(f"http://localhost:{port}/api/scores")
        assert "error" in body

    def test_dashboard_html_embeds_api_key(self, auth_server):
        port, _, _ = auth_server
        req = _urlreq.Request(f"http://localhost:{port}/dashboard")
        with _urlreq.urlopen(req, timeout=5) as r:
            body = r.read().decode()
        assert "monitor-secret-key" in body
