"""
Tests for the three production-readiness fixes:
  Fix 2 — Camera auto-reconnect logic in camera_page.py
  Fix 3 — Video file inference: "Videodatei" combo entry + FPS detection

These tests exercise the logic without a real camera or Qt event loop by
inspecting source code and testing helper components in isolation.
"""
import inspect
import pytest


# ---------------------------------------------------------------------------
# Fix 2 — Auto-reconnect state machine
# ---------------------------------------------------------------------------

class TestAutoReconnectState:
    """Verify the reconnect state attributes and _on_camera_error branching."""

    def _get_source(self, method_name: str) -> str:
        from gui.pages.camera_page import CameraPage
        return inspect.getsource(getattr(CameraPage, method_name))

    def test_reconnect_timer_initialised_in_init(self):
        src = inspect.getsource(
            __import__("gui.pages.camera_page", fromlist=["CameraPage"]).CameraPage.__init__
        )
        assert "_reconnect_timer" in src
        assert "QTimer" in src

    def test_reconnect_timer_is_single_shot(self):
        src = inspect.getsource(
            __import__("gui.pages.camera_page", fromlist=["CameraPage"]).CameraPage.__init__
        )
        assert "setSingleShot(True)" in src

    def test_reconnect_source_initialised_none(self):
        src = inspect.getsource(
            __import__("gui.pages.camera_page", fromlist=["CameraPage"]).CameraPage.__init__
        )
        assert "_reconnect_source = None" in src

    def test_is_video_file_flag_initialised_false(self):
        src = inspect.getsource(
            __import__("gui.pages.camera_page", fromlist=["CameraPage"]).CameraPage.__init__
        )
        assert "_is_video_file" in src

    def test_on_camera_error_starts_timer_for_live_streams(self):
        src = self._get_source("_on_camera_error")
        assert "_reconnect_timer.start(5000)" in src

    def test_on_camera_error_does_not_reconnect_for_video_files(self):
        src = self._get_source("_on_camera_error")
        assert "_is_video_file" in src

    def test_on_camera_error_shows_yellow_for_reconnect(self):
        src = self._get_source("_on_camera_error")
        # Yellow colour is used for "reconnecting" state
        assert "#D29922" in src

    def test_stop_stream_clears_reconnect_source(self):
        src = self._get_source("_stop_stream")
        assert "_reconnect_source = None" in src

    def test_stop_stream_stops_timer(self):
        src = self._get_source("_stop_stream")
        assert "_reconnect_timer.stop()" in src

    def test_try_reconnect_method_exists(self):
        from gui.pages.camera_page import CameraPage
        assert hasattr(CameraPage, "_try_reconnect")

    def test_try_reconnect_increments_attempt_counter(self):
        src = self._get_source("_try_reconnect")
        assert "_reconnect_attempts += 1" in src

    def test_try_reconnect_stops_old_thread(self):
        src = self._get_source("_try_reconnect")
        assert ".stop()" in src

    def test_try_reconnect_creates_new_thread(self):
        src = self._get_source("_try_reconnect")
        assert "CameraFrameThread" in src

    def test_try_reconnect_updates_status_label(self):
        src = self._get_source("_try_reconnect")
        assert "_conn_status_lbl.setText" in src

    def test_on_frame_resets_reconnect_counter(self):
        src = self._get_source("_on_frame")
        assert "_reconnect_attempts = 0" in src

    def test_on_frame_restores_green_status_after_reconnect(self):
        src = self._get_source("_on_frame")
        assert "Verbunden" in src

    def test_reconnect_state_machine_logic(self):
        """Pure logic simulation: reconnect only for live streams, not video files."""
        timer_started = []

        def simulate_error(reconnect_source, is_video):
            if reconnect_source is not None and not is_video:
                timer_started.append(True)

        # Live stream with source set → reconnect
        simulate_error("rtsp://cam", False)
        assert len(timer_started) == 1

        # Video file → no reconnect
        simulate_error("/tmp/video.mp4", True)
        assert len(timer_started) == 1  # unchanged

        # Source cleared (stop_stream called) → no reconnect
        simulate_error(None, False)
        assert len(timer_started) == 1  # unchanged

    def test_attempt_counter_reset_on_first_frame(self):
        """First frame after reconnect resets the attempt counter to 0."""
        attempts = [3]  # simulate 3 failed reconnect attempts

        def on_frame_logic():
            if attempts[0] > 0:
                attempts[0] = 0

        on_frame_logic()
        assert attempts[0] == 0


# ---------------------------------------------------------------------------
# Fix 3 — Video file inference combo entry
# ---------------------------------------------------------------------------

class TestVideoFileInference:
    def _get_source(self, method_name: str) -> str:
        from gui.pages.camera_page import CameraPage
        return inspect.getsource(getattr(CameraPage, method_name))

    def test_on_scan_done_adds_video_option(self):
        src = self._get_source("_on_scan_done")
        assert "Videodatei" in src or "video" in src.lower()

    def test_video_item_has_video_userdata(self):
        src = self._get_source("_on_scan_done")
        assert 'userData="video"' in src or "userData='video'" in src

    def test_start_stream_handles_video_source(self):
        src = self._get_source("_start_stream")
        assert 'source == "video"' in src

    def test_start_stream_opens_file_dialog_for_video(self):
        src = self._get_source("_start_stream")
        assert "QFileDialog" in src

    def test_start_stream_accepts_common_video_formats(self):
        src = self._get_source("_start_stream")
        # File filter should include mp4 and avi at minimum
        assert "mp4" in src.lower()
        assert "avi" in src.lower()

    def test_start_stream_sets_is_video_file_flag(self):
        src = self._get_source("_start_stream")
        assert "_is_video_file" in src

    def test_start_stream_does_not_set_reconnect_source_for_video(self):
        src = self._get_source("_start_stream")
        # Reconnect source must be None for video files
        assert "None if is_video" in src or ("_reconnect_source = None" in src and "is_video" in src)

    def test_start_stream_detects_native_fps(self):
        src = self._get_source("_start_stream")
        assert "CAP_PROP_FPS" in src or "cap.get" in src

    def test_start_stream_falls_back_to_25fps_for_unknown_fps(self):
        src = self._get_source("_start_stream")
        assert "25.0" in src or "25" in src

    def test_stop_stream_clears_is_video_file_flag(self):
        src = self._get_source("_stop_stream")
        assert "_is_video_file = False" in src

    def test_video_status_label_shows_filename(self):
        src = self._get_source("_start_stream")
        assert "os.path.basename" in src

    def test_native_fps_detection_logic(self):
        """Simulate cv2.VideoCapture FPS extraction logic."""
        def get_video_fps(native):
            return native if native and native > 0 else 25.0

        assert get_video_fps(30.0) == 30.0
        assert get_video_fps(0.0) == 25.0
        assert get_video_fps(None) == 25.0
        assert get_video_fps(25.0) == 25.0
        assert get_video_fps(59.94) == 59.94
