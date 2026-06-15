"""Shared monitor state + JPEG frame ring-buffer."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import cv2


_FRAME_BUF_MAX = 200   # max JPEG frames kept in memory (~10 MB at quality 70)
_FRAME_BUF_INTERVAL = 0.5  # seconds between buffer pushes (≈2 fps) to limit CPU

class _MonitorState:
    """Central store for score history, alarm data, and the frame ring-buffer.

    All public attributes are protected by ``_lock``; the frame ring-buffer
    has its own separate ``_frame_buf_lock`` so frame writes from the camera
    thread never contend with JSON-serialisation of score data.

    Frame ring-buffer
    -----------------
    push_frame(frame)   — add a JPEG-compressed copy (throttled to ~2 fps)
    get_frames(n)       — return the n most-recent JPEG bytes

    Hot-swap
    --------
    pending_model_path  — set by POST /api/deploy; consumed by run_monitor()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model_name: str = ""
        self.threshold: float = 0.0
        self.score: float = 0.0
        self.is_alarm: bool = False
        self.event_count: int = 0
        self.cam_status: str = "Nicht verbunden"
        self.output_dir: str = ""
        self.api_key: str = ""
        self.score_buffer: list = []          # [{ts, score, threshold, alarm}]
        self.latest_alarm: dict = {}          # {ts, score, threshold, frame_filename}
        self.start_time: float = time.time()
        # Frame ring-buffer: stores JPEG bytes for GET /api/frames
        self._frame_buf: list = []            # list[bytes]
        self._frame_buf_lock = threading.Lock()
        self._frame_buf_last_t: float = 0.0
        # Hot-swap: set by POST /api/deploy, consumed by run_monitor main loop
        self.pending_model_path: str = ""

    def push_score(self, score: float, threshold: float) -> None:
        entry = {
            "ts":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "score":     round(score, 6),
            "threshold": round(threshold, 6),
            "alarm":     score > threshold,
        }
        with self._lock:
            self.score_buffer.append(entry)
            if len(self.score_buffer) > 500:
                self.score_buffer = self.score_buffer[-500:]
            self.score = score
            self.is_alarm = score > threshold

    def push_alarm(self, score: float, threshold: float, frame_filename: str) -> None:
        with self._lock:
            self.latest_alarm = {
                "ts":             datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "score":          round(score, 6),
                "threshold":      round(threshold, 6),
                "frame_filename": frame_filename,
            }
            self.event_count += 1

    def push_frame(self, frame) -> None:
        """Store a JPEG-compressed copy of frame in the ring buffer (throttled to ~2 fps)."""
        now = time.time()
        if now - self._frame_buf_last_t < _FRAME_BUF_INTERVAL:
            return
        self._frame_buf_last_t = now
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        data = buf.tobytes()
        with self._frame_buf_lock:
            self._frame_buf.append(data)
            if len(self._frame_buf) > _FRAME_BUF_MAX:
                del self._frame_buf[0]

    def get_frames(self, n: int) -> list:
        """Return up to n most-recent JPEG frame bytes from the ring buffer."""
        with self._frame_buf_lock:
            return list(self._frame_buf[-n:])

    def snapshot(self) -> dict:
        with self._lock:
            with self._frame_buf_lock:
                frame_count = len(self._frame_buf)
            return {
                "model_name":   self.model_name,
                "threshold":    self.threshold,
                "score":        self.score,
                "is_alarm":     self.is_alarm,
                "event_count":  self.event_count,
                "cam_status":   self.cam_status,
                "uptime_s":     int(time.time() - self.start_time),
                "score_count":  len(self.score_buffer),
                "frame_count":  frame_count,
            }
