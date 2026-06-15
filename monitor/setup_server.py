"""Multi-channel Setup-Wizard server — built on the shared ``core.http_router``.

Training happens exclusively in Picture Studio; this wizard only handles camera
preview, ROI drawing, and model deployment (binary upload). Web UI routes:

    GET    /setup                       — HTML wizard page
    GET    /setup/cameras               — list of discovered USB cameras
    GET    /setup/status                — JSON snapshot of _SetupState
    GET    /setup/channels/{id}/frame.jpg — JPEG preview from the camera thread
    POST   /setup/channels/add          — create a channel + start its camera thread
    POST   /setup/channels/{id}/roi     — set or clear ROI for a channel
    POST   /setup/channels/{id}/deploy  — upload model binary for a channel
    POST   /setup/go_live               — transition phase → 'live'
    DELETE /setup/channels/{id}         — remove a channel and stop its camera thread
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np

from core.http_router import Request, Response, Router, RouterServer
from monitor.camera import _CameraThread
from monitor.web import load_html


class _SetupChannel:
    """
    Represents one camera channel in the multi-channel setup wizard.

    Holds camera source, optional ROI, the deployed model path, and the most
    recent frame captured by the associated camera thread.  All frame access
    goes through get_frame()/set_frame() to stay thread-safe.
    """

    def __init__(self, channel_id: int, camera_source: Union[int, str]) -> None:
        self.channel_id: int = channel_id
        self.camera_source: Union[int, str] = camera_source
        self.roi: Optional[list] = None          # [x, y, w, h] in canvas pixels, or None
        self.model_path: str = ""                # populated after deploy
        self.status: str = "pending"             # pending | ready
        self.last_frame: Optional[np.ndarray] = None
        self.cam_thread: Optional[_CameraThread] = None
        self._lock: threading.Lock = threading.Lock()

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame in a thread-safe manner."""
        with self._lock:
            return self.last_frame

    def set_frame(self, frame: np.ndarray) -> None:
        """Store the most recent frame in a thread-safe manner."""
        with self._lock:
            self.last_frame = frame.copy()

    def to_dict(self) -> dict:
        """Serialise channel metadata (no frame data, no thread reference)."""
        return {
            "channel_id":    self.channel_id,
            "camera_source": self.camera_source,
            "roi":           self.roi,
            "model_path":    self.model_path,
            "status":        self.status,
        }


class _SetupState:
    """
    Thread-safe state container for the multi-channel Setup-Wizard.

    Tracks all configured channels and the overall wizard phase
    (setup | live | error).
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self.channels: list = []    # list of _SetupChannel
        self.phase: str = "setup"
        self.error: str = ""
        self._next_id: int = 0

    def add_channel(self, camera_source: Union[int, str]) -> "_SetupChannel":
        """Create a new channel, append it to the list, and return it."""
        with self._lock:
            ch = _SetupChannel(self._next_id, camera_source)
            self._next_id += 1
            self.channels.append(ch)
        return ch

    def remove_channel(self, channel_id: int) -> bool:
        """Remove a channel by id.  Returns True if found and removed."""
        with self._lock:
            before = len(self.channels)
            self.channels = [c for c in self.channels if c.channel_id != channel_id]
            return len(self.channels) < before

    def get_channel(self, channel_id: int) -> Optional["_SetupChannel"]:
        """Return the channel with the given id, or None."""
        with self._lock:
            for ch in self.channels:
                if ch.channel_id == channel_id:
                    return ch
        return None

    def all_ready(self) -> bool:
        """True iff all channels have a non-empty model_path (status == 'ready')."""
        with self._lock:
            if not self.channels:
                return False
            return all(c.model_path != "" for c in self.channels)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of the current state."""
        with self._lock:
            return {
                "phase":    self.phase,
                "error":    self.error,
                "channels": [c.to_dict() for c in self.channels],
            }


# ── route handlers ──────────────────────────────────────────────────────────

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _channel_frame(state: _SetupState, req: Request) -> Response:
    try:
        ch_id = int(req.params["id"])
    except ValueError:
        return Response.error("not found", 404)
    ch = state.get_channel(ch_id)
    if ch is None:
        return Response.error("channel not found", 404)
    frame = ch.get_frame()
    if frame is None:
        placeholder = np.full((120, 160, 3), 60, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 50])
    else:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return Response(204, b"", "")
    return Response.data(buf.tobytes(), "image/jpeg", headers=dict(_NO_CACHE))


def _add_channel(state: _SetupState, req: Request) -> Response:
    body = req.json()
    src = body.get("camera_source", 0)
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    ch = state.add_channel(src)

    def _on_frame(frame: np.ndarray) -> None:
        ch.set_frame(frame)

    cam_thread = _CameraThread(source=src, fps=15.0, callback=_on_frame, reconnect_delay=5.0)
    ch.cam_thread = cam_thread
    cam_thread.start()
    return Response.json({"ok": True, "channel_id": ch.channel_id})


def _set_roi(state: _SetupState, req: Request) -> Response:
    try:
        ch_id = int(req.params["id"])
    except ValueError:
        return Response.error("bad request", 400)
    ch = state.get_channel(ch_id)
    if ch is None:
        return Response.error("channel not found", 404)
    body = req.json()
    roi = body.get("roi")
    if roi is not None:
        roi = [float(v) for v in roi[:4]]
    ch.roi = roi
    return Response.json({"ok": True, "roi": ch.roi})


def _deploy(state: _SetupState, output_dir: str, req: Request) -> Response:
    try:
        ch_id = int(req.params["id"])
    except ValueError:
        return Response.error("bad request", 400)
    ch = state.get_channel(ch_id)
    if ch is None:
        return Response.error("channel not found", 404)

    ct = req.headers.get("Content-Type", "")
    if "multipart/form-data" not in ct:
        return Response.error("expected multipart/form-data", 400)
    boundary = None
    for part in ct.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip()
    if not boundary:
        return Response.error("missing boundary", 400)

    raw = req.body
    parts_raw = raw.split(("--" + boundary).encode())
    model_data = None
    for seg in parts_raw[1:]:
        if b'name="model"' in seg or b"name='model'" in seg:
            idx = seg.find(b"\r\n\r\n")
            if idx >= 0:
                model_data = seg[idx + 4:]
                if model_data.endswith(b"\r\n"):
                    model_data = model_data[:-2]
            break
    if not model_data:
        return Response.error("model field not found in upload", 400)

    os.makedirs(os.path.join(output_dir, "setup_models"), exist_ok=True)
    ext = ".pth"
    for seg in parts_raw[1:]:
        if b'name="model"' in seg or b"name='model'" in seg:
            if b".onnx" in seg[:200]:
                ext = ".onnx"
            break
    model_file = os.path.join(output_dir, "setup_models", f"channel_{ch_id}{ext}")
    with open(model_file, "wb") as f:
        f.write(model_data)
    ch.model_path = model_file
    ch.status = "ready"
    return Response.json({"ok": True, "model_path": model_file})


def _go_live(state: _SetupState) -> Response:
    with state._lock:
        state.phase = "live"
    return Response.json({"ok": True})


def _delete_channel(state: _SetupState, req: Request) -> Response:
    try:
        ch_id = int(req.params["id"])
    except ValueError:
        return Response.error("bad request", 400)
    ch = state.get_channel(ch_id)
    if ch and ch.cam_thread:
        ch.cam_thread.stop()
    if state.remove_channel(ch_id):
        return Response.json({"ok": True, "removed": ch_id})
    return Response.error("channel not found", 404)


def build_router(state: _SetupState, output_dir: str, discovered_cameras: list) -> Router:
    """Register all setup-wizard routes (no authentication on the wizard)."""
    r = Router(cors=True)
    r.get("/", lambda req: Response.redirect("/setup", 301))
    r.get("/setup", lambda req: Response.html(load_html("setup")))
    r.get("/setup/cameras", lambda req: Response.json(
        [{"index": idx, "label": label} for idx, label in discovered_cameras]))
    r.get("/setup/status", lambda req: Response.json(state.snapshot()))
    r.get("/setup/channels/{id}/frame.jpg", lambda req: _channel_frame(state, req))
    r.post("/setup/channels/add", lambda req: _add_channel(state, req))
    r.post("/setup/channels/{id}/roi", lambda req: _set_roi(state, req))
    r.post("/setup/channels/{id}/deploy", lambda req: _deploy(state, output_dir, req))
    r.post("/setup/go_live", lambda req: _go_live(state))
    r.delete("/setup/channels/{id}", lambda req: _delete_channel(state, req))
    return r


class SetupApiServer:
    """Runs the multi-channel Setup-Wizard HTTP server on a daemon thread."""

    def __init__(self, port: int, state: _SetupState,
                 output_dir: str = "monitor_logs", discovered_cameras: list = ()) -> None:
        self._port = port
        self._srv = RouterServer(
            port, build_router(state, output_dir, list(discovered_cameras)),
            thread_name="setup-api",
        )

    def start(self) -> None:
        self._srv.start()

    def stop(self) -> None:
        self._srv.stop()


# Backward-compatible alias.
_SetupApiServer = SetupApiServer


def run_setup(
    camera_source: Union[int, str, None],
    setup_port: int = 8765,
    output_dir: str = "monitor_logs",
    discovered_cameras: list = [],
) -> list:
    """
    Start the multi-channel Setup-Wizard and block until go_live is called.

    Returns a list of channel dicts
    ``[{"channel_id", "camera_source", "roi", "model_path", "status"}, ...]``.
    """
    os.makedirs(os.path.join(output_dir, "setup_models"), exist_ok=True)

    state = _SetupState()

    # Optionally pre-populate one channel
    if camera_source is not None:
        ch = state.add_channel(camera_source)

        def _on_frame(frame: np.ndarray) -> None:
            ch.set_frame(frame)

        cam_thread = _CameraThread(
            source=camera_source, fps=15.0, callback=_on_frame, reconnect_delay=5.0,
        )
        ch.cam_thread = cam_thread
        cam_thread.start()

    server = SetupApiServer(
        port=setup_port, state=state, output_dir=output_dir,
        discovered_cameras=discovered_cameras,
    )
    server.start()

    print(
        f"\n"
        f"╔═══════════════════════════════════════════════════╗\n"
        f"║  PictureStudio Monitor – Einrichtungsmodus        ║\n"
        f"║  Web-Interface: http://localhost:{setup_port}/setup      ║\n"
        f"╚═══════════════════════════════════════════════════╝\n"
    )

    # Block until go_live or error
    while state.phase not in ("live", "error"):
        time.sleep(0.5)

    # Stop all channel camera threads
    with state._lock:
        for ch in state.channels:
            if ch.cam_thread:
                ch.cam_thread.stop()

    server.stop()

    with state._lock:
        return [c.to_dict() for c in state.channels]
