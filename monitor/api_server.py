"""Standalone monitor REST API — built on the shared ``core.http_router``.

Endpoints (auth via ``state.api_key``; ``/dashboard`` and ``/api/status`` are
always public):

    GET  /api/status        — JSON snapshot of _MonitorState
    GET  /api/scores        — recent score buffer
    GET  /api/latest_alarm  — last alarm metadata
    GET  /api/frame/<name>  — alarm JPEG from disk
    GET  /api/frames        — ZIP of ring-buffer frames (training data)
    GET  /dashboard         — HTML dashboard
    POST /api/deploy        — upload a new .pth model (hot-swap)
"""
from __future__ import annotations

import io
import os
import zipfile

from core.http_router import Request, Response, Router, RouterServer
from monitor.state import _FRAME_BUF_MAX, _MonitorState
from monitor.web import load_html


def _scores(state: _MonitorState, req: Request) -> Response:
    limit = req.query_int("limit", 120)
    with state._lock:
        buf = state.score_buffer[-limit:]
    return Response.json({"scores": buf, "count": len(buf)})


def _latest_alarm(state: _MonitorState) -> Response:
    with state._lock:
        alarm = dict(state.latest_alarm)
    return Response.json(alarm)


def _frame(state: _MonitorState, req: Request) -> Response:
    fname = req.params["name"]
    if not fname or "/" in fname or ".." in fname:
        return Response.error("Invalid filename", 400)
    fpath = os.path.join(state.output_dir, fname)
    if not os.path.isfile(fpath):
        return Response.error("Not found", 404)
    try:
        with open(fpath, "rb") as f:
            data = f.read()
        return Response.data(data, "image/jpeg")
    except Exception as exc:
        return Response.error(str(exc), 500)


def _frames(state: _MonitorState, req: Request) -> Response:
    n = min(req.query_int("n", 150), _FRAME_BUF_MAX)
    frames = state.get_frames(n)
    if not frames:
        return Response.error("Kein Frame im Puffer — Kamera läuft?", 503)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i, jpg in enumerate(frames):
            zf.writestr(f"frame_{i:04d}.jpg", jpg)
    return Response.data(
        buf.getvalue(), "application/zip",
        headers={"Content-Disposition": 'attachment; filename="frames.zip"'},
    )


def _dashboard(state: _MonitorState) -> Response:
    key = state.api_key
    html = load_html("dashboard").replace("__API_KEY__", f"'{key}'" if key else "''")
    return Response.html(html)


def _deploy(state: _MonitorState, req: Request) -> Response:
    ct = req.headers.get("Content-Type", "")
    raw = req.body
    if not raw:
        return Response.error("Leerer Body", 400)

    model_data = None
    if "multipart/form-data" in ct:
        boundary = None
        for part in ct.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[len("boundary="):].strip().encode()
        if boundary:
            for p in raw.split(b"--" + boundary)[1:]:
                if b"\r\n\r\n" in p:
                    _, body = p.split(b"\r\n\r\n", 1)
                    body = body.rstrip(b"\r\n--")
                    if len(body) > 100:
                        model_data = body
                        break
    else:
        model_data = raw

    if not model_data:
        return Response.error("Kein Modell-Inhalt", 400)

    out_dir = state.output_dir or "monitor_logs"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "deployed_model.pth")
    try:
        with open(save_path, "wb") as fh:
            fh.write(model_data)
    except Exception as exc:
        return Response.error(f"Speicherfehler: {exc}", 500)

    state.pending_model_path = save_path
    print(f"\n[Deploy] Neues Modell empfangen → {save_path} ({len(model_data)} Bytes)")
    return Response.json({"ok": True, "model_path": save_path, "size_bytes": len(model_data)})


def build_router(state: _MonitorState) -> Router:
    """Register all monitor REST routes on a shared :class:`Router`."""
    r = Router(
        api_key=lambda: state.api_key,
        public_paths={"/dashboard", "/api/status"},
        cors=True,
    )
    r.get("/api/status", lambda req: Response.json(state.snapshot()))
    r.get("/api/scores", lambda req: _scores(state, req))
    r.get("/api/latest_alarm", lambda req: _latest_alarm(state))
    r.get("/api/frame/{name}", lambda req: _frame(state, req))
    r.get("/api/frames", lambda req: _frames(state, req))
    r.get("/dashboard", lambda req: _dashboard(state))
    r.post("/api/deploy", lambda req: _deploy(state, req))
    return r


class MonitorApiServer:
    """Runs the monitor REST API on a background daemon thread."""

    def __init__(self, port: int, state: _MonitorState) -> None:
        self._port = port
        self._srv = RouterServer(port, build_router(state), thread_name="monitor-api")

    def start(self) -> None:
        self._srv.start()

    def stop(self) -> None:
        self._srv.stop()


# Backward-compatible alias (run loops construct _MonitorApiServer(port, state)).
_MonitorApiServer = MonitorApiServer
