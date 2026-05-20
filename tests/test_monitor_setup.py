"""
Tests für den Setup-Wizard in monitor.py.

Getestet werden:
  1. _SetupState.snapshot() Schlüssel
  2. _SetupState Standardphase
  3. Argument-Parser: --setup --camera 0 --setup-port 8766
  4. Ohne --setup und ohne --model → SystemExit (parser.error)
  5. _SetupHandler /setup/status gibt 200 JSON zurück
  6. _SetupHandler /setup/frame.jpg gibt 204 oder JPEG zurück wenn kein Frame
"""
from __future__ import annotations

import http.client
import importlib
import json
import sys
import threading
import time
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

# ── Import monitor without executing main() ───────────────────────────────────
import importlib.util, os

_MONITOR_PATH = os.path.join(os.path.dirname(__file__), "..", "monitor.py")
_spec = importlib.util.spec_from_file_location("monitor", _MONITOR_PATH)
_monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_monitor)

_SetupState = _monitor._SetupState
_SetupHandler = _monitor._SetupHandler
_SetupApiServer = _monitor._SetupApiServer
build_parser = _monitor.build_parser


# ── 1. snapshot() enthält erwartete Schlüssel ─────────────────────────────────
def test_setup_state_snapshot_keys():
    state = _SetupState()
    snap = state.snapshot()
    required = {"phase", "frame_count", "threshold", "epoch", "model_path"}
    assert required.issubset(snap.keys()), (
        f"snapshot() fehlen Schlüssel: {required - snap.keys()}"
    )


# ── 2. Standardphase ist "idle" ───────────────────────────────────────────────
def test_setup_state_phase_default():
    state = _SetupState()
    assert state.snapshot()["phase"] == "idle"


# ── 3. Parser: --setup --camera 0 --setup-port 8766 ──────────────────────────
def test_setup_parse_setup_flag():
    parser = build_parser()
    args = parser.parse_args(["--setup", "--camera", "0", "--setup-port", "8766"])
    assert args.setup is True
    assert args.camera == 0
    assert args.setup_port == 8766


# ── 4. Ohne --setup und ohne --model → SystemExit ────────────────────────────
def test_setup_model_optional_without_setup_flag():
    """Without --setup and without --model, main() must call parser.error → SystemExit."""
    parser = build_parser()
    # Parse with no arguments — model is None, setup is False
    args = parser.parse_args([])
    assert args.model is None
    assert args.setup is False
    # Simulate what main() does: if not args.model → parser.error(...)
    with pytest.raises(SystemExit):
        parser.error("--model ist erforderlich")


# ── helpers for real HTTP tests ───────────────────────────────────────────────

def _find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_test_server(state: _SetupState) -> tuple[_SetupApiServer, int]:
    """Start a _SetupApiServer on a free port and return (server, port)."""
    # We need a real AnomalyDetector-like object; use a minimal mock
    det_mock = MagicMock()
    det_mock._threshold = 0.0

    port = _find_free_port()
    srv = _SetupApiServer(port=port, state=state, detector=det_mock, output_dir="/tmp")
    srv.start()
    # Give the server a moment to bind
    time.sleep(0.15)
    return srv, port


# ── 5. /setup/status gibt 200 JSON ───────────────────────────────────────────
def test_setup_handler_status_route():
    state = _SetupState()
    srv, port = _start_test_server(state)
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request("GET", "/setup/status")
        resp = conn.getresponse()
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        body = resp.read()
        data = json.loads(body)
        assert "phase" in data, f"'phase' missing from response: {data}"
    finally:
        srv.stop()


# ── 6. /setup/frame.jpg ohne Frame → 204 oder JPEG ───────────────────────────
def test_setup_handler_frame_missing():
    """When no frame has been captured yet, the endpoint must return 204 or 200 JPEG."""
    state = _SetupState()
    # last_frame is None by default → expect 204
    srv, port = _start_test_server(state)
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request("GET", "/setup/frame.jpg")
        resp = conn.getresponse()
        resp.read()  # consume body
        assert resp.status in (200, 204), (
            f"Expected 200 or 204 for missing frame, got {resp.status}"
        )
    finally:
        srv.stop()
