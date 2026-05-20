"""
Tests für den überarbeiteten Multi-Channel Setup-Wizard in monitor.py.

Keine PyTorch-Abhängigkeit; keine Kamera erforderlich.

Tests:
  1. _SetupChannel.to_dict() enthält channel_id, camera_source, roi, model_path, status
  2. _SetupState.add_channel() → len(channels) == 1
  3. _SetupState.all_ready() → False wenn kein model_path
  4. _SetupState.all_ready() → True wenn model_path gesetzt
  5. Parser: --setup --setup-port 8766 → args.setup=True, args.setup_port=8766
  6. Parser: --channels config.json → args.channels=="config.json"
  7. _SetupHandler /setup/status gibt 200 JSON zurück (echter HTTP-Request)
  8. POST /setup/go_live → phase wird auf "live" gesetzt
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import time

import pytest

# ── Import monitor without executing main() ───────────────────────────────────
import importlib.util

_MONITOR_PATH = os.path.join(os.path.dirname(__file__), "..", "monitor.py")
_spec = importlib.util.spec_from_file_location("monitor", _MONITOR_PATH)
_monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_monitor)

_SetupChannel = _monitor._SetupChannel
_SetupState = _monitor._SetupState
_SetupHandler = _monitor._SetupHandler
_SetupApiServer = _monitor._SetupApiServer
build_parser = _monitor.build_parser


# ── helpers ───────────────────────────────────────────────────────────────────

def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_server(state: _SetupState) -> tuple:
    """Start a _SetupApiServer on a free port and return (server, port)."""
    port = _free_port()
    srv = _SetupApiServer(port=port, state=state, output_dir="/tmp")
    srv.start()
    time.sleep(0.15)
    return srv, port


# ── 1. _SetupChannel.to_dict() ────────────────────────────────────────────────

def test_setup_channel_to_dict():
    """to_dict() must expose all required keys without frame or thread data."""
    ch = _SetupChannel(channel_id=0, camera_source=1)
    d = ch.to_dict()
    required = {"channel_id", "camera_source", "roi", "model_path", "status"}
    assert required.issubset(d.keys()), f"Fehlende Schlüssel: {required - d.keys()}"
    assert d["channel_id"] == 0
    assert d["camera_source"] == 1
    assert d["roi"] is None
    assert d["model_path"] == ""
    assert d["status"] == "pending"


# ── 2. _SetupState.add_channel() ──────────────────────────────────────────────

def test_setup_state_add_channel():
    """add_channel() appends one channel to the list."""
    state = _SetupState()
    ch = state.add_channel(0)
    assert len(state.channels) == 1
    assert ch.channel_id == 0
    assert ch.camera_source == 0


# ── 3. all_ready() → False when no model ─────────────────────────────────────

def test_setup_state_all_ready_false():
    """all_ready() returns False when a channel has no model_path."""
    state = _SetupState()
    state.add_channel(0)
    assert state.all_ready() is False


# ── 4. all_ready() → True when model set ─────────────────────────────────────

def test_setup_state_all_ready_true():
    """all_ready() returns True when all channels have a non-empty model_path."""
    state = _SetupState()
    ch = state.add_channel(0)
    ch.model_path = "/tmp/model.pth"
    ch.status = "ready"
    assert state.all_ready() is True


# ── 5. --setup --setup-port 8766 ─────────────────────────────────────────────

def test_parse_setup_flag():
    """--setup and --setup-port are parsed correctly."""
    parser = build_parser()
    args = parser.parse_args(["--setup", "--setup-port", "8766"])
    assert args.setup is True
    assert args.setup_port == 8766


# ── 6. --channels config.json ────────────────────────────────────────────────

def test_parse_channels_arg():
    """--channels path is accepted by the parser."""
    parser = build_parser()
    args = parser.parse_args(["--channels", "config.json"])
    assert args.channels == "config.json"


# ── 7. GET /setup/status returns 200 JSON ────────────────────────────────────

def test_setup_handler_status_returns_json():
    """An HTTP GET to /setup/status must return 200 with JSON containing 'phase'."""
    state = _SetupState()
    srv, port = _start_server(state)
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request("GET", "/setup/status")
        resp = conn.getresponse()
        body = resp.read()
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(body)
        assert "phase" in data, f"'phase' missing: {data}"
        assert "channels" in data, f"'channels' missing: {data}"
    finally:
        srv.stop()


# ── 8. POST /setup/go_live sets phase to "live" ───────────────────────────────

def test_setup_handler_go_live_sets_phase():
    """POST /setup/go_live must transition the state phase to 'live'."""
    state = _SetupState()
    srv, port = _start_server(state)
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request("POST", "/setup/go_live", body=b"", headers={"Content-Length": "0"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        # Verify state was updated
        assert state.phase == "live", f"Expected phase='live', got '{state.phase}'"
    finally:
        srv.stop()
