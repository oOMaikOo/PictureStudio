"""
PictureStudio Monitor — standalone anomaly-detection daemon.

Split from a single 2 482-line module into focused sub-modules:

    monitor.camera        — camera discovery, terminal select, capture thread
    monitor.state         — shared state + JPEG frame ring-buffer
    monitor.imaging       — ROI crop, overlay, HUD, alarm saving
    monitor.web           — load HTML frontends from monitor_web/
    monitor.api_server    — REST API (on core.http_router)
    monitor.setup_server  — multi-channel Setup-Wizard (on core.http_router)
    monitor.runner        — single- and multi-channel run loops
    monitor.cli           — argument parser + main entry point

The public names below are re-exported so ``from monitor import X`` keeps
working for existing tooling and tests. Run with ``python monitor.py``.
"""
from __future__ import annotations

# Imported in dependency order so sub-modules resolve their siblings cleanly.
from monitor.state import _MonitorState
from monitor.imaging import apply_roi, composite_overlay, draw_hud, save_alarm
from monitor.camera import (
    _CameraThread,
    _VIDEO_EXTENSIONS,
    _discover_cameras,
    _terminal_camera_select,
    find_camera_index,
)
from monitor.api_server import MonitorApiServer, _MonitorApiServer
from monitor.setup_server import (
    SetupApiServer,
    _SetupApiServer,
    _SetupChannel,
    _SetupState,
    run_setup,
)
from monitor.runner import run_monitor, run_monitor_multi
from monitor.cli import build_parser, main

__all__ = [
    "_MonitorState",
    "apply_roi", "composite_overlay", "draw_hud", "save_alarm",
    "_CameraThread", "_VIDEO_EXTENSIONS",
    "_discover_cameras", "_terminal_camera_select", "find_camera_index",
    "MonitorApiServer", "_MonitorApiServer",
    "SetupApiServer", "_SetupApiServer",
    "_SetupChannel", "_SetupState", "run_setup",
    "run_monitor", "run_monitor_multi",
    "build_parser", "main",
]
