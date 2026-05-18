"""
Save/load monitoring profiles as JSON.

A profile captures everything needed to run the headless monitor daemon
without the GUI: model path, camera source, thresholds, MQTT config, etc.
"""
import json
import os
from typing import Optional


PROFILE_VERSION = 1


def default_profile() -> dict:
    """Return a monitoring profile with safe defaults for all fields."""
    return {
        "version": PROFILE_VERSION,
        "model_path": "",
        "model_format": "pytorch",   # "pytorch" | "onnx"
        "threshold": 0.02,
        "camera_source": 0,          # int (USB index) or str (RTSP/HTTP URL)
        "save_dir": "",
        "smooth_n": 5,
        "roi": None,                 # null or [x1, y1, x2, y2] normalised 0–1
        "mqtt": {
            "enabled": False,
            "host": "localhost",
            "port": 1883,
            "topic": "picture_studio/anomaly",
            "username": "",
            "password": "",
        },
        "scoring_interval": 3,       # score every N-th frame
        "save_anomalies": True,
    }


def save_profile(profile: dict, path: str) -> None:
    """Write profile JSON atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_profile(path: str) -> dict:
    """Load profile JSON; raises ValueError on version mismatch."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ver = data.get("version", 0)
    if ver != PROFILE_VERSION:
        raise ValueError(f"Unbekannte Profil-Version {ver} (erwartet {PROFILE_VERSION})")
    base = default_profile()
    base.update(data)
    return base
