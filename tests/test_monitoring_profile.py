"""
Tests for core/monitoring_profile.py — save_profile, load_profile, default_profile
"""
import json
import os

import pytest

from core.monitoring_profile import (
    default_profile,
    save_profile,
    load_profile,
    PROFILE_VERSION,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_profile_has_required_keys():
    """default_profile() returns a dict with all expected top-level keys."""
    profile = default_profile()

    required_keys = {
        "version",
        "model_path",
        "model_format",
        "threshold",
        "camera_source",
        "save_dir",
        "smooth_n",
        "roi",
        "mqtt",
        "scoring_interval",
        "save_anomalies",
    }
    for key in required_keys:
        assert key in profile, f"Key '{key}' missing from default_profile()"


def test_profile_version():
    """version field in default_profile() matches PROFILE_VERSION constant."""
    profile = default_profile()
    assert profile["version"] == PROFILE_VERSION


def test_default_profile_mqtt_subkeys():
    """The nested 'mqtt' dict contains all expected sub-keys."""
    mqtt = default_profile()["mqtt"]
    for key in ("enabled", "host", "port", "topic", "username", "password"):
        assert key in mqtt, f"MQTT key '{key}' missing"


def test_save_load_roundtrip(tmp_path):
    """save_profile then load_profile produces an equal dict."""
    profile = default_profile()
    profile["model_path"] = "/tmp/model.pth"
    profile["threshold"] = 0.05

    path = str(tmp_path / "test.profile.json")
    save_profile(profile, path)
    loaded = load_profile(path)

    # Core values must survive the roundtrip
    assert loaded["model_path"] == "/tmp/model.pth"
    assert loaded["threshold"] == 0.05
    assert loaded["version"] == PROFILE_VERSION


def test_save_creates_valid_json(tmp_path):
    """File written by save_profile is valid, parseable JSON."""
    path = str(tmp_path / "profile.json")
    save_profile(default_profile(), path)

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)  # must not raise

    assert isinstance(data, dict)


def test_load_missing_file_raises(tmp_path):
    """Loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_profile(str(tmp_path / "does_not_exist.json"))


def test_load_wrong_version_raises(tmp_path):
    """load_profile raises ValueError when the version field doesn't match."""
    bad_profile = default_profile()
    bad_profile["version"] = PROFILE_VERSION + 99
    path = str(tmp_path / "bad_version.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bad_profile, fh)

    with pytest.raises(ValueError, match="Version"):
        load_profile(path)
