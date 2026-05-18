"""
Unit tests for core/audit.py → AuditTrail

Tests cover: file creation, JSON-line format, all convenience methods,
multiple appends, timestamp format, details preservation, and directory
auto-creation.
"""
import json
import os
import tempfile
from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_lines(path: str):
    """Return all non-empty lines from a JSONL file as parsed dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_trail(tmp_dir: str, name: str = "TestProjekt"):
    from core.audit import AuditTrail
    return AuditTrail(tmp_dir, name)


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

class TestFileCreation:
    def test_creates_audit_jsonl_in_given_directory(self, tmp_path):
        from core.audit import AuditTrail
        trail = AuditTrail(str(tmp_path), "MyProject")
        trail.log("test_action")
        assert os.path.isfile(str(tmp_path / "audit.jsonl"))

    def test_creates_directory_if_not_exist(self, tmp_path):
        from core.audit import AuditTrail
        new_dir = str(tmp_path / "deeply" / "nested" / "dir")
        trail = AuditTrail(new_dir, "proj")
        trail.log("create_test")
        assert os.path.isfile(os.path.join(new_dir, "audit.jsonl"))

    def test_existing_directory_does_not_raise(self, tmp_path):
        from core.audit import AuditTrail
        # Should not raise even if directory already exists
        AuditTrail(str(tmp_path))
        AuditTrail(str(tmp_path))  # second instantiation


# ---------------------------------------------------------------------------
# JSON line format
# ---------------------------------------------------------------------------

class TestJsonLineFormat:
    def test_log_writes_valid_json_line(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("my_action")
        lines = _read_lines(str(tmp_path / "audit.jsonl"))
        assert len(lines) == 1

    def test_entry_has_required_keys(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("some_action", entity="label_x", details={"k": "v"})
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        for key in ("ts", "action", "entity", "details", "project"):
            assert key in entry, f"Missing key: {key}"

    def test_action_field_matches_argument(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("label_added", "red_label")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "label_added"

    def test_entity_field_matches_argument(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("roi_added", entity="image_001.jpg")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["entity"] == "image_001.jpg"

    def test_details_dict_is_preserved(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        details = {"color": "#FF0000", "count": 42, "flag": True}
        trail.log("label_added", details=details)
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["details"]["color"] == "#FF0000"
        assert entry["details"]["count"] == 42
        assert entry["details"]["flag"] is True

    def test_project_name_stored_in_entry(self, tmp_path):
        from core.audit import AuditTrail
        trail = AuditTrail(str(tmp_path), "MySpecialProject")
        trail.log("test")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["project"] == "MySpecialProject"

    def test_empty_entity_is_empty_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("no_entity_action")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["entity"] == ""

    def test_details_defaults_to_empty_dict(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("bare_action")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["details"] == {}


# ---------------------------------------------------------------------------
# Timestamp format
# ---------------------------------------------------------------------------

class TestTimestampFormat:
    def test_timestamp_is_iso8601(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("ts_test")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        ts = entry["ts"]
        # Should parse as a valid datetime without raising
        parsed = datetime.fromisoformat(ts)
        assert parsed.year >= 2024

    def test_timestamp_has_date_and_time_parts(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log("ts_test")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        ts = entry["ts"]
        # ISO 8601 contains a "T" separator
        assert "T" in ts or " " in ts


# ---------------------------------------------------------------------------
# Multiple appends
# ---------------------------------------------------------------------------

class TestMultipleAppends:
    def test_multiple_log_calls_produce_multiple_lines(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        for i in range(5):
            trail.log(f"action_{i}", entity=f"entity_{i}")
        lines = _read_lines(str(tmp_path / "audit.jsonl"))
        assert len(lines) == 5

    def test_lines_are_appended_in_order(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        actions = ["first", "second", "third"]
        for a in actions:
            trail.log(a)
        lines = _read_lines(str(tmp_path / "audit.jsonl"))
        for i, a in enumerate(actions):
            assert lines[i]["action"] == a

    def test_new_instance_appends_to_existing_file(self, tmp_path):
        from core.audit import AuditTrail
        trail1 = AuditTrail(str(tmp_path), "proj")
        trail1.log("first_session_action")

        trail2 = AuditTrail(str(tmp_path), "proj")
        trail2.log("second_session_action")

        lines = _read_lines(str(tmp_path / "audit.jsonl"))
        assert len(lines) == 2
        assert lines[0]["action"] == "first_session_action"
        assert lines[1]["action"] == "second_session_action"


# ---------------------------------------------------------------------------
# Convenience methods — correct action strings
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def test_log_label_added_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_label_added("cat", "#FF0000")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "label_added"
        assert entry["entity"] == "cat"
        assert entry["details"]["color"] == "#FF0000"

    def test_log_label_removed_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_label_removed("dog")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "label_removed"
        assert entry["entity"] == "dog"

    def test_log_label_renamed_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_label_renamed("old_name", "new_name")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "label_renamed"
        assert entry["entity"] == "old_name"

    def test_log_label_renamed_includes_new_name_in_details(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_label_renamed("alpha", "beta")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["details"].get("new_name") == "beta"

    def test_log_image_labeled_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_image_labeled("/data/images/cat001.jpg", "cat")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "image_labeled"
        # entity should be the basename only
        assert entry["entity"] == "cat001.jpg"
        assert entry["details"]["label"] == "cat"

    def test_log_roi_added_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_roi_added("/data/img.jpg", "roi_abc", "rect")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "roi_added"
        assert entry["entity"] == "img.jpg"
        assert entry["details"]["roi_id"] == "roi_abc"
        assert entry["details"]["type"] == "rect"

    def test_log_roi_deleted_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_roi_deleted("/data/img.jpg", "roi_xyz")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "roi_deleted"
        assert entry["details"]["roi_id"] == "roi_xyz"

    def test_log_training_started_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        config = {"epochs": 10, "lr": 0.001}
        trail.log_training_started("run_42", config)
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "training_started"
        assert entry["entity"] == "run_42"
        assert entry["details"]["config"] == config

    def test_log_training_finished_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        metrics = {"accuracy": 0.95}
        trail.log_training_finished("run_42", metrics)
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "training_finished"
        assert entry["details"]["metrics"] == metrics

    def test_log_project_saved_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_project_saved("/some/path/project.json")
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "project_saved"

    def test_log_inference_action_string(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        trail.log_inference("/data/test_set", "resnet18", 200)
        entry = _read_lines(str(tmp_path / "audit.jsonl"))[0]
        assert entry["action"] == "inference_run"
        assert entry["details"]["image_count"] == 200


# ---------------------------------------------------------------------------
# get_entries
# ---------------------------------------------------------------------------

class TestGetEntries:
    def test_get_entries_returns_empty_list_for_nonexistent_file(self, tmp_path):
        from core.audit import AuditTrail
        trail = AuditTrail(str(tmp_path), "proj")
        # File doesn't exist yet (no log call)
        entries = trail.get_entries()
        assert entries == []

    def test_get_entries_returns_all_logged_entries(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        for i in range(10):
            trail.log(f"action_{i}")
        entries = trail.get_entries(limit=200)
        assert len(entries) == 10

    def test_get_entries_limit_respected(self, tmp_path):
        trail = _make_trail(str(tmp_path))
        for i in range(20):
            trail.log(f"action_{i}")
        entries = trail.get_entries(limit=5)
        assert len(entries) == 5
