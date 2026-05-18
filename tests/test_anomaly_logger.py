"""
Unit tests for core/anomaly_logger.py → AnomalyEventLogger

Tests cover: CSV header creation, event_count property, log_event() appending,
score_pct calculation, session-spanning row count, frame_path storage,
and zero-division guard when threshold is 0.
"""
import csv
import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger(path: str):
    from core.anomaly_logger import AnomalyEventLogger
    return AnomalyEventLogger(path)


def _read_data_rows(path: str):
    """Return all CSV data rows (excluding header) as dicts."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# File creation and header
# ---------------------------------------------------------------------------

class TestFileCreation:
    def test_creates_file_on_first_instantiation(self, tmp_path):
        log_path = str(tmp_path / "events.csv")
        _make_logger(log_path)
        assert os.path.isfile(log_path)

    def test_creates_parent_directories_if_needed(self, tmp_path):
        log_path = str(tmp_path / "deep" / "nested" / "events.csv")
        _make_logger(log_path)
        assert os.path.isfile(log_path)

    def test_new_file_contains_header_row(self, tmp_path):
        log_path = str(tmp_path / "events.csv")
        _make_logger(log_path)
        with open(log_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["timestamp", "score", "threshold", "score_pct", "frame_path"]

    def test_new_file_has_no_data_rows(self, tmp_path):
        log_path = str(tmp_path / "events.csv")
        _make_logger(log_path)
        rows = _read_data_rows(log_path)
        assert rows == []


# ---------------------------------------------------------------------------
# event_count property
# ---------------------------------------------------------------------------

class TestEventCount:
    def test_event_count_starts_at_zero_for_new_file(self, tmp_path):
        logger = _make_logger(str(tmp_path / "e.csv"))
        assert logger.event_count == 0

    def test_event_count_increments_after_log_event(self, tmp_path):
        logger = _make_logger(str(tmp_path / "e.csv"))
        logger.log_event(0.05, 0.10)
        assert logger.event_count == 1

    def test_event_count_increments_multiple_times(self, tmp_path):
        logger = _make_logger(str(tmp_path / "e.csv"))
        for _ in range(7):
            logger.log_event(0.02, 0.10)
        assert logger.event_count == 7

    def test_event_count_restored_from_existing_file(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger1 = _make_logger(log_path)
        for _ in range(5):
            logger1.log_event(0.03, 0.10)

        from core.anomaly_logger import AnomalyEventLogger
        logger2 = AnomalyEventLogger(log_path)
        assert logger2.event_count == 5

    def test_re_instantiation_continues_count_from_existing(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger1 = _make_logger(log_path)
        logger1.log_event(0.03, 0.10)
        logger1.log_event(0.04, 0.10)

        from core.anomaly_logger import AnomalyEventLogger
        logger2 = AnomalyEventLogger(log_path)
        logger2.log_event(0.05, 0.10)
        assert logger2.event_count == 3


# ---------------------------------------------------------------------------
# log_path property
# ---------------------------------------------------------------------------

class TestLogPath:
    def test_log_path_property_returns_correct_path(self, tmp_path):
        log_path = str(tmp_path / "mylog.csv")
        logger = _make_logger(log_path)
        assert logger.log_path == log_path


# ---------------------------------------------------------------------------
# log_event() — row content
# ---------------------------------------------------------------------------

class TestLogEvent:
    def test_log_event_appends_one_row(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(0.05, 0.10)
        rows = _read_data_rows(log_path)
        assert len(rows) == 1

    def test_multiple_log_events_produce_multiple_rows(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        for i in range(4):
            logger.log_event(0.01 * (i + 1), 0.10)
        rows = _read_data_rows(log_path)
        assert len(rows) == 4

    def test_score_stored_as_float_string(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(0.123456, 0.5)
        row = _read_data_rows(log_path)[0]
        assert abs(float(row["score"]) - 0.123456) < 1e-5

    def test_threshold_stored_correctly(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(0.05, 0.25)
        row = _read_data_rows(log_path)[0]
        assert abs(float(row["threshold"]) - 0.25) < 1e-5

    def test_frame_path_stored_correctly(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(0.05, 0.10, frame_path="/tmp/alarm_001.jpg")
        row = _read_data_rows(log_path)[0]
        assert row["frame_path"] == "/tmp/alarm_001.jpg"

    def test_frame_path_defaults_to_empty_string(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(0.05, 0.10)
        row = _read_data_rows(log_path)[0]
        assert row["frame_path"] == ""

    def test_timestamp_present_in_row(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(0.05, 0.10)
        row = _read_data_rows(log_path)[0]
        assert row["timestamp"] != ""


# ---------------------------------------------------------------------------
# score_pct calculation
# ---------------------------------------------------------------------------

class TestScorePct:
    def test_score_pct_is_integer_percentage_of_threshold(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        # score=0.05, threshold=0.10 → pct = int(0.05/0.10*100) = 50
        logger.log_event(0.05, 0.10)
        row = _read_data_rows(log_path)[0]
        assert int(row["score_pct"]) == 50

    def test_score_pct_above_100_when_score_exceeds_threshold(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        # score=0.20, threshold=0.10 → pct = 200
        logger.log_event(0.20, 0.10)
        row = _read_data_rows(log_path)[0]
        assert int(row["score_pct"]) == 200

    def test_score_pct_is_zero_when_threshold_is_zero(self, tmp_path):
        """No division by zero — should produce 0 when threshold=0."""
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(0.05, 0.0)
        row = _read_data_rows(log_path)[0]
        assert int(row["score_pct"]) == 0

    def test_score_pct_truncates_not_rounds(self, tmp_path):
        """int() truncates: 1/3 * 100 = 33, not 34."""
        log_path = str(tmp_path / "e.csv")
        logger = _make_logger(log_path)
        logger.log_event(1 / 3, 1.0)   # pct = int(33.33…) = 33
        row = _read_data_rows(log_path)[0]
        assert int(row["score_pct"]) == 33


# ---------------------------------------------------------------------------
# Existing file — no second header written
# ---------------------------------------------------------------------------

class TestExistingFile:
    def test_existing_file_does_not_get_second_header(self, tmp_path):
        log_path = str(tmp_path / "e.csv")
        logger1 = _make_logger(log_path)
        logger1.log_event(0.01, 0.10)

        from core.anomaly_logger import AnomalyEventLogger
        logger2 = AnomalyEventLogger(log_path)
        logger2.log_event(0.02, 0.10)

        # Only one header row → DictReader should produce exactly 2 data rows
        rows = _read_data_rows(log_path)
        assert len(rows) == 2
