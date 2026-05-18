"""
Anomaly event logger: appends CSV rows for each triggered alarm.
Columns: timestamp, score, threshold, score_pct, frame_path
"""
import csv
import os
from datetime import datetime


class AnomalyEventLogger:
    """
    Append-only CSV logger for anomaly alarm events.

    Creates the CSV file (with header row) on first instantiation.
    When the file already exists, counts pre-existing rows so that
    event_count reflects the total across sessions.
    Used by CameraPage and the REST API server.
    """

    _HEADERS = ["timestamp", "score", "threshold", "score_pct", "frame_path"]

    def __init__(self, log_path: str):
        """
        Initialise logger writing to *log_path*.

        Creates parent directories as needed. Writes the CSV header on a new
        file, or reads the existing row count for a pre-existing file.
        """
        self._log_path = log_path
        self._count = 0
        dir_ = os.path.dirname(log_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._HEADERS)
        else:
            # Count data rows (subtract 1 for the header)
            with open(log_path, "r", encoding="utf-8") as f:
                self._count = max(0, sum(1 for _ in f) - 1)

    @property
    def log_path(self) -> str:
        """Absolute path to the CSV log file."""
        return self._log_path

    @property
    def event_count(self) -> int:
        """Total number of events logged in this file (across all sessions)."""
        return self._count

    def log_event(self, score: float, threshold: float, frame_path: str = "") -> None:
        """
        Append one anomaly event row to the CSV log.

        Computes *score_pct* as (score / threshold * 100) rounded to an integer.
        Increments the internal event counter.
        """
        ts = datetime.now().isoformat(timespec="milliseconds")
        pct = int(score / threshold * 100) if threshold > 0 else 0
        with open(self._log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [ts, f"{score:.6f}", f"{threshold:.6f}", pct, frame_path]
            )
        self._count += 1
