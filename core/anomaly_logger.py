"""
Anomaly event logger: appends CSV rows for each triggered alarm.
Columns: timestamp, score, threshold, score_pct, frame_path
"""
import csv
import os
from datetime import datetime


class AnomalyEventLogger:
    """Append-only CSV logger for anomaly alarm events."""

    _HEADERS = ["timestamp", "score", "threshold", "score_pct", "frame_path"]

    def __init__(self, log_path: str):
        self._log_path = log_path
        self._count = 0
        dir_ = os.path.dirname(log_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._HEADERS)
        else:
            with open(log_path, "r", encoding="utf-8") as f:
                self._count = max(0, sum(1 for _ in f) - 1)

    @property
    def log_path(self) -> str:
        return self._log_path

    @property
    def event_count(self) -> int:
        return self._count

    def log_event(self, score: float, threshold: float, frame_path: str = "") -> None:
        ts = datetime.now().isoformat(timespec="milliseconds")
        pct = int(score / threshold * 100) if threshold > 0 else 0
        with open(self._log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [ts, f"{score:.6f}", f"{threshold:.6f}", pct, frame_path]
            )
        self._count += 1
