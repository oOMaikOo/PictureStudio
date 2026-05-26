"""
SQLite-backed search index for fast project queries.

The JSON file remains the authoritative source of truth. This index is an
in-memory SQLite database built from a loaded Project, giving O(1) SQL queries
for label counts, filtering, and unlabeled images without Python loops.

Usage:
    idx = ProjectSearchIndex()
    idx.rebuild(project)
    counts = idx.get_label_counts()          # {"gut": 120, "schlecht": 45}
    paths  = idx.get_images_by_label("gut")  # ["/path/img1.jpg", ...]
    unlabeled = idx.get_unlabeled()
"""
import sqlite3
import threading
from typing import Optional


class ProjectSearchIndex:
    """In-memory SQLite index rebuilt from a Project instance on demand."""

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._dirty = True
        self._lock = threading.Lock()   # guards _conn replacement in rebuild()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def rebuild(self, project) -> None:
        """(Re-)build the index from *project*. Call after every save/load."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("""
            CREATE TABLE images (
                path     TEXT PRIMARY KEY,
                label    TEXT,           -- primary / first label (or NULL)
                labeled  INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE multi_labels (
                image_path TEXT NOT NULL,
                label      TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX idx_label      ON images(label)")
        conn.execute("CREATE INDEX idx_labeled    ON images(labeled)")
        conn.execute("CREATE INDEX idx_multi_path ON multi_labels(image_path)")
        conn.execute("CREATE INDEX idx_multi_lbl  ON multi_labels(label)")

        is_multi = getattr(project.config, "multi_label", False)
        rows = []
        multi_rows = []
        for path in project.images:
            if is_multi:
                lbls = project.image_multi_labels.get(path, [])
                primary = lbls[0] if lbls else None
                labeled = 1 if lbls else 0
                for lbl in lbls:
                    multi_rows.append((path, lbl))
            else:
                primary = project.image_labels.get(path)
                labeled = 1 if primary else 0
            rows.append((path, primary, labeled))

        conn.executemany("INSERT INTO images VALUES (?,?,?)", rows)
        if multi_rows:
            conn.executemany("INSERT INTO multi_labels VALUES (?,?)", multi_rows)
        conn.commit()

        with self._lock:
            old = self._conn
            self._conn = conn
            self._dirty = False
        if old:
            old.close()

    def invalidate(self) -> None:
        """Mark index as stale (call after project mutations)."""
        with self._lock:
            self._dirty = True

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._conn is not None and not self._dirty

    # ── queries ────────────────────────────────────────────────────────────────

    def get_label_counts(self) -> dict:
        """Return {label: count} for all labeled images (excludes unlabeled)."""
        with self._lock:
            if self._conn is None or self._dirty:
                return {}
            rows = self._conn.execute(
                "SELECT label, COUNT(*) FROM images WHERE labeled=1 GROUP BY label"
            ).fetchall()
        return {row[0]: row[1] for row in rows if row[0]}

    def get_images_by_label(self, label: str) -> list:
        """Return all image paths with the given primary label."""
        with self._lock:
            if self._conn is None or self._dirty:
                return []
            return [
                r[0] for r in self._conn.execute(
                    "SELECT path FROM images WHERE label=?", (label,)
                ).fetchall()
            ]

    def get_unlabeled(self) -> list:
        """Return all image paths that have no label assigned."""
        with self._lock:
            if self._conn is None or self._dirty:
                return []
            return [
                r[0] for r in self._conn.execute(
                    "SELECT path FROM images WHERE labeled=0"
                ).fetchall()
            ]

    def get_labeled_count(self) -> int:
        with self._lock:
            if self._conn is None or self._dirty:
                return 0
            return self._conn.execute(
                "SELECT COUNT(*) FROM images WHERE labeled=1"
            ).fetchone()[0]

    def get_images_with_any_label(self, labels: list) -> list:
        """Return image paths that have at least one of the given labels."""
        with self._lock:
            if self._conn is None or self._dirty or not labels:
                return []
            placeholders = ",".join("?" * len(labels))
            multi = [
                r[0] for r in self._conn.execute(
                    f"SELECT DISTINCT image_path FROM multi_labels WHERE label IN ({placeholders})",
                    labels,
                ).fetchall()
            ]
            if multi:
                return multi
            return [
                r[0] for r in self._conn.execute(
                    f"SELECT path FROM images WHERE label IN ({placeholders})",
                    labels,
                ).fetchall()
            ]

    def search_paths(self, query: str) -> list:
        """Return image paths whose filename contains *query* (case-insensitive)."""
        with self._lock:
            if self._conn is None or self._dirty:
                return []
            q = f"%{query.lower()}%"
            return [
                r[0] for r in self._conn.execute(
                    "SELECT path FROM images WHERE LOWER(path) LIKE ?", (q,)
                ).fetchall()
            ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
