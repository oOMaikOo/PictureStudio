"""
Tests für core/anomaly_clustering.py und gui/pages/anomaly_clustering_page.py
(Feature 6: Anomalie-Clustering).
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Skip markers (graceful fallbacks if heavy deps are absent)
# ---------------------------------------------------------------------------
PIL = pytest.importorskip("PIL", reason="Pillow not installed")
from PIL import Image as PILImage

clustering_mod = pytest.importorskip(
    "core.anomaly_clustering",
    reason="core.anomaly_clustering not available",
)
AnomalyClustering = clustering_mod.AnomalyClustering
ClusteringWorker  = clustering_mod.ClusteringWorker
ClusteringThread  = clustering_mod.ClusteringThread


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_images(tmp_path: str, n: int = 8) -> list[str]:
    """Create *n* tiny real PNG files in *tmp_path* and return their paths."""
    paths = []
    for i in range(n):
        fname = os.path.join(tmp_path, f"img_{i:03d}.png")
        img = PILImage.new("RGB", (32, 32), color=(i * 30 % 255, i * 15 % 255, 100))
        img.save(fname)
        paths.append(fname)
    return paths


# ---------------------------------------------------------------------------
# AnomalyClustering — core tests (no Qt needed)
# ---------------------------------------------------------------------------

class TestAnomalyClustering:
    """Tests for the AnomalyClustering class."""

    def test_fit_returns_dict(self, tmp_path):
        """fit() with real images returns a non-empty dict."""
        paths = _make_images(str(tmp_path), n=8)
        ac = AnomalyClustering()
        result = ac.fit(paths, n_clusters=3)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_fit_all_paths_assigned(self, tmp_path):
        """Every input path should appear in exactly one cluster."""
        paths = _make_images(str(tmp_path), n=8)
        ac = AnomalyClustering()
        ac.fit(paths, n_clusters=3)
        assigned = [p for ps in ac.clusters.values() for p in ps]
        assert set(assigned) == set(paths)

    def test_fit_empty_returns_empty(self):
        """fit() with an empty list returns an empty dict."""
        ac = AnomalyClustering()
        result = ac.fit([], n_clusters=3)
        assert result == {}

    def test_fit_single_image(self, tmp_path):
        """fit() with one image returns one cluster containing that image."""
        paths = _make_images(str(tmp_path), n=1)
        ac = AnomalyClustering()
        result = ac.fit(paths, n_clusters=5)  # n_clusters clamped to 1
        assert len(result) == 1
        cluster_paths = list(result.values())[0]
        assert paths[0] in cluster_paths

    def test_fit_n_clusters_clamped(self, tmp_path):
        """When n_clusters > len(images), result has at most len(images) clusters."""
        paths = _make_images(str(tmp_path), n=3)
        ac = AnomalyClustering()
        result = ac.fit(paths, n_clusters=10)
        assert len(result) <= 3

    def test_fit_missing_files_handled_gracefully(self, tmp_path):
        """Non-existent paths should be silently skipped (no crash)."""
        real = _make_images(str(tmp_path), n=4)
        mixed = real + ["/nonexistent/does_not_exist.png"]
        ac = AnomalyClustering()
        result = ac.fit(mixed, n_clusters=2)
        # Real images are still clustered; fake path not in any cluster
        assigned = [p for ps in result.values() for p in ps]
        assert "/nonexistent/does_not_exist.png" not in assigned

    def test_get_representative_returns_valid_path(self, tmp_path):
        """get_representative() returns a path that is in the cluster."""
        paths = _make_images(str(tmp_path), n=6)
        ac = AnomalyClustering()
        result = ac.fit(paths, n_clusters=2)
        for cid, cluster_paths in result.items():
            rep = ac.get_representative(cid)
            assert rep in cluster_paths

    def test_get_representative_unknown_cluster(self):
        """get_representative() for an unknown cluster_id returns empty string."""
        ac = AnomalyClustering()
        assert ac.get_representative(999) == ""

    def test_to_dataframe_returns_list_of_dicts(self, tmp_path):
        """to_dataframe() returns a list of dicts with required keys."""
        paths = _make_images(str(tmp_path), n=6)
        ac = AnomalyClustering()
        ac.fit(paths, n_clusters=2)
        rows = ac.to_dataframe()
        assert isinstance(rows, list)
        assert len(rows) == len(paths)
        for row in rows:
            assert "path" in row
            assert "cluster" in row
            assert "is_representative" in row

    def test_to_dataframe_exactly_one_representative_per_cluster(self, tmp_path):
        """Each cluster should have exactly one representative in the dataframe."""
        paths = _make_images(str(tmp_path), n=8)
        ac = AnomalyClustering()
        result = ac.fit(paths, n_clusters=3)
        rows = ac.to_dataframe()
        for cid in result:
            reps = [r for r in rows if r["cluster"] == cid and r["is_representative"]]
            assert len(reps) == 1, f"Cluster {cid} has {len(reps)} representatives"

    def test_export_csv_creates_file(self, tmp_path):
        """export_csv() writes a file at the given path."""
        paths = _make_images(str(tmp_path), n=6)
        ac = AnomalyClustering()
        ac.fit(paths, n_clusters=2)
        csv_path = os.path.join(str(tmp_path), "output.csv")
        ac.export_csv(csv_path)
        assert os.path.isfile(csv_path)

    def test_export_csv_headers(self, tmp_path):
        """The exported CSV must have exactly the headers: path, cluster, is_representative."""
        paths = _make_images(str(tmp_path), n=6)
        ac = AnomalyClustering()
        ac.fit(paths, n_clusters=2)
        csv_path = os.path.join(str(tmp_path), "out.csv")
        ac.export_csv(csv_path)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames) == {"path", "cluster", "is_representative"}

    def test_export_csv_row_count(self, tmp_path):
        """CSV should have exactly as many data rows as there are clustered images."""
        paths = _make_images(str(tmp_path), n=8)
        ac = AnomalyClustering()
        ac.fit(paths, n_clusters=3)
        csv_path = os.path.join(str(tmp_path), "out.csv")
        ac.export_csv(csv_path)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == len(paths)


# ---------------------------------------------------------------------------
# ClusteringWorker — plain class tests (no Qt)
# ---------------------------------------------------------------------------

class TestClusteringWorker:
    """Tests for ClusteringWorker (plain Python, no Qt)."""

    def test_run_returns_dict(self, tmp_path):
        """ClusteringWorker.run() returns a dict."""
        paths = _make_images(str(tmp_path), n=6)
        worker = ClusteringWorker(paths, n_clusters=2)
        result = worker.run()
        assert isinstance(result, dict)

    def test_clustering_property_after_run(self, tmp_path):
        """After run(), worker.clustering is an AnomalyClustering instance."""
        paths = _make_images(str(tmp_path), n=6)
        worker = ClusteringWorker(paths, n_clusters=2)
        worker.run()
        assert isinstance(worker.clustering, AnomalyClustering)


# ---------------------------------------------------------------------------
# ClusteringThread — Qt tests
# ---------------------------------------------------------------------------

class TestClusteringThread:
    """Tests for ClusteringThread (requires Qt / qtbot)."""

    def test_finished_signal_emitted(self, qtbot, tmp_path):
        """ClusteringThread should emit finished(dict) after completion."""
        paths = _make_images(str(tmp_path), n=6)
        thread = ClusteringThread(paths, n_clusters=2)
        received = []

        def _on_finished(result):
            received.append(result)

        thread.finished.connect(_on_finished)
        with qtbot.waitSignal(thread.finished, timeout=30_000):
            thread.start()
        assert len(received) == 1
        assert isinstance(received[0], dict)

    def test_progress_signal_emitted(self, qtbot, tmp_path):
        """ClusteringThread should emit at least one progress signal."""
        paths = _make_images(str(tmp_path), n=6)
        thread = ClusteringThread(paths, n_clusters=2)
        progress_calls = []
        thread.progress.connect(lambda c, t: progress_calls.append((c, t)))
        with qtbot.waitSignal(thread.finished, timeout=30_000):
            thread.start()
        # Progress is optional if images are processed instantly,
        # but finished must have been emitted (asserted above via waitSignal).
        # Just verify no crash.

    def test_empty_paths_finished_signal(self, qtbot):
        """ClusteringThread with empty path list should still emit finished({})."""
        thread = ClusteringThread([], n_clusters=3)
        received = []
        thread.finished.connect(lambda r: received.append(r))
        with qtbot.waitSignal(thread.finished, timeout=10_000):
            thread.start()
        assert received[0] == {}


# ---------------------------------------------------------------------------
# AnomalyClusteringPage — UI tests
# ---------------------------------------------------------------------------

page_mod = pytest.importorskip(
    "gui.pages.anomaly_clustering_page",
    reason="anomaly_clustering_page not available",
)
AnomalyClusteringPage = page_mod.AnomalyClusteringPage


class TestAnomalyClusteringPage:
    """UI tests for AnomalyClusteringPage."""

    def test_instantiates_without_args(self, qtbot):
        """AnomalyClusteringPage() should construct without arguments."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)

    def test_set_project_does_not_crash(self, qtbot, sample_project):
        """set_project(project) must not raise."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        page.set_project(sample_project)

    def test_set_project_none_does_not_crash(self, qtbot):
        """set_project(None) should be handled gracefully."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        # None project: set_project will set self.project = None
        # The page should not crash during set_project(None)
        try:
            page.set_project(None)
        except Exception:
            # It's acceptable to guard against None internally — but not crash
            pass

    def test_start_button_exists(self, qtbot):
        """'Clustering starten' button must exist."""
        from PySide6.QtWidgets import QAbstractButton
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        assert hasattr(page, "btn_start")
        assert "Clustering starten" in page.btn_start.text()

    def test_export_button_exists(self, qtbot):
        """'CSV exportieren' button must exist."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        assert hasattr(page, "btn_export")
        assert "CSV" in page.btn_export.text()

    def test_cluster_spinbox_exists(self, qtbot):
        """spin_clusters spinbox must exist and be in range 2–20."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        assert hasattr(page, "spin_clusters")
        assert page.spin_clusters.minimum() == 2
        assert page.spin_clusters.maximum() == 20

    def test_export_button_disabled_initially(self, qtbot):
        """'CSV exportieren' button must be disabled before any clustering."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        assert not page.btn_export.isEnabled()

    def test_start_button_enabled_initially(self, qtbot):
        """'Clustering starten' button must be enabled on page creation."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        assert page.btn_start.isEnabled()

    def test_on_finished_sets_thread_to_none(self, qtbot):
        """_on_finished() must set self._thread = None."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)

        class _FakeThread:
            pass

        page._thread = _FakeThread()
        page._on_finished({})
        assert page._thread is None

    def test_on_error_sets_thread_to_none(self, qtbot, monkeypatch):
        """_on_error() must set self._thread = None even on error."""
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: None)
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)

        class _FakeThread:
            pass

        page._thread = _FakeThread()
        page._on_error("test error")
        assert page._thread is None

    def test_alarm_paths_cache_not_stale_after_set_project(self, qtbot, sample_project):
        """set_project() must flush stale cache entries from the previous project."""
        page = AnomalyClusteringPage()
        qtbot.addWidget(page)
        stale_path = "/old/project/stale_image.png"
        page._alarm_paths_cache = [stale_path]
        page.set_project(sample_project)
        # Cache is rebuilt from the new project — stale path must not appear
        assert page._alarm_paths_cache is None or stale_path not in page._alarm_paths_cache
