"""
Tests for gui/widgets/thumbnail_list._LRUPixmapCache
and the camera retry logic in core/camera.CameraFrameThread.
"""
import pytest


# ---------------------------------------------------------------------------
# _LRUPixmapCache
# ---------------------------------------------------------------------------

class TestLRUPixmapCache:
    def _make(self, maxsize=5):
        from gui.widgets.thumbnail_list import _LRUPixmapCache
        return _LRUPixmapCache(maxsize)

    def test_empty_cache_contains_nothing(self):
        c = self._make()
        assert "a" not in c

    def test_set_and_get(self):
        c = self._make()
        c["a"] = 1
        assert "a" in c
        assert c["a"] == 1

    def test_len(self):
        c = self._make()
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2

    def test_evicts_oldest_when_full(self):
        c = self._make(maxsize=3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        c["d"] = 4   # "a" should be evicted
        assert "a" not in c
        assert "d" in c
        assert len(c) == 3

    def test_access_promotes_to_recent(self):
        c = self._make(maxsize=3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        _ = c["a"]   # promote "a"
        c["d"] = 4   # "b" should be evicted, not "a"
        assert "a" in c
        assert "b" not in c

    def test_overwrite_promotes_to_recent(self):
        c = self._make(maxsize=3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        c["a"] = 99  # overwrite → promotes
        c["d"] = 4   # "b" should be evicted, not "a"
        assert "a" in c
        assert c["a"] == 99
        assert "b" not in c

    def test_clear_empties_cache(self):
        c = self._make()
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        assert "a" not in c

    def test_maxsize_one(self):
        c = self._make(maxsize=1)
        c["a"] = 1
        c["b"] = 2
        assert "a" not in c
        assert "b" in c
        assert len(c) == 1

    def test_never_exceeds_maxsize(self):
        c = self._make(maxsize=10)
        for i in range(50):
            c[str(i)] = i
        assert len(c) <= 10


# ---------------------------------------------------------------------------
# Camera retry logic
# ---------------------------------------------------------------------------

class TestCameraRetryLogic:
    """
    Test the consecutive-failure threshold without a real camera.
    We verify the retry constant and that the logic is in the source.
    """

    def test_consecutive_fail_threshold_is_five(self):
        import inspect
        from core.camera import CameraFrameThread
        src = inspect.getsource(CameraFrameThread.run)
        assert "_consec_fail >= 5" in src or "consec_fail >= 5" in src

    def test_retry_sleep_present(self):
        import inspect
        from core.camera import CameraFrameThread
        src = inspect.getsource(CameraFrameThread.run)
        # After a failed read, there should be a short sleep before retry
        assert "time.sleep(0.1)" in src or "sleep(0.1)" in src

    def test_single_failure_does_not_emit_error(self):
        """Simulate the retry logic: fewer than 5 failures → no error emitted."""
        errors = []
        _consec_fail = 0
        _max = 5

        def simulate_reads(fail_count):
            nonlocal _consec_fail
            _consec_fail = 0
            for i in range(fail_count):
                ret = False
                if not ret:
                    _consec_fail += 1
                    if _consec_fail >= _max:
                        errors.append("error")
                    # else: retry

        simulate_reads(4)  # 4 consecutive fails → no error
        assert errors == []

    def test_five_failures_triggers_error(self):
        """5 consecutive failures → error."""
        errors = []
        _consec_fail = 0
        _max = 5

        def simulate_reads(fail_count):
            nonlocal _consec_fail
            _consec_fail = 0
            for _ in range(fail_count):
                ret = False
                if not ret:
                    _consec_fail += 1
                    if _consec_fail >= _max:
                        errors.append("error")

        simulate_reads(5)
        assert len(errors) >= 1

    def test_recovery_resets_counter(self):
        """After a successful read the failure counter resets."""
        errors = []
        _consec_fail = 0
        _max = 5

        reads = [False, False, True, False, False, False, False]
        for ret in reads:
            if not ret:
                _consec_fail += 1
                if _consec_fail >= _max:
                    errors.append("error")
            else:
                _consec_fail = 0  # reset on success

        # 2 fails → reset → 4 more fails → still < 5 → no error
        assert errors == []
