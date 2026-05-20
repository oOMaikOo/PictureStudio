"""Tests für core/hyperparameter_tuning.py"""
from __future__ import annotations
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hyperparameter_tuning import HPTWorker, HPTThread


def test_hpt_worker_instantiates():
    """HPTWorker() mit project=None darf nicht crashen."""
    w = HPTWorker(project=None, n_trials=1, timeout=10)
    assert w is not None


def test_hpt_worker_no_optuna_raises_import_error(monkeypatch):
    """Wenn optuna fehlt, muss HPTWorker.run() ImportError werfen."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("optuna not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    w = HPTWorker(project=None, n_trials=1)
    with pytest.raises(ImportError):
        w.run()


def test_hpt_thread_instantiates(qtbot):
    """HPTThread() muss ohne Crash instanziierbar sein."""
    t = HPTThread(project=None, n_trials=1)
    assert t is not None


def test_hpt_thread_has_signals(qtbot):
    """HPTThread muss progress, finished und error Signale haben."""
    t = HPTThread(project=None, n_trials=1)
    assert hasattr(t, "progress")
    assert hasattr(t, "finished")
    assert hasattr(t, "error")
