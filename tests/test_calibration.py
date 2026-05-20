"""Tests für core/calibration.py"""
from __future__ import annotations
import json, os, sys, pytest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import TemperatureScaler


def test_temperature_scaler_init():
    ts = TemperatureScaler()
    assert ts.temperature == 1.0


def test_is_fitted_false_initially():
    ts = TemperatureScaler()
    assert not ts.is_fitted


def test_fit_changes_temperature():
    ts = TemperatureScaler()
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((50, 3))
    labels = rng.integers(0, 3, size=50)
    T = ts.fit(logits, labels)
    assert isinstance(T, float)
    assert T > 0.0


def test_is_fitted_true_after_fit():
    ts = TemperatureScaler()
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((20, 2)) * 5   # overconfident
    labels = rng.integers(0, 2, size=20)
    ts.fit(logits, labels)
    assert ts.is_fitted


def test_calibrate_preserves_shape():
    ts = TemperatureScaler()
    ts.temperature = 1.5
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    out = ts.calibrate(probs)
    assert out.shape == probs.shape


def test_calibrate_probabilities_sum_to_one():
    ts = TemperatureScaler()
    ts.temperature = 2.0
    rng = np.random.default_rng(7)
    raw = rng.dirichlet([1, 1, 1], size=10)
    out = ts.calibrate(raw)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(10), atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    ts = TemperatureScaler()
    ts.temperature = 2.718
    path = str(tmp_path / "cal.json")
    ts.save(path)
    ts2 = TemperatureScaler()
    ts2.load(path)
    assert abs(ts2.temperature - 2.718) < 1e-6


def test_save_load_json_format(tmp_path):
    ts = TemperatureScaler()
    ts.temperature = 1.234
    path = str(tmp_path / "cal.json")
    ts.save(path)
    with open(path) as f:
        data = json.load(f)
    assert "temperature" in data
    assert abs(data["temperature"] - 1.234) < 1e-6
