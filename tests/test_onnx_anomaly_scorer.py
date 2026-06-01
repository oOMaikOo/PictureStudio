"""
Tests for core/onnx_anomaly_scorer.py — OnnxAnomalyScorer

All tests skip when onnxruntime is not installed.
"""
import os
import json

import numpy as np
import pytest

# Skip entire module if onnxruntime is not available
ort = pytest.importorskip("onnxruntime")


from core.onnx_anomaly_scorer import OnnxAnomalyScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_onnx(path: str) -> None:
    """
    Write a tiny valid ONNX autoencoder model to *path*.
    Input/output: float32 (1, 3, 128, 128).
    The model is the identity function (input == output) so MSE == 0.
    """
    import onnx
    from onnx import helper, TensorProto

    # Graph: identity (output = input)
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 128, 128])
    identity_node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([identity_node], "test_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    with open(path, "wb") as fh:
        fh.write(model.SerializeToString())


def _skip_if_no_onnx():
    """Skip test if the onnx package (model builder) is not installed."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        pytest.skip("onnx package not installed — cannot build test model")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_import():
    """OnnxAnomalyScorer is importable from core.onnx_anomaly_scorer."""
    assert OnnxAnomalyScorer is not None


def test_init_missing_file_raises(tmp_path):
    """Passing a nonexistent .onnx path raises an exception on __init__."""
    nonexistent = str(tmp_path / "no_model.onnx")
    with pytest.raises(Exception):  # ort raises InvalidGraph or similar
        OnnxAnomalyScorer(nonexistent)


def test_threshold_property(tmp_path):
    """threshold property getter/setter work correctly."""
    _skip_if_no_onnx()
    onnx_path = str(tmp_path / "model.onnx")
    _make_minimal_onnx(onnx_path)

    scorer = OnnxAnomalyScorer(onnx_path, threshold=0.05)
    assert scorer.threshold == pytest.approx(0.05)

    scorer.threshold = 0.1
    assert scorer.threshold == pytest.approx(0.1)

    # Minimum clamp: value below 1e-6 should be set to 1e-6
    scorer.threshold = 0.0
    assert scorer.threshold >= 1e-6


def test_score_returns_float(tmp_path):
    """score() returns a non-negative float for a random frame."""
    _skip_if_no_onnx()
    onnx_path = str(tmp_path / "model.onnx")
    _make_minimal_onnx(onnx_path)

    scorer = OnnxAnomalyScorer(onnx_path)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = scorer.score(frame)

    assert isinstance(result, float)
    assert result >= 0.0


def test_is_anomaly_respects_threshold(tmp_path):
    """is_anomaly() returns (score, bool) and bool reflects threshold comparison."""
    _skip_if_no_onnx()
    onnx_path = str(tmp_path / "model.onnx")
    _make_minimal_onnx(onnx_path)

    # With identity model, score is ~0, so a high threshold means not anomaly
    scorer = OnnxAnomalyScorer(onnx_path, threshold=999.0)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    score, is_anom = scorer.is_anomaly(frame)

    assert isinstance(score, float)
    assert is_anom is False  # score near 0 << threshold 999


def test_from_path_reads_meta_sidecar(tmp_path):
    """from_path() picks up threshold from .meta.json sidecar when present."""
    _skip_if_no_onnx()
    onnx_path = str(tmp_path / "model.onnx")
    _make_minimal_onnx(onnx_path)

    meta = {"threshold": 0.042, "metadata": {"base_ch": 16}}
    with open(onnx_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    scorer = OnnxAnomalyScorer.from_path(onnx_path)
    assert scorer.threshold == pytest.approx(0.042)
    assert scorer.metadata.get("base_ch") == 16
