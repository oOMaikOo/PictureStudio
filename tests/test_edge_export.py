"""Tests für core/edge_export.py"""
from __future__ import annotations
import os, pytest
from core.edge_export import EdgeExporter

torch = pytest.importorskip("torch", reason="torch not installed")


def test_edge_exporter_instantiates():
    e = EdgeExporter()
    assert e is not None


def test_has_coreml_returns_bool():
    result = EdgeExporter.has_coreml()
    assert isinstance(result, bool)


def test_has_quantization_returns_bool():
    result = EdgeExporter.has_quantization()
    assert isinstance(result, bool)


def test_export_quantized_onnx_with_dummy_model(tmp_path):
    """Exportiert ein kleines Modell zu ONNX (ohne Quantisierung für Speed)."""
    import torch, torch.nn as nn
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 64 * 64, 10),
    )
    # Speichere als fake checkpoint
    ckpt_path = str(tmp_path / "dummy.pth")
    torch.save(model, ckpt_path)

    exporter = EdgeExporter()
    # Monkey-patch _load_model so it returns our simple model
    exporter._load_model = lambda path: model

    out_path = str(tmp_path / "model.onnx")
    result = exporter.export_quantized_onnx(
        ckpt_path, out_path, image_size=64, quantize=False
    )
    assert os.path.isfile(result)
    assert result == out_path
