"""
Tests for core/report.py — generate_html_report
"""
import os

import pytest

from core.report import generate_html_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_run_data(**overrides):
    """Return a minimal run_data dict sufficient for generate_html_report."""
    data = {
        "run_id": "run_2025-01-01_001",
        "timestamp": "2025-01-01T12:00:00",
        "model_type": "ResNet18",
        "device": "cpu",
        "train_size": 80,
        "test_size": 20,
        "class_names": ["gut", "schlecht"],
        "hyperparameters": {"lr": 0.001, "batch_size": 32},
        "software_versions": {"torch": "2.0", "python": "3.11"},
        "metrics": {
            "accuracy": 0.95,
            "macro_f1": 0.94,
            "macro_precision": 0.93,
            "macro_recall": 0.95,
            "per_class": {
                "gut":      {"precision": 0.96, "recall": 0.94, "f1": 0.95, "support": 10},
                "schlecht": {"precision": 0.90, "recall": 0.96, "f1": 0.93, "support": 10},
            },
            "confusion_matrix": [[9, 1], [1, 9]],
        },
        "history": {
            "train_loss": [0.5, 0.3],
            "val_loss":   [0.6, 0.35],
            "train_acc":  [0.7, 0.9],
            "val_acc":    [0.65, 0.88],
        },
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_report_returns_output_path(tmp_path):
    """generate_html_report returns the path it was given."""
    out = str(tmp_path / "report.html")
    result = generate_html_report(_minimal_run_data(), out)
    assert result == out


def test_report_file_is_non_empty_html(tmp_path):
    """The written file is a non-empty HTML document."""
    out = str(tmp_path / "report.html")
    generate_html_report(_minimal_run_data(), out)

    with open(out, encoding="utf-8") as fh:
        content = fh.read()

    assert len(content) > 100
    assert "<html" in content
    assert "</html>" in content


def test_model_name_appears_in_report(tmp_path):
    """The model / architecture name appears somewhere in the generated HTML."""
    out = str(tmp_path / "report.html")
    generate_html_report(_minimal_run_data(), out, project_name="MeinProjekt")

    with open(out, encoding="utf-8") as fh:
        content = fh.read()

    assert "ResNet18" in content
    assert "MeinProjekt" in content


def test_report_with_empty_metrics(tmp_path):
    """generate_html_report handles an empty metrics dict without raising."""
    run_data = _minimal_run_data()
    run_data["metrics"] = {}
    run_data["history"] = {}
    run_data["class_names"] = []

    out = str(tmp_path / "empty_report.html")
    generate_html_report(run_data, out)

    with open(out, encoding="utf-8") as fh:
        content = fh.read()

    assert "<html" in content


def test_report_creates_parent_directory(tmp_path):
    """generate_html_report creates intermediate directories if needed."""
    out = str(tmp_path / "nested" / "sub" / "report.html")
    generate_html_report(_minimal_run_data(), out)
    assert os.path.isfile(out)


def test_run_id_in_report(tmp_path):
    """The run_id appears in the HTML title / body."""
    out = str(tmp_path / "report.html")
    generate_html_report(_minimal_run_data(run_id="test-run-xyz"), out)

    with open(out, encoding="utf-8") as fh:
        content = fh.read()

    assert "test-run-xyz" in content
