"""Tests für gui/dialogs/model_comparison_dialog.py"""
from __future__ import annotations
import sys, os, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.dialogs.model_comparison_dialog import ModelComparisonDialog


SAMPLE_RUNS = [
    {"name": "Run A", "accuracy": 0.92, "macro_f1": 0.91, "architecture": "resnet18", "is_best": True},
    {"name": "Run B", "accuracy": 0.88, "macro_f1": 0.87, "architecture": "mobilenetv2", "is_best": False},
    {"name": "Run C", "accuracy": 0.85, "macro_f1": 0.84, "architecture": "efficientnet_b0", "is_best": False},
]


def test_instantiates_with_empty_runs(qtbot):
    dlg = ModelComparisonDialog([])
    qtbot.addWidget(dlg)
    assert dlg is not None


def test_instantiates_with_runs(qtbot):
    dlg = ModelComparisonDialog(SAMPLE_RUNS)
    qtbot.addWidget(dlg)


def test_table_has_correct_row_count(qtbot):
    dlg = ModelComparisonDialog(SAMPLE_RUNS)
    qtbot.addWidget(dlg)
    assert dlg.table.rowCount() == len(SAMPLE_RUNS)


def test_table_has_five_columns(qtbot):
    dlg = ModelComparisonDialog(SAMPLE_RUNS)
    qtbot.addWidget(dlg)
    assert dlg.table.columnCount() == 5


def test_best_run_shown(qtbot):
    dlg = ModelComparisonDialog(SAMPLE_RUNS)
    qtbot.addWidget(dlg)
    star_items = [
        dlg.table.item(r, 4)
        for r in range(dlg.table.rowCount())
        if dlg.table.item(r, 4) and dlg.table.item(r, 4).text() == "★"
    ]
    assert len(star_items) == 1
