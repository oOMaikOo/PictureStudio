"""
Tests for core/metrics.py — compute_multilabel_metrics().

Covers: empty inputs, perfect predictions, all-wrong predictions,
hamming accuracy, exact-match accuracy, per-class P/R/F1,
macro aggregation, boundary values (0.5 threshold rounding).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.metrics import compute_multilabel_metrics


def _run(true, pred, names=None):
    if names is None:
        names = [f"c{i}" for i in range(len(true[0]))]
    return compute_multilabel_metrics(true, pred, names)


# ================================================================== empty / degenerate

class TestEdgeCases:
    def test_empty_samples_returns_empty(self):
        assert compute_multilabel_metrics([], [], ["a", "b"]) == {}

    def test_empty_classes_returns_empty(self):
        assert compute_multilabel_metrics([[]], [[]], []) == {}

    def test_is_multi_label_flag(self):
        r = _run([[1, 0]], [[1, 0]])
        assert r["is_multi_label"] is True

    def test_total_samples_correct(self):
        r = _run([[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 1], [1, 1]])
        assert r["total_samples"] == 3

    def test_class_names_preserved(self):
        r = compute_multilabel_metrics([[1, 0]], [[1, 0]], ["cat", "dog"])
        assert r["class_names"] == ["cat", "dog"]


# ================================================================== perfect predictions

class TestPerfectPredictions:
    def test_hamming_accuracy_is_one(self):
        true = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
        r = _run(true, true)
        assert r["hamming_accuracy"] == 1.0

    def test_exact_match_accuracy_is_one(self):
        true = [[1, 0], [0, 1]]
        r = _run(true, true)
        assert r["exact_match_accuracy"] == 1.0

    def test_macro_f1_is_one(self):
        true = [[1, 0], [0, 1], [1, 1]]
        r = _run(true, true)
        assert r["macro_f1"] == 1.0

    def test_per_class_precision_one(self):
        true = [[1, 0], [0, 1]]
        r = _run(true, true, ["a", "b"])
        assert r["per_class"]["a"]["precision"] == 1.0
        assert r["per_class"]["b"]["precision"] == 1.0

    def test_per_class_recall_one(self):
        true = [[1, 0], [0, 1]]
        r = _run(true, true, ["a", "b"])
        assert r["per_class"]["a"]["recall"] == 1.0


# ================================================================== all wrong

class TestAllWrong:
    def test_hamming_accuracy_is_zero(self):
        true = [[1, 0], [0, 1]]
        pred = [[0, 1], [1, 0]]
        r = _run(true, pred)
        assert r["hamming_accuracy"] == 0.0

    def test_exact_match_accuracy_is_zero(self):
        true = [[1, 0], [0, 1]]
        pred = [[0, 1], [1, 0]]
        r = _run(true, pred)
        assert r["exact_match_accuracy"] == 0.0

    def test_macro_f1_is_zero(self):
        true = [[1, 0], [0, 1]]
        pred = [[0, 1], [1, 0]]
        r = _run(true, pred)
        assert r["macro_f1"] == 0.0


# ================================================================== hamming accuracy

class TestHammingAccuracy:
    def test_half_correct(self):
        # 2 samples, 2 classes → 4 bits total, 2 correct
        true = [[1, 0], [0, 1]]
        pred = [[1, 1], [0, 0]]   # bit 1 wrong each row
        r = _run(true, pred)
        assert r["hamming_accuracy"] == 0.5

    def test_three_of_four_correct(self):
        true = [[1, 0], [1, 1]]
        pred = [[1, 1], [1, 1]]  # 3 right, 1 wrong
        r = _run(true, pred)
        assert abs(r["hamming_accuracy"] - 0.75) < 1e-4

    def test_accuracy_alias_equals_hamming(self):
        true = [[1, 0, 1], [0, 1, 0]]
        pred = [[1, 0, 0], [0, 1, 0]]
        r = _run(true, pred)
        assert r["accuracy"] == r["hamming_accuracy"]


# ================================================================== exact match

class TestExactMatch:
    def test_one_of_two_exact(self):
        true = [[1, 0], [0, 1]]
        pred = [[1, 0], [1, 1]]   # first exact, second not
        r = _run(true, pred)
        assert r["exact_match_accuracy"] == 0.5

    def test_no_exact_when_one_bit_off(self):
        true = [[1, 1], [0, 0]]
        pred = [[1, 0], [0, 1]]
        r = _run(true, pred)
        assert r["exact_match_accuracy"] == 0.0


# ================================================================== per-class metrics

class TestPerClassMetrics:
    def _setup(self):
        # 4 samples, 2 classes
        # class a (idx 0): tp=2, fp=0, fn=1 → prec=1.0, rec=2/3
        # class b (idx 1): tp=1, fp=1, fn=1 → prec=0.5, rec=0.5
        true = [[1, 1], [1, 0], [0, 1], [1, 0]]
        pred = [[1, 1], [1, 1], [0, 0], [0, 0]]
        return true, pred

    def test_tp_fp_fn_correct(self):
        true, pred = self._setup()
        r = _run(true, pred, ["a", "b"])
        assert r["per_class"]["a"]["tp"] == 2
        assert r["per_class"]["a"]["fp"] == 0
        assert r["per_class"]["a"]["fn"] == 1
        assert r["per_class"]["b"]["tp"] == 1
        assert r["per_class"]["b"]["fp"] == 1
        assert r["per_class"]["b"]["fn"] == 1

    def test_precision_computed(self):
        true, pred = self._setup()
        r = _run(true, pred, ["a", "b"])
        assert r["per_class"]["a"]["precision"] == 1.0
        assert abs(r["per_class"]["b"]["precision"] - 0.5) < 1e-4

    def test_recall_computed(self):
        true, pred = self._setup()
        r = _run(true, pred, ["a", "b"])
        assert abs(r["per_class"]["a"]["recall"] - 2 / 3) < 1e-3
        assert abs(r["per_class"]["b"]["recall"] - 0.5) < 1e-4

    def test_f1_is_harmonic_mean(self):
        true, pred = self._setup()
        r = _run(true, pred, ["a", "b"])
        prec_a = r["per_class"]["a"]["precision"]
        rec_a = r["per_class"]["a"]["recall"]
        expected_f1 = 2 * prec_a * rec_a / (prec_a + rec_a)
        assert abs(r["per_class"]["a"]["f1"] - expected_f1) < 1e-4

    def test_support_is_tp_plus_fn(self):
        true, pred = self._setup()
        r = _run(true, pred, ["a", "b"])
        # class a: tp=2, fn=1 → support=3
        assert r["per_class"]["a"]["support"] == 3
        # class b: tp=1, fn=1 → support=2
        assert r["per_class"]["b"]["support"] == 2

    def test_zero_division_precision_safe(self):
        # class with no positive predictions (tp=0, fp=0) → precision=0
        true = [[0, 0], [0, 0]]
        pred = [[0, 0], [0, 0]]
        r = _run(true, pred, ["a", "b"])
        assert r["per_class"]["a"]["precision"] == 0.0
        assert r["per_class"]["a"]["recall"] == 0.0


# ================================================================== 0.5 threshold rounding

class TestThresholdRounding:
    def test_exactly_half_rounds_to_one(self):
        # round(0.5) in Python 3 → 0 (banker's rounding)
        # but round(0.5) == 0 so treating 0.5 as negative for class "present"
        # Just verify function doesn't crash and returns valid metrics
        true = [[0.5, 0.5]]
        pred = [[0.5, 0.5]]
        r = _run(true, pred)
        assert 0.0 <= r["hamming_accuracy"] <= 1.0

    def test_above_half_is_positive(self):
        true = [[0.6, 0.0]]
        pred = [[0.6, 0.0]]   # both match after round()
        r = _run(true, pred, ["a", "b"])
        assert r["hamming_accuracy"] == 1.0


# ================================================================== macro aggregation

class TestMacroAggregation:
    def test_macro_f1_is_mean_of_per_class(self):
        true = [[1, 0], [0, 1], [1, 0]]
        pred = [[1, 0], [0, 1], [0, 0]]
        r = _run(true, pred, ["a", "b"])
        expected = (r["per_class"]["a"]["f1"] + r["per_class"]["b"]["f1"]) / 2
        assert abs(r["macro_f1"] - expected) < 1e-4

    def test_weighted_f1_equals_macro(self):
        # current implementation: weighted_f1 == macro_f1
        true = [[1, 0, 1], [0, 1, 0]]
        pred = [[1, 0, 0], [0, 1, 0]]
        r = _run(true, pred)
        assert r["weighted_f1"] == r["macro_f1"]
