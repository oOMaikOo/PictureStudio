"""
Unit tests for core.metrics — compute_metrics, ROC/AUC, top-k accuracy.
Note: compute_metrics expects integer class indices, not string names.
"""
import pytest


CLASSES = ["gut", "schlecht", "neutral"]


def _to_idx(labels, class_names):
    """Convert string labels to integer indices."""
    idx_map = {n: i for i, n in enumerate(class_names)}
    return [idx_map[l] for l in labels]


def _make_perfect(classes=CLASSES, n=30):
    """Balanced perfect predictions as integer indices."""
    true, pred = [], []
    for i in range(n):
        idx = i % len(classes)
        true.append(idx)
        pred.append(idx)
    return true, pred


def _make_binary(n=20):
    """Binary (2-class) predictions with some errors."""
    true = [0] * 10 + [1] * 10
    pred = [0] * 8 + [1] * 2 + [1] * 8 + [0] * 2
    return true, pred


# ---------------------------------------------------------------------------
# Basic metric structure
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_dict(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        assert isinstance(result, dict)

    def test_required_keys(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_f1", "per_class", "confusion_matrix"):
            assert key in result, f"missing key: {key}"

    def test_perfect_accuracy(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_perfect_f1(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        assert result["macro_f1"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_weighted_f1(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        assert result["weighted_f1"] == pytest.approx(1.0, abs=1e-6)

    def test_imperfect_accuracy_below_one(self):
        from core.metrics import compute_metrics
        cls = ["gut", "schlecht"]
        true = _to_idx(["gut", "gut", "schlecht", "schlecht"], cls)
        pred = _to_idx(["gut", "schlecht", "schlecht", "gut"], cls)
        result = compute_metrics(true, pred, cls)
        assert result["accuracy"] < 1.0

    def test_confusion_matrix_shape(self):
        from core.metrics import compute_metrics
        cls = CLASSES[:2]
        t, p = _make_perfect(cls)
        result = compute_metrics(t, p, cls)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)

    def test_per_class_contains_classes(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        per_class = result["per_class"]
        for cls in CLASSES:
            assert cls in per_class

    def test_per_class_has_f1_key(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        for cls_data in result["per_class"].values():
            assert "f1" in cls_data
            assert "precision" in cls_data
            assert "recall" in cls_data

    def test_weighted_f1_key(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        assert "weighted_f1" in result

    def test_confusion_matrix_diagonal_perfect(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect(CLASSES[:2], n=10)
        result = compute_metrics(t, p, CLASSES[:2])
        cm = result["confusion_matrix"]
        # Off-diagonals should be 0
        assert cm[0][1] == 0
        assert cm[1][0] == 0


# ---------------------------------------------------------------------------
# ROC / AUC (binary)
# ---------------------------------------------------------------------------

class TestROCAUC:
    def _make_probs_binary(self, n=20):
        import random
        random.seed(42)
        true = [0] * 10 + [1] * 10
        probs = [
            [0.8 + random.uniform(-0.1, 0.1), 0.2] if t == 0
            else [0.2, 0.8 + random.uniform(-0.1, 0.1)]
            for t in true
        ]
        return true, probs

    def test_roc_auc_present_for_binary(self):
        from core.metrics import compute_metrics
        true, probs = self._make_probs_binary()
        result = compute_metrics(true, true, ["gut", "schlecht"], pred_probs=probs)
        assert "roc_auc" in result

    def test_roc_auc_range(self):
        from core.metrics import compute_metrics
        true, probs = self._make_probs_binary()
        result = compute_metrics(true, true, ["gut", "schlecht"], pred_probs=probs)
        auc = result.get("roc_auc")
        if auc is not None:
            assert 0.0 <= auc <= 1.0

    def test_roc_curve_keys(self):
        from core.metrics import compute_metrics
        true, probs = self._make_probs_binary()
        result = compute_metrics(true, true, ["gut", "schlecht"], pred_probs=probs)
        if "roc_curve" in result and result["roc_curve"]:
            curve = result["roc_curve"]
            assert "fpr" in curve
            assert "tpr" in curve

    def test_no_roc_for_multiclass(self):
        from core.metrics import compute_metrics
        t, p = _make_perfect()
        probs = [[1/3, 1/3, 1/3]] * len(t)
        result = compute_metrics(t, p, CLASSES, pred_probs=probs)
        # roc_auc not expected for >2 classes — just no crash
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Top-K accuracy (multi-class)
# ---------------------------------------------------------------------------

class TestTopKAccuracy:
    def _make_topk_probs(self, classes, n=30):
        k = len(classes)
        true, probs = [], []
        for i in range(n):
            idx = i % k
            true.append(idx)
            row = [0.05] * k
            row[idx] = 0.8
            probs.append(row)
        return true, probs

    def test_top3_accuracy_gte_top1(self):
        from core.metrics import compute_metrics
        true, probs = self._make_topk_probs(CLASSES)
        pred = [row.index(max(row)) for row in probs]
        result = compute_metrics(true, pred, CLASSES, pred_probs=probs)
        top3 = result.get("top3_accuracy", None)
        if top3 is not None:
            assert top3 >= result["accuracy"] - 1e-9

    def test_top3_at_most_one(self):
        from core.metrics import compute_metrics
        true, probs = self._make_topk_probs(CLASSES)
        pred = [row.index(max(row)) for row in probs]
        result = compute_metrics(true, pred, CLASSES, pred_probs=probs)
        top3 = result.get("top3_accuracy", None)
        if top3 is not None:
            assert top3 <= 1.0

    def test_top3_present_for_multiclass_with_probs(self):
        from core.metrics import compute_metrics
        true, probs = self._make_topk_probs(CLASSES)
        pred = [row.index(max(row)) for row in probs]
        result = compute_metrics(true, pred, CLASSES, pred_probs=probs)
        assert "top3_accuracy" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_class(self):
        from core.metrics import compute_metrics
        true = [0] * 10
        pred = [0] * 10
        result = compute_metrics(true, pred, ["gut"])
        assert result["accuracy"] == pytest.approx(1.0)

    def test_all_wrong_binary(self):
        from core.metrics import compute_metrics
        true = [0] * 5 + [1] * 5
        pred = [1] * 5 + [0] * 5
        result = compute_metrics(true, pred, ["gut", "schlecht"])
        assert result["accuracy"] == pytest.approx(0.0)

    def test_empty_probs_no_crash(self):
        from core.metrics import compute_metrics
        true = [0, 1]
        pred = [0, 1]
        result = compute_metrics(true, pred, ["gut", "schlecht"], pred_probs=None)
        assert isinstance(result, dict)

    def test_empty_input_returns_empty(self):
        from core.metrics import compute_metrics
        result = compute_metrics([], [], CLASSES)
        assert result == {}

    def test_format_metrics_text(self):
        from core.metrics import compute_metrics, format_metrics_text
        t, p = _make_perfect()
        result = compute_metrics(t, p, CLASSES)
        text = format_metrics_text(result)
        assert isinstance(text, str)
        assert "Accuracy" in text
