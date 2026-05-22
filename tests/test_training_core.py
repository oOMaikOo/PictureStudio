"""
Tests for core/training.py — FocalLoss, EarlyStopping, train_one_epoch, evaluate.

Does NOT test TrainingWorker.run() end-to-end (that is covered by test_integration.py).
Tests the building blocks individually so failures are easy to diagnose.
"""
import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

from core.training import FocalLoss, EarlyStopping, train_one_epoch, evaluate


# ------------------------------------------------------------------ helpers

def _simple_model(in_features: int = 4, n_classes: int = 2) -> nn.Module:
    return nn.Sequential(nn.Linear(in_features, n_classes))


def _dummy_loader(n_samples: int = 16, in_features: int = 4, n_classes: int = 2,
                  batch_size: int = 8):
    X = torch.randn(n_samples, in_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


# ================================================================== FocalLoss

class TestFocalLoss:
    def test_gamma_zero_equals_crossentropy(self):
        """gamma=0 → FocalLoss ≈ CrossEntropyLoss (up to fp precision)."""
        torch.manual_seed(0)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        fl = FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()
        assert abs(fl(logits, targets).item() - ce(logits, targets).item()) < 1e-4

    def test_focal_loss_is_lower_than_ce_for_high_confidence(self):
        """With gamma>0, well-classified examples contribute less loss."""
        # Force model to be confident: large logit for correct class
        n = 10
        logits = torch.zeros(n, 2)
        logits[:, 0] = 10.0          # very confident on class 0
        targets = torch.zeros(n, dtype=torch.long)
        fl = FocalLoss(gamma=2.0)
        ce = nn.CrossEntropyLoss()
        assert fl(logits, targets).item() < ce(logits, targets).item()

    def test_returns_scalar(self):
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = FocalLoss(gamma=2.0)(logits, targets)
        assert loss.dim() == 0

    def test_loss_is_nonnegative(self):
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        assert FocalLoss(gamma=2.0)(logits, targets).item() >= 0.0

    def test_perfect_predictions_approach_zero(self):
        """High-confidence correct predictions → very small focal loss."""
        n, c = 10, 3
        logits = torch.full((n, c), -10.0)
        targets = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            logits[i, 0] = 10.0      # correct class very confident
        loss = FocalLoss(gamma=2.0)(logits, targets).item()
        assert loss < 0.01

    def test_per_class_weight_accepted(self):
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        weight = torch.tensor([1.0, 2.0, 0.5])
        loss = FocalLoss(gamma=2.0, weight=weight)(logits, targets)
        assert loss.item() >= 0.0


# ================================================================== EarlyStopping

class TestEarlyStopping:
    def test_no_stop_on_improvement(self):
        es = EarlyStopping(patience=3)
        for val in [0.5, 0.6, 0.7, 0.8]:
            assert not es.step(val)

    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(0.8)           # best
        es.step(0.8)           # no improve → counter 1
        es.step(0.8)           # no improve → counter 2
        assert not es.triggered
        es.step(0.8)           # no improve → counter 3 → trigger
        assert es.triggered

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(0.5)           # best=0.5
        es.step(0.5)           # counter 1
        es.step(0.5)           # counter 2
        es.step(0.6)           # improve → counter reset
        assert es.counter == 0
        assert not es.triggered

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(0.5)           # best=0.5
        es.step(0.55)          # 0.55 - 0.5 = 0.05 < 0.1 → no improve
        assert es.counter == 1
        es.step(0.61)          # 0.61 - 0.5 = 0.11 > 0.1 → improve
        assert es.counter == 0

    def test_first_step_never_triggers(self):
        es = EarlyStopping(patience=1)
        assert not es.step(0.5)

    def test_step_returns_bool(self):
        es = EarlyStopping(patience=5)
        result = es.step(0.5)
        assert isinstance(result, bool)


# ================================================================== train_one_epoch

class TestTrainOneEpoch:
    def test_returns_loss_and_accuracy(self):
        model = _simple_model()
        loader = _dummy_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_loss_is_nonnegative(self):
        model = _simple_model()
        loader = _dummy_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loss, _ = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert loss >= 0.0

    def test_accuracy_in_range(self):
        model = _simple_model()
        loader = _dummy_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        _, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert 0.0 <= acc <= 1.0

    def test_model_weights_change(self):
        model = _simple_model()
        before = model[0].weight.data.clone()
        loader = _dummy_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert not torch.equal(model[0].weight.data, before)

    def test_focal_loss_accepted(self):
        model = _simple_model()
        loader = _dummy_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss, acc = train_one_epoch(
            model, loader, FocalLoss(gamma=2.0), optimizer, torch.device("cpu")
        )
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0


# ================================================================== evaluate

class TestEvaluate:
    def test_returns_four_values(self):
        model = _simple_model()
        loader = _dummy_loader(n_samples=8)
        criterion = nn.CrossEntropyLoss()
        result = evaluate(model, loader, criterion, torch.device("cpu"))
        assert len(result) == 4

    def test_loss_nonnegative(self):
        model = _simple_model()
        loader = _dummy_loader(n_samples=8)
        criterion = nn.CrossEntropyLoss()
        loss, _, _, _ = evaluate(model, loader, criterion, torch.device("cpu"))
        assert loss >= 0.0

    def test_accuracy_in_range(self):
        model = _simple_model()
        loader = _dummy_loader(n_samples=8)
        criterion = nn.CrossEntropyLoss()
        _, acc, _, _ = evaluate(model, loader, criterion, torch.device("cpu"))
        assert 0.0 <= acc <= 1.0

    def test_preds_and_labels_length_match(self):
        n = 16
        model = _simple_model()
        loader = _dummy_loader(n_samples=n)
        criterion = nn.CrossEntropyLoss()
        _, _, preds, labels = evaluate(model, loader, criterion, torch.device("cpu"))
        assert len(preds) == n
        assert len(labels) == n

    def test_preds_are_valid_class_indices(self):
        model = _simple_model(n_classes=3)
        loader = _dummy_loader(n_samples=12, n_classes=3)
        criterion = nn.CrossEntropyLoss()
        _, _, preds, _ = evaluate(model, loader, criterion, torch.device("cpu"))
        assert all(0 <= p < 3 for p in preds)

    def test_no_gradient_computed(self):
        """evaluate() should not accumulate gradients."""
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = _dummy_loader(n_samples=8)
        criterion = nn.CrossEntropyLoss()
        evaluate(model, loader, criterion, torch.device("cpu"))
        assert model[0].weight.grad is None
