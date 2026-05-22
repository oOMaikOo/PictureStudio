"""
Extended tests for core/model_manager.py — ModelManager.

Covers: corrupt JSON recovery, mark_as_best exclusivity, archive visibility,
delete with file removal, register roundtrip, get_best when none flagged.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.model_manager import ModelManager


# ------------------------------------------------------------------ helpers

def _result(arch="resnet18", acc=0.9, path=""):
    return {
        "model_type": arch,
        "best_model_path": path,
        "class_names": ["a", "b"],
        "hyperparameters": {"image_size": 224},
        "metrics": {"accuracy": acc},
        "run_id": "abc123",
        "timestamp": "2024-01-01T00:00:00",
        "train_size": 10,
        "val_size": 3,
        "test_size": 2,
    }


# ================================================================== corrupt JSON recovery

class TestCorruptJsonRecovery:
    def test_loads_empty_on_corrupt_json(self, tmp_path):
        reg = tmp_path / "model_registry.json"
        reg.write_text("{broken json{{")
        mgr = ModelManager(str(tmp_path))
        assert mgr.get_all(include_archived=True) == []

    def test_loads_empty_on_empty_file(self, tmp_path):
        reg = tmp_path / "model_registry.json"
        reg.write_text("")
        mgr = ModelManager(str(tmp_path))
        assert mgr.get_all() == []

    def test_continues_after_corrupt_file(self, tmp_path):
        reg = tmp_path / "model_registry.json"
        reg.write_text("null")
        mgr = ModelManager(str(tmp_path))
        mgr.register(_result(), name="new_model")
        assert len(mgr.get_all()) == 1

    def test_missing_registry_file_is_fine(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        assert mgr.get_all() == []


# ================================================================== register

class TestRegister:
    def test_returns_model_record(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_result(), name="my_model")
        assert rec.name == "my_model"

    def test_record_persisted_to_disk(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        mgr.register(_result(), name="saved")
        mgr2 = ModelManager(str(tmp_path))
        names = [m.name for m in mgr2.get_all()]
        assert "saved" in names

    def test_architecture_stored(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_result(arch="efficientnet"))
        assert rec.architecture == "efficientnet"

    def test_auto_name_when_empty(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_result(arch="resnet18"), name="")
        assert "resnet18" in rec.name

    def test_model_id_is_unique(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r1 = mgr.register(_result())
        r2 = mgr.register(_result())
        assert r1.model_id != r2.model_id


# ================================================================== mark_as_best

class TestMarkAsBest:
    def test_flags_correct_model(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r1 = mgr.register(_result())
        r2 = mgr.register(_result())
        mgr.mark_as_best(r2.model_id)
        assert mgr.get_by_id(r2.model_id).is_best is True

    def test_clears_previous_best(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r1 = mgr.register(_result())
        r2 = mgr.register(_result())
        mgr.mark_as_best(r1.model_id)
        mgr.mark_as_best(r2.model_id)
        assert mgr.get_by_id(r1.model_id).is_best is False

    def test_only_one_best_at_a_time(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        records = [mgr.register(_result()) for _ in range(5)]
        mgr.mark_as_best(records[3].model_id)
        best_count = sum(1 for m in mgr.get_all() if m.is_best)
        assert best_count == 1

    def test_get_best_returns_flagged(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result())
        mgr.mark_as_best(r.model_id)
        assert mgr.get_best().model_id == r.model_id

    def test_get_best_none_when_none_flagged(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        mgr.register(_result())
        assert mgr.get_best() is None

    def test_best_persisted_across_reload(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result())
        mgr.mark_as_best(r.model_id)
        mgr2 = ModelManager(str(tmp_path))
        assert mgr2.get_best().model_id == r.model_id

    def test_mark_archived_model_as_best(self, tmp_path):
        """mark_as_best should work even if the model is archived."""
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result())
        mgr.archive(r.model_id)
        mgr.mark_as_best(r.model_id)
        assert mgr.get_by_id(r.model_id).is_best is True


# ================================================================== archive

class TestArchive:
    def test_archived_hidden_from_get_all(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result())
        mgr.archive(r.model_id)
        assert r.model_id not in [m.model_id for m in mgr.get_all()]

    def test_archived_visible_with_flag(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result())
        mgr.archive(r.model_id)
        assert r.model_id in [m.model_id for m in mgr.get_all(include_archived=True)]

    def test_archive_nonexistent_noop(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        mgr.archive("deadbeef")   # should not raise

    def test_archive_persisted(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result())
        mgr.archive(r.model_id)
        mgr2 = ModelManager(str(tmp_path))
        archived = [m for m in mgr2.get_all(include_archived=True) if m.archived]
        assert any(m.model_id == r.model_id for m in archived)


# ================================================================== delete

class TestDelete:
    def test_removes_record(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result())
        mgr.delete(r.model_id)
        assert mgr.get_by_id(r.model_id) is None

    def test_delete_nonexistent_noop(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        mgr.delete("deadbeef")   # should not raise

    def test_delete_file_removes_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "model.pth"
        checkpoint.write_bytes(b"fake checkpoint")
        mgr = ModelManager(str(tmp_path))
        res = dict(_result(), best_model_path=str(checkpoint))
        r = mgr.register(res)
        mgr.delete(r.model_id, delete_file=True)
        assert not checkpoint.exists()

    def test_delete_without_file_flag_keeps_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "model.pth"
        checkpoint.write_bytes(b"fake checkpoint")
        mgr = ModelManager(str(tmp_path))
        res = dict(_result(), best_model_path=str(checkpoint))
        r = mgr.register(res)
        mgr.delete(r.model_id, delete_file=False)
        assert checkpoint.exists()


# ================================================================== get_by_id

class TestGetById:
    def test_returns_correct_record(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        r = mgr.register(_result(), name="target")
        assert mgr.get_by_id(r.model_id).name == "target"

    def test_returns_none_for_unknown_id(self, tmp_path):
        mgr = ModelManager(str(tmp_path))
        assert mgr.get_by_id("00000000") is None
