"""
Unit tests for core/model_manager.py → ModelRecord and ModelManager

Tests cover: ModelRecord round-trip serialisation, formatting helpers,
and all ModelManager CRUD operations (register, get_all, get_by_id,
get_best, mark_as_best, archive, delete, update_metadata), plus
registry persistence across instantiations.
"""
import json
import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_training_result(arch: str = "resnet18", acc: float = 0.90) -> dict:
    return {
        "run_id": "run_test_001",
        "model_type": arch,
        "timestamp": "2025-01-01T12:00:00",
        "best_model_path": "/tmp/model.pth",
        "class_names": ["cat", "dog", "fish"],
        "hyperparameters": {"image_size": 224, "lr": 0.001, "epochs": 10},
        "metrics": {"accuracy": acc, "macro_f1": acc - 0.02,
                    "macro_precision": acc - 0.01, "macro_recall": acc - 0.03},
        "train_size": 800,
        "val_size": 100,
        "test_size": 100,
        "software_versions": {"torch": "2.0", "python": "3.11"},
    }


def _sample_record_data(**kwargs) -> dict:
    base = {
        "model_id": "abc12345",
        "name": "TestModel",
        "description": "A test model",
        "architecture": "resnet18",
        "version": "3",
        "created_at": "2025-01-01T12:00:00",
        "model_path": "/tmp/model.pth",
        "onnx_path": "",
        "class_names": ["cat", "dog"],
        "image_size": 224,
        "hyperparameters": {"lr": 0.001},
        "metrics": {"accuracy": 0.92, "macro_f1": 0.91},
        "train_size": 600,
        "val_size": 100,
        "test_size": 50,
        "is_best": False,
        "archived": False,
        "run_id": "run_001",
        "software_versions": {},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# ModelRecord
# ---------------------------------------------------------------------------

class TestModelRecord:
    def test_to_dict_round_trip_preserves_all_fields(self):
        from core.model_manager import ModelRecord
        data = _sample_record_data()
        record = ModelRecord(data)
        result = record.to_dict()
        for key, value in data.items():
            assert result[key] == value, f"Field mismatch for '{key}'"

    def test_from_dict_sets_all_attributes(self):
        from core.model_manager import ModelRecord
        data = _sample_record_data(is_best=True, archived=True)
        record = ModelRecord(data)
        assert record.model_id == "abc12345"
        assert record.name == "TestModel"
        assert record.architecture == "resnet18"
        assert record.is_best is True
        assert record.archived is True
        assert record.class_names == ["cat", "dog"]
        assert record.image_size == 224

    def test_missing_keys_use_safe_defaults(self):
        from core.model_manager import ModelRecord
        record = ModelRecord({})
        assert record.model_id == ""
        assert record.name == ""
        assert record.architecture == ""
        assert record.metrics == {}
        assert record.is_best is False
        assert record.archived is False
        assert record.class_names == []
        assert record.image_size == 224

    def test_accuracy_str_formats_as_percentage(self):
        from core.model_manager import ModelRecord
        record = ModelRecord(_sample_record_data(metrics={"accuracy": 0.9567, "macro_f1": 0.90}))
        assert record.accuracy_str() == "95.67%"

    def test_accuracy_str_zero_when_accuracy_missing(self):
        from core.model_manager import ModelRecord
        record = ModelRecord({"metrics": {}})
        assert record.accuracy_str() == "0.00%"

    def test_f1_str_formats_as_percentage(self):
        from core.model_manager import ModelRecord
        record = ModelRecord(_sample_record_data(metrics={"accuracy": 0.90, "macro_f1": 0.8842}))
        assert record.f1_str() == "88.42%"

    def test_f1_str_zero_when_macro_f1_missing(self):
        from core.model_manager import ModelRecord
        record = ModelRecord({"metrics": {"accuracy": 0.9}})
        assert record.f1_str() == "0.00%"

    def test_accuracy_str_perfect_score(self):
        from core.model_manager import ModelRecord
        record = ModelRecord({"metrics": {"accuracy": 1.0}})
        assert record.accuracy_str() == "100.00%"


# ---------------------------------------------------------------------------
# ModelManager — registry persistence
# ---------------------------------------------------------------------------

class TestModelManagerRegistry:
    def test_creates_models_directory_if_not_exists(self, tmp_path):
        from core.model_manager import ModelManager
        models_dir = str(tmp_path / "new_models_dir")
        ModelManager(models_dir)
        assert os.path.isdir(models_dir)

    def test_registry_json_written_after_register(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        mgr.register(_sample_training_result())
        registry_path = str(tmp_path / "model_registry.json")
        assert os.path.isfile(registry_path)

    def test_registry_json_is_valid_json(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        mgr.register(_sample_training_result())
        with open(str(tmp_path / "model_registry.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_reload_restores_registered_models(self, tmp_path):
        from core.model_manager import ModelManager
        mgr1 = ModelManager(str(tmp_path))
        record = mgr1.register(_sample_training_result(arch="resnet18"))
        original_id = record.model_id

        mgr2 = ModelManager(str(tmp_path))
        models = mgr2.get_all()
        assert len(models) == 1
        assert models[0].model_id == original_id

    def test_reload_restores_multiple_models(self, tmp_path):
        from core.model_manager import ModelManager
        mgr1 = ModelManager(str(tmp_path))
        mgr1.register(_sample_training_result(arch="resnet18"))
        mgr1.register(_sample_training_result(arch="efficientnet_b0", acc=0.95))

        mgr2 = ModelManager(str(tmp_path))
        assert len(mgr2.get_all()) == 2


# ---------------------------------------------------------------------------
# ModelManager — register
# ---------------------------------------------------------------------------

class TestModelManagerRegister:
    def test_register_returns_model_record(self, tmp_path):
        from core.model_manager import ModelManager, ModelRecord
        mgr = ModelManager(str(tmp_path))
        result = mgr.register(_sample_training_result())
        assert isinstance(result, ModelRecord)

    def test_register_assigns_non_empty_model_id(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        record = mgr.register(_sample_training_result())
        assert record.model_id != ""

    def test_register_uses_provided_name(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        record = mgr.register(_sample_training_result(), name="MyBestModel")
        assert record.name == "MyBestModel"

    def test_register_uses_provided_description(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        record = mgr.register(_sample_training_result(), description="First experiment")
        assert record.description == "First experiment"

    def test_register_stores_architecture(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        record = mgr.register(_sample_training_result(arch="efficientnet_b0"))
        assert record.architecture == "efficientnet_b0"

    def test_register_stores_metrics(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        record = mgr.register(_sample_training_result(acc=0.88))
        assert abs(record.metrics.get("accuracy", 0) - 0.88) < 1e-6


# ---------------------------------------------------------------------------
# ModelManager — get_all
# ---------------------------------------------------------------------------

class TestModelManagerGetAll:
    def test_get_all_returns_empty_list_for_new_manager(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        assert mgr.get_all() == []

    def test_get_all_returns_registered_model(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        mgr.register(_sample_training_result())
        assert len(mgr.get_all()) == 1

    def test_get_all_excludes_archived_by_default(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        mgr.archive(rec.model_id)
        assert len(mgr.get_all()) == 0

    def test_get_all_include_archived_true_returns_archived(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        mgr.archive(rec.model_id)
        assert len(mgr.get_all(include_archived=True)) == 1

    def test_get_all_returns_both_archived_and_active_when_flag_set(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec1 = mgr.register(_sample_training_result(arch="resnet18"))
        mgr.register(_sample_training_result(arch="vgg16"))
        mgr.archive(rec1.model_id)
        assert len(mgr.get_all(include_archived=True)) == 2
        assert len(mgr.get_all(include_archived=False)) == 1


# ---------------------------------------------------------------------------
# ModelManager — get_by_id
# ---------------------------------------------------------------------------

class TestModelManagerGetById:
    def test_get_by_id_returns_correct_record(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        found = mgr.get_by_id(rec.model_id)
        assert found is not None
        assert found.model_id == rec.model_id

    def test_get_by_id_returns_none_for_unknown_id(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        assert mgr.get_by_id("nonexistent_id") is None

    def test_get_by_id_distinguishes_between_multiple_records(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec1 = mgr.register(_sample_training_result(arch="resnet18"))
        rec2 = mgr.register(_sample_training_result(arch="vgg16"))
        assert mgr.get_by_id(rec1.model_id).architecture == "resnet18"
        assert mgr.get_by_id(rec2.model_id).architecture == "vgg16"


# ---------------------------------------------------------------------------
# ModelManager — delete
# ---------------------------------------------------------------------------

class TestModelManagerDelete:
    def test_delete_removes_model_from_list(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        mgr.delete(rec.model_id)
        assert mgr.get_by_id(rec.model_id) is None

    def test_delete_nonexistent_id_does_not_raise(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        mgr.delete("ghost_id")  # should not raise

    def test_delete_persisted_to_registry(self, tmp_path):
        from core.model_manager import ModelManager
        mgr1 = ModelManager(str(tmp_path))
        rec = mgr1.register(_sample_training_result())
        mgr1.delete(rec.model_id)

        mgr2 = ModelManager(str(tmp_path))
        assert mgr2.get_by_id(rec.model_id) is None

    def test_delete_leaves_other_models_intact(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec1 = mgr.register(_sample_training_result(arch="resnet18"))
        rec2 = mgr.register(_sample_training_result(arch="vgg16"))
        mgr.delete(rec1.model_id)
        remaining = mgr.get_all()
        assert len(remaining) == 1
        assert remaining[0].model_id == rec2.model_id


# ---------------------------------------------------------------------------
# ModelManager — get_best / mark_as_best
# ---------------------------------------------------------------------------

class TestModelManagerBest:
    def test_get_best_returns_none_when_no_model_is_best(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        mgr.register(_sample_training_result())
        assert mgr.get_best() is None

    def test_mark_as_best_sets_is_best_flag(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        mgr.mark_as_best(rec.model_id)
        assert mgr.get_best() is not None
        assert mgr.get_best().model_id == rec.model_id

    def test_mark_as_best_clears_previous_best(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec1 = mgr.register(_sample_training_result(arch="resnet18"))
        rec2 = mgr.register(_sample_training_result(arch="vgg16"))
        mgr.mark_as_best(rec1.model_id)
        mgr.mark_as_best(rec2.model_id)

        best = mgr.get_best()
        assert best.model_id == rec2.model_id

        # Old model must no longer be flagged
        old = mgr.get_by_id(rec1.model_id)
        assert old.is_best is False

    def test_only_one_model_is_best_at_a_time(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        recs = [mgr.register(_sample_training_result()) for _ in range(4)]
        mgr.mark_as_best(recs[2].model_id)
        best_count = sum(1 for m in mgr.get_all(include_archived=True) if m.is_best)
        assert best_count == 1

    def test_mark_as_best_persisted_to_registry(self, tmp_path):
        from core.model_manager import ModelManager
        mgr1 = ModelManager(str(tmp_path))
        rec = mgr1.register(_sample_training_result())
        mgr1.mark_as_best(rec.model_id)

        mgr2 = ModelManager(str(tmp_path))
        assert mgr2.get_best() is not None
        assert mgr2.get_best().model_id == rec.model_id


# ---------------------------------------------------------------------------
# ModelManager — archive
# ---------------------------------------------------------------------------

class TestModelManagerArchive:
    def test_archive_sets_archived_flag(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        mgr.archive(rec.model_id)
        found = mgr.get_by_id(rec.model_id)
        assert found.archived is True

    def test_archive_persisted_to_registry(self, tmp_path):
        from core.model_manager import ModelManager
        mgr1 = ModelManager(str(tmp_path))
        rec = mgr1.register(_sample_training_result())
        mgr1.archive(rec.model_id)

        mgr2 = ModelManager(str(tmp_path))
        found = mgr2.get_by_id(rec.model_id)
        assert found.archived is True

    def test_archive_nonexistent_id_does_not_raise(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        mgr.archive("no_such_id")  # should not raise


# ---------------------------------------------------------------------------
# ModelManager — update_metadata
# ---------------------------------------------------------------------------

class TestModelManagerUpdateMetadata:
    def test_update_metadata_changes_name(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        mgr.update_metadata(rec.model_id, name="BetterName")
        assert mgr.get_by_id(rec.model_id).name == "BetterName"

    def test_update_metadata_changes_description(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        rec = mgr.register(_sample_training_result())
        mgr.update_metadata(rec.model_id, description="New description text")
        assert mgr.get_by_id(rec.model_id).description == "New description text"

    def test_update_metadata_persisted_to_registry(self, tmp_path):
        from core.model_manager import ModelManager
        mgr1 = ModelManager(str(tmp_path))
        rec = mgr1.register(_sample_training_result())
        mgr1.update_metadata(rec.model_id, name="SavedName")

        mgr2 = ModelManager(str(tmp_path))
        assert mgr2.get_by_id(rec.model_id).name == "SavedName"

    def test_update_metadata_nonexistent_id_does_not_raise(self, tmp_path):
        from core.model_manager import ModelManager
        mgr = ModelManager(str(tmp_path))
        mgr.update_metadata("ghost", name="x")  # should not raise
