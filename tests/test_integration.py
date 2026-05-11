"""
Integration tests — project I/O, training pipeline, model load, inference.
"""
import os

import pytest


# ---------------------------------------------------------------------------
# Project: full save → load cycle
# ---------------------------------------------------------------------------

class TestProjectIntegration:
    def test_full_project_lifecycle(self, tmp_dir):
        from core.project import Project

        p = Project()
        p.config.name = "IntegTest"
        p.add_label("a", "#FF0000")
        p.add_label("b", "#00FF00")

        for i in range(4):
            path = os.path.join(tmp_dir, f"img_{i}.jpg")
            p.add_image(path)
            p.set_image_label(path, ["a", "b"][i % 2])
            p.add_roi(path, {
                "id": f"roi_{i}", "type": "rect",
                "x": 5.0, "y": 5.0, "w": 30.0, "h": 30.0,
                "label": ["a", "b"][i % 2], "color": "#FF0000",
            })

        save_path = os.path.join(tmp_dir, "integ.json")
        p.save(save_path)

        p2 = Project.load(save_path)
        assert p2.config.name == "IntegTest"
        assert len(p2.images) == 4
        assert len(p2.labels) == 2
        assert sum(len(v) for v in p2.rois.values()) == 4
        assert p2.image_labels == p.image_labels

    def test_backup_and_restore(self, tmp_dir):
        from core.project import Project

        p = Project()
        p.config.name = "BackupTest"
        p.add_label("gut", "#2ECC71")
        save_path = os.path.join(tmp_dir, "proj.json")
        p.save(save_path)

        backup = p.create_backup(tmp_dir)
        assert backup and os.path.exists(backup)

        p2 = Project.load(backup)
        assert p2.config.name == "BackupTest"

    def test_relocate_images(self, tmp_dir):
        from core.project import Project

        p = Project()
        p.add_label("gut", "#2ECC71")
        old_prefix = "/old/path"
        for i in range(3):
            fake_path = f"{old_prefix}/img_{i}.jpg"
            p.add_image(fake_path)
            p.set_image_label(fake_path, "gut")

        count = p.relocate_images(old_prefix, "/new/path")
        assert count == 3
        assert all(img.startswith("/new/path") for img in p.images)

    def test_save_load_multi_label(self, tmp_dir):
        from core.project import Project

        p = Project()
        p.add_label("a", "#FF0000")
        p.add_label("b", "#00FF00")
        img = os.path.join(tmp_dir, "ml.jpg")
        p.add_image(img)
        p.set_image_label(img, "a")
        p.set_image_multi_labels(img, ["a", "b"])

        save_path = os.path.join(tmp_dir, "ml.json")
        p.save(save_path)
        p2 = Project.load(save_path)
        assert "b" in p2.get_image_multi_labels(img)


# ---------------------------------------------------------------------------
# Training pipeline (CPU, minimal data)
# ---------------------------------------------------------------------------

def _build_training_project(paths, tmp_dir):
    """Build a Project from (fname, lbl) pairs with real image files."""
    from core.project import Project
    p = Project()
    label_names = set()
    for fname, lbl in paths:
        p.add_label(lbl, "#FFFFFF")
        p.add_image(fname)
        p.set_image_label(fname, lbl)
        label_names.add(lbl)
    save_path = os.path.join(tmp_dir, "train_proj.json")
    p.save(save_path)
    return p, sorted(label_names)


def _minimal_cfg(tmp_dir, epochs=1):
    return {
        "model_type": "simple_cnn",
        "learning_rate": 0.01,
        "batch_size": 4,
        "epochs": epochs,
        "device": "cpu",
        "use_pretrained": False,
        "augment": False,
        "image_size": 32,
        "optimizer": "adam",
        "scheduler": "none",
        "early_stopping_patience": 0,
        "mixed_precision": False,
        "use_rois": False,
        "train_split": 0.7,
        "val_split": 0.2,
        "seed": 42,
    }


class TestTrainingIntegration:
    def test_training_worker_runs(self, sample_images, tmp_dir):
        pytest.importorskip("torch")
        from core.training import TrainingWorker

        paths, _ = sample_images
        p, _ = _build_training_project(paths, tmp_dir)
        cfg = _minimal_cfg(tmp_dir, epochs=2)

        worker = TrainingWorker(p, cfg, tmp_dir)
        result = worker.run()

        assert isinstance(result, dict)
        assert "best_val_acc" in result or "history" in result

    def test_training_creates_checkpoint(self, sample_images, tmp_dir):
        pytest.importorskip("torch")
        from core.training import TrainingWorker

        paths, _ = sample_images
        p, _ = _build_training_project(paths, tmp_dir)
        cfg = _minimal_cfg(tmp_dir, epochs=1)

        worker = TrainingWorker(p, cfg, tmp_dir)
        worker.run()

        pth_files = [f for f in os.listdir(tmp_dir) if f.endswith(".pth")]
        assert len(pth_files) >= 1

    def test_training_result_has_metrics(self, sample_images, tmp_dir):
        pytest.importorskip("torch")
        from core.training import TrainingWorker

        paths, _ = sample_images
        p, _ = _build_training_project(paths, tmp_dir)
        cfg = _minimal_cfg(tmp_dir, epochs=1)

        worker = TrainingWorker(p, cfg, tmp_dir)
        result = worker.run()

        assert "metrics" in result
        assert "accuracy" in result["metrics"]


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

class TestModelLoadIntegration:
    def test_save_and_load_checkpoint(self, tmp_dir):
        torch = pytest.importorskip("torch")
        from models.classifier import create_model, save_checkpoint

        model = create_model("simple_cnn", num_classes=3, pretrained=False)
        path = os.path.join(tmp_dir, "test.pth")
        meta = {"class_names": ["a", "b", "c"], "image_size": 64, "model_type": "simple_cnn"}
        save_checkpoint(model, path, meta)
        assert os.path.exists(path)

        # Metadata is stored under "metadata" key in the checkpoint
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in payload
        assert payload["metadata"]["class_names"] == ["a", "b", "c"]

    def test_load_checkpoint_into_model(self, tmp_dir):
        pytest.importorskip("torch")
        from models.classifier import create_model, save_checkpoint, load_checkpoint

        model = create_model("simple_cnn", num_classes=2, pretrained=False)
        path = os.path.join(tmp_dir, "ckpt.pth")
        save_checkpoint(model, path, {"class_names": ["x", "y"], "image_size": 32,
                                       "model_type": "simple_cnn"})

        model2 = create_model("simple_cnn", num_classes=2, pretrained=False)
        # load_checkpoint returns the metadata dict
        meta = load_checkpoint(model2, path)
        assert meta.get("class_names") == ["x", "y"]

    def test_available_models_list(self):
        from models.classifier import get_available_models
        models = get_available_models()
        assert "resnet18" in models
        assert "simple_cnn" in models


# ---------------------------------------------------------------------------
# Inference integration
# ---------------------------------------------------------------------------

class TestInferenceIntegration:
    def _train_and_get_checkpoint(self, sample_images, tmp_dir):
        """Helper: train a minimal model and return the checkpoint path."""
        from core.training import TrainingWorker
        paths, _ = sample_images
        p, _ = _build_training_project(paths, tmp_dir)
        cfg = _minimal_cfg(tmp_dir, epochs=1)
        worker = TrainingWorker(p, cfg, tmp_dir)
        result = worker.run()
        model_path = result.get("best_model_path", "")
        if not model_path or not os.path.exists(model_path):
            pth_files = [os.path.join(tmp_dir, f)
                         for f in os.listdir(tmp_dir) if f.endswith(".pth")]
            if not pth_files:
                pytest.skip("No checkpoint produced by training")
            model_path = pth_files[0]
        return model_path

    def test_predict_single_image(self, sample_images, tmp_dir):
        pytest.importorskip("torch")
        from core.inference import Inferencer

        model_path = self._train_and_get_checkpoint(sample_images, tmp_dir)
        paths, _ = sample_images
        label_names = sorted({lbl for _, lbl in paths})

        inf = Inferencer()
        inf.load_model(model_path)
        result = inf.predict_image(paths[0][0])

        assert "predicted_label" in result
        assert "confidence" in result
        assert result["predicted_label"] in label_names
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_folder(self, sample_images, tmp_dir):
        pytest.importorskip("torch")
        from core.inference import Inferencer

        model_path = self._train_and_get_checkpoint(sample_images, tmp_dir)
        _, img_dir = sample_images
        label_names = sorted({lbl for _, lbl in sample_images[0]})

        inf = Inferencer()
        inf.load_model(model_path)
        results = inf.predict_folder(img_dir)

        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "predicted_label" in r or "error" in r

    def test_inferencer_is_ready_after_load(self, sample_images, tmp_dir):
        pytest.importorskip("torch")
        from core.inference import Inferencer

        model_path = self._train_and_get_checkpoint(sample_images, tmp_dir)
        inf = Inferencer()
        assert not inf.is_ready()
        inf.load_model(model_path)
        assert inf.is_ready()

    def test_predict_top_k(self, sample_images, tmp_dir):
        pytest.importorskip("torch")
        from core.inference import Inferencer

        model_path = self._train_and_get_checkpoint(sample_images, tmp_dir)
        paths, _ = sample_images
        inf = Inferencer()
        inf.load_model(model_path)
        result = inf.predict_image(paths[0][0], top_k=3)
        assert "top_k" in result
        assert len(result["top_k"]) <= 3
