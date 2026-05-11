"""
Model library: track, compare, load, export all trained models for a project.
"""
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from utils.logging_utils import get_logger

log = get_logger()

MODEL_REGISTRY_FILE = "model_registry.json"


class ModelRecord:
    """Metadata for a single trained model."""

    def __init__(self, data: Dict):
        self.model_id: str = data.get("model_id", "")
        self.name: str = data.get("name", "")
        self.description: str = data.get("description", "")
        self.architecture: str = data.get("architecture", "")
        self.version: str = data.get("version", "1.0")
        self.created_at: str = data.get("created_at", "")
        self.model_path: str = data.get("model_path", "")
        self.onnx_path: str = data.get("onnx_path", "")
        self.class_names: List[str] = data.get("class_names", [])
        self.image_size: int = data.get("image_size", 224)
        self.hyperparameters: Dict = data.get("hyperparameters", {})
        self.metrics: Dict = data.get("metrics", {})
        self.train_size: int = data.get("train_size", 0)
        self.val_size: int = data.get("val_size", 0)
        self.test_size: int = data.get("test_size", 0)
        self.is_best: bool = data.get("is_best", False)
        self.archived: bool = data.get("archived", False)
        self.run_id: str = data.get("run_id", "")
        self.software_versions: Dict = data.get("software_versions", {})

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "architecture": self.architecture,
            "version": self.version,
            "created_at": self.created_at,
            "model_path": self.model_path,
            "onnx_path": self.onnx_path,
            "class_names": self.class_names,
            "image_size": self.image_size,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "is_best": self.is_best,
            "archived": self.archived,
            "run_id": self.run_id,
            "software_versions": self.software_versions,
        }

    def accuracy_str(self) -> str:
        acc = self.metrics.get("accuracy", 0)
        return f"{acc*100:.2f}%"

    def f1_str(self) -> str:
        f1 = self.metrics.get("macro_f1", 0)
        return f"{f1*100:.2f}%"


class ModelManager:
    """Manages the model library for a project directory."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self._registry_path = os.path.join(models_dir, MODEL_REGISTRY_FILE)
        self._models: List[ModelRecord] = []
        self._load()

    # ------------------------------------------------------------------ registry

    def _load(self) -> None:
        if not os.path.exists(self._registry_path):
            return
        try:
            with open(self._registry_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._models = [ModelRecord(d) for d in data.get("models", [])]
        except Exception as exc:
            log.warning("Modell-Registry konnte nicht geladen werden: %s", exc)
            self._models = []

    def _save(self) -> None:
        data = {"models": [m.to_dict() for m in self._models]}
        with open(self._registry_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ CRUD

    def register(self, training_result: Dict, name: str = "", description: str = "") -> ModelRecord:
        """Add a training result to the registry."""
        import uuid
        model_id = str(uuid.uuid4())[:8]
        run_id = training_result.get("run_id", model_id)
        arch = training_result.get("model_type", "unknown")
        created = training_result.get("timestamp", datetime.now().isoformat())

        record = ModelRecord({
            "model_id": model_id,
            "name": name or f"{arch}_{run_id}",
            "description": description,
            "architecture": arch,
            "version": str(len(self._models) + 1),
            "created_at": created,
            "model_path": training_result.get("best_model_path", ""),
            "class_names": training_result.get("class_names", []),
            "image_size": training_result.get("hyperparameters", {}).get("image_size", 224),
            "hyperparameters": training_result.get("hyperparameters", {}),
            "metrics": training_result.get("metrics", {}),
            "train_size": training_result.get("train_size", 0),
            "val_size": training_result.get("val_size", 0),
            "test_size": training_result.get("test_size", 0),
            "run_id": run_id,
            "software_versions": training_result.get("software_versions", {}),
        })
        self._models.append(record)
        self._save()
        log.info("Modell registriert: %s (%s)", record.name, model_id)
        return record

    def get_all(self, include_archived: bool = False) -> List[ModelRecord]:
        if include_archived:
            return list(self._models)
        return [m for m in self._models if not m.archived]

    def get_by_id(self, model_id: str) -> Optional[ModelRecord]:
        return next((m for m in self._models if m.model_id == model_id), None)

    def get_best(self) -> Optional[ModelRecord]:
        return next((m for m in self._models if m.is_best), None)

    def mark_as_best(self, model_id: str) -> None:
        for m in self._models:
            m.is_best = m.model_id == model_id
        self._save()

    def archive(self, model_id: str) -> None:
        m = self.get_by_id(model_id)
        if m:
            m.archived = True
            self._save()

    def delete(self, model_id: str, delete_file: bool = False) -> None:
        m = self.get_by_id(model_id)
        if not m:
            return
        if delete_file and os.path.exists(m.model_path):
            try:
                os.remove(m.model_path)
            except OSError:
                pass
        self._models = [x for x in self._models if x.model_id != model_id]
        self._save()

    def update_metadata(self, model_id: str, name: str = None, description: str = None) -> None:
        m = self.get_by_id(model_id)
        if m:
            if name is not None:
                m.name = name
            if description is not None:
                m.description = description
            self._save()

    # ------------------------------------------------------------------ export

    def export_onnx(self, model_id: str) -> str:
        """Export model to ONNX format. Returns onnx_path."""
        try:
            import torch
            from models.classifier import create_model, load_checkpoint
        except ImportError as exc:
            raise RuntimeError(f"PyTorch nicht verfügbar: {exc}")

        m = self.get_by_id(model_id)
        if not m:
            raise ValueError(f"Modell {model_id} nicht gefunden.")
        if not os.path.exists(m.model_path):
            raise FileNotFoundError(f"Modelldatei nicht gefunden: {m.model_path}")

        model = create_model(m.architecture, len(m.class_names), pretrained=False)
        load_checkpoint(model, m.model_path)
        model.eval()

        dummy = torch.randn(1, 3, m.image_size, m.image_size)
        onnx_path = m.model_path.replace(".pth", ".onnx")

        torch.onnx.export(
            model, dummy, onnx_path,
            export_params=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        m.onnx_path = onnx_path
        self._save()
        log.info("ONNX exportiert: %s", onnx_path)
        return onnx_path

    # ------------------------------------------------------------------ comparison

    def compare(self, model_ids: List[str]) -> List[Dict]:
        """Return metric comparison table for given model IDs."""
        rows = []
        for mid in model_ids:
            m = self.get_by_id(mid)
            if m:
                rows.append({
                    "name": m.name,
                    "architecture": m.architecture,
                    "accuracy": m.metrics.get("accuracy", 0),
                    "f1": m.metrics.get("macro_f1", 0),
                    "precision": m.metrics.get("macro_precision", 0),
                    "recall": m.metrics.get("macro_recall", 0),
                    "train_size": m.train_size,
                    "created_at": m.created_at[:19],
                    "is_best": m.is_best,
                })
        return rows
