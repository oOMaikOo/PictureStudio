from __future__ import annotations
import logging
from typing import Optional, Callable
import numpy as np

log = logging.getLogger(__name__)


class HPTWorker:
    """Plain worker für Hyperparameter-Tuning via Optuna."""

    SEARCH_SPACE = {
        "lr": (1e-4, 1e-2, "log"),
        "batch_size": [8, 16, 32],
        "model_type": ["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0"],
        "optimizer": ["adam", "sgd"],
    }

    def __init__(
        self,
        project,
        n_trials: int = 20,
        timeout: float = 300.0,
        device: str = "cpu",
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        self._project = project
        self._n_trials = n_trials
        self._timeout = timeout
        self._device = device
        self._progress_callback = progress_callback
        self._study = None

    def run(self) -> dict:
        """
        Führt Optuna-Studie durch. Gibt zurück:
        {"best_params": dict, "best_value": float, "trials": list[dict]}
        Jeder Trial: {"number": int, "value": float|None, "params": dict, "state": str}
        Raises ImportError wenn optuna nicht installiert.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError(
                "optuna ist nicht installiert. Bitte: pip install optuna"
            ) from exc

        from core.training import TrainingWorker

        study = optuna.create_study(direction="maximize")
        self._study = study

        def objective(trial):
            cfg = {
                "learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                "model_type": trial.suggest_categorical(
                    "model_type", ["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0"]
                ),
                "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
                "epochs": 5,
                "image_size": 224,
                "seed": 42,
                "train_split": 0.7,
                "val_split": 0.2,
                "test_split": 0.1,
                "use_pretrained": True,
                "device": self._device,
                "augmentation": {
                    "rotation": True,
                    "flip": True,
                    "brightness": False,
                    "contrast": False,
                    "scale": False,
                },
            }
            try:
                import tempfile
                save_dir = tempfile.mkdtemp(prefix="hpt_trial_")
                worker = TrainingWorker(self._project, cfg, save_dir)
                result = worker.run()
                val_acc = result.get("metrics", {}).get("accuracy", 0.0)
                return float(val_acc)
            except Exception as exc:
                log.warning("HPT Trial %d fehlgeschlagen: %s", trial.number, exc)
                return 0.0
            finally:
                if self._progress_callback:
                    best = study.best_value if study.trials else 0.0
                    self._progress_callback(trial.number + 1, self._n_trials, best)

        study.optimize(objective, n_trials=self._n_trials, timeout=self._timeout)

        trials = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ]
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": trials,
        }

    def stop(self) -> None:
        """Stoppt die laufende Optuna-Studie (graceful)."""
        if self._study:
            self._study.stop()


def _make_hpt_thread():
    """Build and return the concrete QThread subclass for HPT."""
    from PySide6.QtCore import QThread, Signal as _Signal

    class _HT(QThread):
        progress = _Signal(int, int, float)  # trial_num, total_trials, best_value
        finished = _Signal(dict)
        error = _Signal(str)

        def __init__(self, project, n_trials: int = 20, timeout: float = 300.0,
                     device: str = "cpu", parent=None):
            super().__init__(parent)
            self._worker = HPTWorker(
                project, n_trials, timeout, device,
                progress_callback=self._on_progress,
            )

        def _on_progress(self, trial_num: int, total: int, best: float) -> None:
            self.progress.emit(trial_num, total, best)

        def run(self) -> None:
            try:
                result = self._worker.run()
                self.finished.emit(result)
            except Exception as exc:
                self.error.emit(str(exc))

        def stop(self) -> None:
            self._worker.stop()

    return _HT


try:
    _HPTQThread = _make_hpt_thread()

    class HPTThread(_HPTQThread):  # type: ignore[no-redef]
        """QThread der HPTWorker.run() im Hintergrund ausführt.

        Signals: progress(int, int, float), finished(dict), error(str)
        """

except Exception:
    class HPTThread:  # type: ignore[no-redef]
        """Stub für Umgebungen ohne PySide6."""

        def __init__(self, project, n_trials=20, timeout=300, device="cpu", parent=None):
            self._worker = HPTWorker(project, n_trials, timeout, device)

        def run(self):
            return self._worker.run()
