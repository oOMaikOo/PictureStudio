"""
Enhanced training pipeline: early stopping, LR scheduler, mixed precision,
GPU/CPU selection, checkpoint resume, reproducibility.
"""
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

log = logging.getLogger("ImageLabelingStudio.training")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from utils.logging_utils import get_logger
from utils.reproducibility import set_seed, get_software_versions

log = get_logger()


# ------------------------------------------------------------------ Focal Loss

class FocalLoss(nn.Module if HAS_TORCH else object):
    """Focal Loss for single-label classification.

    Reduces the loss contribution of easy (high-confidence) examples so training
    focuses on hard examples. Particularly useful for imbalanced datasets.

    gamma=0 → equivalent to standard CrossEntropyLoss
    gamma=2 → standard Focal Loss (Lin et al., 2017)
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # per-class weights tensor (optional)

    def forward(self, logits, targets):
        import torch.nn.functional as F
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()


# ------------------------------------------------------------------ epoch helpers

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Run one training epoch for single-label classification.

    Supports optional GradScaler for mixed-precision (CUDA AMP).
    Returns (avg_loss, accuracy) as floats.
    """
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return (total_loss / total if total else 0.0,
            correct / total if total else 0.0)


def evaluate(model, loader, criterion, device):
    """
    Evaluate a single-label classifier on *loader* without updating weights.

    Returns (avg_loss, accuracy, all_predictions, all_true_labels).
    """
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return (total_loss / total if total else 0.0,
            correct / total if total else 0.0,
            all_preds, all_labels)


# ------------------------------------------------------------------ Multi-label helpers

def train_one_epoch_multilabel(model, loader, criterion, optimizer, device, scaler=None):
    """Training loop for multi-label (BCEWithLogitsLoss). Returns (loss, hamming_acc)."""
    model.train()
    total_loss = correct_bits = total_bits = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct_bits += (preds == labels).sum().item()
        total_bits += labels.numel()
    n = len(loader.dataset)
    return (total_loss / n if n else 0.0,
            correct_bits / total_bits if total_bits else 0.0)


def evaluate_multilabel(model, loader, criterion, device, threshold: float = 0.5):
    """Multi-label eval. Returns (loss, hamming_acc, pred_matrix, true_matrix)."""
    model.eval()
    total_loss = correct_bits = total_bits = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= threshold).float()
            correct_bits += (preds == labels).sum().item()
            total_bits += labels.numel()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    n = len(loader.dataset)
    return (total_loss / n if n else 0.0,
            correct_bits / total_bits if total_bits else 0.0,
            all_preds, all_labels)


# ------------------------------------------------------------------ EarlyStopping

class EarlyStopping:
    """
    Monitors a validation metric and signals when training should stop.

    Stops when the metric has not improved by more than *min_delta* for
    *patience* consecutive epochs. Designed to track validation accuracy
    (higher is better).
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ------------------------------------------------------------------ TrainingWorker

class TrainingWorker:
    """
    Qt-free training worker: builds datasets, runs the training loop, and
    returns a result dict compatible with TrainingThread.finished.

    Designed to be instantiated inside TrainingThread.run() so that all
    blocking operations happen off the GUI thread.  Callbacks are used instead
    of Qt signals to keep this class dependency-free.
    """

    def __init__(
        self,
        project,
        training_config: Dict,
        save_dir: str,
        progress_callback: Callable = None,
        log_callback: Callable = None,
        stop_flag: Callable = None,
    ):
        """
        Parameters
        ----------
        project           : Current Project instance (read-only during training).
        training_config   : Dict of hyperparameters (matches DEFAULT_TRAIN_CONFIG structure).
        save_dir          : Directory where checkpoints are saved.
        progress_callback : Called with (epoch, total, tl, vl, ta, va) each epoch.
        log_callback      : Called with a log message string each step.
        stop_flag         : Zero-argument callable; returning True aborts training.
        """
        self.project = project
        self.cfg = training_config
        self.save_dir = save_dir
        self._progress = progress_callback
        self._log = log_callback
        self._stop = stop_flag

    def _emit_log(self, msg: str) -> None:
        """Forward a log message to the Python logger and the GUI callback."""
        log.info(msg)
        if self._log:
            self._log(msg)

    def _emit_progress(self, *args) -> None:
        """Forward a progress update to the GUI callback if set."""
        if self._progress:
            self._progress(*args)

    def run(self) -> Dict:
        """
        Execute the full training pipeline and return the result dict.

        Steps: device selection, dataset creation, model + optimiser + scheduler
        setup, epoch loop (with optional AMP, early stopping, and checkpoint
        saving), final test-set evaluation, and dataset snapshot.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch ist nicht installiert.")

        from core.dataset import create_datasets
        from models.classifier import create_model, save_checkpoint

        seed = self.cfg.get("seed", 42)
        set_seed(seed)

        # Device selection
        device_pref = self.cfg.get("device", "auto")
        if device_pref == "cpu":
            device = torch.device("cpu")
        elif device_pref == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_pref == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self._emit_log(f"Gerät: {device}")

        # Mixed precision (only CUDA)
        use_amp = self.cfg.get("mixed_precision", False) and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            self._emit_log("Mixed Precision (AMP) aktiviert.")

        # Multi-label mode
        is_ml = (self.cfg.get("multi_label", False) or
                 getattr(self.project.config, "multi_label", False))
        if is_ml:
            self._emit_log("Modus: Multi-Label-Klassifikation (BCEWithLogitsLoss)")

        # Datasets
        self._emit_log("Erstelle Datasets...")
        use_rois = self.cfg.get("use_rois", True)
        train_ds, val_ds, test_ds, class_names = create_datasets(
            self.project, self.cfg, use_rois=use_rois
        )
        self._emit_log(
            f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | "
            f"Klassen: {class_names}"
        )

        batch_size = self.cfg.get("batch_size", 16)

        # Weighted sampler + loss for class imbalance (single-label only)
        class_weights_tensor = None
        if self.cfg.get("class_balance", False) and not is_ml and hasattr(train_ds, "samples"):
            from torch.utils.data import WeightedRandomSampler
            class_indices = [s[2] for s in train_ds.samples]
            counts = torch.bincount(torch.tensor(class_indices, dtype=torch.long))
            weights = 1.0 / counts.float().clamp(min=1)
            sample_weights = weights[torch.tensor(class_indices, dtype=torch.long)]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
            # Normalise weights so the loss scale stays comparable across runs
            class_weights_tensor = (weights / weights.sum() * len(weights)).to(device)
            self._emit_log(
                f"Klassenausgleich aktiv (WeightedSampler + gewichteter Loss): "
                f"{dict(zip(class_names, counts.tolist()))}"
            )
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model
        model = create_model(
            self.cfg.get("model_type", "resnet18"),
            len(class_names),
            pretrained=self.cfg.get("use_pretrained", True),
        ).to(device)

        use_focal = self.cfg.get("focal_loss", False) and not is_ml
        if is_ml:
            criterion = nn.BCEWithLogitsLoss()
        elif use_focal:
            gamma = self.cfg.get("focal_gamma", 2.0)
            criterion = FocalLoss(gamma=gamma, weight=class_weights_tensor)
            self._emit_log(f"Focal Loss aktiv (γ={gamma:.1f})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        lr = self.cfg.get("learning_rate", 0.001)
        opt_name = self.cfg.get("optimizer", "adam").lower()
        wd = self.cfg.get("weight_decay", 1e-4)

        if opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Scheduler
        sched_name = self.cfg.get("scheduler", "reduce_on_plateau")
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.get("epochs", 20)
            )
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )

        # Early stopping
        early_stop_patience = self.cfg.get("early_stopping_patience", 0)
        early_stopping = EarlyStopping(patience=early_stop_patience) if early_stop_patience > 0 else None

        # Resume from checkpoint
        start_epoch = 1
        resume_path = self.cfg.get("resume_checkpoint", "")
        if resume_path and os.path.exists(resume_path):
            self._emit_log(f"Setze Training fort von: {resume_path}")
            from models.classifier import load_checkpoint
            meta = load_checkpoint(model, resume_path)
            start_epoch = meta.get("epoch", 0) + 1
            model.to(device)

        epochs = self.cfg.get("epochs", 20)
        os.makedirs(self.save_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_val_acc = 0.0
        best_model_path = ""
        history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]}

        for epoch in range(start_epoch, epochs + 1):
            if self._stop and self._stop():
                self._emit_log("Training vom Benutzer gestoppt.")
                break

            t0 = time.time()
            if is_ml:
                tl, ta = train_one_epoch_multilabel(model, train_loader, criterion, optimizer, device, scaler)
                vl, va, _, _ = evaluate_multilabel(model, val_loader, criterion, device)
            else:
                tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
                vl, va, _, _ = evaluate(model, val_loader, criterion, device)

            cur_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["train_acc"].append(ta)
            history["val_acc"].append(va)
            history["lr"].append(cur_lr)

            # Scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl)
            else:
                scheduler.step()

            elapsed = time.time() - t0
            self._emit_log(
                f"E {epoch:3d}/{epochs} | "
                f"Train {tl:.4f}/{ta*100:.1f}% | "
                f"Val {vl:.4f}/{va*100:.1f}% | "
                f"LR {cur_lr:.2e} | {elapsed:.1f}s"
            )
            self._emit_progress(epoch, epochs, tl, vl, ta, va)

            # Checkpoint: save every epoch (overwrite "last")
            last_ckpt = os.path.join(self.save_dir, f"last_{run_id}.pth")
            save_checkpoint(model, last_ckpt, {
                "class_names": class_names, "model_type": self.cfg.get("model_type"),
                "image_size": self.cfg.get("image_size", 224), "epoch": epoch, "run_id": run_id,
            })

            # Save best
            if va >= best_val_acc:
                best_val_acc = va
                best_model_path = os.path.join(self.save_dir, f"best_{run_id}.pth")
                save_checkpoint(model, best_model_path, {
                    "class_names": class_names,
                    "model_type": self.cfg.get("model_type", "resnet18"),
                    "image_size": self.cfg.get("image_size", 224),
                    "epoch": epoch, "val_acc": va, "run_id": run_id,
                })
                self._emit_log(f"  ✓ Bestes Modell (Val-Acc: {va*100:.2f}%)")

            # Early stopping
            if early_stopping and early_stopping.step(va):
                self._emit_log(
                    f"Early Stopping nach {early_stop_patience} Epochen ohne Verbesserung."
                )
                break

        # ── Best val metrics ────────────────────────────────────────────────
        from core.metrics import compute_metrics
        best_epoch_idx = history["val_acc"].index(max(history["val_acc"])) if history["val_acc"] else 0
        best_val_metrics = {
            "val_acc":  history["val_acc"][best_epoch_idx]  if history["val_acc"]  else 0.0,
            "val_loss": history["val_loss"][best_epoch_idx] if history["val_loss"] else 0.0,
            "train_acc":history["train_acc"][best_epoch_idx] if history["train_acc"] else 0.0,
            "epoch":    best_epoch_idx + 1,
        }

        # ── Load best checkpoint before test evaluation ──────────────────────
        if best_model_path and os.path.exists(best_model_path):
            self._emit_log("Lade bestes Modell für Test-Evaluation…")
            from models.classifier import load_checkpoint
            load_checkpoint(model, best_model_path)
            model.to(device)

        # ── Final evaluation on held-out test set ────────────────────────────
        self._emit_log(f"Evaluiere Test-Set ({len(test_ds)} Bilder, nie während Training gesehen)…")
        if is_ml:
            _, test_acc, test_preds, test_labels = evaluate_multilabel(
                model, test_loader, criterion, device)
            from core.metrics import compute_multilabel_metrics
            test_metrics = compute_multilabel_metrics(test_labels, test_preds, class_names)
            self._emit_log(
                f"Hamming-Accuracy: {test_acc*100:.2f}%  |  "
                f"Test-F1 (macro): {test_metrics.get('macro_f1', 0)*100:.2f}%"
            )
            test_predictions = [
                {
                    "path": test_ds.samples[i][0],
                    "true_labels": [class_names[j] for j, v in enumerate(test_labels[i])
                                    if round(v) == 1],
                    "pred_labels": [class_names[j] for j, v in enumerate(test_preds[i])
                                    if round(v) == 1],
                }
                for i in range(len(test_preds))
            ]
        else:
            _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
            test_metrics = compute_metrics(test_labels, test_preds, class_names)
            self._emit_log(
                f"Test-Accuracy: {test_acc*100:.2f}%  |  "
                f"Test-F1 (macro): {test_metrics.get('macro_f1', 0)*100:.2f}%"
            )
            test_predictions = [
                {
                    "path":       test_ds.samples[i][0],
                    "true_label": class_names[test_labels[i]],
                    "pred_label": class_names[test_preds[i]],
                }
                for i in range(len(test_preds))
            ]

        timestamp = datetime.now().isoformat()
        result = {
            "run_id":           run_id,
            "timestamp":        timestamp,
            "model_type":       self.cfg.get("model_type", "resnet18"),
            "hyperparameters":  dict(self.cfg),
            "class_names":      class_names,
            "history":          history,
            "metrics":          test_metrics,       # kept for backward compat
            "test_metrics":     test_metrics,
            "test_predictions": test_predictions,
            "best_val_metrics": best_val_metrics,
            "best_model_path":  best_model_path,
            "final_model_path": last_ckpt,
            "software_versions": get_software_versions(),
            "train_size":       len(train_ds),
            "val_size":         len(val_ds),
            "test_size":        len(test_ds),
            "device":           str(device),
            "early_stopped":    early_stopping.triggered if early_stopping else False,
            "multi_label":      is_ml,
        }

        # ── Dataset snapshot: record which images+labels were used ───────────
        import json as _json
        snapshot = {
            "run_id":    run_id,
            "timestamp": timestamp,
            "samples":   [
                {"path": s[0], "label": class_names[s[2]], "split": split}
                for ds, split in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]
                for s in ds.samples
            ],
        }
        snap_path = os.path.join(self.save_dir, f"dataset_snapshot_{run_id}.json")
        try:
            with open(snap_path, "w", encoding="utf-8") as fh:
                _json.dump(snapshot, fh, indent=2, ensure_ascii=False)
            result["dataset_snapshot_path"] = snap_path
            self._emit_log(f"Dataset-Snapshot gespeichert: {snap_path}")
        except Exception as exc:
            self._emit_log(f"Snapshot-Warnung: {exc}")

        return result
