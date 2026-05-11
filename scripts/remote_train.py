#!/usr/bin/env python3
"""
Standalone remote training script.
Uploaded to the GPU server and executed there — no local app imports needed.

Usage:
    python remote_train.py /path/to/config.json

Config JSON schema:
{
  "samples":        [["relative/path/img.jpg", "label_name"], ...],
  "class_names":    ["gut", "schlecht", "neutral"],
  "image_base_dir": "/remote/workdir/images",
  "output_dir":     "/remote/workdir/output",
  "model_type":     "resnet18",
  "use_pretrained": true,
  "image_size":     224,
  "batch_size":     16,
  "epochs":         20,
  "learning_rate":  0.001,
  "optimizer":      "adam",
  "scheduler":      "reduce_on_plateau",
  "early_stopping_patience": 5,
  "seed":           42,
  "device":         "auto",
  "mixed_precision": false,
  "train_split":    0.7,
  "val_split":      0.2,
  "augmentation":   {"flip": true, "rotation": true, "brightness": true, "scale": false},
  "run_id":         "20250507_143000"
}

Output lines (parsed by RemoteTrainingThread):
  LOG  <message>
  PROGRESS epoch=1 total=20 train_loss=0.69 val_loss=0.71 train_acc=0.50 val_acc=0.48 lr=0.001
  BEST best_model=<path> val_acc=0.92
  DONE run_id=<id> best_model=<path> val_acc=0.92 accuracy=0.91 f1=0.90
  ERROR <message>
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _log(msg: str) -> None:
    print(f"LOG  {msg}", flush=True)


def _progress(epoch, total, tl, vl, ta, va, lr) -> None:
    print(
        f"PROGRESS epoch={epoch} total={total} "
        f"train_loss={tl:.6f} val_loss={vl:.6f} "
        f"train_acc={ta:.6f} val_acc={va:.6f} lr={lr:.8f}",
        flush=True,
    )


def _done(result: dict) -> None:
    print(
        f"DONE run_id={result['run_id']} "
        f"best_model={result['best_model_path']} "
        f"val_acc={result['best_val_acc']:.6f} "
        f"accuracy={result['metrics'].get('accuracy', 0):.6f} "
        f"f1={result['metrics'].get('macro_f1', 0):.6f}",
        flush=True,
    )


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------- Dataset

class ImageDataset:
    def __init__(self, samples: List[Tuple], image_size: int, augment: bool, aug_cfg: Dict):
        try:
            import torch
            from torch.utils.data import Dataset
            from torchvision import transforms
            from PIL import Image

            class _DS(Dataset):
                def __init__(self_, s, transform):
                    self_.samples = s
                    self_.transform = transform

                def __len__(self_):
                    return len(self_.samples)

                def __getitem__(self_, idx):
                    path, cls_idx = self_.samples[idx]
                    img = Image.open(path).convert("RGB")
                    return self_.transform(img), cls_idx

            aug = []
            if augment:
                if aug_cfg.get("flip", True):
                    aug.append(transforms.RandomHorizontalFlip())
                if aug_cfg.get("rotation", False):
                    aug.append(transforms.RandomRotation(15))
                if aug_cfg.get("brightness", False):
                    aug.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
                if aug_cfg.get("scale", False):
                    aug.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
                else:
                    aug.append(transforms.Resize((image_size, image_size)))
            else:
                aug.append(transforms.Resize((image_size, image_size)))

            aug += [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            self.dataset = _DS(samples, transforms.Compose(aug))
        except Exception as exc:
            raise RuntimeError(f"Dataset-Fehler: {exc}") from exc

    def __len__(self):
        return len(self.dataset)

    def as_loader(self, batch_size: int, shuffle: bool):
        from torch.utils.data import DataLoader
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def split_samples(samples, train_ratio, val_ratio, seed):
    random.seed(seed)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    train, val, test = [], [], []
    for cls_samples in by_class.values():
        random.shuffle(cls_samples)
        n = len(cls_samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        train.extend(cls_samples[:n_train])
        val.extend(cls_samples[n_train : n_train + n_val])
        test.extend(cls_samples[n_train + n_val :])
    random.shuffle(train)
    return train, val, test


# --------------------------------------------------------- Model

def create_model(model_type: str, num_classes: int, pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    weights = "DEFAULT" if pretrained else None
    model_type = model_type.lower()

    if model_type == "resnet18":
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_type == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_type == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_type == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        # SimpleCNN fallback
        m = _SimpleCNN(num_classes)
    return m


class _SimpleCNN:
    def __new__(cls, num_classes):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(4),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5), nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
                    nn.Dropout(0.3), nn.Linear(512, nc),
                )

            def forward(self, x):
                return self.classifier(self.features(x).view(x.size(0), -1))

        return _Net(num_classes)


# --------------------------------------------------------- Training loop

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    import torch
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    import torch
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def compute_simple_metrics(true_labels, pred_labels, n_classes):
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    total = len(true_labels)
    correct = sum(cm[i][i] for i in range(n_classes))
    accuracy = correct / total if total else 0.0
    f1s = []
    for i in range(n_classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n_classes) if j != i)
        fn = sum(cm[i][j] for j in range(n_classes) if j != i)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = sum(f1s) / n_classes if n_classes else 0.0
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4), "confusion_matrix": cm}


# --------------------------------------------------------- Main

def main(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    run_id = cfg.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed = cfg.get("seed", 42)
    set_seed(seed)

    import torch
    import torch.nn as nn

    # Device
    device_pref = cfg.get("device", "auto")
    if device_pref == "cpu":
        device = torch.device("cpu")
    elif device_pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    _log(f"Gerät: {device}" + (f" ({gpu_name})" if gpu_name else ""))

    # Dataset
    class_names = cfg["class_names"]
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    image_base = cfg["image_base_dir"]

    raw_samples = []
    for rel_path, label in cfg["samples"]:
        abs_path = os.path.join(image_base, rel_path)
        if label in label_to_idx and os.path.isfile(abs_path):
            raw_samples.append((abs_path, label_to_idx[label]))

    if not raw_samples:
        print("ERROR Keine Trainingsbilder gefunden. Prüfe image_base_dir und samples.", flush=True)
        sys.exit(1)

    _log(f"{len(raw_samples)} Bilder, {len(class_names)} Klassen: {class_names}")

    train_ratio = cfg.get("train_split", 0.7)
    val_ratio = cfg.get("val_split", 0.2)
    train_s, val_s, test_s = split_samples(raw_samples, train_ratio, val_ratio, seed)
    _log(f"Split — Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    aug_cfg = cfg.get("augmentation", {})
    img_size = cfg.get("image_size", 224)
    batch = cfg.get("batch_size", 16)

    train_ds = ImageDataset(train_s, img_size, augment=True, aug_cfg=aug_cfg)
    val_ds   = ImageDataset(val_s,   img_size, augment=False, aug_cfg={})
    test_ds  = ImageDataset(test_s,  img_size, augment=False, aug_cfg={})

    train_loader = train_ds.as_loader(batch, shuffle=True)
    val_loader   = val_ds.as_loader(batch, shuffle=False)
    test_loader  = test_ds.as_loader(batch, shuffle=False)

    # Model
    model = create_model(
        cfg.get("model_type", "resnet18"),
        len(class_names),
        pretrained=cfg.get("use_pretrained", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = cfg.get("learning_rate", 0.001)
    wd = cfg.get("weight_decay", 1e-4)
    opt_name = cfg.get("optimizer", "adam").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    sched_name = cfg.get("scheduler", "reduce_on_plateau")
    epochs = cfg.get("epochs", 20)
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        _log("Mixed Precision (AMP) aktiviert.")

    patience = cfg.get("early_stopping_patience", 0)
    es_counter = 0
    es_best = None

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    best_val_acc = 0.0
    best_model_path = ""

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(vl)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        _log(
            f"E {epoch:3d}/{epochs} | "
            f"Train {tl:.4f}/{ta*100:.1f}% | "
            f"Val {vl:.4f}/{va*100:.1f}% | "
            f"LR {cur_lr:.2e} | {elapsed:.1f}s"
        )
        _progress(epoch, epochs, tl, vl, ta, va, cur_lr)

        # Save last checkpoint
        last_path = os.path.join(output_dir, f"last_{run_id}.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "metadata": {
                "class_names": class_names,
                "model_type": cfg.get("model_type", "resnet18"),
                "image_size": img_size,
                "epoch": epoch,
                "run_id": run_id,
                "val_acc": va,
            },
        }, last_path)

        if va >= best_val_acc:
            best_val_acc = va
            best_model_path = os.path.join(output_dir, f"best_{run_id}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "class_names": class_names,
                    "model_type": cfg.get("model_type", "resnet18"),
                    "image_size": img_size,
                    "epoch": epoch,
                    "run_id": run_id,
                    "val_acc": va,
                },
            }, best_model_path)
            print(f"BEST best_model={best_model_path} val_acc={va:.6f}", flush=True)

        # Early stopping
        if patience > 0:
            if es_best is None or va > es_best + 1e-4:
                es_best = va
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= patience:
                    _log(f"Early Stopping nach {patience} Epochen ohne Verbesserung.")
                    break

    # Final test evaluation
    _log("Evaluiere Testdaten...")
    _, test_acc, test_preds, test_labels_list = evaluate(model, test_loader, criterion, device)
    metrics = compute_simple_metrics(test_labels_list, test_preds, len(class_names))
    _log(f"Test-Accuracy: {test_acc*100:.2f}%  |  F1: {metrics['macro_f1']*100:.2f}%")

    result = {
        "run_id": run_id,
        "best_model_path": best_model_path,
        "best_val_acc": best_val_acc,
        "class_names": class_names,
        "metrics": metrics,
        "model_type": cfg.get("model_type", "resnet18"),
    }
    _done(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remote_train.py <config.json>", flush=True)
        sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception as exc:
        import traceback
        print(f"ERROR {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
