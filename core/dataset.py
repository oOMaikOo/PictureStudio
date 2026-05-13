"""
Enhanced dataset management: analysis, stratified splits, COCO/YOLO/CSV export,
duplicate detection, class balance warnings.
"""
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from utils.config import IMAGE_FORMATS, MIN_IMAGES_PER_CLASS
from utils.logging_utils import get_logger

log = get_logger()


# ================================================================== Dataset


class ImageROIDataset(Dataset if HAS_TORCH else object):
    """PyTorch Dataset – yields (tensor, class_idx) from images or ROI crops."""

    def __init__(
        self,
        samples: List[Tuple],  # (img_path, roi_or_None, class_idx)
        image_size: int = 224,
        augment: bool = False,
        augmentation_cfg: Dict = None,
    ):
        self.samples = samples
        aug_cfg = augmentation_cfg or {}
        aug_transforms = []
        if augment:
            if aug_cfg.get("flip", True):
                aug_transforms.append(transforms.RandomHorizontalFlip())
                aug_transforms.append(transforms.RandomVerticalFlip(p=0.1))
            if aug_cfg.get("rotation", True):
                deg = float(aug_cfg.get("rotation_degrees", 15))
                aug_transforms.append(transforms.RandomRotation(deg))
            if aug_cfg.get("brightness", True) or aug_cfg.get("contrast", True):
                bri = float(aug_cfg.get("brightness_strength", 0.3))
                aug_transforms.append(transforms.ColorJitter(
                    brightness=bri if aug_cfg.get("brightness") else 0,
                    contrast=bri if aug_cfg.get("contrast") else 0,
                    saturation=0.1,
                ))
            if aug_cfg.get("blur", False):
                radius = int(aug_cfg.get("blur_radius", 3))
                radius = radius if radius % 2 == 1 else radius + 1
                aug_transforms.append(transforms.GaussianBlur(radius, sigma=(0.1, 2.0)))
            if aug_cfg.get("scale", False):
                scale_lo = float(aug_cfg.get("scale_min", 0.8))
                aug_transforms.append(
                    transforms.RandomResizedCrop(image_size, scale=(scale_lo, 1.0))
                )
            else:
                aug_transforms.append(transforms.Resize((image_size, image_size)))
        else:
            aug_transforms.append(transforms.Resize((image_size, image_size)))

        aug_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(aug_transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, roi, class_idx = self.samples[idx]
        image = PILImage.open(img_path).convert("RGB")
        if roi is not None:
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            iw, ih = image.size
            x, y = max(0, x), max(0, y)
            w, h = max(1, min(w, iw - x)), max(1, min(h, ih - y))
            image = image.crop((x, y, x + w, y + h))
        return self.transform(image), class_idx


class MultiLabelImageDataset(Dataset if HAS_TORCH else object):
    """Dataset for multi-label classification — returns (tensor, float_tensor[num_classes])."""

    def __init__(
        self,
        samples: List[Tuple],          # (img_path, roi_or_None, List[int])
        num_classes: int,
        image_size: int = 224,
        augment: bool = False,
        augmentation_cfg: Dict = None,
    ):
        self.samples = samples
        self.num_classes = num_classes
        aug_cfg = augmentation_cfg or {}
        aug_transforms = []
        if augment:
            if aug_cfg.get("flip", True):
                aug_transforms.append(transforms.RandomHorizontalFlip())
                aug_transforms.append(transforms.RandomVerticalFlip(p=0.1))
            if aug_cfg.get("rotation", True):
                deg = float(aug_cfg.get("rotation_degrees", 15))
                aug_transforms.append(transforms.RandomRotation(deg))
            if aug_cfg.get("brightness", True) or aug_cfg.get("contrast", True):
                bri = float(aug_cfg.get("brightness_strength", 0.3))
                aug_transforms.append(transforms.ColorJitter(
                    brightness=bri if aug_cfg.get("brightness") else 0,
                    contrast=bri if aug_cfg.get("contrast") else 0,
                    saturation=0.1,
                ))
            if aug_cfg.get("blur", False):
                radius = int(aug_cfg.get("blur_radius", 3))
                radius = radius if radius % 2 == 1 else radius + 1
                aug_transforms.append(transforms.GaussianBlur(radius, sigma=(0.1, 2.0)))
            if aug_cfg.get("scale", False):
                scale_lo = float(aug_cfg.get("scale_min", 0.8))
                aug_transforms.append(
                    transforms.RandomResizedCrop(image_size, scale=(scale_lo, 1.0))
                )
            else:
                aug_transforms.append(transforms.Resize((image_size, image_size)))
        else:
            aug_transforms.append(transforms.Resize((image_size, image_size)))

        aug_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(aug_transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, roi, class_indices = self.samples[idx]
        image = PILImage.open(img_path).convert("RGB")
        if roi is not None:
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            iw, ih = image.size
            x, y = max(0, x), max(0, y)
            w, h = max(1, min(w, iw - x)), max(1, min(h, ih - y))
            image = image.crop((x, y, x + w, y + h))
        tensor = self.transform(image)
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for ci in class_indices:
            if 0 <= ci < self.num_classes:
                label_vec[ci] = 1.0
        return tensor, label_vec


# ================================================================== Helpers


def build_samples(project, use_rois: bool = True) -> Tuple[List[Tuple], List[str]]:
    label_names = sorted(project.labels.keys())
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    samples = []
    for img_path in project.images:
        rois = project.get_rois(img_path)
        added = False
        if use_rois and rois:
            for roi in rois:
                lbl = roi.get("label", "")
                if lbl and lbl in label_to_idx:
                    samples.append((img_path, roi, label_to_idx[lbl]))
                    added = True
        if not added:
            # Fallback: use image-level label (covers unlabeled ROIs and no-ROI cases)
            lbl = project.get_image_label(img_path)
            if lbl and lbl in label_to_idx:
                samples.append((img_path, None, label_to_idx[lbl]))
    return samples, label_names


def split_samples(
    samples: List,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Stratified split by class index, reproducible via seed."""
    random.seed(seed)
    by_class: Dict[int, List] = defaultdict(list)
    for s in samples:
        by_class[s[2]].append(s)
    train, val, test = [], [], []
    for cls_samples in by_class.values():
        random.shuffle(cls_samples)
        n = len(cls_samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        train.extend(cls_samples[:n_train])
        val.extend(cls_samples[n_train:n_train + n_val])
        test.extend(cls_samples[n_train + n_val:])
    for lst in [train, val, test]:
        random.shuffle(lst)
    return train, val, test


def build_multi_label_samples(project, use_rois: bool = True) -> Tuple[List[Tuple], List[str]]:
    """Build samples for multi-label classification: (img_path, roi_or_None, List[int])."""
    label_names = sorted(project.labels.keys())
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    samples = []
    for img_path in project.images:
        multi_lbls = project.get_image_multi_labels(img_path)
        if not multi_lbls:
            continue
        indices = [label_to_idx[l] for l in multi_lbls if l in label_to_idx]
        if not indices:
            continue
        rois = project.get_rois(img_path)
        if use_rois and rois:
            for roi in rois:
                roi_lbl = roi.get("label", "")
                roi_indices = ([label_to_idx[roi_lbl]] if roi_lbl and roi_lbl in label_to_idx
                               else indices)
                samples.append((img_path, roi, roi_indices))
        else:
            samples.append((img_path, None, indices))
    return samples, label_names


def create_multi_label_datasets(project, training_config: Dict, use_rois: bool = True):
    """Build train/val/test multi-label datasets. Returns (train_ds, val_ds, test_ds, class_names)."""
    samples, class_names = build_multi_label_samples(project, use_rois=use_rois)
    if not samples:
        raise ValueError("Keine Multi-Label-Bilder gefunden.")
    if len(class_names) < 2:
        raise ValueError("Mindestens 2 Klassen erforderlich.")

    # Stratify at image level by first label
    seed = training_config.get("seed", 42)
    train_ratio = training_config.get("train_split", 0.7)
    val_ratio = training_config.get("val_split", 0.2)
    all_paths = list({s[0] for s in samples})
    random.seed(seed)
    random.shuffle(all_paths)
    n = len(all_paths)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    train_paths = set(all_paths[:n_train])
    val_paths = set(all_paths[n_train:n_train + n_val])
    test_paths = set(all_paths[n_train + n_val:])

    train_s = [s for s in samples if s[0] in train_paths]
    val_s = [s for s in samples if s[0] in val_paths]
    test_s = [s for s in samples if s[0] in test_paths]

    num_classes = len(class_names)
    aug_cfg = training_config.get("augmentation", {})
    img_size = training_config.get("image_size", 224)
    return (
        MultiLabelImageDataset(train_s, num_classes, img_size, augment=True, augmentation_cfg=aug_cfg),
        MultiLabelImageDataset(val_s, num_classes, img_size, augment=False),
        MultiLabelImageDataset(test_s, num_classes, img_size, augment=False),
        class_names,
    )


def create_datasets(project, training_config: Dict, use_rois: bool = True):
    """Build train/val/test datasets from project. Returns (train_ds, val_ds, test_ds, class_names)."""
    if getattr(project.config, "multi_label", False) or training_config.get("multi_label", False):
        return create_multi_label_datasets(project, training_config, use_rois)

    samples, class_names = build_samples(project, use_rois=use_rois)
    if not samples:
        raise ValueError("Keine gelabelten Bilder/ROIs gefunden.")
    if len(class_names) < 2:
        raise ValueError("Mindestens 2 Klassen erforderlich.")

    counts = Counter(s[2] for s in samples)
    for idx, name in enumerate(class_names):
        if counts[idx] < MIN_IMAGES_PER_CLASS:
            log.warning("Klasse '%s' hat nur %d Samples", name, counts[idx])

    train_s, val_s, test_s = split_samples(
        samples,
        train_ratio=training_config.get("train_split", 0.7),
        val_ratio=training_config.get("val_split", 0.2),
        seed=training_config.get("seed", 42),
    )
    aug_cfg = training_config.get("augmentation", {})
    img_size = training_config.get("image_size", 224)
    return (
        ImageROIDataset(train_s, img_size, augment=True, augmentation_cfg=aug_cfg),
        ImageROIDataset(val_s, img_size, augment=False),
        ImageROIDataset(test_s, img_size, augment=False),
        class_names,
    )


# ================================================================== Analysis


def analyze_dataset(project) -> Dict:
    """Return a comprehensive analysis of the project dataset."""
    result = {
        "total": len(project.images),
        "labeled": project.get_labeled_image_count(),
        "unlabeled": len(project.get_unlabeled_images()),
        "formats": Counter(),
        "sizes": [],
        "missing_files": [],
        "corrupt_files": [],
        "duplicates": [],
        "label_counts": project.get_label_counts(),
        "roi_counts": project.get_label_counts(use_rois=True),
        "warnings": [],
    }

    seen_hashes = {}
    for img_path in project.images:
        ext = os.path.splitext(img_path)[1].lower()
        result["formats"][ext] += 1

        if not os.path.exists(img_path):
            result["missing_files"].append(img_path)
            continue

        # Check hash for duplicates (fast: use first 64kB)
        try:
            h = hashlib.md5()
            with open(img_path, "rb") as fh:
                h.update(fh.read(65536))
            digest = h.hexdigest()
            if digest in seen_hashes:
                result["duplicates"].append((img_path, seen_hashes[digest]))
            else:
                seen_hashes[digest] = img_path
        except OSError:
            result["corrupt_files"].append(img_path)
            continue

        # Image size
        if HAS_PIL:
            try:
                with PILImage.open(img_path) as im:
                    result["sizes"].append(im.size)
            except Exception:
                result["corrupt_files"].append(img_path)

    # Class balance warning
    counts = [v for v in result["label_counts"].values() if v > 0]
    if len(counts) >= 2:
        ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
        if ratio > 5:
            result["warnings"].append(
                f"Starke Klassenungleichheit (Ratio {ratio:.1f}:1). "
                "Erwäge Oversampling oder Gewichtung."
            )
    for lbl, cnt in result["label_counts"].items():
        if 0 < cnt < MIN_IMAGES_PER_CLASS:
            result["warnings"].append(
                f"Klasse '{lbl}' hat nur {cnt} Samples (< {MIN_IMAGES_PER_CLASS})."
            )

    if result["missing_files"]:
        result["warnings"].append(f"{len(result['missing_files'])} Bild(er) nicht gefunden.")
    if result["duplicates"]:
        result["warnings"].append(f"{len(result['duplicates'])} Duplikat(e) gefunden.")
    if result["corrupt_files"]:
        result["warnings"].append(f"{len(result['corrupt_files'])} fehlerhafte Bild(er).")

    # Size stats
    if result["sizes"]:
        ws = [s[0] for s in result["sizes"]]
        hs = [s[1] for s in result["sizes"]]
        result["size_stats"] = {
            "min_w": min(ws), "max_w": max(ws),
            "min_h": min(hs), "max_h": max(hs),
            "unique_sizes": len(set(result["sizes"])),
        }
    return result


# ================================================================== Export


def create_stratified_split(
    project,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Convenience wrapper: returns (train_paths, val_paths, test_paths)."""
    label_names = sorted(project.labels.keys())
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    samples = [
        (p, None, label_to_idx[lbl])
        for p, lbl in project.image_labels.items()
        if lbl and lbl in label_to_idx
    ]
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
    train_s, val_s, test_s = split_samples(
        samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    return [s[0] for s in train_s], [s[0] for s in val_s], [s[0] for s in test_s]


def export_coco(project, output_path: str) -> None:
    """Export annotations in COCO JSON format (bounding boxes)."""
    label_names = sorted(project.labels.keys())
    cat_id = {n: i + 1 for i, n in enumerate(label_names)}

    images, annotations = [], []
    ann_id = 1
    for img_id, img_path in enumerate(project.images, 1):
        if not os.path.exists(img_path):
            continue
        fname = os.path.basename(img_path)
        w, h = 0, 0
        if HAS_PIL:
            try:
                with PILImage.open(img_path) as im:
                    w, h = im.size
            except Exception:
                pass
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for roi in project.get_rois(img_path):
            lbl = roi.get("label", "")
            if lbl not in cat_id:
                continue
            x, y, rw, rh = roi.get("x", 0), roi.get("y", 0), roi.get("w", 0), roi.get("h", 0)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id[lbl],
                "bbox": [x, y, rw, rh],
                "area": rw * rh,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": project.config.name, "version": "1.0"},
        "categories": [{"id": v, "name": k} for k, v in cat_id.items()],
        "images": images,
        "annotations": annotations,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    log.info("COCO exportiert: %s", output_path)


def export_yolo(project, output_dir: str) -> None:
    """Export annotations in YOLO txt format (one .txt per image)."""
    label_names = sorted(project.labels.keys())
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    os.makedirs(output_dir, exist_ok=True)

    # Write classes.txt
    with open(os.path.join(output_dir, "classes.txt"), "w") as fh:
        fh.write("\n".join(label_names))

    for img_path in project.images:
        if not os.path.exists(img_path):
            continue
        rois = project.get_rois(img_path)
        if not rois:
            continue
        w, h = 1, 1
        if HAS_PIL:
            try:
                with PILImage.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue

        lines = []
        for roi in rois:
            lbl = roi.get("label", "")
            if lbl not in label_to_idx:
                continue
            rx, ry, rw, rh = roi.get("x", 0), roi.get("y", 0), roi.get("w", 0), roi.get("h", 0)
            cx = (rx + rw / 2) / w
            cy = (ry + rh / 2) / h
            nw, nh = rw / w, rh / h
            lines.append(f"{label_to_idx[lbl]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            with open(os.path.join(output_dir, stem + ".txt"), "w") as fh:
                fh.write("\n".join(lines))
    log.info("YOLO exportiert: %s", output_dir)


def export_csv(project, output_path: str) -> None:
    """Export image labels and ROIs as CSV."""
    try:
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image_path", "filename", "image_label",
                            "roi_id", "roi_type", "roi_x", "roi_y",
                            "roi_w", "roi_h", "roi_label"],
            )
            writer.writeheader()
            for img_path in project.images:
                img_lbl = project.get_image_label(img_path)
                rois = project.get_rois(img_path)
                if rois:
                    for roi in rois:
                        writer.writerow({
                            "image_path": img_path,
                            "filename": os.path.basename(img_path),
                            "image_label": img_lbl,
                            "roi_id": roi.get("id", ""),
                            "roi_type": roi.get("type", "rect"),
                            "roi_x": roi.get("x", ""),
                            "roi_y": roi.get("y", ""),
                            "roi_w": roi.get("w", ""),
                            "roi_h": roi.get("h", ""),
                            "roi_label": roi.get("label", ""),
                        })
                else:
                    writer.writerow({
                        "image_path": img_path,
                        "filename": os.path.basename(img_path),
                        "image_label": img_lbl,
                        "roi_id": "", "roi_type": "",
                        "roi_x": "", "roi_y": "",
                        "roi_w": "", "roi_h": "",
                        "roi_label": "",
                    })
        log.info("CSV exportiert: %s", output_path)
    except Exception as exc:
        raise RuntimeError(f"CSV-Export fehlgeschlagen: {exc}") from exc
