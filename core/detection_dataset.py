"""
Convert PictureStudio project ROIs to YOLO dataset format.

YOLO label format per line: class_idx x_center y_center width height  (all normalized 0–1)
Directory structure:
  <output_dir>/
    images/train/   images/val/
    labels/train/   labels/val/
    data.yaml
"""
import os
import random
import shutil
from typing import Dict, List, Tuple


def prepare_yolo_dataset(
    project,
    output_dir: str,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[str, Dict]:
    """
    Convert project ROI annotations to a YOLO-compatible dataset.

    Only images that have at least one labeled ROI are included.
    Returns (data_yaml_path, stats_dict).
    """
    from PIL import Image as _Image

    random.seed(seed)

    # Collect annotated images
    class_names = list(project.labels.keys())
    if not class_names:
        raise ValueError("Das Projekt hat keine Labels definiert.")

    annotated: List[Tuple[str, List[Dict]]] = []
    for img_path in project.images:
        rois = project.get_rois(img_path)
        labeled = [
            r for r in rois
            if r.get("label") and r["label"] in class_names
        ]
        if labeled:
            annotated.append((img_path, labeled))

    if not annotated:
        raise ValueError(
            "Keine Bilder mit beschrifteten ROIs gefunden.\n"
            "Bitte zuerst im Labeling-Editor ROIs mit Labels versehen."
        )

    # Shuffle and split
    random.shuffle(annotated)
    n_train = max(1, int(len(annotated) * train_split))
    splits = {
        "train": annotated[:n_train],
        "val":   annotated[n_train:] or annotated[:1],  # at least 1 val image
    }

    # Create directory structure
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    n_annotations = 0
    for split, items in splits.items():
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        for img_path, rois in items:
            try:
                with _Image.open(img_path) as im:
                    iw, ih = im.size
            except Exception:
                continue

            img_name = os.path.basename(img_path)
            # Avoid filename collisions by keeping subfolder prefix
            dest_name = img_name
            dest_img = os.path.join(img_dir, dest_name)
            if os.path.exists(dest_img):
                base, ext = os.path.splitext(img_name)
                dest_name = f"{base}_{abs(hash(img_path)) % 100000}{ext}"
                dest_img  = os.path.join(img_dir, dest_name)
            shutil.copy(img_path, dest_img)

            lbl_name = os.path.splitext(dest_name)[0] + ".txt"
            with open(os.path.join(lbl_dir, lbl_name), "w") as f:
                for roi in rois:
                    cls_idx = class_names.index(roi["label"])
                    x, y = float(roi.get("x", 0)), float(roi.get("y", 0))
                    w, h = float(roi.get("w", 0)), float(roi.get("h", 0))
                    if w <= 0 or h <= 0:
                        continue
                    xc = (x + w / 2) / iw
                    yc = (y + h / 2) / ih
                    wn = w / iw
                    hn = h / ih
                    # Clamp to [0, 1]
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    wn = max(0.0, min(1.0, wn))
                    hn = max(0.0, min(1.0, hn))
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                    n_annotations += 1

    # Write data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    stats = {
        "n_images":      len(annotated),
        "n_train":       len(splits["train"]),
        "n_val":         len(splits["val"]),
        "n_classes":     len(class_names),
        "class_names":   class_names,
        "n_annotations": n_annotations,
    }
    return yaml_path, stats
