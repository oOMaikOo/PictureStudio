"""
Enhanced project: versioning, autosave, backup, file validation, dashboard data.
Backward-compatible with MVP JSON format.
"""
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from utils.config import DEFAULT_TRAIN_CONFIG, DEFAULT_SSH_CONFIG
from utils.logging_utils import get_logger

log = get_logger()

PROJECT_FORMAT_VERSION = "2.0"


@dataclass
class ProjectConfig:
    """
    Lightweight metadata block stored in the project JSON under the "config" key.

    Serialised via dataclasses.asdict() on save and reconstructed field-by-field
    on load for backward compatibility with older project files.
    """
    name: str = ""
    description: str = ""
    created_at: str = ""
    modified_at: str = ""
    image_dir: str = ""
    version: str = PROJECT_FORMAT_VERSION
    author: str = ""
    tags: List[str] = field(default_factory=list)
    multi_label: bool = False
    project_type: str = "image"   # "image" | "video"


class Project:
    """Central data store for a labeling project – fully serialisable to JSON."""

    def __init__(self):
        self.config = ProjectConfig()
        self.project_path: str = ""

        # Data
        self.images: List[str] = []
        self.labels: Dict[str, Dict] = {}        # name -> {color, description, parent}
        self.image_labels: Dict[str, str] = {}   # img_path -> primary label
        self.image_multi_labels: Dict[str, List[str]] = {}  # img_path -> [label, ...]
        self.rois: Dict[str, List[Dict]] = {}    # img_path -> [roi_dict, ...]

        # Training
        self.training_runs: List[Dict] = []
        self.current_model_path: str = ""
        self.training_config: Dict = dict(DEFAULT_TRAIN_CONFIG)
        self.ssh_config: Dict = dict(DEFAULT_SSH_CONFIG)

        # Inference
        self.inference_results: List[Dict] = []

        # Dataset splits (persistent)
        self.dataset_splits: Dict[str, List] = {}  # run_id -> {train, val, test}

        # ROI templates
        self.roi_templates: List[Dict] = []

        # Active Learning queue: images with uncertain predictions pending review
        # Each entry: {path, predicted_label, confidence, added_at}
        self.active_learning_queue: List[Dict] = []

        # Quality-assurance flags per image: {img_path: {uncertain, comment}}
        self.image_label_flags: Dict[str, Dict] = {}

    @property
    def is_multi_label(self) -> bool:
        """True when the project is in multi-label classification mode."""
        return self.config.multi_label

    # ------------------------------------------------------------------ save / load

    def save(self, path: str = None) -> None:
        """
        Serialise the project to JSON at *path* (or the existing project_path).

        Uses an atomic write (temp file → os.replace) to avoid corrupting an
        existing file on crash or power loss.
        """
        if path:
            self.project_path = path
        if not self.project_path:
            raise ValueError("Kein Projektpfad gesetzt.")

        self.config.modified_at = datetime.now().isoformat()
        self.config.version = PROJECT_FORMAT_VERSION

        data = {
            "format_version": PROJECT_FORMAT_VERSION,
            "config": asdict(self.config),
            "images": self.images,
            "labels": self.labels,
            "image_labels": self.image_labels,
            "image_multi_labels": self.image_multi_labels,
            "rois": self.rois,
            "training_runs": self.training_runs,
            "current_model_path": self.current_model_path,
            "training_config": self.training_config,
            "ssh_config": self.ssh_config,
            "inference_results": self.inference_results,
            "dataset_splits": self.dataset_splits,
            "roi_templates": self.roi_templates,
            "active_learning_queue": self.active_learning_queue,
            "image_label_flags": self.image_label_flags,
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.project_path)), exist_ok=True)
        # Write atomically via temp file
        tmp = self.project_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.project_path)
        log.debug("Projekt gespeichert: %s", self.project_path)

    @classmethod
    def load(cls, path: str) -> "Project":
        """
        Load a project from the JSON file at *path*.

        Missing keys (from older format versions) are silently ignored;
        new fields are initialised to their defaults. Merges training_config
        and ssh_config with current defaults so new keys are always present.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise FileNotFoundError(f"Projektdatei nicht gefunden: {path}")
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Projektdatei ist beschädigt und kann nicht gelesen werden.\n"
                f"Datei: {path}\nDetail: {exc}"
            )
        except OSError as exc:
            raise OSError(f"Projektdatei konnte nicht geöffnet werden: {exc}") from exc

        project = cls()
        project.project_path = path

        cfg = data.get("config", {})
        for key in asdict(project.config):
            if key in cfg:
                setattr(project.config, key, cfg[key])

        project.images = data.get("images", [])
        project.labels = data.get("labels", {})
        project.image_labels = data.get("image_labels", {})
        project.image_multi_labels = data.get("image_multi_labels", {})
        project.rois = data.get("rois", {})
        project.training_runs = data.get("training_runs", [])
        project.current_model_path = data.get("current_model_path", "")
        project.training_config = {**DEFAULT_TRAIN_CONFIG, **data.get("training_config", {})}
        project.ssh_config = {**DEFAULT_SSH_CONFIG, **data.get("ssh_config", {})}
        project.inference_results = data.get("inference_results", [])
        project.dataset_splits = data.get("dataset_splits", {})
        project.roi_templates = data.get("roi_templates", [])
        project.active_learning_queue = data.get("active_learning_queue", [])
        project.image_label_flags = data.get("image_label_flags", {})
        return project

    # ------------------------------------------------------------------ backup

    def create_backup(self, backup_dir: str = None) -> str:
        """Copy the project JSON to a dated backup file. Returns backup path."""
        if not self.project_path or not os.path.exists(self.project_path):
            return ""
        base = backup_dir or os.path.join(self.get_project_dir(), "backups")
        os.makedirs(base, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = os.path.splitext(os.path.basename(self.project_path))[0]
        dest = os.path.join(base, f"{stem}_{ts}.json")
        shutil.copy2(self.project_path, dest)
        log.info("Backup erstellt: %s", dest)
        return dest

    # ------------------------------------------------------------------ file validation

    def validate_image_files(self) -> Dict[str, List[str]]:
        """Check which images exist, are missing, or are inaccessible."""
        ok, missing, unreadable = [], [], []
        for p in self.images:
            if not os.path.exists(p):
                missing.append(p)
            elif not os.path.isfile(p):
                unreadable.append(p)
            else:
                ok.append(p)
        return {"ok": ok, "missing": missing, "unreadable": unreadable}

    def relocate_images(self, old_prefix: str, new_prefix: str) -> int:
        """Replace path prefix for batch image relocation. Returns updated count."""
        count = 0
        new_images = []
        for p in self.images:
            if p.startswith(old_prefix):
                new_p = new_prefix + p[len(old_prefix):]
                new_images.append(new_p)
                if p in self.image_labels:
                    self.image_labels[new_p] = self.image_labels.pop(p)
                if p in self.rois:
                    self.rois[new_p] = self.rois.pop(p)
                count += 1
            else:
                new_images.append(p)
        self.images = new_images
        return count

    # ------------------------------------------------------------------ helpers

    def get_project_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.project_path)) if self.project_path else ""

    def get_models_dir(self) -> str:
        d = os.path.join(self.get_project_dir(), "models")
        os.makedirs(d, exist_ok=True)
        return d

    # Images
    def add_image(self, image_path: str) -> bool:
        """Add *image_path* to the project. Returns False if already present."""
        if image_path not in self.images:
            self.images.append(image_path)
            return True
        return False

    def remove_image(self, image_path: str) -> None:
        """Remove an image and all associated labels, ROIs, and flags."""
        self.images = [p for p in self.images if p != image_path]
        self.image_labels.pop(image_path, None)
        self.image_multi_labels.pop(image_path, None)
        self.rois.pop(image_path, None)
        self.image_label_flags.pop(image_path, None)

    # Labels
    def add_label(self, name: str, color: str = "#E74C3C", description: str = "",
                  parent: str = "") -> None:
        """Add or overwrite a label definition in the project."""
        self.labels[name] = {"color": color, "description": description, "parent": parent}

    def remove_label(self, name: str) -> None:
        """
        Delete a label and clean up all references.

        Removes the label from image_labels, image_multi_labels, and ROI label fields.
        """
        self.labels.pop(name, None)
        for img in list(self.image_labels):
            if self.image_labels[img] == name:
                del self.image_labels[img]
        for img in list(self.image_multi_labels):
            self.image_multi_labels[img] = [l for l in self.image_multi_labels[img] if l != name]
        for roi_list in self.rois.values():
            for roi in roi_list:
                if roi.get("label") == name:
                    roi["label"] = ""

    def rename_label(self, old: str, new: str) -> None:
        """
        Rename a label in place, updating all references.

        Updates labels dict, image_labels, image_multi_labels, and ROI label fields.
        No-op if *old* does not exist or *old == new*.
        """
        if old not in self.labels or old == new:
            return
        self.labels[new] = self.labels.pop(old)
        for img in list(self.image_labels):
            if self.image_labels[img] == old:
                self.image_labels[img] = new
        for img in list(self.image_multi_labels):
            self.image_multi_labels[img] = [new if l == old else l for l in self.image_multi_labels[img]]
        for roi_list in self.rois.values():
            for roi in roi_list:
                if roi.get("label") == old:
                    roi["label"] = new

    def get_label_color(self, name: str) -> str:
        """Return the hex colour for *name*, or '#888888' if the label does not exist."""
        return self.labels.get(name, {}).get("color", "#888888")

    def get_root_labels(self) -> List[str]:
        """Return labels that have no parent (top-level nodes in a label hierarchy)."""
        return [n for n, d in self.labels.items() if not d.get("parent")]

    def get_child_labels(self, parent: str) -> List[str]:
        """Return all labels whose parent field equals *parent*."""
        return [n for n, d in self.labels.items() if d.get("parent") == parent]

    # Image labels
    def set_image_label(self, image_path: str, label: str) -> None:
        """Assign the primary label for *image_path*. Passing an empty string removes it."""
        if label:
            self.image_labels[image_path] = label
        else:
            self.image_labels.pop(image_path, None)

    def get_image_label(self, image_path: str) -> str:
        """Return the primary label for *image_path*, or an empty string."""
        return self.image_labels.get(image_path, "")

    def set_image_multi_labels(self, image_path: str, labels: List[str]) -> None:
        """Store the full multi-label list for *image_path*, filtering out empty strings."""
        self.image_multi_labels[image_path] = [l for l in labels if l]

    def get_image_multi_labels(self, image_path: str) -> List[str]:
        """
        Return the effective multi-label list for *image_path*.

        Merges the primary label (from image_labels) into the multi-label list
        so that images labelled in single-label mode are included transparently.
        """
        primary = self.get_image_label(image_path)
        multi = self.image_multi_labels.get(image_path, [])
        # Merge primary into multi if not present
        if primary and primary not in multi:
            return [primary] + multi
        return multi or ([primary] if primary else [])

    # ROIs
    def add_roi(self, image_path: str, roi_data: Dict) -> None:
        """Append a ROI dict to the list for *image_path*, creating the list if needed."""
        self.rois.setdefault(image_path, []).append(roi_data)

    def remove_roi(self, image_path: str, roi_id: str) -> None:
        """Remove the ROI with the given *roi_id* from *image_path*'s list."""
        if image_path in self.rois:
            self.rois[image_path] = [r for r in self.rois[image_path] if r.get("id") != roi_id]

    def update_roi(self, image_path: str, roi_id: str, roi_data: Dict) -> None:
        """Replace the ROI dict matching *roi_id* with *roi_data* in-place."""
        for i, roi in enumerate(self.rois.get(image_path, [])):
            if roi.get("id") == roi_id:
                self.rois[image_path][i] = roi_data
                return

    def get_rois(self, image_path: str) -> List[Dict]:
        """Return the list of ROI dicts for *image_path* (empty list if none)."""
        return self.rois.get(image_path, [])

    # ROI templates
    def add_roi_template(self, name: str, roi_data: Dict) -> None:
        self.roi_templates.append({"name": name, "roi": roi_data})

    def get_roi_templates(self) -> List[Dict]:
        return self.roi_templates

    def apply_roi_template(self, template_name: str, image_paths: List[str]) -> int:
        """Apply a named ROI template to multiple images. Returns count."""
        import uuid
        tmpl = next((t for t in self.roi_templates if t["name"] == template_name), None)
        if not tmpl:
            return 0
        count = 0
        for img in image_paths:
            roi = dict(tmpl["roi"])
            roi["id"] = str(uuid.uuid4())[:8]
            self.add_roi(img, roi)
            count += 1
        return count

    # Training runs
    def add_training_run(self, run_data: Dict) -> None:
        self.training_runs.append(run_data)

    def get_last_training_run(self) -> Optional[Dict]:
        return self.training_runs[-1] if self.training_runs else None

    # Dataset splits
    def save_split(self, run_id: str, train: List, val: List, test: List) -> None:
        self.dataset_splits[run_id] = {"train": train, "val": val, "test": test}

    # Statistics
    def get_label_counts(self, use_rois: bool = False) -> Dict[str, int]:
        """
        Return a dict mapping each label name to the number of images (or ROIs) assigned.

        When *use_rois* is True, counts ROI-level labels instead of image-level labels.
        In multi-label mode, an image contributes one count per active label.
        """
        counts: Dict[str, int] = {lbl: 0 for lbl in self.labels}
        if use_rois:
            for roi_list in self.rois.values():
                for roi in roi_list:
                    lbl = roi.get("label", "")
                    if lbl:
                        counts[lbl] = counts.get(lbl, 0) + 1
        elif self.config.multi_label:
            for lbls in self.image_multi_labels.values():
                for lbl in lbls:
                    if lbl:
                        counts[lbl] = counts.get(lbl, 0) + 1
        else:
            for lbl in self.image_labels.values():
                counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def get_labeled_image_count(self) -> int:
        """Return the count of images that have at least one label assigned."""
        if self.config.multi_label:
            return sum(1 for p in self.images if self.image_multi_labels.get(p))
        return len(self.image_labels)

    def get_roi_count(self) -> int:
        """Return the total number of ROIs across all images in the project."""
        return sum(len(v) for v in self.rois.values())

    def get_unlabeled_images(self) -> List[str]:
        """Return a list of image paths that have no label assigned."""
        if self.config.multi_label:
            return [p for p in self.images if not self.image_multi_labels.get(p)]
        return [p for p in self.images if not self.get_image_label(p)]

    def migrate_to_multi_label(self) -> int:
        """Copy image_labels into image_multi_labels. Returns count of migrated images."""
        count = 0
        for path, lbl in self.image_labels.items():
            if not lbl:
                continue
            existing = self.image_multi_labels.get(path, [])
            if lbl not in existing:
                self.image_multi_labels[path] = [lbl] + [x for x in existing if x != lbl]
                count += 1
        self.config.multi_label = True
        return count

    def migrate_to_single_label(self) -> int:
        """Set image_labels from the first multi-label. Returns count of migrated images."""
        count = 0
        for path, lbls in self.image_multi_labels.items():
            if lbls:
                self.image_labels[path] = lbls[0]
                count += 1
        self.config.multi_label = False
        return count

    def get_images_by_label(self, label: str) -> List[str]:
        """Return all image paths whose primary label equals *label*."""
        return [p for p, l in self.image_labels.items() if l == label]

    # Label Quality Flags
    def set_label_flag(self, image_path: str, uncertain: bool, comment: str = "") -> None:
        """
        Set or clear the QA uncertain flag for *image_path*.

        When both *uncertain* is False and *comment* is empty, the entry is
        removed from image_label_flags to keep the dict compact.
        """
        if uncertain or comment:
            self.image_label_flags[image_path] = {"uncertain": uncertain, "comment": comment}
        else:
            self.image_label_flags.pop(image_path, None)

    def get_label_flag(self, image_path: str) -> Dict:
        """Return the QA flag dict {uncertain, comment} for *image_path*, or {}."""
        return self.image_label_flags.get(image_path, {})

    def is_label_uncertain(self, image_path: str) -> bool:
        """Return True when *image_path* is flagged as uncertain."""
        return bool(self.image_label_flags.get(image_path, {}).get("uncertain"))

    def get_uncertain_images(self) -> List[str]:
        """Return all image paths currently flagged as uncertain."""
        return [p for p in self.images if self.is_label_uncertain(p)]

    def clear_label_flag(self, image_path: str) -> None:
        """Remove the QA flag for *image_path* if present."""
        self.image_label_flags.pop(image_path, None)

    # Active Learning Queue
    def add_to_al_queue(self, path: str, predicted_label: str, confidence: float) -> bool:
        """Add an image to the AL queue. Returns False if already queued."""
        if any(e["path"] == path for e in self.active_learning_queue):
            return False
        self.active_learning_queue.append({
            "path": path,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "added_at": datetime.now().isoformat(),
        })
        return True

    def remove_from_al_queue(self, path: str) -> None:
        self.active_learning_queue = [e for e in self.active_learning_queue if e["path"] != path]

    def get_al_queue(self) -> List[Dict]:
        return list(self.active_learning_queue)

    def clear_al_queue(self) -> None:
        self.active_learning_queue.clear()

    def get_unlabeled_al_queue(self) -> List[Dict]:
        """Queue entries that still have no image-level label assigned."""
        return [e for e in self.active_learning_queue
                if not self.get_image_label(e["path"])]

    def get_dashboard_data(self) -> Dict:
        """Compile a summary dict used by DashboardPage to populate its statistics widgets."""
        last_run = self.get_last_training_run()
        metrics = last_run.get("metrics", {}) if last_run else {}
        return {
            "total_images": len(self.images),
            "labeled_images": self.get_labeled_image_count(),
            "unlabeled_images": len(self.get_unlabeled_images()),
            "total_rois": self.get_roi_count(),
            "label_counts": self.get_label_counts(),
            "roi_counts": self.get_label_counts(use_rois=True),
            "total_labels": len(self.labels),
            "training_runs": len(self.training_runs),
            "last_run_ts": last_run.get("timestamp", "") if last_run else "",
            "last_run_accuracy": metrics.get("accuracy", 0),
            "last_run_f1": metrics.get("macro_f1", 0),
            "current_model": os.path.basename(self.current_model_path),
            "project_name": self.config.name,
        }
