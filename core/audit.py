"""
Audit trail: log every significant change to labels, ROIs, training config.
Stored as a JSONL file alongside the project.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

from utils.logging_utils import get_logger

log = get_logger()


class AuditTrail:
    """
    Append-only JSONL audit log for a project.

    Each entry is a JSON object (one per line) with keys:
      ts, action, entity, details, project.
    Convenience wrappers cover all common actions; the generic ``log()``
    method can be used for custom events.
    Instantiated by MainWindow and passed to every page via set_project().
    """

    def __init__(self, project_dir: str, project_name: str = ""):
        """
        Initialise the audit trail in *project_dir*/audit.jsonl.

        Creates *project_dir* if it does not exist.
        """
        os.makedirs(project_dir, exist_ok=True)
        self._file = os.path.join(project_dir, "audit.jsonl")
        self._project_name = project_name

    # ------------------------------------------------------------------ public

    def log(self, action: str, entity: str = "", details: Optional[Dict] = None) -> None:
        """
        Append one event to the JSONL log.

        Parameters
        ----------
        action  : Short identifier (e.g. "label_added", "training_started").
        entity  : Affected object name (label name, image filename, run ID …).
        details : Optional dict with additional context.
        """
        entry = {
            "ts": datetime.now().isoformat(),
            "action": action,
            "entity": entity,
            "details": details or {},
            "project": self._project_name,
        }
        try:
            with open(self._file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Audit-Log konnte nicht geschrieben werden: %s", exc)

    def log_label_added(self, name: str, color: str) -> None:
        self.log("label_added", name, {"color": color})

    def log_label_removed(self, name: str) -> None:
        self.log("label_removed", name)

    def log_label_renamed(self, old: str, new: str) -> None:
        self.log("label_renamed", old, {"new_name": new})

    def log_image_labeled(self, image_path: str, label: str) -> None:
        self.log("image_labeled", os.path.basename(image_path), {"label": label})

    def log_roi_added(self, image_path: str, roi_id: str, roi_type: str) -> None:
        self.log("roi_added", os.path.basename(image_path), {"roi_id": roi_id, "type": roi_type})

    def log_roi_deleted(self, image_path: str, roi_id: str) -> None:
        self.log("roi_deleted", os.path.basename(image_path), {"roi_id": roi_id})

    def log_training_started(self, run_id: str, config: Dict) -> None:
        self.log("training_started", run_id, {"config": config})

    def log_training_finished(self, run_id: str, metrics: Dict) -> None:
        self.log("training_finished", run_id, {"metrics": metrics})

    def log_project_saved(self, path: str) -> None:
        self.log("project_saved", "", {"path": path})

    def log_project_backup(self, backup_path: str) -> None:
        self.log("project_backup", "", {"backup_path": backup_path})

    def log_inference(self, folder: str, model: str, count: int) -> None:
        self.log("inference_run", folder, {"model": model, "image_count": count})

    # ------------------------------------------------------------------ read

    def get_entries(self, limit: int = 200) -> List[Dict]:
        """
        Return the most recent *limit* entries from the log file.

        Silently skips malformed JSON lines. Returns an empty list if the
        file does not exist or cannot be read.
        """
        if not os.path.exists(self._file):
            return []
        entries = []
        try:
            with open(self._file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        return entries[-limit:]

    def get_label_history(self, label_name: str) -> List[Dict]:
        """Return all audit entries whose entity matches *label_name* (up to 1000)."""
        return [e for e in self.get_entries(1000) if e.get("entity") == label_name]

    def format_entries(self, entries: List[Dict]) -> str:
        """Format a list of audit entries as a human-readable multi-line string (newest first)."""
        lines = []
        for e in reversed(entries):
            ts = e.get("ts", "")[:19].replace("T", " ")
            action = e.get("action", "")
            entity = e.get("entity", "")
            details = e.get("details", {})
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items() if k != "config")
            lines.append(f"[{ts}] {action:25s}  {entity}  {detail_str}")
        return "\n".join(lines) if lines else "Noch keine Einträge."
