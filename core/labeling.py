"""Label management helpers (thin layer over Project)."""
from typing import Dict, List, Optional
from utils.config import DEFAULT_COLORS


class LabelManager:
    """
    Convenience wrapper around the project's label dict.
    Useful for non-GUI code that needs to manipulate labels.
    """

    def __init__(self, project=None):
        self.project = project

    def get_labels(self) -> Dict[str, Dict]:
        return self.project.labels if self.project else {}

    def get_label_names(self) -> List[str]:
        return list(self.project.labels.keys()) if self.project else []

    def get_color(self, label_name: str) -> str:
        if not self.project:
            return DEFAULT_COLORS[0]
        return self.project.labels.get(label_name, {}).get("color", DEFAULT_COLORS[0])

    def next_available_color(self) -> str:
        if not self.project:
            return DEFAULT_COLORS[0]
        used = {info["color"] for info in self.project.labels.values()}
        for c in DEFAULT_COLORS:
            if c not in used:
                return c
        return DEFAULT_COLORS[0]

    def add_label(self, name: str, color: Optional[str] = None, description: str = "") -> bool:
        if not self.project or name in self.project.labels:
            return False
        color = color or self.next_available_color()
        self.project.add_label(name, color, description)
        return True

    def remove_label(self, name: str) -> bool:
        if not self.project or name not in self.project.labels:
            return False
        self.project.remove_label(name)
        return True
