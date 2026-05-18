"""Label management helpers (thin layer over Project)."""
from typing import Dict, List, Optional
from utils.config import DEFAULT_COLORS


class LabelManager:
    """
    Convenience wrapper around the project's label dict.

    Thin façade over Project.labels; useful for non-GUI code that needs to
    add, remove, or inspect labels without directly touching project internals.
    """

    def __init__(self, project=None):
        self.project = project

    def get_labels(self) -> Dict[str, Dict]:
        """Return the full label dict {name: {color, description, parent}}."""
        return self.project.labels if self.project else {}

    def get_label_names(self) -> List[str]:
        """Return a list of all label name strings."""
        return list(self.project.labels.keys()) if self.project else []

    def get_color(self, label_name: str) -> str:
        """Return the hex colour string for *label_name*, or the first default colour."""
        if not self.project:
            return DEFAULT_COLORS[0]
        return self.project.labels.get(label_name, {}).get("color", DEFAULT_COLORS[0])

    def next_available_color(self) -> str:
        """Return the first colour from DEFAULT_COLORS not already in use."""
        if not self.project:
            return DEFAULT_COLORS[0]
        used = {info["color"] for info in self.project.labels.values()}
        for c in DEFAULT_COLORS:
            if c not in used:
                return c
        return DEFAULT_COLORS[0]

    def add_label(self, name: str, color: Optional[str] = None, description: str = "") -> bool:
        """
        Add a new label to the project.

        Returns False if no project is set or the label already exists.
        A colour is auto-selected when *color* is not provided.
        """
        if not self.project or name in self.project.labels:
            return False
        color = color or self.next_available_color()
        self.project.add_label(name, color, description)
        return True

    def remove_label(self, name: str) -> bool:
        """Remove *name* from the project. Returns False when not found."""
        if not self.project or name not in self.project.labels:
            return False
        self.project.remove_label(name)
        return True
