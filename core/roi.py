"""ROI data helpers (thin wrappers; actual storage is in Project)."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import uuid


@dataclass
class ROI:
    """
    Dataclass representing a single Region of Interest on an image.

    Stored as plain dicts in Project.rois; this class is a convenience
    wrapper for creating, validating, and serialising ROI objects.
    Supported types: "rect" (axis-aligned bounding box) and "polygon".
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    image_path: str = ""
    roi_type: str = "rect"          # "rect" | "polygon"
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    label: str = ""
    color: str = "#E74C3C"
    points: List = field(default_factory=list)  # for polygon

    def to_dict(self) -> dict:
        """Serialise to a plain dict compatible with Project.rois storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ROI":
        """Construct an ROI from a dict, ignoring unknown keys for forward compatibility."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_valid(self) -> bool:
        """Return True when the ROI has a positive width and height."""
        return self.w > 0 and self.h > 0
