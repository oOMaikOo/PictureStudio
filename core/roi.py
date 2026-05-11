"""ROI data helpers (thin wrappers; actual storage is in Project)."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import uuid


@dataclass
class ROI:
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
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ROI":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_valid(self) -> bool:
        return self.w > 0 and self.h > 0
