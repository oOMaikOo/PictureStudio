"""Load the monitor HTML frontends from ``monitor_web/`` (bundle-aware)."""
from __future__ import annotations

import os
import sys
from functools import lru_cache


def _web_dir() -> str:
    """Directory holding the .html templates (repo root or PyInstaller bundle)."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "monitor_web")


@lru_cache(maxsize=None)
def load_html(name: str) -> str:
    """Return the contents of ``monitor_web/<name>.html`` (cached)."""
    with open(os.path.join(_web_dir(), f"{name}.html"), encoding="utf-8") as f:
        return f.read()
