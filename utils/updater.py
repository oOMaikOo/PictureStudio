"""
Update check against GitHub Releases and local CHANGELOG “What's new” text.

Network failures are non-fatal — callers treat ``None`` / empty string as
“nothing to show”.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_REPO = "oOMaikOo/Picture"
_TAG_RE = re.compile(r"^v?(\d+(?:\.\d+)*)", re.I)


def parse_version(text: str) -> Tuple[int, ...]:
    """Parse ``1.2.3`` / ``v1.2.3`` into an int tuple; unknown → ``(0,)``."""
    if not text:
        return (0,)
    m = _TAG_RE.match(str(text).strip())
    if not m:
        return (0,)
    return tuple(int(p) for p in m.group(1).split("."))


def is_newer(remote: str, local: str) -> bool:
    """Return True when *remote* version is strictly greater than *local*."""
    return parse_version(remote) > parse_version(local)


def fetch_latest_release_tag(
    repo: str = DEFAULT_REPO,
    timeout: float = 5.0,
    opener=None,
) -> Optional[str]:
    """
    GET GitHub ``/releases/latest`` and return ``tag_name`` (e.g. ``v2.5.3``).

    *opener* is an optional callable ``(Request) -> file-like`` for tests.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "PictureStudio-UpdateCheck",
        },
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tag = (data.get("tag_name") or "").strip()
        return tag or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError, ValueError) as exc:
        log.debug("Update check failed: %s", exc)
        return None


def changelog_whats_new(changelog_text: str, since_version: str) -> str:
    """
    Extract CHANGELOG sections newer than *since_version*.

    Prefer the ``## Unreleased`` block; then any ``## x.y.z`` headers newer
    than *since_version*. Returns a plain-text summary (max ~2 KB).
    """
    if not changelog_text:
        return ""
    lines = changelog_text.replace("\r\n", "\n").split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_key: Optional[str] = None
    current_body: list[str] = []

    def _flush() -> None:
        nonlocal current_key, current_body
        if current_key is not None:
            sections.append((current_key, current_body))
        current_key, current_body = None, []

    for line in lines:
        if line.startswith("## "):
            _flush()
            current_key = line[3:].strip()
            current_body = []
        elif current_key is not None:
            current_body.append(line)
    _flush()

    chunks: list[str] = []
    since = parse_version(since_version)
    for key, body in sections:
        key_l = key.lower()
        if key_l.startswith("unreleased"):
            text = "\n".join(body).strip()
            if text:
                chunks.append(f"Unreleased\n{text}")
            continue
        m = _TAG_RE.match(key)
        if not m:
            continue
        if parse_version(m.group(1)) > since:
            text = "\n".join(body).strip()
            if text:
                chunks.append(f"{key}\n{text}")

    out = "\n\n".join(chunks).strip()
    if len(out) > 2000:
        out = out[:1990].rstrip() + "…"
    return out


def load_changelog(path: Optional[str] = None) -> str:
    """Read CHANGELOG.md from *path* or the project root next to this package."""
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "CHANGELOG.md")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""
