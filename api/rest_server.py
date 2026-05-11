"""
Lightweight REST API server for Image Labeling Studio.
Runs in a background daemon thread; no extra dependencies required.

Endpoints
---------
GET  /api/status                — server status + project summary
GET  /api/project               — full project statistics
GET  /api/labels                — label definitions (name, color, description)
GET  /api/images[?labeled=true] — all images with their label(s)
GET  /api/images/<filename>     — single image info + ROIs
POST /api/images/label          — assign a label  {path, label}
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse


class _ProjectHandler(BaseHTTPRequestHandler):
    # Shared state injected by RestApiServer before starting
    project = None
    request_count: int = 0

    # ------------------------------------------------------------------ util

    def log_message(self, fmt, *args) -> None:
        pass  # suppress default stdout noise

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg: str, status: int = 400) -> None:
        self._send_json({"error": msg}, status)

    def _read_body(self) -> Optional[dict]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length else {}
        except Exception:
            return None

    def _resolve_image(self, raw: str) -> Optional[str]:
        """Resolve a full path or bare filename to a project image path."""
        proj = _ProjectHandler.project
        if not proj:
            return None
        if raw in proj.images:
            return raw
        matches = [p for p in proj.images if os.path.basename(p) == raw]
        return matches[0] if matches else None

    # ------------------------------------------------------------------ CORS pre-flight

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ------------------------------------------------------------------ GET

    def do_GET(self) -> None:
        _ProjectHandler.request_count += 1
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)
        proj = _ProjectHandler.project

        # /api/status
        if path == "/api/status":
            self._send_json({
                "status": "running",
                "project": proj.config.name if proj else None,
                "project_path": proj.project_path if proj else None,
                "total_images": len(proj.images) if proj else 0,
                "labeled_images": proj.get_labeled_image_count() if proj else 0,
                "multi_label": proj.config.multi_label if proj else False,
                "requests_served": _ProjectHandler.request_count,
                "version": "1.0",
            })
            return

        if proj is None:
            self._err("No project loaded", 503)
            return

        # /api/project
        if path == "/api/project":
            self._send_json({
                "name": proj.config.name,
                "description": proj.config.description,
                "image_dir": proj.config.image_dir,
                "multi_label": proj.config.multi_label,
                "total_images": len(proj.images),
                "labeled_images": proj.get_labeled_image_count(),
                "unlabeled_images": len(proj.get_unlabeled_images()),
                "total_rois": proj.get_roi_count(),
                "labels": list(proj.labels.keys()),
                "label_counts": proj.get_label_counts(),
                "training_runs": len(proj.training_runs),
                "current_model": os.path.basename(proj.current_model_path),
            })
            return

        # /api/labels
        if path == "/api/labels":
            labels = [
                {"name": name, "color": info.get("color", "#888"),
                 "description": info.get("description", "")}
                for name, info in proj.labels.items()
            ]
            self._send_json({"labels": labels, "count": len(labels)})
            return

        # /api/images
        if path == "/api/images":
            filter_labeled = qs.get("labeled", [None])[0]
            images = []
            for img_path in proj.images:
                if proj.is_multi_label:
                    lbls = proj.get_image_multi_labels(img_path)
                else:
                    lbl = proj.get_image_label(img_path)
                    lbls = [lbl] if lbl else []
                is_labeled = bool(lbls)

                if filter_labeled == "true" and not is_labeled:
                    continue
                if filter_labeled == "false" and is_labeled:
                    continue

                images.append({
                    "path": img_path,
                    "filename": os.path.basename(img_path),
                    "label": lbls[0] if lbls else "",
                    "labels": lbls,
                    "roi_count": len(proj.get_rois(img_path)),
                })
            self._send_json({"images": images, "count": len(images)})
            return

        # /api/images/<filename>
        if path.startswith("/api/images/"):
            fname = path[len("/api/images/"):]
            img_path = self._resolve_image(fname)
            if img_path is None:
                self._err(f"Image '{fname}' not found in project", 404)
                return
            if proj.is_multi_label:
                lbls = proj.get_image_multi_labels(img_path)
            else:
                lbl = proj.get_image_label(img_path)
                lbls = [lbl] if lbl else []
            self._send_json({
                "path": img_path,
                "filename": os.path.basename(img_path),
                "label": lbls[0] if lbls else "",
                "labels": lbls,
                "rois": proj.get_rois(img_path),
            })
            return

        self._err("Endpoint not found", 404)

    # ------------------------------------------------------------------ POST

    def do_POST(self) -> None:
        _ProjectHandler.request_count += 1
        path = urlparse(self.path).path.rstrip("/")
        proj = _ProjectHandler.project

        body = self._read_body()
        if body is None:
            self._err("Invalid JSON body")
            return

        if proj is None:
            self._err("No project loaded", 503)
            return

        # POST /api/images/label  — assign a label to one image
        if path == "/api/images/label":
            raw_path = body.get("path", "")
            label = body.get("label", "")
            if not raw_path:
                self._err("'path' field required")
                return
            img_path = self._resolve_image(raw_path)
            if img_path is None:
                self._err(f"Image '{raw_path}' not found in project", 404)
                return
            if label and label not in proj.labels:
                self._err(f"Label '{label}' not defined. "
                          f"Known labels: {list(proj.labels.keys())}", 422)
                return
            proj.set_image_label(img_path, label)
            self._send_json({"ok": True, "path": img_path, "label": label})
            return

        # POST /api/images/multilabel  — assign multiple labels
        if path == "/api/images/multilabel":
            raw_path = body.get("path", "")
            labels = body.get("labels", [])
            if not raw_path:
                self._err("'path' field required")
                return
            if not isinstance(labels, list):
                self._err("'labels' must be an array")
                return
            img_path = self._resolve_image(raw_path)
            if img_path is None:
                self._err(f"Image '{raw_path}' not found in project", 404)
                return
            unknown = [l for l in labels if l and l not in proj.labels]
            if unknown:
                self._err(f"Unknown labels: {unknown}. "
                          f"Known: {list(proj.labels.keys())}", 422)
                return
            proj.set_image_multi_labels(img_path, labels)
            if labels:
                proj.set_image_label(img_path, labels[0])
            self._send_json({"ok": True, "path": img_path, "labels": labels})
            return

        self._err("Endpoint not found", 404)


class RestApiServer:
    """
    Manages a single-threaded HTTP server in a background daemon thread.
    Thread-safe for read operations against the project; label writes are
    fire-and-forget (no UI undo stack integration).
    """

    def __init__(self):
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: int = 8765
        self._status_cb: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------ config

    def set_status_callback(self, cb: Callable[[str], None]) -> None:
        self._status_cb = cb

    def set_project(self, project) -> None:
        _ProjectHandler.project = project

    # ------------------------------------------------------------------ state

    @property
    def is_running(self) -> bool:
        return self._server is not None

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}/api/" if self.is_running else ""

    # ------------------------------------------------------------------ control

    def start(self, port: int = 8765) -> bool:
        if self.is_running:
            return True
        self._port = port
        _ProjectHandler.request_count = 0
        try:
            server = HTTPServer(("", port), _ProjectHandler)
            server.allow_reuse_address = True
            self._server = server
            self._thread = threading.Thread(
                target=server.serve_forever,
                name="RestApiThread",
                daemon=True,
            )
            self._thread.start()
            self._notify(f"Läuft auf http://localhost:{port}/api/")
            return True
        except OSError as exc:
            self._server = None
            self._thread = None
            self._notify(f"Startfehler: {exc}")
            return False

    def stop(self) -> None:
        if not self._server:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None
        self._notify("Gestoppt")

    def _notify(self, msg: str) -> None:
        if self._status_cb:
            self._status_cb(msg)
