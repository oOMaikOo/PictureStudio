"""
Lightweight REST API server for Picture Studio.
Runs in a background daemon thread; no extra dependencies required.

Endpoints
---------
GET  /api/status                — server status + project summary
GET  /api/project               — full project statistics
GET  /api/labels                — label definitions (name, color, description)
GET  /api/images[?labeled=true] — all images with their label(s)
GET  /api/images/<filename>     — single image info + ROIs
GET  /api/scores                — last N anomaly scores (live monitoring feed)
GET  /api/events[?limit=N]      — recent anomaly events from CSV log
GET  /dashboard                 — self-contained HTML monitoring dashboard
POST /api/images/label          — assign a label  {path, label}
POST /api/classify              — live inference  {path, top_k?} or {image_b64, top_k?}
"""
import base64
import json
import logging
import os
import tempfile
import threading
import time as _time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

_MAX_IMAGE_BYTES: int = 50 * 1024 * 1024   # 50 MB — max decoded image size
_MAX_BODY_BYTES:  int = 70 * 1024 * 1024   # 70 MB — max raw request body

from utils.config import APP_VERSION

log = logging.getLogger("ImageLabelingStudio.rest_server")


class _ProjectHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the REST API.

    Class-level attributes are shared across all requests (single-threaded
    serve_forever loop):
      project        — current Project (set by RestApiServer.set_project).
      inferencer     — Inferencer for /api/classify, or None.
      score_buffer   — Rolling list of {ts, score, threshold, alarm} dicts.
      event_log_path — Path to anomaly_events.csv for /api/events.
    """

    # Shared state injected by RestApiServer before starting
    project = None
    inferencer = None   # core.inference.Inferencer or None
    request_count: int = 0
    score_buffer: list = []          # rolling list of (timestamp, score, threshold)
    event_log_path: str = ""         # path to anomaly_events.csv (set externally)
    alarm_frame_dir: str = ""        # directory where alarm JPEG snapshots are saved
    latest_alarm: dict = {}          # most recent alarm: {ts, score, threshold, frame_path}
    _api_key: str = ""               # shared secret; empty = no auth required
    # Multi-camera per-channel state: list of dicts, one per channel
    mc_channels: list = []           # [{channel, score, threshold, is_alarm, event_count,
                                     #   cam_status, score_buffer, latest_alarm}, ...]
    # Rate-limiting for /api/classify: one request at a time, max 10/s
    _classify_lock = threading.Lock()
    _classify_last_t: float = 0.0
    _CLASSIFY_MIN_INTERVAL: float = 0.1   # seconds between calls (10 req/s)

    # ------------------------------------------------------------------ util

    def log_message(self, fmt, *args) -> None:
        pass  # suppress the default BaseHTTPRequestHandler stdout noise

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Serialise *data* to JSON and send it as the HTTP response with CORS headers."""
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
        """Send a JSON error response with the given HTTP status code."""
        self._send_json({"error": msg}, status)

    def _read_body(self) -> Optional[dict]:
        """Parse the JSON request body. Returns None and logs a warning on failure."""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            log.warning("REST: ungültiger Content-Length Header von %s", self.client_address)
            return None
        if not (0 <= length <= _MAX_BODY_BYTES):
            log.warning("REST: Content-Length %d überschreitet Limit %d", length, _MAX_BODY_BYTES)
            return None
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            log.warning("REST: JSON-Parsefehler im Request-Body: %s", exc)
            return None
        except Exception as exc:
            log.warning("REST: Body-Lesefehler: %s", exc)
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

    # ------------------------------------------------------------------ auth

    # Public endpoints that never require a key (monitoring dashboards need them)
    _PUBLIC_PATHS = {"/api/status", "/dashboard", "/dashboard/"}

    def _check_auth(self) -> bool:
        """Return True if the request is authorised (or no key is configured)."""
        key = _ProjectHandler._api_key
        if not key:
            return True
        provided = (
            self.headers.get("X-Api-Key", "")
            or self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        return provided == key

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
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)
        proj = _ProjectHandler.project

        if path not in self._PUBLIC_PATHS and not self._check_auth():
            self._err("Unauthorized — provide X-Api-Key header", 401)
            return

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
                "version": APP_VERSION,
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

        # /api/scores — live anomaly score buffer
        if path == "/api/scores":
            try:
                limit = max(1, min(int(qs.get("limit", ["120"])[0]), 2000))
            except (ValueError, TypeError):
                self._err("'limit' muss eine positive Ganzzahl sein", 400)
                return
            buf = _ProjectHandler.score_buffer[-limit:]
            self._send_json({"scores": buf, "count": len(buf)})
            return

        # /api/events — recent anomaly events from CSV log
        if path == "/api/events":
            try:
                limit = max(1, min(int(qs.get("limit", ["100"])[0]), 2000))
            except (ValueError, TypeError):
                self._err("'limit' muss eine positive Ganzzahl sein", 400)
                return
            events = self._read_event_log(limit)
            self._send_json({"events": events, "count": len(events)})
            return

        # /api/latest_alarm — last alarm event (score, threshold, frame filename)
        if path == "/api/latest_alarm":
            self._send_json(_ProjectHandler.latest_alarm or {})
            return

        # /api/frame/<filename> — serve a saved alarm frame JPEG
        if path.startswith("/api/frame/"):
            fname = path[len("/api/frame/"):]
            frame_dir = _ProjectHandler.alarm_frame_dir
            if not frame_dir:
                self._err("No alarm frame directory configured", 503)
                return
            if not fname:
                self._err("Invalid filename", 400)
                return
            # Prevent path traversal: resolve both paths and verify containment
            safe_dir = os.path.realpath(frame_dir)
            fpath = os.path.realpath(os.path.join(frame_dir, fname))
            if not fpath.startswith(safe_dir + os.sep):
                log.warning("REST: Path-Traversal-Versuch: '%s' von %s", fname, self.client_address)
                self._err("Invalid filename", 400)
                return
            if not os.path.isfile(fpath):
                self._err(f"Frame '{fname}' not found", 404)
                return
            try:
                with open(fpath, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                self._err(str(exc), 500)
            return

        # /dashboard — HTML monitoring dashboard
        if path in ("/dashboard", "/dashboard/"):
            html = _build_dashboard_html(_ProjectHandler._api_key)
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        # /api/mc/channels — multi-camera channel summary
        if path == "/api/mc/channels":
            channels = _ProjectHandler.mc_channels
            summary = [
                {
                    "channel":     ch.get("channel", i),
                    "score":       ch.get("score", 0.0),
                    "threshold":   ch.get("threshold", 0.0),
                    "is_alarm":    ch.get("is_alarm", False),
                    "event_count": ch.get("event_count", 0),
                    "cam_status":  ch.get("cam_status", "Gestoppt"),
                }
                for i, ch in enumerate(channels)
            ]
            self._send_json({"channels": summary, "count": len(summary)})
            return

        # /api/mc/scores?channel=N — per-channel score buffer
        if path == "/api/mc/scores":
            try:
                channel = int(qs.get("channel", ["0"])[0])
            except ValueError:
                self._err("'channel' must be an integer", 400)
                return
            limit = int(qs.get("limit", ["120"])[0])
            channels = _ProjectHandler.mc_channels
            if channel < 0 or channel >= len(channels):
                self._err(f"Channel {channel} not found", 404)
                return
            buf = channels[channel].get("score_buffer", [])[-limit:]
            self._send_json({"channel": channel, "scores": buf, "count": len(buf)})
            return

        # /api/mc/latest_alarm?channel=N — per-channel latest alarm
        if path == "/api/mc/latest_alarm":
            try:
                channel = int(qs.get("channel", ["0"])[0])
            except ValueError:
                self._err("'channel' must be an integer", 400)
                return
            channels = _ProjectHandler.mc_channels
            if channel < 0 or channel >= len(channels):
                self._err(f"Channel {channel} not found", 404)
                return
            self._send_json(channels[channel].get("latest_alarm", {}))
            return

        self._err("Endpoint not found", 404)

    def _read_event_log(self, limit: int) -> list:
        path = _ProjectHandler.event_log_path
        if not path or not os.path.isfile(path):
            return []
        import csv as csv_mod
        rows = []
        try:
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
        except Exception as exc:
            log.warning("Event-Log konnte nicht gelesen werden (%s): %s", path, exc)
            return []
        return rows[-limit:]

    # ------------------------------------------------------------------ POST

    def do_POST(self) -> None:
        _ProjectHandler.request_count += 1
        path = urlparse(self.path).path.rstrip("/")
        proj = _ProjectHandler.project

        if not self._check_auth():
            self._err("Unauthorized — provide X-Api-Key header", 401)
            return

        body = self._read_body()
        if body is None:
            self._err("Ungültiger JSON-Body — Content-Type: application/json erwartet", 400)
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

        # POST /api/classify — live inference on a single image
        if path == "/api/classify":
            if not _ProjectHandler._classify_lock.acquire(blocking=False):
                self._err("Eine Klassifizierung läuft bereits. Bitte kurz warten.", 429)
                return
            try:
                now = _time.monotonic()
                elapsed = now - _ProjectHandler._classify_last_t
                if elapsed < _ProjectHandler._CLASSIFY_MIN_INTERVAL:
                    self._err(
                        f"Rate limit: max 10 Anfragen/Sekunde. "
                        f"Bitte {_ProjectHandler._CLASSIFY_MIN_INTERVAL - elapsed:.2f}s warten.",
                        429,
                    )
                    return
                _ProjectHandler._classify_last_t = now
            finally:
                _ProjectHandler._classify_lock.release()

            inf = _ProjectHandler.inferencer
            if inf is None or not inf.is_ready():
                self._err("No model loaded. Load a model via the Models page first.", 503)
                return

            try:
                top_k = max(1, min(int(body.get("top_k", 3)), 20))
            except (ValueError, TypeError):
                self._err("'top_k' muss eine Ganzzahl zwischen 1 und 20 sein", 400)
                return
            img_path = body.get("path", "")
            if not isinstance(img_path, str):
                self._err("'path' muss ein String sein", 400)
                return
            b64 = body.get("image_b64", "")

            tmp_path = None
            if b64:
                # Guard against DoS: check encoded length before decoding
                if len(b64) > _MAX_IMAGE_BYTES * 4 // 3 + 4:
                    self._err(
                        f"Image too large — max {_MAX_IMAGE_BYTES // 1_048_576} MB", 413
                    )
                    return
                try:
                    img_bytes = base64.b64decode(b64)
                except Exception as exc:
                    self._err(f"Base64 decode error: {exc}")
                    return
                if len(img_bytes) > _MAX_IMAGE_BYTES:
                    self._err(
                        f"Decoded image too large — max {_MAX_IMAGE_BYTES // 1_048_576} MB", 413
                    )
                    return
                suffix = ".png" if img_bytes[:4] == b'\x89PNG' else ".jpg"
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(img_bytes)
                        tmp_path = f.name
                    img_path = tmp_path
                except Exception as exc:
                    self._err(f"Temp-Datei konnte nicht erstellt werden: {exc}", 500)
                    return
            elif not img_path:
                self._err("Provide 'path' (file path) or 'image_b64' (base64 image).")
                return
            elif not os.path.isfile(img_path):
                self._err(f"File not found: {img_path}", 404)
                return

            try:
                result = inf.classify_single(img_path, top_k=top_k)
                response = {
                    "predicted_label": result.get("predicted_label", ""),
                    "confidence":      result.get("confidence", 0.0),
                    "top_k":           result.get("top_k", []),
                    "low_confidence":  result.get("low_confidence", False),
                    "path":            img_path,
                }
                self._send_json(response)
            except Exception as exc:
                self._err(f"Inference error: {exc}", 500)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            return

        self._err("Endpoint not found", 404)


# ── Dashboard HTML ─────────────────────────────────────────────────────────

def _build_dashboard_html(api_key: str = "") -> str:
    # json.dumps produces a valid JS string literal with proper escaping for
    # quotes, backslashes, and control characters — prevents XSS via the key.
    _key_js = json.dumps(api_key) if api_key else '""'
    _html = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Picture Studio – Monitoring Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #22272E; color: #CDD9E5;
         padding: 20px; }
  h1 { color: #5DADE2; margin-bottom: 16px; font-size: 22px; }
  h2 { color: #7FC3F5; font-size: 14px; margin-bottom: 8px; letter-spacing: .05em; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 14px; margin-bottom: 20px; }
  .card { background: #2D333B; border-radius: 8px; padding: 16px; border: 1px solid #373E47; }
  .card .val { font-size: 30px; font-weight: bold; color: #58D68D; margin: 6px 0; }
  .card .val.alarm { color: #E74C3C; }
  .card .lbl { font-size: 11px; color: #768390; }
  #chart-wrap { background: #2D333B; border-radius: 8px; padding: 16px;
                border: 1px solid #373E47; margin-bottom: 20px; }
  canvas { width: 100%; height: 180px; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #373E47; color: #CDD9E5; padding: 8px; text-align: left; }
  td { padding: 7px 8px; border-bottom: 1px solid #373E47; }
  tr.alarm td { background: rgba(231,76,60,.15); }
  #status-dot { display: inline-block; width: 10px; height: 10px;
                border-radius: 50%; background: #58D68D; margin-right: 6px; }
  #status-dot.err { background: #E74C3C; }
  .section { background: #2D333B; border-radius: 8px; padding: 16px;
             border: 1px solid #373E47; }
  .footer { font-size: 10px; color: #545D68; margin-top: 14px; text-align: center; }
  #alarm-frame-wrap { background: #2D333B; border-radius: 8px; padding: 16px;
                      border: 2px solid #E74C3C; margin-bottom: 20px; display:none; }
  #alarm-frame-wrap h2 { color: #E74C3C; }
  #alarm-img { max-width: 100%; border-radius: 6px; margin-top: 8px; display: block; }
  #alarm-meta { font-size: 11px; color: #768390; margin-top: 6px; }
</style>
</head>
<body>
<h1><span id="status-dot"></span>Picture Studio – Live Monitor</h1>

<div class="grid">
  <div class="card">
    <div class="lbl">Letzter Score</div>
    <div class="val" id="last-score">–</div>
    <div class="lbl" id="score-pct">–</div>
  </div>
  <div class="card">
    <div class="lbl">Schwellwert</div>
    <div class="val" id="threshold">–</div>
  </div>
  <div class="card">
    <div class="lbl">Alarme (Session)</div>
    <div class="val alarm" id="alarm-count">–</div>
  </div>
  <div class="card">
    <div class="lbl">Gesamt-Events (Log)</div>
    <div class="val" id="event-count">–</div>
  </div>
</div>

<div id="chart-wrap">
  <h2>Score-Verlauf (letzte 120 Frames)</h2>
  <canvas id="chart"></canvas>
</div>

<div id="alarm-frame-wrap">
  <h2>⚠  Letzter Alarm-Frame</h2>
  <img id="alarm-img" src="" alt="Alarm-Frame">
  <div id="alarm-meta"></div>
</div>

<div class="section">
  <h2>Letzte Anomalie-Events</h2>
  <table>
    <thead><tr><th>Zeit</th><th>Score</th><th>%</th><th>Bild</th></tr></thead>
    <tbody id="events-body"><tr><td colspan="4" style="color:#545D68">Lade…</td></tr></tbody>
  </table>
</div>

<div class="section" id="mc-section" style="margin-top:16px;display:none">
  <h2>Multi-Kamera-Kanäle</h2>
  <div id="mc-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-top:10px"></div>
</div>

<div class="footer">Aktualisiert alle 3 Sekunden &middot; Picture Studio</div>

<script>
const BASE = window.location.origin;
const API_KEY = __API_KEY__;
const _headers = API_KEY ? {'X-Api-Key': API_KEY} : {};
function apiFetch(url) { return fetch(url, {headers: _headers}); }
let scores = [];
let threshold = 0;
let alarmCount = 0;

function draw(canvas, scores, thr) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight || 180;
  ctx.clearRect(0, 0, W, H);
  if (!scores.length) return;
  const vals = scores.map(s => s.score);
  const max = Math.max(...vals, thr * 1.5, 0.001);
  const n = vals.length;
  const dx = W / Math.max(n - 1, 1);

  // threshold line
  const ty = H - (thr / max) * H;
  ctx.strokeStyle = '#E67E22';
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke();
  ctx.setLineDash([]);

  // score line
  ctx.lineWidth = 2;
  vals.forEach((v, i) => {
    const x = i * dx;
    const y = H - (v / max) * H;
    ctx.strokeStyle = v > thr ? '#E74C3C' : '#2ECC71';
    if (i === 0) { ctx.beginPath(); ctx.moveTo(x, y); }
    else { ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y); }
  });
}

let lastAlarmFilename = '';

async function fetchData() {
  try {
    const [sRes, eRes, aRes] = await Promise.all([
      apiFetch(BASE + '/api/scores?limit=120'),
      apiFetch(BASE + '/api/events?limit=20'),
      apiFetch(BASE + '/api/latest_alarm'),
    ]);
    const sData = await sRes.json();
    const eData = await eRes.json();
    const aData = await aRes.json();

    scores = sData.scores || [];
    alarmCount = scores.filter(s => s.alarm).length;
    const last = scores[scores.length - 1];
    threshold = last ? last.threshold : 0;

    document.getElementById('status-dot').className = '';
    if (last) {
      document.getElementById('last-score').textContent = last.score.toFixed(5);
      document.getElementById('last-score').className = 'val' + (last.alarm ? ' alarm' : '');
      const pct = threshold > 0 ? Math.round(last.score / threshold * 100) : 0;
      document.getElementById('score-pct').textContent = pct + '% des Schwellwerts';
      document.getElementById('threshold').textContent = threshold.toFixed(5);
    }
    document.getElementById('alarm-count').textContent = alarmCount;
    document.getElementById('event-count').textContent = eData.count;

    const canvas = document.getElementById('chart');
    draw(canvas, scores, threshold);

    // Latest alarm frame
    const frameWrap = document.getElementById('alarm-frame-wrap');
    if (aData.frame_filename) {
      frameWrap.style.display = 'block';
      if (aData.frame_filename !== lastAlarmFilename) {
        lastAlarmFilename = aData.frame_filename;
        document.getElementById('alarm-img').src =
          BASE + '/api/frame/' + aData.frame_filename + '?t=' + Date.now();
      }
      document.getElementById('alarm-meta').textContent =
        aData.ts + '  |  Score: ' + (aData.score||0).toFixed(5) +
        '  |  Schwellwert: ' + (aData.threshold||0).toFixed(5);
    } else {
      frameWrap.style.display = 'none';
    }

    const tbody = document.getElementById('events-body');
    tbody.innerHTML = '';
    const events = (eData.events || []).slice().reverse();
    if (!events.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#545D68">Keine Events</td></tr>';
    } else {
      events.forEach(ev => {
        const tr = document.createElement('tr');
        if (parseFloat(ev.score) > parseFloat(ev.threshold)) tr.className = 'alarm';
        const frameLink = ev.frame_path
          ? '<a href="' + BASE + '/api/frame/' + ev.frame_path + '" target="_blank" style="color:#5DADE2">' + ev.frame_path + '</a>'
          : '–';
        tr.innerHTML = '<td>' + (ev.timestamp_utc||ev.timestamp||'–') + '</td>'
          + '<td>' + parseFloat(ev.score||0).toFixed(5) + '</td>'
          + '<td>' + (ev.score_pct||'–') + '%</td>'
          + '<td style="font-size:10px">' + frameLink + '</td>';
        tbody.appendChild(tr);
      });
    }
  } catch(e) {
    document.getElementById('status-dot').className = 'err';
    console.warn('fetch error', e);
  }

  // Multi-camera section
  try {
    const mcRes = await apiFetch(BASE + '/api/mc/channels');
    if (mcRes.ok) {
      const mcData = await mcRes.json();
      const channels = mcData.channels || [];
      const mcSection = document.getElementById('mc-section');
      const mcGrid = document.getElementById('mc-grid');
      if (channels.length > 0) {
        mcSection.style.display = 'block';
        mcGrid.innerHTML = channels.map((ch, i) => {
          const pct = ch.threshold > 0 ? Math.round(ch.score / ch.threshold * 100) : 0;
          const alarmStyle = ch.is_alarm ? 'color:#E74C3C' : 'color:#58D68D';
          return `<div style="background:#1C2A3A;border-radius:6px;padding:10px;border:1px solid #2C3E50">
            <div style="font-size:11px;color:#5DADE2;font-weight:bold;margin-bottom:6px">Kanal ${i+1}</div>
            <div style="font-size:13px;${alarmStyle}">${ch.is_alarm?'ANOMALIE':'Normal'}</div>
            <div style="font-size:11px;color:#768390">Score: ${ch.score.toFixed(5)}</div>
            <div style="font-size:11px;color:#768390">Thr: ${ch.threshold.toFixed(5)}</div>
            <div style="font-size:11px;color:#E67E22">Alarme: ${ch.event_count}</div>
            <div style="font-size:10px;color:#545D68;margin-top:3px">${ch.cam_status}</div>
          </div>`;
        }).join('');
      } else {
        mcSection.style.display = 'none';
      }
    }
  } catch(e) { /* multi-camera section not available — hide silently */ }
}

fetchData();
setInterval(fetchData, 3000);
</script>
</body>
</html>"""
    return _html.replace("__API_KEY__", _key_js)


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

    def set_api_key(self, key: str) -> None:
        """Set (or clear) the shared API key. Empty string disables auth."""
        _ProjectHandler._api_key = key or ""

    def set_inferencer(self, inferencer) -> None:
        _ProjectHandler.inferencer = inferencer

    def push_score(self, score: float, threshold: float) -> None:
        """Thread-safe: push one score to the live score buffer (keeps last 500)."""
        from datetime import datetime
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "alarm": score > threshold,
        }
        _ProjectHandler.score_buffer.append(entry)
        if len(_ProjectHandler.score_buffer) > 500:
            _ProjectHandler.score_buffer = _ProjectHandler.score_buffer[-500:]

    def set_event_log_path(self, path: str) -> None:
        _ProjectHandler.event_log_path = path

    def set_alarm_frame_dir(self, path: str) -> None:
        """Set the directory from which alarm frame JPEGs are served."""
        _ProjectHandler.alarm_frame_dir = path

    def push_latest_alarm(self, frame_path: str, score: float, threshold: float) -> None:
        """Record the most recent alarm event for the dashboard."""
        from datetime import datetime
        _ProjectHandler.latest_alarm = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "frame_path": frame_path,
            "frame_filename": os.path.basename(frame_path),
        }

    # ---- Multi-camera per-channel state ----

    def set_mc_channel_count(self, count: int) -> None:
        """(Re-)initialize the per-channel state list to *count* entries."""
        channels = _ProjectHandler.mc_channels
        while len(channels) < count:
            idx = len(channels)
            channels.append({
                "channel": idx, "score": 0.0, "threshold": 0.0,
                "is_alarm": False, "event_count": 0,
                "cam_status": "Gestoppt",
                "score_buffer": [], "latest_alarm": {},
            })
        while len(channels) > count:
            channels.pop()

    def push_mc_score(self, channel: int, score: float, threshold: float) -> None:
        """Push a live score for one multi-camera channel (thread-safe by GIL)."""
        from datetime import datetime, timezone
        channels = _ProjectHandler.mc_channels
        if channel < 0 or channel >= len(channels):
            return
        ch = channels[channel]
        ch["score"] = round(score, 6)
        ch["threshold"] = round(threshold, 6)
        ch["is_alarm"] = score > threshold
        entry = {
            "ts":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "score":     round(score, 6),
            "threshold": round(threshold, 6),
            "alarm":     score > threshold,
        }
        buf = ch.setdefault("score_buffer", [])
        buf.append(entry)
        if len(buf) > 500:
            ch["score_buffer"] = buf[-500:]

    def push_mc_alarm(
        self, channel: int, score: float, threshold: float, frame_filename: str
    ) -> None:
        """Record a per-channel alarm event (thread-safe by GIL)."""
        from datetime import datetime, timezone
        channels = _ProjectHandler.mc_channels
        if channel < 0 or channel >= len(channels):
            return
        ch = channels[channel]
        ch["event_count"] = ch.get("event_count", 0) + 1
        ch["latest_alarm"] = {
            "ts":             datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "score":          round(score, 6),
            "threshold":      round(threshold, 6),
            "frame_filename": frame_filename,
        }

    def set_mc_cam_status(self, channel: int, status: str) -> None:
        """Update the camera status label for one channel."""
        channels = _ProjectHandler.mc_channels
        if 0 <= channel < len(channels):
            channels[channel]["cam_status"] = status

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
            server = ThreadingHTTPServer(("", port), _ProjectHandler)
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
