#!/usr/bin/env python3
"""
PictureStudio Monitor-Client — Standalone Anomalie-Erkennung für den Produktionseinsatz.

Lädt ein trainiertes Modell und verbindet sich mit der Kamera.
ROI und Schwellwert werden automatisch aus den Modell-Metadaten übernommen.

Verwendung:
    python monitor.py --model MODEL_PFAD [Optionen]

Kameraquellen:
    --camera INDEX        USB-Kamera-Index (Standard: aus Modell-Metadaten)
    --url URL             IP-Kamera (rtsp://…, http://…) oder Video-Datei (mp4, avi, …)

Alarmierung:
    --mqtt-host HOST      MQTT-Broker für Alarm-Publishing
    --api-port PORT       Eingebauten REST-Server + Dashboard starten

Beispiele:
    python monitor.py --model anomalie.pth
    python monitor.py --model anomalie.pth --url rtsp://admin:pass@192.168.1.100:554/stream
    python monitor.py --model anomalie.pth --url /pfad/video.mp4 --headless
    python monitor.py --model anomalie.pth --mqtt-host 192.168.1.50 --api-port 8766
    python monitor.py --model anomalie.onnx --headless --output /var/log/anomalien
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional, Union
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Project root on path so core/ imports work from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.anomaly_detector import AnomalyDetector
from core.camera import list_usb_cameras

try:
    from core.onnx_anomaly_scorer import OnnxAnomalyScorer, HAS_ORT
except ImportError:
    OnnxAnomalyScorer = None
    HAS_ORT = False

try:
    from core.mqtt_client import MQTTAlarmClient, HAS_MQTT
except ImportError:
    MQTTAlarmClient = None  # type: ignore[assignment,misc]
    HAS_MQTT = False

_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv', '.webm', '.ts'}


# ---------------------------------------------------------------------------
# Camera thread — USB index, IP/RTSP URL, or video file
# ---------------------------------------------------------------------------

class _CameraThread(threading.Thread):
    """
    Grabs frames from a cv2.VideoCapture source and calls *callback* for each.

    Source can be:
      - int   → USB camera index
      - str   → RTSP/HTTP URL  or  local video file path

    For live streams (USB + URL), auto-reconnect is attempted every
    *reconnect_delay* seconds when the stream drops (0 = disabled).
    Video files are played once without reconnect.
    """

    def __init__(
        self,
        source: Union[int, str],
        fps: float,
        callback: Callable[[np.ndarray], None],
        reconnect_delay: float = 5.0,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="monitor-cam")
        self._source = source
        self._fps = fps
        self._callback = callback
        self._reconnect_delay = reconnect_delay
        self._on_status = on_status
        self._running = False
        self._is_video = self._detect_video(source)
        self.error: Optional[str] = None

    @staticmethod
    def _detect_video(source: Union[int, str]) -> bool:
        if isinstance(source, int):
            return False
        ext = os.path.splitext(str(source))[1].lower()
        return ext in _VIDEO_EXTENSIONS

    def run(self) -> None:
        self._running = True
        attempt = 0

        while self._running:
            attempt += 1
            if attempt > 1 and self._on_status:
                self._on_status(f"Reconnect #{attempt - 1}…")

            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                self.error = f"Quelle konnte nicht geöffnet werden: {self._source}"
                if self._reconnect_delay > 0 and not self._is_video:
                    if self._on_status:
                        self._on_status(
                            f"Verbindung fehlgeschlagen — erneuter Versuch in "
                            f"{self._reconnect_delay:.0f} s"
                        )
                    time.sleep(self._reconnect_delay)
                    continue
                break

            self.error = None
            if self._on_status:
                s = "Wiedergabe" if self._is_video else "Verbunden"
                self._on_status(s)

            # Use native FPS for video files, configured FPS for live streams
            if self._is_video:
                native = cap.get(cv2.CAP_PROP_FPS)
                delay = 1.0 / max(native, 1.0) if native > 0 else 1.0 / max(self._fps, 1.0)
            else:
                delay = 1.0 / max(self._fps, 1.0)

            consec_fail = 0
            t_next = time.perf_counter()

            while self._running:
                now = time.perf_counter()
                if now < t_next:
                    time.sleep(0.005)
                    continue
                t_next = now + delay

                ret, frame = cap.read()
                if ret and frame is not None:
                    consec_fail = 0
                    try:
                        self._callback(frame)
                    except Exception:
                        pass
                else:
                    consec_fail += 1
                    if self._is_video or consec_fail >= 5:
                        break
                    time.sleep(0.1)

            cap.release()

            if not self._running:
                break
            if self._is_video:
                if self._on_status:
                    self._on_status("Video beendet")
                break
            if self._reconnect_delay <= 0:
                break
            if self._on_status:
                self._on_status(
                    f"Verbindung unterbrochen — Reconnect in {self._reconnect_delay:.0f} s"
                )
            time.sleep(self._reconnect_delay)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Shared monitor state (thread-safe; updated by the frame callback)
# ---------------------------------------------------------------------------

class _MonitorState:
    """Central store for score history and alarm data used by the REST API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model_name: str = ""
        self.threshold: float = 0.0
        self.score: float = 0.0
        self.is_alarm: bool = False
        self.event_count: int = 0
        self.cam_status: str = "Nicht verbunden"
        self.output_dir: str = ""
        self.api_key: str = ""
        self.score_buffer: list = []          # [{ts, score, threshold, alarm}]
        self.latest_alarm: dict = {}          # {ts, score, threshold, frame_filename}
        self.start_time: float = time.time()

    def push_score(self, score: float, threshold: float) -> None:
        entry = {
            "ts":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "score":     round(score, 6),
            "threshold": round(threshold, 6),
            "alarm":     score > threshold,
        }
        with self._lock:
            self.score_buffer.append(entry)
            if len(self.score_buffer) > 500:
                self.score_buffer = self.score_buffer[-500:]
            self.score = score
            self.is_alarm = score > threshold

    def push_alarm(self, score: float, threshold: float, frame_filename: str) -> None:
        with self._lock:
            self.latest_alarm = {
                "ts":             datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "score":          round(score, 6),
                "threshold":      round(threshold, 6),
                "frame_filename": frame_filename,
            }
            self.event_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "model_name":   self.model_name,
                "threshold":    self.threshold,
                "score":        self.score,
                "is_alarm":     self.is_alarm,
                "event_count":  self.event_count,
                "cam_status":   self.cam_status,
                "uptime_s":     int(time.time() - self.start_time),
                "score_count":  len(self.score_buffer),
            }


# ---------------------------------------------------------------------------
# Mini REST API (standalone HTTP server, no Qt required)
# ---------------------------------------------------------------------------

_MONITOR_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PictureStudio Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#22272E;color:#CDD9E5;padding:20px}
h1{color:#5DADE2;margin-bottom:16px;font-size:20px}
h2{color:#7FC3F5;font-size:13px;margin-bottom:8px;letter-spacing:.05em;text-transform:uppercase}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.card{background:#2D333B;border-radius:8px;padding:14px;border:1px solid #373E47}
.card .val{font-size:28px;font-weight:bold;color:#58D68D;margin:4px 0}
.card .val.alarm{color:#E74C3C}
.card .lbl{font-size:11px;color:#768390}
#chart-wrap{background:#2D333B;border-radius:8px;padding:14px;border:1px solid #373E47;margin-bottom:20px}
canvas{width:100%;height:160px;display:block}
#alarm-wrap{background:#2D333B;border-radius:8px;padding:14px;border:2px solid #E74C3C;margin-bottom:20px;display:none}
#alarm-wrap h2{color:#E74C3C}
#alarm-img{max-width:100%;border-radius:6px;margin-top:8px}
#alarm-meta{font-size:11px;color:#768390;margin-top:4px}
#dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#58D68D;margin-right:6px}
#dot.err{background:#E74C3C}
.footer{font-size:10px;color:#545D68;margin-top:14px;text-align:center}
</style>
</head>
<body>
<h1><span id="dot"></span>PictureStudio Monitor</h1>
<div class="grid">
  <div class="card"><div class="lbl">Letzter Score</div><div class="val" id="score">–</div></div>
  <div class="card"><div class="lbl">Schwellwert</div><div class="val" id="thr">–</div></div>
  <div class="card"><div class="lbl">Alarme</div><div class="val alarm" id="alarms">–</div></div>
  <div class="card"><div class="lbl">Laufzeit</div><div class="val" id="uptime">–</div></div>
  <div class="card"><div class="lbl">Kamera</div><div class="val" id="cam" style="font-size:14px">–</div></div>
</div>
<div id="chart-wrap"><h2>Score-Verlauf (letzte 120 Frames)</h2><canvas id="chart"></canvas></div>
<div id="alarm-wrap"><h2>⚠ Letzter Alarm-Frame</h2>
  <img id="alarm-img" src="" alt="">
  <div id="alarm-meta"></div>
</div>
<div class="footer">Aktualisiert alle 3 s &middot; PictureStudio Monitor-Client</div>
<script>
const BASE=window.location.origin;
const KEY="__API_KEY__";
const H=KEY?{'X-Api-Key':KEY}:{};
function api(u){return fetch(BASE+u,{headers:H}).then(r=>r.json())}
let lastFrame='';
function fmt(s){const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;
  return h?`${h}h ${m}m`:`${m}m ${sec}s`}
function draw(canvas,scores,thr){
  const ctx=canvas.getContext('2d');
  const W=canvas.width=canvas.offsetWidth,H=canvas.height=canvas.offsetHeight||160;
  ctx.clearRect(0,0,W,H);
  if(!scores.length)return;
  const vals=scores.map(s=>s.score);
  const mx=Math.max(...vals,thr*1.5,0.001);
  const dx=W/Math.max(vals.length-1,1);
  const ty=H-(thr/mx)*H;
  ctx.strokeStyle='#E67E22';ctx.setLineDash([6,4]);ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(0,ty);ctx.lineTo(W,ty);ctx.stroke();ctx.setLineDash([]);
  ctx.lineWidth=2;
  vals.forEach((v,i)=>{
    const x=i*dx,y=H-(v/mx)*H;
    ctx.strokeStyle=v>thr?'#E74C3C':'#2ECC71';
    if(i===0){ctx.beginPath();ctx.moveTo(x,y);}
    else{ctx.lineTo(x,y);ctx.stroke();ctx.beginPath();ctx.moveTo(x,y);}
  });
}
async function tick(){
  try{
    const[st,sc,al]=await Promise.all([
      api('/api/status'),api('/api/scores?limit=120'),api('/api/latest_alarm')
    ]);
    document.getElementById('dot').className='';
    document.getElementById('score').textContent=st.score.toFixed(5);
    document.getElementById('score').className='val'+(st.is_alarm?' alarm':'');
    document.getElementById('thr').textContent=st.threshold.toFixed(5);
    document.getElementById('alarms').textContent=st.event_count;
    document.getElementById('uptime').textContent=fmt(st.uptime_s);
    document.getElementById('cam').textContent=st.cam_status;
    draw(document.getElementById('chart'),sc.scores||[],st.threshold);
    const aw=document.getElementById('alarm-wrap');
    if(al.frame_filename){
      aw.style.display='block';
      if(al.frame_filename!==lastFrame){lastFrame=al.frame_filename;
        document.getElementById('alarm-img').src=BASE+'/api/frame/'+al.frame_filename+'?t='+Date.now();}
      document.getElementById('alarm-meta').textContent=
        al.ts+'  |  Score: '+(al.score||0).toFixed(5)+'  |  Thr: '+(al.threshold||0).toFixed(5);
    }else{aw.style.display='none';}
  }catch(e){document.getElementById('dot').className='err';}
}
tick();setInterval(tick,3000);
</script>
</body>
</html>"""


class _MonitorHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for the standalone monitor REST API."""

    state: _MonitorState = None  # type: ignore[assignment]

    def log_message(self, fmt, *args) -> None:
        pass

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        key = _MonitorHandler.state.api_key
        if not key:
            return True
        provided = (
            self.headers.get("X-Api-Key", "")
            or self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        return provided == key

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Api-Key")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)
        st = _MonitorHandler.state

        # /dashboard and /api/status are public
        if path not in ("/dashboard", "/api/status") and not self._check_auth():
            self._send_json({"error": "Unauthorized — provide X-Api-Key header"}, 401)
            return

        if path == "/api/status":
            self._send_json(st.snapshot())
            return

        if path == "/api/scores":
            limit = int(qs.get("limit", ["120"])[0])
            with st._lock:
                buf = st.score_buffer[-limit:]
            self._send_json({"scores": buf, "count": len(buf)})
            return

        if path == "/api/latest_alarm":
            with st._lock:
                alarm = dict(st.latest_alarm)
            self._send_json(alarm)
            return

        if path.startswith("/api/frame/"):
            fname = path[len("/api/frame/"):]
            if not fname or "/" in fname or ".." in fname:
                self._send_json({"error": "Invalid filename"}, 400)
                return
            fpath = os.path.join(st.output_dir, fname)
            if not os.path.isfile(fpath):
                self._send_json({"error": "Not found"}, 404)
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
                self._send_json({"error": str(exc)}, 500)
            return

        if path in ("/dashboard", "/dashboard/"):
            key = st.api_key
            html = _MONITOR_DASHBOARD_HTML.replace("__API_KEY__", f"'{key}'" if key else "''")
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self._send_json({"error": "Not found"}, 404)


class _MonitorApiServer:
    """Runs the monitor REST API in a background daemon thread."""

    def __init__(self, port: int, state: _MonitorState) -> None:
        self._port = port
        _MonitorHandler.state = state
        self._server = HTTPServer(("", port), _MonitorHandler)
        self._server.allow_reuse_address = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="monitor-api",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()


# ---------------------------------------------------------------------------
# Helpers (unchanged from v1.0)
# ---------------------------------------------------------------------------

def find_camera_index(camera_source: Optional[str]) -> int:
    if not camera_source:
        return 0
    try:
        cameras = list_usb_cameras()
    except Exception:
        cameras = []
    for idx, label in cameras:
        if camera_source in label or label in camera_source:
            return idx
    print(f"[Warnung] Kamera '{camera_source}' nicht gefunden — verwende Index 0.")
    return 0


def apply_roi(frame: np.ndarray, roi: Optional[list]) -> np.ndarray:
    if not roi or len(roi) != 4:
        return frame
    h, w = frame.shape[:2]
    x1 = max(0, int(roi[0] * w)); y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w)); y2 = min(h, int(roi[3] * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return frame
    return frame[y1:y2, x1:x2]


def composite_overlay(full: np.ndarray, roi_overlay: np.ndarray, roi: list) -> np.ndarray:
    out = full.copy()
    h, w = full.shape[:2]
    x1 = max(0, int(roi[0] * w)); y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w)); y2 = min(h, int(roi[3] * h))
    oh, ow = y2 - y1, x2 - x1
    if oh > 0 and ow > 0:
        resized = cv2.resize(roi_overlay, (ow, oh))
        out[y1:y2, x1:x2] = resized
    return out


def draw_hud(
    frame: np.ndarray, score: float, threshold: float,
    is_anomaly: bool, event_count: int, roi: Optional[list],
    cam_status: str = "",
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if roi and len(roi) == 4:
        x1 = int(roi[0] * w); y1 = int(roi[1] * h)
        x2 = int(roi[2] * w); y2 = int(roi[3] * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    bar_w, bar_h, bar_x = 220, 14, 10
    bar_y = h - 56
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
    fill = int(min(score / max(threshold * 3, 1e-9), 1.0) * bar_w)
    bar_color = (0, 0, 210) if is_anomaly else (0, 170, 0)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    thr_x = bar_x + bar_w // 3
    cv2.line(out, (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2), (0, 200, 255), 2)

    txt_color = (0, 0, 210) if is_anomaly else (0, 200, 0)
    state = "ANOMALIE" if is_anomaly else "Normal"
    cv2.putText(out, f"Score: {score:.5f}   {state}   Alarme: {event_count}",
                (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 1, cv2.LINE_AA)
    footer = f"Schwellwert: {threshold:.5f}   Q / ESC = Beenden"
    if cam_status:
        footer += f"   [{cam_status}]"
    cv2.putText(out, footer, (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    if is_anomaly:
        cv2.rectangle(out, (0, 0), (w, 32), (0, 0, 180), -1)
        cv2.putText(out, "  ANOMALIE ERKANNT", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def save_alarm(
    frame: np.ndarray, score: float, threshold: float,
    output_dir: str, log_path: str, event_count: int,
) -> str:
    ts = datetime.now(timezone.utc)
    fname = f"alarm_{ts.strftime('%Y%m%dT%H%M%SZ')}.jpg"
    fpath = os.path.join(output_dir, fname)
    try:
        cv2.imwrite(fpath, frame)
    except Exception as exc:
        print(f"\n[Warnung] Bild konnte nicht gespeichert werden: {exc}")
        fname = ""

    write_header = not os.path.exists(log_path)
    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp_utc", "score", "threshold", "score_pct", "frame_file"])
            pct = int(score / threshold * 100) if threshold > 0 else 0
            writer.writerow([ts.isoformat(), f"{score:.6f}", f"{threshold:.6f}", pct, fname])
    except Exception as exc:
        print(f"\n[Warnung] CSV-Schreibfehler: {exc}")

    print(f"\n[ALARM #{event_count}]  {ts.strftime('%H:%M:%S')}  "
          f"Score={score:.5f}  →  {fname}")
    return fname


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def run_monitor(
    model_path: str,
    *,
    camera_source: Union[int, str, None] = None,   # int=USB index, str=URL/file, None=auto
    threshold_override: Optional[float] = None,
    output_dir: str = "monitor_logs",
    fps: float = 15.0,
    cooldown: float = 10.0,
    headless: bool = False,
    no_notify: bool = False,
    reconnect_delay: float = 5.0,
    mqtt_host: str = "",
    mqtt_port: int = 1883,
    mqtt_topic: str = "picture_studio/monitor",
    mqtt_user: str = "",
    mqtt_pass: str = "",
    api_port: int = 0,
    api_key: str = "",
) -> int:
    """Run the monitor loop. Returns the number of alarm events logged (≥0), or -1 on error."""

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Lade Modell: {model_path}")
    if model_path.lower().endswith(".onnx"):
        if not HAS_ORT:
            print("Fehler: onnxruntime nicht installiert.  pip install onnxruntime")
            return -1
        det = OnnxAnomalyScorer.from_path(model_path)
        print("  ONNX-Modell geladen (kein PyTorch nötig)")
    else:
        det = AnomalyDetector()
        try:
            det.load(model_path)
        except Exception as exc:
            print(f"Fehler beim Laden des Modells: {exc}")
            return -1

    meta = det.metadata
    roi: Optional[list] = meta.get("roi")
    threshold: float    = det.threshold

    if threshold_override is not None:
        det.threshold = threshold_override
        threshold = det.threshold

    print(f"  Beschreibung : {meta.get('description', '-')}")
    print(f"  Trainiert    : {meta.get('trained_at', '-')}")
    print(f"  ROI          : {'aktiv ' + str(roi) if roi else 'nein'}")
    print(f"  Schwellwert  : {threshold:.6f}")

    # ── Determine camera source ─────────────────────────────────────────────────
    if camera_source is None:
        # Auto-detect from model metadata
        meta_cam = meta.get("camera_source", "")
        cam_source: Union[int, str] = find_camera_index(meta_cam)
        print(f"  Kamera       : Index {cam_source} (aus Modell-Metadaten)")
    elif isinstance(camera_source, int):
        cam_source = camera_source
        print(f"  Kamera       : Index {cam_source} (manuell)")
    else:
        cam_source = camera_source
        ext = os.path.splitext(str(cam_source))[1].lower()
        kind = "Videodatei" if ext in _VIDEO_EXTENSIONS else "URL/Stream"
        print(f"  Kamera       : {kind} — {cam_source}")

    # ── Alarm notifier (E-Mail / Webhook) ──────────────────────────────────────
    if no_notify:
        notifier = None
        print("  Benachrichtigung: deaktiviert (--no-notify)")
    else:
        try:
            from core.alarm_notifier import AlarmNotifier
            from utils.settings import AppSettings
            from PySide6.QtCore import QCoreApplication
            _app = QCoreApplication.instance() or QCoreApplication([])
            notifier = AlarmNotifier(AppSettings().get_alarm_notifier_config())
        except Exception:
            try:
                from core.alarm_notifier import AlarmNotifier
                notifier = AlarmNotifier()
            except Exception:
                notifier = None

    # ── MQTT client ────────────────────────────────────────────────────────────
    mqtt_client = None
    if mqtt_host:
        if not HAS_MQTT:
            print("[Warnung] paho-mqtt nicht installiert — MQTT deaktiviert.  pip install paho-mqtt")
        elif MQTTAlarmClient is not None:
            mqtt_client = MQTTAlarmClient(
                host=mqtt_host, port=mqtt_port, topic=mqtt_topic,
                username=mqtt_user, password=mqtt_pass,
            )
            if mqtt_client.connect():
                time.sleep(0.5)   # give connection time to establish
                status = "verbunden" if mqtt_client.connected else "Verbindung ausstehend"
                print(f"  MQTT         : {mqtt_host}:{mqtt_port}/{mqtt_topic} ({status})")
            else:
                print(f"[Warnung] MQTT-Verbindung fehlgeschlagen: {mqtt_client.last_error}")
                mqtt_client = None

    # ── Shared state ───────────────────────────────────────────────────────────
    state = _MonitorState()
    state.model_name = os.path.basename(model_path)
    state.threshold  = threshold
    state.output_dir = output_dir
    state.api_key    = api_key
    state.cam_status = "Verbinde…"

    # ── Mini REST API ──────────────────────────────────────────────────────────
    api_server = None
    if api_port > 0:
        try:
            api_server = _MonitorApiServer(api_port, state)
            api_server.start()
            print(f"  REST-API     : http://localhost:{api_port}/api/status")
            print(f"  Dashboard    : http://localhost:{api_port}/dashboard")
            if api_key:
                print(f"  API-Key      : {api_key}")
        except OSError as exc:
            print(f"[Warnung] REST-API konnte nicht gestartet werden: {exc}")
            api_server = None

    # ── Output dir + log ───────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "monitor_events.csv")

    # ── Thread-safe frame state ────────────────────────────────────────────────
    latest_frame:   Optional[np.ndarray] = None
    latest_display: Optional[np.ndarray] = None
    score_val:   float = 0.0
    is_anom:     bool  = False
    last_alarm_t: float = -cooldown
    frame_lock = threading.Lock()

    # ── Frame callback (runs in camera thread) ─────────────────────────────────
    def on_frame(frame: np.ndarray) -> None:
        nonlocal latest_frame, latest_display, score_val, is_anom, last_alarm_t

        cropped = apply_roi(frame, roi)
        try:
            score, _rec, overlay_crop, _bbox = det.score_detailed(cropped)
        except Exception:
            return

        is_anomaly = score > threshold

        if roi:
            display = composite_overlay(frame, overlay_crop, roi)
        else:
            display = overlay_crop

        with frame_lock:
            score_val      = score
            is_anom        = is_anomaly
            latest_frame   = frame.copy()
            latest_display = display

        # Update shared state for REST API
        state.push_score(score, threshold)

        if headless:
            s = "ANOMALIE" if is_anomaly else "Normal  "
            print(f"\r[{datetime.now(timezone.utc).strftime('%H:%M:%S')}]  "
                  f"Score: {score:.5f}  Thr: {threshold:.5f}  {s}  "
                  f"Alarme: {state.event_count}   ", end="", flush=True)

        if is_anomaly:
            now = time.perf_counter()
            if now - last_alarm_t >= cooldown:
                last_alarm_t = now
                fname = save_alarm(frame, score, threshold, output_dir, log_path,
                                   state.event_count + 1)
                state.push_alarm(score, threshold, fname)

                fpath = os.path.join(output_dir, fname) if fname else ""
                model_name = os.path.basename(model_path)

                if notifier:
                    try:
                        notifier.notify(score, threshold, frame_path=fpath, model_name=model_name)
                    except Exception as exc:
                        print(f"\n[Warnung] Benachrichtigung fehlgeschlagen: {exc}")

                if mqtt_client:
                    try:
                        mqtt_client.publish_alarm(score, threshold, frame_path=fpath)
                    except Exception as exc:
                        print(f"\n[Warnung] MQTT fehlgeschlagen: {exc}")

    def on_cam_status(msg: str) -> None:
        state.cam_status = msg
        if headless:
            print(f"\n[Kamera] {msg}")

    # ── Camera thread ──────────────────────────────────────────────────────────
    cam = _CameraThread(
        source=cam_source,
        fps=fps,
        callback=on_frame,
        reconnect_delay=reconnect_delay,
        on_status=on_cam_status,
    )
    cam.start()
    time.sleep(0.6)

    if cam.error and not cam._is_video:
        # For live streams, the thread keeps retrying — only fatal for video files
        if isinstance(cam_source, str) and not cam._is_video:
            pass   # reconnect loop running
        else:
            print(f"Kamera-Fehler: {cam.error}")
            if api_server:
                api_server.stop()
            return -1

    print(f"\nMonitor läuft. Ausgabe → {output_dir}")
    if not headless:
        print("Zum Beenden: Q oder ESC drücken.\n")
    else:
        print("Headless-Modus. Zum Beenden: Strg+C\n")

    # ── Display / event loop ───────────────────────────────────────────────────
    try:
        if headless:
            while True:
                time.sleep(0.1)
        else:
            win = f"PictureStudio Monitor — {os.path.basename(model_path)}"
            cv2.namedWindow(win, cv2.WINDOW_RESIZABLE)
            while True:
                with frame_lock:
                    disp  = latest_display
                    score = score_val
                    ia    = is_anom
                    ec    = state.event_count
                    cs    = state.cam_status

                if disp is not None:
                    hud = draw_hud(disp, score, threshold, ia, ec, roi, cs)
                    cv2.imshow(win, hud)

                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break

            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cam.join(timeout=3.0)
        if mqtt_client:
            mqtt_client.disconnect()
        if api_server:
            api_server.stop()
        if headless:
            print()

    print(f"\nMonitor beendet. {state.event_count} Alarm-Event(s) protokolliert.")
    if state.event_count > 0:
        print(f"Log-Datei: {log_path}")

    return state.event_count


# ---------------------------------------------------------------------------
# Setup-Wizard — state, HTML, HTTP handler, server, run_setup()
# ---------------------------------------------------------------------------

_SETUP_HTML: str = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PictureStudio — Einrichtung</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#22272E;color:#CDD9E5;padding:24px;min-height:100vh}
h1{color:#5DADE2;margin-bottom:20px;font-size:22px;letter-spacing:.02em}
/* Wizard steps */
.steps{display:flex;gap:0;margin-bottom:28px;user-select:none}
.step{flex:1;padding:10px 6px;text-align:center;font-size:12px;color:#545D68;background:#2D333B;
      border:1px solid #373E47;cursor:default;transition:color .2s,background .2s}
.step:first-child{border-radius:6px 0 0 6px}
.step:last-child{border-radius:0 6px 6px 0}
.step.active{color:#CDD9E5;background:#1C6EA4;border-color:#2185C5}
.step.done{color:#58D68D;background:#1E3A2F;border-color:#2ECC71}
.step-num{display:block;font-size:18px;font-weight:bold;margin-bottom:2px}
/* Cards */
.card{background:#2D333B;border-radius:8px;padding:20px;border:1px solid #373E47;margin-bottom:16px}
.panel{display:none}.panel.active{display:block}
/* Preview */
#preview-img,#preview-img2{max-width:100%;border-radius:6px;display:block;background:#1B2028;
  min-height:200px;width:100%;object-fit:contain}
.preview-placeholder{color:#545D68;font-size:13px;text-align:center;padding:60px 0}
/* Controls */
label{display:block;margin-bottom:4px;font-size:12px;color:#768390}
input[type=range]{width:100%;accent-color:#5DADE2;margin:6px 0 2px}
input[type=text],input[type=number]{background:#1B2028;border:1px solid #373E47;color:#CDD9E5;
  border-radius:4px;padding:6px 10px;width:100%;font-size:13px}
.range-val{font-size:11px;color:#5DADE2;font-weight:bold}
/* Progress bar */
.progress-wrap{background:#1B2028;border-radius:4px;height:18px;overflow:hidden;margin:10px 0}
.progress-bar{height:100%;background:#1C6EA4;transition:width .4s;width:0%}
.progress-label{font-size:11px;color:#768390;margin-top:2px}
/* Buttons */
.btn{display:inline-block;padding:9px 20px;border-radius:5px;border:none;cursor:pointer;
     font-size:13px;font-weight:600;transition:background .15s,opacity .15s}
.btn-primary{background:#1C6EA4;color:#fff}
.btn-primary:hover{background:#2185C5}
.btn-primary:disabled{background:#2D333B;color:#545D68;cursor:not-allowed}
.btn-success{background:#1A8754;color:#fff}
.btn-success:hover{background:#1DB05E}
.btn-success:disabled{background:#2D333B;color:#545D68;cursor:not-allowed}
.btn-danger{background:#8B2222;color:#fff}
.btn-danger:hover{background:#A52626}
.btn-sm{padding:5px 12px;font-size:12px}
.btn-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:14px}
/* Status badge */
.badge{display:inline-block;padding:3px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge-idle{background:#373E47;color:#768390}
.badge-capturing{background:#1C6EA4;color:#fff}
.badge-training{background:#7D4E00;color:#FFC107}
.badge-training_done{background:#1A8754;color:#58D68D}
.badge-done{background:#1A8754;color:#58D68D}
.badge-error{background:#8B2222;color:#ff6b6b}
/* Info row */
.info-row{display:flex;gap:16px;font-size:12px;color:#768390;margin-top:6px;flex-wrap:wrap}
.info-row span b{color:#CDD9E5}
/* Threshold display */
.thr-display{font-size:32px;font-weight:bold;color:#58D68D;margin:8px 0}
/* Success screen */
.success-box{text-align:center;padding:40px 20px}
.success-box .icon{font-size:56px;margin-bottom:12px}
.success-box h2{color:#58D68D;font-size:20px;margin-bottom:8px}
.success-box p{color:#768390;font-size:13px}
/* Section heading */
h2{color:#7FC3F5;font-size:14px;margin-bottom:12px;letter-spacing:.05em;text-transform:uppercase}
.sep{border:none;border-top:1px solid #373E47;margin:16px 0}
</style>
</head>
<body>
<h1>PictureStudio — Einrichtungsmodus</h1>

<!-- Step indicator -->
<div class="steps" id="steps">
  <div class="step active" id="step-ind-1"><span class="step-num">①</span>Kamera-Preview</div>
  <div class="step" id="step-ind-2"><span class="step-num">②</span>Aufnahme</div>
  <div class="step" id="step-ind-3"><span class="step-num">③</span>Training</div>
  <div class="step" id="step-ind-4"><span class="step-num">④</span>Live-Start</div>
</div>

<!-- Panel 1: Camera Preview -->
<div class="panel active" id="panel-1">
  <div class="card">
    <h2>Kamera-Vorschau</h2>
    <img id="preview-img" src="" alt="Kamera wird geladen…" onerror="this.src=''">
    <div id="preview-placeholder" class="preview-placeholder">Kamera wird geladen…</div>
    <div class="info-row" style="margin-top:10px">
      <span>Status: <b id="cam-status-1">–</b></span>
    </div>
  </div>
  <div class="btn-row">
    <button class="btn btn-primary" onclick="gotoStep(2)">Weiter → Aufnahme</button>
  </div>
</div>

<!-- Panel 2: Capture -->
<div class="panel" id="panel-2">
  <div class="card">
    <h2>Normalframes aufnehmen</h2>
    <img id="preview-img2" src="" alt="">
    <hr class="sep">
    <label>Ziel-Frames: <span class="range-val" id="target-val">150</span></label>
    <input type="range" id="target-slider" min="50" max="500" value="150"
           oninput="document.getElementById('target-val').textContent=this.value">
    <div class="progress-wrap" style="margin-top:12px">
      <div class="progress-bar" id="cap-bar"></div>
    </div>
    <div class="progress-label" id="cap-label">0 / 150 Frames</div>
    <div class="info-row">
      <span>Phase: <b><span id="cap-phase-badge" class="badge badge-idle">idle</span></b></span>
      <span>Gesammelt: <b id="cap-count">0</b></span>
    </div>
    <div class="btn-row" style="margin-top:14px">
      <button class="btn btn-primary" id="btn-start-cap" onclick="startCapture()">Aufnehmen starten</button>
      <button class="btn btn-danger btn-sm" id="btn-stop-cap" style="display:none" onclick="stopCapture()">Aufnahme stoppen</button>
    </div>
  </div>
  <div class="btn-row">
    <button class="btn btn-success" id="btn-goto-train" disabled onclick="gotoStep(3)">→ Training</button>
    <span style="font-size:11px;color:#545D68">(mindestens 30 Frames nötig)</span>
  </div>
</div>

<!-- Panel 3: Training -->
<div class="panel" id="panel-3">
  <div class="card">
    <h2>Modell trainieren</h2>
    <label>Epochen (10–100)</label>
    <input type="number" id="train-epochs" value="40" min="10" max="100">
    <label style="margin-top:10px">Modell-Name</label>
    <input type="text" id="train-name" value="monitor_model">
    <hr class="sep">
    <div class="progress-wrap">
      <div class="progress-bar" id="train-bar" style="background:#7D4E00"></div>
    </div>
    <div class="progress-label" id="train-label">Bereit</div>
    <div class="info-row">
      <span>Phase: <b><span id="train-phase-badge" class="badge badge-idle">idle</span></b></span>
      <span id="train-loss-info" style="display:none">Epoch <b id="train-ep">–</b>/<b id="train-total">–</b> — Loss: <b id="train-loss">–</b></span>
    </div>
    <div class="btn-row" style="margin-top:14px">
      <button class="btn btn-primary" id="btn-start-train" onclick="startTraining()">Training starten</button>
    </div>
  </div>
</div>

<!-- Panel 4: Live Start -->
<div class="panel" id="panel-4">
  <div class="card" id="panel4-config">
    <h2>Live-Monitoring starten</h2>
    <p style="font-size:13px;color:#768390;margin-bottom:12px">Berechneter Anomalie-Schwellwert:</p>
    <div class="thr-display" id="thr-display">–</div>
    <label>Schwellwert anpassen (±50%)</label>
    <input type="range" id="thr-slider" min="0" max="100" value="50" step="1"
           oninput="onThrSlider(this.value)">
    <div style="font-size:11px;color:#768390;margin-top:2px">
      Min: <span id="thr-min">–</span> &nbsp; Mitte: <span id="thr-mid">–</span> &nbsp; Max: <span id="thr-max">–</span>
    </div>
    <div class="btn-row" style="margin-top:18px">
      <button class="btn btn-success" id="btn-go-live" onclick="goLive()">Live-Monitoring starten</button>
    </div>
  </div>
  <div class="success-box" id="panel4-success" style="display:none">
    <div class="icon">✓</div>
    <h2>Einrichtung abgeschlossen!</h2>
    <p>Live-Monitoring wurde gestartet.<br>Dieses Fenster kann geschlossen werden.</p>
  </div>
</div>

<script>
// ─── State ───────────────────────────────────────────────────────────────────
let currentStep = 1;
let state = {};
let baseThr = 0;        // threshold from training
let adjustedThr = 0;    // currently selected threshold
let frameTimerRunning = false;

// ─── Navigation ──────────────────────────────────────────────────────────────
function gotoStep(n) {
  for (let i = 1; i <= 4; i++) {
    document.getElementById('panel-' + i).classList.toggle('active', i === n);
    const ind = document.getElementById('step-ind-' + i);
    ind.classList.remove('active', 'done');
    if (i < n) ind.classList.add('done');
    else if (i === n) ind.classList.add('active');
  }
  currentStep = n;
}

// ─── Frame preview polling ────────────────────────────────────────────────────
function startFramePolling() {
  if (frameTimerRunning) return;
  frameTimerRunning = true;
  pollFrame();
}

function pollFrame() {
  const url = '/setup/frame.jpg?t=' + Date.now();
  const imgs = [document.getElementById('preview-img'), document.getElementById('preview-img2')];
  imgs.forEach(img => {
    if (img) {
      img.onload = () => { document.getElementById('preview-placeholder').style.display = 'none'; };
      img.src = url;
    }
  });
  setTimeout(pollFrame, 200);
}

// ─── Status polling ───────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/setup/status');
    const s = await r.json();
    state = s;
    updateUI(s);
  } catch (e) { /* ignore */ }
  setTimeout(poll, 1000);
}

function updateUI(s) {
  // Panel 2 updates
  const target = parseInt(document.getElementById('target-slider').value) || s.target_frames || 150;
  const count = s.frame_count || 0;
  const pct = Math.min(count / Math.max(target, 1) * 100, 100);
  document.getElementById('cap-bar').style.width = pct + '%';
  document.getElementById('cap-label').textContent = count + ' / ' + target + ' Frames';
  document.getElementById('cap-count').textContent = count;
  setBadge('cap-phase-badge', s.phase);
  document.getElementById('btn-goto-train').disabled = count < 30;
  const capturing = s.phase === 'capturing';
  document.getElementById('btn-start-cap').style.display = capturing ? 'none' : 'inline-block';
  document.getElementById('btn-stop-cap').style.display = capturing ? 'inline-block' : 'none';

  // Panel 3 updates
  setBadge('train-phase-badge', s.phase);
  if (s.phase === 'training' || s.phase === 'training_done') {
    const ep = s.epoch || 0;
    const tot = s.total_epochs || 40;
    const pct3 = Math.min(ep / Math.max(tot, 1) * 100, 100);
    document.getElementById('train-bar').style.width = pct3 + '%';
    document.getElementById('train-label').textContent =
      'Epoch ' + ep + '/' + tot + ' — Loss: ' + (s.loss || 0).toFixed(6);
    document.getElementById('train-loss-info').style.display = 'inline';
    document.getElementById('train-ep').textContent = ep;
    document.getElementById('train-total').textContent = tot;
    document.getElementById('train-loss').textContent = (s.loss || 0).toFixed(6);
    document.getElementById('btn-start-train').disabled = true;
  }
  if (s.phase === 'training_done' && currentStep === 3) {
    // auto-advance to step 4
    baseThr = s.threshold || 0;
    adjustedThr = baseThr;
    initThrSlider(baseThr);
    gotoStep(4);
  }

  // Panel 4 updates
  if (s.threshold && currentStep === 4) {
    document.getElementById('thr-display').textContent = s.threshold.toFixed(6);
  }

  // Panel 1 camera status
  if (s.phase) {
    document.getElementById('cam-status-1').textContent = s.phase;
  }
}

function setBadge(id, phase) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = phase || 'idle';
  el.className = 'badge badge-' + (phase || 'idle').replace(/_/g, '-');
}

// ─── Threshold slider ─────────────────────────────────────────────────────────
function initThrSlider(thr) {
  baseThr = thr;
  adjustedThr = thr;
  document.getElementById('thr-display').textContent = thr.toFixed(6);
  document.getElementById('thr-min').textContent = (thr * 0.5).toFixed(6);
  document.getElementById('thr-mid').textContent = thr.toFixed(6);
  document.getElementById('thr-max').textContent = (thr * 1.5).toFixed(6);
  document.getElementById('thr-slider').value = 50;
}

function onThrSlider(val) {
  // val 0..100 maps to 0.5*base..1.5*base
  const factor = 0.5 + (val / 100.0);
  adjustedThr = baseThr * factor;
  document.getElementById('thr-display').textContent = adjustedThr.toFixed(6);
  // debounced POST
  clearTimeout(window._thrTimer);
  window._thrTimer = setTimeout(() => {
    fetch('/setup/threshold', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({threshold: adjustedThr})
    });
  }, 400);
}

// ─── API calls ────────────────────────────────────────────────────────────────
async function startCapture() {
  const target = parseInt(document.getElementById('target-slider').value) || 150;
  await fetch('/setup/capture/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({target: target})
  });
}

async function stopCapture() {
  await fetch('/setup/capture/stop', {method: 'POST'});
}

async function startTraining() {
  const epochs = parseInt(document.getElementById('train-epochs').value) || 40;
  const name = document.getElementById('train-name').value.trim() || 'monitor_model';
  document.getElementById('btn-start-train').disabled = true;
  document.getElementById('train-label').textContent = 'Training wird gestartet…';
  await fetch('/setup/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({epochs: epochs, model_name: name, threshold: null})
  });
}

async function goLive() {
  document.getElementById('btn-go-live').disabled = true;
  await fetch('/setup/go_live', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({camera: 0})
  });
  document.getElementById('panel4-config').style.display = 'none';
  document.getElementById('panel4-success').style.display = 'block';
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
startFramePolling();
poll();
</script>
</body>
</html>"""


class _SetupState:
    """Thread-safe state container for the Setup-Wizard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase: str = "idle"
        self.frame_count: int = 0
        self.target_frames: int = 150
        self.epoch: int = 0
        self.total_epochs: int = 40
        self.loss: float = 0.0
        self.threshold: float = 0.0
        self.model_path: str = ""
        self.error: str = ""
        self.last_frame: Optional[np.ndarray] = None

    def snapshot(self) -> dict:
        """Return all scalar fields as a dict (without the raw frame)."""
        with self._lock:
            return {
                "phase":         self.phase,
                "frame_count":   self.frame_count,
                "target_frames": self.target_frames,
                "epoch":         self.epoch,
                "total_epochs":  self.total_epochs,
                "loss":          self.loss,
                "threshold":     self.threshold,
                "model_path":    self.model_path,
                "error":         self.error,
            }


class _SetupHandler(BaseHTTPRequestHandler):
    """HTTP handler for the Setup-Wizard web UI."""

    state: "_SetupState" = None      # type: ignore[assignment]
    detector: "AnomalyDetector" = None  # type: ignore[assignment]
    output_dir: str = "monitor_logs"

    def log_message(self, fmt, *args) -> None:  # silence request logs
        pass

    # ── helpers ────────────────────────────────────────────────────────────────

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    # ── OPTIONS ────────────────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── GET ────────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/setup")
            self._cors()
            self.end_headers()
            return

        if path == "/setup":
            self._send_html(_SETUP_HTML)
            return

        if path == "/setup/frame.jpg":
            st = _SetupHandler.state
            with st._lock:
                frame = st.last_frame
            if frame is None:
                # No frame yet — return 204 No Content
                self.send_response(204)
                self._cors()
                self.end_headers()
                return
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                self.send_response(204)
                self._cors()
                self.end_headers()
                return
            data = buf.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/setup/status":
            self._send_json(_SetupHandler.state.snapshot())
            return

        self._send_json({"error": "Not found"}, 404)

    # ── POST ───────────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        st = _SetupHandler.state
        det = _SetupHandler.detector

        if path == "/setup/capture/start":
            body = self._read_json_body()
            with st._lock:
                st.target_frames = int(body.get("target", 150))
                st.phase = "capturing"
            self._send_json({"ok": True})
            return

        if path == "/setup/capture/stop":
            with st._lock:
                st.phase = "idle"
            self._send_json({"ok": True})
            return

        if path == "/setup/train":
            body = self._read_json_body()
            epochs = int(body.get("epochs", 40))
            model_name = str(body.get("model_name", "monitor_model")).strip() or "monitor_model"
            thr_override = body.get("threshold")

            with st._lock:
                st.phase = "training"
                st.total_epochs = epochs
                st.epoch = 0

            def _train_thread() -> None:
                try:
                    def _progress(epoch: int, total: int, loss: float) -> None:
                        with st._lock:
                            st.epoch = epoch
                            st.total_epochs = total
                            st.loss = loss

                    computed_thr = det.train(
                        epochs=epochs,
                        progress_cb=_progress,
                    )
                    # Final threshold
                    final_thr = float(thr_override) if thr_override is not None else computed_thr
                    # Save model
                    os.makedirs(_SetupHandler.output_dir, exist_ok=True)
                    model_path = os.path.join(_SetupHandler.output_dir, model_name + ".pth")
                    det.save(model_path)
                    with st._lock:
                        st.threshold = final_thr
                        st.model_path = model_path
                        st.phase = "training_done"
                except Exception as exc:
                    with st._lock:
                        st.error = str(exc)
                        st.phase = "error"

            threading.Thread(target=_train_thread, daemon=True, name="setup-train").start()
            self._send_json({"ok": True})
            return

        if path == "/setup/threshold":
            body = self._read_json_body()
            thr = body.get("threshold")
            if thr is not None:
                with st._lock:
                    st.threshold = float(thr)
                # Also propagate to the detector so run_monitor gets the right value
                det._threshold = float(thr)
            self._send_json({"ok": True, "threshold": st.threshold})
            return

        if path == "/setup/go_live":
            with st._lock:
                st.phase = "done"
            self._send_json({"ok": True})
            return

        self._send_json({"error": "Not found"}, 404)


class _SetupApiServer:
    """Runs the Setup-Wizard HTTP server in a background daemon thread."""

    def __init__(
        self,
        port: int,
        state: _SetupState,
        detector: "AnomalyDetector",
        output_dir: str = "monitor_logs",
    ) -> None:
        self._port = port
        _SetupHandler.state = state
        _SetupHandler.detector = detector
        _SetupHandler.output_dir = output_dir
        self._server = HTTPServer(("", port), _SetupHandler)
        self._server.allow_reuse_address = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="setup-api",
            daemon=True,
        )

    def start(self) -> None:
        """Start the HTTP server thread."""
        self._thread.start()

    def stop(self) -> None:
        """Shut down the HTTP server."""
        self._server.shutdown()


def run_setup(
    camera_source: Union[int, str, None],
    setup_port: int = 8765,
    output_dir: str = "monitor_logs",
) -> dict:
    """
    Start the Setup-Wizard and block until the user clicks 'Live-Monitoring starten'.

    Returns a dict: {"model_path": str, "threshold": float, "camera_source": ...}
    Raises RuntimeError if the user's training fails.
    """
    det = AnomalyDetector()
    state = _SetupState()

    # Resolve camera source
    if camera_source is None:
        cam_source: Union[int, str] = 0
    else:
        cam_source = camera_source

    # Frame callback — stores latest frame; collects frames when capturing
    def _on_frame(frame: np.ndarray) -> None:
        with state._lock:
            state.last_frame = frame.copy()
            if state.phase == "capturing":
                target = state.target_frames
                count = state.frame_count
        # collect outside the lock to avoid holding it during heavy work
        if state.phase == "capturing" and count < target:
            det.collect_frame(frame)
            with state._lock:
                state.frame_count = det.n_collected()

    cam = _CameraThread(
        source=cam_source,
        fps=15.0,
        callback=_on_frame,
        reconnect_delay=5.0,
    )
    cam.start()

    server = _SetupApiServer(
        port=setup_port,
        state=state,
        detector=det,
        output_dir=output_dir,
    )
    server.start()

    print(f"\nSetup-Wizard läuft.")
    print(f"Öffne im Browser: http://localhost:{setup_port}/setup\n")

    # Block until done or error
    while state.phase not in ("done", "error"):
        time.sleep(0.5)

    cam.stop()
    cam.join(timeout=3.0)
    server.stop()

    if state.phase == "error":
        raise RuntimeError(f"Setup-Fehler: {state.error}")

    return {
        "model_path":    state.model_path,
        "threshold":     state.threshold,
        "camera_source": cam_source,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description="PictureStudio Monitor-Client — Standalone Anomalie-Erkennung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # USB-Kamera (Index aus Modell-Metadaten)
  python monitor.py --model models/anomalie.pth

  # IP-Kamera / RTSP-Stream
  python monitor.py --model models/anomalie.pth \\
      --url rtsp://admin:pass@192.168.1.100:554/stream

  # Video-Datei analysieren (einmalig)
  python monitor.py --model models/anomalie.pth --url /pfad/video.mp4 --headless

  # Headless + MQTT + REST-Dashboard
  python monitor.py --model models/anomalie.pth \\
      --url rtsp://kamera.local/stream \\
      --headless --mqtt-host 192.168.1.50 --api-port 8766

  # ONNX-Modell auf Raspberry Pi (kein PyTorch nötig)
  python monitor.py --model models/anomalie.onnx \\
      --url rtsp://kamera.local/stream \\
      --headless --output /var/log/anomalien

  # Systemd-Dienst: kein Reconnect-Timeout (endlos)
  python monitor.py --model anomalie.pth --headless --reconnect-delay 10
        """,
    )

    # Setup-Wizard
    setup_grp = parser.add_argument_group("Einrichtungsmodus (Setup-Wizard)")
    setup_grp.add_argument("--setup", action="store_true",
                           help="Setup-Wizard starten (kein --model nötig)")
    setup_grp.add_argument("--setup-port", type=int, default=8765, metavar="PORT",
                           help="HTTP-Port für den Setup-Wizard (Standard: 8765)")

    # Modell
    parser.add_argument("--model", nargs="?", default=None, metavar="PFAD",
                        help="Pfad zur trainierten .pth- oder .onnx-Modelldatei (Pflicht außer mit --setup)")

    # Kamera
    cam_grp = parser.add_argument_group("Kameraquelle (Standard: aus Modell-Metadaten)")
    src = cam_grp.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, metavar="INDEX",
                     help="USB-Kamera-Index überschreiben (z. B. 0, 1, 2)")
    src.add_argument("--url", metavar="URL",
                     help="IP-Kamera-URL (rtsp://…, http://…) oder Video-Datei-Pfad (mp4, avi, …)")

    # Allgemein
    parser.add_argument("--threshold", type=float, metavar="WERT",
                        help="Anomalie-Schwellwert überschreiben (Standard: aus Modell)")
    parser.add_argument("--output", default="monitor_logs", metavar="VERZ",
                        help="Ausgabeverzeichnis für Logs und Alarm-Bilder (Standard: monitor_logs)")
    parser.add_argument("--fps", type=float, default=15.0, metavar="FPS",
                        help="Bilder pro Sekunde für live Streams (Standard: 15; Videos: nativ)")
    parser.add_argument("--cooldown", type=float, default=10.0, metavar="SEK",
                        help="Mindestabstand zwischen Alarm-Saves in Sekunden (Standard: 10)")
    parser.add_argument("--headless", action="store_true",
                        help="Kein OpenCV-Fenster — nur Terminal und CSV-Log")
    parser.add_argument("--no-notify", action="store_true",
                        help="E-Mail/Webhook-Benachrichtigungen deaktivieren")
    parser.add_argument("--reconnect-delay", type=float, default=5.0, metavar="SEK",
                        help="Sekunden bis zum Reconnect-Versuch bei Verbindungsverlust (0=deaktiviert, Standard: 5)")

    # MQTT
    mqtt_grp = parser.add_argument_group("MQTT (Alarm-Publishing an Broker)")
    mqtt_grp.add_argument("--mqtt-host", default="", metavar="HOST",
                          help="MQTT-Broker-Hostname oder IP")
    mqtt_grp.add_argument("--mqtt-port", type=int, default=1883, metavar="PORT",
                          help="MQTT-Broker-Port (Standard: 1883)")
    mqtt_grp.add_argument("--mqtt-topic", default="picture_studio/monitor", metavar="TOPIC",
                          help="MQTT-Topic (Standard: picture_studio/monitor)")
    mqtt_grp.add_argument("--mqtt-user", default="", metavar="USER",
                          help="MQTT-Benutzername (optional)")
    mqtt_grp.add_argument("--mqtt-pass", default="", metavar="PASS",
                          help="MQTT-Passwort (optional)")

    # REST API
    api_grp = parser.add_argument_group("REST-API + Dashboard (eingebaut)")
    api_grp.add_argument("--api-port", type=int, default=0, metavar="PORT",
                         help="Port für eingebauten REST-Server + HTML-Dashboard (0=deaktiviert)")
    api_grp.add_argument("--api-key", default="", metavar="KEY",
                         help="API-Key für REST-Authentifizierung (optional)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.setup:
        # Setup-Modus — kein --model erforderlich
        if args.url:
            camera_source: Union[int, str, None] = args.url
        elif args.camera is not None:
            camera_source = args.camera
        else:
            camera_source = 0  # Default USB-Kamera

        result = run_setup(
            camera_source=camera_source,
            setup_port=args.setup_port,
            output_dir=args.output,
        )
        print(f"\nEinrichtung abgeschlossen. Starte Live-Monitoring...")
        run_monitor(
            model_path=result["model_path"],
            camera_source=result.get("camera_source", camera_source),
            threshold_override=result["threshold"],
            output_dir=args.output,
            fps=args.fps,
            cooldown=args.cooldown,
            headless=args.headless,
            api_port=args.api_port,
            api_key=args.api_key,
            reconnect_delay=args.reconnect_delay,
        )
        return

    # Normaler Modus — --model ist Pflicht wenn nicht --setup
    if not args.model:
        parser.error("--model ist erforderlich (oder --setup für Einrichtungsmodus verwenden)")
    if not os.path.isfile(args.model):
        parser.error(f"Modell-Datei nicht gefunden: {args.model}")

    # Resolve camera source
    if args.url:
        camera_source = args.url
    elif args.camera is not None:
        camera_source = args.camera
    else:
        camera_source = None   # auto-detect from model metadata

    result = run_monitor(
        model_path=args.model,
        camera_source=camera_source,
        threshold_override=args.threshold,
        output_dir=args.output,
        fps=args.fps,
        cooldown=args.cooldown,
        headless=args.headless,
        no_notify=args.no_notify,
        reconnect_delay=args.reconnect_delay,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        mqtt_user=args.mqtt_user,
        mqtt_pass=args.mqtt_pass,
        api_port=args.api_port,
        api_key=args.api_key,
    )
    sys.exit(0 if result >= 0 else 1)


if __name__ == "__main__":
    main()
