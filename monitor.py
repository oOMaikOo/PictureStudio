#!/usr/bin/env python3
"""
PictureStudio Monitor-Client — Standalone Anomalie-Erkennung für den Produktionseinsatz.

Lädt ein trainiertes Modell und verbindet sich mit der Kamera.
ROI und Schwellwert werden automatisch aus den Modell-Metadaten übernommen.

Verwendung:
    python monitor.py                         # interaktive Kamera-Auswahl + Browser-Setup
    python monitor.py --model anomalie.pth    # direkt starten mit bestehendem Modell

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


def _discover_cameras() -> list:
    """Scan for available USB cameras and return [(index, label)] list."""
    try:
        cams = list_usb_cameras()
        return cams if cams else []
    except Exception:
        return []


def _terminal_camera_select(cameras: list) -> list:
    """Interactive terminal camera selection with multi-select support.

    Prints a numbered list of discovered cameras and asks the user to select
    one or more by index. Returns a list of camera source values (int indices).
    Supports: "0", "0 1 2", "0,1,2", "all", empty string = all.
    """
    if not cameras:
        print("  Keine USB-Kameras gefunden. IP-Kamera/RTSP-URL kann im Web-Interface eingegeben werden.")
        return []

    print("\n  Verfügbare Kameras:")
    for i, (idx, label) in enumerate(cameras):
        print(f"    [{i}]  Index {idx} — {label}")

    print()
    try:
        raw = input("  Auswahl (z.B. '0', '0 1', '0,1,2' oder 'alle'): ").strip()
    except (EOFError, KeyboardInterrupt):
        return []

    if not raw or raw.lower() in ("alle", "all", "*"):
        return [idx for idx, _ in cameras]

    selected = []
    for tok in raw.replace(",", " ").split():
        try:
            pos = int(tok)
            if 0 <= pos < len(cameras):
                selected.append(cameras[pos][0])
            else:
                print(f"  [Warnung] Position {pos} ungültig — übersprungen.")
        except ValueError:
            print(f"  [Warnung] '{tok}' ist keine Zahl — übersprungen.")
    return selected


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
                if attempt == 1 and isinstance(self._source, int) and os.uname().sysname == "Darwin":
                    print(f"[Kamera] Kamera-Index {self._source} konnte nicht geöffnet werden.")
                    print("[Kamera] macOS: Bitte Terminal-Kamerazugriff erlauben:")
                    print("[Kamera]   Systemeinstellungen → Datenschutz & Sicherheit → Kamera → Terminal ✓")
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
            first_frame = True
            # Live cameras (especially built-in on macOS) need up to 30 read()
            # calls before delivering the first frame — don't break too early.
            warmup_limit = 3 if self._is_video else 60
            t_next = time.perf_counter()

            while self._running:
                now = time.perf_counter()
                if now < t_next:
                    time.sleep(0.005)
                    continue
                t_next = now + delay

                ret, frame = cap.read()
                if ret and frame is not None:
                    if first_frame:
                        first_frame = False
                        src_label = self._source if isinstance(self._source, str) else f"Index {self._source}"
                        print(f"[Kamera] Erster Frame von {src_label} empfangen "
                              f"({frame.shape[1]}×{frame.shape[0]})")
                    consec_fail = 0
                    try:
                        self._callback(frame)
                    except Exception:
                        pass
                else:
                    consec_fail += 1
                    if consec_fail >= warmup_limit:
                        print(f"[Kamera] Quelle {self._source}: {consec_fail} "
                              f"aufeinanderfolgende Fehler — trenne Verbindung.")
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
# Setup-Wizard — multi-channel, no training on edge device
# ---------------------------------------------------------------------------

_SETUP_HTML: str = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PictureStudio Monitor – Einrichtung</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#22272E;color:#CDD9E5;padding:16px;min-height:100vh}
h1{color:#5DADE2;margin-bottom:14px;font-size:19px;letter-spacing:.02em}
/* Top bar */
.topbar{display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:5px;padding:7px 16px;border-radius:5px;
     border:none;cursor:pointer;font-size:12px;font-weight:600;transition:background .15s}
.btn-primary{background:#1C6EA4;color:#fff}.btn-primary:hover{background:#2185C5}
.btn-success{background:#1A8754;color:#fff}.btn-success:hover{background:#1DB05E}
.btn-success:disabled{background:#2D333B;color:#545D68;cursor:not-allowed;opacity:.6}
.btn-danger{background:#8B2222;color:#fff}.btn-danger:hover{background:#A52626}
.btn-sm{padding:4px 10px;font-size:11px}
/* Add channel controls */
#add-area{display:flex;gap:6px;align-items:center}
#cam-input{background:#1B2028;border:1px solid #373E47;color:#CDD9E5;border-radius:4px;
           padding:5px 8px;width:120px;font-size:12px}
/* Grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
/* Channel card */
.ch-card{background:#2D333B;border-radius:8px;border:1px solid #373E47;padding:12px;position:relative}
.ch-card .ch-title{font-size:13px;font-weight:bold;color:#7FC3F5;margin-bottom:8px}
.ch-canvas-wrap{position:relative;width:100%;padding-top:56.25%;background:#1B2028;border-radius:5px;overflow:hidden;margin-bottom:8px}
.ch-canvas-wrap img,.ch-canvas-wrap canvas{position:absolute;top:0;left:0;width:100%;height:100%}
.ch-canvas-wrap img{object-fit:contain}
.ch-canvas-wrap canvas{cursor:crosshair}
.ch-info{display:flex;gap:10px;font-size:11px;color:#768390;margin-bottom:8px;flex-wrap:wrap}
.ch-info span b{color:#CDD9E5}
.badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:600}
.badge-ok{background:#1E3A2F;color:#58D68D}.badge-wait{background:#3D2B00;color:#FFC107}
.badge-pending{background:#373E47;color:#768390}
.ch-actions{display:flex;gap:6px;flex-wrap:wrap}
/* Success */
.success-full{display:none;text-align:center;padding:60px 20px}
.success-full .icon{font-size:64px;margin-bottom:14px}
.success-full h2{color:#58D68D;font-size:22px;margin-bottom:8px}
.success-full p{color:#768390;font-size:13px}
</style>
</head>
<body>
<h1>PictureStudio Monitor – Einrichtung</h1>
<div class="topbar" id="topbar">
  <div id="add-area">
    <input id="cam-input" type="text" placeholder="Kamera-Index (0-8) oder URL" value="0">
    <button class="btn btn-primary" onclick="addChannel()">+ Kanal hinzufügen</button>
  </div>
  <button class="btn btn-success" id="btn-live" disabled onclick="goLive()">Alle live starten</button>
</div>
<div id="cam-discovery" style="margin-bottom:12px;display:none">
  <span style="font-size:12px;color:#768390;margin-right:8px">Erkannte Kameras:</span>
  <span id="cam-discovery-list"></span>
</div>
<div class="grid" id="grid"></div>
<div class="success-full" id="success-full">
  <div class="icon">&#10003;</div>
  <h2>Live-Modus gestartet</h2>
  <p>Einrichtung abgeschlossen.<br>Dieses Fenster kann geschlossen werden.</p>
</div>

<script>
// ─── State ───────────────────────────────────────────────────────────────────
let channels = [];      // [{channel_id, camera_source, roi, model_path, status}, ...]
let phase = 'setup';
// ROI drawing state per channel
let roiState = {};      // {id: {drawing, sx, sy, ex, ey}}

// ─── Polling ─────────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/setup/status');
    const d = await r.json();
    phase = d.phase || 'setup';
    channels = d.channels || [];
    if (phase === 'live') {
      showSuccess();
      return;
    }
    updateGrid();
  } catch(e) {}
  setTimeout(pollStatus, 1000);
}

// ─── Grid rendering ───────────────────────────────────────────────────────────
function updateGrid() {
  const grid = document.getElementById('grid');
  const ids = new Set(channels.map(c => c.channel_id));
  // Remove cards for deleted channels
  [...grid.querySelectorAll('.ch-card')].forEach(el => {
    if (!ids.has(parseInt(el.dataset.id))) el.remove();
  });
  channels.forEach(ch => {
    let card = grid.querySelector('.ch-card[data-id="'+ch.channel_id+'"]');
    if (!card) {
      card = buildCard(ch);
      grid.appendChild(card);
    }
    refreshCard(card, ch);
  });
  // Live button
  const allReady = channels.length > 0 && channels.every(c => c.status === 'ready');
  document.getElementById('btn-live').disabled = !allReady;
}

function buildCard(ch) {
  const id = ch.channel_id;
  const div = document.createElement('div');
  div.className = 'ch-card';
  div.dataset.id = id;
  div.innerHTML = `
    <div class="ch-title">Kanal ${id} &nbsp;<span class="badge badge-pending" id="badge-${id}">pending</span></div>
    <div class="ch-canvas-wrap">
      <img id="img-${id}" src="" alt="">
      <canvas id="canvas-${id}"></canvas>
    </div>
    <div class="ch-info">
      <span>ROI: <b id="roi-info-${id}">—</b></span>
      <span>Modell: <b id="model-info-${id}">—</b></span>
    </div>
    <div class="ch-actions">
      <button class="btn btn-sm btn-primary" onclick="resetROI(${id})">ROI zuruecksetzen</button>
      <button class="btn btn-sm btn-danger" onclick="removeChannel(${id})">Entfernen</button>
    </div>`;
  // Start frame polling for this channel
  startFramePoll(id);
  // ROI drawing
  setupROICanvas(id);
  return div;
}

function refreshCard(card, ch) {
  const id = ch.channel_id;
  const badge = document.getElementById('badge-'+id);
  if (badge) {
    badge.textContent = ch.status;
    badge.className = 'badge ' + (ch.status === 'ready' ? 'badge-ok' : ch.status === 'pending' ? 'badge-pending' : 'badge-wait');
  }
  const roiInfo = document.getElementById('roi-info-'+id);
  if (roiInfo) roiInfo.textContent = ch.roi ? '['+ch.roi.map(v=>Math.round(v)).join(',')+']' : '—';
  const modelInfo = document.getElementById('model-info-'+id);
  if (modelInfo) modelInfo.textContent = ch.model_path ? '✓ bereit' : '⏳ ausstehend';
  // Draw ROI on canvas
  drawROI(id, ch.roi);
}

// ─── Frame polling per channel ────────────────────────────────────────────────
function startFramePoll(id) {
  function poll() {
    const img = document.getElementById('img-'+id);
    if (!img) return; // card removed
    img.src = '/setup/channels/'+id+'/frame.jpg?t='+Date.now();
    setTimeout(poll, 400);
  }
  poll();
}

// ─── ROI drawing ──────────────────────────────────────────────────────────────
function setupROICanvas(id) {
  roiState[id] = {drawing: false, sx:0, sy:0, ex:0, ey:0};
  // Use delegation — canvas created in buildCard, wire events after DOM insert
  setTimeout(() => {
    const canvas = document.getElementById('canvas-'+id);
    if (!canvas) return;
    canvas.addEventListener('mousedown', e => {
      const r = canvas.getBoundingClientRect();
      roiState[id].drawing = true;
      roiState[id].sx = e.clientX - r.left;
      roiState[id].sy = e.clientY - r.top;
      roiState[id].ex = roiState[id].sx;
      roiState[id].ey = roiState[id].sy;
    });
    canvas.addEventListener('mousemove', e => {
      if (!roiState[id] || !roiState[id].drawing) return;
      const r = canvas.getBoundingClientRect();
      roiState[id].ex = e.clientX - r.left;
      roiState[id].ey = e.clientY - r.top;
      drawDragROI(canvas, roiState[id]);
    });
    canvas.addEventListener('mouseup', e => {
      if (!roiState[id] || !roiState[id].drawing) return;
      roiState[id].drawing = false;
      const r = canvas.getBoundingClientRect();
      roiState[id].ex = e.clientX - r.left;
      roiState[id].ey = e.clientY - r.top;
      const W = r.width, H = r.height;
      const x = Math.min(roiState[id].sx, roiState[id].ex);
      const y = Math.min(roiState[id].sy, roiState[id].ey);
      const w = Math.abs(roiState[id].ex - roiState[id].sx);
      const h = Math.abs(roiState[id].ey - roiState[id].sy);
      if (w > 4 && h > 4) {
        // Send pixel coords
        fetch('/setup/channels/'+id+'/roi', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({roi:[x,y,w,h]})
        });
      }
    });
  }, 50);
}

function drawDragROI(canvas, rs) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#F39C12';
  ctx.lineWidth = 2;
  const x = Math.min(rs.sx, rs.ex), y = Math.min(rs.sy, rs.ey);
  const w = Math.abs(rs.ex-rs.sx), h = Math.abs(rs.ey-rs.sy);
  ctx.strokeRect(x, y, w, h);
}

function drawROI(id, roi) {
  const canvas = document.getElementById('canvas-'+id);
  if (!canvas) return;
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!roi || roi.length !== 4) return;
  // roi is [x, y, w, h] in pixels at capture resolution
  // We draw them proportionally on the display canvas
  ctx.strokeStyle = '#2ECC71';
  ctx.lineWidth = 2;
  // The server stores the roi in display-canvas-pixel coords
  ctx.strokeRect(roi[0], roi[1], roi[2], roi[3]);
}

// ─── API calls ────────────────────────────────────────────────────────────────
async function addChannel() {
  const val = document.getElementById('cam-input').value.trim();
  const src = (val === '' || val === null) ? 0 : (isNaN(val) ? val : parseInt(val));
  await fetch('/setup/channels/add', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({camera_source: src})
  });
}

async function removeChannel(id) {
  await fetch('/setup/channels/'+id, {method:'DELETE'});
}

async function resetROI(id) {
  await fetch('/setup/channels/'+id+'/roi', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({roi: null})
  });
}

async function goLive() {
  document.getElementById('btn-live').disabled = true;
  await fetch('/setup/go_live', {method:'POST'});
  showSuccess();
}

function showSuccess() {
  document.getElementById('topbar').style.display='none';
  document.getElementById('grid').style.display='none';
  document.getElementById('success-full').style.display='block';
}

// ─── Discovered cameras ───────────────────────────────────────────────────────
async function loadDiscoveredCameras() {
  try {
    const r = await fetch('/setup/cameras');
    const cams = await r.json();
    const area = document.getElementById('cam-discovery');
    const list = document.getElementById('cam-discovery-list');
    if (!cams || cams.length === 0) return;
    list.innerHTML = cams.map(c =>
      `<button class="btn btn-sm btn-primary" style="margin:2px" onclick="addCameraByIndex(${c.index})">
        + ${c.label || 'Index '+c.index}
       </button>`
    ).join('');
    area.style.display = 'block';
  } catch(e) {}
}
async function addCameraByIndex(idx) {
  await fetch('/setup/channels/add', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({camera_source: idx})
  });
}
loadDiscoveredCameras();

// ─── Boot ─────────────────────────────────────────────────────────────────────
pollStatus();
</script>
</body>
</html>"""


class _SetupChannel:
    """
    Represents one camera channel in the multi-channel setup wizard.

    Holds camera source, optional ROI, the deployed model path, and the most
    recent frame captured by the associated camera thread.  All frame access
    goes through get_frame()/set_frame() to stay thread-safe.
    """

    def __init__(self, channel_id: int, camera_source: Union[int, str]) -> None:
        self.channel_id: int = channel_id
        self.camera_source: Union[int, str] = camera_source
        self.roi: Optional[list] = None          # [x, y, w, h] in canvas pixels, or None
        self.model_path: str = ""                # populated after deploy
        self.status: str = "pending"             # pending | ready
        self.last_frame: Optional[np.ndarray] = None
        self.cam_thread: Optional[_CameraThread] = None
        self._lock: threading.Lock = threading.Lock()

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame in a thread-safe manner."""
        with self._lock:
            return self.last_frame

    def set_frame(self, frame: np.ndarray) -> None:
        """Store the most recent frame in a thread-safe manner."""
        with self._lock:
            self.last_frame = frame.copy()

    def to_dict(self) -> dict:
        """Serialise channel metadata (no frame data, no thread reference)."""
        return {
            "channel_id":    self.channel_id,
            "camera_source": self.camera_source,
            "roi":           self.roi,
            "model_path":    self.model_path,
            "status":        self.status,
        }


class _SetupState:
    """
    Thread-safe state container for the multi-channel Setup-Wizard.

    Tracks all configured channels and the overall wizard phase
    (setup | live | error).
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self.channels: list = []    # list of _SetupChannel
        self.phase: str = "setup"
        self.error: str = ""
        self._next_id: int = 0

    def add_channel(self, camera_source: Union[int, str]) -> "_SetupChannel":
        """Create a new channel, append it to the list, and return it."""
        with self._lock:
            ch = _SetupChannel(self._next_id, camera_source)
            self._next_id += 1
            self.channels.append(ch)
        return ch

    def remove_channel(self, channel_id: int) -> bool:
        """Remove a channel by id.  Returns True if found and removed."""
        with self._lock:
            before = len(self.channels)
            self.channels = [c for c in self.channels if c.channel_id != channel_id]
            return len(self.channels) < before

    def get_channel(self, channel_id: int) -> Optional["_SetupChannel"]:
        """Return the channel with the given id, or None."""
        with self._lock:
            for ch in self.channels:
                if ch.channel_id == channel_id:
                    return ch
        return None

    def all_ready(self) -> bool:
        """True iff all channels have a non-empty model_path (status == 'ready')."""
        with self._lock:
            if not self.channels:
                return False
            return all(c.model_path != "" for c in self.channels)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of the current state."""
        with self._lock:
            return {
                "phase":    self.phase,
                "error":    self.error,
                "channels": [c.to_dict() for c in self.channels],
            }


class _SetupHandler(BaseHTTPRequestHandler):
    """
    HTTP handler for the multi-channel Setup-Wizard web UI.

    Class-level attributes are set by _SetupApiServer before the server starts.
    All routes are prefixed with /setup/.
    """

    state: "_SetupState" = None      # type: ignore[assignment]
    output_dir: str = "monitor_logs"
    discovered_cameras: list = []   # [(index, label)] from startup scan

    def log_message(self, fmt, *args) -> None:
        """Suppress access log output."""
        pass

    # ── helpers ────────────────────────────────────────────────────────────────

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
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

    def _parse_channel_id(self, parts: list) -> Optional[int]:
        """Extract numeric channel id from a URL path segments list."""
        try:
            return int(parts[0])
        except (IndexError, ValueError):
            return None

    # ── OPTIONS ────────────────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── GET ────────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # Redirect root to /setup
        if path in ("/", ""):
            self.send_response(301)
            self.send_header("Location", "/setup")
            self._cors()
            self.end_headers()
            return

        if path == "/setup":
            self._send_html(_SETUP_HTML)
            return

        # GET /setup/cameras — list of discovered USB cameras
        if path == "/setup/cameras":
            self._send_json([
                {"index": idx, "label": label}
                for idx, label in _SetupHandler.discovered_cameras
            ])
            return

        if path == "/setup/status":
            self._send_json(_SetupHandler.state.snapshot())
            return

        # /setup/channels/{id}/frame.jpg
        if path.startswith("/setup/channels/"):
            rest = path[len("/setup/channels/"):]  # e.g. "0/frame.jpg"
            parts = rest.split("/")
            ch_id = self._parse_channel_id(parts)
            if ch_id is None:
                self._send_json({"error": "not found"}, 404)
                return
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "frame.jpg":
                ch = _SetupHandler.state.get_channel(ch_id)
                if ch is None:
                    self._send_json({"error": "channel not found"}, 404)
                    return
                frame = ch.get_frame()
                if frame is None:
                    # Return a small black placeholder JPEG so <img> never breaks
                    placeholder = np.zeros((120, 160, 3), dtype=np.uint8)
                    ok, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 50])
                else:
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
            self._send_json({"error": "not found"}, 404)
            return

        self._send_json({"error": "not found"}, 404)

    # ── DELETE ─────────────────────────────────────────────────────────────────

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        st = _SetupHandler.state

        # DELETE /setup/channels/{id}
        if path.startswith("/setup/channels/"):
            rest = path[len("/setup/channels/"):]
            parts = rest.split("/")
            ch_id = self._parse_channel_id(parts)
            if ch_id is None:
                self._send_json({"error": "bad request"}, 400)
                return
            ch = st.get_channel(ch_id)
            if ch and ch.cam_thread:
                ch.cam_thread.stop()
            removed = st.remove_channel(ch_id)
            if removed:
                self._send_json({"ok": True, "removed": ch_id})
            else:
                self._send_json({"error": "channel not found"}, 404)
            return

        self._send_json({"error": "not found"}, 404)

    # ── POST ───────────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        st = _SetupHandler.state

        # POST /setup/channels/add
        if path == "/setup/channels/add":
            body = self._read_json_body()
            src = body.get("camera_source", 0)
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            ch = st.add_channel(src)

            def _on_frame(frame: np.ndarray) -> None:
                ch.set_frame(frame)

            cam_thread = _CameraThread(
                source=src,
                fps=15.0,
                callback=_on_frame,
                reconnect_delay=5.0,
            )
            ch.cam_thread = cam_thread
            cam_thread.start()
            self._send_json({"ok": True, "channel_id": ch.channel_id})
            return

        # POST /setup/channels/{id}/roi or /setup/channels/{id}/deploy
        if path.startswith("/setup/channels/"):
            rest = path[len("/setup/channels/"):]
            parts = rest.split("/")
            ch_id = self._parse_channel_id(parts)
            if ch_id is None:
                self._send_json({"error": "bad request"}, 400)
                return
            sub = parts[1] if len(parts) > 1 else ""

            ch = st.get_channel(ch_id)
            if ch is None:
                self._send_json({"error": "channel not found"}, 404)
                return

            if sub == "roi":
                body = self._read_json_body()
                roi = body.get("roi")
                if roi is not None:
                    roi = [float(v) for v in roi[:4]]
                ch.roi = roi
                self._send_json({"ok": True, "roi": ch.roi})
                return

            if sub == "deploy":
                # Multipart/form-data with field "model"
                ct = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ct:
                    self._send_json({"error": "expected multipart/form-data"}, 400)
                    return
                # Parse boundary
                boundary = None
                for part in ct.split(";"):
                    part = part.strip()
                    if part.startswith("boundary="):
                        boundary = part[len("boundary="):].strip()
                if not boundary:
                    self._send_json({"error": "missing boundary"}, 400)
                    return
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                # Find file data between boundaries
                bnd_bytes = ("--" + boundary).encode()
                parts_raw = raw.split(bnd_bytes)
                model_data: Optional[bytes] = None
                for seg in parts_raw[1:]:
                    if b"name=\"model\"" in seg or b"name='model'" in seg:
                        # header ends at \r\n\r\n
                        idx = seg.find(b"\r\n\r\n")
                        if idx >= 0:
                            model_data = seg[idx + 4:]
                            # strip trailing \r\n--
                            if model_data.endswith(b"\r\n"):
                                model_data = model_data[:-2]
                        break
                if not model_data:
                    self._send_json({"error": "model field not found in upload"}, 400)
                    return
                os.makedirs(
                    os.path.join(_SetupHandler.output_dir, "setup_models"), exist_ok=True
                )
                # Determine extension from Content-Disposition or default to .pth
                ext = ".pth"
                for seg in parts_raw[1:]:
                    if b"name=\"model\"" in seg or b"name='model'" in seg:
                        if b".onnx" in seg[:200]:
                            ext = ".onnx"
                        break
                model_file = os.path.join(
                    _SetupHandler.output_dir,
                    "setup_models",
                    f"channel_{ch_id}{ext}",
                )
                with open(model_file, "wb") as f:
                    f.write(model_data)
                ch.model_path = model_file
                ch.status = "ready"
                self._send_json({"ok": True, "model_path": model_file})
                return

            self._send_json({"error": "not found"}, 404)
            return

        # POST /setup/go_live
        if path == "/setup/go_live":
            with st._lock:
                st.phase = "live"
            self._send_json({"ok": True})
            return

        self._send_json({"error": "not found"}, 404)


class _SetupApiServer:
    """Runs the multi-channel Setup-Wizard HTTP server in a background daemon thread."""

    def __init__(
        self,
        port: int,
        state: "_SetupState",
        output_dir: str = "monitor_logs",
    ) -> None:
        self._port = port
        _SetupHandler.state = state
        _SetupHandler.output_dir = output_dir
        _SetupHandler.discovered_cameras = getattr(_SetupHandler, 'discovered_cameras', [])
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
    discovered_cameras: list = [],
) -> list:
    """
    Start the multi-channel Setup-Wizard and block until go_live is called.

    Training happens exclusively in Picture Studio; this wizard only handles
    camera preview, ROI drawing, and model deployment (binary upload).

    Parameters
    ----------
    camera_source:
        Optional initial camera source (USB index or URL/RTSP string).
        If provided, a first channel is created automatically.
    setup_port:
        TCP port for the embedded HTTP server.
    output_dir:
        Directory where deployed model files are stored under setup_models/.

    Returns
    -------
    list of dict
        [{"channel_id": int, "camera_source": ..., "roi": [...], "model_path": str}, ...]
    """
    os.makedirs(os.path.join(output_dir, "setup_models"), exist_ok=True)

    state = _SetupState()

    # Optionally pre-populate one channel
    if camera_source is not None:
        ch = state.add_channel(camera_source)

        def _on_frame(frame: np.ndarray) -> None:
            ch.set_frame(frame)

        cam_thread = _CameraThread(
            source=camera_source,
            fps=15.0,
            callback=_on_frame,
            reconnect_delay=5.0,
        )
        ch.cam_thread = cam_thread
        cam_thread.start()

    _SetupHandler.discovered_cameras = discovered_cameras
    server = _SetupApiServer(port=setup_port, state=state, output_dir=output_dir)
    server.start()

    print(
        f"\n"
        f"╔═══════════════════════════════════════════════════╗\n"
        f"║  PictureStudio Monitor – Einrichtungsmodus        ║\n"
        f"║  Web-Interface: http://localhost:{setup_port}/setup      ║\n"
        f"╚═══════════════════════════════════════════════════╝\n"
    )

    # Block until go_live or error
    while state.phase not in ("live", "error"):
        time.sleep(0.5)

    # Stop all channel camera threads
    with state._lock:
        for ch in state.channels:
            if ch.cam_thread:
                ch.cam_thread.stop()

    server.stop()

    with state._lock:
        return [c.to_dict() for c in state.channels]


# ---------------------------------------------------------------------------
# Multi-channel monitor loop
# ---------------------------------------------------------------------------

def run_monitor_multi(
    channels: list,
    output_dir: str = "monitor_logs",
    fps: float = 15.0,
    cooldown: float = 10.0,
    headless: bool = True,
    api_port: int = 0,
    api_key: str = "",
    reconnect_delay: float = 5.0,
    mqtt_host: str = "",
    mqtt_port: int = 1883,
    mqtt_topic: str = "picture_studio/monitor",
    mqtt_user: str = "",
    mqtt_pass: str = "",
) -> int:
    """
    Run the multi-channel monitor loop.

    For each channel dict (from run_setup() or a channels.json file), this
    function starts a camera thread and loads the associated anomaly detector
    (.pth via AnomalyDetector.load() or .onnx via OnnxAnomalyScorer).
    ROI is applied per-channel.  Alarms are written to per-channel CSV files.

    Parameters
    ----------
    channels:
        List of dicts with keys channel_id, camera_source, roi, model_path.
    output_dir:
        Directory for alarm images and CSV logs.
    fps, cooldown, headless, api_port, api_key, reconnect_delay:
        Same semantics as run_monitor().
    mqtt_host, mqtt_port, mqtt_topic, mqtt_user, mqtt_pass:
        MQTT broker settings for alarm publishing.

    Returns
    -------
    int
        0 on clean exit, -1 on configuration error.
    """
    if not channels:
        print("Keine Kanäle konfiguriert.")
        return -1

    os.makedirs(output_dir, exist_ok=True)

    # ── MQTT ──────────────────────────────────────────────────────────────────
    mqtt_client = None
    if mqtt_host:
        if not HAS_MQTT:
            print("[Warnung] paho-mqtt nicht installiert — MQTT deaktiviert.")
        elif MQTTAlarmClient is not None:
            mqtt_client = MQTTAlarmClient(
                host=mqtt_host, port=mqtt_port, topic=mqtt_topic,
                username=mqtt_user, password=mqtt_pass,
            )
            if mqtt_client.connect():
                time.sleep(0.3)
            else:
                print(f"[Warnung] MQTT-Verbindung fehlgeschlagen: {mqtt_client.last_error}")
                mqtt_client = None

    # ── Per-channel setup ─────────────────────────────────────────────────────
    state = _MonitorState()
    state.output_dir = output_dir
    state.api_key = api_key

    cam_threads: list = []
    last_alarm_t: dict = {}   # channel_id -> float

    for ch_dict in channels:
        ch_id = ch_dict.get("channel_id", 0)
        model_path = ch_dict.get("model_path", "")
        roi = ch_dict.get("roi")
        cam_src = ch_dict.get("camera_source", 0)

        if not model_path or not os.path.isfile(model_path):
            print(f"[Kanal {ch_id}] Modell nicht gefunden: {model_path} — übersprungen")
            continue

        # Load detector (lazy torch import inside AnomalyDetector)
        if model_path.lower().endswith(".onnx"):
            if not HAS_ORT or OnnxAnomalyScorer is None:
                print(f"[Kanal {ch_id}] onnxruntime nicht verfügbar — übersprungen")
                continue
            det = OnnxAnomalyScorer.from_path(model_path)
        else:
            det = AnomalyDetector()
            try:
                det.load(model_path)
            except Exception as exc:
                print(f"[Kanal {ch_id}] Fehler beim Laden: {exc} — übersprungen")
                continue

        threshold = det.threshold
        log_path = os.path.join(output_dir, f"channel_{ch_id}_events.csv")
        last_alarm_t[ch_id] = -cooldown

        def _make_callback(cid, detector, croi, cthr, clog):
            def on_frame(frame: np.ndarray) -> None:
                cropped = apply_roi(frame, croi)
                try:
                    score = detector.score(cropped)
                except Exception:
                    return
                state.push_score(score, cthr)
                if headless:
                    s = "ANOMALIE" if score > cthr else "Normal  "
                    print(
                        f"\r[Kanal {cid}] Score: {score:.5f}  {s}   ",
                        end="", flush=True,
                    )
                if score > cthr:
                    now = time.perf_counter()
                    if now - last_alarm_t[cid] >= cooldown:
                        last_alarm_t[cid] = now
                        fname = save_alarm(
                            frame, score, cthr, output_dir, clog, state.event_count + 1
                        )
                        state.push_alarm(score, cthr, fname)
                        if mqtt_client:
                            try:
                                fpath = os.path.join(output_dir, fname) if fname else ""
                                mqtt_client.publish_alarm(score, cthr, frame_path=fpath)
                            except Exception:
                                pass
            return on_frame

        cb = _make_callback(ch_id, det, roi, threshold, log_path)
        cam = _CameraThread(
            source=cam_src,
            fps=fps,
            callback=cb,
            reconnect_delay=reconnect_delay,
        )
        cam.start()
        cam_threads.append(cam)
        print(f"  Kanal {ch_id}: Kamera={cam_src}  ROI={roi}  Modell={os.path.basename(model_path)}")

    if not cam_threads:
        print("Kein einziger Kanal konnte gestartet werden.")
        return -1

    # ── Optional REST API ─────────────────────────────────────────────────────
    api_server = None
    if api_port > 0:
        try:
            api_server = _MonitorApiServer(api_port, state)
            api_server.start()
            print(f"  REST-API: http://localhost:{api_port}/api/status")
        except OSError as exc:
            print(f"[Warnung] REST-API konnte nicht gestartet werden: {exc}")

    print(f"\nMulti-Channel-Monitor läuft ({len(cam_threads)} Kanal/Kanäle). Strg+C zum Beenden.\n")

    _WATCHDOG_INTERVAL = 5.0
    _t_watchdog = time.perf_counter()

    try:
        while True:
            time.sleep(0.1)
            now = time.perf_counter()
            if now - _t_watchdog >= _WATCHDOG_INTERVAL:
                _t_watchdog = now
                for i, cam in enumerate(cam_threads):
                    if not cam.is_alive():
                        print(f"[Watchdog] Kanal {i}: Thread beendet — starte neu…")
                        new_cam = _CameraThread(
                            source=cam._source,
                            fps=fps,
                            callback=cam._callback,
                            reconnect_delay=reconnect_delay,
                        )
                        new_cam.start()
                        cam_threads[i] = new_cam
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cam_threads:
            cam.stop()
        for cam in cam_threads:
            cam.join(timeout=3.0)
        if mqtt_client:
            mqtt_client.disconnect()
        if api_server:
            api_server.stop()
        if headless:
            print()

    print(f"\nMonitor beendet. {state.event_count} Alarm-Event(s) protokolliert.")
    return 0


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

  # Multi-Channel Setup-Wizard (Training in Picture Studio)
  python monitor.py --setup --setup-port 8765

  # Multi-Channel aus gespeicherter Konfiguration
  python monitor.py --channels /pfad/channels.json --headless
        """,
    )

    # Setup-Wizard
    setup_grp = parser.add_argument_group("Einrichtungsmodus (Setup-Wizard, kein Training auf Edge-Gerät)")
    setup_grp.add_argument("--setup", action="store_true",
                           help="Multi-Channel Setup-Wizard starten (kein --model nötig)")
    setup_grp.add_argument("--setup-port", type=int, default=8765, metavar="PORT",
                           help="HTTP-Port für den Setup-Wizard (Standard: 8765)")
    setup_grp.add_argument("--channels", default=None, metavar="PFAD",
                           help="Pfad zu einer JSON-Datei mit gespeicherter Kanal-Konfiguration")

    # Modell
    parser.add_argument("--model", nargs="?", default=None, metavar="PFAD",
                        help="Pfad zur trainierten .pth- oder .onnx-Modelldatei (Pflicht außer mit --setup/--channels)")

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

    # ── Setup-Wizard (multi-channel, no training on edge device) ──────────────
    if args.setup:
        if args.url:
            camera_source: Union[int, str, None] = args.url
        elif args.camera is not None:
            camera_source = args.camera
        else:
            camera_source = None  # wizard starts with no channels pre-configured

        channels = run_setup(
            camera_source=camera_source,
            setup_port=args.setup_port,
            output_dir=args.output,
            discovered_cameras=_discover_cameras(),
        )
        if not channels:
            print("Keine Kanäle konfiguriert.")
            sys.exit(1)
        print(f"\nEinrichtung abgeschlossen. Starte Live-Monitoring mit {len(channels)} Kanal/Kanäle...")
        if len(channels) == 1:
            ch = channels[0]
            run_monitor(
                model_path=ch["model_path"],
                camera_source=ch.get("camera_source"),
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        else:
            run_monitor_multi(
                channels=channels,
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        return

    # ── Channels JSON (pre-configured multi-channel) ───────────────────────────
    if args.channels:
        import json as _json
        try:
            with open(args.channels, encoding="utf-8") as fh:
                channels = _json.load(fh)
        except Exception as exc:
            print(f"Fehler beim Lesen der Kanal-Konfiguration: {exc}")
            sys.exit(1)
        run_monitor_multi(
            channels=channels,
            output_dir=args.output,
            fps=args.fps,
            cooldown=args.cooldown,
            headless=args.headless,
            api_port=args.api_port,
            api_key=args.api_key,
            reconnect_delay=args.reconnect_delay,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            mqtt_topic=args.mqtt_topic,
            mqtt_user=args.mqtt_user,
            mqtt_pass=args.mqtt_pass,
        )
        return

    # ── No arguments: interactive camera selection → auto setup wizard ─────────
    if not args.model:
        import webbrowser
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  PictureStudio Monitor — Kamera-Erkennung                   ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")
        print("  Scanne verfügbare Kameras…")
        discovered = _discover_cameras()
        if discovered:
            print(f"  {len(discovered)} Kamera(s) gefunden.")
        else:
            print("  Keine USB-Kameras gefunden (IP-Kameras können im Browser eingegeben werden).")

        # B) Terminal-Auswahl
        selected = _terminal_camera_select(discovered)

        # A) Web-UI starten
        setup_port = args.setup_port
        print(f"\n  Starte Setup-Wizard auf http://localhost:{setup_port}/setup …")
        webbrowser.open(f"http://localhost:{setup_port}/setup")

        # Pre-populate selected cameras as channels
        first_source: Union[int, str, None] = selected[0] if selected else None
        channels = run_setup(
            camera_source=first_source,
            setup_port=setup_port,
            output_dir=args.output,
            discovered_cameras=discovered,
        )

        # Add remaining selected cameras (beyond the first) as extra channels
        # (they can also be added via the web UI, but pre-populate if terminal selection was used)

        if not channels:
            print("Keine Kanäle konfiguriert.")
            sys.exit(1)
        print(f"\nEinrichtung abgeschlossen. Starte Live-Monitoring mit {len(channels)} Kanal/Kanäle…")
        if len(channels) == 1:
            ch = channels[0]
            run_monitor(
                model_path=ch["model_path"],
                camera_source=ch.get("camera_source"),
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        else:
            run_monitor_multi(
                channels=channels,
                output_dir=args.output,
                fps=args.fps,
                cooldown=args.cooldown,
                headless=args.headless,
                api_port=args.api_port,
                api_key=args.api_key,
                reconnect_delay=args.reconnect_delay,
                mqtt_host=args.mqtt_host,
                mqtt_port=args.mqtt_port,
                mqtt_topic=args.mqtt_topic,
                mqtt_user=args.mqtt_user,
                mqtt_pass=args.mqtt_pass,
            )
        return
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
