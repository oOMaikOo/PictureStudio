#!/usr/bin/env python3
"""
PictureStudio Monitor-Client — Standalone Anomalie-Erkennung für den Produktionseinsatz.

Lädt ein trainiertes Modell und verbindet sich automatisch mit der Kamera,
die beim Training verwendet wurde. ROI und Schwellwert werden aus den
Modell-Metadaten übernommen.

Verwendung:
    python monitor.py --model MODEL_PFAD [Optionen]

Optionen:
    --model PFAD          Pfad zur trainierten .pt-Modelldatei (Pflicht)
    --camera INDEX        Kamera-Index manuell überschreiben
    --threshold WERT      Anomalie-Schwellwert überschreiben
    --output VERZ         Ausgabeverzeichnis für Logs/Bilder (Standard: monitor_logs)
    --fps FPS             Bilder pro Sekunde (Standard: 15)
    --cooldown SEK        Mindestabstand zwischen Alarm-Saves in Sekunden (Standard: 10)
    --headless            Kein Fenster — nur Terminal + CSV
"""

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import cv2
import numpy as np

# Project root on path so core/ imports work when called from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.anomaly_detector import AnomalyDetector
from core.camera import list_usb_cameras


# ---------------------------------------------------------------------------
# Camera thread (no Qt dependency)
# ---------------------------------------------------------------------------

class _CameraThread(threading.Thread):
    """Grabs frames from a cv2.VideoCapture and calls *callback* for each."""

    def __init__(self, camera_idx: int, fps: float,
                 callback: Callable[[np.ndarray], None]) -> None:
        super().__init__(daemon=True, name=f"monitor-cam-{camera_idx}")
        self._idx = camera_idx
        self._delay = 1.0 / max(fps, 1.0)
        self._callback = callback
        self._running = False
        self.error: Optional[str] = None

    def run(self) -> None:
        cap = cv2.VideoCapture(self._idx)
        if not cap.isOpened():
            self.error = f"Kamera {self._idx} konnte nicht geöffnet werden."
            return
        self._running = True
        t_next = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            if now < t_next:
                time.sleep(0.005)
                continue
            t_next = now + self._delay
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    self._callback(frame)
                except Exception:
                    pass
        cap.release()

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_camera_index(camera_source: Optional[str]) -> int:
    """Return the OpenCV index that best matches *camera_source* from metadata.

    Falls back to 0 if no match is found or *camera_source* is empty.
    """
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
    """Crop *frame* to the normalised ROI [x1,y1,x2,y2]. Returns frame unchanged if roi is None."""
    if not roi or len(roi) != 4:
        return frame
    h, w = frame.shape[:2]
    x1 = max(0, int(roi[0] * w));  y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w));  y2 = min(h, int(roi[3] * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return frame
    return frame[y1:y2, x1:x2]


def composite_overlay(full: np.ndarray, roi_overlay: np.ndarray,
                       roi: list) -> np.ndarray:
    """Paste the roi_overlay back onto *full* at the ROI position."""
    out = full.copy()
    h, w = full.shape[:2]
    x1 = max(0, int(roi[0] * w));  y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w));  y2 = min(h, int(roi[3] * h))
    oh, ow = y2 - y1, x2 - x1
    if oh > 0 and ow > 0:
        resized = cv2.resize(roi_overlay, (ow, oh))
        out[y1:y2, x1:x2] = resized
    return out


def draw_hud(frame: np.ndarray, score: float, threshold: float,
             is_anomaly: bool, event_count: int,
             roi: Optional[list]) -> np.ndarray:
    """Return a copy of *frame* with score bar, status text and alarm banner."""
    out = frame.copy()
    h, w = out.shape[:2]

    # ROI rectangle
    if roi and len(roi) == 4:
        x1 = int(roi[0] * w); y1 = int(roi[1] * h)
        x2 = int(roi[2] * w); y2 = int(roi[3] * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Score bar
    bar_w, bar_h, bar_x = 220, 14, 10
    bar_y = h - 56
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
    fill = int(min(score / max(threshold * 3, 1e-9), 1.0) * bar_w)
    bar_color = (0, 0, 210) if is_anomaly else (0, 170, 0)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    thr_x = bar_x + bar_w // 3   # threshold sits at 1/3 of the 3× display range
    cv2.line(out, (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2), (0, 200, 255), 2)

    # Status text
    txt_color = (0, 0, 210) if is_anomaly else (0, 200, 0)
    state = "ANOMALIE" if is_anomaly else "Normal"
    cv2.putText(out,
                f"Score: {score:.5f}   {state}   Alarme: {event_count}",
                (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 1, cv2.LINE_AA)
    cv2.putText(out,
                f"Schwellwert: {threshold:.5f}   Q / ESC = Beenden",
                (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Red alarm banner at the top
    if is_anomaly:
        cv2.rectangle(out, (0, 0), (w, 32), (0, 0, 180), -1)
        cv2.putText(out, "  ⚠  ANOMALIE ERKANNT", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def save_alarm(frame: np.ndarray, score: float, threshold: float,
               output_dir: str, log_path: str, event_count: int) -> str:
    """Save alarm JPEG and append a row to the CSV log. Returns the filename."""
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
                writer.writerow(["timestamp_utc", "score", "threshold",
                                  "score_pct", "frame_file"])
            pct = int(score / threshold * 100) if threshold > 0 else 0
            writer.writerow([ts.isoformat(), f"{score:.6f}", f"{threshold:.6f}",
                              pct, fname])
    except Exception as exc:
        print(f"\n[Warnung] CSV-Schreibfehler: {exc}")

    print(f"\n[ALARM #{event_count}]  {ts.strftime('%H:%M:%S')}  "
          f"Score={score:.5f}  →  {fname}")
    return fname


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def run_monitor(model_path: str,
                camera_idx_override: Optional[int],
                threshold_override: Optional[float],
                output_dir: str,
                fps: float,
                cooldown: float,
                headless: bool) -> int:
    """Run the monitor loop. Returns the number of alarm events logged."""

    # --- Load model -------------------------------------------------------
    print(f"Lade Modell: {model_path}")
    det = AnomalyDetector()
    try:
        det.load(model_path)
    except Exception as exc:
        print(f"Fehler beim Laden des Modells: {exc}")
        return -1

    meta = det.metadata
    camera_source: str  = meta.get("camera_source", "")
    roi: Optional[list] = meta.get("roi")   # [x1,y1,x2,y2] normalised or None
    threshold: float    = det.threshold

    if threshold_override is not None:
        det.threshold = threshold_override
        threshold = det.threshold

    print(f"  Beschreibung : {meta.get('description', '-')}")
    print(f"  Trainiert    : {meta.get('trained_at', '-')}")
    print(f"  Kamera       : {camera_source or '-'}")
    print(f"  ROI          : {'aktiv ' + str(roi) if roi else 'nein'}")
    print(f"  Schwellwert  : {threshold:.6f}")

    # --- Find camera ------------------------------------------------------
    if camera_idx_override is not None:
        cam_idx = camera_idx_override
        print(f"Kamera: Index {cam_idx} (manuell)")
    else:
        cam_idx = find_camera_index(camera_source)
        print(f"Kamera: Index {cam_idx} (aus Metadaten)")

    # --- Prepare output dir -----------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "monitor_events.csv")

    # --- Shared state protected by a lock ---------------------------------
    latest_frame:   Optional[np.ndarray] = None
    latest_display: Optional[np.ndarray] = None
    score_val:   float = 0.0
    is_anom:     bool  = False
    event_count: int   = 0
    last_alarm_t: float = -cooldown   # allow immediate first alarm
    lock = threading.Lock()

    # --- Frame callback (runs in camera thread) ---------------------------
    def on_frame(frame: np.ndarray) -> None:
        nonlocal latest_frame, latest_display, score_val, is_anom
        nonlocal event_count, last_alarm_t

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

        with lock:
            score_val      = score
            is_anom        = is_anomaly
            latest_frame   = frame.copy()
            latest_display = display

        if headless:
            state = "ANOMALIE" if is_anomaly else "Normal  "
            print(f"\r[{datetime.now(timezone.utc).strftime('%H:%M:%S')}]  "
                  f"Score: {score:.5f}  Thr: {threshold:.5f}  {state}  "
                  f"Alarme: {event_count}   ", end="", flush=True)

        if is_anomaly:
            now = time.perf_counter()
            if now - last_alarm_t >= cooldown:
                last_alarm_t = now
                event_count += 1
                save_alarm(frame, score, threshold, output_dir, log_path, event_count)

    # --- Start camera thread ----------------------------------------------
    cam = _CameraThread(cam_idx, fps, on_frame)
    cam.start()
    time.sleep(0.6)   # give the thread a moment to open the device

    if cam.error:
        print(f"Kamera-Fehler: {cam.error}")
        return -1

    print(f"\nMonitor läuft. Ausgabe → {output_dir}")
    if not headless:
        print("Zum Beenden: Q oder ESC drücken.\n")
    else:
        print("Headless-Modus. Zum Beenden: Strg+C\n")

    # --- Display / event loop ---------------------------------------------
    try:
        if headless:
            while True:
                time.sleep(0.1)
        else:
            win = f"PictureStudio Monitor — {os.path.basename(model_path)}"
            cv2.namedWindow(win, cv2.WINDOW_RESIZABLE)
            while True:
                with lock:
                    disp  = latest_display
                    score = score_val
                    ia    = is_anom
                    ec    = event_count

                if disp is not None:
                    hud = draw_hud(disp, score, threshold, ia, ec, roi)
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
        if headless:
            print()

    print(f"\nMonitor beendet. {event_count} Alarm-Event(s) protokolliert.")
    if event_count > 0:
        print(f"Log-Datei: {log_path}")

    return event_count


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
  python monitor.py --model models/mein_modell.pt
  python monitor.py --model models/mein_modell.pt --camera 1
  python monitor.py --model models/mein_modell.pt --threshold 0.0015
  python monitor.py --model models/mein_modell.pt --headless --output /var/log/anomalien
  python monitor.py --model models/mein_modell.pt --fps 10 --cooldown 30
        """,
    )
    parser.add_argument("--model", required=True, metavar="PFAD",
                        help="Pfad zur trainierten .pt-Modelldatei")
    parser.add_argument("--camera", type=int, default=None, metavar="INDEX",
                        help="Kamera-Index überschreiben (Standard: aus Modell-Metadaten)")
    parser.add_argument("--threshold", type=float, default=None, metavar="WERT",
                        help="Anomalie-Schwellwert überschreiben (Standard: aus Modell)")
    parser.add_argument("--output", default="monitor_logs", metavar="VERZEICHNIS",
                        help="Ausgabeverzeichnis für Logs und Alarm-Bilder (Standard: monitor_logs)")
    parser.add_argument("--fps", type=float, default=15.0, metavar="FPS",
                        help="Bilder pro Sekunde (Standard: 15)")
    parser.add_argument("--cooldown", type=float, default=10.0, metavar="SEK",
                        help="Mindestabstand zwischen Alarm-Saves in Sekunden (Standard: 10)")
    parser.add_argument("--headless", action="store_true",
                        help="Kein Anzeigefenster — nur Terminal-Ausgabe und CSV-Log")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        parser.error(f"Modell-Datei nicht gefunden: {args.model}")

    sys.exit(0 if run_monitor(
        model_path=args.model,
        camera_idx_override=args.camera,
        threshold_override=args.threshold,
        output_dir=args.output,
        fps=args.fps,
        cooldown=args.cooldown,
        headless=args.headless,
    ) >= 0 else 1)


if __name__ == "__main__":
    main()
