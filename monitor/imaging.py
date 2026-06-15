"""Frame helpers: ROI crop, overlay compositing, HUD drawing, alarm saving."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np


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
