#!/usr/bin/env python3
"""
Picture Studio – Headless Anomaly Monitor Daemon
=================================================
Runs without a GUI.  Reads a monitoring profile exported from the camera
dialog and monitors a video stream in real time.

Usage
-----
    python scripts/monitor_daemon.py --profile /path/to/monitor_profile.json
    python scripts/monitor_daemon.py --help

The daemon:
  1. Opens the camera (USB index or RTSP/HTTP URL).
  2. Loads the autoencoder model (PyTorch .pth or ONNX .onnx).
  3. Scores every N-th frame against the threshold.
  4. On alarm: saves the frame and appends a row to anomaly_events.csv.
  5. Optionally publishes a JSON payload to an MQTT broker.
  6. Prints a one-line status to stdout every 5 seconds.
  7. Stops cleanly on Ctrl+C / SIGTERM.
"""
import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Make project root importable even when called as a script ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


# ── Graceful shutdown ──────────────────────────────────────────────────────

_running = True


def _handle_signal(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── ONNX / PyTorch inference ───────────────────────────────────────────────

def _load_model(model_path: str, model_format: str):
    """Return a callable that takes a (1,1,128,128) float32 ndarray and returns score."""
    if model_format == "onnx" or model_path.endswith(".onnx"):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name

            def _score_onnx(tensor: np.ndarray) -> float:
                recon = sess.run([out_name], {input_name: tensor})[0]
                return float(np.mean((tensor - recon) ** 2))

            print(f"[daemon] ONNX model loaded: {model_path}")
            return _score_onnx
        except ImportError:
            print("[daemon] onnxruntime not installed; falling back to PyTorch for .onnx model")

    # PyTorch fallback
    import torch
    data = torch.load(model_path, map_location="cpu", weights_only=False)
    from core.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    detector.load(model_path)
    print(f"[daemon] PyTorch model loaded: {model_path}")

    def _score_torch(tensor: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.from_numpy(tensor)
            recon = detector._model(t)
            return float(torch.mean((t - recon) ** 2).item())

    return _score_torch


# ── Frame preprocessing ────────────────────────────────────────────────────

def _preprocess(frame: np.ndarray, roi=None, size: int = 128) -> np.ndarray:
    """Crop ROI (optional), resize to size×size, normalise to [0,1] float32."""
    if roi is not None:
        h, w = frame.shape[:2]
        x1 = int(roi[0] * w); y1 = int(roi[1] * h)
        x2 = int(roi[2] * w); y2 = int(roi[3] * h)
        x1, x2 = sorted([max(0, x1), min(w, x2)])
        y1, y2 = sorted([max(0, y1), min(h, y2)])
        if x2 > x1 and y2 > y1:
            frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    arr = small.astype(np.float32) / 255.0
    return arr.reshape(1, 1, size, size)


# ── CSV event logger ───────────────────────────────────────────────────────

class _CsvLogger:
    def __init__(self, path: str):
        self.path = path
        self.count = 0
        exists = os.path.isfile(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if not exists:
            self._w.writerow(["timestamp", "score", "threshold", "score_pct", "frame_path"])
            self._f.flush()

    def log(self, score: float, threshold: float, frame_path: str):
        pct = int(score / threshold * 100) if threshold > 0 else 0
        self._w.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            round(score, 6),
            round(threshold, 6),
            pct,
            frame_path,
        ])
        self._f.flush()
        self.count += 1

    def close(self):
        self._f.close()


# ── MQTT publisher (optional) ──────────────────────────────────────────────

def _make_mqtt_publisher(cfg: dict):
    if not cfg.get("enabled"):
        return None
    try:
        import paho.mqtt.client as mqtt_mod
    except ImportError:
        print("[daemon] paho-mqtt not installed; MQTT disabled")
        return None
    client = mqtt_mod.Client(client_id="picture_daemon")
    username = cfg.get("username", "")
    password = cfg.get("password", "")
    if username:
        client.username_pw_set(username, password)
    topic = cfg.get("topic", "picture_studio/anomaly")
    qos = int(cfg.get("qos", 0))

    try:
        client.connect(cfg.get("host", "localhost"), int(cfg.get("port", 1883)), keepalive=60)
        client.loop_start()
        print(f"[daemon] MQTT connected to {cfg.get('host')}:{cfg.get('port')}")
    except Exception as exc:
        print(f"[daemon] MQTT connect failed: {exc}")
        return None

    def publish(score: float, threshold: float, frame_path: str):
        pct = int(score / threshold * 100) if threshold > 0 else 0
        payload = json.dumps({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "score_pct": pct,
            "alarm": True,
            "frame_path": frame_path,
        })
        client.publish(topic, payload, qos=qos)

    return publish


# ── Main daemon loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Picture Studio headless anomaly monitor daemon"
    )
    parser.add_argument("--profile", required=True, help="Path to monitoring profile JSON")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable anomaly frame saving (overrides profile)")
    args = parser.parse_args()

    # Load profile
    from core.monitoring_profile import load_profile
    try:
        profile = load_profile(args.profile)
    except Exception as exc:
        sys.exit(f"[daemon] Cannot load profile: {exc}")

    model_path = profile.get("model_path", "")
    model_format = profile.get("model_format", "pytorch")
    threshold = float(profile.get("threshold", 0.02))
    camera_source = profile.get("camera_source", 0)
    save_dir = profile.get("save_dir", "/tmp/anomalies")
    smooth_n = int(profile.get("smooth_n", 5))
    roi = profile.get("roi")
    scoring_interval = int(profile.get("scoring_interval", 3))
    save_anomalies = bool(profile.get("save_anomalies", True)) and not args.no_save
    mqtt_cfg = profile.get("mqtt", {})

    if not model_path or not os.path.isfile(model_path):
        sys.exit(f"[daemon] Model not found: {model_path}")

    os.makedirs(save_dir, exist_ok=True)

    # Load model
    try:
        score_fn = _load_model(model_path, model_format)
    except Exception as exc:
        sys.exit(f"[daemon] Cannot load model: {exc}")

    # Open camera
    cam = cv2.VideoCapture(camera_source)
    if not cam.isOpened():
        sys.exit(f"[daemon] Cannot open camera: {camera_source}")
    print(f"[daemon] Camera opened: {camera_source}")

    # Set up logging and MQTT
    log_path = os.path.join(save_dir, "anomaly_events.csv")
    csv_logger = _CsvLogger(log_path)
    mqtt_publish = _make_mqtt_publisher(mqtt_cfg)

    print(f"[daemon] Threshold: {threshold:.5f}  |  Smooth: {smooth_n} frames")
    print(f"[daemon] Saving anomaly frames: {save_anomalies}  →  {save_dir}")
    print(f"[daemon] Event log: {log_path}")
    print("[daemon] Running — press Ctrl+C to stop\n")

    frame_idx = 0
    alarm_streak = 0
    total_frames = 0
    total_alarms = 0
    last_status_time = time.time()
    capture_idx = 0

    while _running:
        ok, frame = cam.read()
        if not ok:
            print("[daemon] Camera read failed; retrying in 1 s…")
            time.sleep(1)
            continue

        total_frames += 1
        frame_idx += 1

        if frame_idx % scoring_interval != 0:
            continue

        # Score
        tensor = _preprocess(frame, roi)
        score = score_fn(tensor)
        is_anomaly = score > threshold
        alarm_streak = (alarm_streak + 1) if is_anomaly else 0
        smoothed_alarm = alarm_streak >= smooth_n

        if smoothed_alarm:
            total_alarms += 1
            frame_path = ""
            if save_anomalies:
                capture_idx += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"anomaly_{ts}_{capture_idx:04d}.png"
                frame_path = os.path.join(save_dir, fname)
                cv2.imwrite(frame_path, frame)
            csv_logger.log(score, threshold, frame_path)
            if mqtt_publish:
                mqtt_publish(score, threshold, frame_path)
            pct = int(score / threshold * 100) if threshold > 0 else 0
            print(f"[ALARM] {datetime.now().isoformat(timespec='seconds')} "
                  f"score={score:.5f} ({pct}%)  frame={frame_path}")

        # Periodic status line
        now = time.time()
        if now - last_status_time >= 5:
            pct = int(score / threshold * 100) if threshold > 0 else 0
            print(f"[status] frames={total_frames}  score={score:.5f} ({pct}%)  "
                  f"alarms={total_alarms}  streak={alarm_streak}")
            last_status_time = now

    cam.release()
    csv_logger.close()
    print(f"\n[daemon] Stopped. Total frames: {total_frames}, alarms: {total_alarms}")


if __name__ == "__main__":
    main()
