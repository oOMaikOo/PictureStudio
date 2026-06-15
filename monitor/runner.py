"""Single- and multi-channel monitor run loops (with model hot-swap)."""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Union

import cv2
import numpy as np

from core.anomaly_detector import AnomalyDetector
from monitor.camera import _CameraThread, _VIDEO_EXTENSIONS, find_camera_index
from monitor.imaging import apply_roi, composite_overlay, draw_hud, save_alarm
from monitor.state import _MonitorState
from monitor.api_server import _MonitorApiServer

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


def run_monitor(
    model_path: str = "",
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
    """Run the monitor loop. Returns alarm event count (≥0), or -1 on error.

    If model_path is empty, starts in collection-only mode: camera runs and
    frames are buffered for GET /api/frames, but no scoring occurs until a
    model is deployed via POST /api/deploy.
    """

    # ── Load model (optional — empty = collection-only mode) ───────────────────
    # det_ref[0] is replaced atomically by the hot-swap check in the main loop.
    det_ref: list = [None]
    roi: Optional[list] = None
    threshold: float = 0.0

    def _load_detector(path: str) -> bool:
        """Load or reload the anomaly detector from path. Returns True on success."""
        nonlocal roi, threshold
        try:
            if path.lower().endswith(".onnx"):
                if not HAS_ORT:
                    print("Fehler: onnxruntime nicht installiert.  pip install onnxruntime")
                    return False
                d = OnnxAnomalyScorer.from_path(path)
                print("  ONNX-Modell geladen (kein PyTorch nötig)")
            else:
                d = AnomalyDetector()
                d.load(path)
            if threshold_override is not None:
                d.threshold = threshold_override
            meta = d.metadata
            roi       = meta.get("roi")
            threshold = d.threshold
            print(f"  Beschreibung : {meta.get('description', '-')}")
            print(f"  Trainiert    : {meta.get('trained_at', '-')}")
            print(f"  ROI          : {'aktiv ' + str(roi) if roi else 'nein'}")
            print(f"  Schwellwert  : {threshold:.6f}")
            det_ref[0] = d
            return True
        except Exception as exc:
            print(f"Fehler beim Laden des Modells: {exc}")
            return False

    if model_path:
        print(f"Lade Modell: {model_path}")
        if not _load_detector(model_path):
            return -1
    else:
        print("Kein Modell angegeben — Sammel-Modus (Frames puffern für GET /api/frames).")
        print("Modell deployen: POST /api/deploy oder PictureStudio → Fleet → Deployen")

    # ── Determine camera source ─────────────────────────────────────────────────
    if camera_source is None:
        if det_ref[0] is not None:
            meta_cam = det_ref[0].metadata.get("camera_source", "")
            cam_source: Union[int, str] = find_camera_index(meta_cam)
            print(f"  Kamera       : Index {cam_source} (aus Modell-Metadaten)")
        else:
            cam_source = 0
            print(f"  Kamera       : Index {cam_source} (Standard)")
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
        if model_path:
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
                time.sleep(0.5)
                status = "verbunden" if mqtt_client.connected else "Verbindung ausstehend"
                print(f"  MQTT         : {mqtt_host}:{mqtt_port}/{mqtt_topic} ({status})")
            else:
                print(f"[Warnung] MQTT-Verbindung fehlgeschlagen: {mqtt_client.last_error}")
                mqtt_client = None

    # ── Shared state ───────────────────────────────────────────────────────────
    state = _MonitorState()
    state.model_name = os.path.basename(model_path) if model_path else "—"
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
            print(f"  Frames-API   : http://localhost:{api_port}/api/frames")
            print(f"  Deploy-API   : http://localhost:{api_port}/api/deploy  (POST)")
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

        # Always buffer frames for GET /api/frames (training data collection)
        state.push_frame(frame)

        det = det_ref[0]
        if det is None:
            # Collection-only mode: store raw frame for display, no scoring
            with frame_lock:
                latest_frame   = frame.copy()
                latest_display = frame.copy()
            return

        cropped = apply_roi(frame, roi)
        try:
            score, _rec, overlay_crop, _bbox = det.score_detailed(cropped)
        except Exception:
            return

        is_anomaly = score > threshold

        display = composite_overlay(frame, overlay_crop, roi) if roi else overlay_crop

        with frame_lock:
            score_val      = score
            is_anom        = is_anomaly
            latest_frame   = frame.copy()
            latest_display = display

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
                cur_model = os.path.basename(state.model_name)
                fname = save_alarm(frame, score, threshold, output_dir, log_path,
                                   state.event_count + 1)
                state.push_alarm(score, threshold, fname)

                fpath = os.path.join(output_dir, fname) if fname else ""

                if notifier:
                    try:
                        notifier.notify(score, threshold, frame_path=fpath,
                                        model_name=cur_model)
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
        if isinstance(cam_source, str) and not cam._is_video:
            pass   # reconnect loop running
        else:
            print(f"Kamera-Fehler: {cam.error}")
            if api_server:
                api_server.stop()
            return -1

    mode_str = "Sammel-Modus (kein Modell)" if det_ref[0] is None else "Anomalie-Erkennung aktiv"
    print(f"\nMonitor läuft ({mode_str}). Ausgabe → {output_dir}")
    if not headless:
        print("Zum Beenden: Q oder ESC drücken.\n")
    else:
        print("Headless-Modus. Zum Beenden: Strg+C\n")

    # ── Display / event loop with hot-swap support ─────────────────────────────
    def _check_hot_swap() -> None:
        """If a new model was POSTed, reload it (called from main loop)."""
        nonlocal threshold
        path = state.pending_model_path
        if not path:
            return
        state.pending_model_path = ""
        print(f"\n[Deploy] Lade neues Modell: {path}")
        if _load_detector(path):
            state.model_name = os.path.basename(path)
            state.threshold  = threshold
            print(f"[Deploy] Modell aktiv: {state.model_name}  Schwellwert: {threshold:.6f}")
        else:
            print("[Deploy] Laden fehlgeschlagen — altes Modell bleibt aktiv.")

    try:
        if headless:
            while True:
                _check_hot_swap()
                time.sleep(0.5)
        else:
            win_title = f"PictureStudio Monitor — {os.path.basename(model_path) if model_path else 'Sammel-Modus'}"
            cv2.namedWindow(win_title, cv2.WINDOW_RESIZABLE)
            while True:
                _check_hot_swap()
                with frame_lock:
                    disp  = latest_display
                    score = score_val
                    ia    = is_anom
                    ec    = state.event_count
                    cs    = state.cam_status

                if disp is not None:
                    hud = draw_hud(disp, score, threshold, ia, ec, roi, cs)
                    cv2.imshow(win_title, hud)

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
