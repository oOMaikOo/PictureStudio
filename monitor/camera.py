"""Camera discovery, terminal selection, and the frame-capture thread."""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional, Union

import cv2
import numpy as np

from core.camera import list_usb_cameras


_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv', '.webm', '.ts'}


def _discover_cameras() -> list:
    """Scan for available USB cameras and return [(index, label)] list."""
    try:
        cams = list_usb_cameras()
        return cams if cams else []
    except Exception as exc:
        log.warning("Kamera-Suche fehlgeschlagen: %s", exc)
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

            # macOS AVFoundation: after open() the sensor needs several seconds to
            # adjust exposure — frames arrive with ret=True but are pure black.
            # Sleep briefly, then discard dark frames for up to 5 s.
            # Require 3 consecutive bright frames so we don't stop on a fluke.
            if (not self._is_video and isinstance(self._source, int)
                    and os.uname().sysname == "Darwin"):
                time.sleep(0.8)  # let AVFoundation session stabilise
                bright = 0
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    ret_w, f_w = cap.read()
                    if ret_w and f_w is not None:
                        m = float(f_w.mean())
                        if m > 5.0:
                            bright += 1
                            if bright >= 3:
                                print(f"[Kamera] Index {self._source}: Sensor bereit "
                                      f"(Helligkeit {m:.1f})")
                                break
                        else:
                            bright = 0
                    time.sleep(0.04)
                else:
                    print(f"[Kamera] Warnung: Index {self._source} liefert nach 5 s "
                          f"noch schwarze Frames — prüfe Kamera-Berechtigung für Terminal")

            consec_fail = 0
            first_frame = True
            # Live cameras (especially built-in on macOS) need up to 30 read()
            # calls before delivering the first frame — don't break too early.
            warmup_limit = 3 if self._is_video else 60
            t_next = time.perf_counter()

            try:
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
                        except Exception as _cb_exc:
                            import logging as _logging
                            _logging.getLogger(__name__).warning(
                                "Frame-Callback-Fehler: %s", _cb_exc
                            )
                    else:
                        consec_fail += 1
                        if consec_fail >= warmup_limit:
                            print(f"[Kamera] Quelle {self._source}: {consec_fail} "
                                  f"aufeinanderfolgende Fehler — trenne Verbindung.")
                            break
                        time.sleep(0.1)
            finally:
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


def find_camera_index(camera_source: Optional[str]) -> int:
    if not camera_source:
        return 0
    try:
        cameras = list_usb_cameras()
    except Exception as exc:
        log.warning("Kamera-Suche fehlgeschlagen: %s", exc)
        cameras = []
    for idx, label in cameras:
        if camera_source in label or label in camera_source:
            return idx
    print(f"[Warnung] Kamera '{camera_source}' nicht gefunden — verwende Index 0.")
    return 0
