"""
Camera capture backend: USB cameras and IP camera streams via OpenCV.
"""
import json
import platform
import subprocess
import time
from datetime import datetime
from typing import Union

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

# On macOS, CAP_ANY automatically selects AVFoundation — do NOT pass
# CAP_AVFOUNDATION explicitly as OpenCV 4.x rejects integer-index captures with it.
_BACKEND = cv2.CAP_ANY


def _macos_avfoundation_names() -> list[str]:
    """
    Return camera display names in the same index order that OpenCV uses on macOS.

    OpenCV orders cameras as: external/non-builtin cameras first (in AVFoundation
    enumeration order), then built-in cameras (FaceTime etc.) last.
    This matches `.builtInWideAngleCamera` always coming after `.external` devices
    in OpenCV's internal AVCaptureDeviceDiscoverySession query.

    Strategy:
    1. Swift — queries DiscoverySession, splits into external vs built-in, outputs
       external first then built-in (= OpenCV order).
    2. system_profiler — plain listing order as fallback (no reversal).
    """
    # --- Swift (most reliable — matches OpenCV's internal enumeration order) ---
    _swift_script = (
        "import AVFoundation;"
        "let s=AVCaptureDevice.DiscoverySession("
        "deviceTypes:[.builtInWideAngleCamera,.external,.continuityCamera],"
        "mediaType:.video,position:.unspecified);"
        "var ext=[String](),bi=[String]();"
        "for d in s.devices{"
        "if d.deviceType == .builtInWideAngleCamera{bi.append(d.localizedName)}"
        "else{ext.append(d.localizedName)}};"
        "for n in ext+bi{print(n)}"
    )
    try:
        r = subprocess.run(
            ["swift", "-e", _swift_script],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            names = [l.strip() for l in r.stdout.splitlines() if l.strip()]
            if names:
                return names
    except Exception:
        pass

    # --- system_profiler fallback ---
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=6,
        )
        data = json.loads(out.stdout)
        return [cam.get("_name", "") for cam in data.get("SPCameraDataType", [])]
    except Exception:
        return []


def list_usb_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """
    Return (index, label) pairs for every camera device OpenCV can open.

    All cameras are included — no device type is excluded.  The label is the
    human-readable name from AVFoundation when available, otherwise "Kamera N".
    """
    # names[i] is the display name for OpenCV index i (macOS only).
    names: list[str] = _macos_avfoundation_names() if platform.system() == "Darwin" else []

    found = []
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, _BACKEND)
            if not cap.isOpened():
                cap.release()
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            res = f" ({w}×{h})" if w > 0 and h > 0 else ""
            name = names[i] if i < len(names) else ""
            label = f"{name}{res}" if name else f"Kamera {i}{res}"
            found.append((i, label))
        except Exception:
            pass
    return found


def apply_timestamp(frame: np.ndarray) -> np.ndarray:
    """Return a copy of frame with current date/time burned into the bottom-left."""
    out = frame.copy()
    text = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 1280)
    thickness = max(1, int(scale * 1.5))
    _, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 10, h - 10 - baseline
    cv2.putText(out, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def frame_to_qimage(bgr: np.ndarray) -> QImage:
    """
    Convert a BGR numpy frame to a QImage suitable for display in Qt widgets.

    Returns a deep copy so the numpy buffer can be freed without corrupting the QImage.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # QImage does not own the buffer — .copy() makes it self-contained
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class CameraFrameThread(QThread):
    """
    Background QThread that continuously reads frames from a camera source and
    emits them as numpy BGR arrays.

    Signals
    -------
    frame_ready(np.ndarray) : Emitted for every successfully captured frame.
    error(str)              : Emitted on open/read failure; thread then exits.
    """

    frame_ready = Signal(object)   # numpy BGR array
    error = Signal(str)

    def __init__(self, source: Union[int, str], fps: float = 15.0, parent=None):
        """
        Parameters
        ----------
        source : Integer USB device index or RTSP/HTTP URL string.
        fps    : Target frame rate; the thread sleeps between reads to match it.
        """
        super().__init__(parent)
        self.source = source
        self._frame_interval = 1.0 / max(fps, 1.0)
        self._running = False

    def run(self) -> None:
        """Main thread loop: open the device, emit frames, then release on exit."""
        self._running = True
        backend = _BACKEND if isinstance(self.source, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.source, backend)
        if not cap.isOpened():
            self.error.emit(f"Kamera konnte nicht geöffnet werden: {self.source}")
            return

        try:
            # Some devices (e.g. iPhone Continuity Camera) need up to ~1 s to
            # deliver the first frame after open() — retry before giving up.
            ret, frame = False, None
            for _ in range(30):
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.05)

            if not ret:
                self.error.emit(
                    "Kamera geöffnet, aber kein Bild empfangen.\n"
                    "Bitte prüfen ob das Gerät bereit ist und Kamera-Zugriff erteilt wurde."
                )
                return

            # emit the already-read first frame, then continue normally
            self.frame_ready.emit(frame)

            while self._running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    if self._running:
                        self.error.emit("Kein Bild empfangen – Verbindung unterbrochen.")
                    break
                self.frame_ready.emit(frame)
                wait = self._frame_interval - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
        finally:
            cap.release()

    def stop(self) -> None:
        """Request the thread to stop and wait up to 3 seconds for it to finish."""
        self._running = False
        self.wait(3000)
