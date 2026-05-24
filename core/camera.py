"""
Camera capture backend: USB cameras and IP camera streams via OpenCV.
"""
import json
import logging
import platform
import subprocess
import threading
import time
from datetime import datetime
from typing import Union

log = logging.getLogger("ImageLabelingStudio.camera")

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

_CAM_PROP_MAP: dict[str, int] = {
    "brightness":  cv2.CAP_PROP_BRIGHTNESS,
    "contrast":    cv2.CAP_PROP_CONTRAST,
    "saturation":  cv2.CAP_PROP_SATURATION,
    "sharpness":   cv2.CAP_PROP_SHARPNESS,
    "exposure":    cv2.CAP_PROP_EXPOSURE,
    "gain":        cv2.CAP_PROP_GAIN,
}


def apply_cam_props(cap: cv2.VideoCapture, props: dict) -> None:
    """Apply camera property values to an open VideoCapture (silently ignores unsupported props)."""
    for name, value in props.items():
        prop_id = _CAM_PROP_MAP.get(name)
        if prop_id is not None:
            cap.set(prop_id, value)


def apply_frame_filter(frame: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Apply a named preprocessing filter and return a BGR frame.
    filter_name: "none" | "grayscale" | "canny" | "sobel" | "laplacian"
    """
    if filter_name == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if filter_name == "canny":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if filter_name == "sobel":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
        return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    if filter_name == "laplacian":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return frame  # "none" oder unbekannt

# On macOS, CAP_ANY automatically selects AVFoundation — do NOT pass
# CAP_AVFOUNDATION explicitly as OpenCV 4.x rejects integer-index captures with it.
_BACKEND = cv2.CAP_ANY


def _macos_avfoundation_names() -> list[str]:
    """
    Return camera display names in the exact order OpenCV uses on macOS,
    with mobile/continuity cameras (iPhone, iPad) already excluded.

    Observed OpenCV ordering on macOS:
      1. Real USB cameras  — uniqueID starts with "0x" (USB bus address)
      2. Built-in cameras  — deviceType == .builtInWideAngleCamera (FaceTime etc.)
      3. Virtual/mobile    — uniqueID is UUID format → EXCLUDED

    The Swift script classifies each device by uniqueID prefix and deviceType,
    then outputs only groups 1 + 2 in that order (= OpenCV's index order).
    system_profiler is used as a best-effort fallback when Swift is unavailable.
    """
    # --- Swift (definitive) ---
    _swift_script = (
        "import AVFoundation;"
        "let s=AVCaptureDevice.DiscoverySession("
        "deviceTypes:[.builtInWideAngleCamera,.external,.continuityCamera],"
        "mediaType:.video,position:.unspecified);"
        "var usb=[String](),bi=[String]();"
        "for d in s.devices{"
        # Built-in (FaceTime etc.) → group 2
        "if d.deviceType == .builtInWideAngleCamera{bi.append(d.localizedName)}"
        # Real USB devices have hex bus-address uniqueIDs like "0x1110000046d0828"
        # Mobile/virtual devices have UUID-format uniqueIDs — skip them
        "else if d.uniqueID.lowercased().hasPrefix(\"0x\"){usb.append(d.localizedName)}"
        "};"
        # Output: USB cameras first, then built-in — matches OpenCV's index order
        "for n in usb+bi{print(n)}"
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
    except Exception as exc:
        log.debug("Swift-Kamera-Auflistung fehlgeschlagen: %s", exc)

    # --- system_profiler fallback (ordering may not match OpenCV — best effort) ---
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=6,
        )
        data = json.loads(out.stdout)
        return [cam.get("_name", "") for cam in data.get("SPCameraDataType", [])]
    except Exception as exc:
        log.debug("system_profiler-Fallback fehlgeschlagen: %s", exc)
        return []


def list_usb_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """
    Return (index, label) pairs for USB and built-in cameras OpenCV can open.
    Mobile/continuity cameras (iPhone, iPad) are excluded.

    On macOS, OpenCV always assigns mobile/virtual cameras to the HIGHEST indices
    (after all real USB and built-in devices). When Swift provides a filtered name
    list (USB + built-in only), any OpenCV index beyond that list is a mobile device
    and is skipped without even opening it.
    """
    # names[i] → display name for OpenCV index i.
    # When Swift succeeds, this list already excludes mobile cameras and is in
    # the correct OpenCV order: USB cameras first, then built-in.
    names: list[str] = _macos_avfoundation_names() if platform.system() == "Darwin" else []

    found = []
    for i in range(max_index):
        name = names[i] if i < len(names) else ""

        # If Swift provided a non-empty list and this index has no entry, it is a
        # mobile/virtual device at the tail of OpenCV's enumeration — skip it.
        if names and not name:
            continue

        try:
            cap = cv2.VideoCapture(i, _BACKEND)
            if not cap.isOpened():
                cap.release()
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            res = f" ({w}×{h})" if w > 0 and h > 0 else ""
            label = f"{name}{res}" if name else f"Kamera {i}{res}"
            found.append((i, label))
        except Exception as exc:
            log.debug("Kamera-Index %d nicht öffenbar: %s", i, exc)
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
        self._pending_props: dict = {}
        self._props_lock = threading.Lock()

    def set_cam_props(self, props: dict) -> None:
        """Queue camera property updates; applied on the next frame loop tick."""
        with self._props_lock:
            self._pending_props.update(props)

    def run(self) -> None:
        """Main thread loop: open the device, emit frames, then release on exit."""
        self._running = True
        backend = _BACKEND if isinstance(self.source, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.source, backend)
        if not cap.isOpened():
            self.error.emit(f"Kamera konnte nicht geöffnet werden: {self.source}")
            return

        with self._props_lock:
            pending = dict(self._pending_props)
            self._pending_props.clear()
        if pending:
            apply_cam_props(cap, pending)

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

            _consec_fail = 0
            while self._running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    _consec_fail += 1
                    if _consec_fail >= 5:
                        if self._running:
                            self.error.emit("Kein Bild empfangen – Verbindung unterbrochen.")
                        break
                    time.sleep(0.1)
                    continue
                _consec_fail = 0
                self.frame_ready.emit(frame)
                with self._props_lock:
                    pending = dict(self._pending_props)
                    self._pending_props.clear()
                if pending:
                    apply_cam_props(cap, pending)
                wait = self._frame_interval - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
        finally:
            cap.release()

    def stop(self) -> None:
        """Request the thread to stop and wait up to 3 seconds for it to finish."""
        self._running = False
        self.wait(3000)
