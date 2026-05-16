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

# Use AVFoundation explicitly on macOS — required for Continuity Camera (iPhone)
# and for correct device naming via system_profiler.
_BACKEND = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def _macos_camera_names() -> list[str]:
    """Return camera display names in AVFoundation order via system_profiler."""
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
    """Return (index, label) pairs for every responsive camera device."""
    sys_names = _macos_camera_names() if platform.system() == "Darwin" else []
    found = []
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, _BACKEND)
            if cap.isOpened():
                # Try reading one frame to confirm the device is actually usable
                ret, _ = cap.read()
                cap.release()
                if not ret:
                    continue
                name = sys_names[i] if i < len(sys_names) and sys_names[i] else f"Kamera {i}"
                found.append((i, name))
            else:
                cap.release()
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
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class CameraFrameThread(QThread):
    """Background thread that continuously reads frames from a camera source."""

    frame_ready = Signal(object)   # numpy BGR array
    error = Signal(str)

    def __init__(self, source: Union[int, str], fps: float = 15.0, parent=None):
        super().__init__(parent)
        self.source = source
        self._frame_interval = 1.0 / max(fps, 1.0)
        self._running = False

    def run(self) -> None:
        self._running = True
        backend = _BACKEND if isinstance(self.source, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.source, backend)
        if not cap.isOpened():
            self.error.emit(f"Kamera konnte nicht geöffnet werden: {self.source}")
            return

        try:
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
        self._running = False
        self.wait(3000)
