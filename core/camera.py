"""
Camera capture backend: USB cameras and IP camera streams via OpenCV.
"""
import time
from typing import Union

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def list_usb_cameras(max_index: int = 8) -> list[tuple[int, str]]:
    """Return (index, label) pairs for every responsive USB camera."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append((i, f"USB Kamera {i}"))
            cap.release()
    return found


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
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error.emit(f"Kamera konnte nicht geöffnet werden: {self.source}")
            return

        while self._running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if self._running:
                    self.error.emit("Kein Bild empfangen – Verbindung unterbrochen.")
                break
            self.frame_ready.emit(frame)
            elapsed = time.perf_counter() - t0
            wait = self._frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)

        cap.release()

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
