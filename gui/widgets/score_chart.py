"""
Live anomaly score chart: a compact QPainter line chart for camera monitoring.
"""
from typing import List

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFont

_WINDOW = 120  # max samples shown at once


class ScoreChart(QWidget):
    """
    Displays the last N anomaly scores as a coloured line.
    Scores below threshold are green; scores above are red.
    A dashed orange horizontal line marks the threshold.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scores: List[float] = []
        self._threshold: float = 0.02
        self.setMinimumHeight(72)
        self.setMaximumHeight(90)

    def update_data(self, scores: List[float], threshold: float) -> None:
        """
        Replace the displayed score history and redraw.

        Only the last ``_WINDOW`` scores are kept so the chart never grows
        unbounded. A ``self.update()`` call queues a repaint.
        """
        self._scores = scores[-_WINDOW:]
        self._threshold = threshold
        self.update()

    def paintEvent(self, _) -> None:
        """
        Draw the score chart using QPainter.

        Renders a background fill, a dashed threshold line, and coloured line
        segments connecting consecutive score samples (green below threshold,
        red above). The y-axis is scaled to fit both the data range and the
        threshold with a small margin.
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        pad = 4

        p.fillRect(0, 0, w, h, QColor("#22272E"))

        if not self._scores:
            p.setPen(QColor("#545D68"))
            p.setFont(QFont("", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Keine Daten")
            p.end()
            return

        thr = self._threshold
        lo = min(min(self._scores), thr) * 0.85
        hi = max(max(self._scores), thr) * 1.15
        if hi <= lo:
            hi = lo + 1e-6

        def ypx(v: float) -> int:
            ratio = (v - lo) / (hi - lo)
            return int(h - pad - ratio * (h - 2 * pad))

        # Threshold dashed line
        ty = ypx(thr)
        p.setPen(QPen(QColor("#D29922"), 1, Qt.DashLine))
        p.drawLine(pad, ty, w - pad, ty)

        n = len(self._scores)
        if n < 2:
            p.end()
            return

        x_step = (w - 2 * pad) / (n - 1)
        for i in range(1, n):
            x1 = int(pad + (i - 1) * x_step)
            x2 = int(pad + i * x_step)
            y1 = ypx(self._scores[i - 1])
            y2 = ypx(self._scores[i])
            color = QColor("#F85149") if self._scores[i] > thr else QColor("#3FB950")
            p.setPen(QPen(color, 1.5))
            p.drawLine(x1, y1, x2, y2)

        p.end()
