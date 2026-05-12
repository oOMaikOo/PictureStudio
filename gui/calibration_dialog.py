"""
Threshold calibration dialog for anomaly detection.
Shows score distribution histogram and suggests optimal thresholds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QTextEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class ThresholdCalibrationDialog(QDialog):
    """
    Displays a score-distribution histogram and helps the user pick a threshold.
    After accept(), `selected_threshold` holds the chosen value.
    """

    def __init__(self, scores: list[float], threshold: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Schwellwert kalibrieren")
        self.setMinimumSize(700, 520)
        self.selected_threshold: Optional[float] = None
        self._scores = np.array(scores, dtype=np.float32)
        self._threshold = threshold
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Chart area
        if HAS_MPL:
            self._fig = Figure(figsize=(7, 3.5), facecolor="#1a1a2e")
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(260)
            root.addWidget(self._canvas)
        else:
            self._ascii_box = QTextEdit()
            self._ascii_box.setReadOnly(True)
            self._ascii_box.setFont(QFont("Courier New", 9))
            root.addWidget(self._ascii_box)

        # Statistics
        stats_grp = QGroupBox("Statistik")
        sl = QHBoxLayout(stats_grp)
        self._stat_lbl = QLabel()
        self._stat_lbl.setWordWrap(True)
        self._stat_lbl.setStyleSheet("font-size:11px; color:#BDC3C7;")
        sl.addWidget(self._stat_lbl)
        root.addWidget(stats_grp)

        # Suggested thresholds
        sug_grp = QGroupBox("Vorschläge")
        sg = QHBoxLayout(sug_grp)
        self._sug_btns: list[QPushButton] = []
        for label, attr in [
            ("µ+1σ", "p1"), ("µ+2σ", "p2"), ("µ+2.5σ", "p25"),
            ("µ+3σ", "p3"), ("95. Pz.", "p95"), ("99. Pz.", "p99"),
        ]:
            btn = QPushButton(label)
            btn.setProperty("_val_attr", attr)
            btn.setStyleSheet(
                "background:#1565C0;color:white;padding:4px 10px;border-radius:4px;"
            )
            btn.clicked.connect(self._on_suggestion)
            self._sug_btns.append(btn)
            sg.addWidget(btn)
        root.addWidget(sug_grp)

        # Manual threshold
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Schwellwert:"))
        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setRange(0.00001, 1.0)
        self._thr_spin.setDecimals(6)
        self._thr_spin.setSingleStep(0.001)
        self._thr_spin.setValue(self._threshold)
        self._thr_spin.valueChanged.connect(self._on_manual_change)
        thr_row.addWidget(self._thr_spin)
        self._alarm_pct_lbl = QLabel("")
        self._alarm_pct_lbl.setStyleSheet("color:#E74C3C; font-size:11px;")
        thr_row.addWidget(self._alarm_pct_lbl)
        thr_row.addStretch()
        root.addLayout(thr_row)

        # Buttons
        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Schwellwert übernehmen")
        apply_btn.setStyleSheet(
            "background:#2ECC71;color:white;font-weight:bold;padding:7px;border-radius:4px;"
        )
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    def _populate(self) -> None:
        s = self._scores
        if len(s) == 0:
            return

        mu, sigma = float(s.mean()), float(s.std())
        self._suggestions = {
            "p1":  mu + 1.0 * sigma,
            "p2":  mu + 2.0 * sigma,
            "p25": mu + 2.5 * sigma,
            "p3":  mu + 3.0 * sigma,
            "p95": float(np.percentile(s, 95)),
            "p99": float(np.percentile(s, 99)),
        }

        labels = {"p1": "µ+1σ", "p2": "µ+2σ", "p25": "µ+2.5σ",
                  "p3": "µ+3σ", "p95": "95. Pz.", "p99": "99. Pz."}
        for btn in self._sug_btns:
            attr = btn.property("_val_attr")
            val = self._suggestions.get(attr, 0)
            btn.setText(f"{labels[attr]}\n{val:.5f}")
            btn.setProperty("_val", val)

        stat_txt = (
            f"n={len(s)}   µ={mu:.5f}   σ={sigma:.5f}   "
            f"min={float(s.min()):.5f}   max={float(s.max()):.5f}"
        )
        self._stat_lbl.setText(stat_txt)

        if HAS_MPL:
            self._draw_mpl()
        else:
            self._draw_ascii()

        self._update_alarm_pct(self._threshold)

    def _draw_mpl(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        ax.hist(self._scores, bins=40, color="#3498DB", alpha=0.8, label="Scores")
        ax.axvline(self._threshold, color="#E74C3C", linewidth=2, label=f"Schwellwert ({self._threshold:.5f})")
        ax.set_xlabel("Rekonstruktionsfehler (MSE)", color="#aaa")
        ax.set_ylabel("Häufigkeit", color="#aaa")
        ax.set_title("Score-Verteilung", color="#eee")
        ax.legend(facecolor="#16213e", labelcolor="#eee", fontsize=9)
        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()

    def _draw_ascii(self) -> None:
        hist, edges = np.histogram(self._scores, bins=20)
        peak = max(hist) or 1
        lines = ["Score-Verteilung (ASCII):\n"]
        for count, edge in zip(hist, edges):
            bar = "█" * int(30 * count / peak)
            marker = " ◄ Schwellwert" if edges[0] <= self._threshold <= edge else ""
            lines.append(f"  {edge:.5f}  {bar}{marker}")
        self._ascii_box.setPlainText("\n".join(lines))

    def _update_alarm_pct(self, thr: float) -> None:
        if len(self._scores) == 0:
            return
        pct = (self._scores > thr).mean() * 100
        self._alarm_pct_lbl.setText(f"→ {pct:.1f}% aller Frames würden Alarm auslösen")
        if HAS_MPL:
            for line in self._fig.axes[0].lines:
                if "Schwellwert" in (line.get_label() or ""):
                    line.set_xdata([thr, thr])
                    line.set_label(f"Schwellwert ({thr:.5f})")
            ax = self._fig.axes[0]
            ax.legend(facecolor="#16213e", labelcolor="#eee", fontsize=9)
            self._canvas.draw_idle()

    def _on_suggestion(self) -> None:
        val = self.sender().property("_val")
        if val is not None:
            self._thr_spin.setValue(float(val))

    def _on_manual_change(self, val: float) -> None:
        self._update_alarm_pct(val)

    def _apply(self) -> None:
        self.selected_threshold = self._thr_spin.value()
        self.accept()
