"""
Dashboard page: project overview with statistics cards.
"""
import os
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QGroupBox, QPushButton, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor


class StatCard(QFrame):
    """Single statistics card widget."""

    def __init__(self, title: str, value: str = "–", color: str = "#3498DB", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(
            f"StatCard {{ background: #16213e; border-radius: 8px; "
            f"border: 2px solid {color}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        self._value_label = QLabel(value)
        self._value_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self._value_label.setFont(font)
        self._value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self._value_label)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(title_label)

    def set_value(self, value: str) -> None:
        self._value_label.setText(str(value))


class DashboardPage(QWidget):
    """Project overview page."""

    open_project_requested = Signal()
    new_project_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        # Title
        title = QLabel("Projekt-Dashboard")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #3498DB;")
        layout.addWidget(title)

        # No-project hint
        self._no_project_widget = QWidget()
        np_layout = QHBoxLayout(self._no_project_widget)
        np_layout.addStretch()
        new_btn = QPushButton("Neues Projekt erstellen")
        new_btn.setStyleSheet("padding:10px 20px; font-size:13px; background:#2ECC71; color:white;")
        new_btn.clicked.connect(self.new_project_requested)
        open_btn = QPushButton("Projekt öffnen")
        open_btn.setStyleSheet("padding:10px 20px; font-size:13px; background:#3498DB; color:white;")
        open_btn.clicked.connect(self.open_project_requested)
        np_layout.addWidget(new_btn)
        np_layout.addWidget(open_btn)
        np_layout.addStretch()
        layout.addWidget(self._no_project_widget)

        # Stats cards grid
        cards_group = QGroupBox("Datensatz-Übersicht")
        cards_grid = QGridLayout(cards_group)
        cards_grid.setSpacing(12)

        self._cards = {
            "total_images":    StatCard("Bilder gesamt",       "–", "#3498DB"),
            "labeled_images":  StatCard("Gelabelt",             "–", "#2ECC71"),
            "unlabeled_images":StatCard("Ungelabelt",           "–", "#E74C3C"),
            "total_rois":      StatCard("ROIs",                 "–", "#9B59B6"),
            "total_labels":    StatCard("Klassen",              "–", "#F39C12"),
            "training_runs":   StatCard("Trainingsläufe",       "–", "#1ABC9C"),
        }
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for (r, c), card in zip(positions, self._cards.values()):
            cards_grid.addWidget(card, r, c)
        layout.addWidget(cards_group)

        # Last training
        last_group = QGroupBox("Letztes Training")
        last_layout = QGridLayout(last_group)

        self._last_labels = {}
        for row, (key, label) in enumerate([
            ("last_run_ts",       "Zeitstempel"),
            ("last_run_accuracy", "Accuracy"),
            ("last_run_f1",       "F1-Score (Macro)"),
            ("current_model",     "Aktuelles Modell"),
        ]):
            last_layout.addWidget(QLabel(label + ":"), row, 0)
            val_lbl = QLabel("–")
            val_lbl.setStyleSheet("color: #2ECC71; font-weight: bold;")
            last_layout.addWidget(val_lbl, row, 1)
            self._last_labels[key] = val_lbl
        layout.addWidget(last_group)

        # Class distribution
        self._class_group = QGroupBox("Klassenverteilung")
        self._class_layout = QVBoxLayout(self._class_group)
        layout.addWidget(self._class_group)

        # Warnings
        self._warn_group = QGroupBox("Warnungen / Hinweise")
        self._warn_layout = QVBoxLayout(self._warn_group)
        self._warn_label = QLabel("Keine Warnungen.")
        self._warn_label.setWordWrap(True)
        self._warn_layout.addWidget(self._warn_label)
        layout.addWidget(self._warn_group)

        layout.addStretch()

    def set_project(self, project) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        if not self.project:
            self._no_project_widget.setVisible(True)
            return
        self._no_project_widget.setVisible(False)
        data = self.project.get_dashboard_data()

        # Cards
        self._cards["total_images"].set_value(str(data["total_images"]))
        self._cards["labeled_images"].set_value(str(data["labeled_images"]))
        self._cards["unlabeled_images"].set_value(str(data["unlabeled_images"]))
        self._cards["total_rois"].set_value(str(data["total_rois"]))
        self._cards["total_labels"].set_value(str(data["total_labels"]))
        self._cards["training_runs"].set_value(str(data["training_runs"]))

        # Last training
        ts = data.get("last_run_ts", "")[:19].replace("T", " ")
        acc = data.get("last_run_accuracy", 0)
        f1 = data.get("last_run_f1", 0)
        self._last_labels["last_run_ts"].setText(ts or "–")
        self._last_labels["last_run_accuracy"].setText(f"{acc*100:.2f}%" if acc else "–")
        self._last_labels["last_run_f1"].setText(f"{f1*100:.2f}%" if f1 else "–")
        self._last_labels["current_model"].setText(data.get("current_model", "") or "–")

        # Class distribution bars
        for i in reversed(range(self._class_layout.count())):
            self._class_layout.itemAt(i).widget().deleteLater()

        label_counts = data.get("label_counts", {})
        total_labeled = sum(label_counts.values()) or 1
        colors = [info.get("color", "#888") for info in self.project.labels.values()]

        for (lbl, cnt), color in zip(label_counts.items(), colors):
            row = QHBoxLayout()
            name_lbl = QLabel(f"{lbl}")
            name_lbl.setFixedWidth(150)
            name_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            row.addWidget(name_lbl)
            bar = QFrame()
            pct = int(cnt / total_labeled * 300)
            bar.setFixedSize(max(pct, 4), 18)
            bar.setStyleSheet(f"background: {color}; border-radius: 3px;")
            row.addWidget(bar)
            cnt_lbl = QLabel(f" {cnt}")
            cnt_lbl.setStyleSheet("color: #aaa;")
            row.addWidget(cnt_lbl)
            row.addStretch()
            container_w = QWidget()
            container_w.setLayout(row)
            self._class_layout.addWidget(container_w)

        # Warnings
        warns = []
        if data["unlabeled_images"] > 0:
            warns.append(f"⚠ {data['unlabeled_images']} Bild(er) noch nicht gelabelt.")
        counts = list(label_counts.values())
        if len(counts) >= 2 and min(counts) > 0:
            ratio = max(counts) / min(counts)
            if ratio > 5:
                warns.append(f"⚠ Klassenungleichgewicht: Ratio {ratio:.1f}:1. Augmentation empfohlen.")
        from utils.config import MIN_IMAGES_PER_CLASS
        for lbl, cnt in label_counts.items():
            if 0 < cnt < MIN_IMAGES_PER_CLASS:
                warns.append(f"⚠ Klasse '{lbl}' hat nur {cnt} Bilder (Mindestens {MIN_IMAGES_PER_CLASS} empfohlen).")

        self._warn_label.setText("\n".join(warns) if warns else "✓ Keine Warnungen.")
        self._warn_label.setStyleSheet("color: #F39C12;" if warns else "color: #2ECC71;")
