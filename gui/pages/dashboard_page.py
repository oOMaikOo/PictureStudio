"""
Dashboard page: project overview with statistics cards.
"""
import os
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QGroupBox, QPushButton, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor


class StatCard(QFrame):
    """
    A compact statistics card with a large coloured value label and a smaller title.

    Used inside ``DashboardPage`` to display counts such as total images,
    labeled images, ROIs, etc.  The accent colour is applied both to the
    value text and to the top border of the card frame.
    """

    def __init__(self, title: str, value: str = "–", color: str = "#388BFD", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(
            f"StatCard {{ background: #161B22; border-radius: 10px; "
            f"border: 1px solid #30363D; border-top: 3px solid {color}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        self._value_label = QLabel(value)
        self._value_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(26)
        font.setBold(True)
        self._value_label.setFont(font)
        self._value_label.setStyleSheet(f"color: {color}; background: transparent; border: none;")
        layout.addWidget(self._value_label)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #8B949E; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(title_label)

    def set_value(self, value: str) -> None:
        """Update the large value text displayed on the card."""
        self._value_label.setText(str(value))


class DashboardPage(QWidget):
    """
    First page shown after application start and after a project is loaded.

    Displays:
    - Action buttons (new / open project) and a recent-projects list when no
      project is active.
    - Six ``StatCard`` widgets showing dataset statistics once a project is loaded.
    - Last-training metrics (timestamp, accuracy, F1, model name).
    - A per-class distribution bar chart.
    - Contextual warnings (unlabeled images, class imbalance, low sample counts).

    Signals
    -------
    open_project_requested : Emitted when the user clicks "Projekt öffnen".
    new_project_requested  : Emitted when the user clicks "+ Neues Projekt".
    open_recent_requested  : Emitted with the file path when a recent entry is clicked.
    """

    open_project_requested = Signal()
    new_project_requested = Signal()
    open_recent_requested = Signal(str)   # emits file path
    navigate_to_label_requested = Signal(str)  # emits label name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._recent_projects: List[str] = []
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
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #388BFD;")
        layout.addWidget(title)

        # No-project hint
        self._no_project_widget = QWidget()
        np_layout = QVBoxLayout(self._no_project_widget)
        np_layout.setSpacing(16)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        _btn_ss = (
            "QPushButton {{ background:{bg}; color:white; border:none; border-radius:6px;"
            " padding:10px 24px; font-size:13px; font-weight:bold; }}"
            "QPushButton:hover {{ background:{hov}; }}"
        )
        new_btn = QPushButton("+ Neues Projekt")
        new_btn.setStyleSheet(_btn_ss.format(bg="#1F6FEB", hov="#388BFD"))
        new_btn.clicked.connect(self.new_project_requested)
        open_btn = QPushButton("Projekt öffnen…")
        open_btn.setStyleSheet(_btn_ss.format(bg="#21262D", hov="#30363D"))
        open_btn.clicked.connect(self.open_project_requested)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        np_layout.addLayout(btn_row)

        # Recent projects section (shown when there are recents)
        self._recent_group = QGroupBox("Zuletzt geöffnet")
        self._recent_layout = QVBoxLayout(self._recent_group)
        self._recent_layout.setSpacing(4)
        np_layout.addWidget(self._recent_group)
        self._recent_group.hide()

        layout.addWidget(self._no_project_widget)

        # Stats cards grid
        cards_group = QGroupBox("Datensatz-Übersicht")
        cards_grid = QGridLayout(cards_group)
        cards_grid.setSpacing(12)

        self._cards = {
            "total_images":    StatCard("Bilder gesamt",  "–", "#388BFD"),
            "labeled_images":  StatCard("Gelabelt",        "–", "#3FB950"),
            "unlabeled_images":StatCard("Ungelabelt",      "–", "#F85149"),
            "total_rois":      StatCard("ROIs",            "–", "#BC8CFF"),
            "total_labels":    StatCard("Klassen",         "–", "#D29922"),
            "training_runs":   StatCard("Trainingsläufe",  "–", "#39C5CF"),
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
            val_lbl.setStyleSheet("color: #3FB950; font-weight: bold;")
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
        """Accept a new (or newly-loaded) project and refresh the display."""
        self.project = project
        self.refresh()

    def set_recent_projects(self, paths: List[str]) -> None:
        """Update the recent-projects list, filtering out paths that no longer exist."""
        self._recent_projects = [p for p in paths if os.path.exists(p)]
        self._refresh_recent_section()

    def _refresh_recent_section(self) -> None:
        """Rebuild the recent-project button list inside the 'Zuletzt geöffnet' group."""
        # Clear existing buttons
        for i in reversed(range(self._recent_layout.count())):
            w = self._recent_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        if not self._recent_projects:
            self._recent_group.hide()
            return

        self._recent_group.show()
        for path in self._recent_projects[:8]:
            btn = QPushButton(f"  {os.path.basename(path)}")
            btn.setToolTip(path)
            btn.setStyleSheet(
                "QPushButton { text-align:left; background:#161B22; border:1px solid #30363D;"
                " border-radius:4px; padding:6px 10px; color:#ADBAC7; }"
                "QPushButton:hover { background:#1F6FEB; color:white; border-color:#1F6FEB; }"
            )
            btn.clicked.connect(lambda _, p=path: self.open_recent_requested.emit(p))
            self._recent_layout.addWidget(btn)

    def refresh(self) -> None:
        """Re-read all statistics from the project and repopulate every UI section."""
        if not self.project:
            self._no_project_widget.setVisible(True)
            self._refresh_recent_section()
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

        # Class distribution bars (with percentage)
        for i in reversed(range(self._class_layout.count())):
            w = self._class_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        label_counts = data.get("label_counts", {})
        total_labeled = sum(label_counts.values()) or 1
        colors = [info.get("color", "#888") for info in self.project.labels.values()]
        max_count = max(label_counts.values(), default=1) or 1

        for (lbl, cnt), color in zip(label_counts.items(), colors):
            row = QHBoxLayout()
            name_lbl = QLabel(lbl)
            name_lbl.setFixedWidth(140)
            name_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            row.addWidget(name_lbl)

            bar_container = QFrame()
            bar_container.setFixedHeight(18)
            bar_container.setStyleSheet("background: #21262D; border-radius: 3px;")
            bar_container.setMinimumWidth(200)
            bar = QFrame(bar_container)
            fill_w = max(int(cnt / max_count * 200), 4 if cnt > 0 else 0)
            bar.setGeometry(0, 0, fill_w, 18)
            bar.setStyleSheet(f"background: {color}; border-radius: 3px;")
            row.addWidget(bar_container, stretch=1)

            pct = cnt / total_labeled * 100
            cnt_lbl = QLabel(f"  {cnt}  ({pct:.1f}%)")
            cnt_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            cnt_lbl.setFixedWidth(100)
            row.addWidget(cnt_lbl)

            container_w = QWidget()
            container_w.setLayout(row)
            container_w.setCursor(Qt.PointingHandCursor)
            container_w.setToolTip(f"Klick: Beschriftungsseite auf '{lbl}' filtern")
            container_w.mousePressEvent = (
                lambda _e, lb=lbl: self.navigate_to_label_requested.emit(lb)
            )
            self._class_layout.addWidget(container_w)

        # Warnings
        warns = []
        if data["unlabeled_images"] > 0:
            warns.append(f"⚠ {data['unlabeled_images']} Bild(er) noch nicht gelabelt.")
        counts = list(label_counts.values())
        if len(counts) >= 2 and min(counts) > 0:
            ratio = max(counts) / min(counts)
            if ratio > 5:
                warns.append(
                    f"⚠ Klassenungleichgewicht: Ratio {ratio:.1f}:1 — "
                    "'Klassenausgleich' im Training aktivieren."
                )
        from utils.config import MIN_IMAGES_PER_CLASS
        for lbl, cnt in label_counts.items():
            if 0 < cnt < MIN_IMAGES_PER_CLASS:
                warns.append(
                    f"⚠ Klasse '{lbl}' hat nur {cnt} Bilder "
                    f"(Mindestens {MIN_IMAGES_PER_CLASS} empfohlen)."
                )

        self._warn_label.setText("\n".join(warns) if warns else "✓ Keine Warnungen.")
        self._warn_label.setStyleSheet("color: #D29922;" if warns else "color: #3FB950;")
