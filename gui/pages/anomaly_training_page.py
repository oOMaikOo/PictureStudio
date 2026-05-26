"""
Anomalie-Training-Seite (Stack-Index 17, Video-Modus).

Zeigt einen 3-Schritt-Guide und bietet direkten Einstieg in den
CameraCaptureDialog zum Sammeln von Frames und Trainieren des Autoencoders.
"""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QGroupBox,
)
from PySide6.QtCore import Qt, Signal


class AnomalyTrainingPage(QWidget):
    """Landing-Seite für Anomalie-Training im Video-Workflow (Stack 17)."""

    open_capture_requested = Signal()   # main_window: zu CameraPage + Dialog öffnen

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._camera_page = None
        self._build_ui()

    def set_project(self, project, audit=None) -> None:
        self._refresh_model_status()

    def set_camera_page(self, camera_page) -> None:
        """Referenz auf CameraPage injizieren (nach Konstruktion durch MainWindow)."""
        self._camera_page = camera_page

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_model_status()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        from utils.i18n import tr
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(20)

        # Header
        title = QLabel(tr("anomalytraining.title"))
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #5DADE2;")
        root.addWidget(title)

        subtitle = QLabel(tr("anomalytraining.subtitle"))
        subtitle.setStyleSheet("color: #8B949E; font-size: 12px;")
        root.addWidget(subtitle)

        # 3-step guide
        steps_group = QGroupBox(tr("anomalytraining.steps_group"))
        steps_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #8B949E; border: 1px solid #30363D;"
            " border-radius: 6px; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        steps_layout = QVBoxLayout(steps_group)
        steps_layout.setSpacing(12)

        for num, color, title_key, desc_key in [
            (1, "#388BFD", "anomalytraining.step1_title", "anomalytraining.step1_desc"),
            (2, "#3FB950", "anomalytraining.step2_title", "anomalytraining.step2_desc"),
            (3, "#BC8CFF", "anomalytraining.step3_title", "anomalytraining.step3_desc"),
        ]:
            row = QHBoxLayout()

            badge = QLabel(str(num))
            badge.setFixedSize(28, 28)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background: {color}; color: white; border-radius: 14px;"
                " font-weight: bold; font-size: 13px;"
            )
            row.addWidget(badge)

            text_col = QVBoxLayout()
            text_col.setSpacing(2)
            step_title = QLabel(tr(title_key))
            step_title.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
            step_desc = QLabel(tr(desc_key))
            step_desc.setStyleSheet("color: #8B949E; font-size: 11px;")
            step_desc.setWordWrap(True)
            text_col.addWidget(step_title)
            text_col.addWidget(step_desc)
            row.addLayout(text_col, stretch=1)

            steps_layout.addLayout(row)

        root.addWidget(steps_group)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        train_btn = QPushButton(tr("anomalytraining.open_dialog_btn"))
        train_btn.setFixedHeight(44)
        train_btn.setStyleSheet(
            "QPushButton { background: #1F6FEB; color: white; border-radius: 6px;"
            " font-size: 13px; font-weight: bold; padding: 0 20px; }"
            "QPushButton:hover { background: #388BFD; }"
        )
        train_btn.clicked.connect(self.open_capture_requested)
        btn_row.addWidget(train_btn)

        load_btn = QPushButton(tr("anomalytraining.load_model_btn"))
        load_btn.setFixedHeight(44)
        load_btn.setStyleSheet(
            "QPushButton { background: #21262D; color: #E6EDF3; border: 1px solid #30363D;"
            " border-radius: 6px; font-size: 12px; padding: 0 16px; }"
            "QPushButton:hover { background: #30363D; }"
        )
        load_btn.clicked.connect(self._load_model)
        btn_row.addWidget(load_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Model status card
        status_card = QFrame()
        status_card.setStyleSheet(
            "QFrame { background: #161B22; border: 1px solid #30363D;"
            " border-radius: 8px; }"
        )
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(4)

        status_header = QLabel(tr("anomalytraining.current_model_label"))
        status_header.setStyleSheet("color: #8B949E; font-size: 10px; font-weight: bold;")
        status_layout.addWidget(status_header)

        self._model_status_lbl = QLabel(tr("anomalytraining.no_model"))
        self._model_status_lbl.setStyleSheet("color: #484F58; font-size: 12px;")
        status_layout.addWidget(self._model_status_lbl)

        root.addWidget(status_card)
        root.addStretch()

    # ------------------------------------------------------------------ actions

    def _load_model(self) -> None:
        from utils.i18n import tr
        path, _ = QFileDialog.getOpenFileName(
            self, tr("anomalytraining.load_dialog_title"), "", "Modell (*.pth)"
        )
        if not path or not self._camera_page:
            return
        self._camera_page._load_model_from_path(path)
        self._refresh_model_status()

    def _refresh_model_status(self) -> None:
        from utils.i18n import tr
        if not self._camera_page:
            return
        model_path = getattr(self._camera_page, "_model_path", None)
        detector = getattr(self._camera_page, "_detector", None)
        if model_path and detector:
            name = os.path.basename(model_path)
            thr = detector.threshold
            self._model_status_lbl.setText(tr("anomalytraining.model_info", name=name, thr=thr))
            self._model_status_lbl.setStyleSheet("color: #3FB950; font-size: 12px;")
        else:
            self._model_status_lbl.setText(tr("anomalytraining.no_model"))
            self._model_status_lbl.setStyleSheet("color: #484F58; font-size: 12px;")
