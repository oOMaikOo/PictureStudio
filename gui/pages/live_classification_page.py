"""
Live classification page: run the trained image classifier on a live camera or
video stream in real time (top-k overlay), and capture interesting frames back
into the project dataset — closing the loop video → classification → labeling.
"""
import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QSpinBox, QLineEdit, QFileDialog, QMessageBox, QSizePolicy,
    QCheckBox,
)

from core.camera import CameraFrameThread, list_usb_cameras
from core.inference import Inferencer
from utils.i18n import tr


class LiveClassificationPage(QWidget):
    """Classify a live video stream with the project's image classifier."""

    # Emitted with the number of frames added to the project (loop back to Kern 1).
    images_added = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.project = None
        self._inferencer = Inferencer()
        self._camera_thread: Optional[CameraFrameThread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_result: Optional[dict] = None
        self._frame_counter = 0
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ── Left control panel ────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(300)
        pl = QVBoxLayout(panel)

        title = QLabel(tr("liveclass.title"))
        title.setStyleSheet("font-size:16px;font-weight:bold;")
        pl.addWidget(title)

        # Model
        model_grp = QGroupBox(tr("liveclass.model_group"))
        ml = QVBoxLayout(model_grp)
        self._model_lbl = QLabel(tr("liveclass.no_model"))
        self._model_lbl.setWordWrap(True)
        self._model_lbl.setStyleSheet("color:#7F8C8D;font-size:11px;")
        ml.addWidget(self._model_lbl)
        load_btn = QPushButton(tr("liveclass.load_model_btn"))
        load_btn.clicked.connect(self._load_model)
        ml.addWidget(load_btn)
        pl.addWidget(model_grp)

        # Camera
        cam_grp = QGroupBox(tr("liveclass.camera_group"))
        cl = QVBoxLayout(cam_grp)
        self._cam_combo = QComboBox()
        self._cam_combo.addItem(tr("liveclass.cam_placeholder"), userData=None)
        cl.addWidget(self._cam_combo)
        scan_btn = QPushButton(tr("liveclass.scan_btn"))
        scan_btn.clicked.connect(self._scan_cameras)
        cl.addWidget(scan_btn)
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText(tr("liveclass.url_placeholder"))
        cl.addWidget(self._url_edit)
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton(tr("liveclass.start_btn"))
        self._start_btn.clicked.connect(self._start)
        btn_row.addWidget(self._start_btn)
        self._stop_btn = QPushButton(tr("liveclass.stop_btn"))
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._stop_btn)
        cl.addLayout(btn_row)
        pl.addWidget(cam_grp)

        # Interval
        int_row = QHBoxLayout()
        int_row.addWidget(QLabel(tr("liveclass.interval_label")))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 30)
        self._interval_spin.setValue(5)
        int_row.addWidget(self._interval_spin)
        int_row.addStretch()
        pl.addLayout(int_row)

        # Minimum confidence: below this a prediction is shown as "uncertain"
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel(tr("liveclass.minconf_label")))
        self._minconf_spin = QSpinBox()
        self._minconf_spin.setRange(0, 100)
        self._minconf_spin.setValue(50)
        self._minconf_spin.setSuffix(" %")
        conf_row.addWidget(self._minconf_spin)
        conf_row.addStretch()
        pl.addLayout(conf_row)

        # Optional ROI: crop the frame before classification (percent of frame)
        roi_grp = QGroupBox(tr("liveclass.roi_group"))
        rl = QVBoxLayout(roi_grp)
        self._roi_cb = QCheckBox(tr("liveclass.roi_enable"))
        self._roi_cb.toggled.connect(self._on_roi_toggled)
        rl.addWidget(self._roi_cb)
        grid = QHBoxLayout()
        self._roi_spins = {}
        for key, default in (("x1", 10), ("y1", 10), ("x2", 90), ("y2", 90)):
            sp = QSpinBox()
            sp.setRange(0, 100)
            sp.setValue(default)
            sp.setSuffix("%")
            sp.setEnabled(False)
            sp.setMaximumWidth(62)
            self._roi_spins[key] = sp
            grid.addWidget(sp)
        rl.addLayout(grid)
        pl.addWidget(roi_grp)

        # Capture back to dataset
        self._capture_btn = QPushButton(tr("liveclass.capture_btn"))
        self._capture_btn.clicked.connect(self._capture_to_project)
        pl.addWidget(self._capture_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#7F8C8D;font-size:10px;")
        self._status_lbl.setWordWrap(True)
        pl.addWidget(self._status_lbl)
        pl.addStretch()
        root.addWidget(panel)

        # ── Center: video + result ────────────────────────────────────────
        center = QVBoxLayout()
        self._preview = QLabel(tr("liveclass.no_signal"))
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(480, 360)
        self._preview.setStyleSheet("background:#1A252F;color:#7F8C8D;border-radius:6px;")
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center.addWidget(self._preview, 1)

        self._result_lbl = QLabel(tr("liveclass.result_idle"))
        self._result_lbl.setAlignment(Qt.AlignCenter)
        self._result_lbl.setStyleSheet(
            "font-size:18px;font-weight:bold;padding:8px;"
            "background:#1A252F;border-radius:6px;color:#7F8C8D;"
        )
        center.addWidget(self._result_lbl)

        self._topk_lbl = QLabel("")
        self._topk_lbl.setAlignment(Qt.AlignCenter)
        self._topk_lbl.setStyleSheet("font-size:12px;color:#AAB4BE;")
        center.addWidget(self._topk_lbl)
        root.addLayout(center, 1)

    # ------------------------------------------------------------------ API

    def set_project(self, project, audit=None) -> None:
        """Receive the active project; preload its current classifier if any."""
        self.project = project
        path = getattr(getattr(project, "config", None), "current_model_path", "") or \
            getattr(project, "current_model_path", "")
        if path and os.path.isfile(path):
            self._try_load(path)

    # ------------------------------------------------------------------ model

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("liveclass.load_model_btn"), "", "PyTorch-Modell (*.pth)"
        )
        if path:
            self._try_load(path)

    def _try_load(self, path: str) -> None:
        try:
            self._inferencer.load_model(path)
            self._model_lbl.setText(tr(
                "liveclass.model_loaded",
                name=os.path.basename(path),
                classes=", ".join(self._inferencer.class_names),
            ))
            self._model_lbl.setStyleSheet("color:#27AE60;font-size:11px;")
        except Exception as exc:
            self._model_lbl.setText(tr("liveclass.model_error", err=str(exc)))
            self._model_lbl.setStyleSheet("color:#E74C3C;font-size:11px;")

    # ------------------------------------------------------------------ camera

    def _scan_cameras(self) -> None:
        self._cam_combo.clear()
        try:
            cams = list_usb_cameras()
        except Exception:
            cams = []
        for idx, label in cams:
            self._cam_combo.addItem(label, userData=idx)
        if not cams:
            self._cam_combo.addItem(tr("liveclass.no_camera"), userData=None)

    def _start(self) -> None:
        url = self._url_edit.text().strip()
        if url:
            source = int(url) if url.isdigit() else url
        else:
            source = self._cam_combo.currentData()
        if source is None:
            QMessageBox.warning(self, tr("common.info"), tr("liveclass.no_source"))
            return
        self._frame_counter = 0
        self._camera_thread = CameraFrameThread(source, fps=15, parent=self)
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.error.connect(self._on_cam_error)
        self._camera_thread.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText(tr("liveclass.running"))

    def _stop(self) -> None:
        if self._camera_thread is not None:
            self._camera_thread.stop()
            self._camera_thread = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText(tr("liveclass.stopped"))

    def _on_cam_error(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    # ------------------------------------------------------------------ frame loop

    def _on_roi_toggled(self, checked: bool) -> None:
        for sp in self._roi_spins.values():
            sp.setEnabled(checked)

    def _roi_px(self, frame: np.ndarray):
        """Return the ROI as pixel (x1, y1, x2, y2), or None if disabled/too small."""
        if not self._roi_cb.isChecked():
            return None
        h, w = frame.shape[:2]
        x1 = int(self._roi_spins["x1"].value() / 100 * w)
        y1 = int(self._roi_spins["y1"].value() / 100 * h)
        x2 = int(self._roi_spins["x2"].value() / 100 * w)
        y2 = int(self._roi_spins["y2"].value() / 100 * h)
        x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
        y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)

    def _confident(self, result: Optional[dict]) -> bool:
        return bool(result) and result["confidence"] * 100 >= self._minconf_spin.value()

    def _on_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        self._frame_counter += 1

        roi = self._roi_px(frame)
        crop = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame

        interval = self._interval_spin.value()
        if self._inferencer.is_ready() and self._frame_counter % interval == 0:
            try:
                self._last_result = self._inferencer.predict_frame(crop, top_k=3)
                self._update_result(self._last_result)
            except Exception as exc:
                self._status_lbl.setText(tr("liveclass.infer_error", err=str(exc)))

        self._show_frame(self._overlay(frame, self._last_result, roi))

    def _overlay(self, frame: np.ndarray, result: Optional[dict], roi=None) -> np.ndarray:
        out = frame.copy()
        if roi is not None:
            cv2.rectangle(out, (roi[0], roi[1]), (roi[2], roi[3]), (0, 200, 255), 2)
        if not result:
            return out
        confident = self._confident(result)
        label = result["predicted_label"] if confident else tr("liveclass.uncertain")
        conf = result["confidence"]
        color = (0, 170, 0) if confident else (0, 170, 230)
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (26, 37, 47), -1)
        cv2.putText(out, f"{label}   {conf*100:.1f}%", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return out

    def _update_result(self, result: dict) -> None:
        confident = self._confident(result)
        label = result["predicted_label"] if confident else tr("liveclass.uncertain")
        conf = result["confidence"]
        color = "#27AE60" if confident else "#E67E22"
        self._result_lbl.setText(f"{label}    {conf*100:.1f}%")
        self._result_lbl.setStyleSheet(
            f"font-size:18px;font-weight:bold;padding:8px;"
            f"background:#1A252F;border-radius:6px;color:{color};"
        )
        parts = [f"{t['label']}: {t['prob']*100:.1f}%" for t in result.get("top_k", [])]
        self._topk_lbl.setText("   |   ".join(parts))

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._preview.width(), self._preview.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._preview.setPixmap(pix)

    # ------------------------------------------------------------------ capture loop

    def _capture_to_project(self) -> None:
        if self.project is None:
            QMessageBox.warning(self, tr("common.info"), tr("liveclass.no_project"))
            return
        if self._last_frame is None:
            QMessageBox.warning(self, tr("common.info"), tr("liveclass.capture_no_frame"))
            return
        base = (getattr(self.project.config, "image_dir", "") or
                (os.path.dirname(self.project.project_path) if getattr(self.project, "project_path", "") else "") or
                os.getcwd())
        out_dir = os.path.join(base, "live_captures")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]
        path = os.path.join(out_dir, f"live_{ts}.jpg")
        try:
            cv2.imwrite(path, self._last_frame)
        except Exception as exc:
            QMessageBox.warning(self, tr("common.error"), str(exc))
            return
        if self.project.add_image(path):
            self.images_added.emit(1)
            self._status_lbl.setText(tr("liveclass.captured", name=os.path.basename(path)))

    # ------------------------------------------------------------------ lifecycle

    def stop_camera(self) -> None:
        """Stop the camera thread (called on app close / page switch)."""
        self._stop()
