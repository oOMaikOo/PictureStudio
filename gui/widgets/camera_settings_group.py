"""
Camera property sliders + preprocessing filter selector.

`CameraPage` and `CameraCaptureDialog` both show the same two group boxes and
used to define the slider ranges, defaults and filter entries separately — two
copies that had to be kept in step by hand. This widget owns them once.

It knows nothing about cameras: it emits what the user changed and leaves
applying it to the owner, which is what differs between the two call sites
(`CameraPage._camera_thread` vs. `CameraCaptureDialog._frame_thread`).
"""
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSlider, QLabel, QPushButton, QComboBox,
)
from PySide6.QtCore import Qt, Signal


class CameraSettingsGroup(QWidget):
    """
    Two group boxes: collapsible "Kamera-Einstellungen" (five property sliders
    plus a reset button) and "Vorverarbeitung" (filter dropdown).

    Signals
    -------
    prop_changed(str, int)   A single slider moved — property name and value.
    props_reset(dict)        Reset was pressed — the full set of defaults.
    filter_changed(str)      Another preprocessing filter was selected.
    """

    prop_changed = Signal(str, int)
    props_reset = Signal(dict)
    filter_changed = Signal(str)

    # (tr-key, property name, minimum, maximum, default)
    SLIDERS = [
        ("camera.brightness_label", "brightness", -64, 64, 0),
        ("camera.contrast_label",   "contrast",     0, 95, 0),
        ("camera.saturation_label", "saturation",   0, 100, 0),
        ("camera.sharpness_label",  "sharpness",    0, 7,  0),
        ("camera.exposure_label",   "exposure",   -13, -1, -6),
    ]

    # (tr-key, filter key as understood by core.camera.apply_frame_filter)
    FILTERS = [
        ("camera.filter_none",      "none"),
        ("camera.filter_grayscale", "grayscale"),
        ("camera.filter_canny",     "canny"),
        ("camera.filter_sobel",     "sobel"),
        ("camera.filter_laplacian", "laplacian"),
    ]

    def __init__(self, cam_props: Optional[dict] = None,
                 filter_name: str = "none", parent=None):
        """
        Parameters
        ----------
        cam_props   Start values per property name; missing ones use the default.
        filter_name Preselected filter key; unknown values fall back to "none".
        """
        super().__init__(parent)
        from utils.i18n import tr

        cam_props = cam_props or {}
        self._sliders: Dict[str, QSlider] = {}
        self._value_labels: Dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── Camera settings ──────────────────────────────────────────────────
        self.settings_group = QGroupBox(tr("camera.settings_group"))
        self.settings_group.setCheckable(True)
        self.settings_group.setChecked(False)   # collapsed by default
        form = QFormLayout(self.settings_group)
        form.setSpacing(4)

        for key, prop, lo, hi, default in self.SLIDERS:
            start = cam_props.get(prop, default)
            form.addRow(tr(key), self._build_slider_row(prop, lo, hi, start))

        reset_btn = QPushButton(tr("camera.reset_btn"))
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self.reset)
        form.addRow("", reset_btn)
        root.addWidget(self.settings_group)

        # ── Preprocessing filter ─────────────────────────────────────────────
        self.filter_group = QGroupBox(tr("camera.filter_group"))
        self._filter_form = QFormLayout(self.filter_group)
        self._filter_form.setSpacing(4)

        self.filter_combo = QComboBox()
        for key, value in self.FILTERS:
            self.filter_combo.addItem(tr(key), value)
        index = self.filter_combo.findData(filter_name)
        if index >= 0:
            self.filter_combo.setCurrentIndex(index)
        self.filter_combo.currentIndexChanged.connect(
            lambda _i: self.filter_changed.emit(self.filter_name()))
        self._filter_form.addRow(tr("camera.filter_label"), self.filter_combo)
        root.addWidget(self.filter_group)

    def _build_slider_row(self, prop: str, lo: int, hi: int, start: int) -> QHBoxLayout:
        """Build one slider + value-label row and register it under *prop*."""
        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(start)
        slider.setFixedHeight(18)

        value_lbl = QLabel(str(start))
        value_lbl.setFixedWidth(30)
        value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row.addWidget(slider)
        row.addWidget(value_lbl)

        def _on_change(value: int, p=prop, lbl=value_lbl):
            lbl.setText(str(value))
            self.prop_changed.emit(p, value)

        slider.valueChanged.connect(_on_change)
        self._sliders[prop] = slider
        self._value_labels[prop] = value_lbl
        return row

    # ── Public API ───────────────────────────────────────────────────────────

    def cam_props(self) -> Dict[str, int]:
        """Current value of every property slider."""
        return {prop: sl.value() for prop, sl in self._sliders.items()}

    def filter_name(self) -> str:
        """Key of the selected preprocessing filter."""
        return self.filter_combo.currentData() or "none"

    def defaults(self) -> Dict[str, int]:
        """Neutral value of every property slider."""
        return {prop: default for _k, prop, _lo, _hi, default in self.SLIDERS}

    def reset(self) -> Dict[str, int]:
        """
        Move every slider back to its default and emit ``props_reset``.

        Emits once for the whole set rather than per slider, so the owner sends
        a single ``set_cam_props()`` to the camera. Returns the defaults.
        """
        defaults = self.defaults()
        for prop, value in defaults.items():
            slider = self._sliders[prop]
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            self._value_labels[prop].setText(str(value))
        self.props_reset.emit(defaults)
        return defaults

    def add_filter_row(self, label: str, widget: QWidget) -> None:
        """Append an owner-specific row to the preprocessing group."""
        self._filter_form.addRow(label, widget)
