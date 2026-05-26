"""
Labeling page: image list + ROI editor + label assignment.
Uses the enhanced ROI editor (v2) and lazy thumbnail list.
"""
import logging
import os
from typing import Optional, List, Dict

log = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QComboBox, QListWidget, QListWidgetItem,
    QScrollArea, QLineEdit, QMessageBox, QCheckBox, QButtonGroup,
    QRadioButton, QFrame, QToolBar, QSizePolicy, QMenu, QProgressBar,
    QTextEdit, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QPoint, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap, QKeySequence, QShortcut, QUndoStack

from gui.widgets.thumbnail_list import LazyThumbnailList
from gui.widgets.roi_editor import ROIEditor, DRAW_RECT, DRAW_ELLIPSE, DRAW_POLYGON
from utils.config import DEFAULT_COLORS


class LabelingPage(QWidget):
    """Main labeling workspace for the Picture Studio application (stack index 2).

    Presents a three-panel layout (thumbnail list / ROI editor / controls) and
    wires together all labeling interactions:

    - Lazy thumbnail list with multi-select, search, sort and chip filters.
    - ROI editor (rect / ellipse / polygon) backed by an undo/redo stack.
    - Single-label combo and multi-label checkbox mode (switchable at runtime).
    - Uncertain-flag (QA) toggle with optional free-text comment.
    - Bulk-label assignment for multiple selected images.
    - Active Learning (AL) queue panel for reviewing model predictions.
    - Per-class progress bars and keyboard-shortcut reference.

    Signals:
        project_changed: Emitted after label changes are saved to the project.
        al_retrain_requested: Emitted when the user completes the AL queue and
            requests an immediate retraining run.
    """

    project_changed    = Signal()   # emitted after saving changes
    al_retrain_requested = Signal() # user finished AL queue and wants to retrain

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._current_image: str = ""
        self._audit = None
        self._undo_stack = QUndoStack(self)
        self._undo_stack.setUndoLimit(100)
        self._pre_labeler = None
        self._pre_label_thread = None
        # Debounce stats refreshes: rapid label changes (e.g. bulk-accept) coalesce
        # into a single _do_update_stats() call 80 ms after the last trigger.
        self._stats_timer = QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.setInterval(80)
        self._stats_timer.timeout.connect(self._do_update_stats)
        self._build_ui()
        self._setup_shortcuts()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """Create the three-column horizontal splitter and populate each panel."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([210, 700, 270])

    # ---- Left: image list ----

    def _build_left_panel(self) -> QGroupBox:
        """Build the left column: image list, filters, progress bar and AL/bulk panels."""
        from utils.i18n import tr
        box = QGroupBox(tr("labeling.images_group"))
        v = QVBoxLayout(box)

        load_btn = QPushButton("Ordner laden…")
        load_btn.clicked.connect(self._load_folder)
        v.addWidget(load_btn)

        filter_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Suchen…")
        self.search_edit.textChanged.connect(self._filter_list)
        filter_row.addWidget(self.search_edit)
        v.addLayout(filter_row)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Dateiname A→Z",
            "Dateiname Z→A",
            "Label A→Z",
            "Erst ungelabelt",
            "Erst gelabelt",
        ])
        self.sort_combo.currentIndexChanged.connect(self._filter_list)
        v.addWidget(self.sort_combo)

        # Scrollable label chip row
        self._chip_scroll = QScrollArea()
        self._chip_scroll.setFixedHeight(32)
        self._chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._chip_scroll.setFrameShape(QFrame.NoFrame)
        self._chip_scroll.setWidgetResizable(True)
        self._chip_widget = QWidget()
        self._chip_hbox = QHBoxLayout(self._chip_widget)
        self._chip_hbox.setContentsMargins(0, 2, 0, 2)
        self._chip_hbox.setSpacing(3)
        self._chip_hbox.addStretch()
        self._chip_scroll.setWidget(self._chip_widget)
        self._label_chip_btns: Dict[str, QPushButton] = {}
        v.addWidget(self._chip_scroll)

        filter_cb_row = QHBoxLayout()
        self.unlabeled_only_cb = QCheckBox(tr("labeling.filter_unlabeled"))
        self.unlabeled_only_cb.toggled.connect(self._filter_list)
        filter_cb_row.addWidget(self.unlabeled_only_cb)

        self.roi_only_cb = QCheckBox("Nur mit ROIs")
        self.roi_only_cb.toggled.connect(self._filter_list)
        filter_cb_row.addWidget(self.roi_only_cb)

        reset_filter_btn = QPushButton("✕")
        reset_filter_btn.setFixedSize(20, 20)
        reset_filter_btn.setToolTip("Filter zurücksetzen")
        reset_filter_btn.clicked.connect(self._reset_filters)
        filter_cb_row.addWidget(reset_filter_btn)
        v.addLayout(filter_cb_row)

        filter_cb_row2 = QHBoxLayout()
        self.uncertain_only_cb = QCheckBox(tr("labeling.filter_uncertain"))
        self.uncertain_only_cb.toggled.connect(self._filter_list)
        filter_cb_row2.addWidget(self.uncertain_only_cb)
        v.addLayout(filter_cb_row2)

        self._visible_count_label = QLabel("")
        self._visible_count_label.setAlignment(Qt.AlignCenter)
        self._visible_count_label.setStyleSheet("color:#5D9CEC; font-size:9px;")
        self._visible_count_label.hide()
        v.addWidget(self._visible_count_label)

        from utils.settings import AppSettings
        self.thumb_list = LazyThumbnailList(thumb_size=AppSettings().get_thumbnail_size())
        self.thumb_list.image_selected.connect(self._on_image_selected)
        self.thumb_list.selection_changed.connect(self._on_selection_changed)
        self.thumb_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.thumb_list.customContextMenuRequested.connect(self._on_thumb_context_menu)
        v.addWidget(self.thumb_list)

        # Drag & Drop onto the thumbnail list
        from gui.widgets.drop_mixin import ImageDropFilter
        self._drop_filter = ImageDropFilter(self.thumb_list)
        self._drop_filter.files_dropped.connect(self._on_files_dropped)

        # Numeric key filter: 1-9 assigns labels, Space advances — even when
        # the thumbnail list (not the ROI editor) has keyboard focus.
        from PySide6.QtCore import QObject, QEvent as _QEvent
        _page = self

        class _NumKeyFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == _QEvent.KeyPress:
                    k = event.key()
                    if Qt.Key_1 <= k <= Qt.Key_9:
                        _page._quick_assign_label(k - Qt.Key_1)
                        return True
                    if k == Qt.Key_Space:
                        _page._next_image()
                        return True
                return False

        self._num_key_filter = _NumKeyFilter(self.thumb_list)
        self.thumb_list.installEventFilter(self._num_key_filter)

        shortcut_hint = QLabel("1–9: Label  ·  Space: weiter  ·  Entf: löschen")
        shortcut_hint.setAlignment(Qt.AlignCenter)
        shortcut_hint.setStyleSheet("color:#444D56; font-size:9px;")
        v.addWidget(shortcut_hint)

        # ── Labeling progress bar ────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(10)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background:#1a1a2e; border:none; border-radius:5px; }"
            "QProgressBar::chunk { background:#2ECC71; border-radius:5px; }"
        )
        v.addWidget(self._progress_bar)

        self.img_count_label = QLabel("0 / 0 gelabelt")
        self.img_count_label.setAlignment(Qt.AlignCenter)
        self.img_count_label.setStyleSheet("color:#aaa; font-size:10px;")
        v.addWidget(self.img_count_label)

        # ── Massen-Labeling Panel (hidden until 2+ images selected) ──────────
        self._bulk_panel = QFrame()
        self._bulk_panel.setFrameShape(QFrame.StyledPanel)
        self._bulk_panel.setStyleSheet(
            "QFrame { background:#1A3A5C; border:1px solid #2980B9; border-radius:5px; }"
        )
        bp = QVBoxLayout(self._bulk_panel)
        bp.setContentsMargins(6, 6, 6, 6)
        bp.setSpacing(4)

        self._bulk_info = QLabel("0 Bilder ausgewählt")
        self._bulk_info.setStyleSheet("color:#AED6F1; font-weight:bold; font-size:11px;")
        bp.addWidget(self._bulk_info)

        bulk_row = QHBoxLayout()
        self._bulk_label_combo = QComboBox()
        self._bulk_label_combo.addItem("(kein)")
        bulk_row.addWidget(self._bulk_label_combo)

        bulk_btn = QPushButton("Allen zuweisen")
        bulk_btn.setStyleSheet(
            "background:#2980B9; color:white; font-weight:bold;"
            "border-radius:4px; padding:3px 8px;"
        )
        bulk_btn.clicked.connect(self._bulk_assign_label)
        bulk_row.addWidget(bulk_btn)
        bp.addLayout(bulk_row)

        self._bulk_panel.hide()
        v.addWidget(self._bulk_panel)

        # ── Active Learning Queue Panel ──────────────────────────────────────
        self._al_panel = QFrame()
        self._al_panel.setFrameShape(QFrame.StyledPanel)
        self._al_panel.setStyleSheet(
            "QFrame { background:#2C1A0A; border:1px solid #E67E22; border-radius:5px; }"
        )
        alp = QVBoxLayout(self._al_panel)
        alp.setContentsMargins(6, 6, 6, 6)
        alp.setSpacing(4)

        al_header = QHBoxLayout()
        self._al_count_label = QLabel("🔄 AL-Queue: 0")
        self._al_count_label.setStyleSheet(
            "color:#E67E22; font-weight:bold; font-size:11px;"
        )
        al_header.addWidget(self._al_count_label)
        al_header.addStretch()
        al_clear_btn = QPushButton("✕")
        al_clear_btn.setFixedSize(20, 20)
        al_clear_btn.setToolTip("Queue leeren")
        al_clear_btn.setStyleSheet("color:#E67E22; font-weight:bold;")
        al_clear_btn.clicked.connect(self._al_clear_queue)
        al_header.addWidget(al_clear_btn)
        alp.addLayout(al_header)

        self._al_suggestion_label = QLabel("")
        self._al_suggestion_label.setStyleSheet("color:#F0B27A; font-size:10px;")
        self._al_suggestion_label.setWordWrap(True)
        alp.addWidget(self._al_suggestion_label)

        al_nav_row = QHBoxLayout()
        self._al_next_btn = QPushButton("→ Nächstes")
        self._al_next_btn.setStyleSheet(
            "background:#E67E22; color:white; font-weight:bold;"
            "border-radius:4px; padding:3px 8px;"
        )
        self._al_next_btn.clicked.connect(self._al_next_image)
        al_nav_row.addWidget(self._al_next_btn)

        self._al_done_btn = QPushButton("✓ Gelabelt")
        self._al_done_btn.setStyleSheet(
            "background:#27AE60; color:white; border-radius:4px; padding:3px 8px;"
        )
        self._al_done_btn.setToolTip("Bild als gelabelt markieren und aus Queue entfernen")
        self._al_done_btn.clicked.connect(self._al_mark_done)
        al_nav_row.addWidget(self._al_done_btn)

        self._al_accept_btn = QPushButton("⚡ Übernehmen")
        self._al_accept_btn.setStyleSheet(
            "background:#1F6FEB; color:white; border-radius:4px; padding:3px 8px;"
        )
        self._al_accept_btn.setToolTip(
            "Vorgeschlagenes Label für dieses Bild übernehmen und zum nächsten springen"
        )
        self._al_accept_btn.clicked.connect(self._al_accept_suggestion)
        al_nav_row.addWidget(self._al_accept_btn)
        alp.addLayout(al_nav_row)

        # Bulk accept row
        al_bulk_row = QHBoxLayout()
        self._al_bulk_btn = QPushButton("⚡ Alle ≥80% übernehmen")
        self._al_bulk_btn.setStyleSheet(
            "background:#BC8CFF; color:white; border-radius:4px;"
            "padding:3px 8px; font-size:10px;"
        )
        self._al_bulk_btn.setToolTip(
            "Alle Queue-Bilder mit Confidence ≥ 80% automatisch labeln"
        )
        self._al_bulk_btn.clicked.connect(self._al_bulk_accept)
        al_bulk_row.addWidget(self._al_bulk_btn)
        alp.addLayout(al_bulk_row)

        self._al_panel.hide()
        v.addWidget(self._al_panel)

        return box

    # ---- Center: ROI editor + toolbar ----

    def _build_center_panel(self) -> QGroupBox:
        """Build the center column: draw-mode toolbar, ROI editor and mask editor tabs."""
        box = QGroupBox("Bildansicht & ROI-Editor")
        v = QVBoxLayout(box)

        # Draw mode toolbar
        toolbar = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)

        for label, mode, key in [
            ("Rect (R)", DRAW_RECT, "R"),
            ("Ellipse (E)", DRAW_ELLIPSE, "E"),
            ("Polygon (G)", DRAW_POLYGON, "G"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(f"Zeichenmodus: {label} (Taste: {key})")
            btn.clicked.connect(lambda _, m=mode: self._set_draw_mode(m))
            self._mode_group.addButton(btn)
            toolbar.addWidget(btn)
        self._mode_group.buttons()[0].setChecked(True)

        toolbar.addSeparator_() if hasattr(toolbar, 'addSeparator_') else toolbar.addSpacing(10)
        del_btn = QPushButton("Löschen (Del)")
        del_btn.clicked.connect(self.roi_editor.delete_selected if hasattr(self, 'roi_editor') else lambda: None)
        toolbar.addWidget(del_btn)

        zoom_reset_btn = QPushButton("Zoom zurücksetzen")
        zoom_reset_btn.clicked.connect(lambda: self.roi_editor.reset_zoom())
        toolbar.addWidget(zoom_reset_btn)
        toolbar.addStretch()

        # Undo / Redo buttons wired to the undo stack
        self._undo_btn = QPushButton("↩ Rückgängig")
        self._undo_btn.setEnabled(False)
        self._undo_btn.setToolTip("Rückgängig (Strg+Z)")
        self._undo_btn.clicked.connect(self._undo_stack.undo)
        toolbar.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("↪ Wiederholen")
        self._redo_btn.setEnabled(False)
        self._redo_btn.setToolTip("Wiederholen (Strg+Y)")
        self._redo_btn.clicked.connect(self._undo_stack.redo)
        toolbar.addWidget(self._redo_btn)

        self._undo_stack.canUndoChanged.connect(self._undo_btn.setEnabled)
        self._undo_stack.canRedoChanged.connect(self._redo_btn.setEnabled)
        self._undo_stack.undoTextChanged.connect(
            lambda t: self._undo_btn.setToolTip(f"Rückgängig: {t} (Strg+Z)"))
        self._undo_stack.redoTextChanged.connect(
            lambda t: self._redo_btn.setToolTip(f"Wiederholen: {t} (Strg+Y)"))

        v.addLayout(toolbar)

        from gui.widgets.roi_editor import ROIEditor
        from gui.widgets.mask_editor import MaskEditorPanel
        from PySide6.QtWidgets import QTabWidget as _QTabWidget

        self._center_tabs = _QTabWidget()

        self.roi_editor = ROIEditor()
        self.roi_editor.roi_added.connect(self._on_roi_added)
        self.roi_editor.roi_deleted.connect(self._on_roi_deleted)
        self.roi_editor.roi_selected.connect(self._on_roi_selected)
        self.roi_editor.roi_moved.connect(self._on_roi_moved)
        self.roi_editor.label_quick_assign.connect(self._quick_assign_label)
        self.roi_editor.mode_changed.connect(self._on_mode_changed)
        self.roi_editor.whole_image_roi_requested.connect(self._create_whole_image_roi)
        self.roi_editor.space_pressed.connect(self._next_image)
        self.roi_editor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_editor.customContextMenuRequested.connect(self._on_image_context_menu)
        self._center_tabs.addTab(self.roi_editor, "🔲 ROI / Klassifikation")

        self.mask_editor = MaskEditorPanel()
        self.mask_editor.mask_saved.connect(lambda p: None)  # no-op for now
        self._center_tabs.addTab(self.mask_editor, "🎨 Segmentierungsmaske")

        self._center_tabs.currentChanged.connect(self._on_center_tab_changed)
        v.addWidget(self._center_tabs)

        # Fix del_btn binding
        del_btn.clicked.disconnect()
        del_btn.clicked.connect(self.roi_editor.delete_selected)

        self.img_path_label = QLabel("Kein Bild geladen")
        self.img_path_label.setAlignment(Qt.AlignCenter)
        self.img_path_label.setStyleSheet("color:#aaa;font-size:10px;")
        v.addWidget(self.img_path_label)
        return box

    # ---- Right: controls ----

    def _build_right_panel(self) -> QScrollArea:
        """Build the right column: label controls, ROI list, navigation and statistics."""
        from utils.i18n import tr
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        v = QVBoxLayout(container)
        scroll.setWidget(container)

        # Image label (single-label combo or multi-label checkboxes)
        img_lbl_box = QGroupBox(tr("labeling.labels_group"))
        ib = QVBoxLayout(img_lbl_box)
        ib.setSpacing(4)

        self._ml_toggle_btn = QPushButton(tr("labeling.multi_label_cb"))
        self._ml_toggle_btn.setCheckable(True)
        self._ml_toggle_btn.setToolTip(tr("labeling.multi_label_tooltip"))
        self._ml_toggle_btn.setStyleSheet(
            "QPushButton{background:#2a2a3a;color:#888;border:1px solid #444;"
            "border-radius:3px;padding:3px 6px;font-size:10px;}"
            "QPushButton:checked{background:#1A3A5C;color:#3498DB;"
            "border-color:#2980B9;font-weight:bold;}"
        )
        self._ml_toggle_btn.clicked.connect(self._toggle_multi_label_mode)
        ib.addWidget(self._ml_toggle_btn)

        self._label_stack = QStackedWidget()

        # Page 0: single-label combo
        _sp = QWidget()
        _spl = QVBoxLayout(_sp)
        _spl.setContentsMargins(0, 0, 0, 0)
        self.img_label_combo = QComboBox()
        self.img_label_combo.addItem("(kein)")
        self.img_label_combo.currentTextChanged.connect(self._assign_image_label)
        _spl.addWidget(self.img_label_combo)
        self._label_stack.addWidget(_sp)

        # Page 1: multi-label checkboxes in a scroll area
        _ml_scroll = QScrollArea()
        _ml_scroll.setWidgetResizable(True)
        _ml_scroll.setMaximumHeight(120)
        _ml_scroll.setFrameShape(QFrame.NoFrame)
        self._ml_cb_widget = QWidget()
        self._multi_label_layout = QVBoxLayout(self._ml_cb_widget)
        self._multi_label_layout.setContentsMargins(2, 2, 2, 2)
        self._multi_label_layout.setSpacing(2)
        self._multi_label_layout.addStretch()
        _ml_scroll.setWidget(self._ml_cb_widget)
        self._multi_label_cbs: Dict[str, QCheckBox] = {}
        self._label_stack.addWidget(_ml_scroll)

        ib.addWidget(self._label_stack)

        flag_row = QHBoxLayout()
        self._uncertain_btn = QPushButton(tr("labeling.uncertain_cb"))
        self._uncertain_btn.setCheckable(True)
        self._uncertain_btn.setFixedHeight(24)
        self._uncertain_btn.setToolTip("Label als unsicher markieren (benötigt QA-Review)")
        self._uncertain_btn.setStyleSheet(
            "QPushButton{background:#2a2a3a;color:#888;border:1px solid #444;"
            "border-radius:3px;padding:2px 8px;font-size:10px;}"
            "QPushButton:checked{background:#4A2800;color:#E67E22;"
            "border-color:#E67E22;font-weight:bold;}"
        )
        self._uncertain_btn.clicked.connect(self._toggle_uncertain)
        flag_row.addWidget(self._uncertain_btn)

        self._uncertain_comment_edit = QLineEdit()
        self._uncertain_comment_edit.setPlaceholderText(tr("labeling.comment_placeholder"))
        self._uncertain_comment_edit.setStyleSheet("font-size:10px;")
        self._uncertain_comment_edit.setVisible(False)
        self._uncertain_comment_edit.editingFinished.connect(self._on_comment_changed)
        flag_row.addWidget(self._uncertain_comment_edit)
        ib.addLayout(flag_row)

        v.addWidget(img_lbl_box)

        # Draw mode (mirrored as radio group in right panel)
        mode_box = QGroupBox("Zeichenmodus (R/E/G)")
        mb = QVBoxLayout(mode_box)
        self._right_mode_btns: Dict[str, QPushButton] = {}
        for label, mode in [("Rechteck", DRAW_RECT), ("Ellipse", DRAW_ELLIPSE), ("Polygon", DRAW_POLYGON)]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self._set_draw_mode(m))
            mb.addWidget(btn)
            self._right_mode_btns[mode] = btn
        self._right_mode_btns[DRAW_RECT].setChecked(True)
        v.addWidget(mode_box)

        # ROI list
        roi_box = QGroupBox("ROIs dieses Bildes")
        rb = QVBoxLayout(roi_box)

        roi_hint = QLabel(
            "ROI = Interessensbereich. Jedes Bild kann eigene ROIs haben.\n"
            "Beim Training wird nur der ROI-Ausschnitt klassifiziert."
        )
        roi_hint.setWordWrap(True)
        roi_hint.setStyleSheet("color:#aaa;font-size:9px;")
        rb.addWidget(roi_hint)

        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(160)
        self.roi_list.currentRowChanged.connect(self._on_roi_list_select)
        rb.addWidget(self.roi_list)

        roi_lbl_row = QHBoxLayout()
        roi_lbl_row.addWidget(QLabel("Label:"))
        self.roi_label_combo = QComboBox()
        self.roi_label_combo.addItem("(kein)")
        roi_lbl_row.addWidget(self.roi_label_combo)
        rb.addLayout(roi_lbl_row)

        assign_btn = QPushButton("ROI-Label zuweisen")
        assign_btn.clicked.connect(self._assign_roi_label)
        rb.addWidget(assign_btn)

        del_roi_btn = QPushButton("Ausgewählten ROI löschen")
        del_roi_btn.clicked.connect(self._delete_roi_from_list)
        rb.addWidget(del_roi_btn)
        v.addWidget(roi_box)

        # ROI — apply to all images
        apply_box = QGroupBox("ROI auf alle Bilder")
        ab = QVBoxLayout(apply_box)

        apply_hint = QLabel(
            "Gleicher ROI für alle Bilder (z.B. feste Kamera, Fließband):"
        )
        apply_hint.setWordWrap(True)
        apply_hint.setStyleSheet("color:#aaa;font-size:9px;")
        ab.addWidget(apply_hint)

        apply_all_btn = QPushButton("ROIs dieses Bildes → alle Bilder")
        apply_all_btn.setToolTip(
            "Kopiert die ROIs des aktuellen Bildes auf alle anderen Bilder im Projekt.\n"
            "Vorhandene ROIs werden überschrieben."
        )
        apply_all_btn.setStyleSheet(
            "background:#1565C0;color:white;padding:5px;border-radius:4px;"
        )
        apply_all_btn.clicked.connect(self._apply_roi_to_all)
        ab.addWidget(apply_all_btn)

        apply_size_btn = QPushButton("ROI-Größe → alle Bilder")
        apply_size_btn.setToolTip(
            "Überträgt nur Breite und Höhe des ausgewählten ROI auf alle Bilder.\n"
            "Die Position (x/y) bleibt pro Bild erhalten — zum Verschieben einfach\n"
            "den ROI mit der Maus ziehen (im Rechteck-/Ellipse-Modus)."
        )
        apply_size_btn.setStyleSheet(
            "background:#00695C;color:white;padding:5px;border-radius:4px;"
        )
        apply_size_btn.clicked.connect(self._apply_roi_size_to_all)
        ab.addWidget(apply_size_btn)

        clear_all_rois_btn = QPushButton("Alle ROIs löschen")
        clear_all_rois_btn.setStyleSheet(
            "background:#B71C1C;color:white;padding:5px;border-radius:4px;"
        )
        clear_all_rois_btn.clicked.connect(self._clear_all_rois)
        ab.addWidget(clear_all_rois_btn)
        v.addWidget(apply_box)

        # ROI Validation
        val_box = QGroupBox("ROI-Validierung")
        vb = QVBoxLayout(val_box)
        val_btn = QPushButton("ROIs dieses Bildes prüfen")
        val_btn.clicked.connect(self._validate_rois)
        vb.addWidget(val_btn)
        v.addWidget(val_box)

        # Keyboard labeling
        kb_box = QGroupBox("⌨ Tastatur-Labeling")
        kbv = QVBoxLayout(kb_box)
        kbv.setSpacing(4)

        self._auto_advance_cb = QCheckBox("Auto-weiter nach Label (1–9)")
        self._auto_advance_cb.setToolTip(
            "Nach dem Zuweisen eines Labels per Zahlentaste\n"
            "automatisch zum nächsten Bild wechseln."
        )
        self._auto_advance_cb.setChecked(True)
        kbv.addWidget(self._auto_advance_cb)

        kb_hint = QLabel(
            "<small>"
            "<b>1–9</b> Label zuweisen<br>"
            "<b>W</b> Ganzbild-ROI<br>"
            "<b>Space / N</b> Nächstes Bild<br>"
            "<b>P</b> Vorheriges Bild<br>"
            "<b>R / E / G</b> Zeichenmodus<br>"
            "<b>Del</b> ROI löschen<br>"
            "<b>Pfeile</b> ROI verschieben (2 px)<br>"
            "<b>Ctrl+C/V</b> ROI kopieren/einfügen<br>"
            "<b>Ctrl+Z/Y</b> Rückgängig/Wiederholen"
            "</small>"
        )
        kb_hint.setStyleSheet("color:#888; padding:2px;")
        kb_hint.setWordWrap(True)
        kbv.addWidget(kb_hint)

        v.addWidget(kb_box)

        # Navigation
        nav_box = QGroupBox("Navigation (← / →)")
        nb = QHBoxLayout(nav_box)
        prev_btn = QPushButton("◀")
        prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("▶")
        next_btn.clicked.connect(self._next_image)
        nb.addWidget(prev_btn)
        nb.addWidget(next_btn)
        v.addWidget(nav_box)

        # Stats + per-class progress
        stats_box = QGroupBox("Statistik & Fortschritt")
        sb = QVBoxLayout(stats_box)
        sb.setSpacing(3)
        self.stats_label = QLabel("–")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("font-size:10px;")
        sb.addWidget(self.stats_label)

        self._class_progress = QTextEdit()
        self._class_progress.setReadOnly(True)
        self._class_progress.setFixedHeight(130)
        self._class_progress.setStyleSheet(
            "background:#12121F; border:none; font-size:10px;"
        )
        sb.addWidget(self._class_progress)

        self._qa_review_btn = QPushButton("QA-Review…")
        self._qa_review_btn.setFlat(True)
        self._qa_review_btn.setStyleSheet("color:#E67E22; font-size:10px; padding:3px;")
        self._qa_review_btn.setToolTip("Unsichere Labels überprüfen")
        self._qa_review_btn.clicked.connect(self._open_qa_review)
        sb.addWidget(self._qa_review_btn)

        v.addWidget(stats_box)

        # Shortcut reference panel
        shortcuts_box = QGroupBox("Tastenkürzel")
        sb2 = QVBoxLayout(shortcuts_box)
        shortcuts_box.setStyleSheet("QGroupBox { font-size: 10px; color: #555; }")
        _shortcuts = [
            ("←  /  →",      "Voriges / Nächstes Bild"),
            ("1 – 9",         "Label direkt zuweisen"),
            ("Leertaste",     "Nächstes Bild"),
            ("Entf",          "Label entfernen"),
            ("U",             "Unsicher markieren"),
            ("Strg+Z / Y",    "Rückgängig / Wiederholen"),
        ]
        for key, desc in _shortcuts:
            row = QHBoxLayout()
            key_lbl = QLabel(key)
            key_lbl.setFixedWidth(72)
            key_lbl.setStyleSheet(
                "background:#21262D; color:#7EE787; font-family:monospace;"
                " font-size:10px; border-radius:3px; padding:1px 4px;"
            )
            row.addWidget(key_lbl)
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color:#8B949E; font-size:10px;")
            row.addWidget(desc_lbl)
            row.addStretch()
            w = QWidget()
            w.setLayout(row)
            sb2.addWidget(w)
        v.addWidget(shortcuts_box)

        # Pre-Labeling panel
        pre_box = QGroupBox("🤖 Pre-Labeling")
        pre_box.setStyleSheet("QGroupBox { font-size: 11px; }")
        pb = QVBoxLayout(pre_box)

        pre_info = QLabel(
            "Modell auf ungelabelte Bilder anwenden und Labels vorschlagen."
        )
        pre_info.setWordWrap(True)
        pre_info.setStyleSheet("color:#aaa; font-size:10px;")
        pb.addWidget(pre_info)

        pre_model_row = QHBoxLayout()
        self._pre_model_lbl = QLabel("Kein Modell geladen.")
        self._pre_model_lbl.setStyleSheet("color:#ccc; font-size:10px;")
        self._pre_model_lbl.setWordWrap(True)
        pre_model_row.addWidget(self._pre_model_lbl, 1)
        self._pre_load_btn = QPushButton("📂")
        self._pre_load_btn.setFixedWidth(28)
        self._pre_load_btn.setToolTip("Trainiertes .pth-Modell laden")
        self._pre_load_btn.clicked.connect(self._pre_load_model)
        pre_model_row.addWidget(self._pre_load_btn)
        pb.addLayout(pre_model_row)

        pre_thr_row = QHBoxLayout()
        pre_thr_row.addWidget(QLabel("Min. Konfidenz:"))
        from PySide6.QtWidgets import QDoubleSpinBox as _DSB
        self._pre_thr_spin = _DSB()
        self._pre_thr_spin.setRange(0.0, 1.0)
        self._pre_thr_spin.setSingleStep(0.05)
        self._pre_thr_spin.setValue(0.75)
        self._pre_thr_spin.setDecimals(2)
        self._pre_thr_spin.setToolTip(
            "Vorschläge unter diesem Schwellwert werden übersprungen."
        )
        pre_thr_row.addWidget(self._pre_thr_spin)
        pb.addLayout(pre_thr_row)

        self._pre_only_unlabeled_cb = QCheckBox("Nur ungelabelte Bilder")
        self._pre_only_unlabeled_cb.setChecked(True)
        self._pre_only_unlabeled_cb.setStyleSheet("font-size: 10px;")
        pb.addWidget(self._pre_only_unlabeled_cb)

        self._pre_run_btn = QPushButton("▶ Vorschläge generieren")
        self._pre_run_btn.setEnabled(False)
        self._pre_run_btn.setStyleSheet(
            "QPushButton{background:#1A4A2A;color:#58D68D;border:1px solid #27AE60;"
            "border-radius:4px;padding:4px 8px;}"
            "QPushButton:disabled{color:#555;border-color:#333;background:#1a1a2a;}"
        )
        self._pre_run_btn.clicked.connect(self._pre_run)
        pb.addWidget(self._pre_run_btn)

        self._pre_apply_btn = QPushButton("✅ Vorschläge übernehmen")
        self._pre_apply_btn.setEnabled(False)
        self._pre_apply_btn.setToolTip(
            "Alle Vorschläge über dem Schwellwert als Labels speichern (Undo möglich)."
        )
        self._pre_apply_btn.setStyleSheet(
            "QPushButton{background:#1F6FEB;color:white;border-radius:4px;padding:4px 8px;}"
            "QPushButton:disabled{color:#555;background:#1a1a2a;border:1px solid #333;}"
        )
        self._pre_apply_btn.clicked.connect(self._pre_apply)
        pb.addWidget(self._pre_apply_btn)

        self._pre_progress = QProgressBar()
        self._pre_progress.setVisible(False)
        self._pre_progress.setFixedHeight(8)
        self._pre_progress.setTextVisible(False)
        pb.addWidget(self._pre_progress)

        self._pre_status = QLabel("")
        self._pre_status.setWordWrap(True)
        self._pre_status.setStyleSheet("color:#aaa; font-size:10px;")
        pb.addWidget(self._pre_status)

        v.addWidget(pre_box)

        v.addStretch()
        return scroll

    def _setup_shortcuts(self) -> None:
        """Register all page-level keyboard shortcuts (navigation, undo/redo, flag)."""
        # Navigation
        QShortcut(QKeySequence("N"),          self, self._next_image)
        QShortcut(QKeySequence("P"),          self, self._prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_image)
        QShortcut(QKeySequence(Qt.Key_Left),  self, self._prev_image)
        # Undo / Redo
        QShortcut(QKeySequence("Ctrl+Z"),       self, self._undo_stack.undo)
        QShortcut(QKeySequence("Ctrl+Y"),       self, self._undo_stack.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._undo_stack.redo)
        # Label clear (Delete key)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._clear_current_label)
        # Toggle uncertain flag
        QShortcut(QKeySequence("U"), self, self._shortcut_toggle_uncertain)

    def _clear_current_label(self) -> None:
        """Delete shortcut: remove label from current image."""
        if not self.project or not self._current_image:
            return
        if self.project.is_multi_label:
            from gui.labeling_commands import SetMultiLabelsCommand
            self._undo_stack.push(
                SetMultiLabelsCommand(
                    self, self._current_image, [],
                    list(self.project.get_image_multi_labels(self._current_image))
                )
            )
        else:
            self._assign_label_direct(self._current_image, "")
        mw = self.window()
        if hasattr(mw, "statusBar"):
            mw.statusBar().showMessage("Label gelöscht — Strg+Z zum Rückgängigmachen", 3000)

    def _shortcut_toggle_uncertain(self) -> None:
        """U shortcut: toggle the uncertain flag on the current image."""
        if not self.project or not self._current_image:
            return
        # click() toggles checked state AND emits clicked(bool) → _toggle_uncertain
        self._uncertain_btn.click()

    def closeEvent(self, event) -> None:
        self._stats_timer.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------ project

    def set_project(self, project, audit=None) -> None:
        """Bind a new project and refresh all UI state.

        Parameters:
            project: The active ``Project`` instance.
            audit: Optional ``AuditLog`` used to record label changes.
        """
        self.project = project
        self._audit = audit
        self._current_image = ""
        self._pre_labeler = None
        self._undo_stack.clear()
        self._refresh_label_combos()
        self._refresh_thumb_list()
        self._update_stats()
        self.refresh_al_queue_panel()
        from utils.i18n import tr
        is_ml = getattr(project.config, "multi_label", False) if project else False
        self._ml_toggle_btn.blockSignals(True)
        self._ml_toggle_btn.setChecked(is_ml)
        self._ml_toggle_btn.setText(
            "Multi-Label deaktivieren" if is_ml else "Multi-Label aktivieren"
        )
        self._ml_toggle_btn.blockSignals(False)
        self._label_stack.setCurrentIndex(1 if is_ml else 0)
        has_images = bool(project and project.images)
        self._pre_run_btn.setEnabled(has_images and self._pre_labeler is not None)

    def _refresh_label_combos(self) -> None:
        """Repopulate the label combo, ROI combo, bulk combo, chips and multi-label checkboxes."""
        labels = list(self.project.labels.keys()) if self.project else []

        # img_label_combo: add key hints for 1-9
        self.img_label_combo.blockSignals(True)
        self.img_label_combo.clear()
        self.img_label_combo.addItem("(kein)")
        for i, lbl in enumerate(labels):
            hint = f"[{i+1}] {lbl}" if i < 9 else lbl
            self.img_label_combo.addItem(hint, lbl)   # display text with hint, user data = real label
        self.img_label_combo.blockSignals(False)

        # roi and bulk combos: plain labels
        for combo in [self.roi_label_combo, self._bulk_label_combo]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(kein)")
            combo.addItems(labels)
            combo.blockSignals(False)

        # Rebuild label filter chips
        self._rebuild_label_chips(labels)

        # Rebuild multi-label checkboxes
        self._rebuild_multi_label_cbs(labels)

    def _refresh_thumb_list(self) -> None:
        """Rebuild the thumbnail list from the project's image list, syncing labels and flags."""
        self.thumb_list.clear_all()
        if not self.project:
            return
        self.thumb_list.set_root_dir(self.project.config.image_dir or None)
        for img_path in self.project.images:
            if self.project.is_multi_label:
                lbls = self.project.get_image_multi_labels(img_path)
                display = ", ".join(lbls) if lbls else ""
                color = self.project.get_label_color(lbls[0]) if lbls else ""
            else:
                lbl = self.project.get_image_label(img_path)
                display, color = lbl, (self.project.get_label_color(lbl) if lbl else "")
            self.thumb_list.add_image(img_path, display, color)
            if self.project.is_label_uncertain(img_path):
                self.thumb_list.update_flag(img_path, True)
        self._update_stats()

    def on_labels_changed(self) -> None:
        """Called by MainWindow when the project's label set changes; refreshes all label UI."""
        self._refresh_label_combos()
        self._refresh_thumb_list()
        if self._current_image:
            self._load_image(self._current_image)
        self._update_stats()

    # ------------------------------------------------------------------ image loading

    def _on_files_dropped(self, paths: list) -> None:
        """Handle files/folders dragged onto the thumbnail list."""
        from utils.i18n import tr
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("common.no_project_msg"))
            return
        added = 0
        for path in paths:
            if self.project.add_image(path):
                lbl = self.project.get_image_label(path)
                color = self.project.get_label_color(lbl) if lbl else ""
                self.thumb_list.add_image(path, lbl, color)
                added += 1
        if added:
            self._update_stats()

    def _load_folder(self) -> None:
        """Open a folder chooser, scan for images and add new ones to the project."""
        from utils.i18n import tr
        if not self.project:
            QMessageBox.warning(self, tr("common.no_project"), tr("common.no_project_msg"))
            return
        from utils.config import IMAGE_FORMATS
        folder = QFileDialog.getExistingDirectory(self, tr("data.dlg.folder_select"))
        if not folder:
            return
        added = 0
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() in IMAGE_FORMATS:
                img_path = os.path.join(folder, fname)
                if self.project.add_image(img_path):
                    lbl = self.project.get_image_label(img_path)
                    color = self.project.get_label_color(lbl) if lbl else ""
                    self.thumb_list.add_image(img_path, lbl, color)
                    added += 1
        self.project.config.image_dir = folder
        self._update_stats()
        if added:
            QMessageBox.information(self, tr("common.saved"), tr("data.msg.images_loaded", added=added, folder=folder))

    from PySide6.QtWidgets import QFileDialog  # needed for _load_folder

    def _filter_list(self) -> None:
        """Re-apply all active filters (search text, label chips, checkboxes, sort order)."""
        if not self.project:
            return
        search = self.search_edit.text()
        only_unlabeled = self.unlabeled_only_cb.isChecked()
        only_roi = getattr(self, 'roi_only_cb', None) and self.roi_only_cb.isChecked()

        # Label set from chips (None = all)
        label_set = None
        if not only_unlabeled and self._label_chip_btns:
            checked = [lbl for lbl, chip in self._label_chip_btns.items() if chip.isChecked()]
            if checked:
                label_set = set(checked)

        # ROI filter
        roi_paths = None
        if only_roi:
            roi_paths = {p for p in self.project.images if self.project.get_rois(p)}

        # Sort
        sort_idx = self.sort_combo.currentIndex() if hasattr(self, 'sort_combo') else 0
        if sort_idx == 1:
            self.thumb_list.sort_items(lambda p: os.path.basename(p).lower(), reverse=True)
        elif sort_idx == 2:
            def _lbl_az(p):
                lbl = self.project.get_image_label(p) or ""
                return (0 if lbl else 1, lbl.lower(), os.path.basename(p).lower())
            self.thumb_list.sort_items(_lbl_az)
        elif sort_idx == 3:
            self.thumb_list.sort_items(
                lambda p: (1 if self.project.get_image_label(p) else 0,
                           os.path.basename(p).lower())
            )
        elif sort_idx == 4:
            self.thumb_list.sort_items(
                lambda p: (0 if self.project.get_image_label(p) else 1,
                           os.path.basename(p).lower())
            )
        else:
            self.thumb_list.sort_items(lambda p: os.path.basename(p).lower())

        uncertain_paths = None
        if hasattr(self, 'uncertain_only_cb') and self.uncertain_only_cb.isChecked():
            uncertain_paths = set(self.project.get_uncertain_images())

        self.thumb_list.filter(
            text=search,
            label_set=label_set,
            only_unlabeled=only_unlabeled,
            roi_paths=roi_paths,
            uncertain_paths=uncertain_paths,
        )

        # Show visible count when any filter is active
        visible = self.thumb_list.count_visible()
        total = len(self.thumb_list._items)
        if visible < total:
            self._visible_count_label.setText(f"{visible} / {total} sichtbar")
            self._visible_count_label.show()
        else:
            self._visible_count_label.hide()

    def _rebuild_label_chips(self, labels: List[str]) -> None:
        """Recreate the horizontal row of toggle-button filter chips above the thumbnail list."""
        while self._chip_hbox.count() > 0:
            item = self._chip_hbox.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._label_chip_btns.clear()

        from utils.i18n import tr
        self._all_chip = QPushButton(tr("labeling.filter_all"))
        self._all_chip.setCheckable(True)
        self._all_chip.setChecked(True)
        self._all_chip.setFixedHeight(22)
        self._all_chip.setStyleSheet(
            "QPushButton{background:#3D4560;color:white;border-radius:3px;"
            "padding:1px 7px;font-size:10px;border:none;}"
            "QPushButton:checked{background:#6C79C0;color:white;}"
        )
        self._all_chip.clicked.connect(self._on_all_chip_clicked)
        self._chip_hbox.addWidget(self._all_chip)

        for lbl in labels:
            color = (self.project.get_label_color(lbl) if self.project else "#888") or "#888"
            chip = QPushButton(lbl)
            chip.setCheckable(True)
            chip.setFixedHeight(22)
            chip.setStyleSheet(
                f"QPushButton{{background:#2a2a3a;color:{color};"
                f"border:1px solid {color};border-radius:3px;"
                f"padding:1px 6px;font-size:10px;}}"
                f"QPushButton:checked{{background:{color};color:#111;}}"
            )
            chip.clicked.connect(self._on_label_chip_clicked)
            self._chip_hbox.addWidget(chip)
            self._label_chip_btns[lbl] = chip

        self._chip_hbox.addStretch()

    def _on_all_chip_clicked(self) -> None:
        """Deselect all individual label chips and show all images."""
        for chip in self._label_chip_btns.values():
            chip.setChecked(False)
        self._all_chip.setChecked(True)
        self._filter_list()

    def _on_label_chip_clicked(self) -> None:
        """Keep the 'Alle' chip state consistent when individual label chips change."""
        any_checked = any(c.isChecked() for c in self._label_chip_btns.values())
        self._all_chip.setChecked(not any_checked)
        self._filter_list()

    def _reset_filters(self) -> None:
        """Clear all filter controls and show the full image list."""
        self.search_edit.blockSignals(True)
        self.search_edit.clear()
        self.search_edit.blockSignals(False)
        self.unlabeled_only_cb.blockSignals(True)
        self.unlabeled_only_cb.setChecked(False)
        self.unlabeled_only_cb.blockSignals(False)
        self.roi_only_cb.blockSignals(True)
        self.roi_only_cb.setChecked(False)
        self.roi_only_cb.blockSignals(False)
        for chip in self._label_chip_btns.values():
            chip.setChecked(False)
        if hasattr(self, '_all_chip'):
            self._all_chip.setChecked(True)
        if hasattr(self, 'uncertain_only_cb'):
            self.uncertain_only_cb.blockSignals(True)
            self.uncertain_only_cb.setChecked(False)
            self.uncertain_only_cb.blockSignals(False)
        self._filter_list()

    def filter_by_label(self, label_name: str) -> None:
        """Activate the chip filter for `label_name` and clear all other filters."""
        self._reset_filters()
        if label_name in self._label_chip_btns:
            if hasattr(self, '_all_chip'):
                self._all_chip.setChecked(False)
            self._label_chip_btns[label_name].setChecked(True)
            self._filter_list()

    # ------------------------------------------------------------------ image selection

    @Slot(list)
    def _on_selection_changed(self, paths: list) -> None:
        """Show/hide the bulk panel and update the selection counter."""
        multi = len(paths) >= 2
        self._bulk_panel.setVisible(multi)
        if multi:
            self._bulk_info.setText(f"{len(paths)} Bilder ausgewählt")

    @Slot(str)
    def _on_image_selected(self, image_path: str) -> None:
        """Flush ROIs of the previous image then load the newly selected one."""
        self._save_current_rois()
        self._load_image(image_path)

    def _on_center_tab_changed(self, idx: int) -> None:
        """Lazily load the mask editor when the user switches to the segmentation tab."""
        if idx == 1 and self._current_image:
            self.mask_editor.load_image(self._current_image)

    def _load_image(self, image_path: str) -> None:
        """Display an image in the ROI editor, restore its ROIs and sync all label controls."""
        self._current_image = image_path
        self.img_path_label.setText(os.path.basename(image_path))
        self.roi_editor.load_image(image_path)
        if self._center_tabs.currentIndex() == 1:
            self.mask_editor.load_image(image_path)
        rois = self.project.get_rois(image_path) if self.project else []
        self.roi_editor.load_rois(rois)
        self._refresh_roi_list()
        if self.project and self.project.is_multi_label:
            self._load_multi_label_checkboxes(image_path)
        else:
            lbl = self.project.get_image_label(image_path) if self.project else ""
            self.img_label_combo.blockSignals(True)
            idx = self.img_label_combo.findData(lbl) if lbl else 0
            self.img_label_combo.setCurrentIndex(max(0, idx))
            self.img_label_combo.blockSignals(False)

        if self.project:
            flag = self.project.get_label_flag(image_path)
            uncertain = flag.get("uncertain", False)
            comment = flag.get("comment", "")
            self._uncertain_btn.blockSignals(True)
            self._uncertain_btn.setChecked(uncertain)
            self._uncertain_btn.blockSignals(False)
            self._uncertain_comment_edit.setVisible(uncertain)
            self._uncertain_comment_edit.setText(comment if uncertain else "")

    def _save_current_rois(self) -> None:
        """Write the ROI editor's current state back into ``project.rois`` for the active image."""
        if not self.project or not self._current_image:
            return
        self.project.rois[self._current_image] = self.roi_editor.get_all_roi_data()

    def _refresh_roi_list(self) -> None:
        """Repopulate the ROI list widget for the currently displayed image."""
        self.roi_list.clear()
        if not self.project or not self._current_image:
            return
        for roi in self.project.get_rois(self._current_image):
            roi_id = roi.get("id", "?")
            label = roi.get("label", "–")
            roi_type = roi.get("type", "rect")
            x, y = int(roi.get("x", 0)), int(roi.get("y", 0))
            w, h = int(roi.get("w", 0)), int(roi.get("h", 0))
            item = QListWidgetItem(f"[{roi_type}] {roi_id[:4]}  {label}  ({x},{y} {w}×{h})")
            item.setData(Qt.UserRole, roi_id)
            color = roi.get("color", "#888")
            item.setForeground(QColor(color))
            self.roi_list.addItem(item)

    # ------------------------------------------------------------------ context menus

    def _label_context_menu(self, global_pos: QPoint, image_path: str) -> None:
        """Show a popup with all project labels; assigns the chosen one."""
        if not self.project or not image_path:
            return
        menu = QMenu(self)
        menu.setTitle("Label zuweisen")
        for lbl_name, lbl_info in self.project.labels.items():
            action = menu.addAction(lbl_name)
            color = lbl_info.get("color", "#888888")
            action.setData((image_path, lbl_name))
            # Colour the action text to match the label colour
            try:
                from PySide6.QtGui import QIcon, QPixmap
                pix = QPixmap(14, 14)
                pix.fill(QColor(color))
                action.setIcon(QIcon(pix))
            except Exception:
                pass
        menu.addSeparator()
        menu.addAction("(kein Label)").setData((image_path, ""))
        menu.addSeparator()
        remove_act = menu.addAction("🗑 Bild aus Datensatz entfernen")
        remove_act.setData(None)
        chosen = menu.exec(global_pos)
        if chosen is remove_act:
            self._remove_images([image_path])
        elif chosen:
            data = chosen.data()
            if data is not None:
                img, lbl = data
                self._assign_label_direct(img, lbl)

    def _assign_label_direct(self, image_path: str, label: str) -> None:
        """Push a SetImageLabelCommand onto the undo stack."""
        if not self.project:
            return
        from gui.labeling_commands import SetImageLabelCommand
        old = self.project.get_image_label(image_path)
        if old == label:
            return
        self._undo_stack.push(SetImageLabelCommand(self, image_path, label, old))
        if label:
            mw = self.window()
            if hasattr(mw, "statusBar"):
                mw.statusBar().showMessage(f"Label gesetzt: {label}", 2000)

    # --- actual worker called by commands ---

    def _do_set_image_label(self, image_path: str, label: str) -> None:
        """Mutate the project, update the thumbnail and combo; called by SetImageLabelCommand."""
        if not self.project:
            return
        self.project.set_image_label(image_path, label)
        color = self.project.get_label_color(label) if label else ""
        if self.project.is_multi_label:
            # Also sync image_multi_labels so the thumbnail shows all active labels
            if label:
                multi = list(self.project.get_image_multi_labels(image_path))
                if label not in multi:
                    self.project.set_image_multi_labels(image_path, [label] + multi)
            all_lbls = self.project.get_image_multi_labels(image_path)
            self.thumb_list.update_label(image_path, ", ".join(all_lbls), color)
        else:
            self.thumb_list.update_label(image_path, label, color)
        self.thumb_list.update_flag(image_path, self.project.is_label_uncertain(image_path))
        if self._audit:
            self._audit.log_image_labeled(image_path, label)
        if image_path == self._current_image:
            if self.project.is_multi_label:
                self._load_multi_label_checkboxes(image_path)
            else:
                self.img_label_combo.blockSignals(True)
                idx = self.img_label_combo.findData(label) if label else 0
                self.img_label_combo.setCurrentIndex(max(0, idx))
                self.img_label_combo.blockSignals(False)
        self._update_stats()
        if label:
            al_paths = {e["path"] for e in self.project.active_learning_queue}
            if image_path in al_paths:
                self.project.remove_from_al_queue(image_path)
                self.refresh_al_queue_panel()

    def _do_set_multi_labels(self, image_path: str, labels: list) -> None:
        """Mutate multi-labels and sync the primary label; called by SetMultiLabelsCommand."""
        if not self.project:
            return
        self.project.set_image_multi_labels(image_path, labels)
        # Sync primary label to first multi-label
        primary = labels[0] if labels else ""
        self.project.set_image_label(image_path, primary)
        color = self.project.get_label_color(primary) if primary else ""
        self.thumb_list.update_label(image_path, ", ".join(labels), color)
        self.thumb_list.update_flag(image_path, self.project.is_label_uncertain(image_path))
        if self._audit:
            self._audit.log_image_labeled(image_path, str(labels))
        if image_path == self._current_image:
            self._load_multi_label_checkboxes(image_path)
        self._update_stats()
        if labels:
            al_paths = {e["path"] for e in self.project.active_learning_queue}
            if image_path in al_paths:
                self.project.remove_from_al_queue(image_path)
                self.refresh_al_queue_panel()

    def _rebuild_multi_label_cbs(self, labels: List[str]) -> None:
        """Recreate the multi-label checkbox list in the right panel for the given label names."""
        while self._multi_label_layout.count() > 0:
            item = self._multi_label_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._multi_label_cbs.clear()
        for lbl in labels:
            color = (self.project.get_label_color(lbl) if self.project else "#888") or "#888"
            cb = QCheckBox(lbl)
            cb.setStyleSheet(f"color: {color}; font-size: 11px;")
            cb.toggled.connect(self._on_multi_label_cb_changed)
            self._multi_label_layout.addWidget(cb)
            self._multi_label_cbs[lbl] = cb
        self._multi_label_layout.addStretch()

    def _load_multi_label_checkboxes(self, image_path: str) -> None:
        """Tick/untick the multi-label checkboxes to reflect the current image's labels."""
        if not self.project:
            return
        active = set(self.project.get_image_multi_labels(image_path))
        for lbl, cb in self._multi_label_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(lbl in active)
            cb.blockSignals(False)

    def _on_multi_label_cb_changed(self) -> None:
        """Push a SetMultiLabelsCommand when the user ticks/unticks a multi-label checkbox."""
        if not self.project or not self._current_image:
            return
        new_labels = [lbl for lbl, cb in self._multi_label_cbs.items() if cb.isChecked()]
        old_labels = list(self.project.get_image_multi_labels(self._current_image))
        if new_labels == old_labels:
            return
        from gui.labeling_commands import SetMultiLabelsCommand
        self._undo_stack.push(SetMultiLabelsCommand(
            self, self._current_image, new_labels, old_labels
        ))

    # ------------------------------------------------------------------ QA uncertain flag

    def _toggle_uncertain(self, checked: bool) -> None:
        """Toggle the uncertain QA flag for the current image via the undo stack."""
        if not self.project or not self._current_image:
            self._uncertain_btn.blockSignals(True)
            self._uncertain_btn.setChecked(False)
            self._uncertain_btn.blockSignals(False)
            return
        old_flag = self.project.get_label_flag(self._current_image)
        old_u = old_flag.get("uncertain", False)
        old_c = old_flag.get("comment", "")
        new_c = self._uncertain_comment_edit.text() if checked else ""
        self._uncertain_comment_edit.setVisible(checked)
        if not checked:
            self._uncertain_comment_edit.clear()
        from gui.labeling_commands import SetLabelFlagCommand
        self._undo_stack.push(SetLabelFlagCommand(
            self, self._current_image, checked, new_c, old_u, old_c
        ))

    def _on_comment_changed(self) -> None:
        """Persist a new uncertain-flag comment when the user finishes editing the field."""
        if not self.project or not self._current_image:
            return
        if not self._uncertain_btn.isChecked():
            return
        old_flag = self.project.get_label_flag(self._current_image)
        old_u = old_flag.get("uncertain", False)
        old_c = old_flag.get("comment", "")
        new_c = self._uncertain_comment_edit.text()
        if new_c == old_c:
            return
        from gui.labeling_commands import SetLabelFlagCommand
        self._undo_stack.push(SetLabelFlagCommand(
            self, self._current_image, old_u, new_c, old_u, old_c
        ))

    def _do_set_label_flag(self, image_path: str, uncertain: bool, comment: str) -> None:
        """Apply an uncertain-flag mutation and refresh the thumbnail; called by SetLabelFlagCommand."""
        if not self.project:
            return
        self.project.set_label_flag(image_path, uncertain, comment)
        self.thumb_list.update_flag(image_path, uncertain)
        if image_path == self._current_image:
            self._uncertain_btn.blockSignals(True)
            self._uncertain_btn.setChecked(uncertain)
            self._uncertain_btn.blockSignals(False)
            self._uncertain_comment_edit.setVisible(uncertain)
            self._uncertain_comment_edit.setText(comment if uncertain else "")
        self._update_stats()

    def _open_qa_review(self) -> None:
        """Open the QA review dialog and refresh the thumbnail list on close."""
        if not self.project:
            return
        from gui.qa_review_dialog import QAReviewDialog
        dlg = QAReviewDialog(self.project, self)
        dlg.exec()
        self._refresh_thumb_list()
        if self._current_image:
            flag = self.project.get_label_flag(self._current_image)
            uncertain = flag.get("uncertain", False)
            comment = flag.get("comment", "")
            self._uncertain_btn.blockSignals(True)
            self._uncertain_btn.setChecked(uncertain)
            self._uncertain_btn.blockSignals(False)
            self._uncertain_comment_edit.setVisible(uncertain)
            self._uncertain_comment_edit.setText(comment if uncertain else "")

    def _toggle_multi_label_mode(self, checked: bool) -> None:
        """Switch the project between single-label and multi-label mode with user confirmation."""
        from utils.i18n import tr
        if not self.project:
            self._ml_toggle_btn.setChecked(False)
            return
        if checked:
            existing = sum(1 for p in self.project.images
                           if self.project.get_image_label(p))
            if existing > 0:
                reply = QMessageBox.question(
                    self, "Multi-Label aktivieren",
                    f"{existing} Bilder haben Einzel-Labels.\n\n"
                    "Diese werden als erste Multi-Labels übernommen. Fortfahren?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    self._ml_toggle_btn.blockSignals(True)
                    self._ml_toggle_btn.setChecked(False)
                    self._ml_toggle_btn.blockSignals(False)
                    return
                self.project.migrate_to_multi_label()
            else:
                self.project.config.multi_label = True
            self._ml_toggle_btn.setText("Multi-Label deaktivieren")
            self._label_stack.setCurrentIndex(1)
        else:
            reply = QMessageBox.question(
                self, "Multi-Label deaktivieren",
                "Nur der erste Label jedes Bildes wird behalten.\nFortfahren?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self._ml_toggle_btn.blockSignals(True)
                self._ml_toggle_btn.setChecked(True)
                self._ml_toggle_btn.blockSignals(False)
                return
            self.project.migrate_to_single_label()
            self._ml_toggle_btn.setText("Multi-Label aktivieren")
            self._label_stack.setCurrentIndex(0)
        self._refresh_thumb_list()
        if self._current_image:
            self._load_image(self._current_image)

    def _on_thumb_context_menu(self, pos: QPoint) -> None:
        """Right-click: label menu for single image or bulk menu for multi-selection."""
        selected = self.thumb_list.get_selected_paths()
        if len(selected) >= 2:
            self._bulk_context_menu(self.thumb_list.mapToGlobal(pos), selected)
        else:
            item = self.thumb_list.itemAt(pos)
            if not item:
                return
            self._label_context_menu(self.thumb_list.mapToGlobal(pos),
                                     item.data(Qt.UserRole))

    def _bulk_context_menu(self, global_pos: QPoint, paths: list) -> None:
        """Context menu for bulk-labeling N selected images."""
        if not self.project:
            return
        menu = QMenu(self)
        menu.setTitle(f"{len(paths)} Bilder labeln")
        header = menu.addAction(f"─── {len(paths)} Bilder ───")
        header.setEnabled(False)
        menu.addSeparator()
        for lbl_name, lbl_info in self.project.labels.items():
            action = menu.addAction(lbl_name)
            color = lbl_info.get("color", "#888888")
            action.setData(lbl_name)
            try:
                from PySide6.QtGui import QPixmap
                pix = QPixmap(14, 14)
                pix.fill(QColor(color))
                action.setIcon(QIcon(pix))
            except Exception:
                pass
        menu.addSeparator()
        menu.addAction("(kein Label)").setData("")
        menu.addSeparator()
        remove_act = menu.addAction(f"🗑 {len(paths)} Bilder aus Datensatz entfernen")
        remove_act.setData(None)
        chosen = menu.exec(global_pos)
        if chosen is remove_act:
            self._remove_images(paths)
        elif chosen:
            label = chosen.data()
            if label is None:
                return
            old_labels = {p: self.project.get_image_label(p) for p in paths}
            from gui.labeling_commands import BulkSetImageLabelCommand
            self._undo_stack.push(BulkSetImageLabelCommand(self, paths, label, old_labels))

    def _on_image_context_menu(self, pos: QPoint) -> None:
        """Right-click on the main image view → label assignment menu for current image.
        Suppressed when in polygon mode (right-click closes polygon there)."""
        from gui.widgets.roi_editor import DRAW_POLYGON
        if self.roi_editor._mode == DRAW_POLYGON:
            return
        self._label_context_menu(self.roi_editor.mapToGlobal(pos), self._current_image)

    # ------------------------------------------------------------------ label assignment

    def _assign_image_label(self, _display_text: str) -> None:
        """Slot for the single-label combo; extracts the real label from item user data."""
        if not self.project or not self._current_image:
            return
        if self.project.is_multi_label:
            return  # checkboxes handle multi-label assignment
        data = self.img_label_combo.currentData()
        lbl = data if data is not None else (_display_text if _display_text != "(kein)" else "")
        self._assign_label_direct(self._current_image, lbl)

    def _quick_assign_label(self, index: int) -> None:
        """Assign the label at *index* (0-based) via keyboard shortcut key 1–9."""
        if not self.project or not self._current_image:
            return
        labels = list(self.project.labels.keys())
        if index >= len(labels):
            return
        lbl = labels[index]
        if self.project.is_multi_label:
            old = list(self.project.get_image_multi_labels(self._current_image))
            new = [l for l in old if l != lbl] if lbl in old else old + [lbl]
            from gui.labeling_commands import SetMultiLabelsCommand
            self._undo_stack.push(SetMultiLabelsCommand(self, self._current_image, new, old))
        else:
            self._assign_label_direct(self._current_image, lbl)
            if getattr(self, "_auto_advance_cb", None) and self._auto_advance_cb.isChecked():
                self._next_image()

    def _bulk_assign_label(self) -> None:
        """Assign one label to all currently selected images (one undo step)."""
        if not self.project:
            return
        paths = self.thumb_list.get_selected_paths()
        if len(paths) < 2:
            return
        label = self._bulk_label_combo.currentText()
        if label == "(kein)":
            label = ""
        old_labels = {p: self.project.get_image_label(p) for p in paths}
        from gui.labeling_commands import BulkSetImageLabelCommand
        self._undo_stack.push(BulkSetImageLabelCommand(self, paths, label, old_labels))

    # ------------------------------------------------------------------ ROI events

    @Slot(dict)
    def _on_roi_added(self, roi_data: dict) -> None:
        """Receive a new ROI from the editor, stamp its label/color and push AddROICommand."""
        if not self.project or not self._current_image:
            return
        roi_data["label"] = self.roi_editor.current_label
        roi_data["color"] = self.roi_editor.current_color
        from gui.labeling_commands import AddROICommand
        # Remove from editor first (command redo() will re-add it)
        self.roi_editor.delete_roi(roi_data["id"])
        self._undo_stack.push(AddROICommand(self, self._current_image, roi_data))

    def _do_add_roi(self, image_path: str, roi_data: dict) -> None:
        """Persist a new ROI in the project and add it visually to the editor; called by AddROICommand."""
        if not self.project:
            return
        self.project.add_roi(image_path, roi_data)
        if image_path == self._current_image:
            self.roi_editor.add_roi_item(roi_data)
            self._refresh_roi_list()
        if self._audit:
            self._audit.log_roi_added(image_path, roi_data["id"], roi_data.get("type", "rect"))
        self._update_stats()

    @Slot(str)
    def _on_roi_deleted(self, roi_id: str) -> None:
        """Receive a delete request from the editor and push DeleteROICommand onto the undo stack."""
        if not self.project or not self._current_image:
            return
        rois = self.project.get_rois(self._current_image)
        roi_data = next((r for r in rois if r.get("id") == roi_id), None)
        if roi_data is None:
            return
        from gui.labeling_commands import DeleteROICommand
        self._undo_stack.push(DeleteROICommand(self, self._current_image, roi_data))

    def _do_delete_roi(self, image_path: str, roi_id: str) -> None:
        """Remove an ROI from the project and the editor view; called by DeleteROICommand."""
        if not self.project:
            return
        self.project.remove_roi(image_path, roi_id)
        if image_path == self._current_image:
            self.roi_editor.delete_roi(roi_id)
            self._refresh_roi_list()
        if self._audit:
            self._audit.log_roi_deleted(image_path, roi_id)
        self._update_stats()

    @Slot(str)
    def _on_roi_selected(self, roi_id: str) -> None:
        """Sync the ROI list widget selection when the user clicks an ROI in the editor."""
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            if item.data(Qt.UserRole) == roi_id:
                self.roi_list.setCurrentRow(i)
                break

    @Slot(dict)
    def _on_roi_moved(self, roi_data: dict) -> None:
        """Push a MoveROICommand when the user drags an ROI to a new position."""
        if not self.project or not self._current_image:
            return
        rois = self.project.get_rois(self._current_image)
        old_data = next((r for r in rois if r.get("id") == roi_data.get("id")), None)
        if old_data is None:
            return
        from gui.labeling_commands import MoveROICommand
        self._undo_stack.push(MoveROICommand(self, self._current_image, roi_data, old_data))

    def _do_move_roi(self, image_path: str, roi_data: dict) -> None:
        """Update the ROI geometry in the project and refresh the editor; called by MoveROICommand."""
        if not self.project:
            return
        self.project.update_roi(image_path, roi_data["id"], roi_data)
        if image_path == self._current_image:
            self.roi_editor.update_roi_geometry(roi_data)

    def _on_roi_list_select(self, row: int) -> None:
        """Sync the ROI label combo when the user selects a row in the ROI list widget."""
        item = self.roi_list.item(row)
        if not item or not self.project or not self._current_image:
            return
        roi_id = item.data(Qt.UserRole)
        rois = self.project.get_rois(self._current_image)
        roi = next((r for r in rois if r.get("id") == roi_id), None)
        if roi:
            lbl = roi.get("label", "")
            self.roi_label_combo.blockSignals(True)
            idx = self.roi_label_combo.findText(lbl) if lbl else 0
            self.roi_label_combo.setCurrentIndex(max(0, idx))
            self.roi_label_combo.blockSignals(False)

    def _assign_roi_label(self) -> None:
        """Push AssignROILabelCommand to assign the selected label to the currently chosen ROI."""
        item = self.roi_list.currentItem()
        if not item or not self.project or not self._current_image:
            return
        roi_id = item.data(Qt.UserRole)
        new_label = self.roi_label_combo.currentText()
        if new_label == "(kein)":
            new_label = ""
        new_color = self.project.get_label_color(new_label) if new_label else "#E74C3C"
        rois = self.project.get_rois(self._current_image)
        roi = next((r for r in rois if r.get("id") == roi_id), None)
        if roi is None:
            return
        old_label = roi.get("label", "")
        old_color = roi.get("color", "#E74C3C")
        if old_label == new_label:
            return
        from gui.labeling_commands import AssignROILabelCommand
        self._undo_stack.push(AssignROILabelCommand(
            self, self._current_image, roi_id,
            new_label, new_color, old_label, old_color,
        ))

    def _do_assign_roi_label(self, image_path: str, roi_id: str,
                             label: str, color: str) -> None:
        """Mutate the ROI label/color in the project and refresh the editor; called by AssignROILabelCommand."""
        if not self.project:
            return
        rois = self.project.get_rois(image_path)
        for roi in rois:
            if roi.get("id") == roi_id:
                roi["label"] = label
                roi["color"] = color
                break
        if image_path == self._current_image:
            self.roi_editor.update_roi_label(roi_id, label, color)
            self.roi_editor.current_label = label
            self.roi_editor.current_color = color
            self._refresh_roi_list()
        self._update_stats()

    def _delete_roi_from_list(self) -> None:
        """Delete the ROI currently selected in the list widget via the undo stack."""
        item = self.roi_list.currentItem()
        if not item or not self.project or not self._current_image:
            return
        roi_id = item.data(Qt.UserRole)
        rois = self.project.get_rois(self._current_image)
        roi_data = next((r for r in rois if r.get("id") == roi_id), None)
        if roi_data is None:
            return
        from gui.labeling_commands import DeleteROICommand
        self._undo_stack.push(DeleteROICommand(self, self._current_image, roi_data))

    def _apply_roi_to_all(self) -> None:
        """Copy the current image's ROIs to every image in the project."""
        from utils.i18n import tr
        if not self.project or not self._current_image:
            QMessageBox.warning(self, tr("common.warning"), "Bitte zuerst ein Bild auswählen.")
            return
        self._save_current_rois()
        src_rois = self.project.get_rois(self._current_image)
        if not src_rois:
            QMessageBox.warning(self, tr("common.warning"),
                                "Das aktuelle Bild hat keine ROIs zum Kopieren.")
            return
        n = len(self.project.images)
        reply = QMessageBox.question(
            self, "ROIs übertragen",
            f"Die {len(src_rois)} ROI(s) dieses Bildes werden auf alle "
            f"{n} Bilder kopiert.\nVorhandene ROIs werden überschrieben.\n\n"
            "⚠ Diese Aktion kann nicht rückgängig gemacht werden.\n\nFortfahren?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        import copy, uuid as _uuid
        for img_path in self.project.images:
            if img_path == self._current_image:
                continue
            new_rois = []
            for roi in src_rois:
                r = copy.deepcopy(roi)
                r["id"] = str(_uuid.uuid4())[:8]  # unique id per image
                new_rois.append(r)
            self.project.rois[img_path] = new_rois
        QMessageBox.information(
            self, "Fertig",
            f"ROIs auf {n - 1} weitere Bilder übertragen."
        )

    def _apply_roi_size_to_all(self) -> None:
        """Copy only w+h of the selected ROI to every image; per-image x/y is kept."""
        from utils.i18n import tr
        if not self.project or not self._current_image:
            QMessageBox.warning(self, tr("common.warning"), "Bitte zuerst ein Bild auswählen.")
            return
        self._save_current_rois()
        src_rois = self.project.get_rois(self._current_image)
        if not src_rois:
            QMessageBox.warning(self, tr("common.warning"),
                                "Das aktuelle Bild hat keine ROIs.")
            return

        # Use the selected ROI if available, otherwise the first one
        sel_items = [i for i in self.roi_editor._scene.selectedItems()
                     if hasattr(i, "roi_data")]
        src_roi = sel_items[0].roi_data if sel_items else src_rois[0]
        src_w, src_h = src_roi.get("w", 0), src_roi.get("h", 0)
        src_type     = src_roi.get("type", "rect")
        src_label    = src_roi.get("label", "")
        src_color    = src_roi.get("color", "#E74C3C")

        n = len(self.project.images)
        reply = QMessageBox.question(
            self, "ROI-Größe übertragen",
            f"Breite ({src_w:.0f} px) und Höhe ({src_h:.0f} px) des ROI werden auf alle "
            f"{n} Bilder übertragen.\nBilder mit einem bestehenden ROI behalten ihre Position.\n"
            "Bilder ohne ROI erhalten einen neuen ROI an der aktuellen Position.\n\nFortfahren?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        import uuid as _uuid
        updated = created = 0
        for img_path in self.project.images:
            existing = self.project.get_rois(img_path)
            if existing:
                # Update first ROI's size only
                r = existing[0]
                r["w"] = src_w
                r["h"] = src_h
                self.project.update_roi(img_path, r["id"], r)
                updated += 1
            else:
                # No ROI yet — create one at source position with source size
                new_roi = {
                    "id":    str(_uuid.uuid4())[:8],
                    "type":  src_type,
                    "x":     src_roi.get("x", 0),
                    "y":     src_roi.get("y", 0),
                    "w":     src_w,
                    "h":     src_h,
                    "label": src_label,
                    "color": src_color,
                }
                self.project.add_roi(img_path, new_roi)
                created += 1

        # Refresh editor if the current image was affected
        self.roi_editor.load_rois(self.project.get_rois(self._current_image))
        self._refresh_roi_list()
        self._update_stats()
        QMessageBox.information(
            self, "Fertig",
            f"Größe übertragen: {updated} ROIs aktualisiert, {created} neue ROIs erstellt."
        )

    # ------------------------------------------------------------------ pre-labeling

    @Slot()
    def _pre_load_model(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Modell laden", "", "PyTorch Checkpoint (*.pth)"
        )
        if not path:
            return
        try:
            from core.pre_labeling import PreLabeler
            pl = PreLabeler()
            meta = pl.load_model(path)
            self._pre_labeler = pl
            classes = ", ".join(pl.class_names[:5])
            if len(pl.class_names) > 5:
                classes += "…"
            self._pre_model_lbl.setText(
                f"✅ {os.path.basename(path)}\n({len(pl.class_names)} Klassen: {classes})"
            )
            self._pre_run_btn.setEnabled(bool(self.project and self.project.images))
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox as _QMB
            from utils.i18n import tr
            _QMB.critical(self, tr("common.error"), str(exc))

    @Slot()
    def _pre_run(self):
        if not self.project or not self._pre_labeler:
            return
        only_unlabeled = self._pre_only_unlabeled_cb.isChecked()
        if only_unlabeled:
            candidates = [p for p in self.project.images if not self.project.get_image_label(p)]
        else:
            candidates = list(self.project.images)
        if not candidates:
            self._pre_status.setText("Keine Bilder zum Analysieren.")
            return

        self._pre_suggestions = []
        self._pre_progress.setMaximum(len(candidates))
        self._pre_progress.setValue(0)
        self._pre_progress.setVisible(True)
        self._pre_run_btn.setEnabled(False)
        self._pre_apply_btn.setEnabled(False)
        self._pre_status.setText(f"Analysiere {len(candidates)} Bilder…")

        from core.pre_labeling import PreLabelingThread
        roi = None
        if self.project.rois:
            # Use first project ROI as crop template (same as inference page fallback)
            for p, rois in self.project.rois.items():
                if rois:
                    roi = rois[0]
                    break

        t = PreLabelingThread(
            self._pre_labeler,
            candidates,
            list(self.project.labels.keys()),
            confidence_threshold=self._pre_thr_spin.value(),
            roi=roi,
            parent=self,
        )
        t.progress.connect(lambda c, tot: self._pre_progress.setValue(c))
        t.finished.connect(self._pre_on_done)
        t.error.connect(self._pre_on_error)
        self._pre_label_thread = t
        t.start()

    @Slot(list)
    def _pre_on_done(self, results: list):
        self._pre_label_thread = None
        self._pre_progress.setVisible(False)
        self._pre_run_btn.setEnabled(True)
        self._pre_suggestions = results
        accepted = [r for r in results if not r["skip"] and not r["error"]]
        skipped  = [r for r in results if r["skip"]]
        errors   = [r for r in results if r["error"]]
        thr      = self._pre_thr_spin.value()
        self._pre_status.setText(
            f"{len(accepted)} Vorschläge ≥ {thr:.0%}  •  "
            f"{len(skipped)} unter Schwellwert  •  "
            f"{len(errors)} Fehler"
        )
        self._pre_apply_btn.setEnabled(bool(accepted))

    @Slot(str)
    def _pre_on_error(self, msg: str):
        self._pre_label_thread = None
        self._pre_progress.setVisible(False)
        self._pre_run_btn.setEnabled(True)
        self._pre_status.setText(f"Fehler: {msg}")

    @Slot()
    def _pre_apply(self):
        if not self.project or not hasattr(self, "_pre_suggestions"):
            return
        accepted = [r for r in self._pre_suggestions if not r["skip"] and not r["error"]]
        if not accepted:
            return
        old_labels = {r["path"]: self.project.get_image_label(r["path"]) for r in accepted}
        label_map  = {r["path"]: r["label"] for r in accepted}
        from gui.labeling_commands import BulkSetImageLabelCommand
        self._undo_stack.push(
            BulkSetImageLabelCommand(
                self, list(label_map.keys()), "", old_labels, label_map=label_map
            )
        )
        self._pre_apply_btn.setEnabled(False)
        self._pre_status.setText(f"✅ {len(accepted)} Labels übernommen (Undo mit Strg+Z).")

    # ------------------------------------------------------------------ remove images

    def _remove_images(self, paths: list) -> None:
        """Remove one or more images from the project after user confirmation."""
        if not self.project or not paths:
            return
        n = len(paths)
        reply = QMessageBox.question(
            self, "Bilder entfernen",
            f"{'Dieses Bild' if n == 1 else f'Diese {n} Bilder'} aus dem Datensatz entfernen?\n"
            "Die Dateien auf der Festplatte bleiben erhalten.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Determine the next image to select if the current one is being removed
        next_path = None
        if self._current_image in paths:
            all_paths = self.thumb_list.get_all_paths()
            remaining = [p for p in all_paths if p not in paths]
            # Find the first image after the removed block
            for p in all_paths:
                if p not in paths and all_paths.index(p) > all_paths.index(self._current_image):
                    next_path = p
                    break
            if next_path is None and remaining:
                next_path = remaining[-1]

        # Remove from project and thumbnail list
        for path in paths:
            self.project.remove_image(path)
            self.thumb_list.remove_image(path)

        # Handle editor state
        if self._current_image in paths:
            self._current_image = ""
            if next_path:
                self.thumb_list.select_path(next_path)
            else:
                self.roi_editor.clear_image()
                self.img_path_label.setText("")
                self._refresh_roi_list()

        self._update_stats()
        self.project_changed.emit()

    def _clear_all_rois(self) -> None:
        """Delete every ROI in the project after user confirmation."""
        if not self.project:
            return
        total = self.project.get_roi_count()
        if total == 0:
            QMessageBox.information(self, "Keine ROIs", "Es sind keine ROIs vorhanden.")
            return
        reply = QMessageBox.question(
            self, "Alle ROIs löschen",
            f"Alle {total} ROIs aus dem Projekt löschen?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.project.rois.clear()
        self.roi_editor.load_rois([])
        self._refresh_roi_list()
        self._update_stats()

    def _validate_rois(self) -> None:
        """Ask the ROI editor to validate bounds and show any warnings in a dialog."""
        warnings = self.roi_editor.validate_rois()
        if warnings:
            QMessageBox.warning(self, "ROI-Validierung", "\n".join(warnings))
        else:
            QMessageBox.information(self, "ROI-Validierung", "Alle ROIs liegen innerhalb des Bildes.")

    # ------------------------------------------------------------------ draw mode

    def _set_draw_mode(self, mode: str) -> None:
        """Switch the ROI editor and the right-panel draw-mode buttons to *mode*."""
        self.roi_editor.set_mode(mode)
        for m, btn in self._right_mode_btns.items():
            btn.setChecked(m == mode)

    def _on_mode_changed(self, mode: str) -> None:
        """Sync the right-panel draw-mode buttons when the editor changes mode internally."""
        for m, btn in self._right_mode_btns.items():
            btn.setChecked(m == mode)

    # ------------------------------------------------------------------ whole-image ROI

    def _create_whole_image_roi(self) -> None:
        """Create a rectangle ROI that covers the entire loaded image (W key)."""
        if not self.project or not self._current_image:
            return
        rect = self.roi_editor._image_rect
        if rect.isNull():
            return
        import uuid
        label = self.roi_editor.current_label
        color = self.roi_editor.current_color
        roi = {
            "id":    str(uuid.uuid4())[:8],
            "type":  "rect",
            "x":     0.0,
            "y":     0.0,
            "w":     rect.width(),
            "h":     rect.height(),
            "label": label,
            "color": color,
        }
        # Go through the undo-stack command path
        from gui.labeling_commands import AddROICommand
        self._undo_stack.push(AddROICommand(self, self._current_image, roi))

    # ------------------------------------------------------------------ navigation

    def _prev_image(self) -> None:
        """Navigate to the previous image in the thumbnail list."""
        self._save_current_rois()
        paths = self.thumb_list.get_all_paths()
        if self._current_image in paths:
            idx = paths.index(self._current_image)
            if idx > 0:
                self.thumb_list.select_path(paths[idx - 1])

    def _next_image(self) -> None:
        """Navigate to the next image in the thumbnail list."""
        self._save_current_rois()
        paths = self.thumb_list.get_all_paths()
        if self._current_image in paths:
            idx = paths.index(self._current_image)
            if idx < len(paths) - 1:
                self.thumb_list.select_path(paths[idx + 1])

    # ------------------------------------------------------------------ Active Learning Queue

    def refresh_al_queue_panel(self) -> None:
        """Refresh the AL queue panel after queue changes or project load."""
        if not self.project:
            self._al_panel.hide()
            return
        queue = self.project.get_al_queue()
        unlabeled = self.project.get_unlabeled_al_queue()
        if not queue:
            self._al_panel.hide()
            return
        self._al_panel.show()
        self._al_count_label.setText(
            f"🔄 AL-Queue: {len(unlabeled)} offen / {len(queue)} gesamt"
        )
        # Show suggestion for current image if it's in queue
        entry = next((e for e in queue if e["path"] == self._current_image), None)
        if entry:
            self._al_suggestion_label.setText(
                f"Vorschlag: {entry['predicted_label']} "
                f"({entry['confidence']*100:.0f}% Conf.)"
            )
        else:
            self._al_suggestion_label.setText(
                "Klicke '→ Nächstes', um das nächste ungelabelte Queue-Bild zu öffnen."
                if unlabeled else "Alle Queue-Bilder wurden gelabelt."
            )
        self._al_next_btn.setEnabled(bool(unlabeled))
        in_queue = bool(
            self._current_image and
            any(e["path"] == self._current_image for e in queue)
        )
        self._al_done_btn.setEnabled(in_queue)
        self._al_accept_btn.setEnabled(in_queue and bool(
            next((e for e in queue if e["path"] == self._current_image), {})
            .get("predicted_label", "") in self.project.labels
        ))

    def _al_next_image(self) -> None:
        """Jump to the next unlabeled image in the AL queue."""
        if not self.project:
            return
        unlabeled = self.project.get_unlabeled_al_queue()
        if not unlabeled:
            return
        target = unlabeled[0]["path"]
        # Make sure the image is in the project
        if target not in self.project.images:
            self.project.remove_from_al_queue(target)
            self.refresh_al_queue_panel()
            return
        self._save_current_rois()
        self.thumb_list.select_path(target)
        # Show prediction suggestion after loading
        entry = unlabeled[0]
        self._al_suggestion_label.setText(
            f"Vorschlag: {entry['predicted_label']} "
            f"({entry['confidence']*100:.0f}% Conf.)"
        )
        self._al_done_btn.setEnabled(True)

    def _al_mark_done(self) -> None:
        """Manually mark the current image as done and remove from queue."""
        if not self.project or not self._current_image:
            return
        self.project.remove_from_al_queue(self._current_image)
        self.refresh_al_queue_panel()
        # Auto-advance to next queue image
        unlabeled = self.project.get_unlabeled_al_queue()
        if unlabeled:
            self._al_next_image()
        else:
            remaining = self.project.get_al_queue()
            if not remaining:
                self._al_panel.hide()
                reply = QMessageBox.question(
                    self,
                    "AL-Queue fertig",
                    "Alle Queue-Bilder wurden gelabelt!\n\n"
                    "Jetzt neu trainieren, um das Modell zu verbessern?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.al_retrain_requested.emit()

    def _al_clear_queue(self) -> None:
        """Remove all entries from the AL queue after user confirmation and hide the panel."""
        if not self.project:
            return
        n = len(self.project.get_al_queue())
        if n == 0:
            return
        reply = QMessageBox.question(
            self, "Queue leeren",
            f"Alle {n} Einträge aus der AL-Queue entfernen?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.project.clear_al_queue()
            self._al_panel.hide()

    def _al_accept_suggestion(self) -> None:
        """Apply the predicted label for the current image and advance to next."""
        if not self.project or not self._current_image:
            return
        queue = self.project.get_al_queue()
        entry = next((e for e in queue if e["path"] == self._current_image), None)
        if not entry:
            return
        suggested = entry.get("predicted_label", "")
        if suggested and suggested in self.project.labels:
            self._assign_label_direct(self._current_image, suggested)
        self._al_mark_done()

    def _al_bulk_accept(self) -> None:
        """Auto-label all AL queue images with confidence ≥ 80%."""
        if not self.project:
            return
        queue = self.project.get_al_queue()
        eligible = [
            e for e in queue
            if e.get("confidence", 0) >= 0.80
            and e.get("predicted_label", "") in self.project.labels
            and e.get("path", "") in self.project.images
        ]
        if not eligible:
            QMessageBox.information(
                self, "Keine Kandidaten",
                "Keine Queue-Einträge mit Confidence ≥ 80% und bekanntem Label gefunden."
            )
            return
        reply = QMessageBox.question(
            self, "Bulk-Accept",
            f"{len(eligible)} Bilder werden automatisch gelabelt.\nFortfahren?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        from gui.labeling_commands import SetImageLabelCommand
        assignments = {e["path"]: e["predicted_label"] for e in eligible}
        self._undo_stack.beginMacro(f"Bulk-Accept ({len(eligible)} Bilder)")
        for path, label in assignments.items():
            old = self.project.get_image_label(path)
            self._undo_stack.push(SetImageLabelCommand(self, path, label, old))
        self._undo_stack.endMacro()
        for path in assignments:
            self.project.remove_from_al_queue(path)
        self.refresh_al_queue_panel()
        self._do_update_stats()   # thumbnails already updated via update_label(); just refresh stats once
        QMessageBox.information(
            self, "Fertig",
            f"{len(eligible)} Labels übernommen.\n"
            "Im Labeling-Reiter zur Kontrolle prüfen."
        )

    # ------------------------------------------------------------------ stats & progress

    def _update_stats(self) -> None:
        """Schedule a stats refresh; rapid calls within 80 ms are coalesced into one."""
        self._stats_timer.start()

    def _do_update_stats(self) -> None:
        """Recompute labeling progress and refresh the progress bar, count label and per-class bars."""
        if not self.project:
            self.stats_label.setText("–")
            self._progress_bar.setValue(0)
            self.img_count_label.setText("0 / 0 gelabelt")
            self._class_progress.setHtml("")
            return

        total  = len(self.project.images)
        labeled = self.project.get_labeled_image_count()
        pct    = int(labeled / total * 100) if total else 0
        roi_cnt = self.project.get_roi_count()

        # ── Top progress bar ──────────────────────────────────────────────
        self._progress_bar.setValue(pct)
        # Colour the bar: red→yellow→green gradient by percentage
        if pct >= 80:
            chunk_color = "#2ECC71"
        elif pct >= 40:
            chunk_color = "#F39C12"
        else:
            chunk_color = "#E74C3C"
        self._progress_bar.setStyleSheet(
            "QProgressBar { background:#1a1a2e; border:none; border-radius:5px; }"
            f"QProgressBar::chunk {{ background:{chunk_color}; border-radius:5px; }}"
        )
        self.img_count_label.setText(
            f"{labeled} / {total} gelabelt  ({pct}%)"
            + (f"  ·  {roi_cnt} ROIs" if roi_cnt else "")
        )

        # ── Summary text ──────────────────────────────────────────────────
        uncertain_cnt = len(self.project.get_uncertain_images())
        self.stats_label.setText(
            f"Bilder: {total}   Gelabelt: {labeled}   ROIs: {roi_cnt}"
            + (f"   ⚠ Unsicher: {uncertain_cnt}" if uncertain_cnt else "")
        )

        # ── Per-class HTML bars ───────────────────────────────────────────
        self._class_progress.setHtml(
            self._render_class_progress_html(total)
        )

    def _render_class_progress_html(self, total: int) -> str:
        """Return an HTML table with per-class count, inline progress bar and percentage."""
        if not self.project or total == 0:
            return ""
        counts = self.project.get_label_counts(use_rois=False)
        unlabeled = total - self.project.get_labeled_image_count()

        rows = []
        for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            color = self.project.get_label_color(lbl)
            pct   = int(cnt / total * 100)
            rows.append((lbl, cnt, pct, color))

        # Also show unlabeled count
        if unlabeled > 0:
            rows.append(("(kein Label)", unlabeled,
                         int(unlabeled / total * 100), "#555555"))

        html = (
            "<style>"
            "body { margin:4px; font-family:sans-serif; font-size:10px; color:#ccc; }"
            "table { width:100%; border-collapse:collapse; }"
            "td { padding:2px 4px; vertical-align:middle; }"
            ".bar-bg { background:#1a1a2e; border-radius:3px; height:7px; }"
            ".bar-fill { border-radius:3px; height:7px; }"
            "</style><table>"
        )
        for lbl, cnt, pct, color in rows:
            fill_w = max(1, pct)
            html += (
                f"<tr>"
                f"<td style='width:90px;white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis;max-width:90px;color:{color}'>"
                f"<b>{lbl[:14]}</b></td>"
                f"<td style='width:30px;text-align:right;color:#888'>{cnt}</td>"
                f"<td>"
                f"<div class='bar-bg'>"
                f"<div class='bar-fill' style='width:{fill_w}%;background:{color};'></div>"
                f"</div>"
                f"</td>"
                f"<td style='width:28px;text-align:right;color:#666'>{pct}%</td>"
                f"</tr>"
            )
        html += "</table>"
        return html
