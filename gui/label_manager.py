"""
Dialog for creating, editing, and deleting class labels.
"""
from typing import Dict, List, Optional, Callable

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QLabel,
    QColorDialog, QMessageBox, QWidget, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QIcon

from utils.config import DEFAULT_COLORS


class LabelManagerDialog(QDialog):
    """Modal dialog for managing class labels."""

    labels_changed = Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Labels verwalten")
        self.setMinimumSize(420, 380)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # List
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(QLabel("Vorhandene Labels:"))
        layout.addWidget(self.list_widget)

        # Edit row
        edit_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Label-Name (z.B. gut, schlecht, Fehler_A)")
        edit_row.addWidget(self.name_edit)

        self.color_btn = QPushButton("Farbe")
        self.color_btn.setFixedWidth(80)
        self.color_btn.clicked.connect(self._pick_color)
        self._selected_color = DEFAULT_COLORS[0]
        self._update_color_btn()
        edit_row.addWidget(self.color_btn)
        layout.addLayout(edit_row)

        # Buttons
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Hinzufügen")
        add_btn.clicked.connect(self._add_label)
        btn_row.addWidget(add_btn)

        edit_btn = QPushButton("Umbenennen")
        edit_btn.clicked.connect(self._rename_label)
        btn_row.addWidget(edit_btn)

        del_btn = QPushButton("Löschen")
        del_btn.clicked.connect(self._delete_label)
        btn_row.addWidget(del_btn)

        layout.addLayout(btn_row)

        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _color_icon(self, color_str: str) -> QIcon:
        pix = QPixmap(16, 16)
        pix.fill(QColor(color_str))
        return QIcon(pix)

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for name, info in self.project.labels.items():
            item = QListWidgetItem(self._color_icon(info["color"]), name)
            item.setData(Qt.UserRole, info["color"])
            self.list_widget.addItem(item)

    def _on_selection_changed(self, row: int) -> None:
        item = self.list_widget.item(row)
        if item:
            self.name_edit.setText(item.text())
            self._selected_color = item.data(Qt.UserRole)
            self._update_color_btn()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._selected_color), self, "Farbe wählen")
        if color.isValid():
            self._selected_color = color.name()
            self._update_color_btn()

    def _update_color_btn(self) -> None:
        pix = QPixmap(16, 16)
        pix.fill(QColor(self._selected_color))
        self.color_btn.setIcon(QIcon(pix))
        self.color_btn.setText("Farbe")

    def _add_label(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Fehler", "Bitte einen Namen eingeben.")
            return
        if name in self.project.labels:
            QMessageBox.warning(self, "Fehler", f"Label '{name}' existiert bereits.")
            return
        # Auto-assign color if default is taken
        used_colors = {info["color"] for info in self.project.labels.values()}
        color = self._selected_color
        if color in used_colors:
            for c in DEFAULT_COLORS:
                if c not in used_colors:
                    color = c
                    break
        self.project.add_label(name, color)
        self.name_edit.clear()
        self._refresh_list()
        self.labels_changed.emit()

    def _rename_label(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        old_name = item.text()
        new_name = self.name_edit.text().strip()
        if not new_name or new_name == old_name:
            return
        if new_name in self.project.labels:
            QMessageBox.warning(self, "Fehler", f"Label '{new_name}' existiert bereits.")
            return
        self.project.rename_label(old_name, new_name)
        self._refresh_list()
        self.labels_changed.emit()

    def _delete_label(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        name = item.text()
        reply = QMessageBox.question(
            self, "Löschen", f"Label '{name}' wirklich löschen?\n"
                             "Alle zugehörigen Zuweisungen werden entfernt.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.project.remove_label(name)
            self._refresh_list()
            self.labels_changed.emit()


class LabelSelector(QWidget):
    """
    Compact widget to display labels as colored buttons for quick assignment.
    """

    label_selected = Signal(str)  # label name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._buttons: Dict[str, QPushButton] = {}
        self._active: str = ""

    def refresh(self, labels: Dict) -> None:
        """Rebuild buttons from label dict."""
        for btn in self._buttons.values():
            self._layout.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()

        for name, info in labels.items():
            btn = QPushButton(name)
            btn.setCheckable(True)
            color = info.get("color", "#888888")
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; color: white; "
                f"border-radius: 4px; padding: 4px 8px; font-weight: bold; }}"
                f"QPushButton:checked {{ border: 3px solid white; }}"
            )
            btn.clicked.connect(lambda checked, n=name: self._on_click(n))
            self._layout.addWidget(btn)
            self._buttons[name] = btn

        self._layout.addStretch()

    def _on_click(self, name: str) -> None:
        # Toggle: clicking active label deselects it
        if self._active == name:
            self._active = ""
            self._buttons[name].setChecked(False)
            self.label_selected.emit("")
        else:
            if self._active and self._active in self._buttons:
                self._buttons[self._active].setChecked(False)
            self._active = name
            self.label_selected.emit(name)

    def set_active_label(self, name: str) -> None:
        if self._active and self._active in self._buttons:
            self._buttons[self._active].setChecked(False)
        self._active = name
        if name and name in self._buttons:
            self._buttons[name].setChecked(True)

    def get_active_label(self) -> str:
        return self._active
