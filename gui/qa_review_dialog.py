"""
QA Review Dialog — lets a reviewer confirm or correct uncertain-flagged labels.
"""
import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QPushButton, QMessageBox, QMenu,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon


class QAReviewDialog(QDialog):
    """
    Shows all images with uncertain labels.
    Reviewer can confirm the existing label or change it.
    Both actions remove the uncertain flag.
    """

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Label-Qualitätssicherung (QA-Review)")
        self.resize(920, 580)
        self._build_ui()
        self._load_items()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)

        self._header = QLabel()
        self._header.setStyleSheet(
            "font-size:13px;font-weight:bold;color:#E67E22;padding:4px;"
        )
        v.addWidget(self._header)

        splitter = QSplitter(Qt.Horizontal)

        # Left — list of uncertain images
        self._list = QListWidget()
        self._list.setMinimumWidth(280)
        self._list.currentRowChanged.connect(self._on_row_changed)
        splitter.addWidget(self._list)

        # Right — preview + actions
        right = QVBoxLayout()
        right_w = QLabel()  # placeholder parent
        from PySide6.QtWidgets import QWidget
        rw = QWidget()
        right = QVBoxLayout(rw)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(280)
        self._preview.setStyleSheet(
            "background:#111;border-radius:6px;color:#666;"
        )
        self._preview.setText("Kein Bild ausgewählt")
        right.addWidget(self._preview)

        self._img_info = QLabel()
        self._img_info.setStyleSheet("color:#aaa;font-size:10px;")
        self._img_info.setWordWrap(True)
        right.addWidget(self._img_info)

        self._comment_edit = QLineEdit()
        self._comment_edit.setPlaceholderText("Kommentar (optional)…")
        self._comment_edit.setStyleSheet("font-size:10px;")
        right.addWidget(self._comment_edit)

        btn_row = QHBoxLayout()

        self._confirm_btn = QPushButton("✓ Label bestätigen")
        self._confirm_btn.setStyleSheet(
            "background:#27AE60;color:white;padding:7px 14px;"
            "font-weight:bold;border-radius:4px;"
        )
        self._confirm_btn.setToolTip(
            "Existierendes Label als korrekt markieren und Flag entfernen"
        )
        self._confirm_btn.clicked.connect(self._confirm_current)
        btn_row.addWidget(self._confirm_btn)

        self._change_btn = QPushButton("✕ Label ändern…")
        self._change_btn.setStyleSheet(
            "background:#E74C3C;color:white;padding:7px 14px;"
            "font-weight:bold;border-radius:4px;"
        )
        self._change_btn.setToolTip(
            "Anderes Label auswählen und Flag entfernen"
        )
        self._change_btn.clicked.connect(self._change_current)
        btn_row.addWidget(self._change_btn)

        self._skip_btn = QPushButton("→ Überspringen")
        self._skip_btn.setToolTip("Zum nächsten unsicheren Bild, ohne Änderung")
        self._skip_btn.clicked.connect(self._skip_current)
        btn_row.addWidget(self._skip_btn)

        right.addLayout(btn_row)

        confirm_all_btn = QPushButton("✓ Alle bestätigen")
        confirm_all_btn.setFlat(True)
        confirm_all_btn.setStyleSheet("color:#2ECC71;font-size:10px;padding:2px;")
        confirm_all_btn.setToolTip("Alle verbleibenden unsicheren Labels als korrekt markieren")
        confirm_all_btn.clicked.connect(self._confirm_all)
        right.addWidget(confirm_all_btn)
        right.addStretch()

        splitter.addWidget(rw)
        splitter.setSizes([320, 600])
        v.addWidget(splitter)

        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.accept)
        v.addWidget(close_btn)

    # ------------------------------------------------------------------ data

    def _load_items(self) -> None:
        self._list.clear()
        uncertain = self.project.get_uncertain_images()
        n = len(uncertain)
        self._header.setText(
            f"{n} unsichere{'s' if n == 1 else ''} Label{'s' if n != 1 else ''} "
            "zur Überprüfung"
        )
        for img_path in uncertain:
            fname = os.path.basename(img_path)
            lbl = self.project.get_image_label(img_path)
            flag = self.project.get_label_flag(img_path)
            comment = flag.get("comment", "")
            lines = [fname, f"  Label: {lbl or '(kein)'}"]
            if comment:
                lines.append(f"  Kommentar: {comment}")
            item = QListWidgetItem("\n".join(lines))
            item.setData(Qt.UserRole, img_path)
            item.setForeground(QColor("#E67E22"))
            self._list.addItem(item)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)
        self._update_buttons()

    def _current_path(self) -> str:
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item else ""

    def _update_buttons(self) -> None:
        has = bool(self._current_path())
        self._confirm_btn.setEnabled(has)
        self._change_btn.setEnabled(has)
        self._skip_btn.setEnabled(has and self._list.count() > 1)

    # ------------------------------------------------------------------ slots

    def _on_row_changed(self, row: int) -> None:
        item = self._list.item(row)
        if not item:
            self._preview.setText("Kein Bild ausgewählt")
            self._img_info.clear()
            self._comment_edit.clear()
            self._update_buttons()
            return

        img_path = item.data(Qt.UserRole)
        lbl = self.project.get_image_label(img_path)
        flag = self.project.get_label_flag(img_path)
        comment = flag.get("comment", "")

        # Load preview
        pix = QPixmap(img_path)
        if not pix.isNull():
            pix = pix.scaled(440, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._preview.setPixmap(pix)
            self._preview.setText("")
        else:
            self._preview.clear()
            self._preview.setText("Bild konnte nicht geladen werden")

        self._img_info.setText(
            f"<b>{os.path.basename(img_path)}</b><br>"
            f"Label: <b>{lbl or '(kein)'}</b><br>"
            f"<span style='color:#555'>{img_path}</span>"
        )
        self._comment_edit.setText(comment)
        self._update_buttons()

    def _confirm_current(self) -> None:
        img_path = self._current_path()
        if not img_path:
            return
        self.project.clear_label_flag(img_path)
        self._load_items()
        if self._list.count() == 0:
            QMessageBox.information(
                self, "QA abgeschlossen",
                "Alle unsicheren Labels wurden überprüft."
            )

    def _change_current(self) -> None:
        img_path = self._current_path()
        if not img_path:
            return
        labels = list(self.project.labels.keys())
        if not labels:
            return

        menu = QMenu(self)
        for lbl_name in labels:
            action = menu.addAction(lbl_name)
            color = self.project.get_label_color(lbl_name)
            pix = QPixmap(14, 14)
            pix.fill(QColor(color))
            action.setIcon(QIcon(pix))
            action.setData(lbl_name)
        menu.addSeparator()
        no_lbl = menu.addAction("(kein Label)")
        no_lbl.setData("")

        chosen = menu.exec(
            self._change_btn.mapToGlobal(self._change_btn.rect().bottomLeft())
        )
        if chosen is not None:
            new_label = chosen.data()
            self.project.set_image_label(img_path, new_label if new_label else "")
            self.project.clear_label_flag(img_path)
            self._load_items()

    def _skip_current(self) -> None:
        cur = self._list.currentRow()
        next_row = (cur + 1) % self._list.count()
        self._list.setCurrentRow(next_row)

    def _confirm_all(self) -> None:
        uncertain = self.project.get_uncertain_images()
        if not uncertain:
            return
        reply = QMessageBox.question(
            self, "Alle bestätigen",
            f"Alle {len(uncertain)} unsicheren Labels als korrekt markieren\n"
            "und Flags entfernen?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            for img_path in list(uncertain):
                self.project.clear_label_flag(img_path)
            self._load_items()
            QMessageBox.information(self, "Fertig", "Alle Flags wurden entfernt.")
