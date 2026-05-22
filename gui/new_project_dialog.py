"""
Dialog for creating a new project: choose name, type (image vs video), description.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QMessageBox,
)
from PySide6.QtCore import Qt


class NewProjectDialog(QDialog):
    """Replaces the simple QInputDialog for new project creation."""

    def __init__(self, parent=None):
        from utils.i18n import tr
        super().__init__(parent)
        self.setWindowTitle(tr("new_project.title"))
        self.setMinimumWidth(460)
        self.setModal(True)

        self.project_name: str = ""
        self.project_type: str = "image"
        self.description: str = ""

        self._build_ui()

    def _build_ui(self) -> None:
        from utils.i18n import tr
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # Name
        layout.addWidget(QLabel(tr("new_project.name_label")))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText(tr("new_project.name_placeholder"))
        self._name_edit.setMinimumHeight(32)
        layout.addWidget(self._name_edit)

        # Description
        layout.addWidget(QLabel(tr("new_project.desc_label")))
        self._desc_edit = QLineEdit()
        self._desc_edit.setPlaceholderText(tr("new_project.desc_placeholder"))
        self._desc_edit.setMinimumHeight(32)
        layout.addWidget(self._desc_edit)

        # Project type
        type_group = QGroupBox(tr("new_project.type_group"))
        tl = QVBoxLayout(type_group)
        tl.setSpacing(6)

        self._img_btn = self._make_type_btn(
            tr("new_project.image_type"),
            tr("new_project.image_desc"),
            checked=True,
        )
        self._img_btn.clicked.connect(lambda: self._select_type("image"))
        tl.addWidget(self._img_btn)

        self._vid_btn = self._make_type_btn(
            tr("new_project.video_type"),
            tr("new_project.video_desc"),
            checked=False,
        )
        self._vid_btn.clicked.connect(lambda: self._select_type("video"))
        tl.addWidget(self._vid_btn)

        layout.addWidget(type_group)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton(tr("common.cancel"))
        cancel_btn.setFixedHeight(34)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton(tr("new_project.create_btn"))
        ok_btn.setDefault(True)
        ok_btn.setFixedHeight(34)
        ok_btn.setStyleSheet(
            "QPushButton { background:#1976D2; color:white; padding:0 18px;"
            " border-radius:4px; font-weight:bold; }"
            "QPushButton:hover { background:#1565C0; }"
        )
        ok_btn.clicked.connect(self._accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _make_type_btn(self, title: str, subtitle: str, checked: bool) -> QPushButton:
        btn = QPushButton(title)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setFixedHeight(64)
        btn.setToolTip(subtitle)
        btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 10px 16px;
                border: 2px solid #444;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:checked {
                border: 2px solid #1976D2;
                background: #1565C025;
                color: #42A5F5;
            }
            QPushButton:hover:!checked {
                border-color: #777;
            }
        """)
        return btn

    def _select_type(self, t: str) -> None:
        self.project_type = t
        self._img_btn.setChecked(t == "image")
        self._vid_btn.setChecked(t == "video")

    def _accept(self) -> None:
        from utils.i18n import tr
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, tr("common.error"), tr("new_project.no_name_msg"))
            self._name_edit.setFocus()
            return
        self.project_name = name
        self.description = self._desc_edit.text().strip()
        self.accept()
