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
        super().__init__(parent)
        self.setWindowTitle("Neues Projekt erstellen")
        self.setMinimumWidth(460)
        self.setModal(True)

        self.project_name: str = ""
        self.project_type: str = "image"
        self.description: str = ""

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # Name
        layout.addWidget(QLabel("Projektname: *"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("z. B. Qualitätskontrolle Linie 3")
        self._name_edit.setMinimumHeight(32)
        layout.addWidget(self._name_edit)

        # Description
        layout.addWidget(QLabel("Beschreibung (optional):"))
        self._desc_edit = QLineEdit()
        self._desc_edit.setPlaceholderText("Kurze Beschreibung des Projekts")
        self._desc_edit.setMinimumHeight(32)
        layout.addWidget(self._desc_edit)

        # Project type
        type_group = QGroupBox("Projekttyp wählen")
        tl = QVBoxLayout(type_group)
        tl.setSpacing(6)

        self._img_btn = self._make_type_btn(
            "📸   Bildklassifikation",
            "Bilder importieren, labeln, Klassifikationsmodelle trainieren und auswerten.",
            checked=True,
        )
        self._img_btn.clicked.connect(lambda: self._select_type("image"))
        tl.addWidget(self._img_btn)

        self._vid_btn = self._make_type_btn(
            "🎬   Videoanalyse & Anomalie",
            "Videos importieren, Frames extrahieren, Live-Kamera und Anomalieerkennung.",
            checked=False,
        )
        self._vid_btn.clicked.connect(lambda: self._select_type("video"))
        tl.addWidget(self._vid_btn)

        layout.addWidget(type_group)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.setFixedHeight(34)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton("Projekt erstellen")
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
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Fehler", "Bitte einen Projektnamen eingeben.")
            self._name_edit.setFocus()
            return
        self.project_name = name
        self.description = self._desc_edit.text().strip()
        self.accept()
