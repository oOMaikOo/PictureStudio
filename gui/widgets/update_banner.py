"""Non-blocking update / What's-new banner for the main window."""
from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from utils.i18n import tr


class UpdateBanner(QFrame):
    """Slim top banner: update available and/or local What's-new blurb."""

    dismissed = Signal()
    open_whats_new = Signal()
    open_release = Signal(str)  # tag or URL hint

    def __init__(self, parent=None):
        super().__init__(parent)
        self._release_tag = ""
        self.setObjectName("UpdateBanner")
        self.setStyleSheet(
            "#UpdateBanner {"
            "  background: #1F3A5F; border-bottom: 1px solid #388BFD;"
            "}"
            "QLabel { color: #E6EDF3; font-size: 12px; }"
            "QPushButton {"
            "  background: #1F6FEB; color: white; border: none;"
            "  border-radius: 4px; padding: 4px 10px;"
            "}"
            "QPushButton:hover { background: #388BFD; }"
            "QPushButton#dismiss {"
            "  background: transparent; color: #8B949E; padding: 2px 8px;"
            "}"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 12, 6)
        self._msg = QLabel("")
        self._msg.setWordWrap(True)
        row.addWidget(self._msg, stretch=1)

        self._whats_btn = QPushButton(tr("update.whats_new_btn"))
        self._whats_btn.clicked.connect(self.open_whats_new.emit)
        row.addWidget(self._whats_btn)

        self._release_btn = QPushButton(tr("update.release_btn"))
        self._release_btn.clicked.connect(
            lambda: self.open_release.emit(self._release_tag)
        )
        row.addWidget(self._release_btn)

        dismiss = QPushButton("✕")
        dismiss.setObjectName("dismiss")
        dismiss.setFixedWidth(28)
        dismiss.clicked.connect(self._on_dismiss)
        row.addWidget(dismiss)
        self.hide()

    def show_update(self, local: str, remote_tag: str, whats_new: str = "") -> None:
        """Show banner for a newer *remote_tag*; optionally attach What's-new."""
        self._release_tag = remote_tag
        self._msg.setText(
            tr("update.available", local=local, remote=remote_tag.lstrip("v"))
        )
        self._release_btn.setVisible(True)
        self._whats_btn.setVisible(bool(whats_new))
        self._whats_new_cache = whats_new
        self.show()

    def show_whats_new_only(self, whats_new: str) -> None:
        """Show banner for local What's-new after an upgrade (no remote)."""
        self._release_tag = ""
        self._whats_new_cache = whats_new
        self._msg.setText(tr("update.whats_new_hint"))
        self._release_btn.setVisible(False)
        self._whats_btn.setVisible(True)
        self.show()

    @property
    def whats_new_text(self) -> str:
        return getattr(self, "_whats_new_cache", "") or ""

    def _on_dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()
