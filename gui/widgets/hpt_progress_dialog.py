"""
Reusable HPT progress dialog with scrolling per-trial log.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QTextEdit, QPushButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class HptProgressDialog(QDialog):
    """
    Modal dialog for Optuna HPT runs.

    Shows:
      - a status label (current trial / best value so far)
      - a progress bar (0 … n_trials)
      - a scrolling monospace log (one line per completed trial)
      - an Abbrechen / Schließen button

    Usage
    -----
    dlg = HptProgressDialog(n_trials, parent=self)
    dlg.setModal(True)
    dlg.show()

    hpt.progress.connect(lambda cur, tot, val: dlg.update_progress(cur, tot, val_str))
    hpt.log.connect(dlg.append_log)
    hpt.finished.connect(lambda _: dlg.set_done())
    dlg.rejected.connect(hpt.stop)
    """

    def __init__(self, n_trials: int, title: str = "Hyperparameter-Suche", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.setMinimumHeight(400)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        v = QVBoxLayout(self)
        v.setSpacing(8)

        self._status = QLabel("Initialisierung…")
        v.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, n_trials)
        self._bar.setValue(0)
        v.addWidget(self._bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setLineWrapMode(QTextEdit.NoWrap)
        v.addWidget(self._log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn = QPushButton("Abbrechen")
        self._btn.clicked.connect(self.reject)
        btn_row.addWidget(self._btn)
        v.addLayout(btn_row)

    # ------------------------------------------------------------------ public

    def update_progress(self, cur: int, tot: int, best_str: str) -> None:
        """Update progress bar and status label after each trial."""
        self._bar.setValue(cur)
        self._status.setText(f"Versuch {cur} / {tot}  —  {best_str}")

    def append_log(self, text: str) -> None:
        """Append one log line and auto-scroll to bottom."""
        self._log.append(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_done(self) -> None:
        """Switch button from Abbrechen to Schließen once the run finishes."""
        self._btn.setText("Schließen")
        self._btn.clicked.disconnect()
        self._btn.clicked.connect(self.accept)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.show()  # re-apply window flags
