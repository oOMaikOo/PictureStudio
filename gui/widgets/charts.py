"""
Training curve charts – uses matplotlib if available, ASCII sparklines otherwise.
"""
from typing import Dict, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QBrush

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class TrainingCurvesWidget(QWidget):
    """Shows loss + accuracy curves for a training run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        if HAS_MPL:
            self._fig = Figure(figsize=(8, 4), facecolor="#1a1a2e")
            self._canvas = FigureCanvas(self._fig)
            self._layout.addWidget(self._canvas)
        else:
            self._text = QTextEdit()
            self._text.setReadOnly(True)
            self._text.setFont(QFont("Courier New", 9))
            self._layout.addWidget(self._text)

    def update_curves(self, history: Dict) -> None:
        if HAS_MPL:
            self._update_mpl(history)
        else:
            self._update_ascii(history)

    def _update_mpl(self, history: Dict) -> None:
        self._fig.clear()
        ax1 = self._fig.add_subplot(1, 2, 1)
        ax2 = self._fig.add_subplot(1, 2, 2)
        style = {"facecolor": "#16213e"}
        for ax in [ax1, ax2]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#aaa")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")
            ax.title.set_color("#eee")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
        if epochs:
            ax1.plot(epochs, history.get("train_loss", []), color="#3498DB", label="Train")
            ax1.plot(epochs, history.get("val_loss", []), color="#E74C3C", label="Val")
            ax1.set_title("Loss")
            ax1.set_xlabel("Epoche")
            ax1.legend(facecolor="#16213e", labelcolor="#eee")

            ax2.plot(epochs, [v * 100 for v in history.get("train_acc", [])], color="#3498DB", label="Train")
            ax2.plot(epochs, [v * 100 for v in history.get("val_acc", [])], color="#2ECC71", label="Val")
            ax2.set_title("Accuracy (%)")
            ax2.set_xlabel("Epoche")
            ax2.legend(facecolor="#16213e", labelcolor="#eee")

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    def _update_ascii(self, history: Dict) -> None:
        lines = ["=== Trainingskurven ===\n"]
        for key, label in [
            ("train_loss", "Train-Loss"), ("val_loss", "Val-Loss"),
            ("train_acc", "Train-Acc %"), ("val_acc", "Val-Acc %"),
        ]:
            values = history.get(key, [])
            if not values:
                continue
            disp = [v * 100 for v in values] if "acc" in key else values
            lines.append(f"{label}:")
            lines.append("  " + _spark(disp))
            lines.append(
                f"  Zuletzt: {disp[-1]:.2f}  Min: {min(disp):.2f}  Max: {max(disp):.2f}\n"
            )
        self._text.setPlainText("\n".join(lines))


class ConfusionMatrixWidget(QWidget):
    """
    Interactive confusion matrix rendered as a QTableWidget.
    Clicking any cell with count > 0 emits cell_clicked(true_idx, pred_idx).
    """

    cell_clicked = Signal(int, int)   # true_class_idx, pred_class_idx

    _COL_DIAG  = QColor("#1b5e20")
    _COL_OFF   = QColor("#6d1212")
    _COL_ZERO  = QColor("#1a1a2e")
    _COL_HDR   = QColor("#1565C0")
    _COL_TEXT_BRIGHT = QColor("#ffffff")
    _COL_TEXT_DIM    = QColor("#555555")

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._hint = QLabel(
            "Klicke auf eine Zelle, um die dazugehörigen Bilder anzuzeigen."
        )
        self._hint.setStyleSheet("color:#888;font-size:10px;padding:2px 4px;")
        layout.addWidget(self._hint)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(True)
        self._table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self._table)

        self._cm: List[List[int]] = []
        self._class_names: List[str] = []

    def set_matrix(self, cm: List[List[int]], class_names: List[str]) -> None:
        self._cm = cm
        self._class_names = class_names

        n = len(class_names)
        if not cm or not n:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._hint.setText("Keine Konfusionsmatrix verfügbar.")
            return

        self._table.setRowCount(n)
        self._table.setColumnCount(n)
        self._table.setHorizontalHeaderLabels(class_names)
        self._table.setVerticalHeaderLabels(class_names)

        max_val = max(val for row in cm for val in row) or 1

        for i, row in enumerate(cm):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val) if val > 0 else "·")
                item.setTextAlignment(Qt.AlignCenter)
                if i == j:
                    bg = self._COL_DIAG
                    fg = self._COL_TEXT_BRIGHT
                elif val > 0:
                    # Intensity proportional to value
                    intensity = int(80 + 120 * val / max_val)
                    bg = QColor(intensity, 18, 18)
                    fg = self._COL_TEXT_BRIGHT
                else:
                    bg = self._COL_ZERO
                    fg = self._COL_TEXT_DIM
                item.setBackground(QBrush(bg))
                item.setForeground(QBrush(fg))
                if val > 0:
                    item.setToolTip(
                        f"Wahr: {class_names[i]}\nVorhergesagt: {class_names[j]}\n"
                        f"Anzahl: {val}\n→ Klicken zum Anzeigen"
                    )
                self._table.setItem(i, j, item)

        self._hint.setText(
            "T = Wahre Klasse (Zeilen), P = Vorhergesagt (Spalten)  ·  "
            "Klicke eine Zelle zum Anzeigen der Bilder."
        )

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if row < len(self._cm) and col < len(self._cm[row]):
            if self._cm[row][col] > 0:
                self.cell_clicked.emit(row, col)


def _spark(values: List[float], width: int = 60) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    chars = "▁▂▃▄▅▆▇█"
    out = ""
    for v in values[-width:]:
        idx = int((v - mn) / rng * (len(chars) - 1))
        out += chars[max(0, min(idx, len(chars) - 1))]
    return out
