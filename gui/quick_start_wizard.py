"""
Quick-Start Wizard — guides new users through their first project.

Shows automatically on first launch (controlled by QSettings key
"wizard/shown_v1").  Can be reopened at any time via Help menu or the
Dashboard quick-start panel.

Two workflows are supported:
  • Image Classification  (steps: project → data → label → train → classify)
  • Video / Anomaly       (steps: project → camera/video → train → monitor)

Signals
-------
navigate_requested(int)  — ask MainWindow to switch to this stack index
new_project_requested()  — ask MainWindow to open the New-Project dialog
open_project_requested() — ask MainWindow to open an existing project
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QWidget, QStackedWidget, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtGui import QFont, QColor, QPalette


# ── Step definitions ─────────────────────────────────────────────────────────

_IMAGE_STEPS = [
    {
        "icon": "🎉",
        "title_key": "wizard.img.s0.title",
        "desc_key":  "wizard.img.s0.desc",
        "title_de":  "Willkommen bei Picture Studio!",
        "desc_de": (
            "In wenigen Schritten erstellst du dein erstes Bildklassifikations-Projekt.\n\n"
            "Du wirst lernen wie du:\n"
            "  • Ein Projekt anlegst und Bilder importierst\n"
            "  • Bilder mit Labels versiehst\n"
            "  • Ein KI-Modell trainierst\n"
            "  • Neue Bilder automatisch klassifizierst\n\n"
            "Klicke auf 'Weiter' um zu beginnen — oder 'Überspringen' um sofort loszulegen."
        ),
        "title_en":  "Welcome to Picture Studio!",
        "desc_en": (
            "In just a few steps you'll create your first image classification project.\n\n"
            "You will learn how to:\n"
            "  • Create a project and import images\n"
            "  • Annotate images with labels\n"
            "  • Train an AI model\n"
            "  • Automatically classify new images\n\n"
            "Click 'Next' to begin — or 'Skip' to jump right in."
        ),
        "action_de":  None,
        "action_en":  None,
        "stack_idx":  None,
    },
    {
        "icon": "📁",
        "title_key": "wizard.img.s1.title",
        "title_de":  "Schritt 1 — Projekt erstellen",
        "desc_de": (
            "Zuerst legst du ein neues Projekt an.\n\n"
            "• Klicke auf '+ Neues Projekt erstellen'\n"
            "• Wähle 'Bildklassifikation' als Projekttyp\n"
            "• Gib einen Namen für dein Projekt ein\n\n"
            "Das Projekt speichert alle deine Bilder, Labels und Modelle an einem Ort."
        ),
        "title_en":  "Step 1 — Create a Project",
        "desc_en": (
            "First, create a new project.\n\n"
            "• Click '+ New Project'\n"
            "• Choose 'Image Classification' as project type\n"
            "• Enter a name for your project\n\n"
            "The project stores all your images, labels, and models in one place."
        ),
        "action_de":  "+ Neues Projekt erstellen",
        "action_en":  "+ Create New Project",
        "stack_idx":  None,    # triggers new_project_requested signal
        "action_type": "new_project",
    },
    {
        "icon": "🖼",
        "title_de":  "Schritt 2 — Bilder importieren",
        "desc_de": (
            "Jetzt importierst du deine Trainingsbilder.\n\n"
            "• Navigiere zur Seite 'Daten'\n"
            "• Klicke 'Bilder laden…' und wähle deinen Bildordner\n"
            "• Unterordner werden automatisch eingeschlossen\n\n"
            "Tipp: Für gute Ergebnisse braucht jede Klasse mindestens 20–50 Bilder."
        ),
        "title_en":  "Step 2 — Import Images",
        "desc_en": (
            "Now import your training images.\n\n"
            "• Navigate to the 'Data' page\n"
            "• Click 'Load Images…' and select your image folder\n"
            "• Subfolders are included automatically\n\n"
            "Tip: For good results, each class needs at least 20–50 images."
        ),
        "action_de":  "→ Zur Daten-Seite",
        "action_en":  "→ Go to Data Page",
        "stack_idx":  1,
        "action_type": "navigate",
    },
    {
        "icon": "🏷",
        "title_de":  "Schritt 3 — Labels anlegen & Bilder beschriften",
        "desc_de": (
            "Erstelle Labels (Klassen) und weise jedem Bild ein Label zu.\n\n"
            "• Gehe zu 'Projekt → Labels verwalten…' um Klassen anzulegen\n"
            "  (z.B. 'gut', 'schlecht', 'Kratzer')\n"
            "• Wechsle zur 'Labeling'-Seite\n"
            "• Klicke auf ein Bild und wähle das passende Label\n\n"
            "Tipp: Mit der Tastatur (1, 2, 3…) geht das Labeling besonders schnell!"
        ),
        "title_en":  "Step 3 — Create Labels & Annotate Images",
        "desc_en": (
            "Create labels (classes) and assign one to each image.\n\n"
            "• Go to 'Project → Manage Labels…' to create classes\n"
            "  (e.g. 'good', 'bad', 'scratch')\n"
            "• Switch to the 'Labeling' page\n"
            "• Click an image and pick the matching label\n\n"
            "Tip: Use keyboard shortcuts (1, 2, 3…) to label images quickly!"
        ),
        "action_de":  "→ Zum Labeling",
        "action_en":  "→ Go to Labeling",
        "stack_idx":  2,
        "action_type": "navigate",
    },
    {
        "icon": "🧠",
        "title_de":  "Schritt 4 — Modell trainieren",
        "desc_de": (
            "Jetzt trainierst du dein erstes KI-Modell!\n\n"
            "• Wechsle zur 'Training'-Seite\n"
            "• Die Standardeinstellungen sind für den Anfang optimal\n"
            "• Klicke 'Training starten'\n"
            "• Das Training dauert je nach Datenmenge 1–15 Minuten\n\n"
            "Architektur-Empfehlung:\n"
            "  EfficientNet-B3 ★ — bestes Ergebnis bei ausreichend Daten\n"
            "  DINOv2 ★★ — Foundation Model, ideal mit wenig Bildern\n"
            "              (< 100 pro Klasse, erster Start lädt ~85 MB)"
        ),
        "title_en":  "Step 4 — Train a Model",
        "desc_en": (
            "Now train your first AI model!\n\n"
            "• Switch to the 'Training' page\n"
            "• Default settings are fine for a first run\n"
            "• Click 'Start Training'\n"
            "• Training takes 1–15 minutes depending on dataset size\n\n"
            "Architecture recommendation:\n"
            "  EfficientNet-B3 ★ — best accuracy with enough data\n"
            "  DINOv2 ★★ — Foundation Model, ideal for few images\n"
            "               (< 100 per class, first run downloads ~85 MB)"
        ),
        "action_de":  "→ Zum Training",
        "action_en":  "→ Go to Training",
        "stack_idx":  3,
        "action_type": "navigate",
    },
    {
        "icon": "🔍",
        "title_de":  "Schritt 5 — Neue Bilder klassifizieren",
        "desc_de": (
            "Dein Modell ist bereit — jetzt kannst du neue Bilder klassifizieren!\n\n"
            "• Wechsle zur 'Klassifikation'-Seite\n"
            "• Lade ein einzelnes Bild oder einen ganzen Ordner\n"
            "• Das Modell sagt dir für jedes Bild die Klasse voraus\n\n"
            "Für große Mengen nutze 'Batch-Klassifikation' in der Seitenleiste.\n\n"
            "Du hast alles! 🎉 Viel Erfolg mit deinem Projekt."
        ),
        "title_en":  "Step 5 — Classify New Images",
        "desc_en": (
            "Your model is ready — now classify new images!\n\n"
            "• Switch to the 'Classification' page\n"
            "• Load a single image or an entire folder\n"
            "• The model predicts the class for each image\n\n"
            "For large volumes use 'Batch' in the sidebar.\n\n"
            "You're all set! 🎉 Good luck with your project."
        ),
        "action_de":  "→ Zur Klassifikation",
        "action_en":  "→ Go to Classification",
        "stack_idx":  5,
        "action_type": "navigate",
    },
]

_VIDEO_STEPS = [
    {
        "icon": "🎉",
        "title_de":  "Willkommen — Videoanalyse & Anomalieerkennung",
        "desc_de": (
            "In wenigen Schritten richtest du deine erste Anomalieerkennung ein.\n\n"
            "Du wirst lernen wie du:\n"
            "  • Ein Videoprojekt anlegst\n"
            "  • Normale Bilder (Gut-Bilder) aufnimmst\n"
            "  • Einen Anomalie-Detektor trainierst\n"
            "  • Live per Kamera Fehler erkennst\n\n"
            "Klicke auf 'Weiter' um zu beginnen."
        ),
        "title_en":  "Welcome — Video Analysis & Anomaly Detection",
        "desc_en": (
            "In a few steps you'll set up your first anomaly detection pipeline.\n\n"
            "You will learn how to:\n"
            "  • Create a video project\n"
            "  • Capture normal (good) images\n"
            "  • Train an anomaly detector\n"
            "  • Detect defects live via camera\n\n"
            "Click 'Next' to begin."
        ),
        "action_de":  None,
        "action_en":  None,
        "stack_idx":  None,
    },
    {
        "icon": "📁",
        "title_de":  "Schritt 1 — Videoprojekt erstellen",
        "desc_de": (
            "Erstelle ein neues Projekt und wähle 'Videoanalyse & Anomalie'.\n\n"
            "• Klicke auf '+ Neues Projekt erstellen'\n"
            "• Wähle 'Videoanalyse & Anomalie' als Projekttyp\n"
            "• Gib deinem Projekt einen Namen\n\n"
            "Die Seitenleiste wechselt danach automatisch auf Video-Seiten."
        ),
        "title_en":  "Step 1 — Create a Video Project",
        "desc_en": (
            "Create a new project and choose 'Video Analysis & Anomaly'.\n\n"
            "• Click '+ New Project'\n"
            "• Choose 'Video Analysis & Anomaly' as project type\n"
            "• Give your project a name\n\n"
            "The sidebar will automatically switch to video pages."
        ),
        "action_de":  "+ Neues Projekt erstellen",
        "action_en":  "+ Create New Project",
        "stack_idx":  None,
        "action_type": "new_project",
    },
    {
        "icon": "📷",
        "title_de":  "Schritt 2 — Gut-Bilder aufnehmen & Detektor trainieren",
        "desc_de": (
            "Trainiere den Detektor mit 'normalen' Bildern (ohne Fehler).\n\n"
            "• Wechsle zur Seite '🧠 Training' in der Seitenleiste\n"
            "• Klicke '🎬 Frames aufnehmen & Trainieren'\n"
            "• Verbinde deine Kamera und sammle mindestens 150 Bilder\n"
            "• Klicke 'Training starten' — dauert 2–5 Minuten\n\n"
            "Tipp: Achte auf konstante Beleuchtung — das verbessert die Ergebnisse erheblich."
        ),
        "title_en":  "Step 2 — Capture Normal Images & Train",
        "desc_en": (
            "Train the detector with 'normal' images (no defects).\n\n"
            "• Switch to the '🧠 Training' page in the sidebar\n"
            "• Click '🎬 Capture Frames & Train'\n"
            "• Connect your camera and collect at least 150 images\n"
            "• Click 'Start Training' — takes 2–5 minutes\n\n"
            "Tip: Consistent lighting greatly improves detection accuracy."
        ),
        "action_de":  "→ Zur Training-Seite",
        "action_en":  "→ Go to Training Page",
        "stack_idx":  17,
        "action_type": "navigate",
    },
    {
        "icon": "🧠",
        "title_de":  "Schritt 3 — Modell prüfen",
        "desc_de": (
            "Nach dem Training wird das Modell automatisch geladen.\n\n"
            "Die Training-Seite zeigt dir:\n"
            "• Den Modellnamen und den automatisch gesetzten Schwellwert\n"
            "• 'Bestehendes Modell laden' um ein bereits trainiertes Modell wiederzuverwenden\n\n"
            "Der Detektor lernt, wie 'normal' aussieht, und schlägt bei Abweichungen Alarm."
        ),
        "title_en":  "Step 3 — Check the Model",
        "desc_en": (
            "After training the model is loaded automatically.\n\n"
            "The Training page shows you:\n"
            "• The model name and the automatically set threshold\n"
            "• 'Load Existing Model' to reuse a previously trained model\n\n"
            "The detector learns what 'normal' looks like and alerts on deviations."
        ),
        "action_de":  "→ Zur Training-Seite",
        "action_en":  "→ Go to Training Page",
        "stack_idx":  17,
        "action_type": "navigate",
    },
    {
        "icon": "🎯",
        "title_de":  "Schritt 4 — Live-Monitoring starten",
        "desc_de": (
            "Dein Detektor ist bereit — starte das Live-Monitoring!\n\n"
            "• Auf der Kamera-Seite: klicke 'Scoring aktivieren'\n"
            "• Zeige ein normales Teil → niedriger Anomalie-Score\n"
            "• Zeige einen Fehler → hoher Score + roter Alarm-Banner\n"
            "• Passe den Schwellwert an deine Toleranz an\n\n"
            "Für mehrere Kameras gleichzeitig: nutze 'Multi-Kamera' in der Seitenleiste.\n\n"
            "Alles eingerichtet! 🎉"
        ),
        "title_en":  "Step 4 — Start Live Monitoring",
        "desc_en": (
            "Your detector is ready — start live monitoring!\n\n"
            "• On the Camera page: click 'Enable Scoring'\n"
            "• Show a normal part → low anomaly score\n"
            "• Show a defect → high score + red alarm banner\n"
            "• Adjust the threshold to match your tolerance\n\n"
            "For multiple cameras: use 'Multi-Camera' in the sidebar.\n\n"
            "All set! 🎉"
        ),
        "action_de":  "→ Zum Live-Monitoring",
        "action_en":  "→ Go to Live Monitoring",
        "stack_idx":  8,
        "action_type": "navigate",
    },
]


# ── Wizard Dialog ─────────────────────────────────────────────────────────────

class QuickStartWizard(QDialog):
    """
    Step-by-step first-start wizard.

    Signals
    -------
    navigate_requested(int)    — switch MainWindow to this stack index
    new_project_requested()    — open the New-Project dialog
    open_project_requested()   — open an existing project
    """

    navigate_requested    = Signal(int)
    new_project_requested = Signal()
    open_project_requested = Signal()

    # Set to True by MainWindow after first display so it isn't shown again.
    SETTINGS_KEY = "wizard/shown_v1"

    def __init__(self, workflow: str = "image", parent=None):
        super().__init__(parent)
        from utils.i18n import tr, current_lang
        self._lang = current_lang()
        self._steps = _IMAGE_STEPS if workflow == "image" else _VIDEO_STEPS
        self._step = 0

        self.setWindowTitle(
            "Schnellstart-Assistent" if self._lang == "de" else "Quick-Start Wizard"
        )
        self.setMinimumSize(780, 520)
        self.resize(820, 560)
        self._build_ui()
        self._show_step(0)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Left sidebar: step list ----
        sidebar = QWidget()
        sidebar.setFixedWidth(190)
        sidebar.setStyleSheet("background: #0D1117;")
        sv = QVBoxLayout(sidebar)
        sv.setContentsMargins(0, 24, 0, 16)
        sv.setSpacing(0)

        logo = QLabel("Picture Studio")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            "color: #388BFD; font-size: 13px; font-weight: bold;"
            " padding: 0 8px 16px 8px; border-bottom: 1px solid #21262D;"
        )
        sv.addWidget(logo)

        self._step_btns: list[QPushButton] = []
        for i, step in enumerate(self._steps):
            btn = QPushButton(f"  {step['icon']}  {self._step_label(i)}")
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.setFixedHeight(44)
            btn.setStyleSheet(
                "QPushButton { text-align: left; padding: 0 12px; border: none;"
                " color: #484F58; background: transparent; font-size: 11px; }"
                "QPushButton:checked { color: #E6EDF3; background: #161B22;"
                " font-weight: bold; border-left: 3px solid #388BFD; }"
                "QPushButton:enabled:!checked { color: #8B949E; }"
                "QPushButton:enabled:hover:!checked { background: #161B22; color: #E6EDF3; }"
            )
            btn.clicked.connect(lambda _, idx=i: self._jump_to(idx))
            sv.addWidget(btn)
            self._step_btns.append(btn)

        sv.addStretch()

        skip_lnk = QPushButton(
            "Wizard überspringen" if self._lang == "de" else "Skip wizard"
        )
        skip_lnk.setStyleSheet(
            "QPushButton { background: transparent; border: none;"
            " color: #545D68; font-size: 10px; padding: 4px; }"
            "QPushButton:hover { color: #8B949E; }"
        )
        skip_lnk.clicked.connect(self.reject)
        sv.addWidget(skip_lnk)

        root.addWidget(sidebar)

        # ---- Vertical divider ----
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet("color: #21262D;")
        root.addWidget(div)

        # ---- Right: content area ----
        right = QWidget()
        right.setStyleSheet("background: #0D1117;")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(32, 32, 32, 24)
        rv.setSpacing(0)

        # Step pages (stacked)
        self._stack = QStackedWidget()
        for step in self._steps:
            self._stack.addWidget(self._build_step_page(step))
        rv.addWidget(self._stack, 1)

        # Bottom navigation
        nav = QHBoxLayout()
        nav.setSpacing(8)

        self._back_btn = QPushButton(
            "◀ Zurück" if self._lang == "de" else "◀ Back"
        )
        self._back_btn.setFixedWidth(100)
        self._back_btn.setStyleSheet(
            "QPushButton { background: #21262D; color: #8B949E; border: none;"
            " border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:hover { background: #30363D; color: #E6EDF3; }"
            "QPushButton:disabled { color: #30363D; }"
        )
        self._back_btn.clicked.connect(self._go_back)

        self._next_btn = QPushButton(
            "Weiter ▶" if self._lang == "de" else "Next ▶"
        )
        self._next_btn.setFixedWidth(120)
        self._next_btn.setStyleSheet(
            "QPushButton { background: #1F6FEB; color: white; border: none;"
            " border-radius: 6px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #388BFD; }"
        )
        self._next_btn.clicked.connect(self._go_next)

        close_btn = QPushButton(
            "Schließen" if self._lang == "de" else "Close"
        )
        close_btn.setFixedWidth(100)
        close_btn.setStyleSheet(
            "QPushButton { background: #21262D; color: #8B949E; border: none;"
            " border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:hover { background: #30363D; color: #E6EDF3; }"
        )
        close_btn.clicked.connect(self.accept)

        nav.addWidget(self._back_btn)
        nav.addStretch()
        nav.addWidget(close_btn)
        nav.addWidget(self._next_btn)
        rv.addLayout(nav)

        root.addWidget(right, 1)

    def _build_step_page(self, step: dict) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: transparent;")
        v = QVBoxLayout(page)
        v.setSpacing(20)
        v.setContentsMargins(0, 0, 0, 24)

        lang = self._lang
        icon_lbl = QLabel(step["icon"])
        icon_lbl.setStyleSheet("font-size: 48px; background: transparent;")
        icon_lbl.setAlignment(Qt.AlignLeft)
        v.addWidget(icon_lbl)

        title = QLabel(step.get(f"title_{lang}", step.get("title_de", "")))
        title.setStyleSheet(
            "color: #E6EDF3; font-size: 18px; font-weight: bold; background: transparent;"
        )
        title.setWordWrap(True)
        v.addWidget(title)

        desc = QLabel(step.get(f"desc_{lang}", step.get("desc_de", "")))
        desc.setStyleSheet(
            "color: #8B949E; font-size: 12px; line-height: 1.5; background: transparent;"
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        v.addWidget(desc, 1)

        action_text = step.get(f"action_{lang}")
        if action_text:
            action_btn = QPushButton(action_text)
            action_btn.setFixedHeight(40)
            action_btn.setMaximumWidth(320)
            action_btn.setStyleSheet(
                "QPushButton { background: #238636; color: white; border: none;"
                " border-radius: 6px; padding: 8px 20px; font-size: 13px; font-weight: bold; }"
                "QPushButton:hover { background: #2EA043; }"
            )
            step_ref = step  # capture for lambda
            action_btn.clicked.connect(
                lambda _=False, s=step_ref: self._do_action(s)
            )
            v.addWidget(action_btn)

        return page

    # ── Navigation ────────────────────────────────────────────────────────────

    def _step_label(self, idx: int) -> str:
        step = self._steps[idx]
        lang = self._lang
        full = step.get(f"title_{lang}", step.get("title_de", ""))
        # Truncate long step labels for sidebar
        return full[:26] + "…" if len(full) > 27 else full

    def _show_step(self, idx: int) -> None:
        self._step = idx
        self._stack.setCurrentIndex(idx)
        last = len(self._steps) - 1

        # Unlock all visited steps
        for i, btn in enumerate(self._step_btns):
            btn.setEnabled(i <= idx)
            btn.setChecked(i == idx)

        self._back_btn.setEnabled(idx > 0)
        label = (
            ("Fertig!" if self._lang == "de" else "Finish!")
            if idx == last
            else ("Weiter ▶" if self._lang == "de" else "Next ▶")
        )
        self._next_btn.setText(label)

    def _go_next(self) -> None:
        if self._step >= len(self._steps) - 1:
            self.accept()
        else:
            self._show_step(self._step + 1)

    def _go_back(self) -> None:
        if self._step > 0:
            self._show_step(self._step - 1)

    def _jump_to(self, idx: int) -> None:
        self._show_step(idx)

    def _do_action(self, step: dict) -> None:
        atype = step.get("action_type")
        if atype == "new_project":
            self.new_project_requested.emit()
            self.accept()
        elif atype == "navigate":
            idx = step.get("stack_idx")
            if idx is not None:
                self.navigate_requested.emit(idx)
                self.accept()

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def should_show_on_startup() -> bool:
        """Return True if the wizard has not yet been shown on this installation."""
        s = QSettings("ImageLabelingStudio", "ILS")
        return not s.value(QuickStartWizard.SETTINGS_KEY, False, type=bool)

    @staticmethod
    def mark_shown() -> None:
        """Record that the wizard has been displayed; won't auto-show again."""
        s = QSettings("ImageLabelingStudio", "ILS")
        s.setValue(QuickStartWizard.SETTINGS_KEY, True)
