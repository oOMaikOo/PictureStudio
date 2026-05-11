"""
Interactive guided tour – floating panel that walks through each page step by step,
highlights the relevant button/widget with an overlay frame.
"""
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QAbstractButton, QWidget, QApplication,
)
from PySide6.QtCore import Qt, QTimer, QPoint, QRect
from PySide6.QtGui import QFont, QColor

# ---------------------------------------------------------------------------
# Tour steps per page index (0–7)
# Each step: (title, description, button_text_to_highlight | None)
# button_text can be a partial match
# ---------------------------------------------------------------------------
TOUR_STEPS = {
    0: [  # Dashboard
        ("Willkommen auf dem Dashboard",
         "Das Dashboard zeigt dir den Projektstand auf einen Blick:\n"
         "Anzahl Bilder, gelabelte Bilder, Klassen und letzte Trainingsmetriken.\n\n"
         "Öffne zunächst ein bestehendes Projekt oder erstelle ein neues.",
         None),
        ("Neues Projekt anlegen",
         "Klicke 'Neues Projekt erstellen' um zu starten.\n"
         "Du vergibst einen Namen und wählst den Speicherort.\n"
         "Das Projekt wird als .json-Datei gespeichert.",
         "Neues Projekt erstellen"),
        ("Vorhandenes Projekt öffnen",
         "Klicke 'Projekt öffnen' um eine bestehende Projektdatei zu laden.\n"
         "Zuletzt geöffnete Projekte findest du unter:\n"
         "Menü → Datei → Zuletzt geöffnet",
         "Projekt öffnen"),
        ("Bilder direkt von der Kamera aufnehmen",
         "Menü Datei → Kamera aufnehmen… (Strg+K)\n"
         "Verbinde eine USB- oder IP-Kamera und nimm Einzelbilder\n"
         "oder Burst-Serien direkt ins Projekt auf.\n\n"
         "Zeitstempel: Systemzeit und -datum lassen sich ins\n"
         "Live-Bild einblenden und/oder dauerhaft einbrennen.",
         None),
    ],
    1: [  # Daten
        ("Daten-Seite",
         "Hier lädst du Bilder, analysierst den Datensatz und\n"
         "exportierst Annotationen in verschiedene Formate.\n\n"
         "Beginne mit dem Laden deiner Bilder.",
         None),
        ("Bilder laden",
         "Klicke 'Bilder laden…' und wähle einen Ordner.\n"
         "Alle .jpg, .png, .bmp und .tiff Dateien werden\n"
         "dem Projekt automatisch hinzugefügt.",
         "Bilder laden"),
        ("Datensatz analysieren",
         "Klicke 'Dataset analysieren' um zu prüfen:\n"
         "• Fehlende Dateien\n"
         "• MD5-Duplikate (identische Bilder)\n"
         "• Klassenungleichgewicht\n"
         "• Bildformat- und Größenstatistiken",
         "Dataset analysieren"),
        ("Fehlende Dateien prüfen",
         "Prüft ob alle Bilddateien noch vorhanden sind.\n"
         "Verschobene Dateien können mit\n"
         "'Bildpfade korrigieren…' aktualisiert werden.",
         "Fehlende Dateien"),
        ("Annotationen exportieren",
         "Exportiere Annotationen als:\n"
         "• COCO JSON → Object-Detection-Frameworks\n"
         "• YOLO TXT → Ultralytics/Darknet\n"
         "• CSV → eigene Tools / Tabellenkalkulation",
         "COCO"),
    ],
    2: [  # Labeling
        ("Labeling-Seite",
         "Hier annotierst du Bilder mit Labels und\n"
         "zeichnest ROIs (Regions of Interest).\n\n"
         "Links: Bildliste  |  Mitte: Editor  |  Rechts: ROI-Details",
         None),
        ("Bilder laden",
         "Klicke 'Ordner laden…' um Bilder direkt\n"
         "in den Editor zu laden.\n"
         "Alternativ: Bilder zuerst auf der Daten-Seite hinzufügen.",
         "Ordner laden"),
        ("Bild auswählen & labeln",
         "Klicke in der Bildliste auf ein Bild.\n"
         "Drücke 1–9 für schnelle Label-Zuweisung,\n"
         "oder nutze das Label-Dropdown oben rechts.\n"
         "Navigation: N = nächstes, P = vorheriges Bild",
         None),
        ("ROI zeichnen – Rechteck",
         "Klicke 'Rechteck' in der Toolbar oder drücke R.\n"
         "Im Bild ziehen um ein Rechteck zu zeichnen.\n"
         "Weitere Modi: E = Ellipse, G = Polygon\n"
         "Abbrechen: Esc | Löschen: Entf",
         "Rechteck"),
        ("ROI-Label zuweisen",
         "Wähle eine ROI aus der Liste rechts.\n"
         "Setze das Label im Dropdown darunter.\n"
         "Klicke 'ROI-Label zuweisen' um es zu speichern.\n"
         "Schneller: ROI auswählen + Taste 1–9",
         "ROI-Label zuweisen"),
        ("ROIs auf alle Bilder übertragen",
         "Hast du ROIs für ein typisches Bild gezeichnet?\n"
         "Klicke 'ROIs dieses Bildes → alle Bilder' um\n"
         "dieselben ROI-Positionen auf alle Bilder anzuwenden.",
         "ROIs dieses Bildes"),
    ],
    3: [  # Training
        ("Training-Seite",
         "Trainiere ein CNN-Modell mit deinen annotierten Bildern.\n"
         "Links: Konfiguration  |  Rechts: Fortschritt & Kurven\n\n"
         "Stelle sicher, dass du Bilder gelabelt hast.",
         None),
        ("Architektur wählen",
         "Wähle im Architektur-Dropdown:\n"
         "• ResNet-18: schnell, guter Ausgangspunkt\n"
         "• MobileNetV2: effizient für CPU\n"
         "• EfficientNet-B0: beste Genauigkeit\n"
         "• SimpleCNN: schnell für erste Tests ohne GPU",
         None),
        ("Hyperparameter einstellen",
         "Empfohlene Startwerte:\n"
         "• Epochen: 20–30\n"
         "• Lernrate: 0.001\n"
         "• Batch-Größe: 32 (GPU) / 8 (CPU)\n"
         "• Gerät: 'auto' wählt automatisch GPU/MPS/CPU\n"
         "• Early Stopping: 5 (stoppt wenn keine Verbesserung)",
         None),
        ("Training starten",
         "Klicke 'Training starten' um zu beginnen.\n"
         "Trainingskurven und Metriken aktualisieren sich live.\n"
         "Das beste Modell wird automatisch gespeichert.\n"
         "Abbruch jederzeit mit 'Training stoppen'.",
         "Training starten"),
        ("Berichte exportieren",
         "Nach dem Training Bericht erstellen:\n"
         "• 'HTML-Bericht erstellen…' → vollständiger Report\n"
         "• 'Excel-Bericht erstellen…' → für Dokumentation\n"
         "Enthält: Metriken, Kurven, Konfusionsmatrix",
         "HTML-Bericht"),
        ("SSH-Ferntraining",
         "Für Training auf einem externen GPU-Server:\n"
         "1. SSH-Profil in Einstellungen anlegen\n"
         "2. SSH-Ferntraining-Checkbox aktivieren\n"
         "3. Profil wählen → 'Verbindung testen'\n"
         "Grünes Signal = bereit. Dann Training starten.",
         "Verbindung testen"),
    ],
    4: [  # Modelle
        ("Modellbibliothek",
         "Hier findest du alle trainierten Modelle des Projekts.\n"
         "Vergleiche Modelle, lade sie für Inferenz\n"
         "oder exportiere als ONNX.",
         None),
        ("Modell für Klassifikation laden",
         "Wähle ein Modell in der Tabelle.\n"
         "Klicke 'In Inferenz laden' um es auf der\n"
         "Klassifikations-Seite zu verwenden.\n"
         "Tipp: F1-Score ist bei ungleichen Klassen\n"
         "aussagekräftiger als Accuracy.",
         "In Inferenz laden"),
        ("ONNX exportieren",
         "ONNX ermöglicht Einsatz in anderen Frameworks:\n"
         "TensorRT, OpenCV DNN, ONNX Runtime.\n"
         "Wähle ein Modell → 'Als ONNX exportieren'.",
         "Als ONNX exportieren"),
        ("Modelle vergleichen",
         "Wähle mehrere Modelle (Strg+Klick) und\n"
         "klicke 'Ausgewählte vergleichen' für eine\n"
         "Gegenüberstellung aller Metriken.",
         "Ausgewählte vergleichen"),
    ],
    5: [  # Klassifikation
        ("Klassifikations-Seite",
         "Klassifiziere neue Bilder mit einem trainierten Modell.\n"
         "Ergebnis: Top-3 Vorhersagen mit Konfidenz-Farbkodierung.\n\n"
         "Zuerst ein Modell laden.",
         None),
        ("Modell laden",
         "Klicke 'Modell laden (.pth)' und wähle\n"
         "eine Modelldatei aus dem Dateisystem.\n"
         "Alternativ: Direkt von der Modelle-Seite laden\n"
         "via 'In Inferenz laden'.",
         "Modell laden"),
        ("Bildordner wählen",
         "Klicke 'Ordner…' und wähle den Ordner\n"
         "mit den zu klassifizierenden Bildern.\n"
         "Einzelne Bilder: 'Einzelbild klassifizieren'",
         "Ordner…"),
        ("Alle Bilder klassifizieren",
         "Klicke 'Alle Bilder klassifizieren'.\n"
         "Konfidenz-Farbkodierung:\n"
         "Grün >90% | Gelb 70–90% | Rot <70%\n"
         "Unsichere Vorhersagen → Niedrig-Konfidenz-Tab",
         "Alle Bilder klassifizieren"),
        ("Ergebnisse filtern & exportieren",
         "Filtere nach Label oder Konfidenz.\n"
         "Niedrig-Konfidenz-Tab: alle Bilder unter Schwellwert.\n"
         "Für Excel-Export: zur Export-Seite wechseln.",
         "Filter anwenden"),
    ],
    6: [  # Export
        ("Excel-Export",
         "Exportiere Klassifikationsergebnisse\n"
         "in eine formatierte Excel-Datei.\n"
         "Spalten sind frei konfigurierbar.",
         None),
        ("Ergebnisse laden",
         "Klicke 'Ergebnisse aus letzter Inferenz laden'\n"
         "um die aktuellen Klassifikationsergebnisse\n"
         "aus dem Projekt zu übernehmen.",
         "Ergebnisse aus letzter"),
        ("Zieldatei wählen",
         "'Datei wählen…' → vorhandene Excel-Datei\n"
         "'Neue Datei erstellen' → neue Excel-Datei\n"
         "Modus 'Anhängen': fügt Zeilen hinzu\n"
         "Modus 'Überschreiben': erstellt neu",
         "Datei wählen"),
        ("Spalten konfigurieren & exportieren",
         "Spalten in der Tabelle:\n"
         "• Checkbox: Spalte ein/ausschalten\n"
         "• Doppelklick: Spalte umbenennen\n\n"
         "Dann 'Excel exportieren' klicken.",
         "Excel exportieren"),
    ],
    7: [  # Einstellungen
        ("Einstellungen",
         "Alle Einstellungen werden automatisch\n"
         "gespeichert und beim nächsten Start\n"
         "wiederhergestellt.",
         None),
        ("Theme & Darstellung",
         "Wähle 'Dunkel' oder 'Hell'.\n"
         "Schriftgröße: 7–16 pt\n"
         "Thumbnail-Größe: beeinflusst Ladezeit\n"
         "im Labeling-Editor",
         None),
        ("Autosave konfigurieren",
         "Autosave-Intervall: Standard 5 Minuten.\n"
         "Deaktivieren → manuell Strg+S nutzen.\n"
         "Backup: erstellt .bak Datei bei jedem Speichern.",
         None),
        ("SSH-Profile anlegen",
         "Klicke 'Profil hinzufügen' für einen\n"
         "neuen SSH-Eintrag:\n"
         "• Name, Host, Benutzername\n"
         "• Key-Pfad (empfohlen) oder Passwort\n"
         "• Port (Standard: 22)",
         "Profil hinzufügen"),
        ("Einstellungen speichern",
         "Klicke 'Einstellungen speichern' um alle\n"
         "Änderungen dauerhaft zu übernehmen.",
         "Einstellungen speichern"),
    ],
}


# ---------------------------------------------------------------------------
# Highlight overlay
# ---------------------------------------------------------------------------

class HighlightOverlay(QFrame):
    """Transparent orange-bordered frame that highlights a widget."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(243,156,18,15);"
            "border: 3px solid #F39C12;"
            "border-radius: 7px;"
        )
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()

    def highlight(self, widget: QWidget | None) -> None:
        if widget is None or not widget.isVisible():
            self.hide()
            return
        pad = 5
        pos = widget.mapTo(self.parent(), QPoint(0, 0))
        self.setGeometry(
            pos.x() - pad, pos.y() - pad,
            widget.width() + 2 * pad, widget.height() + 2 * pad,
        )
        self.raise_()
        self.show()


# ---------------------------------------------------------------------------
# Guide tour panel
# ---------------------------------------------------------------------------

class GuideTour(QFrame):
    """Floating step-by-step guide panel anchored to the main window."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self._main = main_window
        self._steps: list = []
        self._index: int = 0
        self._page_widget: QWidget | None = None

        self._overlay = HighlightOverlay(main_window)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            GuideTour {
                background: #1C2A3A;
                border: 2px solid #2980B9;
                border-radius: 10px;
            }
        """)
        self.setFixedWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        self._header_lbl = QLabel("Geführte Tour")
        self._header_lbl.setStyleSheet("color:#F39C12; font-weight:bold; font-size:11px;")
        header_row.addWidget(self._header_lbl)
        header_row.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(
            "QPushButton{background:transparent;color:#aaa;border:none;font-size:14px;}"
            "QPushButton:hover{color:white;}"
        )
        close_btn.clicked.connect(self.stop)
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)

        # Title
        self._title_lbl = QLabel()
        self._title_lbl.setStyleSheet("color:white; font-weight:bold; font-size:13px;")
        self._title_lbl.setWordWrap(True)
        layout.addWidget(self._title_lbl)

        # Description
        self._desc_lbl = QLabel()
        self._desc_lbl.setStyleSheet("color:#BDC3C7; font-size:11px; line-height:1.5;")
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(90)
        self._desc_lbl.setAlignment(Qt.AlignTop)
        layout.addWidget(self._desc_lbl)

        # Step counter
        self._counter_lbl = QLabel()
        self._counter_lbl.setStyleSheet("color:#7F8C8D; font-size:10px;")
        self._counter_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._counter_lbl)

        # Navigation buttons
        nav = QHBoxLayout()
        self._back_btn = QPushButton("← Zurück")
        self._back_btn.setStyleSheet(
            "QPushButton{background:#2C3E50;color:#BDC3C7;border:1px solid #34495E;"
            "border-radius:5px;padding:5px 10px;}"
            "QPushButton:hover{background:#34495E;color:white;}"
            "QPushButton:disabled{color:#555;}"
        )
        self._back_btn.clicked.connect(self._prev_step)
        nav.addWidget(self._back_btn)

        self._next_btn = QPushButton("Weiter →")
        self._next_btn.setStyleSheet(
            "QPushButton{background:#2980B9;color:white;border:none;"
            "border-radius:5px;padding:5px 10px;font-weight:bold;}"
            "QPushButton:hover{background:#3498DB;}"
        )
        self._next_btn.clicked.connect(self._next_step)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Timer to keep highlight position in sync when window resizes
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(250)
        self._sync_timer.timeout.connect(self._refresh_highlight)

        self.hide()

    # ------------------------------------------------------------------ public

    def start(self, page_index: int, page_widget: QWidget) -> None:
        """Start tour for the given page."""
        self._steps = TOUR_STEPS.get(page_index, [])
        self._page_widget = page_widget
        self._index = 0
        if not self._steps:
            return
        self._update_ui()
        self._reposition()
        self.show()
        self.raise_()
        self._overlay.raise_()
        self._sync_timer.start()

    def stop(self) -> None:
        self._sync_timer.stop()
        self._overlay.hide()
        self.hide()

    # ------------------------------------------------------------------ steps

    def _next_step(self) -> None:
        if self._index < len(self._steps) - 1:
            self._index += 1
            self._update_ui()
        else:
            self.stop()

    def _prev_step(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._update_ui()

    def _update_ui(self) -> None:
        title, desc, btn_text = self._steps[self._index]
        total = len(self._steps)
        self._header_lbl.setText(f"Geführte Tour  •  Schritt {self._index + 1} von {total}")
        self._title_lbl.setText(title)
        self._desc_lbl.setText(desc)
        self._counter_lbl.setText("─" * 36)
        self._back_btn.setEnabled(self._index > 0)
        last = self._index == total - 1
        self._next_btn.setText("Tour beenden" if last else "Weiter →")
        self._next_btn.setStyleSheet(
            "QPushButton{background:%s;color:white;border:none;"
            "border-radius:5px;padding:5px 10px;font-weight:bold;}"
            "QPushButton:hover{background:%s;}"
            % (("#27AE60", "#2ECC71") if last else ("#2980B9", "#3498DB"))
        )
        self._highlight_step(btn_text)
        self.adjustSize()
        self._reposition()

    def _highlight_step(self, btn_text: str | None) -> None:
        if not btn_text or not self._page_widget:
            self._overlay.hide()
            return
        target = self._find_widget(btn_text)
        self._overlay.highlight(target)

    def _find_widget(self, text: str) -> QWidget | None:
        if not self._page_widget:
            return None
        text_lower = text.lower()
        for btn in self._page_widget.findChildren(QAbstractButton):
            if text_lower in btn.text().lower() and btn.isVisible():
                return btn
        # Fallback: search all visible children with matching text
        for lbl in self._page_widget.findChildren(QLabel):
            if text_lower in lbl.text().lower() and lbl.isVisible():
                return lbl
        return None

    def _refresh_highlight(self) -> None:
        if not self.isVisible():
            return
        _, _, btn_text = self._steps[self._index]
        self._highlight_step(btn_text)

    # ------------------------------------------------------------------ positioning

    def _reposition(self) -> None:
        """Keep panel in bottom-right corner of main window."""
        parent = self._main
        margin = 16
        x = parent.width() - self.width() - margin
        y = parent.height() - self.height() - margin - 30  # 30 = statusbar approx
        self.move(max(0, x), max(0, y))
