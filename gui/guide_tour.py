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
# Tour steps per page index (0–9)
# 0=Dashboard, 1=Daten, 2=Labeling, 3=Training, 4=Modelle,
# 5=Klassifikation, 6=Export, 7=Einstellungen, 8=Kamera, 9=Batch
# Each step: (title, description, button_text_to_highlight | None)
# ---------------------------------------------------------------------------
TOUR_STEPS = {
    0: [  # Dashboard
        ("Willkommen bei Picture Studio",
         "Das Dashboard zeigt den Projektstand auf einen Blick:\n"
         "Bilder, Labels, Klassen und letzte Trainingsmetriken.\n\n"
         "Die Navigationsleiste links ist gesperrt — erst ein\n"
         "Projekt öffnen oder neu anlegen um sie freizuschalten.",
         None),
        ("Neues Projekt anlegen",
         "Klicke '+ Neues Projekt' oder Strg+N.\n\n"
         "Im Dialog wählst du:\n"
         "• Projektname und Beschreibung\n"
         "• Projekttyp: 📸 Bildklassifikation\n"
         "  oder 🎬 Videoanalyse & Anomalie\n\n"
         "Der Typ bestimmt welche Seiten in der\n"
         "Sidebar angezeigt werden.",
         "Neues Projekt"),
        ("Vorhandenes Projekt öffnen",
         "Klicke 'Projekt öffnen…' oder Strg+O um eine\n"
         "bestehende Projektdatei (.json) zu laden.\n\n"
         "Zuletzt geöffnete Projekte:\n"
         "Menü → Datei → Zuletzt geöffnet\n\n"
         "Nach dem Laden ist die Sidebar freigeschaltet\n"
         "und zeigt die passenden Seiten für den Projekttyp.",
         "Projekt öffnen"),
        ("Dashboard-Statistiken",
         "Die Karten zeigen auf einen Blick:\n"
         "• Bilder gesamt / gelabelt / ungelabelt\n"
         "• ROIs und Klassen\n"
         "• Anzahl Trainingsläufe\n\n"
         "Darunter: Klassenverteilung mit Balkendiagramm\n"
         "und Warnungen bei Ungleichgewicht oder\n"
         "zu wenig Bildern pro Klasse.",
         None),
    ],
    1: [  # Daten
        ("Daten-Seite",
         "Hier importierst du Bilder oder Videos,\n"
         "analysierst den Datensatz und exportierst\n"
         "Annotationen in verschiedene Formate.",
         None),
        ("Bilder laden",
         "Klicke 'Bilder laden…' → Ordner wählen.\n"
         "Alle .jpg, .png, .bmp, .tiff im Ordner\n"
         "werden automatisch hinzugefügt.\n\n"
         "Alternativ: Bilder direkt ins Fenster ziehen\n"
         "(Drag & Drop funktioniert überall in der App).",
         "Bilder laden"),
        ("Video importieren",
         "Klicke 'Video importieren…' um ein Video\n"
         "(MP4, AVI, MOV, MKV, WebM) zu importieren.\n\n"
         "Frame-Intervall einstellen:\n"
         "• Alle 1 Frame = volle Framerate\n"
         "• Alle 5 Frames = ca. 6 Bilder/s bei 30 fps\n\n"
         "Die extrahierten Frames werden als PNG\n"
         "ins Projektverzeichnis gespeichert.",
         "Video importieren"),
        ("Datensatz analysieren",
         "Klicke 'Dataset analysieren' um zu prüfen:\n"
         "• Fehlende Dateien\n"
         "• MD5-Duplikate (identische Bilder)\n"
         "• Klassenungleichgewicht\n"
         "• Bildformat- und Größenstatistiken\n\n"
         "Bei starkem Ungleichgewicht: auf der\n"
         "Trainingsseite Klassenausgleich aktivieren.",
         "Dataset analysieren"),
        ("Annotationen exportieren",
         "Exportiere Annotationen (ROIs + Labels) als:\n"
         "• COCO JSON → Object-Detection-Frameworks\n"
         "• YOLO TXT → Ultralytics / Darknet\n"
         "• CSV → eigene Tools / Tabellenkalkulation",
         "COCO"),
    ],
    2: [  # Labeling (nur Bildprojekte)
        ("Labeling-Seite",
         "Weise Bildern Klassen zu und zeichne ROIs\n"
         "(Regions of Interest) für die Klassifikation.\n\n"
         "Links: Thumbnail-Liste  |  Mitte: Editor\n"
         "Rechts: ROI-Details und Label-Zuweisung",
         None),
        ("Bilder laden & Label definieren",
         "Bilder zuerst auf der Daten-Seite laden\n"
         "oder hier 'Ordner laden…' klicken.\n\n"
         "Labels definieren:\n"
         "Menü → Projekt → Labels verwalten… (Strg+L)\n"
         "• Namen und Farbe pro Klasse festlegen\n"
         "• Mindestens 2 Klassen für Training nötig",
         "Ordner laden"),
        ("Bild auswählen & schnell labeln",
         "Bild in der Thumbnail-Liste anklicken.\n\n"
         "Schnellzuweisung: Taste 1–9 drücken\n"
         "(Reihenfolge entspricht der Label-Liste)\n\n"
         "Alternativ: Label-Dropdown oben rechts\n\n"
         "Navigation:\n"
         "N = nächstes Bild  |  P = vorheriges Bild",
         None),
        ("ROI zeichnen",
         "Werkzeug in der Toolbar wählen:\n"
         "• R = Rechteck (häufigste Wahl)\n"
         "• E = Ellipse\n"
         "• G = Polygon (für unregelmäßige Formen)\n\n"
         "Im Bild ziehen um ROI zu zeichnen.\n"
         "Esc = Abbrechen  |  Entf = Löschen\n"
         "Strg+C / Strg+V = Kopieren / Einfügen",
         "Rechteck"),
        ("ROI-Label zuweisen",
         "ROI in der rechten Liste auswählen.\n"
         "Label im Dropdown darunter wählen.\n"
         "'ROI-Label zuweisen' klicken.\n\n"
         "Schneller: ROI auswählen + Taste 1–9\n\n"
         "'ROIs dieses Bildes → alle Bilder' überträgt\n"
         "die gleichen ROI-Positionen auf alle Bilder.",
         "ROI-Label zuweisen"),
        ("Segmentierungsmaske malen",
         "Für pixelgenaue Annotation:\n"
         "Tab '🎨 Segmentierungsmaske' im mittleren\n"
         "Bereich wählen.\n\n"
         "Linksklick = malen  |  Rechtsklick = löschen\n"
         "Scroll = Zoom  |  Klasse und Pinselgröße\n"
         "über die obere Toolbar wählen.\n\n"
         "'Maske speichern' speichert als PNG\n"
         "neben der Bilddatei.",
         "Maske speichern"),
    ],
    3: [  # Training (nur Bildprojekte)
        ("Training-Seite",
         "Trainiere ein CNN-Modell auf deinen\n"
         "annotierten Bildern.\n\n"
         "Links: Konfiguration\n"
         "Rechts: Live-Kurven & Metriken\n\n"
         "Voraussetzung: Bilder müssen gelabelt sein.",
         None),
        ("Architektur wählen",
         "Im Architektur-Dropdown:\n"
         "• ResNet-18 — schnell, guter Startpunkt\n"
         "• MobileNetV2 — effizient, gut für CPU\n"
         "• EfficientNet-B0 — beste Genauigkeit\n"
         "• SimpleCNN — kein GPU nötig, für Tests\n\n"
         "Alle außer SimpleCNN nutzen vortrainierte\n"
         "ImageNet-Gewichte (Transfer Learning).",
         None),
        ("Hyperparameter einstellen",
         "Empfohlene Startwerte:\n"
         "• Epochen: 20–50\n"
         "• Lernrate: 0.001\n"
         "• Batch-Größe: 32 (GPU) / 8–16 (CPU)\n"
         "• Gerät: 'auto' wählt GPU > MPS > CPU\n"
         "• Early Stopping: 5 Epochen\n"
         "• Klassenausgleich: bei ungleichen Klassen\n"
         "  (WeightedRandomSampler) aktivieren",
         None),
        ("Training starten",
         "Klicke 'Training starten'.\n"
         "Kurven (Loss, Accuracy) aktualisieren live.\n"
         "Das beste Checkpoint wird automatisch\n"
         "gespeichert. Jederzeit abbrechen mit\n"
         "'Training stoppen'.\n\n"
         "Nach dem Training:\n"
         "• HTML- oder Excel-Bericht erstellen\n"
         "• Modell auf der Modelle-Seite verwalten",
         "Training starten"),
        ("SSH-Ferntraining auf GPU-Server",
         "Training auf einem externen Server:\n"
         "1. Einstellungen → SSH-Profil anlegen\n"
         "   (Host, Benutzer, SSH-Key-Pfad)\n"
         "2. SSH-Ferntraining-Checkbox aktivieren\n"
         "3. Profil wählen → 'Verbindung testen'\n"
         "4. Grünes Signal → Training starten\n\n"
         "Der Log streamt live. Das Checkpoint\n"
         "wird automatisch heruntergeladen.",
         "Verbindung testen"),
    ],
    4: [  # Modelle
        ("Modellbibliothek",
         "Alle trainierten Modelle des Projekts\n"
         "auf einen Blick.\n\n"
         "Tab 📦 Modellbibliothek — aktuelle Modelle\n"
         "Tab 📊 Run-History — alle Trainingsläufe\n"
         "im zeitlichen Verlauf vergleichen",
         None),
        ("Modell laden & als Best markieren",
         "Modell in der Tabelle auswählen.\n\n"
         "'In Inferenz laden' → Modell auf der\n"
         "Klassifikations-Seite verwenden.\n\n"
         "'Als Best markieren' → Modell als\n"
         "Standard für das Projekt setzen.\n\n"
         "Tipp: F1-Score bei ungleichen Klassen\n"
         "aussagekräftiger als Accuracy.",
         "In Inferenz laden"),
        ("ONNX & TorchScript exportieren",
         "Modell für andere Systeme exportieren:\n\n"
         "'Als ONNX exportieren' → .onnx (Opset 17)\n"
         "Einsatz in: ONNX Runtime, OpenCV DNN,\n"
         "TensorRT, Python, C++, C#\n\n"
         "'Als TorchScript exportieren' → .pt\n"
         "Einsatz in: PyTorch C++ API, mobile Apps",
         "Als ONNX exportieren"),
        ("Modelle vergleichen",
         "Mehrere Modelle auswählen (Strg+Klick)\n"
         "→ 'Ausgewählte vergleichen'\n\n"
         "Zeigt Accuracy, F1, Architektur\n"
         "und Best-Markierung im Überblick.\n\n"
         "Run-History-Tab: alle Läufe nach Datum\n"
         "sortiert mit Gerät, Epochen, Train-Acc.",
         "Ausgewählte vergleichen"),
    ],
    5: [  # Klassifikation (nur Bildprojekte)
        ("Klassifikations-Seite",
         "Neue (unbekannte) Bilder mit einem\n"
         "trainierten Modell bewerten.\n\n"
         "Ergebnis: Top-K Vorhersagen mit\n"
         "Konfidenz-Farbkodierung.\n\n"
         "Modell zuerst laden.",
         None),
        ("Modell laden",
         "'Modell laden (.pth)' → Datei wählen\n\n"
         "Oder direkt von der Modelle-Seite:\n"
         "Modell auswählen → 'In Inferenz laden'\n\n"
         "Ensemble — mehrere Modelle kombinieren:\n"
         "'+ Modell hinzufügen' → alle geladenen\n"
         "Modelle werden gemittelt (stabilere\n"
         "Vorhersagen bei schwierigen Bildern).",
         "Modell laden"),
        ("Ordner klassifizieren",
         "1. 'Ordner…' → Ordner mit neuen Bildern\n"
         "2. TTA (Test-Time Augmentation):\n"
         "   Spinner auf 3–5 → mehrere augmentierte\n"
         "   Versionen je Bild, Durchschnitt gebildet\n"
         "   → genauere Ergebnisse bei Grenzfällen\n"
         "3. 'Alle Bilder klassifizieren'\n\n"
         "Farben: Grün >90% | Gelb 70–90% | Rot <70%",
         "Alle Bilder klassifizieren"),
        ("Unsichere Vorhersagen prüfen",
         "Tab 'Niedrige Konfidenz' zeigt alle Bilder\n"
         "unter dem eingestellten Schwellwert.\n\n"
         "Diese Bilder eignen sich für:\n"
         "• Manuelle Überprüfung\n"
         "• Nachtrainieren (Active Learning)\n"
         "• Hinzufügen zur Trainingsdatenbank",
         None),
        ("Automatisch labeln",
         "Hochkonfidente Ergebnisse direkt als\n"
         "Projekt-Labels übernehmen:\n\n"
         "1. Mindest-Konfidenz einstellen (z. B. 0.90)\n"
         "2. 'Auf Projekt anwenden' klicken\n"
         "3. Bilder mit Konfidenz ≥ Schwellwert\n"
         "   bekommen automatisch ein Label\n\n"
         "Danach: Labeling-Seite zur Kontrolle öffnen.",
         "Auf Projekt anwenden"),
    ],
    6: [  # Export
        ("Excel-Export",
         "Klassifikationsergebnisse in eine\n"
         "formatierte Excel-Datei exportieren.\n"
         "Spalten sind frei konfigurierbar.",
         None),
        ("Ergebnisse laden & Datei wählen",
         "'Ergebnisse aus letzter Inferenz laden'\n"
         "übernimmt die aktuellen Ergebnisse.\n\n"
         "Dann Zieldatei wählen:\n"
         "'Datei wählen…' → vorhandene Excel-Datei\n"
         "'Neue Datei erstellen' → neue Excel-Datei\n\n"
         "Modus:\n"
         "• Anhängen: fügt Zeilen unten an\n"
         "• Überschreiben: erstellt neue Datei",
         "Ergebnisse aus letzter"),
        ("Spalten konfigurieren & exportieren",
         "In der Spalten-Tabelle:\n"
         "• Checkbox: Spalte ein-/ausschalten\n"
         "• Doppelklick auf Name: umbenennen\n\n"
         "Verfügbare Spalten z. B.:\n"
         "Dateiname, Vorhersage, Konfidenz,\n"
         "Top-2/3, Zeitstempel, Modellpfad\n\n"
         "→ 'Excel exportieren' klicken.",
         "Excel exportieren"),
    ],
    7: [  # Einstellungen
        ("Einstellungen",
         "Alle Einstellungen werden automatisch\n"
         "gespeichert (QSettings) und beim\n"
         "nächsten Start wiederhergestellt.",
         None),
        ("Theme & Darstellung",
         "Theme: Dunkel (Standard) oder Hell\n"
         "Schriftgröße: 7–16 pt\n"
         "Thumbnail-Größe: kleinere Werte\n"
         "beschleunigen das Laden im Editor",
         None),
        ("Autosave & Backup",
         "Autosave-Intervall: Standard 5 Minuten\n"
         "Deaktivieren → manuell Strg+S nutzen.\n\n"
         "Backup aktivieren: erstellt bei jedem\n"
         "Speichern eine Sicherungskopie (.bak)\n"
         "im Projektverzeichnis.",
         None),
        ("REST-API & Web-Dashboard",
         "REST-Server: Einstellungen → API starten\n"
         "(Standard-Port: 8765)\n\n"
         "Endpunkte:\n"
         "GET  /api/status   → Projektstatus\n"
         "GET  /api/labels   → Klassenliste\n"
         "POST /api/classify → Bild bewerten\n"
         "GET  /api/scores   → Live Score-Puffer\n"
         "GET  /api/events   → Anomalie-Events\n"
         "GET  /dashboard    → Live-Dashboard\n\n"
         "'📊 Dashboard' → öffnet Browser-Dashboard\n"
         "mit Live-Monitoring (aktualisiert alle 3 s).",
         "Dashboard"),
        ("SSH-Profile anlegen",
         "'Profil hinzufügen' für SSH-Ferntraining:\n"
         "• Name (frei wählbar)\n"
         "• Host / IP-Adresse\n"
         "• Benutzername\n"
         "• SSH-Key-Pfad (empfohlen)\n"
         "• Port (Standard: 22)\n\n"
         "Profil auf der Trainingsseite auswählen.",
         "Profil hinzufügen"),
    ],
    8: [  # Live & Anomalie (Videoprojekte, Stack-Index 8)
        ("Live & Anomalie – Überblick",
         "Diese Seite ist der Kern des Videoprojekts:\n"
         "Live-Kamerabild + Anomalieerkennung\n"
         "in einem Workflow.\n\n"
         "Ablauf:\n"
         "1. Kamera verbinden\n"
         "2. Normalframes aufnehmen\n"
         "3. Autoencoder trainieren\n"
         "4. Live-Scoring aktivieren",
         None),
        ("Kamera verbinden",
         "Kamera-Index wählen:\n"
         "• 0 = erste USB-Kamera\n"
         "• 1, 2 … = weitere Kameras\n\n"
         "Klicke '▶ Kamera starten'.\n"
         "Der Live-Stream erscheint im Vorschaubereich.\n\n"
         "Kamera wechseln: erst stoppen,\n"
         "dann Index ändern und neu starten.",
         "Kamera starten"),
        ("Normalframes aufnehmen",
         "Normalprozess vor die Kamera bringen.\n\n"
         "'⚙ Aufnahme & Anomalie-Erkennung…' öffnen.\n\n"
         "'Normalframes aufnehmen' starten:\n"
         "• Mindestens 100 Frames (besser: 200–300)\n"
         "• Kamera NICHT bewegen während Aufnahme\n"
         "• Beleuchtung konstant halten\n\n"
         "Kein Beispiel einer Anomalie nötig —\n"
         "nur normale Frames!",
         "Aufnahme"),
        ("Autoencoder trainieren",
         "'→ Trainieren' klicken (Standard: 40 Epochen).\n\n"
         "Der Autoencoder lernt den Normalzustand.\n"
         "Schwellwert wird automatisch berechnet:\n"
         "Mittelwert + 2,5 × Standardabweichung\n"
         "der Trainings-Rekonstruktionsfehler.\n\n"
         "Nach dem Training: Schwellwert kalibrieren\n"
         "mit '📊 Schwellwert kalibrieren…'.",
         None),
        ("Schwellwert kalibrieren",
         "'📊 Schwellwert kalibrieren…' öffnet ein\n"
         "Histogramm der Score-Verteilung.\n\n"
         "Vorschläge:\n"
         "• µ+1σ → sehr sensitiv (wenige Abweichungen)\n"
         "• µ+2σ → ausgewogen (Empfehlung)\n"
         "• µ+3σ → nur grobe Abweichungen\n\n"
         "Für Fertigungsüberwachung: mit µ+2σ\n"
         "starten und bei Bedarf anpassen.",
         None),
        ("Live-Scoring aktivieren",
         "'→ Live-Scoring' einschalten.\n\n"
         "Anzeige im Live-Bild:\n"
         "✅ Grüner Rahmen = Normalzustand\n"
         "🔴 Roter Rahmen + Alarm = Abweichung\n\n"
         "Heatmap zeigt WELCHER Bereich abweicht.\n"
         "Bounding Box markiert die auffälligste Zone.\n\n"
         "Optional: 'Anomalie-Frames automatisch\n"
         "speichern' zur Dokumentation aktivieren.",
         None),
        ("Modell sichern & exportieren",
         "Trainiertes Modell sichern:\n"
         "'Modell speichern' → .pth-Datei\n"
         "'Modell laden' → bei nächster Sitzung\n"
         "sofort einsatzbereit ohne Neutraining.\n\n"
         "Für Deployment in anderen Systemen:\n"
         "'ONNX exportieren' → .onnx\n"
         "'TorchScript exportieren' → .pt\n\n"
         "ONNX läuft in ONNX Runtime, OpenCV,\n"
         "TensorRT und vielen anderen Frameworks.",
         None),
        ("Monitoring-Profil exportieren",
         "Für 24/7-Betrieb ohne GUI:\n\n"
         "1. Modell trainieren und speichern\n"
         "2. '📋 Profil exportieren…' klicken\n"
         "3. JSON-Datei speichern\n\n"
         "Das Profil enthält:\n"
         "• Modellpfad + Format\n"
         "• Schwellwert & Glättung\n"
         "• Kameraquelle & Speicherordner\n"
         "• ROI-Koordinaten\n"
         "• MQTT-Konfiguration\n\n"
         "Dann headless starten:\n"
         "python scripts/monitor_daemon.py\n"
         "  --profile monitor_profile.json",
         "Profil exportieren"),
        ("Event-Log & MQTT-Alarm",
         "Während Live-Scoring:\n\n"
         "Event-Log (CSV): Jeder Alarm wird\n"
         "automatisch geloggt:\n"
         "• Zeitstempel, Score, Schwellwert\n"
         "• Pfad zum gespeicherten Frame\n"
         "'Log öffnen' → CSV in Standardprogramm\n\n"
         "MQTT: Einstellungen → MQTT konfigurieren\n"
         "Jeder Alarm wird als JSON-Payload\n"
         "an den Broker publiziert.\n\n"
         "pip install paho-mqtt",
         "Log öffnen"),
    ],
    9: [  # Batch-Inferenz
        ("Batch-Inferenz",
         "Klassifiziere einen ganzen Ordner\n"
         "mit einem trainierten Modell in einem Durchlauf.\n\n"
         "Ergebnis: sortierbare Tabelle mit\n"
         "Dateiname, Klasse, Konfidenz, Fehler\n"
         "und CSV-Export.",
         None),
        ("Modell laden",
         "Modell aus dem Projekt wählen\n"
         "(Dropdown zeigt alle Trainingsläufe)\n"
         "oder externe .pth-Datei laden.\n\n"
         "'Ausgewähltes Modell laden' → bereit.",
         "Ausgewähltes Modell laden"),
        ("Bilder auswählen",
         "'Ordner wählen…' → Ordner mit Bildern\n"
         "oder 'Projektbilder verwenden' um alle\n"
         "Bilder des aktuellen Projekts zu nehmen.\n\n"
         "Konfidenz-Filter: nur Ergebnisse über\n"
         "dem Schwellwert werden angezeigt.",
         "Ordner wählen"),
        ("Batch starten & exportieren",
         "'Batch starten' → alle Bilder werden\n"
         "nacheinander klassifiziert.\n\n"
         "Ergebnisse in der Tabelle:\n"
         "• Klick auf Spaltenköpfe → sortieren\n"
         "• Rot = unter Min. Confidence\n\n"
         "'Als CSV exportieren' → Tabelle als\n"
         "CSV-Datei speichern.",
         "Batch starten"),
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
