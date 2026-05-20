# 📤 Excel-Export

> **PictureStudio v2.3.0** — Klassifikationsergebnisse als formatierte Excel-Datei exportieren

---

# 📤 Excel-Export

Klassifikationsergebnisse als formatierte Excel-Datei exportieren.

## Schritt-für-Schritt

**1 – Ergebnisse laden****
*Ergebnisse aus letzter Inferenz laden* – übernimmt die aktuellen
Klassifikationsergebnisse aus dem Projekt.

2 – Zieldatei wählen****
*Datei wählen…* → vorhandene Excel-Datei (für Anhängen-Modus)

*Neue Datei erstellen* → neue Excel-Datei anlegen

3 – Spalten konfigurieren****
In der Tabelle:

• Checkbox = Spalte ein/ausschalten

• Doppelklick auf Name = Spalte umbenennen

Reihenfolge der Tabelle = Reihenfolge in der Excel-Datei

4 – Modus & Exportieren****
Überschreiben:** Neue Datei / bestehende Datei komplett ersetzen**
Anhängen:** Zeilen an bestehende Datei hinzufügen (gleiche Spalten!)**
Dann *Excel exportieren* klicken.

> ⚠️ Voraussetzung:** `pip install openpyxl` muss installiert sein.

## Besonderheiten

  - Zeilen unter Konfidenz-Schwellwert werden **rot** markiert
  - Kopfzeilen werden fett und farbig formatiert
  - Top-2 und Top-3 Vorhersagen können optional hinzugefügt werden
