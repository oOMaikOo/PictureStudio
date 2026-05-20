# 🏷 Labeling & ROIs

> **PictureStudio v2.3.0** — Bilder annotieren, Labels zuweisen und Regionen einzeichnen

---

# 🏷 Labeling & ROIs

Bilder annotieren, Labels zuweisen, Regionen einzeichnen.

## Labels verwalten

**Labels hinzufügen****
*Projekt → Labels verwalten…* (`Strg+L`)

Name + Farbe definieren. Labels können jederzeit umbenannt oder neu eingefärbt werden.

Mindestens 2 Labels für das Training erforderlich.

## Bilder labeln

Schnelles Labeln****
1. Bild in der Liste anklicken

2. `1`–`9` drücken = Label 1–9 zuweisen

3. `N` = nächstes Bild, `P` = vorheriges Bild

Label-Filter: Nur Bilder einer bestimmten Klasse anzeigen.

## ROIs zeichnen

Rechteck** `R` – Im Bild ziehen

**Ellipse** `E` – Im Bild ziehen

**Polygon** `G` – Punkte klicken, Doppelklick zum Abschließen

## ROI bearbeiten

| Taste | Aktion |
|---|---|
| Entf | Ausgewählte ROI löschen |
| Strg+C | ROI kopieren |
| Strg+V | ROI einfügen |
| Pfeiltasten | ROI um 2 px verschieben |
| Esc | Zeichnen abbrechen |

> 💡 **ROIs auf alle Bilder übertragen:** Wenn alle Bilder dieselbe Aufnahmeposition haben, zeichne ROIs einmal und nutze *ROIs dieses Bildes → alle Bilder*.

## Segmentierungsmaske

**Tab 🎨 Segmentierungsmaske**

Für pixelgenaue Annotation:

Linksklick = malen | Rechtsklick = löschen | Scroll = Zoom

Klasse und Pinselgröße über die Toolbar wählen.

*Maske speichern* speichert als PNG neben der Bilddatei.
