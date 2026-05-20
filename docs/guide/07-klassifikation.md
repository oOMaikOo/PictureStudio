# 🔍 Klassifikation

> **PictureStudio v2.3.0** — Neue Bilder mit dem trainierten Modell klassifizieren

---

# 🔍 Klassifikation

Neue Bilder mit dem trainierten Modell klassifizieren.

## Schritt-für-Schritt

**1 – Modell laden****
*Modell laden (.pth)* → Modelldatei wählen.

Schneller: Modelle-Seite → *In Inferenz laden*

Ensemble:** mehrere Modelle per *+ Modell hinzufügen* kombinieren für stabilere Vorhersagen.

**2 – Bildordner wählen****
*Ordner…* → Ordner mit neuen Bildern wählen.

TTA (Test-Time Augmentation):** Spinner auf 3–5 → mehrere augmentierte Versionen
je Bild, Durchschnitt = genauere Ergebnisse bei Grenzfällen.

**3 – Klassifizieren****
*Alle Bilder klassifizieren* → Ergebnis mit Top-K Vorhersagen.

Farbkodierung: Grün >90% | Gelb 70–90% | Rot 4 – Unsichere prüfen****
Niedrig-Konfidenz-Tab**: alle Bilder unter dem eingestellten Schwellwert (Standard: 70%).**
Diese manuell prüfen und ggf. ins Training aufnehmen.

5 – Automatisch labeln**

Hochkonfidente Ergebnisse direkt als Projekt-Labels übernehmen:

Mindest-Konfidenz einstellen → *Auf Projekt anwenden*

Danach Labeling-Seite zur Kontrolle öffnen.

## Konfidenz-Schwellwert

> 💡 Einstellungen → Schwelle 'unsicher' (Standard: 0.70).
 Niedrigerer Wert = mehr Bilder gelten als sicher.
 Höherer Wert = strengere Qualitätskontrolle, mehr im Niedrig-Konfidenz-Tab.
