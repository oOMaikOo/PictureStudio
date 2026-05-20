# 🧠 Training

> **PictureStudio v2.3.0** — CNN-Modell lokal oder remote per SSH trainieren

---

# 🧠 Training

CNN-Modell trainieren – lokal oder auf GPU-Server via SSH.

## Konfiguration

| Parameter | Empfehlung | Beschreibung |
|---|---|---|
| Architektur | ResNet-18 zum Start | Modellarchitektur |
| Epochen | 20–30 | Anzahl Trainingsdurchläufe |
| Lernrate | 0.001 | Schrittgröße der Gewichts-Updates |
| Batch-Größe | 32 (GPU) / 8–16 (CPU) | Bilder pro Trainingsschritt |
| Gerät | auto | Automatisch: GPU > MPS (Apple) > CPU |
| Early Stopping | 5–7 | Stopp nach N Epochen ohne Verbesserung |
| LR-Scheduler | cosine | Lernrate automatisch anpassen |
| Klassenausgleich | bei Ungleichgewicht | WeightedRandomSampler ausgleichen |

## Hyperparameter-Suche (optional)

Schaltfläche **⚙ Hyperparameter-Suche…** klicken (`pip install optuna` erforderlich).
Optuna testet automatisch Kombinationen aus Lernrate, Batch-Größe, Architektur und Optimizer.
Beste Parameter werden direkt in die Trainings-Konfiguration übernommen.

## Training starten & überwachen

> 💡 **Button-Reihenfolge:** ① Hyperparameter-Suche → ② Training starten → ③ Training stoppen

**Training starten**
Klicke *Training starten*. Live-Anzeige:

• **Train-Loss** und **Val-Loss** – sinken idealerweise gemeinsam
• **Train-Acc** und **Val-Acc** – steigen idealerweise gemeinsam

Das beste Modell (höchste Val-Acc) wird automatisch gespeichert.

> ⚠️ Overfitting erkennen:** Train-Loss sinkt, Val-Loss steigt → Modell lernt Trainingsdaten auswendig. Lösung: mehr Daten, Early Stopping aktivieren, größere Augmentierung.

## SSH-Ferntraining

**Einrichten****
1. *Einstellungen → SSH-Profile* – Profil anlegen (Host, User, Key-Pfad)

2. Training-Seite: SSH-Checkbox aktivieren → Profil wählen → Verbindung testen

3. Training starten → Anwendung zippt Daten, lädt hoch, streamt Logs,
   lädt das beste Modell automatisch herunter.

## Nach dem Training

Berichte erstellen**

• *HTML-Bericht* – vollständiger Report mit Kurven und Konfusionsmatrix

• *Excel-Bericht* – Metriken als Tabelle für Dokumentation
