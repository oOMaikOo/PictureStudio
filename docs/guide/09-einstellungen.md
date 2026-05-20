# ⚙ Einstellungen

> **PictureStudio v2.3.0** — REST-API, MQTT, SSH-Profile und alle App-Einstellungen

---

# ⚙ Einstellungen

Alle Einstellungen werden automatisch gespeichert (QSettings) und beim nächsten Start wiederhergestellt.
Nach Änderungen **„Einstellungen speichern"** klicken.

## Erscheinungsbild

| Einstellung | Standard | Beschreibung |
|---|---|---|
| Design | dark | dark = dunkles Theme | light = helles Theme |
| Schriftgröße | 9 pt | 7–16 pt – wirkt nach Neustart vollständig |

## Projekt & Autosave

| Einstellung | Standard | Beschreibung |
|---|---|---|
| Autosave aktiviert | Ja | Projekt automatisch im eingestellten Intervall speichern |
| Autosave-Intervall | 300 s | 30–3600 Sekunden |
| Backup vor Speichern | Ja | Erstellt bei jedem Speichern eine .bak-Sicherungskopie |

## Labeling

| Einstellung | Standard | Beschreibung |
|---|---|---|
| Thumbnail-Größe | 100 px | 60–240 px |
| ROI-Labels im Editor anzeigen | Ja | Label-Texte auf ROI-Rahmen einblenden |

## Inferenz

| Einstellung | Standard | Beschreibung |
|---|---|---|
| Schwelle 'unsicher' | 0.70 | Bilder unter diesem Konfidenzwert erscheinen im Niedrig-Konfidenz-Tab |
| Standard Top-K | 3 | Anzahl angezeigte Top-Vorhersagen (1–5) |

---

## REST-API Server

Integrierter HTTP-Server für externe Steuerung und Monitoring.

**API starten****
Port einstellen (Standard: `8765`) → *API starten* klicken.

### API-Endpunkte

| Methode | Endpunkt | Beschreibung |
|---|---|---|
| GET | /api/status | Server-Status (öffentlich) |
| GET | /dashboard | HTML Live-Dashboard (öffentlich) |
| GET | /api/project | Projektübersicht |
| GET | /api/labels | Alle Label-Definitionen |
| GET | /api/images | Alle Bilder mit Labels |
| POST | /api/images/label | Label zuweisen |
| GET | /api/scores | Live-Score-Puffer |
| GET | /api/events | Anomalie-Event-Liste |

> 💡 Beispiel:**** `curl http://localhost:8765/api/status`
 `curl http://localhost:8765/api/labels -H "X-Api-Key: dein-schluessel"`

## MQTT-Alarm

MQTT einrichten****
1. MQTT-Publishing aktiviert – Checkbox aktivieren

2. Broker-Host – Hostname oder IP (z. B. `localhost`)

3. Port – Standard `1883`

4. Topic – Standard: `picture_studio/anomaly`

5. Benutzername / Passwort – optional

6. Einstellungen speichern

> ⚠️ Voraussetzung:** `pip install paho-mqtt`

## Alarmierung (E-Mail & Webhook)

**E-Mail konfigurieren****
SMTP-Host, Port (587 für TLS), Benutzername und Passwort eintragen.

Webhook konfigurieren****
URL eintragen (z.B. Teams, Slack). Payload: event, timestamp, score, threshold, frame_file.

## Industrieanbindung (OPC-UA & Modbus TCP)

| Protokoll | Typische Anwendung | Standard-Port |
|---|---|---|
| OPC-UA | Siemens S7, Beckhoff, FANUC CNC | 4840 |
| Modbus TCP | Beckhoff, Wago, ältere SPS | 502 |

## SSH-Profile

Profil hinzufügen****
• Profilname, Host, Benutzername, SSH-Key-Pfad (z. B. `~/.ssh/id_rsa`)

> 💡 SSH-Key erstellen:**
 `ssh-keygen -t ed25519 -f ~/.ssh/gpu_key`
 `ssh-copy-id -i ~/.ssh/gpu_key.pub user@server`
