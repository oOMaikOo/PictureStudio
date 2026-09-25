# Settings

> **PictureStudio v2.5.2** — REST API, MQTT, SSH profiles, and all application settings

---

# Settings

All settings are saved automatically (QSettings) and restored on next launch.
Click **"Einstellungen speichern"** (Save settings) after making changes.

## Appearance

| Setting | Default | Description |
|---|---|---|
| Theme | dark | dark = dark theme \| light = light theme |
| Font size | 9 pt | 7–16 pt — takes full effect after restart |

## Project & Autosave

| Setting | Default | Description |
|---|---|---|
| Autosave enabled | Yes | Automatically save the project at the configured interval |
| Autosave interval | 300 s | 30–3600 seconds |
| Backup before saving | Yes | Creates a `.bak` backup copy on every save |

## Labeling

| Setting | Default | Description |
|---|---|---|
| Thumbnail size | 100 px | 60–240 px |
| Show ROI labels in editor | Yes | Display label text on ROI frames |

## Inference

| Setting | Default | Description |
|---|---|---|
| 'Uncertain' threshold | 0.70 | Images below this confidence value appear in the low-confidence tab |
| Default Top-K | 3 | Number of top predictions shown (1–5) |

---

## REST API Server

Integrated HTTP server for external control and monitoring.

**Start the API**
Set the port (default: `8765`) → click *Start API*.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/status | Server status (public) |
| GET | /dashboard | HTML live dashboard (public) |
| GET | /api/project | Project overview |
| GET | /api/labels | All label definitions |
| GET | /api/images | All images with labels |
| POST | /api/images/label | Assign a label |
| GET | /api/scores | Live score buffer |
| GET | /api/events | Anomaly event list |

> 💡 **Examples:**
> `curl http://localhost:8765/api/status`
> `curl http://localhost:8765/api/labels -H "X-Api-Key: your-key"`

## MQTT Alarm

**Configure MQTT**
1. MQTT publishing enabled – check the checkbox
2. Broker host – hostname or IP (e.g. `localhost`)
3. Port – default `1883`
4. Topic – default: `picture_studio/anomaly`
5. Username / Password – optional
6. Save settings

> ⚠️ Prerequisite: `pip install paho-mqtt`

## Alerting (Email & Webhook)

**Configure email**
Enter SMTP host, port (587 for TLS), username, and password.

**Configure webhook**
Enter URL (e.g. Teams, Slack). Payload: event, timestamp, score, threshold, frame_file.

## Industrial Integration (OPC-UA & Modbus TCP)

| Protocol | Typical use | Default port |
|---|---|---|
| OPC-UA | Siemens S7, Beckhoff, FANUC CNC | 4840 |
| Modbus TCP | Beckhoff, Wago, older PLCs | 502 |

## SSH Profiles

**Add a profile**
- Profile name, host, username, SSH key path (e.g. `~/.ssh/id_rsa`)

> 💡 **Create an SSH key:**
> `ssh-keygen -t ed25519 -f ~/.ssh/gpu_key`
> `ssh-copy-id -i ~/.ssh/gpu_key.pub user@server`
