# Changelog

All notable changes to PictureStudio are documented here.

---

## [1.2.0] – 2026-05-19

### Added
- **Multi-camera: channel count selector** — new `QSpinBox` in the toolbar lets you
  choose 1–9 simultaneous monitoring channels (default: 2). The grid rebuilds
  dynamically; existing channel configs (model, camera index) are preserved when
  the count grows.
- **Multi-camera: pagination** — when more than 4 channels are active a navigation
  row (◀ Vorherige / Seite N / Gesamt / Nächste ▶) appears automatically. Each
  page shows up to 4 channels in a 2 × 2 grid; channels on hidden pages keep
  running.
- **Multi-camera: alarm JPEG saving** — every alarm frame is automatically saved as
  `monitor_logs/multi_cam/mc_ch<N>_<YYYYMMDDTHHMMSSZ>.jpg`.
- **REST API: per-channel multi-camera endpoints**
  - `GET /api/mc/channels` — summary of all channels (score, threshold, is_alarm,
    event_count, cam_status).
  - `GET /api/mc/scores?channel=N` — rolling score buffer (up to 500 entries) for
    channel N.
  - `GET /api/mc/latest_alarm?channel=N` — most recent alarm event for channel N.
  - `RestApiServer` gains `set_mc_channel_count()`, `push_mc_score()`,
    `push_mc_alarm()`, `set_mc_cam_status()` for thread-safe state updates.
- **Dashboard: multi-camera section** — the web dashboard (`/dashboard`) shows a live
  per-channel grid (score, alarm state, cam status) that appears automatically when
  channels are registered and hides when none are active.
- **monitor.py: IP/RTSP-URL + video file support** — `--url URL` accepts RTSP/HTTP
  streams and local video files (mp4, avi, mov, mkv, …). `--camera INDEX` and
  `--url` are mutually exclusive.
- **monitor.py: auto-reconnect** — live streams (USB + RTSP/HTTP) reconnect
  automatically after `--reconnect-delay` seconds (default: 5). Video files play
  once without reconnect.
- **monitor.py: MQTT publishing** — `--mqtt-host HOST` publishes alarm events as JSON
  to a configurable broker and topic (`--mqtt-topic`, default:
  `picture_studio/monitor`). Optional auth via `--mqtt-user` / `--mqtt-pass`.
  Graceful no-op when paho-mqtt is not installed.
- **monitor.py: embedded REST API + dashboard** — `--api-port PORT` starts a
  lightweight HTTP server with `/api/status`, `/api/scores`, `/api/latest_alarm`,
  `/api/frame/<file>`, and `/dashboard`. Auth via `--api-key KEY`; status and
  dashboard are always public.

### Changed
- Help dialog section 14 (Multi-Kamera) fully rewritten: documents channel selector,
  pagination, alarm JPEG path, per-channel REST endpoints with curl examples.
- Guided tour step index 10 expanded from 4 → 7 steps covering all new features.
- Feature-overview table in help updated: Multi-Kamera row added, REST-API and
  Kamera/Video rows updated.

---

## [1.1.0] – 2026-05-19

### Added
- **REST API authentication** — optional shared-secret API key (`X-Api-Key` header).
  Generate / show / clear key in Settings → REST-API. Public endpoints (`/api/status`,
  `/dashboard`) never require a key. All other endpoints return HTTP 401 without it.
  The dashboard JS injects the key automatically so it keeps working.
- **Camera auto-reconnect** — when a live camera stream drops, the Live-Monitoring page
  waits 5 seconds and reconnects automatically. Status label turns yellow during reconnect
  and green again on the first successful frame. Manual disconnect stops the cycle.
- **Video file inference in Live-Monitor** — the camera dropdown now includes
  "Videodatei (MP4, AVI, …)". A file dialog opens on selection; native FPS is read from
  the file automatically (fallback: 25 fps). Video playback does not trigger auto-reconnect.

### Fixed
- `show_roi_labels` setting was rendered in the UI but never loaded or saved via QSettings.
  The value is now correctly persisted on save and restored on next launch.

### Changed
- Help dialog and Tour updated for all new features: API key auth workflow,
  auto-reconnect behaviour, video-from-combo usage.
- REST API endpoint table in help now marks public vs. protected endpoints.
- Feature overview table updated.

---

## [1.0.0] – 2026-05-15

### Added
- Initial stable release.
- Image labeling with single- and multi-label mode, ROI editor (rectangle, ellipse, polygon).
- Training pipeline: ResNet18/50, MobileNetV2, EfficientNet-B0; GPU/CPU/MPS; mixed precision.
- SSH remote training with live log streaming.
- Batch inference with auto-labeling and confidence colour coding.
- Anomaly detection: Conv-Autoencoder, heatmap overlay, ROI, alarm deduplication, CSV log.
- Live-Monitoring page with scoring, score chart, alarm banner, and JPEG snapshots.
- Multi-camera monitoring (up to 4 simultaneous feeds).
- Anomaly clustering (DBSCAN/K-Means) for grouping alarm frames.
- ONNX and TorchScript export for edge deployment.
- E-Mail and webhook alarm notifications.
- OPC-UA and Modbus TCP integration for industrial SPS connectivity.
- MQTT alarm publishing (paho-mqtt).
- Standalone monitor client (`monitor.py`) — runs without the GUI.
- REST API with live dashboard (HTML, auto-refresh every 3 s).
- LRU thumbnail cache (max 500 entries) to prevent memory growth.
- CSV and JSON export for inference results.
- Camera stream retry logic: 5 consecutive failures required before error signal.
- Project load error handling with user-friendly messages.
- Thread cleanup on application close (camera, training, clustering, industrial notifier).
