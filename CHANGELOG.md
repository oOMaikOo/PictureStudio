# Changelog

All notable changes to PictureStudio are documented here.

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
