# Troubleshooting

> **PictureStudio v2.5.2** — Common problems and their solutions

---

# Troubleshooting

## Application Won't Start

Install `pip install PySide6`.

Linux: `apt install libxcb-cursor0`

## Training is Very Slow

Set the device to `cuda` or `mps`.

For CPU testing: set image size to 128 px, batch size to 8, and use SimpleCNN.

## ImportError: openpyxl

`pip install openpyxl`

## ImportError: paramiko

`pip install paramiko` – required only for SSH remote training

## MQTT Not Working

`pip install paho-mqtt`

Test the broker connection: `mosquitto_pub -h localhost -t test -m hello`

## Camera Not Found

`pip install opencv-python`

macOS: Check camera access in System Settings → Privacy & Security → Camera.

## SSH Connection Fails

- Check host, username, and key path
- `ssh-add <key_path>`
- `chmod 600 ~/.ssh/id_rsa`

## Anomaly Score Always 0

The autoencoder must be trained first. **Scoring aktiv** must be enabled (green).

## Too Many False Alarms

- Increase or calibrate the threshold
- Increase smoothing to 5–10 frames
- Collect more normal frames and retrain
- Set an ROI and enable the motion filter

## Project File Corrupted

Rename `projekt.bak` → `projekt.json`

## Video File Won't Open

Convert to H.264 using ffmpeg:

`ffmpeg -i input.avi -c:v libx264 output.mp4`
