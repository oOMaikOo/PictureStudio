# Camera & Video Analysis

> **PictureStudio v2.3.0** — Live capture, anomaly detection, and batch analysis with a camera or video file

---

# Camera & Video Analysis

Live capture from USB/IP cameras, video file analysis, and automatic anomaly detection.

Open via: *File → Capture camera…* (`Ctrl+K`)

## Camera Sources

### USB Camera

Select the camera from the dropdown → click *Connect*.
Camera not visible? → Click *Rescan cameras*.

### IP Camera / Network Camera

Enter the URL and click *Connect*.

`rtsp://user:pass@192.168.1.100:554/stream` – RTSP

`http://192.168.1.100:8080/video` – HTTP-MJPEG

### Video File

*Choose file…* → MP4, AVI, MOV, MKV, WebM.

Set playback fps to 0 for original speed.

## Capture Functions

**Single frame** – `Spacebar`

**Burst capture** – set count + interval → click *Start burst*

**Live recording (MP4)** – *Start recording* → saves a continuous MP4 file

## Anomaly Detection (Autoencoder)

> 💡 **How it works:** A convolutional autoencoder learns from normal frames. When an anomaly occurs, the reconstruction error (MSE) exceeds the threshold → alarm.

### Step 1 – Capture Normal Frames

150–500 frames recommended. Click *Start capturing* → frames are collected.

### Step 2 – Train the Autoencoder

Set epochs (default: 40) → click *Training starten*.
Threshold = mean + 2.5 × standard deviation of training reconstruction errors.

### Step 3 – Live Detection

Activate the **Scoring aktiv** button.
Green = Normal | Red = Anomaly. The heatmap highlights deviating regions.

## Batch Analysis

Tab "📁 Batch" → select folder or files → click *Start batch* → export as CSV.

## Camera Settings

Sliders for **Brightness, Contrast, Saturation, Sharpness, and Exposure** are available in two places:

- **CameraPage (Anomaly Detection):** In the left panel under "Kamera-Einstellungen" (collapsed by default). Changes apply live to the running stream.
- **Capture Dialog (Image Classification):** Directly in the *CameraCaptureDialog* — also available via the camera button on the Data page. Values passed from CameraPage are used as initial values.

**Reset** sets all sliders back to neutral values.

## Preprocessing Filters

The filter dropdown is available in both dialogs (anomaly detection *and* image classification capture):

- **No filter** — original frame
- **Grayscale** — returned as BGR
- **Canny edges** — edge lines (thresholds 50/150)
- **Sobel gradient** — gradient magnitude in X and Y
- **Laplacian** — second derivative (fine detail)

## Hyperparameter Search (Anomaly Autoencoder)

Click **⚙ Hyperparameter-Suche…** to start an Optuna study.

Search space: base_ch (8/16/32), lr (1e-4 to 1e-2), batch_size (8/16/32).

**Button order in the capture dialog:** ① Hyperparameter-Suche → ② Training starten → ③ Training stoppen

### Live Progress Display

A dialog shows the search progress:

- **Progress bar** — Trial X / Total
- **Best threshold** — currently lowest reconstruction error
- **Scrolling log** — one line per trial with parameters and result

```
[Trial  1/10]  base_ch=16  lr=1.2341e-03  batch=16  →  Threshold: 0.01234  ★ New best!
[Trial  2/10]  base_ch=32  lr=5.6780e-03  batch= 8  →  Threshold: 0.01891
```

A **★** marks each new best result. When done: close the dialog or apply the parameters directly.

> ⚠️ `pip install optuna`
