"""Tests für Grad-CAM mit _ConvAutoencoder."""
import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _patch_usb_scan():
    """Patch list_usb_cameras to avoid macOS subprocess crash in tests."""
    with patch("gui.camera_capture_dialog.list_usb_cameras", return_value=[]), \
         patch("gui.pages.camera_page.list_usb_cameras", return_value=[]):
        yield


@pytest.fixture
def trained_detector():
    """AnomalyDetector mit 5 Dummy-Frames trainiert."""
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector(base_ch=8)  # klein für Speed
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    for _ in range(5):
        det.collect_frame(frame)
    det.train(epochs=2, batch_size=4)
    return det


def test_compute_gradcam_anomaly_returns_bgr(trained_detector):
    """compute_gradcam_anomaly gibt ein BGR-Frame gleicher Größe zurück."""
    from core.gradcam import compute_gradcam_anomaly
    frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    result = compute_gradcam_anomaly(trained_detector, frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_compute_gradcam_anomaly_untrained_returns_original():
    """Bei untrainiertem Detektor wird der Original-Frame zurückgegeben."""
    from core.gradcam import compute_gradcam_anomaly
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    result = compute_gradcam_anomaly(det, frame)
    assert np.array_equal(result, frame)


def test_compute_gradcam_anomaly_different_base_ch(trained_detector):
    """Funktioniert auch mit base_ch=8 (nicht-Standard)."""
    from core.gradcam import compute_gradcam_anomaly
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = compute_gradcam_anomaly(trained_detector, frame)
    assert result.shape == (64, 64, 3)


def test_camera_page_has_gradcam_checkbox(qtbot):
    """CameraPage hat _gradcam_cb Checkbox."""
    from gui.pages.camera_page import CameraPage
    page = CameraPage()
    qtbot.addWidget(page)
    assert hasattr(page, "_gradcam_cb")


def test_capture_dialog_has_gradcam_checkbox(qtbot):
    """CameraCaptureDialog hat _ae_gradcam_cb Checkbox."""
    from gui.camera_capture_dialog import CameraCaptureDialog
    dlg = CameraCaptureDialog()
    qtbot.addWidget(dlg)
    assert hasattr(dlg, "_ae_gradcam_cb")
    dlg.reject()
