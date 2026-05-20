"""Tests für AnomalyHPTWorker und AnomalyHPTThread."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _patch_usb_scan():
    """Patch list_usb_cameras for all tests to avoid macOS subprocess crash."""
    with patch("gui.camera_capture_dialog.list_usb_cameras", return_value=[]):
        yield


# ── AnomalyDetector base_ch ──────────────────────────────────────────────────

def test_anomaly_detector_accepts_base_ch():
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector(base_ch=8)
    assert det._base_ch == 8


def test_anomaly_detector_default_base_ch():
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    assert det._base_ch == 16


def test_conv_autoencoder_base_ch_8():
    from core.anomaly_detector import _ConvAutoencoder
    import torch
    model = _ConvAutoencoder(base_ch=8)
    x = torch.zeros(1, 3, 128, 128)
    out = model(x)
    assert out.shape == x.shape


def test_conv_autoencoder_base_ch_32():
    from core.anomaly_detector import _ConvAutoencoder
    import torch
    model = _ConvAutoencoder(base_ch=32)
    x = torch.zeros(1, 3, 128, 128)
    out = model(x)
    assert out.shape == x.shape


def test_conv_autoencoder_base_ch_16():
    """Default base_ch=16 must produce correct output shape."""
    from core.anomaly_detector import _ConvAutoencoder
    import torch
    model = _ConvAutoencoder()
    x = torch.zeros(1, 3, 128, 128)
    out = model(x)
    assert out.shape == x.shape


def test_anomaly_detector_train_accepts_lr():
    """train() must forward the lr argument to the Adam optimizer (smoke test)."""
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector(base_ch=8)
    # collect a few synthetic frames
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(5):
        det.collect_frame(frame)
    thr = det.train(epochs=1, batch_size=4, lr=5e-4)
    assert isinstance(thr, float)
    assert thr >= 0


def test_anomaly_detector_save_contains_base_ch(tmp_path):
    """Saved checkpoint must include base_ch."""
    import torch
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector(base_ch=8)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(5):
        det.collect_frame(frame)
    det.train(epochs=1, batch_size=4)
    path = str(tmp_path / "model.pth")
    det.save(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["base_ch"] == 8


def test_anomaly_detector_load_restores_base_ch(tmp_path):
    """load() must rebuild the model with the saved base_ch."""
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector(base_ch=8)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(5):
        det.collect_frame(frame)
    det.train(epochs=1, batch_size=4)
    path = str(tmp_path / "model.pth")
    det.save(path)

    det2 = AnomalyDetector()  # default base_ch=16
    det2.load(path)
    assert det2._base_ch == 8


# ── AnomalyHPTWorker ─────────────────────────────────────────────────────────

def test_anomaly_hpt_worker_raises_without_frames():
    pytest.importorskip("optuna")
    from core.hyperparameter_tuning import AnomalyHPTWorker
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    worker = AnomalyHPTWorker(det, n_trials=1, epochs_per_trial=1)
    with pytest.raises(ValueError, match="Keine Frames"):
        worker.run()


def test_anomaly_hpt_worker_runs_minimal():
    """End-to-end smoke test with 1 trial and 1 epoch."""
    pytest.importorskip("optuna")
    from core.hyperparameter_tuning import AnomalyHPTWorker
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(10):
        det.collect_frame(frame)
    worker = AnomalyHPTWorker(det, n_trials=1, epochs_per_trial=1)
    result = worker.run()
    assert "base_ch" in result
    assert "lr" in result
    assert "batch_size" in result
    assert "best_value" in result
    assert result["base_ch"] in [8, 16, 32]


def test_anomaly_hpt_worker_progress_callback():
    """progress_callback must be called once per trial."""
    pytest.importorskip("optuna")
    from core.hyperparameter_tuning import AnomalyHPTWorker
    from core.anomaly_detector import AnomalyDetector
    calls = []
    det = AnomalyDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(10):
        det.collect_frame(frame)
    worker = AnomalyHPTWorker(
        det, n_trials=2, epochs_per_trial=1,
        progress_callback=lambda t, tot, v: calls.append((t, tot, v)),
    )
    worker.run()
    assert len(calls) == 2
    assert calls[-1][0] == 2
    assert calls[-1][1] == 2


def test_anomaly_hpt_thread_instantiates():
    """AnomalyHPTThread can be instantiated without a QApplication."""
    from core.hyperparameter_tuning import AnomalyHPTThread
    from core.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    t = AnomalyHPTThread(det, n_trials=1)
    assert t is not None


# ── CameraCaptureDialog HPT button ───────────────────────────────────────────

def test_dialog_has_hpt_button(qtbot):
    from gui.camera_capture_dialog import CameraCaptureDialog
    dlg = CameraCaptureDialog()
    qtbot.addWidget(dlg)
    assert hasattr(dlg, "_ae_hpt_btn"), "_ae_hpt_btn fehlt"
    dlg.reject()


def test_hpt_button_disabled_initially(qtbot):
    from gui.camera_capture_dialog import CameraCaptureDialog
    dlg = CameraCaptureDialog()
    qtbot.addWidget(dlg)
    assert not dlg._ae_hpt_btn.isEnabled()
    dlg.reject()


def test_dialog_has_lr_override(qtbot):
    """Dialog must expose _lr_override and _batch_override initialized to None."""
    from gui.camera_capture_dialog import CameraCaptureDialog
    dlg = CameraCaptureDialog()
    qtbot.addWidget(dlg)
    assert dlg._lr_override is None
    assert dlg._batch_override is None
    dlg.reject()
