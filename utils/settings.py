"""
Persistent user settings via QSettings (INI file).
"""
from PySide6.QtCore import QSettings


class AppSettings:
    """
    Wrapper around QSettings providing typed getter/setter pairs.

    Settings are stored in the OS-specific INI / registry location under
    organisation "ImageLabelingStudio" / application "ILS".
    Used by MainWindow, TrainingPage, and SettingsPage.
    """

    _ORG = "ImageLabelingStudio"
    _APP = "ILS"

    def __init__(self):
        self._s = QSettings(self._ORG, self._APP)

    # ---- appearance ----
    def get_theme(self) -> str:
        return self._s.value("appearance/theme", "dark")

    def set_theme(self, theme: str) -> None:
        self._s.setValue("appearance/theme", theme)

    def get_font_size(self) -> int:
        return int(self._s.value("appearance/font_size", 9))

    def set_font_size(self, size: int) -> None:
        self._s.setValue("appearance/font_size", size)

    # ---- project ----
    def get_autosave_enabled(self) -> bool:
        return self._s.value("project/autosave_enabled", True, type=bool)

    def set_autosave_enabled(self, val: bool) -> None:
        self._s.setValue("project/autosave_enabled", val)

    def get_autosave_interval(self) -> int:
        return int(self._s.value("project/autosave_interval", 300))

    def set_autosave_interval(self, seconds: int) -> None:
        self._s.setValue("project/autosave_interval", seconds)

    def get_backup_enabled(self) -> bool:
        return self._s.value("project/backup_enabled", True, type=bool)

    def set_backup_enabled(self, val: bool) -> None:
        self._s.setValue("project/backup_enabled", val)

    def get_recent_projects(self) -> list:
        return self._s.value("project/recent", []) or []

    def add_recent_project(self, path: str) -> None:
        """Prepend *path* to the MRU list, keeping at most 10 entries."""
        recents = self.get_recent_projects()
        if path in recents:
            recents.remove(path)
        recents.insert(0, path)
        self._s.setValue("project/recent", recents[:10])

    # ---- labeling ----
    def get_thumbnail_size(self) -> int:
        return int(self._s.value("labeling/thumbnail_size", 120))

    def set_thumbnail_size(self, size: int) -> None:
        self._s.setValue("labeling/thumbnail_size", size)

    def get_show_roi_labels(self) -> bool:
        return self._s.value("labeling/show_roi_labels", True, type=bool)

    def set_show_roi_labels(self, val: bool) -> None:
        self._s.setValue("labeling/show_roi_labels", val)

    # ---- training ----
    def get_default_device(self) -> str:
        return self._s.value("training/device", "auto")

    def set_default_device(self, device: str) -> None:
        self._s.setValue("training/device", device)

    # ---- inference ----
    def get_low_confidence_threshold(self) -> float:
        return float(self._s.value("inference/low_confidence_threshold", 0.70))

    def set_low_confidence_threshold(self, val: float) -> None:
        self._s.setValue("inference/low_confidence_threshold", val)

    def get_show_top_k(self) -> int:
        return int(self._s.value("inference/show_top_k", 3))

    def set_show_top_k(self, val: int) -> None:
        self._s.setValue("inference/show_top_k", val)

    # ---- alarm notifier ----
    def get_alarm_notifier_config(self) -> dict:
        return self._s.value("alarm/notifier_config", {}) or {}

    def save_alarm_notifier_config(self, cfg: dict) -> None:
        self._s.setValue("alarm/notifier_config", cfg)

    # ---- industrial protocols ----
    def get_industrial_config(self) -> dict:
        return self._s.value("industrial/config", {}) or {}

    def save_industrial_config(self, cfg: dict) -> None:
        self._s.setValue("industrial/config", cfg)

    # ---- mqtt ----
    def get_mqtt_config(self) -> dict:
        return self._s.value("mqtt/config", {}) or {}

    def save_mqtt_config(self, cfg: dict) -> None:
        self._s.setValue("mqtt/config", cfg)

    # ---- ssh profiles ----
    def get_ssh_profiles(self) -> list:
        return self._s.value("ssh/profiles", []) or []

    def save_ssh_profiles(self, profiles: list) -> None:
        self._s.setValue("ssh/profiles", profiles)

    # ---- REST API key ----
    def get_api_key(self) -> str:
        return self._s.value("api/key", "") or ""

    def save_api_key(self, key: str) -> None:
        self._s.setValue("api/key", key)

    # ---- window state ----
    def get_window_geometry(self):
        return self._s.value("window/geometry")

    def save_window_geometry(self, geometry) -> None:
        self._s.setValue("window/geometry", geometry)

    def sync(self) -> None:
        """Flush pending changes to disk immediately."""
        self._s.sync()
