# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Picture Studio — macOS .app bundle (BETA).

Build with:  ./build_macos.sh
or directly: pyinstaller --noconfirm Picture.spec
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# ── App metadata ─────────────────────────────────────────────────────────────
APP_NAME = "Picture Studio"
BUNDLE_ID = "de.picturestudio.app"

# ── Data files bundled into the app ──────────────────────────────────────────
# Assets are loaded at runtime via sys._MEIPASS/assets (see main.py).
# monitor_web/*.html are served by the monitor daemon (loaded via monitor.web).
datas = [
    ("assets", "assets"),
    ("monitor_web", "monitor_web"),
]
# sklearn / matplotlib ship runtime data that their hooks may miss.
datas += collect_data_files("sklearn", includes=["**/*.csv", "**/*.npz"])

# ── Hidden imports (modules pulled in dynamically by their libraries) ─────────
hiddenimports = []
hiddenimports += collect_submodules("sklearn")
hiddenimports += [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "scipy.special",
]

# ── Excludes: heavy/unused packages to keep the bundle lean ──────────────────
excludes = [
    "tkinter", "PyQt5", "PyQt6", "PySide2",
    "pytest", "pytest_qt", "_pytest",
    # Optional extras that are not installed / not needed at runtime:
    "optuna", "ultralytics", "coremltools", "imagehash",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyi_rth_cv2path.py"],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    # Collect cv2 as plain on-disk source (NOT inside the PYZ archive). The
    # OpenCV loader does sys.modules.pop("cv2") + re-import to load its native
    # .so; if the frozen importer owns the cv2 package this re-import recurses
    # ("recursion is detected during loading of cv2 binary extensions").
    module_collection_mode={"cv2": "py"},
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Picture Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Picture Studio",
)

app = BUNDLE(
    coll,
    name="Picture Studio.app",
    icon="assets/icon.icns",
    bundle_identifier=BUNDLE_ID,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": "2.5.0",
        "CFBundleVersion": "2.5.0-beta",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        # Required so macOS allows camera access instead of crashing:
        "NSCameraUsageDescription":
            "Picture Studio benötigt Kamerazugriff für die Live-Videoanalyse.",
        "NSMicrophoneUsageDescription":
            "Picture Studio benötigt ggf. Mikrofonzugriff bei Videoquellen.",
    },
)
