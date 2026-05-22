"""English UI strings."""

STRINGS: dict = {
    # ------------------------------------------------------------------ Sidebar navigation
    "nav.dashboard":        "Dashboard",
    "nav.data":             "Data",
    "nav.labeling":         "Labeling",
    "nav.training":         "Training",
    "nav.models":           "Models",
    "nav.inference":        "Classification",
    "nav.batch":            "Batch",
    "nav.export":           "Export",
    "nav.settings":         "Settings",
    "nav.clustering":       "Clustering",
    "nav.dataset":          "Dataset",
    "nav.objectdetection":  "Object Detection",
    "nav.datadrift":        "Data Drift",
    "nav.camera":           "Live & Anomaly",
    "nav.multicamera":      "Multi-Camera",
    "nav.videoannotation":  "Video Annotation",
    "nav.fleet":            "Fleet",
    "sidebar.tour_btn":     "▶  Start Tour",
    "sidebar.help_btn":     "?  Help (F1)",
    "sidebar.badge.image":  "📸 Image Project",
    "sidebar.badge.video":  "🎬 Video Project",

    # ------------------------------------------------------------------ Menus
    "menu.file":                "File",
    "menu.file.new":            "New Project",
    "menu.file.open":           "Open Project…",
    "menu.file.save":           "Save Project",
    "menu.file.saveas":         "Save Project As…",
    "menu.file.backup":         "Create Backup",
    "menu.file.camera":         "Camera Capture…",
    "menu.file.recent":         "Recent Projects",
    "menu.file.quit":           "Quit",
    "menu.project":             "Project",
    "menu.project.labels":      "Manage Labels…",
    "menu.project.validate":    "Validate Image Files",
    "menu.project.info":        "Project Info…",
    "menu.project.report":      "Create Report…",
    "menu.view":                "View",
    "menu.view.batchinference": "Batch Inference",
    "menu.audit":               "Audit",
    "menu.audit.changelog":     "Show Changelog…",
    "menu.help":                "Help",
    "menu.help.manual":         "Open Manual…  (F1)",
    "menu.help.tour":           "Start Guided Tour",
    "menu.help.log":            "Show Error Log…",
    "menu.help.about":          "About…",
    "menu.help.dashboard":      "Dashboard – Help",
    "menu.help.data":           "Data – Help",
    "menu.help.labeling":       "Labeling – Help",
    "menu.help.training":       "Training – Help",
    "menu.help.models":         "Models – Help",
    "menu.help.inference":      "Classification – Help",
    "menu.help.export":         "Export – Help",
    "menu.help.settings":       "Settings – Help",
    "menu.help.camera":         "Camera – Help",
    "menu.help.shortcuts":      "Keyboard Shortcuts",
    "menu.help.troubleshoot":   "Troubleshooting",
    "menu.help.monitor":        "Monitor Client – Help",
    "menu.help.multicamera":    "Multi-Camera – Help",
    "menu.help.clustering":     "Anomaly Clustering – Help",
    "menu.help.datasetstats":   "Dataset Statistics – Help",
    "menu.help.videoannotation":"Video Annotation – Help",
    "menu.help.fleet":          "Fleet Management – Help",
    "menu.help.modeladvanced":  "Advanced Model – Help",

    # ------------------------------------------------------------------ Status bar
    "statusbar.ready":          "Ready – no project loaded",

    # ------------------------------------------------------------------ About dialog
    "about.beta_warning":       "⚠ Beta version — not for production use",

    # ------------------------------------------------------------------ Crash dialog
    "crash.title":              "Unexpected Error",
    "crash.text":               (
        "An unexpected error occurred.\n"
        "Details have been written to the error log:\n{log_dir}"
    ),

    # ------------------------------------------------------------------ Settings
    "settings.title":           "Settings",
    "settings.lang.group":      "Language / Sprache",
    "settings.lang.label":      "Language:",
    "settings.lang.hint":       "Change takes effect after restarting the app",
    "settings.save_btn":        "Save Settings",
    "settings.saved.title":     "Saved",
    "settings.saved.msg":       "Settings have been saved.",
}
