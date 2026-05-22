"""Deutsche UI-Strings — Standardsprache."""

STRINGS: dict = {
    # ------------------------------------------------------------------ Sidebar navigation
    "nav.dashboard":        "Dashboard",
    "nav.data":             "Daten",
    "nav.labeling":         "Labeling",
    "nav.training":         "Training",
    "nav.models":           "Modelle",
    "nav.inference":        "Klassifikation",
    "nav.batch":            "Batch",
    "nav.export":           "Export",
    "nav.settings":         "Einstellungen",
    "nav.clustering":       "Clustering",
    "nav.dataset":          "Datensatz",
    "nav.objectdetection":  "Objekterkennung",
    "nav.datadrift":        "Data Drift",
    "nav.camera":           "Live & Anomalie",
    "nav.multicamera":      "Multi-Kamera",
    "nav.videoannotation":  "Video-Annotation",
    "nav.fleet":            "Fleet",
    "sidebar.tour_btn":     "▶  Tour starten",
    "sidebar.help_btn":     "?  Hilfe (F1)",
    "sidebar.badge.image":  "📸 Bildprojekt",
    "sidebar.badge.video":  "🎬 Videoprojekt",

    # ------------------------------------------------------------------ Menüs
    "menu.file":                "Datei",
    "menu.file.new":            "Neues Projekt",
    "menu.file.open":           "Projekt öffnen…",
    "menu.file.save":           "Projekt speichern",
    "menu.file.saveas":         "Projekt speichern unter…",
    "menu.file.backup":         "Backup erstellen",
    "menu.file.camera":         "Kamera aufnehmen…",
    "menu.file.recent":         "Zuletzt geöffnet",
    "menu.file.quit":           "Beenden",
    "menu.project":             "Projekt",
    "menu.project.labels":      "Labels verwalten…",
    "menu.project.validate":    "Bilddateien prüfen",
    "menu.project.info":        "Projektinfo…",
    "menu.project.report":      "Bericht erstellen…",
    "menu.view":                "Ansicht",
    "menu.view.batchinference": "Batch-Inferenz",
    "menu.audit":               "Audit",
    "menu.audit.changelog":     "Änderungsprotokoll anzeigen…",
    "menu.help":                "Hilfe",
    "menu.help.manual":         "Handbuch öffnen…  (F1)",
    "menu.help.tour":           "Geführte Tour starten",
    "menu.help.log":            "Fehlerlog anzeigen…",
    "menu.help.about":          "Über…",
    "menu.help.dashboard":      "Dashboard – Hilfe",
    "menu.help.data":           "Daten – Hilfe",
    "menu.help.labeling":       "Labeling – Hilfe",
    "menu.help.training":       "Training – Hilfe",
    "menu.help.models":         "Modelle – Hilfe",
    "menu.help.inference":      "Klassifikation – Hilfe",
    "menu.help.export":         "Export – Hilfe",
    "menu.help.settings":       "Einstellungen – Hilfe",
    "menu.help.camera":         "Kamera – Hilfe",
    "menu.help.shortcuts":      "Tastenkürzel",
    "menu.help.troubleshoot":   "Fehlerbehebung",
    "menu.help.monitor":        "Monitor-Client – Hilfe",
    "menu.help.multicamera":    "Multi-Kamera – Hilfe",
    "menu.help.clustering":     "Anomalie-Clustering – Hilfe",
    "menu.help.datasetstats":   "Datensatz-Statistiken – Hilfe",
    "menu.help.videoannotation":"Video-Annotation – Hilfe",
    "menu.help.fleet":          "Fleet-Management – Hilfe",
    "menu.help.modeladvanced":  "Modell-Erweitert – Hilfe",

    # ------------------------------------------------------------------ Statusleiste
    "statusbar.ready":          "Bereit – kein Projekt geladen",

    # ------------------------------------------------------------------ Über-Dialog
    "about.beta_warning":       "⚠ Beta-Version — nicht für den Produktiveinsatz",

    # ------------------------------------------------------------------ Absturz-Dialog
    "crash.title":              "Unerwarteter Fehler",
    "crash.text":               (
        "Ein unerwarteter Fehler ist aufgetreten.\n"
        "Details wurden in das Fehlerlog geschrieben:\n{log_dir}"
    ),

    # ------------------------------------------------------------------ Einstellungen
    "settings.title":           "Einstellungen",
    "settings.lang.group":      "Sprache / Language",
    "settings.lang.label":      "Sprache:",
    "settings.lang.hint":       "Änderung wirkt nach Neustart der App",
    "settings.save_btn":        "Einstellungen speichern",
    "settings.saved.title":     "Gespeichert",
    "settings.saved.msg":       "Einstellungen wurden gespeichert.",
}
