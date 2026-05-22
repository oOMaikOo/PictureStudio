"""
Minimal i18n engine for Picture Studio.

Usage:
    # In main.py, before creating MainWindow:
    from utils.i18n import init_i18n
    init_i18n("de")   # or "en"

    # In any UI file:
    from utils.i18n import tr
    btn.setText(tr("menu.file.new"))

Strings not found in the current language dict fall back to the key itself
so untranslated strings are visible rather than crashing.
"""

_strings: dict = {}
_lang: str = "de"

SUPPORTED = ("de", "en")


def init_i18n(lang: str) -> None:
    """Load the string dict for *lang*. Must be called before any UI is created."""
    global _strings, _lang
    lang = lang if lang in SUPPORTED else "de"
    _lang = lang
    if lang == "en":
        from locales.en import STRINGS
    else:
        from locales.de import STRINGS
    _strings = STRINGS


def tr(key: str, **kwargs) -> str:
    """Return the translated string for *key*, formatting with *kwargs* if given."""
    text = _strings.get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def current_lang() -> str:
    """Return the currently active language code."""
    return _lang
