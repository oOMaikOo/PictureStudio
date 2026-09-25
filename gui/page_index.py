"""
Stack indices of the pages in ``MainWindow``'s ``QStackedWidget``.

Single source of truth for the page numbering. Four places used to hard-code
the same integers — the page list in ``MainWindow._build_ui()``, the nav tuples
in ``gui/sidebar.py``, the ``TOUR_STEPS`` keys in ``gui/guide_tour.py`` and
``PAGE_TO_SECTION`` in ``gui/help_dialog.py`` — so renumbering meant editing all
four in lockstep and silently broke navigation when one was missed.

``MainWindow`` adds its pages in ``sorted(Page)`` order, which makes the value
below *the* index rather than a description of it. To add a page: append a
member here, map it to its widget in ``MainWindow._build_ui()`` (a missing entry
raises ``KeyError`` at startup), then add a sidebar nav tuple.

``IntEnum`` members compare and hash like plain ints, so they can be passed to
``setCurrentIndex()`` and mixed with ints in sets and dict keys.
"""
from enum import IntEnum


class Page(IntEnum):
    """Index of each page in ``MainWindow.stack``."""

    DASHBOARD        = 0
    DATA             = 1
    LABELING         = 2
    TRAINING         = 3
    MODELS           = 4
    INFERENCE        = 5
    EXPORT           = 6
    SETTINGS         = 7
    CAMERA           = 8
    BATCH            = 9
    MULTI_CAMERA     = 10
    DATASET_STATS    = 11
    VIDEO_ANNOTATION = 12
    FLEET            = 13
    DATA_DRIFT       = 14
    ANOMALY_TRAINING = 15
    LIVE_CLASSIFY    = 16
