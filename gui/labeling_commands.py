"""
QUndoCommand subclasses for all undoable labeling operations.
Each command holds enough state to redo and undo the action,
and calls back into LabelingPage for UI refresh.
"""
import copy
from PySide6.QtGui import QUndoCommand


# ---------------------------------------------------------------------------
# Image-level label
# ---------------------------------------------------------------------------

class BulkSetImageLabelCommand(QUndoCommand):
    """
    Assign one label to N images in a single undoable step.

    Used by the bulk-labeling panel and the right-click context menu
    when multiple thumbnails are selected.
    """

    def __init__(self, page, image_paths: list, new_label: str, old_labels: dict,
                 label_map: dict = None):
        """
        Parameters
        ----------
        page        : LabelingPage instance (provides _do_set_image_label).
        image_paths : Paths to be labelled.
        new_label   : Label to assign (empty string = remove label).
                      Ignored when label_map is provided.
        old_labels  : {path: previous_label} snapshot for undo.
        label_map   : Optional {path: label} for per-image labels (pre-labeling).
        """
        n = len(image_paths)
        if label_map is not None:
            super().__init__(f"Pre-Labeling ({n} Bilder)")
        else:
            super().__init__(f"Massen-Label ({n} Bilder → {new_label or '(kein)'})")
        self._page       = page
        self._paths      = list(image_paths)
        self._new        = new_label
        self._old_labels = dict(old_labels)   # path -> old label
        self._label_map  = dict(label_map) if label_map else None

    def redo(self):
        for path in self._paths:
            lbl = self._label_map[path] if self._label_map else self._new
            self._page._do_set_image_label(path, lbl)

    def undo(self):
        for path in self._paths:
            self._page._do_set_image_label(path, self._old_labels.get(path, ""))


class SetLabelFlagCommand(QUndoCommand):
    """Toggle the QA 'uncertain' flag and optional comment on a single image."""

    def __init__(self, page, image_path: str,
                 new_uncertain: bool, new_comment: str,
                 old_uncertain: bool, old_comment: str):
        state = "unsicher" if new_uncertain else "sicher"
        super().__init__(f"Label-Flag: → {state}")
        self._page        = page
        self._image_path  = image_path
        self._new_u       = new_uncertain
        self._new_c       = new_comment
        self._old_u       = old_uncertain
        self._old_c       = old_comment

    def redo(self):
        self._page._do_set_label_flag(self._image_path, self._new_u, self._new_c)

    def undo(self):
        self._page._do_set_label_flag(self._image_path, self._old_u, self._old_c)


class SetMultiLabelsCommand(QUndoCommand):
    """Replace the full multi-label set for one image in a single undoable step."""

    def __init__(self, page, image_path: str, new_labels: list, old_labels: list):
        new_str = ", ".join(new_labels) if new_labels else "(kein)"
        super().__init__(f"Multi-Labels → {new_str}")
        self._page       = page
        self._image_path = image_path
        self._new        = list(new_labels)
        self._old        = list(old_labels)

    def redo(self):
        self._page._do_set_multi_labels(self._image_path, self._new)

    def undo(self):
        self._page._do_set_multi_labels(self._image_path, self._old)


class SetImageLabelCommand(QUndoCommand):
    """Assign a single label to one image; undoable."""

    def __init__(self, page, image_path: str, new_label: str, old_label: str):
        lbl_from = old_label or "(kein)"
        lbl_to   = new_label or "(kein)"
        super().__init__(f"Label: {lbl_from} → {lbl_to}")
        self._page       = page
        self._image_path = image_path
        self._new        = new_label
        self._old        = old_label

    def redo(self):
        self._page._do_set_image_label(self._image_path, self._new)

    def undo(self):
        self._page._do_set_image_label(self._image_path, self._old)


# ---------------------------------------------------------------------------
# ROI add / delete
# ---------------------------------------------------------------------------

class AddROICommand(QUndoCommand):
    """Add one ROI to an image; undo removes it again."""

    def __init__(self, page, image_path: str, roi_data: dict):
        super().__init__(f"ROI hinzufügen ({roi_data.get('type','rect')})")
        self._page       = page
        self._image_path = image_path
        self._roi        = copy.deepcopy(roi_data)

    def redo(self):
        self._page._do_add_roi(self._image_path, copy.deepcopy(self._roi))

    def undo(self):
        self._page._do_delete_roi(self._image_path, self._roi["id"])


class DeleteROICommand(QUndoCommand):
    """Delete one ROI from an image; undo re-adds it."""

    def __init__(self, page, image_path: str, roi_data: dict):
        super().__init__(f"ROI löschen ({roi_data.get('type','rect')})")
        self._page       = page
        self._image_path = image_path
        self._roi        = copy.deepcopy(roi_data)

    def redo(self):
        self._page._do_delete_roi(self._image_path, self._roi["id"])

    def undo(self):
        self._page._do_add_roi(self._image_path, copy.deepcopy(self._roi))


# ---------------------------------------------------------------------------
# ROI label assignment
# ---------------------------------------------------------------------------

class AssignROILabelCommand(QUndoCommand):
    """Change the label (and colour) of an existing ROI; fully undoable."""

    def __init__(self, page, image_path: str, roi_id: str,
                 new_label: str, new_color: str,
                 old_label: str, old_color: str):
        super().__init__(f"ROI-Label: {old_label or '–'} → {new_label or '–'}")
        self._page       = page
        self._image_path = image_path
        self._roi_id     = roi_id
        self._new_label  = new_label
        self._new_color  = new_color
        self._old_label  = old_label
        self._old_color  = old_color

    def redo(self):
        self._page._do_assign_roi_label(
            self._image_path, self._roi_id, self._new_label, self._new_color)

    def undo(self):
        self._page._do_assign_roi_label(
            self._image_path, self._roi_id, self._old_label, self._old_color)


# ---------------------------------------------------------------------------
# ROI move / resize
# ---------------------------------------------------------------------------

class MoveROICommand(QUndoCommand):
    """
    Record a ROI move or resize; consecutive moves on the same ROI are merged.

    Uses QUndoCommand.mergeWith() so that dragging a ROI produces a single
    undo step rather than one per mouse event.
    """

    def __init__(self, page, image_path: str, new_data: dict, old_data: dict):
        super().__init__("ROI verschieben")
        self._page       = page
        self._image_path = image_path
        self._new        = copy.deepcopy(new_data)
        self._old        = copy.deepcopy(old_data)

    # Merge consecutive moves of the same ROI into a single undo step
    def id(self):
        return 1001

    def mergeWith(self, other) -> bool:
        """Merge *other* into this command when both move the same ROI."""
        if not isinstance(other, MoveROICommand):
            return False
        if other._roi_id() != self._roi_id():
            return False
        # Keep the latest position but retain the original starting position for undo
        self._new = other._new
        return True

    def _roi_id(self):
        """Return the ID of the ROI being moved."""
        return self._new.get("id", "")

    def redo(self):
        self._page._do_move_roi(self._image_path, self._new)

    def undo(self):
        self._page._do_move_roi(self._image_path, self._old)
