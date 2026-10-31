"""Edit menu Undo/Redo bound to LabelingPage._undo_stack."""
import pytest


@pytest.fixture
def win(qtbot):
    from gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_edit_menu_actions_exist(win):
    assert hasattr(win, "_undo_action")
    assert hasattr(win, "_redo_action")
    titles = [a.text() for a in win.menuBar().actions()]
    # First action after File should be Edit (de: Bearbeiten / en: Edit)
    assert any(t in ("Bearbeiten", "Edit") for t in titles)


def test_edit_actions_track_undo_stack(win, sample_project):
    from gui.labeling_commands import SetImageLabelCommand

    win.labeling_page.set_project(sample_project)
    stack = win.labeling_page._undo_stack
    assert not win._undo_action.isEnabled()
    assert not win._redo_action.isEnabled()

    path = sample_project.images[0]
    old = sample_project.get_image_label(path)
    stack.push(SetImageLabelCommand(win.labeling_page, path, "schlecht", old))

    assert win._undo_action.isEnabled()
    assert not win._redo_action.isEnabled()

    win._undo_action.trigger()
    assert not win._undo_action.isEnabled()
    assert win._redo_action.isEnabled()

    win._redo_action.trigger()
    assert win._undo_action.isEnabled()
    assert not win._redo_action.isEnabled()


def test_locale_keys_present():
    from locales.de import STRINGS as de
    from locales.en import STRINGS as en
    for key in ("menu.edit", "menu.edit.undo", "menu.edit.redo"):
        assert key in de
        assert key in en
