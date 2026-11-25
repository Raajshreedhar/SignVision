"""MainWindow (src/ui/window.py) with a FakeController on the offscreen platform: buttons, menus, keys, guide."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest

from src.tracking import Selection
from src.ui.camera_view import hud_text
from src.ui.window import MainWindow
from tests.ui_fakes import HAND, HUD, FakeController, make_result, make_snapshot


def tracking_snapshot():
    return make_snapshot(make_result(Selection(HAND, 1, False, [], HAND), "A", 0.91))


@pytest.fixture
def controller():
    return FakeController()


@pytest.fixture
def window(qapp, controller):
    widget = MainWindow(controller)
    yield widget
    widget.close()


def _show(window):
    window.show()
    assert QTest.qWaitForWindowActive(window)


def _feed(window, controller, text):
    """The worker committed ``text`` and published a frame."""
    controller.text_value = text
    controller.snapshot = tracking_snapshot()
    window.apply_snapshot()


def _click(window, action):
    QTest.mouseClick(window.toolbar.widgetForAction(action), Qt.MouseButton.LeftButton)


def test_toolbar_holds_delete_clear_guide_quit(window):
    labelled = [action for action in window.toolbar.actions() if action.text()]
    assert [action.text() for action in labelled] == ["Delete", "Clear", "Guide", "Quit"]
    assert labelled == [window.delete_action, window.clear_action, window.guide_action, window.quit_action]
    assert [a.objectName() for a in labelled] == ["delete", "clear", "guide", "quit"]
    assert window.toolbar.objectName() == "controls"
    assert not window.toolbar.isMovable() and not window.toolbar.isFloatable()
    assert window.toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    assert window.delete_action.toolTip() == "Delete the last letter"
    assert "(C)" in window.clear_action.toolTip()
    assert "(Q)" in window.quit_action.toolTip()
    assert window.guide_action.isCheckable() and not window.guide_action.isChecked()
    assert window.windowTitle() == "SignVision"


def test_menus_reuse_the_toolbar_actions(window):
    menus = {action.text(): action.menu() for action in window.menuBar().actions()}
    assert list(menus) == ["Edit", "View"]
    edit, view = menus["Edit"].actions(), menus["View"].actions()
    assert len(edit) == 2 and edit[0] is window.delete_action and edit[1] is window.clear_action
    assert len(view) == 1 and view[0] is window.guide_action


def test_accessible_names(window):
    assert window.camera.accessibleName() == "Camera"
    assert window.transcript.accessibleName() == "Text"
    assert window.hud.accessibleName() == "Performance"


def test_a_snapshot_shows_the_text_hud_and_frame(window, controller):
    assert window.hud.text() == ""  # nothing before the first frame
    assert not window.delete_action.isEnabled() and not window.clear_action.isEnabled()
    window.apply_snapshot()  # no snapshot yet: nothing changes
    assert window.camera.snapshot is None

    snap = tracking_snapshot()
    controller.text_value, controller.snapshot = "AB", snap
    window.apply_snapshot()
    assert window.delete_action.isEnabled() and window.clear_action.isEnabled()
    assert window.transcript.displayed_text() == "AB"
    assert window.hud.text() == hud_text(HUD)
    assert window.camera.snapshot is snap


def test_delete_button_removes_the_last_letter(window, controller):
    _show(window)
    _feed(window, controller, "AB")
    _click(window, window.delete_action)
    assert controller.calls == ["delete"]
    assert window.transcript.displayed_text() == "A"
    assert window.delete_action.isEnabled()


def test_clear_button_empties_the_text_and_disables_both(window, controller):
    _show(window)
    _feed(window, controller, "AB")
    _click(window, window.clear_action)
    assert controller.calls == ["clear"]
    assert window.transcript.displayed_text() == ""
    assert not window.delete_action.isEnabled() and not window.clear_action.isEnabled()
    _click(window, window.delete_action)  # disabled: nothing happens
    assert controller.calls == ["clear"]


def test_c_clears_and_q_closes_once(window, controller):
    closed = QSignalSpy(window.closed)
    _show(window)
    _feed(window, controller, "AB")
    QTest.keyClick(window, Qt.Key.Key_C)
    assert controller.calls == ["clear"] and window.transcript.displayed_text() == ""
    QTest.keyClick(window, Qt.Key.Key_Q)
    assert not window.isVisible()
    assert controller.stopped is True
    assert closed.count() == 1
    window.close()  # a second close (as worker.finished does) does not emit again
    assert closed.count() == 1


def test_quit_button_closes_the_window(window, controller):
    closed = QSignalSpy(window.closed)
    _show(window)
    _click(window, window.quit_action)
    assert not window.isVisible() and controller.stopped and closed.count() == 1


def test_guide_button_shows_and_hides_the_guide(window):
    _show(window)
    assert window.guide.isHidden()
    _click(window, window.guide_action)
    assert window.guide_action.isChecked() and window.guide.isVisible()
    assert window.splitter.sizes()[1] == 320
    assert len(window.guide.tiles) == 29
    _click(window, window.guide_action)
    assert not window.guide_action.isChecked() and window.guide.isHidden()


def test_long_text_keeps_the_newest_letters(window, controller):
    _show(window)
    _feed(window, controller, "ABCDEFGHIJ" * 50 + "XYZ")
    shown = window.transcript.displayed_text()
    assert shown.startswith("…") and shown.endswith("XYZ")
    assert len(shown) < 503
