"""CameraView and its pure helpers (src/ui/camera_view.py), rendered offscreen with grab()."""

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtTest import QSignalSpy

from src.tracking import Selection
from src.ui.camera_view import CameraView, OverlayBox, fit_rect, hud_text, label_text, overlay_boxes
from tests.ui_fakes import HAND, HUD, OTHER, gray_frame, make_result, make_snapshot

# HAND (440, 160, 840, 560) in a 1280x720 frame, drawn in a 640x480 view:
# the video is (0, 60, 640, 360) at scale 0.5, so the box is (220, 140)-(420, 340).
LEFT_EDGE_X, MID_Y = 220, 240


@pytest.fixture
def view(qapp):
    widget = CameraView()
    widget.resize(640, 480)
    yield widget
    widget.close()


def _near(color, target, tolerance=80):
    return all(abs(a - b) <= tolerance for a, b in zip(color.getRgb()[:3], target.getRgb()[:3]))


def test_label_text_is_the_label_and_its_confidence():
    assert label_text(make_result(Selection(HAND, 1, False, [], HAND), "A", 0.91)) == "A 91%"


def test_overlay_boxes_for_each_hand_state():
    selected = overlay_boxes(make_result(Selection(HAND, 1, False, [OTHER], HAND), "A", 0.91))
    assert selected == [OverlayBox("other", OTHER, None), OverlayBox("selected", HAND, "A 91%")]
    held = overlay_boxes(make_result(Selection(HAND, 1, True, [], HAND), "B", 0.8))
    assert held == [OverlayBox("held", HAND, "B 80%")]
    assert overlay_boxes(make_result(None, "NOTHING", 1.0)) == []


def test_hud_text_with_and_without_the_classifier():
    assert hud_text(HUD) == "FPS 27.5 · Detector 11.2 ms · Classifier 4.3 ms · MPS"
    assert hud_text({**HUD, "cls_ms": None}) == "FPS 27.5 · Detector 11.2 ms · Classifier – · MPS"


def test_fit_rect_letterboxes_and_pillarboxes():
    assert fit_rect(1280, 720, 640, 480).getRect() == (0, 60, 640, 360)
    x, y, w, h = fit_rect(1280, 720, 1000, 400).getRect()
    assert (x, y, w, h) == pytest.approx((144.44, 0, 711.11, 400), abs=0.01)


def test_selected_box_is_drawn_in_the_accent_color(view):
    snap = make_snapshot(make_result(Selection(HAND, 1, False, [OTHER], HAND), "A", 0.91))
    view.set_snapshot(snap)
    image = view.grab().toImage()
    assert image.pixelColor(5, 5).getRgb()[:3] == (0, 0, 0)
    accent = view.palette().color(QPalette.ColorRole.Accent)
    assert _near(image.pixelColor(LEFT_EDGE_X, MID_Y), accent)
    # OTHER (950, 300, 1100, 470) -> bottom edge at y 60 + 235 = 295, x 475 + 37: thin white at 55 %.
    other = image.pixelColor(512, 295)
    assert 60 < other.red() < 200 and other.red() == other.green() == other.blue()


def test_held_box_is_dashed(view):
    view.set_snapshot(make_snapshot(make_result(Selection(HAND, 1, True, [], HAND), "A", 0.86)))
    image = view.grab().toImage()
    accent = view.palette().color(QPalette.ColorRole.Accent)
    edge = [_near(image.pixelColor(LEFT_EDGE_X, y), accent) for y in range(160, 320)]
    assert any(edge) and not all(edge)  # dashes and gaps along the left edge


def test_no_hand_capsule_sits_at_the_video_top_left(view):
    frame = gray_frame(200)
    view.set_snapshot(make_snapshot(make_result(None, "NOTHING", 1.0), frame=frame, hud={**HUD, "cls_ms": None}))
    image = view.grab().toImage()
    row = 60 + 16 + 14  # the capsule's middle row
    assert image.pixelColor(16 + 4, row).red() == pytest.approx(200 * 0.4, abs=6)  # 60 % black over gray 200
    assert image.pixelColor(8, row).red() == pytest.approx(200, abs=2)  # left of the capsule: the frame
    assert any(image.pixelColor(x, row).red() > 230 for x in range(16 + 10, 16 + 120))  # white text


def test_waiting_state_renders_without_a_snapshot(view):
    image = view.grab().toImage()
    assert image.pixelColor(5, 5).getRgb()[:3] == (0, 0, 0)
    middle = [image.pixelColor(x, 240) for x in range(200, 440)]
    assert any(c.red() > 60 for c in middle)  # the waiting text


def test_frame_painted_fires_once_per_snapshot(view):
    spy = QSignalSpy(view.frame_painted)
    view.set_snapshot(make_snapshot(make_result(None, "NOTHING", 1.0)))
    view.grab()
    view.grab()
    assert spy.count() == 1
    assert spy.at(0)[0] >= 0
    view.set_snapshot(make_snapshot(make_result(None, "NOTHING", 1.0)))
    view.grab()
    assert spy.count() == 2
