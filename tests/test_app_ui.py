"""Headless checks of the live app's UI code (B10): draw_overlay states and the missing-weights exit.

The app itself is never run here: draw_overlay draws onto a zero frame with
cv2.imshow patched to fail, and main() must exit before any model or camera
is touched.
"""

import cv2
import numpy as np
import pytest

from src import app, config
from src.pipeline import FrameResult
from src.tracking import Detection, Selection

HUD = {"fps": 27.5, "det_ms": 11.2, "cls_ms": 4.3, "device": "mps"}
BOX = Detection(440, 160, 840, 560, 0.91)
OTHER = Detection(950, 300, 1100, 470, 0.55)


@pytest.fixture
def drawn(monkeypatch):
    """Record every putText string and rectangle call (still drawing them); make imshow fail."""
    texts, rects = [], []
    real_put, real_rect = cv2.putText, cv2.rectangle

    def put(img, text, *args, **kwargs):
        texts.append(text)
        return real_put(img, text, *args, **kwargs)

    def rect(img, pt1, pt2, color, thickness=1, *args, **kwargs):
        rects.append((tuple(pt1), tuple(pt2), tuple(color), thickness))
        return real_rect(img, pt1, pt2, color, thickness, *args, **kwargs)

    def no_imshow(*_args, **_kwargs):
        raise AssertionError("draw_overlay must not call imshow")

    monkeypatch.setattr(cv2, "putText", put)
    monkeypatch.setattr(cv2, "rectangle", rect)
    monkeypatch.setattr(cv2, "imshow", no_imshow)
    return texts, rects


def frame_result(selection, label, conf):
    probs = np.eye(config.NUM_CLASSES, dtype=np.float32)[config.CLASSES.index(label)]
    cls_ran = selection is not None
    return FrameResult(selection, probs, probs, label, conf, 11.2, 4.3 if cls_ran else 0.0, cls_ran)


def render(result, text, hud=HUD):
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = app.draw_overlay(frame, result, text, hud)
    assert out.shape == (720, 1280, 3) and out.dtype == np.uint8
    assert out.any()  # something was drawn
    return out


def test_selected_box_with_other_hands(drawn):
    texts, rects = drawn
    out = render(frame_result(Selection(BOX, 1, False, [OTHER], BOX), "A", 0.91), "HELLO")
    assert ((440, 160), (840, 560), app.GREEN, 3) in rects
    assert ((950, 300), (1100, 470), app.GRAY, 1) in rects
    assert "A 91%" in texts
    assert "FPS 27.5  det 11.2 ms  cls 4.3 ms  mps" in texts
    assert "Text: HELLO" in texts and "q: quit   c: clear" in texts
    assert "No hand -> NOTHING" not in texts
    assert tuple(out[560, 640]) == app.GREEN  # bottom edge of the selected box
    assert tuple(out[470, 1025]) == app.GRAY  # bottom edge of the dimmed box


def test_held_box_is_yellow(drawn):
    texts, rects = drawn
    out = render(frame_result(Selection(BOX, 1, True, [], BOX), "B", 0.8), "AB")
    assert ((440, 160), (840, 560), app.YELLOW, 2) in rects
    assert all(color != app.GREEN for _, _, color, _ in rects)
    assert "B 80%" in texts and "Text: AB" in texts
    assert tuple(out[560, 640]) == app.YELLOW


def test_no_hand_shows_the_nothing_status(drawn):
    texts, rects = drawn
    render(frame_result(None, "NOTHING", 1.0), "", {**HUD, "cls_ms": None})
    assert "No hand -> NOTHING" in texts
    assert "FPS 27.5  det 11.2 ms  cls - ms  mps" in texts
    assert [color for _, _, color, _ in rects] == [app.BLACK]  # only the text panel, no boxes


def test_long_text_keeps_the_newest_letters_inside_the_panel(drawn):
    texts, _ = drawn
    render(frame_result(None, "NOTHING", 1.0), "ABCDEFGHIJ" * 50 + "XYZ")
    panel = [t for t in texts if t.startswith("Text: ")]
    assert len(panel) == 1
    assert panel[0].startswith("Text: ...") and panel[0].endswith("XYZ")
    assert cv2.getTextSize(panel[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0] <= 1280 - 20


@pytest.mark.parametrize(
    "attr, trainer",
    [("DET_WEIGHTS", "src.train_detector"), ("CLS_WEIGHTS", "src.train_classifier")],
)
def test_main_exits_1_naming_the_trainer_when_weights_are_missing(attr, trainer, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, attr, tmp_path / "missing.pt")
    with pytest.raises(SystemExit) as exc:
        app.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert trainer in err and str(tmp_path / "missing.pt") in err
    assert "Traceback" not in err
