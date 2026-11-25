"""Headless checks of the live app's start-up and wiring: src.app.main() with the Qt window.

No camera and no model is touched: build_pipeline returns a SignPipeline
around fakes, resolve_device returns 'cpu', and cv2.VideoCapture builds a
FakeCapture. The window runs on the offscreen Qt platform (conftest), and a
10 s safety timer ends the event loop if a test would otherwise hang.
"""

import sys
from types import SimpleNamespace

import cv2
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src import app, config
from src.ui.window import MainWindow
from tests.ui_fakes import HAND, FakeCapture, FakeClassifier, FakeDetector, make_pipeline

SAFETY_MS = 10000


@pytest.fixture
def fake_app(qapp, tmp_path, monkeypatch):
    """Weight files that exist, the CPU, fake models and a FakeCapture for main(); returns them to adjust."""
    import src.yolo_env

    for attr in ("DET_WEIGHTS", "CLS_WEIGHTS", "CLS_META"):
        path = tmp_path / attr.lower()
        path.write_text("not used")
        monkeypatch.setattr(config, attr, path)
    monkeypatch.setattr(src.yolo_env, "resolve_device", lambda pref="auto": "cpu")
    setup = SimpleNamespace(
        detector=FakeDetector([HAND], device="cpu"), classifier=FakeClassifier(), capture=FakeCapture(delay_s=0.01),
        timed_out=[],
    )
    monkeypatch.setattr(app, "build_pipeline", lambda device: make_pipeline(setup.detector, setup.classifier))
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: setup.capture)

    safety = QTimer()
    safety.setSingleShot(True)
    safety.timeout.connect(lambda: (setup.timed_out.append(True), QApplication.quit()))
    safety.start(SAFETY_MS)
    yield setup
    safety.stop()
    assert not setup.timed_out, "the event loop only ended through the safety timer"


def test_main_exits_1_naming_pyside6_when_it_is_missing(fake_app, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "src.ui.window", None)  # the import fails as without PySide6
    with pytest.raises(SystemExit) as exc:
        app.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "PySide6" in err and "pip install -r requirements.txt" in err
    assert "Traceback" not in err
    assert fake_app.capture.reads == 0


def test_main_returns_1_when_the_camera_does_not_open(fake_app, capsys):
    fake_app.capture.opened = False
    assert app.main() == 1
    err = capsys.readouterr().err
    assert "could not open the webcam" in err and app.CAMERA_HINT in err


def test_frames_flow_until_q_closes_the_window(fake_app, capsys):
    pressed = []

    def press_q():
        windows = [w for w in QApplication.topLevelWidgets() if isinstance(w, MainWindow) and w.isVisible()]
        pressed.append(len(windows))
        for window in windows:
            QTest.keyClick(window, Qt.Key.Key_Q)

    press = QTimer()
    press.setSingleShot(True)
    press.timeout.connect(press_q)
    press.start(500)
    try:
        assert app.main() == 0
    finally:
        press.stop()
    assert pressed == [1]
    assert fake_app.capture.reads > 0 and fake_app.capture.released
    out = capsys.readouterr().out
    assert "Using device: cpu" in out and "Session latency over" in out


def test_main_returns_1_when_the_camera_stops_returning_frames(fake_app, capsys):
    fake_app.capture.fail = True
    assert app.main() == 1
    assert "returned no frame 5 times in a row" in capsys.readouterr().err
    assert fake_app.capture.released


def test_a_pipeline_error_is_raised_after_the_summary(fake_app, capsys):
    def boom(frame):
        raise RuntimeError("boom")

    fake_app.detector.detect = boom
    with pytest.raises(RuntimeError, match="boom"):
        app.main()
    assert fake_app.capture.released
    assert "No frames were processed." in capsys.readouterr().out


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
