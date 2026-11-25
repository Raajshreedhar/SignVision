"""Fakes shared by the Qt UI tests: detector, classifier, camera, controller and canned results.

A plain helper module (pytest does not collect it), imported as
``from tests.ui_fakes import ...``. Nothing here opens a camera or loads a model.
"""

import time

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from src import config
from src.pipeline import FrameResult, SignPipeline
from src.smoothing import ProbSmoother
from src.tracking import Detection, HandSelector
from src.ui.worker import Snapshot

HAND = Detection(440, 160, 840, 560, 0.91)
OTHER = Detection(950, 300, 1100, 470, 0.55)
HUD = {"fps": 27.5, "det_ms": 11.2, "cls_ms": 4.3, "device": "mps"}


class FakeDetector:
    """Returns the scripted ``dets`` on every call, after sleeping ``delay_s``; counts its calls."""

    def __init__(self, dets=(), device="cpu", delay_s=0.0):
        self.dets = list(dets)
        self.device = device
        self.delay_s = delay_s
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return list(self.dets)


class FakeClassifier:
    """One-hot output for the scripted ``label``."""

    def __init__(self, label="A", crop_pad=config.CROP_PAD, imgsz=config.CLS_IMGSZ):
        self.label = label
        self.crop_pad = crop_pad
        self.imgsz = imgsz

    def classify(self, img):
        probs = np.zeros(config.NUM_CLASSES, np.float32)
        probs[config.CLASSES.index(self.label)] = 1.0
        return probs


def gray_frame(value=96):
    """A (720, 1280, 3) uint8 BGR frame of one gray level."""
    return np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), value, np.uint8)


class FakeCapture:
    """A cv2.VideoCapture stand-in: cycles through ``frames`` (copies), or fails every read while ``fail``."""

    def __init__(self, frames=None, fail=False, delay_s=0.0, opened=True):
        self.frames = list(frames) if frames is not None else [gray_frame()]
        self.fail = fail
        self.delay_s = delay_s
        self.opened = opened
        self.reads = 0
        self.props = {}
        self.released = False

    def read(self):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.reads += 1
        if self.fail:
            return False, None
        return True, self.frames[(self.reads - 1) % len(self.frames)].copy()

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def release(self):
        self.released = True


def make_pipeline(detector, classifier):
    """A real SignPipeline (HandSelector, ProbSmoother, real clock) around the fakes."""
    return SignPipeline(detector, classifier, HandSelector(), ProbSmoother())


def make_result(selection, label, conf):
    """A FrameResult with a one-hot ``label``; the classifier ran only when there is a selection."""
    probs = np.eye(config.NUM_CLASSES, dtype=np.float32)[config.CLASSES.index(label)]
    cls_ran = selection is not None
    return FrameResult(selection, probs, probs, label, conf, 11.2, 4.3 if cls_ran else 0.0, cls_ran)


def make_snapshot(result, frame=None, hud=HUD, read_at=None):
    """A Snapshot of ``frame`` (default: a black 1280x720 frame) with ``result`` and ``hud``."""
    frame = frame if frame is not None else gray_frame(0)
    h, w = frame.shape[:2]
    image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888).copy()
    read_at = time.perf_counter() if read_at is None else read_at
    return Snapshot(image=image, result=result, hud=dict(hud), read_at=read_at)


class FakeController(QObject):
    """Follows the window's controller protocol; records Delete and Clear presses and stop()."""

    frame_ready = Signal()

    def __init__(self, text="", snapshot=None):
        super().__init__()
        self.text_value = text
        self.snapshot = snapshot
        self.calls = []
        self.stopped = False

    def take_snapshot(self):
        snapshot, self.snapshot = self.snapshot, None
        return snapshot

    def text(self):
        return self.text_value

    def delete_last(self):
        self.calls.append("delete")
        self.text_value = self.text_value[:-1]
        return self.text_value

    def clear_text(self):
        self.calls.append("clear")
        self.text_value = ""
        return self.text_value

    def stop(self):
        self.stopped = True


class SnapshotSink(QObject):
    """Lives in the GUI thread and takes every snapshot a worker announces."""

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.snapshots = []
        worker.frame_ready.connect(self.take)

    @Slot()
    def take(self):
        snapshot = self.worker.take_snapshot()
        if snapshot is not None:
            self.snapshots.append(snapshot)


def wait_until(predicate, timeout_s=5.0):
    """Run the Qt event loop until ``predicate()`` is true or ``timeout_s`` passes; return predicate()."""
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    return predicate()
