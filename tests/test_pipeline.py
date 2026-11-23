"""src.pipeline.SignPipeline with a fake detector, classifier and clock (design sections 7.2, 12; N4, N6)."""

import numpy as np
import pytest

from src.config import (
    CLASSES,
    CLS_IMGSZ,
    CONFIDENCE_THRESHOLD,
    COOLDOWN_FRAMES,
    CROP_PAD,
    DEBOUNCE_FRAMES,
    LOST_GRACE_FRAMES,
    NOTHING_IDX,
    NUM_CLASSES,
    SMOOTH_WINDOW,
)
from src.pipeline import SignPipeline
from src.preprocess import classifier_input
from src.smoothing import ProbSmoother
from src.state_machine import CommitStateMachine
from src.tracking import Detection, HandSelector

# N4: long enough per phase for the grace frames, the smoothing window, the
# cooldown and the debounce, so a hand-free phase really releases the latch.
N = COOLDOWN_FRAMES + LOST_GRACE_FRAMES + SMOOTH_WINDOW + DEBOUNCE_FRAMES + 2
HAND = Detection(440, 160, 840, 560, 0.9)
FRAME = np.random.default_rng(7).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
STEP_S = 0.005  # the fake clock advances 5 ms per call


class FakeDetector:
    """Returns the scripted ``dets`` on every call."""

    def __init__(self):
        self.dets = []

    def detect(self, frame):
        return list(self.dets)


class FakeClassifier:
    """One-hot output for the scripted ``label``; keeps every input it was given."""

    def __init__(self, label="A", crop_pad=CROP_PAD, imgsz=CLS_IMGSZ):
        self.label = label
        self.crop_pad = crop_pad
        self.imgsz = imgsz
        self.inputs = []

    def classify(self, img):
        self.inputs.append(img)
        probs = np.zeros(NUM_CLASSES, np.float32)
        probs[CLASSES.index(self.label)] = 1.0
        return probs


class StepClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        self.now += STEP_S
        return self.now


def make_pipeline(detector, classifier):
    return SignPipeline(detector, classifier, HandSelector(), ProbSmoother(), clock=StepClock())


def test_no_detections_give_nothing_without_running_the_classifier():
    classifier = FakeClassifier()
    r = make_pipeline(FakeDetector(), classifier).process(FRAME)
    assert r.selection is None
    assert r.label == "NOTHING" and r.conf == pytest.approx(1.0)
    assert r.cls_ran is False and r.cls_ms == 0
    assert r.det_ms == pytest.approx(STEP_S * 1000)
    np.testing.assert_array_equal(r.probs_raw, np.eye(NUM_CLASSES, dtype=np.float32)[NOTHING_IDX])
    assert classifier.inputs == []


@pytest.mark.parametrize("crop_pad, imgsz", [(CROP_PAD, CLS_IMGSZ), (0.1, 160)])
def test_the_classifier_gets_the_crop_of_the_raw_box(crop_pad, imgsz):  # N6
    detector, classifier = FakeDetector(), FakeClassifier("C", crop_pad, imgsz)
    pipeline = make_pipeline(detector, classifier)
    detector.dets = [Detection(400, 150, 800, 550, 0.9)]
    pipeline.process(FRAME)
    detector.dets = [HAND]
    r = pipeline.process(FRAME)

    assert r.cls_ran is True and r.label == "C"
    assert r.det_ms == pytest.approx(STEP_S * 1000) and r.cls_ms == pytest.approx(STEP_S * 1000)
    assert r.selection.raw == HAND and r.selection.box != HAND  # a continuing track: EMA box != raw box
    crop = classifier.inputs[-1]
    assert crop.shape == (imgsz, imgsz, 3) and crop.dtype == np.uint8
    np.testing.assert_array_equal(crop, classifier_input(FRAME, HAND.xyxy, crop_pad, imgsz))
    assert not np.array_equal(crop, classifier_input(FRAME, r.selection.box.xyxy, crop_pad, imgsz))


def test_the_smoother_resets_exactly_once_per_new_hand():
    detector = FakeDetector()
    pipeline = make_pipeline(detector, FakeClassifier())
    resets = []
    real_reset = pipeline.smoother.reset

    def spy():
        resets.append(frame_no)
        real_reset()

    pipeline.smoother.reset = spy
    gone = LOST_GRACE_FRAMES + 1  # empty frames until the selector reports no hand
    script = [[]] * gone + [[HAND]] * 5 + [[]] * gone + [[HAND]] * 5  # None -> track 1 -> None -> track 2
    tracks = []
    for frame_no, dets in enumerate(script):
        detector.dets = dets
        r = pipeline.process(FRAME)
        tracks.append(None if r.selection is None else r.selection.track_id)

    assert tracks == [None] * gone + [1] * 5 + [1] * LOST_GRACE_FRAMES + [None] + [2] * 5
    assert resets == [gone, gone + 5 + gone]  # entering track 1 and track 2; never on hand -> no hand


def test_hand_a_then_no_hand_then_hand_b_commits_ab():
    detector, classifier = FakeDetector(), FakeClassifier()
    pipeline = make_pipeline(detector, classifier)
    machine = CommitStateMachine(CLASSES, CONFIDENCE_THRESHOLD, DEBOUNCE_FRAMES, COOLDOWN_FRAMES)

    def phase(dets, label):
        detector.dets, classifier.label = dets, label
        for _ in range(N):
            r = pipeline.process(FRAME)
            machine.step(r.label, r.conf)
        return machine.text

    assert phase([HAND], "A") == "A"  # one commit however long the sign is held
    assert phase([], "A") == "A"  # the held frames still classify as A, then NOTHING releases the latch
    assert phase([HAND], "B") == "AB"
