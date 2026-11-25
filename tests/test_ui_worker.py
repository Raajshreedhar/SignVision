"""LiveWorker (src/ui/worker.py) with a fake camera and fake models: frames, commits, snapshots, errors, threading."""

import time

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtTest import QSignalSpy

from src import config
from src.config import COOLDOWN_FRAMES, DEBOUNCE_FRAMES, LOST_GRACE_FRAMES, SMOOTH_WINDOW
from src.perf import LatencyTracker
from src.state_machine import CommitStateMachine
from src.ui.worker import LiveWorker
from tests.ui_fakes import HAND, FakeCapture, FakeClassifier, FakeDetector, SnapshotSink, make_pipeline, wait_until

# As in test_pipeline.py: long enough per phase for the grace frames, the
# smoothing window, the cooldown and the debounce.
N = COOLDOWN_FRAMES + LOST_GRACE_FRAMES + SMOOTH_WINDOW + DEBOUNCE_FRAMES + 2
NO_FRAME_ERROR = "ERROR: the webcam returned no frame 5 times in a row; exiting."


def make_worker(capture=None, detector=None, classifier=None, clock=time.perf_counter):
    detector = detector if detector is not None else FakeDetector()
    classifier = classifier if classifier is not None else FakeClassifier()
    machine = CommitStateMachine(config.CLASSES, config.CONFIDENCE_THRESHOLD, config.DEBOUNCE_FRAMES,
                                 config.COOLDOWN_FRAMES)
    capture = capture if capture is not None else FakeCapture()
    worker = LiveWorker(capture, make_pipeline(detector, classifier), machine, LatencyTracker(), "cpu", clock=clock)
    return worker, detector, classifier


@pytest.fixture
def stopped_threads():
    """Workers started in a test; each is stopped and joined at teardown."""
    workers = []
    yield workers
    for worker in workers:
        worker.stop()
        worker.wait(5000)


def test_step_publishes_a_mirrored_snapshot_with_the_hud(qapp):
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, :640] = (0, 0, 255)  # BGR red on the left
    frame[:, 640:] = (255, 0, 0)  # BGR blue on the right
    worker, _, _ = make_worker(FakeCapture([frame]), FakeDetector([HAND]))
    assert worker.step_once() is True

    snap = worker.take_snapshot()
    assert (snap.image.width(), snap.image.height()) == (1280, 720)
    assert snap.image.pixelColor(10, 10) == QColor(0, 0, 255)  # blue: the frame was mirrored
    assert snap.image.pixelColor(1270, 10) == QColor(255, 0, 0)
    assert set(snap.hud) == {"fps", "det_ms", "cls_ms", "device"}
    assert snap.hud["device"] == "cpu" and snap.hud["cls_ms"] is not None
    assert snap.result.selection is not None and snap.result.label == "A"
    assert snap.read_at <= time.perf_counter()


def test_phases_commit_a_then_ab(qapp):
    worker, detector, classifier = make_worker()

    def phase(dets, label):
        detector.dets, classifier.label = dets, label
        for _ in range(N):
            assert worker.step_once() is True
        return worker.text()

    assert phase([HAND], "A") == "A"
    assert phase([], "A") == "A"
    assert phase([HAND], "B") == "AB"


def test_delete_last_and_clear_text_return_the_new_text(qapp):
    worker, _, _ = make_worker()
    worker.machine.text = "ABC"
    assert worker.delete_last() == "AB" and worker.machine.text == "AB"
    assert worker.clear_text() == "" and worker.text() == ""
    assert worker.delete_last() == ""


def test_frame_ready_waits_until_the_snapshot_is_taken(qapp):
    ticks = iter(range(1, 1000))  # step k reads at tick 2k - 1 and finishes at tick 2k
    worker, _, _ = make_worker(clock=lambda: float(next(ticks)))
    spy = QSignalSpy(worker.frame_ready)
    worker.step_once()
    worker.step_once()
    assert spy.count() == 1
    snap = worker.take_snapshot()
    assert snap.read_at == 3.0  # the second frame's snapshot
    assert worker.take_snapshot() is None  # nothing new since
    worker.step_once()
    assert spy.count() == 2


def test_five_failed_reads_in_a_row_end_the_loop(qapp):
    worker, _, _ = make_worker(FakeCapture(fail=True))
    assert [worker.step_once() for _ in range(4)] == [True] * 4
    assert worker.error is None
    assert worker.step_once() is False
    assert worker.error == NO_FRAME_ERROR

    capture = FakeCapture(fail=True)
    worker, _, _ = make_worker(capture)
    assert [worker.step_once() for _ in range(4)] == [True] * 4
    capture.fail = False
    assert worker.step_once() is True  # a good read resets the count
    capture.fail = True
    assert [worker.step_once() for _ in range(4)] == [True] * 4
    assert worker.error is None
    assert worker.step_once() is False and worker.error == NO_FRAME_ERROR


def test_latency_is_recorded_per_frame_and_cls_only_with_a_hand(qapp):
    worker, detector, _ = make_worker()
    for dets in ([], [], [HAND], [HAND], [HAND]):
        detector.dets = dets
        worker.step_once()
    summary = worker.perf.summary()
    assert summary["total_ms"]["n"] == 5
    assert summary["det_ms"]["n"] == 5
    assert summary["cls_ms"]["n"] == 3


def test_threaded_run_delivers_snapshots_and_stops(qapp, stopped_threads):
    worker, _, _ = make_worker(FakeCapture(delay_s=0.005), FakeDetector([HAND]))
    sink = SnapshotSink(worker)
    stopped_threads.append(worker)
    worker.start()
    assert wait_until(lambda: len(sink.snapshots) >= 3)
    worker.stop()
    assert worker.wait(5000) is True
    assert worker.error is None and worker.exception is None


def test_run_stores_an_exception_instead_of_raising(qapp):
    worker, _, _ = make_worker()

    def boom(frame):
        raise RuntimeError("boom")

    worker.pipeline.process = boom
    worker.run()  # directly, on this thread
    assert isinstance(worker.exception, RuntimeError) and str(worker.exception) == "boom"
    assert worker.error is None


def test_buttons_do_not_wait_for_a_slow_frame(qapp, stopped_threads):
    detector = FakeDetector([HAND], delay_s=0.2)
    worker, _, _ = make_worker(detector=detector)
    worker.machine.text = "AB"
    stopped_threads.append(worker)
    worker.start()
    assert wait_until(lambda: detector.calls >= 1)
    t = time.perf_counter()
    assert worker.delete_last() == "A"
    assert time.perf_counter() - t < 0.1
    worker.stop()
    assert worker.wait(5000) is True
