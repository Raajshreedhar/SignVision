"""The camera loop on a worker thread, so the window stays responsive while the models run.

LiveWorker.run repeats step_once, the old OpenCV loop body without the
drawing: read a frame, mirror it, run SignPipeline.process, step the
CommitStateMachine, and record the frame's latency. Every processed frame is
stepped exactly once and in order, so commit timing does not depend on how
often the window repaints.

The worker keeps only the newest Snapshot (frame image, FrameResult, HUD
values). It emits frame_ready only after the previous snapshot was taken, so
queued events never pile up in the GUI thread. The Delete and Clear buttons
call delete_last and clear_text, which take the same lock the worker holds
around machine.step, so a button press never races a commit.
"""

import threading
import time
from dataclasses import dataclass

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src import config
from src.pipeline import FrameResult


@dataclass(frozen=True)
class Snapshot:
    """One processed frame, as the window shows it."""

    image: QImage  # the mirrored camera frame (an owned copy)
    result: FrameResult
    hud: dict  # fps, det_ms, cls_ms (None when the classifier did not run) and device
    read_at: float  # clock time just before the frame was read


class LiveWorker(QThread):
    """Runs capture -> flip -> SignPipeline -> CommitStateMachine -> LatencyTracker off the GUI thread.

    After run() ends, ``error`` holds the message for a camera that stopped
    returning frames and ``exception`` any exception the loop raised; both
    are None after a normal stop().
    """

    frame_ready = Signal()

    def __init__(self, capture, pipeline, machine, perf, device, clock=time.perf_counter):
        super().__init__()
        self.capture = capture
        self.pipeline = pipeline
        self.machine = machine
        self.perf = perf
        self.device = device
        self.clock = clock
        self.error: str | None = None
        self.exception: BaseException | None = None
        self._failures = 0
        self._stop = threading.Event()
        self._machine_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._latest = None
        self._pending = False

    def step_once(self):
        """Read and process one frame. Returns False when the loop must end (the camera keeps failing)."""
        t = self.clock()
        ok, frame = self.capture.read()
        if not ok:
            self._failures += 1
            if self._failures >= config.MAX_READ_FAILURES:
                self.error = f"ERROR: the webcam returned no frame {self._failures} times in a row; exiting."
                return False
            return True
        self._failures = 0
        frame = cv2.flip(frame, 1)  # mirror, so moving right on screen matches moving right in reality

        r = self.pipeline.process(frame)
        with self._machine_lock:
            self.machine.step(r.label, r.conf)
        cls_ms = r.cls_ms if r.cls_ran else None
        hud = {"fps": self.perf.fps(), "det_ms": r.det_ms, "cls_ms": cls_ms, "device": self.device}
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888).copy()
        self._publish(Snapshot(image=image, result=r, hud=hud, read_at=t))
        self.perf.record(det_ms=r.det_ms, cls_ms=cls_ms, total_ms=(self.clock() - t) * 1000)
        return True

    def _publish(self, snapshot):
        with self._snapshot_lock:
            self._latest = snapshot
            notify = not self._pending
            self._pending = True
        if notify:
            self.frame_ready.emit()

    def run(self):
        """Loop step_once until stop() or the camera gives up; store any exception instead of raising."""
        try:
            while not self._stop.is_set():
                if not self.step_once():
                    break
        except BaseException as exc:  # re-raised on the main thread by src.ui.window.run
            self.exception = exc

    def stop(self):
        """Ask the loop to end after the current frame."""
        self._stop.set()

    def take_snapshot(self):
        """The newest snapshot not yet taken, or None; the next published frame emits frame_ready again."""
        with self._snapshot_lock:
            snapshot, self._latest = self._latest, None
            self._pending = False
        return snapshot

    def text(self):
        """The committed text."""
        with self._machine_lock:
            return self.machine.text

    def delete_last(self):
        """Remove the last letter (the Delete button) and return the new text."""
        with self._machine_lock:
            return self.machine.delete_last()

    def clear_text(self):
        """Empty the text (the Clear button and the c key) and return it."""
        with self._machine_lock:
            self.machine.clear()
            return self.machine.text
