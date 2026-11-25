"""Live ASL detector for SignVision: full-frame YOLO26 detection in a native Qt window.

Every webcam frame is mirrored and scanned whole by the YOLO26 hand detector;
there is no fixed box to hold the hand in. HandSelector picks the signing hand
and keeps following it (solid accent box; dashed while briefly lost; other
hands thin white). The padded square crop of that hand goes through the shared
gray + CLAHE classifier_input to the YOLO26-cls letter classifier, a
sliding-window ProbSmoother steadies the prediction, and CommitStateMachine
turns it into text: a sign held for DEBOUNCE_FRAMES commits once, and a NOTHING
(or a dropped hand) must pass between any two commits. With no hand in view
the pipeline reports NOTHING.

The camera loop runs on a worker thread (src/ui/worker.py) and the PySide6
window (src/ui/window.py) only paints: the committed text at the top, the
camera view with the boxes and the current label, the sign guide as an
optional side pane, and the HUD (FPS, detector ms, classifier ms, device) in
the status bar. The toolbar has Delete, Clear, Guide and Quit; keys: q quits,
c clears the text. The session latency summary prints when the window closes.

Start-up exits 1 with the command to run when a trained model is missing or
the classifier does not match this code, warns when the classifier was built
with a different detector, and falls back to the CPU (one WARNING) if the MPS
warm-up fails. Without PySide6 it exits 1 naming the install command.

cv2, numpy, torch, ultralytics and PySide6 are imported only inside
functions, so `import src.app` (and the CommitStateMachine re-export) stays
headless.

Usage:
    .venv/bin/python -m src.app
"""

import sys
from pathlib import Path

from src import config
from src.state_machine import CommitStateMachine  # re-export

TRAIN_DETECTOR_CMD = ".venv/bin/python -m src.hand_data && .venv/bin/python -m src.train_detector"
TRAIN_CLASSIFIER_CMD = ".venv/bin/python -m src.cls_data && .venv/bin/python -m src.train_classifier"
RETRAIN_CLASSIFIER_CMD = ".venv/bin/python -m src.train_classifier"
NO_DETECTOR_WARNING = (
    "WARNING: classifier trained on uncropped center squares (--no-detector); "
    "accuracy on detector crops will be lower"
)
CAMERA_HINT = (
    "On macOS, allow camera access for the app you run this from (Terminal, iTerm, VS Code, ...) in "
    "System Settings → Privacy & Security → Camera, then start it again."
)
PYSIDE6_MISSING = (
    "ERROR: PySide6 is not installed in this environment. Install the pinned requirements: "
    ".venv/bin/pip install -r requirements.txt"
)


def _rel(path):
    """``path`` relative to the project root when it lies inside it, for short messages."""
    path = Path(path)
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _require_weights():
    """Exit 1, naming the commands to run, when a trained model file is missing."""
    problems = [
        f"No {what} at {_rel(path)}. Train it: {command}"
        for path, what, command in (
            (config.DET_WEIGHTS, "hand detector", TRAIN_DETECTOR_CMD),
            (config.CLS_WEIGHTS, "letter classifier", TRAIN_CLASSIFIER_CMD),
            (config.CLS_META, "letter classifier metadata", TRAIN_CLASSIFIER_CMD),
        )
        if not Path(path).is_file()
    ]
    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)


def _load_models(device):
    """(HandDetector, LetterClassifier) on ``device``; exit 1 if the classifier does not match this code."""
    from src.class_map import ClassOrderError
    from src.pipeline import HandDetector, LetterClassifier, MetaError

    detector = HandDetector(config.DET_WEIGHTS, device)
    try:
        classifier = LetterClassifier(config.CLS_WEIGHTS, config.CLS_META, device)
    except (ClassOrderError, MetaError) as exc:
        print(f"ERROR: {exc}\nretrain the classifier: {RETRAIN_CLASSIFIER_CMD}", file=sys.stderr)
        raise SystemExit(1) from None
    return detector, classifier


def _check_detector_md5(detector, classifier):
    """Warn once when the classifier's crops were built with another detector (design section 4.6)."""
    if classifier.detector_md5 is None:
        print(NO_DETECTOR_WARNING, file=sys.stderr)
    elif classifier.detector_md5 != detector.md5:
        print(
            f"WARNING: {_rel(detector.weights_path)} (md5 {detector.md5}) is not the detector the classifier's "
            f"crops were built with (md5 {classifier.detector_md5}); accuracy may be lower. Rebuild the crops and "
            f"retrain the classifier: {TRAIN_CLASSIFIER_CMD}",
            file=sys.stderr,
        )


def _warm_up(detector, classifier):
    """One predict per model on zero images, so the first camera frame is not slow."""
    import numpy as np

    detector.detect(np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), np.uint8))
    classifier.classify(np.zeros((classifier.imgsz, classifier.imgsz, 3), np.uint8))


def build_pipeline(device):
    """Load and check both models, warm them up, and return a SignPipeline.

    Exits 1 with the command to run when weights are missing or the classifier
    does not match this code. When the warm-up on MPS raises a RuntimeError,
    both models are rebuilt on the CPU and one WARNING is printed;
    pipeline.detector.device then reads 'cpu'.
    """
    from src.pipeline import SignPipeline
    from src.smoothing import ProbSmoother
    from src.tracking import HandSelector

    _require_weights()
    detector, classifier = _load_models(device)
    _check_detector_md5(detector, classifier)
    try:
        _warm_up(detector, classifier)
    except RuntimeError as exc:
        if device != "mps":
            raise
        print(f"WARNING: the MPS warm-up failed ({exc}); running both models on the CPU", file=sys.stderr)
        detector, classifier = _load_models("cpu")
        _warm_up(detector, classifier)
    return SignPipeline(detector, classifier, HandSelector(), ProbSmoother())


def _print_summary(perf):
    summary = perf.summary()
    total = summary["total_ms"]
    if total["n"] == 0:
        print("No frames were processed.")
        return
    print(f"Session latency over {total['n']} frames (ms):")
    for metric in ("det_ms", "cls_ms", "total_ms"):
        s = summary[metric]
        if s["n"]:
            print(f"  {metric:<8}  mean {s['mean']:7.1f}  p50 {s['p50']:7.1f}  p95 {s['p95']:7.1f}  n {s['n']}")
    print(f"  mean FPS {1000.0 / total['mean']:.1f}")


def main():
    """Run the live webcam app in the Qt window until it closes. Returns the exit code."""
    _require_weights()  # before torch and ultralytics load, so a missing model fails fast
    try:
        from src.ui.window import run
    except ImportError as exc:
        print(f"{PYSIDE6_MISSING}\n({exc})", file=sys.stderr)
        raise SystemExit(1) from None

    import cv2

    from src.perf import LatencyTracker
    from src.yolo_env import resolve_device

    device = resolve_device()
    print(f"Using device: {device}")
    pipeline = build_pipeline(device)
    device = pipeline.detector.device  # 'cpu' after an MPS warm-up fallback
    machine = CommitStateMachine(
        classes=config.CLASSES,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        debounce_frames=config.DEBOUNCE_FRAMES,
        cooldown_frames=config.COOLDOWN_FRAMES,
    )
    perf = LatencyTracker()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open the webcam (camera index {config.CAMERA_INDEX}).\n{CAMERA_HINT}",
              file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    try:
        return run(cap, pipeline, machine, perf, device)
    finally:
        cap.release()
        _print_summary(perf)


if __name__ == "__main__":
    sys.exit(main())
