"""Live ASL detector for SignVision: full-frame YOLO26 detection, one OpenCV window.

Every webcam frame is mirrored and scanned whole by the YOLO26 hand detector;
there is no fixed box to hold the hand in. HandSelector picks the signing hand
and keeps following it (green box; yellow while briefly lost; other hands
gray). The padded square crop of that hand goes through the shared
gray + CLAHE classifier_input to the YOLO26-cls letter classifier, a
sliding-window ProbSmoother steadies the prediction, and CommitStateMachine
turns it into text: a sign held for DEBOUNCE_FRAMES commits once, and a NOTHING
(or a dropped hand) must pass between any two commits. With no hand in view
the pipeline reports NOTHING.

The window shows the HUD (FPS, detector ms, classifier ms, device), the boxes,
the current label, and a bottom panel with the committed text. Keys: q quits,
c clears the text.

Start-up exits 1 with the command to run when a trained model is missing or
the classifier does not match this code, warns when the classifier was built
with a different detector, and falls back to the CPU (one WARNING) if the MPS
warm-up fails.

cv2, numpy, torch and ultralytics are imported only inside functions, so
`import src.app` (and the CommitStateMachine re-export) stays headless.

Usage:
    .venv/bin/python -m src.app
"""

import sys
import time
from pathlib import Path

from src import config
from src.state_machine import CommitStateMachine  # re-export

WINDOW = "SignVision"
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

# BGR colors
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
GRAY = (160, 160, 160)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HINT_GRAY = (180, 180, 180)
PANEL_HEIGHT = 70


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


def _corners(det):
    return (int(round(det.x1)), int(round(det.y1))), (int(round(det.x2)), int(round(det.y2)))


def _put(cv2, frame, text, org, scale, color, thickness):
    """Text with a dark outline, readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, org, font, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _panel_text(cv2, text, max_width, scale, thickness):
    """'Text: ...' trimmed from the left so the newest letters stay visible."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    shown, start = "Text: " + text, 0
    while start < len(text) and cv2.getTextSize(shown, font, scale, thickness)[0][0] > max_width:
        start += 1
        shown = "Text: ..." + text[start:]
    return shown


def draw_overlay(frame, result, text, hud):
    """Draw the boxes, label, HUD and text panel onto ``frame`` in place and return it (no imshow).

    ``result`` is a pipeline FrameResult, ``text`` the committed text and
    ``hud`` a dict with fps, det_ms, cls_ms (None when the classifier did not
    run) and device.
    """
    import cv2

    sel = result.selection
    if sel is not None:
        for other in sel.others:
            cv2.rectangle(frame, *_corners(other), GRAY, 1)
        color, thickness = (YELLOW, 2) if sel.held else (GREEN, 3)
        top_left, bottom_right = _corners(sel.box)  # the EMA box
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        _put(cv2, frame, f"{result.label} {result.conf:.0%}", (top_left[0], max(top_left[1] - 10, 20)), 0.9,
             color, 2)
    else:
        # OpenCV's Hershey fonts draw ASCII only, so the arrow is "->".
        _put(cv2, frame, "No hand -> NOTHING", (10, 64), 0.8, WHITE, 2)

    cls_ms = "-" if hud.get("cls_ms") is None else f"{hud['cls_ms']:.1f}"
    _put(cv2, frame, f"FPS {hud['fps']:.1f}  det {hud['det_ms']:.1f} ms  cls {cls_ms} ms  {hud['device']}",
         (10, 28), 0.7, WHITE, 2)

    frame_h, frame_w = frame.shape[:2]
    panel_top = frame_h - PANEL_HEIGHT
    cv2.rectangle(frame, (0, panel_top), (frame_w, frame_h), BLACK, -1)
    cv2.putText(frame, _panel_text(cv2, text, frame_w - 20, 0.9, 2), (10, frame_h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "q: quit   c: clear", (10, panel_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HINT_GRAY, 1,
                cv2.LINE_AA)
    return frame


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
    """Run the live webcam loop. Returns the exit code."""
    _require_weights()  # before torch and ultralytics load, so a missing model fails fast

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

    failures = 0
    try:
        while True:
            t = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures >= config.MAX_READ_FAILURES:
                    print(f"ERROR: the webcam returned no frame {failures} times in a row; exiting.", file=sys.stderr)
                    return 1
                continue
            failures = 0
            frame = cv2.flip(frame, 1)  # mirror, so moving right on screen matches moving right in reality

            r = pipeline.process(frame)
            text = machine.step(r.label, r.conf)
            cls_ms = r.cls_ms if r.cls_ran else None
            draw_overlay(frame, r, text, {"fps": perf.fps(), "det_ms": r.det_ms, "cls_ms": cls_ms, "device": device})
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            perf.record(det_ms=r.det_ms, cls_ms=cls_ms, total_ms=(time.perf_counter() - t) * 1000)

            if key == ord("q"):
                break
            if key == ord("c"):
                machine.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        _print_summary(perf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
