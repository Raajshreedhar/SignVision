"""Headless FPS and latency benchmark of the two-stage pipeline (design section 7.5).

    .venv/bin/python -m src.benchmark [--mode pretrained|finetuned] [--frames 200] [--warmup 10]
           [--device auto|mps|cpu|both] [--matrix] [--no-force-box] [--det-imgsz 512]

Every run feeds one synthetic 1280x720 frame (a gray gradient with the
grassknoted A_test.jpg pasted into the forced box when it is cached) through
SignPipeline.process with a fresh selector and smoother, and times it. No
camera is opened.

  --mode pretrained  the pretrained YOLO26 detector (models/pretrained/yolo26{n,s}.pt)
                     and classifier (yolo26{n,s}-cls.pt). Only the timing matters:
                     the classifier output is a uniform vector. --matrix runs all
                     four {n,s} det x {n,s} cls combinations (default: n + n).
  --mode finetuned   the trained models/hand_detector.pt + models/letter_classifier.pt.

The detector is wrapped in _ForcedBoxDetector by default: it still runs on
every frame, but the pipeline always gets the box (440, 160, 840, 560), so the
classifier also runs on every frame. --no-force-box (finetuned only) measures
true behavior instead. total_ms is process() time, without the camera read and
drawing; the app's HUD shows full-loop FPS.

Results are merged into models/reports/benchmark_<device>.json under the key
'<mode>:det-<d>+cls-<c>@<det imgsz>' (plus ':no-force-box'), so earlier
combinations are kept.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import config
from src.cli import DEVICE_CHOICES, MODEL_CHOICES, imgsz_arg, int_range
from src.config import CLS_IMGSZ, CROP_PAD, FPS_TARGET, NUM_CLASSES
from src.tracking import Detection

FORCED_BOX = (440, 160, 840, 560)  # (x1, y1, x2, y2) in the 1280x720 frame
FRAME_W, FRAME_H = 1280, 720
TRAIN_DETECTOR_CMD = ".venv/bin/python -m src.hand_data && .venv/bin/python -m src.train_detector"
TRAIN_CLASSIFIER_CMD = ".venv/bin/python -m src.cls_data && .venv/bin/python -m src.train_classifier"


class _RawCls:
    """Benchmark-only classifier: the real predict() cost, no class or meta validation, uniform output."""

    def __init__(self, weights, device, imgsz=CLS_IMGSZ):
        from src.yolo_env import load_yolo  # torch + ultralytics, only when a model is built

        self.weights_path = Path(weights)
        self.model = load_yolo(self.weights_path)
        self.device = device
        self.imgsz = imgsz
        self.crop_pad = CROP_PAD

    def classify(self, img):
        result = self.model.predict(img, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        result.probs.data.cpu()  # wait for the device, as LetterClassifier.classify does
        return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)


class _ForcedBoxDetector:
    """Runs the inner detector (so its cost is timed), discards its boxes and returns FORCED_BOX."""

    def __init__(self, inner):
        self.inner = inner
        self.box = Detection(*FORCED_BOX, 0.99)

    def detect(self, frame):
        self.inner.detect(frame)
        return [self.box]


def a_test_image(sources_file=None):
    """Path of the grassknoted A_test.jpg recorded in data/sources.json, or None when it is not cached."""
    from src.download_data import REGISTRY, read_sources

    root = read_sources(sources_file).get("grassknoted")
    if root is None:
        return None
    path = Path(root) / Path(REGISTRY["grassknoted"].test_glob).parent / "A_test.jpg"
    return path if path.is_file() else None


def make_frame(sources_file=None):
    """(frame, description): a 1280x720 BGR gray gradient, with A_test.jpg resized into FORCED_BOX if cached."""
    import cv2

    row = np.linspace(30, 220, FRAME_W).astype(np.uint8)
    frame = np.repeat(np.tile(row, (FRAME_H, 1))[:, :, None], 3, axis=2)
    path = a_test_image(sources_file)
    image = cv2.imread(str(path)) if path is not None else None
    if image is None:
        return frame, "gray gradient (grassknoted A_test.jpg not cached)"
    x1, y1, x2, y2 = FORCED_BOX
    frame[y1:y2, x1:x2] = cv2.resize(image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return frame, f"gray gradient + {path.name} in {FORCED_BOX}"


def run_benchmark(detector, classifier, frame, frames, warmup):
    """Time SignPipeline.process over ``frames`` frames after ``warmup`` untimed ones.

    Returns {det_ms, cls_ms, total_ms: {mean, p50, p95, n}, fps, frames, warmup}.
    cls_ms covers only the frames where the classifier ran; fps = 1000 / mean total_ms.
    """
    from src.perf import LatencyTracker
    from src.pipeline import SignPipeline
    from src.smoothing import ProbSmoother
    from src.tracking import HandSelector

    pipeline = SignPipeline(detector, classifier, HandSelector(), ProbSmoother())
    for _ in range(warmup):
        pipeline.process(frame)
    perf = LatencyTracker()
    for _ in range(frames):
        t = time.perf_counter()
        r = pipeline.process(frame)
        total_ms = (time.perf_counter() - t) * 1000
        perf.record(det_ms=r.det_ms, cls_ms=r.cls_ms if r.cls_ran else None, total_ms=total_ms)
    summary = perf.summary()
    result = {
        metric: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}
        for metric, stats in summary.items()
    }
    result["fps"] = round(1000.0 / summary["total_ms"]["mean"], 2)
    result["frames"] = frames
    result["warmup"] = warmup
    return result


def combo_key(mode, det_size, cls_size, det_imgsz, force_box=True):
    key = f"{mode}:det-{det_size}+cls-{cls_size}@{det_imgsz}"
    return key if force_box else f"{key}:no-force-box"


def write_report(device, combo, result, reports_dir=None):
    """Merge {combo: result} into <reports_dir>/benchmark_<device>.json atomically; return the merged dict."""
    reports_dir = Path(reports_dir) if reports_dir is not None else config.REPORTS_DIR
    path = reports_dir / f"benchmark_{device}.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: {path} is unreadable ({exc}); rewriting it", file=sys.stderr)
            data = {}
    data[combo] = result

    reports_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=reports_dir, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return data


def _fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _model_size(meta_path):
    try:
        return json.loads(Path(meta_path).read_text()).get("model_size", "unknown")
    except (OSError, ValueError):
        return "unknown"


def _finetuned_backends(device, det_imgsz):
    from src.class_map import ClassOrderError
    from src.pipeline import HandDetector, LetterClassifier, MetaError

    detector = HandDetector(config.DET_WEIGHTS, device, imgsz=det_imgsz)
    try:
        classifier = LetterClassifier(config.CLS_WEIGHTS, config.CLS_META, device)
    except (ClassOrderError, MetaError) as exc:
        _fail(f"{exc}\nretrain the classifier: .venv/bin/python -m src.train_classifier")
    return detector, classifier


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark",
        description="Headless FPS / latency benchmark of the detector + classifier pipeline (no camera).",
    )
    parser.add_argument(
        "--mode", choices=("pretrained", "finetuned"), default="pretrained",
        help="pretrained YOLO26 weights (timing only) or the trained models (default pretrained)",
    )
    parser.add_argument("--frames", type=int_range(1), default=200, help="timed frames per run (default 200)")
    parser.add_argument("--warmup", type=int_range(0), default=10, help="untimed warm-up frames (default 10)")
    parser.add_argument(
        "--device", choices=(*DEVICE_CHOICES, "both"), default="auto",
        help="auto = mps when available; both = mps, then cpu",
    )
    parser.add_argument(
        "--matrix", action="store_true", help="pretrained only: all four {n,s} detector x {n,s} classifier combos"
    )
    parser.add_argument(
        "--no-force-box", action="store_true",
        help="finetuned only: use the real detector boxes instead of the fixed box",
    )
    parser.add_argument(
        "--det-imgsz", type=imgsz_arg, default=None,
        help=f"detector image size, a multiple of 32 (default config.DET_IMGSZ = {config.DET_IMGSZ})",
    )
    return parser


def _ms(stats):
    if not stats["n"]:
        return "-"
    return f"{stats['mean']:.1f}/{stats['p50']:.1f}/{stats['p95']:.1f}"


def print_table(rows):
    print(f"\n{'device':<6}  {'combo':<44}  {'fps':>6}  {'det ms mean/p50/p95':>20}  "
          f"{'cls ms mean/p50/p95':>20}  {'total ms mean/p50/p95':>22}  >={FPS_TARGET:g} fps")
    for device, combo, r in rows:
        met = "yes" if r["fps"] >= FPS_TARGET else "no"
        print(f"{device:<6}  {combo:<44}  {r['fps']:>6.1f}  {_ms(r['det_ms']):>20}  {_ms(r['cls_ms']):>20}  "
              f"{_ms(r['total_ms']):>22}  {met}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.matrix and args.mode != "pretrained":
        parser.error("--matrix needs --mode pretrained (only one fine-tuned pair exists at a time)")
    if args.no_force_box and args.mode != "finetuned":
        parser.error("--no-force-box needs --mode finetuned (the pretrained models do not detect hands)")
    det_imgsz = args.det_imgsz if args.det_imgsz is not None else config.DET_IMGSZ

    if args.mode == "finetuned":
        missing = [
            f"{path} is missing; train it: {command}"
            for path, command in (
                (config.DET_WEIGHTS, TRAIN_DETECTOR_CMD),
                (config.CLS_WEIGHTS, TRAIN_CLASSIFIER_CMD),
                (config.CLS_META, TRAIN_CLASSIFIER_CMD),
            )
            if not Path(path).is_file()
        ]
        if missing:
            _fail("\n".join(missing))
        sizes = [(_model_size(config.DET_META), _model_size(config.CLS_META))]
    else:
        sizes = [(d, c) for d in MODEL_CHOICES for c in MODEL_CHOICES] if args.matrix else [("n", "n")]

    from src import yolo_env  # torch + ultralytics

    frame, frame_desc = make_frame()
    print(f"frame: {frame_desc}; {args.warmup} warm-up + {args.frames} timed frames per run")
    devices = ["mps", "cpu"] if args.device == "both" else [args.device]
    rows = []
    for pref in devices:
        device = yolo_env.resolve_device(pref)
        for det_size, cls_size in sizes:
            if args.mode == "finetuned":
                detector, classifier = _finetuned_backends(device, det_imgsz)
                det_weights, cls_weights = config.DET_WEIGHTS, config.CLS_WEIGHTS
            else:
                from src.pipeline import HandDetector

                det_weights = yolo_env.pretrained_path(f"yolo26{det_size}.pt")
                cls_weights = yolo_env.pretrained_path(f"yolo26{cls_size}-cls.pt")
                detector = HandDetector(det_weights, device, imgsz=det_imgsz)
                classifier = _RawCls(cls_weights, device)
            force_box = not args.no_force_box
            if force_box:
                detector = _ForcedBoxDetector(detector)
            combo = combo_key(args.mode, det_size, cls_size, det_imgsz, force_box)
            print(f"[{device}] {combo} ...", flush=True)
            result = run_benchmark(detector, classifier, frame, args.frames, args.warmup)
            result.update(
                device=device, det_imgsz=det_imgsz, force_box=force_box, det_weights=str(det_weights),
                cls_weights=str(cls_weights), frame=frame_desc, fps_target=FPS_TARGET,
                measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            write_report(device, combo, result)
            rows.append((device, combo, result))
    print_table(rows)
    print(f"\nreports: {', '.join(sorted({str(config.REPORTS_DIR / f'benchmark_{d}.json') for d, _, _ in rows}))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
