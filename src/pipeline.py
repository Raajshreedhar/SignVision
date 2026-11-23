"""The two-stage pipeline: model wrappers (design section 7.1) and SignPipeline (section 7.2).

HandDetector wraps the stage-1 YOLO26 hand detector and returns
tracking.Detection boxes. LetterClassifier wraps the stage-2 YOLO26-cls letter
classifier and refuses weights or metadata that do not match this code: the
model's class names must decode through class_map.decode_class_names, and the
meta's classes and preprocess_version must equal config.CLASSES and
config.PREPROCESS_VERSION.

SignPipeline runs one frame through detector -> HandSelector -> square crop of
the raw chosen box -> classifier -> ProbSmoother and returns a FrameResult
with the smoothed label and the per-stage timings. With no hand it skips the
classifier and feeds a NOTHING one-hot to the smoother. It resets the smoother
whenever a new track starts, so two hands' predictions never blend.

Ultralytics (through src.yolo_env) and cv2 (through src.preprocess) are
imported lazily inside the constructors, so importing this module stays light
and pipeline tests can use fakes. Missing weights raise FileNotFoundError from
yolo_env.load_yolo.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.class_map import CLASS_DIRS, ClassOrderError, decode_class_names
from src.config import CLASSES, DET_CONF, DET_IMGSZ, DET_MAX_DET, NOTHING_IDX, NUM_CLASSES, PREPROCESS_VERSION
from src.tracking import Detection, Selection

REQUIRED_META_KEYS = ("classes", "class_dirs", "preprocess_version", "crop_pad", "imgsz", "detector_md5")


class MetaError(Exception):
    """letter_classifier.json is unreadable, incomplete, or built for a different preprocessing."""


def _detections(result):
    """Ultralytics detect Result -> list[Detection] (YOLO26 is NMS-free, so boxes are final)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    return [
        Detection(float(x1), float(y1), float(x2), float(y2), float(c)) for (x1, y1, x2, y2), c in zip(xyxy, conf)
    ]


class HandDetector:
    """The YOLO26 hand detector: frames or image files in, hand boxes out."""

    def __init__(self, weights, device, imgsz=DET_IMGSZ, conf=DET_CONF, max_det=DET_MAX_DET):
        from src.yolo_env import load_yolo, md5_file  # torch + ultralytics, only when a model is built

        self.weights_path = Path(weights)
        self.model = load_yolo(self.weights_path)
        self.md5 = md5_file(self.weights_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det

    def _predict(self, source, **kwargs):
        return self.model.predict(
            source, imgsz=self.imgsz, conf=self.conf, max_det=self.max_det, device=self.device, verbose=False,
            **kwargs,
        )

    def detect(self, frame):
        """Hand boxes in one BGR frame."""
        return _detections(self._predict(frame)[0])

    def detect_batch(self, paths, batch=32):
        """Yield (path, detections, (h, w)) for image files, in input order.

        The paths are split into chunks of ``batch``; each chunk is decoded with
        cv2.imread and passed to one predict(..., batch=len(chunk), stream=True)
        call as BGR arrays, the same input type as live frames. An unreadable
        file yields nothing; the caller counts it as a read failure. (A list of
        path strings would be decoded through PIL instead, whose Ultralytics
        patch runs `pip install pi-heif` on the first unreadable file.)
        """
        import cv2

        paths = [Path(p) for p in paths]
        for start in range(0, len(paths), batch):
            chunk = []
            for path in paths[start:start + batch]:
                image = cv2.imread(str(path))
                if image is not None:
                    chunk.append((path, image))
            if not chunk:
                continue
            results = self._predict([image for _, image in chunk], batch=len(chunk), stream=True)
            for (path, _), result in zip(chunk, results, strict=True):
                yield path, _detections(result), tuple(int(v) for v in result.orig_shape)


def _read_meta(meta_path):
    path = Path(meta_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise MetaError(f"{path} is not readable JSON ({exc}); retrain the classifier") from exc
    if not isinstance(meta, dict):
        raise MetaError(f"{path} must hold a JSON object; retrain the classifier")
    return meta


class LetterClassifier:
    """The YOLO26-cls letter classifier: a preprocessed crop in, 29 class probabilities out."""

    def __init__(self, weights, meta_path, device):
        from src.yolo_env import load_yolo  # torch + ultralytics, only when a model is built

        self.weights_path = Path(weights)
        self.meta_path = Path(meta_path)
        self.model = load_yolo(self.weights_path)
        meta = _read_meta(self.meta_path)

        missing = [key for key in REQUIRED_META_KEYS if key not in meta]
        if missing:
            raise MetaError(f"{self.meta_path} lacks {', '.join(missing)}; retrain the classifier")
        if decode_class_names(self.model.names) != CLASSES:  # raises ClassOrderError with a diff
            raise ClassOrderError(f"{self.weights_path} class names do not decode to config.CLASSES")
        if meta["classes"] != CLASSES:
            raise ClassOrderError(f"{self.meta_path} classes {meta['classes']} != config.CLASSES {CLASSES}")
        if meta["class_dirs"] != CLASS_DIRS:
            raise ClassOrderError(f"{self.meta_path} class_dirs {meta['class_dirs']} != {CLASS_DIRS}")
        if meta["preprocess_version"] != PREPROCESS_VERSION:
            raise MetaError(
                f"{self.meta_path} preprocess_version {meta['preprocess_version']!r} != {PREPROCESS_VERSION!r}; "
                "rebuild src.cls_data and retrain the classifier"
            )
        try:
            self.crop_pad = float(meta["crop_pad"])
            self.imgsz = int(meta["imgsz"])
        except (TypeError, ValueError) as exc:
            raise MetaError(f"{self.meta_path} has an invalid crop_pad or imgsz ({exc})") from exc
        self.detector_md5 = meta["detector_md5"]
        self.meta = meta
        self.device = device

    def classify(self, img):
        """(29,) float32 class probabilities in config.CLASSES order for one preprocessed crop."""
        result = self.model.predict(img, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return result.probs.data.cpu().numpy().astype(np.float32)


@dataclass
class FrameResult:
    """SignPipeline output for one frame."""

    selection: Selection | None
    probs_raw: np.ndarray  # (29,) classifier output, or the NOTHING one-hot when no hand was selected
    probs_smoothed: np.ndarray  # (29,) window mean from the ProbSmoother
    label: str  # argmax of probs_smoothed
    conf: float  # probs_smoothed[label]
    det_ms: float
    cls_ms: float  # 0.0 when the classifier did not run
    cls_ran: bool


class SignPipeline:
    """detector -> selector -> raw-box crop -> classifier -> smoother for one frame (design section 7.2).

    The detector (detect(frame) -> list[Detection]), the classifier
    (crop_pad, imgsz, classify(img) -> (29,) probabilities) and the clock are
    duck-typed, so tests can pass fakes.
    """

    def __init__(self, detector, classifier, selector, smoother, clock=time.perf_counter):
        from src.preprocess import classifier_input  # cv2: imported only when a pipeline is built

        self.detector = detector
        self.classifier = classifier
        self.selector = selector
        self.smoother = smoother
        self.clock = clock
        self._classifier_input = classifier_input
        self._last_track = None

    def process(self, frame):
        """Run one BGR frame through both stages and return a FrameResult."""
        t0 = self.clock()
        dets = self.detector.detect(frame)
        det_ms = (self.clock() - t0) * 1000
        sel = self.selector.update(dets, frame.shape[:2])
        cur = sel.track_id if sel is not None else None
        if cur is not None and cur != self._last_track:  # entering a new hand
            self.smoother.reset()
        self._last_track = cur  # None on no-hand frames
        if sel is not None:
            t1 = self.clock()
            # Crop from the raw chosen box (N6), the same geometry as the training crops.
            x = self._classifier_input(frame, sel.raw.xyxy, self.classifier.crop_pad, self.classifier.imgsz)
            probs_raw = self.classifier.classify(x)
            cls_ms = (self.clock() - t1) * 1000
            cls_ran = True
        else:
            probs_raw = np.eye(NUM_CLASSES, dtype=np.float32)[NOTHING_IDX]
            cls_ms = 0.0
            cls_ran = False
        probs_smoothed = self.smoother.push(probs_raw)
        idx = int(np.argmax(probs_smoothed))
        return FrameResult(
            selection=sel, probs_raw=probs_raw, probs_smoothed=probs_smoothed, label=CLASSES[idx],
            conf=float(probs_smoothed[idx]), det_ms=det_ms, cls_ms=cls_ms, cls_ran=cls_ran,
        )
