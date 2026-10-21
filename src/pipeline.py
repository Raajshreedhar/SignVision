"""Model wrappers for the two-stage pipeline (design section 7.1).

HandDetector wraps the stage-1 YOLO26 hand detector and returns
tracking.Detection boxes. LetterClassifier wraps the stage-2 YOLO26-cls letter
classifier and refuses weights or metadata that do not match this code: the
model's class names must decode through class_map.decode_class_names, and the
meta's classes and preprocess_version must equal config.CLASSES and
config.PREPROCESS_VERSION.

Ultralytics is imported lazily inside the constructors (through
src.yolo_env), so importing this module stays light and pipeline tests can
use fakes. Missing weights raise FileNotFoundError from yolo_env.load_yolo.
"""

import json
from pathlib import Path

import numpy as np

from src.class_map import CLASS_DIRS, ClassOrderError, decode_class_names
from src.config import CLASSES, DET_CONF, DET_IMGSZ, DET_MAX_DET, PREPROCESS_VERSION
from src.tracking import Detection

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
