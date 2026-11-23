"""Hand detections and signing-hand scoring (design section 6). Pure Python, no I/O.

A frame can show several hands. Every consumer ranks them the same way:
  - valid_detections drops weak (conf < DET_CONF) and tiny
    (area < MIN_BOX_AREA_FRAC of the frame) boxes;
  - score = conf * sqrt(area / frame_area): confidence dominates, and among
    confident boxes the larger, nearer hand (the one held up to the camera)
    wins;
  - rank_key = (-score, -conf, -area, x1, y1) is the single deterministic
    tie-break chain.
best_detection is that ranking with no history; cls_data uses it to pick the
crop box of each training image.
"""

import math
from dataclasses import dataclass

from src.config import DET_CONF, MIN_BOX_AREA_FRAC


@dataclass(frozen=True)
class Detection:
    """One detector box in frame pixels plus its confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _xyxy(box):
    return box.xyxy if hasattr(box, "xyxy") else tuple(float(v) for v in box)


def iou(a, b):
    """Intersection over union of two boxes (Detections or (x1, y1, x2, y2)); 0 for zero-area boxes."""
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter / (area_a + area_b - inter)


def _frame_area(frame_hw):
    height, width = frame_hw
    return float(height) * float(width)


def valid_detections(dets, frame_hw):
    """Detections with conf >= DET_CONF and area >= MIN_BOX_AREA_FRAC of the frame, in input order."""
    frame_area = _frame_area(frame_hw)
    return [d for d in dets if d.conf >= DET_CONF and d.area / frame_area >= MIN_BOX_AREA_FRAC]


def score(d, frame_hw):
    """conf * sqrt(area / frame_area)."""
    return d.conf * math.sqrt(d.area / _frame_area(frame_hw))


def rank_key(d, frame_hw):
    """Sort key, best first: higher score, then higher conf, then larger area, then left-most, top-most."""
    return (-score(d, frame_hw), -d.conf, -d.area, d.x1, d.y1)


def best_detection(dets, frame_hw):
    """The best-ranked valid detection, or None when no detection is valid."""
    valid = valid_detections(dets, frame_hw)
    if not valid:
        return None
    return min(valid, key=lambda d: rank_key(d, frame_hw))
