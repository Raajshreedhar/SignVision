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

HandSelector adds history for the live app: it keeps following the current
hand while it still overlaps the previous box and is not clearly beaten
(stickiness), bridges detector dropouts of up to LOST_GRACE_FRAMES frames by
holding the last box, and smooths the drawn box with an EMA inside a track.
The classifier crops from Selection.raw, the raw detection chosen this frame,
so live crops have the same geometry as the raw-box training crops.
"""

import math
from dataclasses import dataclass, replace

from src.config import BOX_EMA_ALPHA, DET_CONF, LOST_GRACE_FRAMES, MIN_BOX_AREA_FRAC, STICKY_IOU, STICKY_RATIO


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


@dataclass(frozen=True)
class Selection:
    """The signing hand chosen for one frame.

    box: the box to draw and to match the next frame against. Inside a track
        it is the EMA of the raw boxes; on a new track it is the raw box.
    track_id: the same while the same hand is followed; a new hand gets a new id.
    held: True on grace frames, where nothing valid was detected and the
        previous selection is carried over.
    others: the other valid detections of this frame (drawn dimmed); empty
        on held frames.
    raw: the raw detection chosen this frame (equal to box on a new track).
        The classifier crops from it. Held frames carry it over with box.
    """

    box: Detection
    track_id: int
    held: bool
    others: list[Detection]
    raw: Detection


class HandSelector:
    """Deterministic, sticky choice of the signing hand across frames (design section 6)."""

    def __init__(self, sticky_iou=STICKY_IOU, sticky_ratio=STICKY_RATIO, ema_alpha=BOX_EMA_ALPHA,
                 grace_frames=LOST_GRACE_FRAMES):
        self.sticky_iou = sticky_iou
        self.sticky_ratio = sticky_ratio
        self.ema_alpha = ema_alpha
        self.grace_frames = grace_frames
        self.next_id = 1
        self.reset()

    def reset(self):
        """Forget the current hand. Track ids keep counting up, so an id is never reused."""
        self.prev = None
        self.missed = 0

    def update(self, dets, frame_hw):
        """Choose this frame's signing hand from the raw detections; None when there is none."""
        valid = sorted(valid_detections(dets, frame_hw), key=lambda d: rank_key(d, frame_hw))
        if not valid:
            self.missed += 1
            if self.prev is not None and self.missed <= self.grace_frames:
                return replace(self.prev, held=True, others=[])
            self.prev = None
            return None

        chosen, same_track = 0, False
        if self.prev is not None:
            # The candidate overlaps the previous (EMA) box most; valid is in
            # rank_key order, so index() breaks IoU ties by rank_key.
            overlaps = [iou(d, self.prev.box) for d in valid]
            candidate = overlaps.index(max(overlaps))
            if (overlaps[candidate] >= self.sticky_iou
                    and score(valid[candidate], frame_hw) >= self.sticky_ratio * score(valid[0], frame_hw)):
                chosen, same_track = candidate, True

        raw = valid[chosen]
        if same_track:
            a, p = self.ema_alpha, self.prev.box
            box = Detection(
                a * raw.x1 + (1 - a) * p.x1, a * raw.y1 + (1 - a) * p.y1,
                a * raw.x2 + (1 - a) * p.x2, a * raw.y2 + (1 - a) * p.y2, raw.conf,
            )
            track_id = self.prev.track_id
        else:
            box, track_id = raw, self.next_id
            self.next_id += 1
        self.missed = 0
        others = [d for i, d in enumerate(valid) if i != chosen]
        self.prev = Selection(box=box, track_id=track_id, held=False, others=others, raw=raw)
        return self.prev
