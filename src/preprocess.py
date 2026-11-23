"""Shared image preprocessing for SignVision.

INVARIANT: this is the only module that may color-convert, CLAHE or resize
classifier inputs. It owns both halves of the classifier input:
  - the crop geometry: square_crop_box / crop_square turn a hand box into a
    padded square that always lies inside the frame (shifted, never padded
    with fake pixels);
  - the pixel transform: preprocess_hand_crop does gray -> resize -> CLAHE ->
    3 identical channels.
classifier_input chains the two. cls_data (training crops), the pipeline
(live crops), the evaluator and the benchmark all call it, so the pixels the
classifier sees are identical at train and inference time. Bump
config.PREPROCESS_VERSION whenever preprocess_hand_crop changes.
"""

import math
import numbers

import cv2
import numpy as np

from src.config import CLAHE_CLIP, CLAHE_TILE, CLS_IMGSZ, CROP_PAD, MIN_CROP_PX

# One module-level CLAHE instance, reused for every call.
_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))


def square_crop_box(box_xyxy, frame_hw, pad=CROP_PAD, min_px=MIN_CROP_PX):
    """Padded square around a box, fully inside the frame.

    side = max(w, h) * (1 + 2*pad), rounded, raised to min_px and then capped
    at min(H, W) (the frame limit wins). The square is centered on the box and
    shifted back inside the frame when it sticks out. Returns integer
    (x1, y1, x2, y2) with x2 - x1 == y2 - y1 == side. Raises ValueError for a
    box that is not 4 finite numbers with x2 > x1 and y2 > y1, or a frame_hw
    that is not two positive ints.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    except (TypeError, ValueError):
        raise ValueError(f"box_xyxy must be 4 numbers (x1, y1, x2, y2), got {box_xyxy!r}") from None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"box_xyxy must be finite, got {box_xyxy!r}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box_xyxy must have x2 > x1 and y2 > y1, got {box_xyxy!r}")
    try:
        height, width = frame_hw
    except (TypeError, ValueError):
        raise ValueError(f"frame_hw must be (height, width), got {frame_hw!r}") from None
    for v in (height, width):
        if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"frame_hw must be two positive ints, got {frame_hw!r}")
    height, width = int(height), int(width)

    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    side = min(max(int(round(side)), min_px), min(height, width))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(int(round(cx - side / 2)), 0), width - side)
    top = min(max(int(round(cy - side / 2)), 0), height - side)
    return (left, top, left + side, top + side)


def crop_square(img, box_xyxy, pad=CROP_PAD):
    """The square_crop_box region of ``img`` as a contiguous copy (never a view)."""
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        raise ValueError(f"img must be an (h, w) or (h, w, c) array, got {getattr(img, 'shape', type(img))}")
    x1, y1, x2, y2 = square_crop_box(box_xyxy, img.shape[:2], pad)
    return img[y1:y2, x1:x2].copy()


def preprocess_hand_crop(crop, size=CLS_IMGSZ):
    """BGR uint8 (h, w, 3) or gray uint8 (h, w) crop -> uint8 (size, size, 3) classifier input.

    Gray conversion, resize to size x size (INTER_AREA when shrinking,
    INTER_LINEAR when enlarging), CLAHE, then the gray image merged into 3
    identical channels, so BGR/RGB order is irrelevant downstream. Resizing
    before CLAHE makes the CLAHE tiles cover the same fraction of the hand
    whatever the crop resolution. Raises ValueError for empty, non-uint8 or
    non-1/3-channel input.
    """
    if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8:
        raise ValueError(f"crop must be a uint8 array, got {getattr(crop, 'dtype', type(crop))}")
    if crop.size == 0:
        raise ValueError(f"crop is empty, shape {crop.shape}")
    if crop.ndim == 3 and crop.shape[2] == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    elif crop.ndim == 2:
        gray = crop
    else:
        raise ValueError(f"crop must be (h, w, 3) BGR or (h, w) gray, got shape {crop.shape}")
    height, width = gray.shape
    interp = cv2.INTER_AREA if height >= size and width >= size else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (size, size), interpolation=interp)
    equalized = _CLAHE.apply(resized)
    return cv2.merge([equalized, equalized, equalized])


def classifier_input(img, box_xyxy, pad=CROP_PAD, size=CLS_IMGSZ):
    """Frame + hand box -> the uint8 (size, size, 3) image the letter classifier sees."""
    return preprocess_hand_crop(crop_square(img, box_xyxy, pad), size)

