"""Shared image preprocessing for SignVision.

INVARIANT (a): preprocess_frame is the ONE place preprocessing is defined.
Both training (via src.dataset) and live inference (src.app) MUST call this
exact function so the pixels the model sees are identical at train and
inference time. Do NOT reimplement grayscale/CLAHE/resize/normalize anywhere
else.
"""

import cv2
import numpy as np

from src.config import IMG_SIZE

# Module-level CLAHE instance (reused across every call for consistency and
# to avoid reallocation per frame).
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_frame(img):
    """Convert a webcam frame or dataset image into model input.

    Steps:
      1. BGR -> grayscale if the image has 3 channels (pass through if already 2D).
      2. Apply CLAHE on the 8-bit gray image for lighting-robust contrast.
      3. Resize to (IMG_SIZE, IMG_SIZE) using INTER_AREA.
      4. Normalize to float32 in [0, 1].

    Returns a (1, IMG_SIZE, IMG_SIZE) float32 array (single channel first).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # CLAHE expects an 8-bit single-channel image.
    gray = _CLAHE.apply(gray)

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    arr = resized.astype(np.float32) / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE)
