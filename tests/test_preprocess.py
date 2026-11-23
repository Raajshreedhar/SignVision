"""Unit tests for src.preprocess: the legacy preprocess_frame and the shared classifier input.

Run: .venv/bin/python tests/test_preprocess.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Allow running directly (python tests/test_preprocess.py) by putting the
# project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLAHE_CLIP, CLAHE_TILE, CLS_IMGSZ, IMG_SIZE, PROJECT_ROOT
from src.preprocess import classifier_input, crop_square, preprocess_frame, preprocess_hand_crop


def test_color_input():
    out = preprocess_frame(np.zeros((200, 200, 3), dtype=np.uint8))
    assert out.dtype == np.float32, f"dtype was {out.dtype}"
    assert out.shape == (1, IMG_SIZE, IMG_SIZE), f"shape was {out.shape}"
    assert out.min() >= 0.0, f"min was {out.min()}"
    assert out.max() <= 1.0, f"max was {out.max()}"


def test_gray_input():
    out = preprocess_frame(np.full((200, 200), 127, dtype=np.uint8))
    assert out.dtype == np.float32, f"dtype was {out.dtype}"
    assert out.shape == (1, IMG_SIZE, IMG_SIZE), f"shape was {out.shape}"
    assert out.min() >= 0.0, f"min was {out.min()}"
    assert out.max() <= 1.0, f"max was {out.max()}"


# --- shared classifier input (preprocess_hand_crop / classifier_input) ---------------


def random_bgr(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("shape", [(300, 300, 3), (300, 300), (60, 80, 3), (224, 224)])
def test_bgr_and_gray_input_give_uint8_224_three_channels(shape):
    img = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    out = preprocess_hand_crop(img)
    assert out.dtype == np.uint8
    assert out.shape == (CLS_IMGSZ, CLS_IMGSZ, 3)


def test_channels_are_identical():
    out = preprocess_hand_crop(random_bgr(250, 250))
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_is_deterministic():
    img = random_bgr(400, 400, seed=5)
    np.testing.assert_array_equal(preprocess_hand_crop(img), preprocess_hand_crop(img.copy()))


def test_pipeline_order_is_gray_resize_clahe():
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    for size_in, interp in ((448, cv2.INTER_AREA), (100, cv2.INTER_LINEAR)):
        img = random_bgr(size_in, size_in, seed=size_in)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        expected = clahe.apply(cv2.resize(gray, (CLS_IMGSZ, CLS_IMGSZ), interpolation=interp))
        np.testing.assert_array_equal(preprocess_hand_crop(img)[..., 0], expected)


def test_clahe_raises_contrast_of_a_low_contrast_image():
    rng = np.random.default_rng(3)
    low = (110 + rng.integers(0, 20, (300, 300))).astype(np.uint8)  # values 110..129
    plain = cv2.resize(low, (CLS_IMGSZ, CLS_IMGSZ), interpolation=cv2.INTER_AREA)
    out = preprocess_hand_crop(low)[..., 0]
    assert out.std() > 1.5 * plain.std()


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 10, 3), np.uint8),
        np.zeros((10, 0), np.uint8),
        np.zeros((32, 32, 3), np.float32),
        np.zeros((32, 32), np.float64),
        np.zeros((32, 32, 4), np.uint8),
        np.zeros((32, 32, 1), np.uint8),
        np.zeros(32, np.uint8),
        [[0, 1], [2, 3]],
    ],
)
def test_invalid_crops_raise(bad):
    with pytest.raises(ValueError):
        preprocess_hand_crop(bad)


def test_classifier_input_equals_crop_then_preprocess():
    frame = random_bgr(720, 1280, seed=9)
    box = (600.4, 300.2, 700.9, 420.0)
    np.testing.assert_array_equal(
        classifier_input(frame, box), preprocess_hand_crop(crop_square(frame, box))
    )
    np.testing.assert_array_equal(
        classifier_input(frame, box, pad=0.1, size=64), preprocess_hand_crop(crop_square(frame, box, 0.1), 64)
    )
    assert classifier_input(frame, box, size=64).shape == (64, 64, 3)


def test_clahe_and_gray_conversion_only_in_preprocess():
    """Single call site: no other src/ module may CLAHE or gray-convert classifier inputs."""
    src = Path(PROJECT_ROOT) / "src"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if path.name == "preprocess.py":
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("createCLAHE", "COLOR_BGR2GRAY"):
            if token in text:
                offenders.append(f"{path.relative_to(src)}: {token}")
    assert not offenders, offenders
    assert "createCLAHE" in (src / "preprocess.py").read_text(encoding="utf-8")


if __name__ == "__main__":
    test_color_input()
    test_gray_input()
    print("test_preprocess.py: all assertions passed")
