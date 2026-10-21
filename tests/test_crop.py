"""src.preprocess crop geometry: square_crop_box and crop_square."""

import random

import numpy as np
import pytest

from src.config import CROP_PAD, MIN_CROP_PX
from src.preprocess import crop_square, square_crop_box

H, W = 720, 1280


def side_of(box):
    x1, y1, x2, y2 = box
    assert x2 - x1 == y2 - y1, f"not square: {box}"
    return x2 - x1


def assert_inside(box, hw=(H, W)):
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= hw[1] and 0 <= y1 < y2 <= hw[0], f"{box} not inside {hw}"
    assert all(isinstance(v, int) for v in box)


def test_centered_box():
    box = square_crop_box((600, 300, 700, 420), (H, W))
    assert box == (566, 276, 734, 444)  # side = 120 * 1.4 = 168, centered on (650, 360)
    assert side_of(box) == round(120 * (1 + 2 * CROP_PAD))


@pytest.mark.parametrize(
    "box, expect_x1, expect_y1",
    [
        ((0, 300, 50, 360), 0, None),  # left edge
        ((1230, 300, 1280, 360), W - 84, None),  # right edge
        ((600, 0, 650, 60), None, 0),  # top edge
        ((600, 660, 650, 720), None, H - 84),  # bottom edge
        ((0, 0, 50, 60), 0, 0),  # top-left corner
        ((1230, 0, 1280, 60), W - 84, 0),  # top-right corner
        ((0, 660, 50, 720), 0, H - 84),  # bottom-left corner
        ((1230, 660, 1280, 720), W - 84, H - 84),  # bottom-right corner
    ],
)
def test_box_at_edges_and_corners_is_shifted_inside_with_side_preserved(box, expect_x1, expect_y1):
    out = square_crop_box(box, (H, W))
    assert side_of(out) == 84  # max(50, 60) * 1.4
    assert_inside(out)
    if expect_x1 is not None:
        assert out[0] == expect_x1
    if expect_y1 is not None:
        assert out[1] == expect_y1


def test_box_larger_than_frame_uses_the_short_side():
    out = square_crop_box((0, 0, W, H), (H, W))
    assert side_of(out) == H
    assert out == (280, 0, 1000, 720)
    assert_inside(out)


def test_tiny_box_is_raised_to_min_crop_px():
    out = square_crop_box((100, 100, 105, 104), (H, W))
    assert side_of(out) == MIN_CROP_PX
    assert_inside(out)


def test_frame_smaller_than_min_crop_px_wins():
    out = square_crop_box((5, 5, 10, 10), (20, 25))
    assert side_of(out) == 20
    assert out == (0, 0, 20, 20)
    assert_inside(out, (20, 25))


def test_pad_zero_keeps_a_square_box():
    assert square_crop_box((10, 20, 110, 120), (H, W), pad=0.0) == (10, 20, 110, 120)


def test_seeded_random_property_loop():
    rng = random.Random(1234)
    for _ in range(2000):
        hw = (rng.randint(1, 900), rng.randint(1, 1600))
        x1, y1 = rng.uniform(-50, hw[1]), rng.uniform(-50, hw[0])
        x2, y2 = x1 + rng.uniform(0.01, hw[1] * 1.2), y1 + rng.uniform(0.01, hw[0] * 1.2)
        pad = rng.choice([0.0, 0.1, CROP_PAD, 0.5])
        out = square_crop_box((x1, y1, x2, y2), hw, pad=pad)
        side = side_of(out)
        assert_inside(out, hw)
        expected = min(max(round(max(x2 - x1, y2 - y1) * (1 + 2 * pad)), MIN_CROP_PX), min(hw))
        assert side == expected


@pytest.mark.parametrize(
    "box",
    [
        (float("nan"), 0, 10, 10),
        (0, 0, float("inf"), 10),
        (10, 0, 5, 10),  # x2 < x1
        (0, 10, 10, 10),  # y2 == y1
        (0, 0, 10),
        ("a", 0, 10, 10),
        None,
    ],
)
def test_invalid_box_raises(box):
    with pytest.raises(ValueError):
        square_crop_box(box, (H, W))


@pytest.mark.parametrize("hw", [(0, 100), (100, -1), (10.5, 100), (100,), None, (True, 100)])
def test_invalid_frame_hw_raises(hw):
    with pytest.raises(ValueError):
        square_crop_box((0, 0, 10, 10), hw)


def test_numpy_scalars_are_accepted():
    box = tuple(np.float32(v) for v in (600, 300, 700, 420))
    assert square_crop_box(box, tuple(np.int64(v) for v in (H, W))) == (566, 276, 734, 444)


def test_crop_square_returns_a_contiguous_copy():
    img = np.arange(H * W * 3, dtype=np.uint32).reshape(H, W, 3).astype(np.uint8)
    crop = crop_square(img, (600, 300, 700, 420))
    assert crop.shape == (168, 168, 3)
    assert crop.flags["C_CONTIGUOUS"] and crop.flags["OWNDATA"]
    np.testing.assert_array_equal(crop, img[276:444, 566:734])
    crop[:] = 0
    assert img[276:444, 566:734].any()  # the frame is untouched


def test_crop_square_gray_image():
    gray = np.zeros((100, 200), np.uint8)
    assert crop_square(gray, (0, 0, 200, 100)).shape == (100, 100)
    with pytest.raises(ValueError):
        crop_square(np.zeros(5, np.uint8), (0, 0, 2, 2))
