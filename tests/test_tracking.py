"""src.tracking helpers: Detection, iou, filtering, scoring and the tie-break chain."""

import dataclasses
import itertools
import math

import pytest

from src.config import DET_CONF, MIN_BOX_AREA_FRAC
from src.tracking import Detection, best_detection, iou, rank_key, score, valid_detections

HW = (720, 1280)
FRAME_AREA = 720 * 1280


def test_detection_properties():
    d = Detection(10.0, 20.0, 110.0, 70.0, 0.9)
    assert d.xyxy == (10.0, 20.0, 110.0, 70.0)
    assert d.area == 100.0 * 50.0
    assert Detection(10, 10, 5, 20, 0.9).area == 0.0  # inverted box has no area
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.conf = 0.1


def test_iou_math():
    a = Detection(0, 0, 10, 10, 0.9)
    assert iou(a, a) == 1.0
    assert iou(a, Detection(20, 20, 30, 30, 0.9)) == 0.0
    assert iou(a, Detection(10, 0, 20, 10, 0.9)) == 0.0  # touching edges
    assert iou(a, Detection(5, 0, 15, 10, 0.9)) == pytest.approx(50 / 150)
    assert iou(a, Detection(2, 2, 4, 4, 0.9)) == pytest.approx(4 / 100)  # contained
    assert iou(a, (5, 0, 15, 10)) == pytest.approx(1 / 3)  # plain tuples work too
    assert iou(a, Detection(3, 3, 3, 8, 0.9)) == 0.0  # zero-area box
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_score_formula():
    d = Detection(0, 0, 320, 180, 0.8)  # 1/16 of the frame
    assert score(d, HW) == pytest.approx(0.8 * math.sqrt(1 / 16))
    assert rank_key(d, HW) == (-score(d, HW), -0.8, -d.area, 0, 0)


def test_large_less_confident_beats_confident_small_but_not_a_weak_one():
    small_confident = Detection(0, 0, 100, 100, 0.90)  # 0.90 * sqrt(10000 / 921600) = 0.094
    large = Detection(500, 200, 800, 500, 0.50)  # 0.50 * sqrt(90000 / 921600) = 0.156
    assert best_detection([small_confident, large], HW) == large
    near_confident = Detection(0, 0, 200, 200, 0.95)  # 0.198
    slightly_larger = Detection(500, 200, 720, 420, 0.60)  # 0.137
    assert best_detection([slightly_larger, near_confident], HW) == near_confident


def test_filter_thresholds():
    side = math.ceil(math.sqrt(MIN_BOX_AREA_FRAC * FRAME_AREA))  # just above the minimum area
    ok = Detection(0, 0, side, side, DET_CONF)
    weak = Detection(0, 0, 300, 300, DET_CONF - 1e-3)
    tiny = Detection(0, 0, side - 2, side - 2, 0.99)
    assert valid_detections([weak, ok, tiny], HW) == [ok]
    assert best_detection([weak, tiny], HW) is None
    assert best_detection([], HW) is None


def test_tie_break_chain():
    hw = (100, 100)
    # equal score: higher confidence wins over the larger box
    high_conf = Detection(50, 50, 60, 60, 0.8)  # 0.8 * sqrt(0.01) = 0.08
    big_low_conf = Detection(0, 0, 20, 20, 0.4)  # 0.4 * sqrt(0.04) = 0.08
    assert score(high_conf, hw) == score(big_low_conf, hw)
    assert best_detection([big_low_conf, high_conf], hw) == high_conf
    # equal score, conf and area: left-most, then top-most
    left = Detection(10, 40, 30, 60, 0.5)
    right = Detection(40, 10, 60, 30, 0.5)
    assert best_detection([right, left], hw) == left
    top = Detection(10, 5, 30, 25, 0.5)
    assert best_detection([left, top], hw) == top
    # the winner never depends on input order (left, right and top score 0.1 > 0.08; top wins on y1)
    dets = [high_conf, big_low_conf, left, right, top]
    winners = {best_detection(list(p), hw) for p in itertools.permutations(dets)}
    assert winners == {top}
