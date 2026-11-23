"""src.tracking: Detection, iou, filtering, scoring, the tie-break chain, and the sticky HandSelector."""

import dataclasses
import itertools
import math

import pytest

from src.config import BOX_EMA_ALPHA, DET_CONF, LOST_GRACE_FRAMES, MIN_BOX_AREA_FRAC, STICKY_IOU, STICKY_RATIO
from src.tracking import (
    Detection,
    HandSelector,
    Selection,
    best_detection,
    iou,
    rank_key,
    score,
    valid_detections,
)

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


# --- HandSelector (design section 6, N6) ------------------------------------------------


def ema(new, prev, alpha=BOX_EMA_ALPHA):
    """Expected EMA box coordinates of ``new`` blended into ``prev``."""
    return tuple(alpha * n + (1 - alpha) * p for n, p in zip(new.xyxy, prev.xyxy))


def test_selector_single_box_starts_track_one():
    d = Detection(500, 200, 700, 400, 0.9)
    assert HandSelector().update([d], HW) == Selection(box=d, track_id=1, held=False, others=[], raw=d)


def test_selector_ignores_weak_and_tiny_boxes():
    sel = HandSelector()
    assert sel.update([], HW) is None
    weak = Detection(500, 200, 700, 400, DET_CONF - 0.01)
    tiny = Detection(0, 0, 10, 10, 0.99)
    assert sel.update([weak, tiny], HW) is None


def test_selector_picks_the_highest_score_and_lists_others_in_rank_order():
    small_confident = Detection(0, 0, 100, 100, 0.90)  # score 0.094
    large = Detection(500, 200, 800, 500, 0.50)  # 0.156
    medium = Detection(900, 100, 1100, 300, 0.70)  # 0.146
    s = HandSelector().update([small_confident, medium, large], HW)
    assert s.box == s.raw == large
    assert s.others == [medium, small_confident]


def test_selection_does_not_depend_on_detection_order():
    dets = [Detection(100, 100, 300, 300, 0.8), Detection(700, 100, 900, 300, 0.8), Detection(400, 400, 520, 520, 0.95)]
    outcomes = {(s.raw, tuple(s.others)) for s in (HandSelector().update(list(p), HW)
                                                   for p in itertools.permutations(dets))}
    assert len(outcomes) == 1


def test_stickiness_keeps_the_current_hand_when_a_slightly_better_one_appears():
    sel = HandSelector()
    first = sel.update([Detection(100, 100, 300, 300, 0.80)], HW)
    moved = Detection(110, 100, 310, 300, 0.80)
    better = Detection(800, 200, 1000, 400, 0.95)
    assert STICKY_RATIO * score(better, HW) <= score(moved, HW) < score(better, HW)
    s = sel.update([better, moved], HW)
    assert s.track_id == first.track_id == 1
    assert s.raw == moved
    assert s.others == [better]


def test_switches_when_the_other_hand_is_clearly_better():
    sel = HandSelector()
    sel.update([Detection(100, 100, 300, 300, 0.80)], HW)
    fading = Detection(110, 100, 310, 300, 0.40)
    clearly_better = Detection(700, 100, 1100, 500, 0.95)
    assert score(fading, HW) < STICKY_RATIO * score(clearly_better, HW)
    s = sel.update([fading, clearly_better], HW)
    assert s.track_id == 2
    assert s.box == s.raw == clearly_better
    assert s.others == [fading]


def test_a_box_that_no_longer_overlaps_starts_a_new_track():
    sel = HandSelector()
    sel.update([Detection(100, 100, 300, 300, 0.9)], HW)
    far = Detection(800, 300, 1000, 500, 0.9)
    s = sel.update([far], HW)
    assert s.track_id == 2 and s.box == far


def test_stickiness_iou_is_measured_against_the_ema_box():
    sel = HandSelector()
    a0 = Detection(300, 100, 500, 300, 0.9)
    a1 = Detection(340, 100, 540, 300, 0.9)
    sel.update([a0], HW)
    second = sel.update([a1], HW)
    assert second.box.xyxy == pytest.approx(ema(a1, a0))  # (324, 100, 524, 300): the EMA lags the raw box
    probe = Detection(224, 100, 424, 300, 0.9)
    assert iou(probe, second.box) >= STICKY_IOU > iou(probe, second.raw)
    assert sel.update([probe], HW).track_id == second.track_id


def test_iou_ties_between_candidates_are_broken_by_rank_key():
    sel = HandSelector()
    first = sel.update([Detection(400, 100, 600, 300, 0.9)], HW)
    left = Detection(300, 100, 500, 300, 0.8)
    right = Detection(500, 100, 700, 300, 0.9)  # same IoU with the previous box, better score
    assert iou(left, first.box) == iou(right, first.box) >= STICKY_IOU
    s = sel.update([left, right], HW)
    assert s.track_id == first.track_id and s.raw == right


def test_grace_frames_hold_the_last_selection_then_return_none():
    sel = HandSelector()
    sel.update([Detection(100, 100, 300, 300, 0.9)], HW)
    last = sel.update([Detection(120, 100, 320, 300, 0.9), Detection(800, 100, 900, 200, 0.9)], HW)
    assert last.others
    for _ in range(LOST_GRACE_FRAMES):
        held = sel.update([], HW)
        assert held.held is True and held.others == []
        assert (held.box, held.raw, held.track_id) == (last.box, last.raw, last.track_id)
    assert sel.update([], HW) is None
    again = sel.update([Detection(120, 100, 320, 300, 0.9)], HW)
    assert again.track_id == last.track_id + 1 and again.box == again.raw  # a new track, no EMA


def test_a_hand_back_within_the_grace_period_keeps_its_track():
    sel = HandSelector()
    first = sel.update([Detection(100, 100, 300, 300, 0.9)], HW)
    weak = Detection(100, 100, 300, 300, DET_CONF - 0.01)  # filtered out, so it counts as no hand
    for _ in range(LOST_GRACE_FRAMES):
        assert sel.update([weak], HW).held is True
    back = sel.update([Detection(110, 100, 310, 300, 0.9)], HW)
    assert back.track_id == first.track_id and back.held is False


def test_new_track_ids_increment_and_reset_never_reuses_one():
    sel = HandSelector()
    a = Detection(100, 100, 300, 300, 0.9)
    b = Detection(800, 300, 1000, 500, 0.9)
    ids = [sel.update([a], HW).track_id, sel.update([b], HW).track_id]
    for _ in range(LOST_GRACE_FRAMES + 1):
        sel.update([], HW)
    ids.append(sel.update([b], HW).track_id)
    sel.reset()
    after_reset = sel.update([b], HW)
    ids.append(after_reset.track_id)
    assert ids == [1, 2, 3, 4]
    assert after_reset.box == b


def test_box_ema_applies_only_within_a_track():
    sel = HandSelector()
    a0 = Detection(100, 100, 300, 300, 0.9)
    a1 = Detection(120, 110, 330, 310, 0.7)
    a2 = Detection(140, 120, 350, 320, 0.8)
    s0 = sel.update([a0], HW)
    s1 = sel.update([a1], HW)
    assert s1.track_id == s0.track_id
    assert s1.box.xyxy == pytest.approx(ema(a1, a0))
    assert s1.box.conf == a1.conf
    s2 = sel.update([a2], HW)
    assert s2.box.xyxy == pytest.approx(ema(a2, s1.box))  # blends into the previous EMA box
    other = Detection(800, 300, 1000, 500, 0.9)
    s3 = sel.update([other], HW)
    assert s3.track_id != s2.track_id and s3.box == other


def test_raw_is_the_chosen_detection_box_the_ema_and_held_frames_keep_both():  # N6
    sel = HandSelector()
    a0 = Detection(100, 100, 300, 300, 0.9)
    new = sel.update([a0], HW)
    assert new.raw == new.box == a0
    a1 = Detection(130, 100, 330, 300, 0.9)
    distractor = Detection(900, 400, 1000, 500, 0.6)
    cont = sel.update([distractor, a1], HW)
    assert cont.raw == a1
    assert cont.box != cont.raw and cont.box.xyxy == pytest.approx(ema(a1, a0))
    held = sel.update([], HW)
    assert held.held is True and held.raw == cont.raw and held.box == cont.box
