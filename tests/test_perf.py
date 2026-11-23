"""src.perf.LatencyTracker: rolling FPS, session mean / p50 / p95, and cls_ms=None handling."""

import pytest

from src.perf import LatencyTracker


def test_fps_is_the_rolling_mean_over_the_window():
    t = LatencyTracker(window=30)
    assert t.fps() == 0.0
    for _ in range(30):
        t.record(det_ms=5, cls_ms=3, total_ms=10)
    assert t.fps() == pytest.approx(100.0)
    for _ in range(30):
        t.record(det_ms=5, cls_ms=3, total_ms=40)
    assert t.fps() == pytest.approx(25.0)  # only the last 30 frames count
    t.record(det_ms=5, cls_ms=3, total_ms=10)
    assert t.fps() == pytest.approx(1000 / ((29 * 40 + 10) / 30))


def test_summary_covers_the_whole_session():
    t = LatencyTracker(window=5)
    for v in range(1, 101):
        t.record(det_ms=v, cls_ms=2 * v, total_ms=3 * v)
    s = t.summary()
    assert s["det_ms"]["n"] == s["cls_ms"]["n"] == s["total_ms"]["n"] == 100
    assert s["det_ms"]["mean"] == pytest.approx(50.5)
    assert s["det_ms"]["p50"] == pytest.approx(50.5)
    assert s["det_ms"]["p95"] == pytest.approx(95.05)  # linear interpolation, as numpy.percentile
    assert s["cls_ms"]["p95"] == pytest.approx(2 * 95.05)
    assert s["total_ms"]["mean"] == pytest.approx(3 * 50.5)


def test_cls_ms_none_is_left_out():
    t = LatencyTracker()
    t.record(det_ms=4, cls_ms=None, total_ms=10)
    t.record(det_ms=6, cls_ms=8, total_ms=20)
    s = t.summary()
    assert s["det_ms"]["n"] == s["total_ms"]["n"] == 2
    assert s["cls_ms"] == {"mean": 8.0, "p50": 8.0, "p95": 8.0, "n": 1}
    assert s["det_ms"]["mean"] == pytest.approx(5.0)


def test_summary_of_an_empty_tracker():
    s = LatencyTracker().summary()
    assert set(s) == {"det_ms", "cls_ms", "total_ms"}
    assert all(v == {"mean": None, "p50": None, "p95": None, "n": 0} for v in s.values())
