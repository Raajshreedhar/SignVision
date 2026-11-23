"""src.smoothing.ProbSmoother: window mean, renormalization, eviction, reset and input validation."""

import numpy as np
import pytest

from src.config import NUM_CLASSES, SMOOTH_WINDOW
from src.smoothing import ProbSmoother

EYE = np.eye(NUM_CLASSES, dtype=np.float32)


def test_defaults_come_from_config():
    s = ProbSmoother()
    assert (s.window, s.num_classes) == (SMOOTH_WINDOW, NUM_CLASSES)


def test_window_mean():
    s = ProbSmoother(window=3)
    np.testing.assert_allclose(s.push(EYE[0]), EYE[0])
    np.testing.assert_allclose(s.push(EYE[1]), (EYE[0] + EYE[1]) / 2)
    out = s.push(EYE[1])
    np.testing.assert_allclose(out, (EYE[0] + 2 * EYE[1]) / 3, rtol=1e-6)
    assert out.dtype == np.float32 and out.shape == (NUM_CLASSES,)


def test_inputs_are_renormalized():
    s = ProbSmoother(window=2)
    np.testing.assert_allclose(s.push(EYE[3] * 4.0), EYE[3])
    uniform = np.ones(NUM_CLASSES)  # sums to 29 -> 1/29 each
    out = s.push(uniform.tolist())  # plain lists are accepted too
    np.testing.assert_allclose(out, (EYE[3] + uniform / NUM_CLASSES) / 2, rtol=1e-6)
    assert float(out.sum()) == pytest.approx(1.0)


def test_the_oldest_vector_is_evicted():
    s = ProbSmoother(window=SMOOTH_WINDOW)
    for _ in range(SMOOTH_WINDOW):
        s.push(EYE[0])
    out = s.push(EYE[1])
    assert out[0] == pytest.approx((SMOOTH_WINDOW - 1) / SMOOTH_WINDOW)
    assert out[1] == pytest.approx(1 / SMOOTH_WINDOW)
    for _ in range(SMOOTH_WINDOW - 1):
        out = s.push(EYE[1])
    np.testing.assert_allclose(out, EYE[1])


def test_reset_forgets_the_history():
    s = ProbSmoother()
    s.push(EYE[0])
    s.push(EYE[0])
    s.reset()
    np.testing.assert_allclose(s.push(EYE[5]), EYE[5])


def _with_first(value):
    p = np.zeros(NUM_CLASSES)
    p[0] = value
    p[1] = 1.0
    return p


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(NUM_CLASSES - 1),
        np.ones(NUM_CLASSES + 1),
        np.ones((1, NUM_CLASSES)),
        _with_first(-0.1),
        _with_first(np.nan),
        _with_first(np.inf),
        np.zeros(NUM_CLASSES),
    ],
    ids=["short", "long", "2d", "negative", "nan", "inf", "zero-sum"],
)
def test_invalid_vectors_raise(bad):
    with pytest.raises(ValueError):
        ProbSmoother().push(bad)


def test_a_rejected_vector_leaves_the_window_unchanged():
    s = ProbSmoother()
    s.push(EYE[0])
    with pytest.raises(ValueError):
        s.push(np.zeros(NUM_CLASSES))
    np.testing.assert_allclose(s.push(EYE[0]), EYE[0])
