"""Shared pytest fixtures for SignVision.

The autouse ``_no_webcam`` fixture replaces ``cv2.VideoCapture`` with a function
that raises, so no automated test run can ever open the laptop camera. The live
app is checked manually by the user.
"""

from pathlib import Path

import numpy as np
import pytest


def _webcam_disabled(*_args, **_kwargs):
    raise RuntimeError("webcam disabled in tests")


@pytest.fixture(autouse=True)
def _no_webcam(monkeypatch):
    """Make every cv2.VideoCapture(...) call raise inside tests."""
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", _webcam_disabled)


@pytest.fixture
def frame_720p():
    """A (720, 1280, 3) uint8 BGR frame holding a horizontal gray gradient."""
    row = np.linspace(0, 255, 1280).astype(np.uint8)
    gray = np.tile(row, (720, 1))
    return np.repeat(gray[:, :, None], 3, axis=2)


@pytest.fixture
def make_image_tree():
    """Factory that writes seeded random images into ``root/<folder>/``.

    Usage: ``make_image_tree(root, {"A": 3, "nothing": 2}, size=(64, 64))``.
    ``size`` is (width, height). Returns ``root`` as a Path.
    """
    import cv2

    def _make(root, counts, size=(64, 64), ext=".jpg", seed=0):
        root = Path(root)
        rng = np.random.default_rng(seed)
        width, height = size
        for folder, n in counts.items():
            folder_dir = root / folder
            folder_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
                out = folder_dir / f"{Path(folder).name}_{i:04d}{ext}"
                if not cv2.imwrite(str(out), img):
                    raise RuntimeError(f"failed to write fixture image {out}")
        return root

    return _make
