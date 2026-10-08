"""Unit tests for the shared preprocess_frame (invariant a).

Run: .venv/bin/python tests/test_preprocess.py
"""

import os
import sys

import numpy as np

# Allow running directly (python tests/test_preprocess.py) by putting the
# project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import IMG_SIZE
from src.preprocess import preprocess_frame


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


if __name__ == "__main__":
    test_color_input()
    test_gray_input()
    print("test_preprocess.py: all assertions passed")
