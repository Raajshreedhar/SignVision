"""Shape test for SignVisionCNN.

Confirms a (2, 1, 64, 64) batch forwards to (2, 29) logits. Run with the venv
interpreter: `.venv/bin/python tests/test_model.py`.
"""

import os
import sys

import torch

# Allow running directly (python tests/test_model.py) by putting the project
# root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import IMG_SIZE, NUM_CLASSES
from src.model import SignVisionCNN


def test_forward_shape():
    model = SignVisionCNN()
    model.eval()
    x = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES), f"expected (2, {NUM_CLASSES}), got {tuple(out.shape)}"


if __name__ == "__main__":
    test_forward_shape()
    print("test_model ok")
