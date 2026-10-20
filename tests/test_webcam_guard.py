"""The autouse conftest guard must keep every test away from the webcam."""

import cv2
import pytest


def test_videocapture_is_disabled():
    with pytest.raises(RuntimeError, match="webcam disabled in tests"):
        cv2.VideoCapture(0)
