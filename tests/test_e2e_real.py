"""Real-weights headless end-to-end test (design section 12).

Pastes the grassknoted A_test.jpg into a 1280x720 gray canvas and runs one
frame through the real SignPipeline on the CPU: the trained hand detector must
find the pasted hand (IoU >= 0.3) and the pipeline must return a canonical
label with positive stage timings. The letter itself is not asserted, to keep
the test stable across retrains. No camera is opened.

The IoU reference is the fist inside the pasted image, not the whole pasted
square: the fist covers only about 43% x 41% of A_test.jpg (the rest is window,
wall and forearm), so even a perfect hand box has an IoU of only 0.17 with the
square.

Skips with a reason when models/hand_detector.pt, models/letter_classifier.pt,
models/letter_classifier.json or the cached A_test.jpg (found through
data/sources.json) are missing.
"""

from pathlib import Path

import numpy as np
import pytest

from src import config
from src.config import CLASSES

pytestmark = [pytest.mark.requires_weights, pytest.mark.requires_data]

PASTE_BOX = (440, 160, 840, 560)  # (x1, y1, x2, y2) of the pasted A_test.jpg in the frame
FRAME_W, FRAME_H = 1280, 720
GRAY = 128
# The fist with the wrist in the 200x200 A_test.jpg, measured from its dark pixels (gray < 80).
HAND_IN_IMAGE = (84, 34, 170, 115)
MIN_IOU = 0.3


@pytest.fixture(scope="module")
def scene():
    """(frame, hand_box): the gray canvas with A_test.jpg resized into PASTE_BOX, and the fist's box in it."""
    import cv2

    from src.benchmark import a_test_image

    missing = [str(p) for p in (config.DET_WEIGHTS, config.CLS_WEIGHTS, config.CLS_META) if not Path(p).is_file()]
    if missing:
        pytest.skip(f"trained models missing ({', '.join(missing)}); run src.train_detector and src.train_classifier")
    path = a_test_image()
    if path is None:
        pytest.skip("grassknoted A_test.jpg not cached (data/sources.json); run "
                    ".venv/bin/python -m src.download_data --sources grassknoted")
    image = cv2.imread(str(path))
    if image is None:
        pytest.skip(f"{path} is not a readable image")

    frame = np.full((FRAME_H, FRAME_W, 3), GRAY, np.uint8)
    x1, y1, x2, y2 = PASTE_BOX
    frame[y1:y2, x1:x2] = cv2.resize(image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    sx, sy = (x2 - x1) / image.shape[1], (y2 - y1) / image.shape[0]
    hx1, hy1, hx2, hy2 = HAND_IN_IMAGE
    return frame, (x1 + hx1 * sx, y1 + hy1 * sy, x1 + hx2 * sx, y1 + hy2 * sy)


def test_real_pipeline_on_pasted_hand(scene):
    from src.pipeline import HandDetector, LetterClassifier, SignPipeline
    from src.smoothing import ProbSmoother
    from src.tracking import HandSelector, iou

    frame, hand_box = scene
    pipeline = SignPipeline(
        HandDetector(config.DET_WEIGHTS, "cpu"),
        LetterClassifier(config.CLS_WEIGHTS, config.CLS_META, "cpu"),
        HandSelector(),
        ProbSmoother(),
    )
    r = pipeline.process(frame)

    assert r.selection is not None, "the detector found no hand anywhere in the frame"
    boxes = [r.selection.raw, *r.selection.others]
    best = max(iou(box, hand_box) for box in boxes)
    assert best >= MIN_IOU, f"best IoU with the pasted hand {hand_box} is {best:.3f}: {boxes}"
    assert r.label in CLASSES
    assert r.probs_raw.shape == (len(CLASSES),)
    assert r.det_ms > 0
    if r.cls_ran:
        assert r.cls_ms > 0
