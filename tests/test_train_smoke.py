"""Trainer smoke round trips and the classifier augmentation pin (design section 12, amendment M1).

The two slow tests run each trainer's smoke_test() in-process on the CPU into
tmp_path: a 1-epoch Ultralytics train, the real post-train steps (copy best.pt,
val, meta), and a reload of the saved weights. They need the pretrained
YOLO26 weights in models/pretrained/ (downloaded on first use) and skip when
those cannot be fetched. models/hand_detector.pt and models/letter_classifier.pt
are never touched.

test_classifier_aug_matches_choice needs no weights and runs in the quick
suite: it pins the M1 choice that the classifier trains with RandAugment and
without ColorJitter.
"""

import json

import numpy as np
import pytest

from src.config import CLASSES, NUM_CLASSES
from src.train_classifier import AUG_ARGS


@pytest.fixture(scope="module")
def yolo_env():
    """src.yolo_env, once the pretrained YOLO26 detector and classifier load (skips when offline)."""
    from src import yolo_env

    try:
        for name in ("yolo26n.pt", "yolo26n-cls.pt"):
            yolo_env.load_yolo(yolo_env.pretrained_path(name))
    except Exception as exc:  # noqa: BLE001 - any download or load failure means the weights are unavailable
        pytest.skip(f"pretrained YOLO26 weights unavailable (offline?): {type(exc).__name__}: {exc}")
    return yolo_env


@pytest.mark.slow
@pytest.mark.requires_weights
def test_detector_smoke_round_trip(yolo_env, tmp_path):
    from src import train_detector

    smoke_dir = tmp_path / "det"
    train_detector.smoke_test(smoke_dir=smoke_dir, device="cpu")

    model = yolo_env.load_yolo(smoke_dir / "hand_detector.pt")
    assert dict(model.names) == {0: "hand"}
    meta = json.loads((smoke_dir / "hand_detector.json").read_text())
    assert meta["names"] == {"0": "hand"}
    assert meta["model_size"] == "n" and meta["device_used"] == "cpu"
    assert set(meta["metrics"]) == {"map50", "map50_95", "precision", "recall"}


@pytest.mark.slow
@pytest.mark.requires_weights
def test_classifier_smoke_round_trip(yolo_env, tmp_path):
    from src import train_classifier
    from src.class_map import decode_class_names
    from src.pipeline import LetterClassifier
    from src.preprocess import preprocess_hand_crop

    smoke_dir = tmp_path / "cls"
    train_classifier.smoke_test(smoke_dir=smoke_dir, device="cpu")
    weights, meta_path = smoke_dir / "letter_classifier.pt", smoke_dir / "letter_classifier.json"

    assert decode_class_names(yolo_env.load_yolo(weights).names) == CLASSES
    meta = json.loads(meta_path.read_text())
    assert meta["classes"] == CLASSES

    classifier = LetterClassifier(weights, meta_path, "cpu")
    crop = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    probs = classifier.classify(preprocess_hand_crop(crop, size=classifier.imgsz))
    assert probs.shape == (NUM_CLASSES,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-3)


def test_classifier_aug_matches_choice(tmp_path, make_image_tree):
    """M1: with auto_augment set, Ultralytics builds RandAugment and drops ColorJitter, so hsv_* stay 0."""
    import src.yolo_env  # noqa: F401  - MPS fallback env var and telemetry opt-out before ultralytics is used
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import ClassificationDataset

    root = make_image_tree(tmp_path / "cls" / "train", {"00_A": 2, "01_B": 2}, size=(40, 40))
    dataset = ClassificationDataset(
        root=str(root), args=get_cfg(overrides={"imgsz": 32, **AUG_ARGS}), augment=True, prefix="train"
    )
    names = [type(t).__name__ for t in dataset.torch_transforms.transforms]
    assert "RandAugment" in names, names
    assert "ColorJitter" not in names, names
    assert AUG_ARGS["auto_augment"] == "randaugment"
    assert AUG_ARGS["hsv_h"] == AUG_ARGS["hsv_s"] == AUG_ARGS["hsv_v"] == 0.0
