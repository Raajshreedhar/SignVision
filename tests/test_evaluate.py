"""src.evaluate.eval_classifier with a fake classifier on a tmp crop tree."""

import json

import cv2
import numpy as np
import pytest

from src.class_map import class_dir_name
from src.config import CLASSES, NUM_CLASSES
from src.evaluate import eval_classifier

IDX = {label: i for i, label in enumerate(CLASSES)}

# (true label, ranking of predicted labels, best first) per image
SPECS = [
    ("A", ["A", "B"]),
    ("A", ["A", "C"]),
    ("A", ["A"]),
    ("A", ["B", "A"]),  # top-1 miss, top-5 hit
    ("B", ["C", "D", "E", "F", "G", "B"]),  # B ranked 6th: top-5 miss
    ("B", ["C", "D", "E", "F", "G", "H", "B"]),
    ("NOTHING", ["NOTHING"]),
    ("NOTHING", ["NOTHING", "A"]),
    ("SPACE", ["DELETE", "A", "SPACE"]),  # top-5 hit
]


def probs_for(ranking):
    p = np.full(NUM_CLASSES, 1e-4, np.float32)
    for k, label in enumerate(ranking):
        p[IDX[label]] = len(ranking) - k
    return p / p.sum()


class FakeClassifier:
    """Reads the spec id stored in pixel (0, 0) and returns that spec's probabilities."""

    def __init__(self):
        self.calls = 0

    def classify(self, img):
        assert img.dtype == np.uint8 and img.shape == (16, 16, 3)
        self.calls += 1
        return probs_for(SPECS[int(img[0, 0, 0])][1])


@pytest.fixture
def split_dir(tmp_path):
    root = tmp_path / "cls_test" / "test"
    for i in range(NUM_CLASSES):
        if i % 5 != 4:  # some class folders are missing entirely
            (root / class_dir_name(i)).mkdir(parents=True)
    for spec_id, (label, _) in enumerate(SPECS):
        folder = root / class_dir_name(IDX[label])
        folder.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(folder / f"img_{spec_id}.png"), np.full((16, 16, 3), spec_id, np.uint8))
    return root


def test_confusion_top1_top5_and_report(split_dir, tmp_path, capsys):
    fake = FakeClassifier()
    report = eval_classifier(None, None, split_dir, "cpu", classifier=fake, reports_dir=tmp_path / "reports")

    assert fake.calls == len(SPECS) == report["n_images"]
    assert report["split"] == "test"  # defaults to the folder name
    confusion = np.array(report["confusion"])
    assert confusion.shape == (NUM_CLASSES, NUM_CLASSES)
    assert report["classes"] == CLASSES
    expected = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for label, ranking in SPECS:
        expected[IDX[label], IDX[ranking[0]]] += 1
    np.testing.assert_array_equal(confusion, expected)  # rows true, columns predicted, canonical order
    assert confusion[IDX["SPACE"], IDX["DELETE"]] == 1

    assert report["top1"] == pytest.approx(5 / 9)
    assert report["top5"] == pytest.approx(7 / 9)
    assert report["per_class"]["A"] == pytest.approx(0.75)
    assert report["per_class"]["B"] == 0.0
    assert report["per_class"]["NOTHING"] == 1.0
    assert report["per_class"]["Z"] is None  # no samples
    assert report["per_class_n"]["A"] == 4

    assert report["weakest_classes"] == [
        {"class": "B", "accuracy": 0.0, "n": 2},
        {"class": "SPACE", "accuracy": 0.0, "n": 1},
        {"class": "A", "accuracy": 0.75, "n": 4},
        {"class": "NOTHING", "accuracy": 1.0, "n": 2},
    ]
    assert [(c["true"], c["pred"], c["count"]) for c in report["most_confused"]] == [
        ("B", "C", 2), ("A", "B", 1), ("SPACE", "DELETE", 1),
    ]

    on_disk = json.loads((tmp_path / "reports" / "classifier_test.json").read_text())
    assert on_disk["top1"] == report["top1"] and on_disk["confusion"] == report["confusion"]
    assert on_disk["weakest_classes"] == report["weakest_classes"]
    assert (tmp_path / "reports" / "confusion_test.png").stat().st_size > 0

    out = capsys.readouterr().out
    assert "weakest classes: B 0.00 (n=2)" in out
    assert "B->C 2" in out


def test_split_name_and_unreadable_files(split_dir, tmp_path):
    (split_dir / class_dir_name(0) / "broken.png").write_bytes(b"not a png")
    report = eval_classifier(
        None, None, split_dir, "cpu", classifier=FakeClassifier(), split_name="val", reports_dir=tmp_path / "r"
    )
    assert report["read_failures"] == 1 and report["n_images"] == len(SPECS)
    assert (tmp_path / "r" / "classifier_val.json").is_file()
    assert (tmp_path / "r" / "confusion_val.png").is_file()


def test_empty_split_reports_none(tmp_path):
    report = eval_classifier(
        None, None, tmp_path / "empty", "cpu", classifier=FakeClassifier(), reports_dir=tmp_path / "r"
    )
    assert report["n_images"] == 0 and report["top1"] is None and report["weakest_classes"] == []


def test_wrong_probability_shape_raises(split_dir, tmp_path):
    class Bad:
        def classify(self, img):
            return np.ones(10, np.float32)

    with pytest.raises(ValueError, match="shape"):
        eval_classifier(None, None, split_dir, "cpu", classifier=Bad(), reports_dir=tmp_path / "r")
