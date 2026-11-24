"""src.evaluate: eval_classifier with a fake classifier on a tmp crop tree, eval_detector with a fake
val_fn on a tmp detection tree, and the CLI's missing-input exits."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src import config, evaluate
from src.class_map import class_dir_name
from src.config import CLASSES, NUM_CLASSES
from src.evaluate import cls_test_detection_rate, eval_classifier, eval_detector

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


# --- eval_detector --------------------------------------------------------------

VAL_IMAGES = {  # source -> file names in images/val, as src.hand_data names them
    "hand_keypoints": ["hand_keypoints__IMG_0001.jpg", "hand_keypoints__IMG_0002.jpg", "hand_keypoints__IMG_0003.jpg"],
    "hagrid30k": ["hagrid30k__0a1b.jpg", "hagrid30k__ff02.jpg"],
    "composite": ["composite__00000.jpg"],
    "egohands": ["egohands__frame_7.jpg"],  # a prefix outside DET_SOURCES sorts after them
}
# map50 per val run; the other metrics are offsets from it, so every run has distinct numbers
RUN_MAP50 = {
    "detector_val": 0.90,
    "detector_val_hand_keypoints": 0.80,
    "detector_val_hagrid30k": 0.70,
    "detector_val_composite": 0.60,
    "detector_val_egohands": 0.50,
}


def fake_metrics(name):
    m = RUN_MAP50[name]
    return {"map50": m, "map50_95": m - 0.25, "precision": m + 0.05, "recall": m - 0.05}


class FakeVal:
    """Records (name, parsed yaml, file names of the list-file val split) per call."""

    def __init__(self):
        self.calls = []

    def __call__(self, data_yaml, name):
        data = yaml.safe_load(Path(data_yaml).read_text())
        val = Path(data["path"]) / data["val"]
        listed = None
        if val.is_file():
            lines = val.read_text().splitlines()
            assert all(Path(line).is_absolute() for line in lines)
            listed = [Path(line).name for line in lines]
        self.calls.append((name, data, listed))
        return fake_metrics(name)


@pytest.fixture
def det_tree(tmp_path):
    """A tmp data/hand_det with hand.yaml, stats.json and empty val images; data/cls/stats.json next to it."""
    root = tmp_path / "hand_det"
    val_dir = root / "images" / "val"
    val_dir.mkdir(parents=True)
    for names in VAL_IMAGES.values():
        for name in names:
            (val_dir / name).write_bytes(b"jpg")
    (root / "hand.yaml").write_text(
        f"# test dataset\npath: {json.dumps(str(root))}\ntrain: images/train\nval: images/val\nnames:\n  0: hand\n"
    )
    (root / "stats.json").write_text(json.dumps({"webcam_view_source": True}))
    (tmp_path / "cls").mkdir()
    (tmp_path / "cls" / "stats.json").write_text(json.dumps({"n_test": 898, "test_fallback": 31}))
    return root


def run_eval_detector(det_tree, tmp_path, monkeypatch, fake):
    monkeypatch.setattr(config, "CLS_DIR", tmp_path / "cls")
    return eval_detector(
        weights=tmp_path / "hand_detector.pt", data_yaml=det_tree / "hand.yaml", imgsz=512, device="cpu",
        reports_dir=tmp_path / "reports", val_fn=fake,
    )


def test_detector_report_schema(det_tree, tmp_path, monkeypatch, capsys):
    fake = FakeVal()
    report = run_eval_detector(det_tree, tmp_path, monkeypatch, fake)

    assert json.loads((tmp_path / "reports" / "detector.json").read_text()) == report
    assert {"map50", "map50_95", "precision", "recall", "per_source", "cls_test_detection_rate",
            "webcam_view_source"} <= set(report)
    assert {k: report[k] for k in ("map50", "map50_95", "precision", "recall")} == pytest.approx(
        fake_metrics("detector_val")
    )
    assert report["n_images"] == sum(len(v) for v in VAL_IMAGES.values())
    assert report["imgsz"] == 512 and report["device"] == "cpu"
    assert report["webcam_view_source"] is True
    assert report["cls_test_detection_rate"] == pytest.approx(1 - 31 / 898)
    assert (report["cls_test_fallback"], report["cls_test_n"]) == (31, 898)

    assert list(report["per_source"]) == ["hand_keypoints", "hagrid30k", "composite", "egohands"]
    for source, entry in report["per_source"].items():
        expected = {**fake_metrics(f"detector_val_{source}"), "n_images": len(VAL_IMAGES[source])}
        assert {k: entry[k] for k in expected} == pytest.approx(expected)
        assert entry["val_dir"] == str((tmp_path / "reports").resolve() / f"detector_val_{source}")

    # the full val split gets the dataset's own yaml; the temporary subset files are gone
    assert [name for name, _, listed in fake.calls if listed is None] == ["detector_val"]
    assert not list((tmp_path / "reports").glob(".detector_val_*"))
    out = capsys.readouterr().out
    assert "detector val: mAP50 0.9000" in out and "hagrid30k" in out


def test_per_source_list_files_hold_only_that_source(det_tree, tmp_path, monkeypatch):
    fake = FakeVal()
    run_eval_detector(det_tree, tmp_path, monkeypatch, fake)

    listed = {name: files for name, _, files in fake.calls if files is not None}
    assert listed == {f"detector_val_{source}": sorted(files) for source, files in VAL_IMAGES.items()}
    for name, data, files in fake.calls:
        if files is not None:  # each subset yaml keeps the dataset root and the class names
            assert Path(data["path"]) == det_tree and data["names"] == {0: "hand"}
            assert data["train"] == data["val"]


def test_detector_report_without_stats(det_tree, tmp_path, monkeypatch):
    (det_tree / "stats.json").unlink()
    (tmp_path / "cls" / "stats.json").unlink()
    report = run_eval_detector(det_tree, tmp_path, monkeypatch, FakeVal())
    assert report["cls_test_detection_rate"] is None and report["webcam_view_source"] is None


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"n_test": 898, "test_fallback": 31}, 1 - 31 / 898),
        ({"n_test": 10, "test_fallback": 0}, 1.0),
        ({"n_test": 4, "test_fallback": 4}, 0.0),
        ({"n_test": 0, "test_fallback": 0}, None),
        ({"test_fallback": 3}, None),
        ({"n_test": 10}, None),
        ({}, None),
        (None, None),
    ],
)
def test_cls_test_detection_rate(stats, expected):
    rate = cls_test_detection_rate(stats)
    assert rate is None if expected is None else rate == pytest.approx(expected)


# --- CLI ------------------------------------------------------------------------


def point_config_at(tmp_path, monkeypatch):
    for attr, path in {
        "CLS_WEIGHTS": tmp_path / "letter_classifier.pt",
        "CLS_META": tmp_path / "letter_classifier.json",
        "CLS_DIR": tmp_path / "cls",
        "CLS_TEST_DIR": tmp_path / "cls_test",
        "DET_WEIGHTS": tmp_path / "hand_detector.pt",
        "HAND_DET_DIR": tmp_path / "hand_det",
    }.items():
        monkeypatch.setattr(config, attr, path)


def test_cli_exits_1_naming_every_missing_input(tmp_path, monkeypatch, capsys):
    point_config_at(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        evaluate.main(["--split", "all", "--detector"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    for command in ("src.train_classifier", "src.cls_data", "src.train_detector", "src.hand_data"):
        assert command in err
    assert "Traceback" not in err


def test_cli_exits_1_when_only_the_detector_weights_are_missing(tmp_path, monkeypatch, capsys):
    point_config_at(tmp_path, monkeypatch)
    for path in (config.CLS_WEIGHTS, config.CLS_META):
        path.write_bytes(b"")
    (config.CLS_DIR / "val").mkdir(parents=True)
    (config.CLS_TEST_DIR / "test").mkdir(parents=True)
    config.HAND_DET_DIR.mkdir()
    (config.HAND_DET_DIR / "hand.yaml").write_text("val: images/val\n")

    with pytest.raises(SystemExit) as exc:
        evaluate.main(["--detector"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"{config.DET_WEIGHTS} is missing" in err and "src.train_detector" in err
    assert err.count("is missing") == 1


def test_cli_rejects_an_unknown_split():
    with pytest.raises(SystemExit) as exc:
        evaluate.main(["--split", "train"])
    assert exc.value.code == 2
