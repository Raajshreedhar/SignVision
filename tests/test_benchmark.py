"""src.benchmark: forced-box detector, timing loop with fake backends, frame, report merge and CLI guards."""

import json

import cv2
import numpy as np
import pytest

from src import benchmark, config
from src.config import CLS_IMGSZ, CROP_PAD, NUM_CLASSES
from src.tracking import Detection


class FakeDetector:
    """Finds nothing; records the frames it was given."""

    def __init__(self):
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame)
        return []


class FakeClassifier:
    crop_pad = CROP_PAD
    imgsz = CLS_IMGSZ

    def __init__(self):
        self.calls = 0

    def classify(self, img):
        self.calls += 1
        return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, np.float32)


def test_forced_box_detector_calls_the_inner_detector_and_returns_the_fixed_box():
    inner = FakeDetector()
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert benchmark._ForcedBoxDetector(inner).detect(frame) == [Detection(440, 160, 840, 560, 0.99)]
    assert len(inner.frames) == 1 and inner.frames[0] is frame


def test_run_benchmark_with_the_forced_box_classifies_every_frame(tmp_path):
    inner, classifier = FakeDetector(), FakeClassifier()
    frame, _ = benchmark.make_frame(sources_file=tmp_path / "missing.json")
    result = benchmark.run_benchmark(benchmark._ForcedBoxDetector(inner), classifier, frame, frames=12, warmup=3)
    assert len(inner.frames) == classifier.calls == 15  # warm-up frames run but are not timed
    for metric in ("det_ms", "cls_ms", "total_ms"):
        assert set(result[metric]) == {"mean", "p50", "p95", "n"}
        assert result[metric]["n"] == 12  # cls_ran on every timed frame
        assert result[metric]["mean"] >= 0  # rounded to 1 us, so a fake stage can read 0.0
    assert result["total_ms"]["mean"] > 0  # includes the real crop + CLAHE
    assert result["total_ms"]["mean"] >= result["det_ms"]["mean"]
    assert result["fps"] == pytest.approx(1000 / result["total_ms"]["mean"], rel=1e-2)
    assert (result["frames"], result["warmup"]) == (12, 3)


def test_run_benchmark_without_a_hand_never_classifies(tmp_path):
    classifier = FakeClassifier()
    frame, _ = benchmark.make_frame(sources_file=tmp_path / "missing.json")
    result = benchmark.run_benchmark(FakeDetector(), classifier, frame, frames=5, warmup=0)
    assert classifier.calls == 0
    assert result["cls_ms"] == {"mean": None, "p50": None, "p95": None, "n": 0}
    assert result["det_ms"]["n"] == result["total_ms"]["n"] == 5


def test_make_frame_pastes_a_test_jpg_into_the_forced_box(tmp_path):
    root = tmp_path / "grassknoted"
    test_dir = root / "asl_alphabet_test" / "asl_alphabet_test"
    test_dir.mkdir(parents=True)
    image = np.random.default_rng(0).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "A_test.jpg"), image)
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"grassknoted": {"path": str(root), "fetched_at": "2026-01-01T00:00:00+00:00"}}))

    frame, desc = benchmark.make_frame(sources_file=sources)
    plain, plain_desc = benchmark.make_frame(sources_file=tmp_path / "missing.json")
    assert frame.shape == plain.shape == (720, 1280, 3) and frame.dtype == plain.dtype == np.uint8
    x1, y1, x2, y2 = benchmark.FORCED_BOX
    expected = cv2.resize(cv2.imread(str(test_dir / "A_test.jpg")), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(frame[y1:y2, x1:x2], expected)
    outside = np.ones(frame.shape[:2], bool)
    outside[y1:y2, x1:x2] = False
    np.testing.assert_array_equal(frame[outside], plain[outside])
    assert "A_test.jpg" in desc and "not cached" in plain_desc


def test_combo_keys():
    assert benchmark.combo_key("pretrained", "s", "n", 512) == "pretrained:det-s+cls-n@512"
    assert benchmark.combo_key("finetuned", "n", "n", 416, force_box=False) == "finetuned:det-n+cls-n@416:no-force-box"


def test_write_report_keeps_earlier_combos(tmp_path):
    frame, _ = benchmark.make_frame(sources_file=tmp_path / "missing.json")
    timed = benchmark.run_benchmark(
        benchmark._ForcedBoxDetector(FakeDetector()), FakeClassifier(), frame, frames=3, warmup=0
    )
    benchmark.write_report("cpu", "pretrained:det-n+cls-n@512", timed, reports_dir=tmp_path)
    benchmark.write_report("cpu", "finetuned:det-n+cls-n@512", {"fps": 1.0}, reports_dir=tmp_path)
    data = json.loads((tmp_path / "benchmark_cpu.json").read_text())
    assert data == {"pretrained:det-n+cls-n@512": timed, "finetuned:det-n+cls-n@512": {"fps": 1.0}}

    merged = benchmark.write_report("cpu", "finetuned:det-n+cls-n@512", {"fps": 2.0}, reports_dir=tmp_path)
    assert merged == json.loads((tmp_path / "benchmark_cpu.json").read_text())
    assert merged["finetuned:det-n+cls-n@512"] == {"fps": 2.0}
    assert merged["pretrained:det-n+cls-n@512"] == timed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["benchmark_cpu.json"]  # no temp files, no other device


def test_cli_rejects_mode_mismatched_flags():
    for argv in (["--mode", "finetuned", "--matrix"], ["--mode", "pretrained", "--no-force-box"]):
        with pytest.raises(SystemExit) as exc:
            benchmark.main(argv)
        assert exc.value.code == 2


@pytest.mark.parametrize("attr, trainer", [("DET_WEIGHTS", "src.train_detector"), ("CLS_WEIGHTS", "src.train_classifier")])
def test_finetuned_with_missing_weights_exits_1_with_the_command(attr, trainer, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, attr, tmp_path / "missing.pt")
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--mode", "finetuned"])
    assert exc.value.code == 1
    assert trainer in capsys.readouterr().err
