"""Trainer guards: the M3 results.csv run-health check, --time mapping, CPU retry, pre-checks."""

import json
import shutil

import pytest

from src import config
from src import train_classifier
from src import train_detector
from src import yolo_env
from src.class_map import CLASS_DIRS

REBUILD = "rebuild with python -m src.cls_data"

DET_COLUMNS = [
    "train/box_loss", "train/cls_loss", "train/l1_loss",
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/l1_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
]
CLS_COLUMNS = ["train/loss", "metrics/accuracy_top1", "metrics/accuracy_top5", "val/loss", "lr/pg0", "lr/pg1", "lr/pg2"]


def write_results(run_dir, columns, rows, pad=False):
    """Write results.csv the way Ultralytics does: epoch,time,<cols> with %.6g values."""
    run_dir.mkdir(parents=True, exist_ok=True)
    header = ["epoch", "time", *columns]
    if pad:
        header = [f"{h:>20s}" for h in header]
    lines = [",".join(header)]
    for epoch, values in enumerate(rows, start=1):
        cells = [epoch, 10.0 * epoch, *values]
        lines.append(",".join(_fmt(v) for v in cells))
    (run_dir / "results.csv").write_text("\n".join(lines) + "\n")
    return run_dir


def _fmt(value):
    return value if isinstance(value, str) else "%.6g" % value


def finite_det_row(**overrides):
    row = dict(zip(DET_COLUMNS, [1.2, 0.9, 0.05, 0.8, 0.7, 0.75, 0.5, 1.1, 0.8, 0.04, 0.001, 0.001, 0.001]))
    row.update(overrides)
    return [row[c] for c in DET_COLUMNS]


def test_all_finite_is_healthy(tmp_path):
    run = write_results(tmp_path / "run", DET_COLUMNS, [finite_det_row(), finite_det_row()])
    assert yolo_env.run_failure_reason(run) is None


def test_nan_box_loss_is_a_failure(tmp_path):
    run = write_results(tmp_path / "run", DET_COLUMNS, [finite_det_row(), finite_det_row(**{"train/box_loss": float("nan")})])
    reason = yolo_env.run_failure_reason(run)
    assert reason is not None and "train/box_loss" in reason


def test_nan_classifier_loss_is_a_failure(tmp_path):
    run = write_results(tmp_path / "run", CLS_COLUMNS, [[float("nan"), 0.5, 0.9, 1.0, 0.001, 0.001, 0.001]])
    reason = yolo_env.run_failure_reason(run)
    assert reason is not None and "train/loss" in reason


def test_inf_metric_is_a_failure(tmp_path):
    run = write_results(tmp_path / "run", DET_COLUMNS, [finite_det_row(**{"metrics/mAP50-95(B)": float("inf")})])
    reason = yolo_env.run_failure_reason(run)
    assert reason is not None and "metrics/mAP50-95(B)" in reason


def test_unparsable_value_is_a_failure(tmp_path):
    run = write_results(tmp_path / "run", DET_COLUMNS, [finite_det_row(**{"val/cls_loss": "oops"})])
    assert "val/cls_loss" in yolo_env.run_failure_reason(run)


def test_only_last_row_matters(tmp_path):
    run = write_results(tmp_path / "run", DET_COLUMNS, [finite_det_row(**{"train/box_loss": float("nan")}), finite_det_row()])
    assert yolo_env.run_failure_reason(run) is None


def test_nan_learning_rate_is_ignored(tmp_path):
    run = write_results(tmp_path / "run", DET_COLUMNS, [finite_det_row(**{"lr/pg0": float("nan")})])
    assert yolo_env.run_failure_reason(run) is None


def test_padded_headers_are_stripped(tmp_path):
    run = write_results(tmp_path / "run", DET_COLUMNS, [finite_det_row(**{"train/cls_loss": float("nan")})], pad=True)
    rows = yolo_env.read_results(run)
    assert "train/cls_loss" in rows[0] and "epoch" in rows[0]
    assert "train/cls_loss" in yolo_env.run_failure_reason(run)


def test_missing_results_csv_is_a_failure(tmp_path):
    (tmp_path / "run").mkdir()
    reason = yolo_env.run_failure_reason(tmp_path / "run")
    assert reason is not None and "missing" in reason


def test_train_time_arg_mapping():
    assert yolo_env.train_time_arg(0) is None
    assert yolo_env.train_time_arg(0.0) is None
    assert yolo_env.train_time_arg(2.0) == 2.0


def test_common_train_kwargs_use_an_absolute_project(tmp_path):
    kwargs = yolo_env.common_train_kwargs(tmp_path / "runs")
    assert kwargs["project"] == str(tmp_path / "runs")
    assert kwargs["seed"] == config.SEED and kwargs["exist_ok"] is True


def test_cpu_retry_command():
    argv = ["--model", "n", "--time", "2.0", "--device", "mps"]
    assert yolo_env.cpu_retry_command("src.train_detector", argv) == (
        ".venv/bin/python -m src.train_detector --model n --time 2.0 --device cpu"
    )
    assert yolo_env.cpu_retry_command("src.train_detector", ["--device=mps"]).endswith("--device cpu")
    assert yolo_env.cpu_retry_command("src.train_classifier", ["--model", "s"]) == (
        ".venv/bin/python -m src.train_classifier --model s --device cpu"
    )


def test_detector_precheck_exits_1_without_dataset(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "HAND_DET_DIR", tmp_path / "hand_det")
    (tmp_path / "hand_det").mkdir()
    with pytest.raises(SystemExit) as exc:
        train_detector.main(["--model", "n"])
    assert exc.value.code == 1
    assert "src.hand_data" in capsys.readouterr().err


def make_cls_tree(root, **stats_overrides):
    """A minimal valid data/cls tree: 29 class folders per split with one file each, plus stats.json."""
    for split in ("train", "val"):
        for name in CLASS_DIRS:
            folder = root / split / name
            folder.mkdir(parents=True)
            (folder / "x.png").write_bytes(b"png")
    stats = {"preprocess_version": config.PREPROCESS_VERSION, "crop_pad": config.CROP_PAD,
             "cls_imgsz": config.CLS_IMGSZ, "detector": {"path": "hand_detector.pt", "md5": "abc"}}
    stats.update(stats_overrides)
    (root / "stats.json").write_text(json.dumps(stats))
    return root


@pytest.fixture
def cls_dir(tmp_path, monkeypatch):
    root = tmp_path / "cls"
    monkeypatch.setattr(config, "CLS_DIR", root)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    return root


def run_classifier_main(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        train_classifier.main(argv)
    assert exc.value.code == 1
    return capsys.readouterr().err


def test_classifier_precheck_accepts_a_valid_tree(cls_dir):
    make_cls_tree(cls_dir)
    assert train_classifier.precheck(cls_dir, config.CLS_IMGSZ)["cls_imgsz"] == config.CLS_IMGSZ


def test_classifier_missing_dataset_exits_1(cls_dir, capsys):
    err = run_classifier_main(["--model", "n"], capsys)
    assert REBUILD in err


def test_classifier_pad_mismatch_exits_1(cls_dir, capsys):
    make_cls_tree(cls_dir, crop_pad=0.3)
    err = run_classifier_main(["--model", "n"], capsys)
    assert REBUILD in err and "crop_pad 0.3" in err


def test_classifier_imgsz_mismatch_exits_1(cls_dir, capsys):
    make_cls_tree(cls_dir)
    err = run_classifier_main(["--model", "n", "--imgsz", "256"], capsys)
    assert REBUILD in err and "--imgsz 256" in err
    shutil.rmtree(cls_dir)
    make_cls_tree(cls_dir, cls_imgsz=192)  # default --imgsz 224 vs crops built at 192
    assert REBUILD in run_classifier_main(["--model", "n"], capsys)


def test_classifier_version_mismatch_exits_1(cls_dir, capsys):
    make_cls_tree(cls_dir, preprocess_version="gray-clahe-v1")
    err = run_classifier_main(["--model", "n"], capsys)
    assert REBUILD in err and "gray-clahe-v1" in err


def test_classifier_missing_class_dir_exits_1(cls_dir, capsys):
    make_cls_tree(cls_dir)
    shutil.rmtree(cls_dir / "val" / "27_DELETE")
    err = run_classifier_main(["--model", "n"], capsys)
    assert REBUILD in err and "27_DELETE" in err


def test_classifier_extra_class_dir_exits_1(cls_dir, capsys):
    make_cls_tree(cls_dir)
    (cls_dir / "train" / "DELETE").mkdir()  # an unprefixed folder would shift the class order
    err = run_classifier_main(["--model", "n"], capsys)
    assert REBUILD in err and "unexpected ['DELETE']" in err


def test_classifier_empty_train_class_exits_1(cls_dir, capsys):
    make_cls_tree(cls_dir)
    (cls_dir / "train" / "28_NOTHING" / "x.png").unlink()
    (cls_dir / "train" / "28_NOTHING" / "notes.txt").write_text("not an image")
    err = run_classifier_main(["--model", "n"], capsys)
    assert REBUILD in err and "28_NOTHING" in err


def test_classifier_resume_without_last_pt_exits_1(cls_dir, capsys):
    make_cls_tree(cls_dir)
    err = run_classifier_main(["--model", "n", "--resume", "--device", "cpu"], capsys)
    assert "last.pt" in err and "letter_classifier_n" in err


def test_detector_resume_without_last_pt_exits_1(tmp_path, monkeypatch, capsys):
    det = tmp_path / "hand_det"
    det.mkdir()
    (det / "hand.yaml").write_text("names:\n  0: hand\n")
    (det / "stats.json").write_text('{"webcam_view_source": true}')
    monkeypatch.setattr(config, "HAND_DET_DIR", det)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    with pytest.raises(SystemExit) as exc:
        train_detector.main(["--model", "n", "--resume", "--device", "cpu"])
    assert exc.value.code == 1
    assert "last.pt" in capsys.readouterr().err
