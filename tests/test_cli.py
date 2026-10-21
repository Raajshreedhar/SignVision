"""Accept/reject cases for the shared CLI validators (design section 9 table)."""

import argparse

import pytest

from src import cli
from src.download_data import sources_arg


def accepts(type_fn, text):
    return type_fn(text)


def rejects(type_fn, text):
    with pytest.raises(argparse.ArgumentTypeError):
        type_fn(text)


@pytest.mark.parametrize("text, value", [("1", 1), ("40", 40), ("1000", 1000)])
def test_epochs_accept(text, value):
    assert accepts(cli.int_range(1, 1000), text) == value


@pytest.mark.parametrize("text", ["0", "1001", "-5", "abc", "1.5", ""])
def test_epochs_reject(text):
    rejects(cli.int_range(1, 1000), text)


@pytest.mark.parametrize("text, value", [("0", 0.0), ("2.0", 2.0), ("24", 24.0)])
def test_time_accept(text, value):
    assert accepts(cli.float_range(0, 24), text) == value


@pytest.mark.parametrize("text", ["-0.1", "24.5", "nan", "inf", "two"])
def test_time_reject(text):
    rejects(cli.float_range(0, 24), text)


@pytest.mark.parametrize("text, value", [("32", 32), ("416", 416), ("512", 512), ("1280", 1280)])
def test_imgsz_accept(text, value):
    assert accepts(cli.imgsz_arg, text) == value


@pytest.mark.parametrize("text", ["0", "16", "500", "1312", "-32", "big"])
def test_imgsz_reject(text):
    rejects(cli.imgsz_arg, text)


@pytest.mark.parametrize("text, value", [("1", 1), ("16", 16), ("-1", -1)])
def test_batch_accept(text, value):
    assert accepts(cli.batch_arg, text) == value


@pytest.mark.parametrize("text", ["0", "-2", "x"])
def test_batch_reject(text):
    rejects(cli.batch_arg, text)


def test_patience_and_min_one_ranges():
    assert cli.int_range(0)("0") == 0
    assert cli.int_range(0)("10") == 10
    rejects(cli.int_range(0), "-1")
    # --max-per-class, --det-batch, --frames: int >= 1; --warmup: int >= 0
    assert cli.int_range(1)("1") == 1
    assert cli.int_range(1)("100000") == 100000
    rejects(cli.int_range(1), "0")


@pytest.mark.parametrize("text, value", [("0.1", 0.1), ("0.5", 0.5), ("0.001", 0.001)])
def test_hagrid_val_frac_accept(text, value):
    assert accepts(cli.float_range(0, 0.5, lo_open=True), text) == value


@pytest.mark.parametrize("text", ["0", "0.0", "-0.1", "0.51", "1"])
def test_hagrid_val_frac_reject(text):
    rejects(cli.float_range(0, 0.5, lo_open=True), text)


def test_existing_pt(tmp_path):
    weights = tmp_path / "hand_detector.pt"
    weights.write_bytes(b"x")
    assert cli.existing_pt(str(weights)) == weights
    rejects(cli.existing_pt, str(tmp_path / "missing.pt"))
    other = tmp_path / "weights.pth"
    other.write_bytes(b"x")
    rejects(cli.existing_pt, str(other))
    folder = tmp_path / "folder.pt"
    folder.mkdir()
    rejects(cli.existing_pt, str(folder))


def test_model_and_device_choices():
    assert cli.MODEL_CHOICES == ("n", "s")
    assert cli.DEVICE_CHOICES == ("auto", "mps", "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=cli.MODEL_CHOICES, default="n")
    parser.add_argument("--device", choices=cli.DEVICE_CHOICES, default="auto")
    args = parser.parse_args(["--model", "s", "--device", "mps"])
    assert (args.model, args.device) == ("s", "mps")
    for bad in (["--model", "m"], ["--device", "cuda"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(bad)
        assert exc.value.code == 2


def test_sources_any_role():
    parse = sources_arg()
    assert parse("grassknoted,hand_keypoints") == ["grassknoted", "hand_keypoints"]
    assert parse(" hagrid30k , hagrid30k ") == ["hagrid30k"]
    rejects(parse, "bogus")
    rejects(parse, ",")


def test_sources_role_filtered():
    assert sources_arg("det")("hand_keypoints,hagrid30k") == ["hand_keypoints", "hagrid30k"]
    rejects(sources_arg("det"), "grassknoted")
    assert sources_arg("cls_train")("grassknoted,ayuraj") == ["grassknoted", "ayuraj"]
    rejects(sources_arg("cls_train"), "hagrid30k")


def test_bad_value_exits_2_through_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=cli.int_range(1, 1000))
    parser.add_argument("--imgsz", type=cli.imgsz_arg)
    for bad in (["--epochs", "0"], ["--imgsz", "500"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(bad)
        assert exc.value.code == 2
