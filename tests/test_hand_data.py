"""src.hand_data: label conversion, HaGRID split, composites (tmp fake trees only)."""

import json
import math

import cv2
import numpy as np
import pytest

from src import config
from src import hand_data as hd

CANVAS = (320, 180)


@pytest.fixture(autouse=True)
def small_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HAND_DET_DIR", tmp_path / "hand_det")
    monkeypatch.setattr(config, "SOURCES_FILE", tmp_path / "sources.json")
    monkeypatch.setattr(config, "ZOOMOUT_COMPOSITES_TRAIN", 6)
    monkeypatch.setattr(config, "ZOOMOUT_COMPOSITES_VAL", 3)
    monkeypatch.setattr(config, "COMPOSITE_CANVAS", CANVAS)


def pose_line(cx=0.5, cy=0.5, w=0.4, h=0.6, cls=0, vis=2, kx=0.5, ncols=hd.POSE_COLUMNS):
    values = [cls, cx, cy, w, h]
    while len(values) < ncols:
        values += [kx, 0.5, vis]
    return " ".join(str(v) for v in values[:ncols])


def write_image(path, size=(48, 32), value=None, seed=0):
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        img = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    else:
        img = np.full((height, width, 3), value, np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


def make_hk_tree(root, n_train=5, n_val=3, bad_train=0, unlabeled_train=0, size=(48, 32)):
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            stem = f"IMG_{split}_{i:04d}"
            write_image(root / "images" / split / f"{stem}.jpg", size=size, seed=i)
            if split == "train" and i < unlabeled_train:
                continue
            bad = split == "train" and unlabeled_train <= i < unlabeled_train + bad_train
            line = pose_line(ncols=hd.POSE_COLUMNS - 1) if bad else pose_line(cx=round(0.3 + 0.01 * (i % 50), 4))
            label = root / "labels" / split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(line + "\n")
    return root


def make_nothing_tree(root, n=3, value=None):
    folder = root / "asl_alphabet_train" / "asl_alphabet_train" / "nothing"
    for i in range(n):
        write_image(folder / f"nothing{i}.jpg", size=(200, 200), value=value, seed=100 + i)
    return root


def write_sources(**paths):
    data = {name: {"path": str(path), "fetched_at": "2025-01-01T00:00:00+00:00"} for name, path in paths.items()}
    config.SOURCES_FILE.write_text(json.dumps(data))


def make_split_dirs(out):
    for kind in ("images", "labels"):
        for split in hd.SPLITS:
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    return out


def read_labels(out, split):
    return {p.stem: p.read_text() for p in sorted((out / "labels" / split).glob("*.txt"))}


def make_hagrid_tree(root, records, missing_images=()):
    """records: {gesture: {image_id: record}}; images are written for every id not in missing_images."""
    base = root / "hagrid-sample-30k-384p"
    for gesture, recs in records.items():
        ann = base / "ann_train_val" / f"{gesture}.json"
        ann.parent.mkdir(parents=True, exist_ok=True)
        ann.write_text(json.dumps(recs))
        for image_id in recs:
            if image_id not in missing_images:
                write_image(base / "hagrid_30k" / f"train_val_{gesture}" / f"{image_id}.jpg", size=(64, 36))
    return root


# --- pose lines ---------------------------------------------------------------


def test_convert_pose_line_keeps_the_box():
    line = pose_line(cx=0.25, cy=0.75, w=0.5, h=0.125, vis=2, kx=1.12)  # visibility 2, landmark off-image
    assert hd.convert_pose_line(line) == "0 0.25 0.75 0.5 0.125"
    assert len(line.split()) == 68


@pytest.mark.parametrize(
    "line",
    [
        pose_line(ncols=67),
        pose_line(ncols=67) + " 0.5 0.5",  # 69 columns
        pose_line(cx=1.2),
        pose_line(w=-0.1),
        pose_line(h="nan"),
        pose_line(cls=1),
        pose_line().replace("0.4", "abc", 1),
        "",
    ],
)
def test_convert_pose_line_rejects_malformed(line):
    assert hd.convert_pose_line(line) is None


def test_find_hk_root_nested(tmp_path):
    root = make_hk_tree(tmp_path / "dl" / "nested" / "hand-keypoints")
    assert hd.find_hk_root(tmp_path / "dl") == root
    with pytest.raises(hd.HandDataError):
        hd.find_hk_root(tmp_path / "missing")


# --- full build: hand-keypoints only -> composites ----------------------------


def test_build_without_hagrid_writes_dataset_composites_and_warning(tmp_path, capsys):
    hk = make_hk_tree(tmp_path / "raw" / "hand_keypoints", n_train=5, n_val=3, unlabeled_train=1)
    gk = make_nothing_tree(tmp_path / "gk")
    write_sources(hand_keypoints=hk, grassknoted=gk)

    stats = hd.build(["hand_keypoints"], 0.1)
    out = config.HAND_DET_DIR

    assert stats["webcam_view_source"] is False
    assert stats["sources"]["hand_keypoints"]["train"]["images"] == 5
    assert stats["sources"]["hand_keypoints"]["train"]["boxes"] == 4
    assert stats["sources"]["hand_keypoints"]["train"]["unlabeled_images"] == 1
    assert stats["sources"]["hand_keypoints"]["val"]["boxes"] == 3
    c = stats["composites"]
    assert (c["train"], c["val"]) == (6, 3)
    assert c["bg_nothing"] + c["bg_gradient"] == 9
    assert stats["totals"]["train"] == {"images": 5 + 6, "boxes": 4 + 6}
    assert stats["totals"]["val"] == {"images": 3 + 3, "boxes": 3 + 3}

    assert (out / "images" / "train" / "hand_keypoints__IMG_train_0001.jpg").is_file()
    assert (out / "labels" / "train" / "hand_keypoints__IMG_train_0000.txt").read_text() == ""  # background
    assert (out / "labels" / "train" / "hand_keypoints__IMG_train_0001.txt").read_text() == "0 0.31 0.5 0.4 0.6\n"
    assert (out / "images" / "val" / "composite__00008.jpg").is_file()  # numbering continues into val

    on_disk = json.loads((out / "stats.json").read_text())
    assert on_disk["webcam_view_source"] is False
    assert on_disk["composites"] == c
    yaml_text = (out / "hand.yaml").read_text()
    assert f"path: {json.dumps(str(out.resolve()))}" in yaml_text
    assert "train: images/train" in yaml_text and "val: images/val" in yaml_text and "0: hand" in yaml_text

    err = capsys.readouterr().err
    assert "WARNING: hagrid30k unavailable (" in err
    assert "+ 9 zoom-out composites" in err
    assert "re-run src.download_data --sources hagrid30k" in err


def test_build_is_a_fresh_rebuild(tmp_path):
    hk = make_hk_tree(tmp_path / "raw" / "hand_keypoints")
    write_sources(hand_keypoints=hk)
    stale = config.HAND_DET_DIR / "images" / "train" / "stale__old.jpg"
    write_image(stale)
    stats = hd.build(["hand_keypoints"], 0.1)
    assert not stale.exists()
    assert stats["composites"]["bg_nothing"] == 0  # grassknoted not cached -> gradients only


def test_bad_line_is_dropped_and_counted(tmp_path):
    hk = make_hk_tree(tmp_path / "raw" / "hand_keypoints", n_train=150, n_val=2, bad_train=1)
    write_sources(hand_keypoints=hk)
    stats = hd.build(["hand_keypoints"], 0.1)
    train = stats["sources"]["hand_keypoints"]["train"]
    assert train["dropped_lines"] == 1
    assert train["label_lines"] == 150
    assert train["boxes"] == 149
    assert train["images"] == 150


def test_more_than_one_percent_bad_lines_aborts(tmp_path, capsys):
    hk = make_hk_tree(tmp_path / "raw" / "hand_keypoints", n_train=50, n_val=2, bad_train=2)
    write_sources(hand_keypoints=hk)
    sentinel = config.HAND_DET_DIR / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("previous build")

    with pytest.raises(hd.HandDataError, match="dropped 2/50"):
        hd.build(["hand_keypoints"], 0.1)
    assert hd.main([]) == 1
    assert "ERROR: hand_keypoints train: dropped 2/50" in capsys.readouterr().err
    assert sentinel.exists()  # validation happens before the old output is deleted


# --- HaGRID -------------------------------------------------------------------


def test_hagrid_box_to_yolo_with_clipping():
    assert hd.hagrid_box_to_yolo([0.1, 0.2, 0.3, 0.4]) == pytest.approx((0.25, 0.4, 0.3, 0.4))
    # left edge and bottom edge clipped to the image
    assert hd.hagrid_box_to_yolo([-0.1, 0.9, 0.3, 0.2]) == pytest.approx((0.1, 0.95, 0.2, 0.1))
    assert hd.hagrid_box_to_yolo([0.5, 0.5, 0.005, 0.01]) is None  # w*h < 1e-4
    assert hd.hagrid_box_to_yolo([1.2, 0.5, 0.1, 0.1]) is None  # entirely outside


def test_hagrid_split_is_deterministic_and_user_disjoint():
    splits = {}
    for u in range(300):
        user = f"user-{u}"
        for i in range(3):
            split = hd.hagrid_split(user, f"img-{u}-{i}", 0.2)
            assert split == hd.hagrid_split(user, f"img-{u}-{i}", 0.2)
            splits.setdefault(user, set()).add(split)
    assert all(len(s) == 1 for s in splits.values())  # one user never spans both splits
    n_val = sum(s == {"val"} for s in splits.values())
    assert 30 < n_val < 90  # roughly 20% of 300 users
    # a missing user id hashes the image id instead
    assert hd.hagrid_split(None, "img-x", 0.5) == hd.hagrid_split("img-x", "unused", 0.5)


def test_build_with_hagrid_includes_webcam_view_source(tmp_path):
    hk = make_hk_tree(tmp_path / "raw" / "hand_keypoints")
    records = {
        "call": {
            "id-a": {"bboxes": [[0.1, 0.2, 0.3, 0.4]], "labels": ["call"], "user_id": "u1"},
            "id-b": {"bboxes": [[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.001, 0.001]], "labels": ["call", "no_gesture"],
                     "user_id": "u2"},
            "id-missing": {"bboxes": [[0.1, 0.1, 0.1, 0.1]], "user_id": "u3"},
        },
        "palm": {"id-c": {"bboxes": [], "labels": [], "user_id": "u4"}},
    }
    hg = make_hagrid_tree(tmp_path / "raw" / "hagrid", records, missing_images={"id-missing"})
    write_sources(hand_keypoints=hk, hagrid30k=hg)

    stats = hd.build(["hand_keypoints", "hagrid30k"], 0.5)
    out = config.HAND_DET_DIR

    assert stats["webcam_view_source"] is True
    assert stats["hagrid_status"] == "included"
    assert stats["composites"] == {"train": 0, "val": 0, "bg_nothing": 0, "bg_gradient": 0}
    hg_stats = stats["sources"]["hagrid30k"]
    assert sum(s["images"] for s in hg_stats.values()) == 3  # id-missing has no image
    assert sum(s["boxes"] for s in hg_stats.values()) == 2
    assert sum(s["dropped_boxes"] for s in hg_stats.values()) == 1
    assert sum(s["background_images"] for s in hg_stats.values()) == 1
    split_a = hd.hagrid_split("u1", "id-a", 0.5)
    assert (out / "labels" / split_a / "hagrid30k__id-a.txt").read_text() == "0 0.250000 0.400000 0.300000 0.400000\n"
    assert (out / "images" / split_a / "hagrid30k__id-a.jpg").is_file()
    assert not list(out.rglob("composite__*"))


def test_hagrid_record_without_bboxes_falls_back_to_composites(tmp_path, capsys):
    hk = make_hk_tree(tmp_path / "raw" / "hand_keypoints")
    records = {"call": {"id-a": {"labels": ["call"], "user_id": "u1"}}}
    hg = make_hagrid_tree(tmp_path / "raw" / "hagrid", records)
    write_sources(hand_keypoints=hk, hagrid30k=hg)

    stats = hd.build(["hand_keypoints", "hagrid30k"], 0.1)
    assert stats["webcam_view_source"] is False
    assert stats["hagrid_status"].startswith("schema error")
    assert "hagrid30k" not in stats["sources"]
    assert stats["composites"]["train"] == 6
    assert not list(config.HAND_DET_DIR.rglob("hagrid30k__*"))
    err = capsys.readouterr().err
    assert "ERROR: hagrid30k aborted" in err and "has no 'bboxes'" in err
    assert "WARNING: hagrid30k unavailable (schema error" in err


# --- composites -----------------------------------------------------------------


def test_transform_box_math():
    # box centered in a 100x50 paste at (200, 40) on a 400x200 canvas
    cx, cy, w, h = hd.transform_box((0.5, 0.5, 0.2, 0.4), (200, 40, 100, 50), (400, 200))
    assert (cx, cy, w, h) == pytest.approx((250 / 400, 65 / 200, 20 / 400, 20 / 200))


def test_composite_box_matches_pasted_region_within_1px(tmp_path):
    # Hand image: black with a white rectangle; the label box covers exactly that rectangle.
    hand = np.zeros((80, 120, 3), np.uint8)
    hand[10:50, 20:80] = 255
    src = tmp_path / "src" / "hand.png"
    src.parent.mkdir()
    cv2.imwrite(str(src), hand)
    line = f"0 {50 / 120} {30 / 80} {60 / 120} {40 / 80}"
    black_bg = [write_image(tmp_path / "bg" / "black.png", size=(200, 200), value=0)]
    out = make_split_dirs(tmp_path / "out")

    counts = hd.make_composites([(src, [line])], [(src, [line])], out, black_bg, seed=7)
    assert counts["train"] == 6 and counts["val"] == 3

    checked = 0
    W, H = CANVAS
    for split in hd.SPLITS:
        for label in sorted((out / "labels" / split).glob("*.txt")):
            img = cv2.imread(str(out / "images" / split / f"{label.stem}.jpg"))
            corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
            if max(int(c.max()) for c in corners) > 30:
                continue  # gradient background; only black-background composites are measured
            ys, xs = np.nonzero(img.max(axis=2) > 128)
            cx, cy, w, h = (float(v) for v in label.read_text().split()[1:])
            x1, x2 = (cx - w / 2) * W, (cx + w / 2) * W
            y1, y2 = (cy - h / 2) * H, (cy + h / 2) * H
            assert abs(xs.min() - x1) <= 1 and abs(xs.max() + 1 - x2) <= 1
            assert abs(ys.min() - y1) <= 1 and abs(ys.max() + 1 - y2) <= 1
            checked += 1
    assert checked >= 3
    assert counts["bg_nothing"] >= checked


def _hk_items(tmp_path, split, n, size):
    items = []
    for i in range(n):
        path = write_image(tmp_path / f"src_{split}" / f"{split}_{i}.jpg", size=size, seed=i)
        items.append((path, [f"0 0.5 0.5 1.0 1.0"]))
    return items


def test_composites_are_deterministic_for_a_seed(tmp_path):
    train = _hk_items(tmp_path, "train", 4, (60, 40))
    val = _hk_items(tmp_path, "val", 2, (60, 40))
    outs = [make_split_dirs(tmp_path / name) for name in ("a", "b", "c")]
    hd.make_composites(train, val, outs[0], [], seed=3)
    hd.make_composites(train, val, outs[1], [], seed=3)
    hd.make_composites(train, val, outs[2], [], seed=4)
    for split in hd.SPLITS:
        assert read_labels(outs[0], split) == read_labels(outs[1], split)
        for img in (outs[0] / "images" / split).glob("*.jpg"):
            assert img.read_bytes() == (outs[1] / "images" / split / img.name).read_bytes()
    assert read_labels(outs[0], "train") != read_labels(outs[2], "train")


def test_composite_splits_come_from_disjoint_source_splits(tmp_path):
    # train sources are 2:1 landscape, val sources 1:2 portrait; full-image boxes reveal the source split.
    train = _hk_items(tmp_path, "train", 4, (80, 40))
    val = _hk_items(tmp_path, "val", 4, (40, 80))
    out = make_split_dirs(tmp_path / "out")
    hd.make_composites(train, val, out, [], seed=1)
    W, H = CANVAS
    for split, expected in (("train", 2.0), ("val", 0.5)):
        labels = read_labels(out, split)
        assert labels
        for text in labels.values():
            _, _, _, w, h = (float(v) for v in text.split())
            assert math.isclose((w * W) / (h * H), expected, rel_tol=0.1)


# --- CLI ------------------------------------------------------------------------


def test_cli_rejects_non_det_sources_and_bad_val_frac():
    for argv in (["--sources", "grassknoted"], ["--hagrid-val-frac", "0"], ["--hagrid-val-frac", "0.6"]):
        with pytest.raises(SystemExit) as exc:
            hd.main(argv)
        assert exc.value.code == 2


def test_cli_missing_hand_keypoints_exits_1(capsys):
    assert hd.main([]) == 1
    err = capsys.readouterr().err
    assert "src.download_data" in err and "hand_keypoints" in err
