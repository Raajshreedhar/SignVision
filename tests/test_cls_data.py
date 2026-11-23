"""src.cls_data: leakage removal, dedupe, deterministic cap, split, crops and stats (tmp trees only)."""

import json
import random
import shutil
import string
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import cls_data, config
from src.class_map import CLASS_DIRS, class_dir_name
from src.cls_data import Item, cap_manifest, dedupe, plan_background_crops, remove_test_leaks, stratified_split
from src.config import CLASSES, CLS_IMGSZ, CROP_PAD, PREPROCESS_VERSION
from src.hand_data import HagridRecord
from src.preprocess import classifier_input
from src.tracking import Detection, iou

GK_FOLDERS = list(string.ascii_uppercase) + ["space", "del", "nothing"]
FAKE_MD5 = "f" * 32


@pytest.fixture(autouse=True)
def tmp_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CLS_DIR", tmp_path / "out" / "cls")
    monkeypatch.setattr(config, "CLS_TEST_DIR", tmp_path / "out" / "cls_test")
    monkeypatch.setattr(config, "SOURCES_FILE", tmp_path / "sources.json")
    monkeypatch.setattr(config, "DET_WEIGHTS", tmp_path / "models" / "hand_detector.pt")
    monkeypatch.setattr(cls_data, "NOTHING_BG_CROPS", 3)


def write_img(path, seed, size=(48, 40)):
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


class FakeDetector:
    """One centered box per image, none for files whose name contains 'nohand'."""

    def __init__(self):
        self.weights_path = Path("/fake/models/hand_detector.pt")
        self.md5 = FAKE_MD5
        self.batches = []

    def detect_batch(self, paths, batch=32):
        self.batches.append(batch)
        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            dets = [] if "nohand" in path.name else [Detection(w * 0.25, h * 0.25, w * 0.75, h * 0.75, 0.9)]
            yield path, dets, (h, w)


@pytest.fixture
def sources(tmp_path):
    """Fake grassknoted, debashishsau, ayuraj, danrasband and hagrid30k downloads + sources.json."""
    raw = tmp_path / "raw"
    seed = iter(range(10_000))

    gk = raw / "grassknoted"
    gk_train = gk / "asl_alphabet_train" / "asl_alphabet_train"
    for folder in GK_FOLDERS:
        for i in range(3):
            write_img(gk_train / folder / f"{folder}{i}.jpg", next(seed))
    write_img(gk_train / "A" / "A_nohand.jpg", next(seed))  # detector finds nothing -> fallback
    write_img(gk_train / "B" / "B_nohand.jpg", next(seed))
    write_img(gk_train / "nothing" / "nothing_nohand.jpg", next(seed))  # NOTHING without a box -> random square
    gk_test = gk / "asl_alphabet_test" / "asl_alphabet_test"
    for folder in GK_FOLDERS:
        if folder != "del":  # like the real test set: no del
            write_img(gk_test / f"{folder}_test.jpg", next(seed))
    shutil.copy2(gk_test / "C_test.jpg", gk_train / "C" / "C_copy_of_test.jpg")  # leak

    deb = raw / "debashishsau"
    deb_train = deb / "ASL_Alphabet_Dataset" / "asl_alphabet_train"
    for folder in list(string.ascii_uppercase[:20]) + ["Space", "Delete", "Blank"]:
        for i in range(2):
            write_img(deb_train / folder / f"{i}.jpeg", next(seed))
    write_img(deb / "ASL_Alphabet_Dataset" / "asl_alphabet_test" / "A" / "0.jpg", next(seed))  # never read
    shutil.copy2(gk_train / "D" / "D0.jpg", deb_train / "D" / "dup_of_grassknoted.jpeg")  # duplicate
    shutil.copy2(gk_train / "E" / "E1.jpg", deb_train / "F" / "dup_other_label.jpeg")  # duplicate, other label

    ay = raw / "ayuraj"
    ay_root = ay / "asl_dataset"
    for folder in list(string.ascii_lowercase[:22]) + ["0", "1"]:
        write_img(ay_root / folder / f"hand_{folder}.jpeg", next(seed))
        write_img(ay_root / "asl_dataset" / folder / f"hand_{folder}.jpeg", next(seed))  # nested copy

    dan = raw / "danrasband"
    for folder in GK_FOLDERS:
        write_img(dan / folder / f"{folder}0001_test.jpg", next(seed))
        write_img(dan / "asl-alphabet-test" / folder / f"{folder}0001_test.jpg", next(seed))
    shutil.copy2(dan / "G" / "G0001_test.jpg", deb_train / "G" / "leak.jpeg")  # leak into debashishsau

    hg = raw / "hagrid"
    base = hg / "hagrid-sample-30k-384p"
    records = {}
    for i in range(6):
        image_id = f"img{i}"
        write_img(base / "hagrid_30k" / "train_val_call" / f"{image_id}.jpg", next(seed), size=(160, 120))
        records[image_id] = {"bboxes": [[0.1, 0.1, 0.2, 0.3]], "labels": ["call"], "user_id": f"u{i}"}
    (base / "ann_train_val").mkdir(parents=True)
    (base / "ann_train_val" / "call.json").write_text(json.dumps(records))

    entries = {"grassknoted": gk, "debashishsau": deb, "ayuraj": ay, "danrasband": dan, "hagrid30k": hg}
    config.SOURCES_FILE.write_text(json.dumps({k: {"path": str(v), "fetched_at": "x"} for k, v in entries.items()}))
    return entries


def pngs(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.png"))


# --- pure helpers ---------------------------------------------------------------------


def items_for(counts, source="grassknoted"):
    return [
        Item(Path(f"/x/{label}/{i}.jpg"), label, source, f"{label}/{i:03d}.jpg", md5=f"{source}-{label}-{i}")
        for label, n in counts.items()
        for i in range(n)
    ]


def test_stratified_split_covers_every_class_in_both_splits():
    items = items_for({"A": 30, "B": 2, "C": 1, "NOTHING": 7})
    train, val = stratified_split(items, 0.1, 42, cls_data._label_index)
    assert sorted(train + val, key=lambda it: it.rel) == sorted(items, key=lambda it: it.rel)
    val_labels, train_labels = Counter(i.label for i in val), Counter(i.label for i in train)
    assert val_labels == {"A": 3, "B": 1, "NOTHING": 1}
    assert train_labels == {"A": 27, "B": 1, "C": 1, "NOTHING": 6}  # a single image stays in train
    assert stratified_split(items, 0.1, 42, cls_data._label_index) == (train, val)  # deterministic


def test_dedupe_keeps_the_first_source_in_source_order():
    gk = Item(Path("/gk/A/1.jpg"), "A", "grassknoted", "A/1.jpg", md5="same")
    deb = Item(Path("/deb/A/1.jpg"), "A", "debashishsau", "A/1.jpg", md5="same")
    ay = Item(Path("/ay/b/1.jpg"), "B", "ayuraj", "b/1.jpg", md5="same")
    other = Item(Path("/ay/c/1.jpg"), "C", "ayuraj", "c/1.jpg", md5="other")
    unique, removed = dedupe([ay, other, deb, gk])
    assert unique == [gk, other]
    assert removed == {"debashishsau": 1, "ayuraj": 1}


def test_remove_test_leaks_counts_per_source(capsys):
    items = items_for({"A": 10}, "grassknoted") + items_for({"A": 200}, "debashishsau")
    leaked = {items[0].md5, items[1].md5, items[10].md5}
    kept, removed = remove_test_leaks(items, leaked)
    assert removed == {"grassknoted": 2, "debashishsau": 1}
    assert len(kept) == len(items) - 3 and not {i.md5 for i in kept} & leaked
    err = capsys.readouterr().err
    assert "grassknoted: 2/10 training images are copies of test images" in err  # > 1%
    assert "debashishsau" not in err  # 1/200 = 0.5%


def test_cap_is_deterministic_regardless_of_walk_order():
    items = items_for({"A": 50, "B": 3, "NOTHING": 20}) + items_for({"A": 40}, "debashishsau")
    shuffled = list(items)
    random.Random(7).shuffle(shuffled)
    capped = cap_manifest(items, 10)
    assert cap_manifest(shuffled, 10) == capped
    assert cap_manifest(list(reversed(items)), 10) == capped
    groups = Counter((i.source, i.label) for i in capped)
    assert groups == {("grassknoted", "A"): 10, ("grassknoted", "B"): 3, ("grassknoted", "NOTHING"): 10,
                      ("debashishsau", "A"): 10}
    a_rels = [i.rel for i in capped if i.source == "grassknoted" and i.label == "A"]
    assert a_rels == sorted(a_rels)
    assert cap_manifest(items, 10, seed=1) != capped


def test_hand_free_rejects_low_iou_overlaps():
    hand = (100, 100, 120, 120)  # 20x20 hand
    assert cls_data.is_hand_free((0, 0, 50, 50), [hand])
    assert not cls_data.is_hand_free((90, 90, 190, 190), [hand])  # contains the whole hand: IoU only 0.04
    assert not cls_data.is_hand_free((105, 105, 108, 108), [(0, 0, 400, 400)])  # inside a large hand box
    assert cls_data.is_hand_free((0, 0, 50, 50), [])


def test_background_crops_avoid_every_hand(tmp_path):
    records = []
    for i in range(30):
        path = write_img(tmp_path / f"bg{i}.jpg", i, size=(200, 120))
        boxes = ((0.05, 0.1, 0.3, 0.5), (0.6, 0.4, 0.25, 0.4))
        records.append(HagridRecord(path, boxes, f"u{i}", f"id{i}"))
    blocked = write_img(tmp_path / "full.jpg", 99, size=(200, 120))
    records.append(HagridRecord(blocked, ((0.0, 0.0, 1.0, 1.0),), "u", "full"))  # no hand-free square

    planned = plan_background_crops(records, 25, seed=3)
    assert len(planned) == 25
    assert plan_background_crops(records, 25, seed=3) == planned
    assert all(it.label == "NOTHING" and it.source == "hagrid30k" for it in planned)
    assert "full" not in {it.rel for it in planned}
    for item in planned:
        x1, y1, x2, y2 = item.box
        side = x2 - x1
        assert side == y2 - y1 and 0.15 * 120 - 1 <= side <= 0.40 * 120 + 1
        assert 0 <= x1 and x2 <= 200 and 0 <= y1 and y2 <= 120
        hands = [(x * 200, y * 120, (x + w) * 200, (y + h) * 120) for x, y, w, h in
                 next(r for r in records if r.image_id == item.rel).bboxes]
        assert all(iou(item.box, hand) < 0.05 for hand in hands)


# --- full build -------------------------------------------------------------------


def test_build_writes_class_dirs_crops_and_stats(sources, capsys):
    detector = FakeDetector()
    stats = cls_data.build(detector=detector, det_batch=7)
    out, test_root = config.CLS_DIR, config.CLS_TEST_DIR / "test"
    assert set(detector.batches) == {7}

    for split_root in (out / "train", out / "val", test_root):
        assert sorted(p.name for p in split_root.iterdir() if p.is_dir()) == CLASS_DIRS
    for i in range(len(CLASSES)):  # every class present in the training data lands in both splits
        assert stats["split_counts"]["train"][class_dir_name(i)] >= 1
        assert stats["split_counts"]["val"][class_dir_name(i)] >= 1

    assert stats["crop_pad"] == CROP_PAD and stats["cls_imgsz"] == CLS_IMGSZ
    assert stats["preprocess_version"] == PREPROCESS_VERSION
    assert stats["detector"] == {"path": "/fake/models/hand_detector.pt", "md5": FAKE_MD5}
    assert json.loads((out / "stats.json").read_text()) == stats

    # leaks: one grassknoted copy of a grassknoted test file, one debashishsau copy of a danrasband file
    assert stats["test_leak_removed"] == {"grassknoted": 1, "debashishsau": 1, "ayuraj": 0}
    assert not list(out.rglob("grassknoted__C_copy_of_test.png")) and not list(out.rglob("debashishsau__leak.png"))
    # duplicates: the debashishsau copies of grassknoted images lose, whatever their label
    assert stats["dedupe_removed"] == {"grassknoted": 0, "debashishsau": 2, "ayuraj": 0}
    assert list(out.rglob("grassknoted__D0.png")) and not list(out.rglob("debashishsau__dup_*.png"))

    # fallback: the two letter images without a detection; NOTHING without one is a random square
    assert stats["fallback"] == {"grassknoted": 2, "debashishsau": 0, "ayuraj": 0}
    assert stats["crop_kinds"]["grassknoted"]["random"] == 1
    assert stats["nothing_detector_crops"]["grassknoted"] == 3
    assert stats["fallback_rate"]["grassknoted"] == pytest.approx(2 / (26 * 3 + 2 + 2 * 3), abs=1e-4)
    a_src = sources["grassknoted"] / "asl_alphabet_train" / "asl_alphabet_train" / "A" / "A_nohand.jpg"
    img = cv2.imread(str(a_src))
    written = cv2.imread(str(next(out.rglob("grassknoted__A_nohand.png"))))
    np.testing.assert_array_equal(written, classifier_input(img, (0, 0, img.shape[1], img.shape[0])))

    # sources: debashishsau aliases, ayuraj digits / nested copy skipped, HaGRID background crops
    assert stats["counts"]["debashishsau"]["SPACE"] == 2 and stats["counts"]["debashishsau"]["NOTHING"] == 2
    assert sum(stats["counts"]["ayuraj"].values()) == 22
    assert stats["unmapped"]["ayuraj"] == ["0", "1", "asl_dataset"]
    assert stats["nothing_bg"] == 3 and stats["counts"]["hagrid30k"] == {"NOTHING": 3}

    # held-out test set: danrasband (one copy) + the 28 grassknoted test files
    assert stats["n_test"] == 29 + 28 == sum(stats["test_counts"].values())
    assert stats["test_sources"] == {"danrasband": 29, "grassknoted": 28}
    assert stats["test_fallback"] == 0
    assert (test_root / "27_DELETE" / "danrasband__del0001_test.png").is_file()
    assert (test_root / "00_A" / "grassknoted__A_test.png").is_file()

    total = stats["totals"]
    assert total["train"] + total["val"] == len(pngs(out)) and total["test"] == len(pngs(test_root))
    for path in list(out.rglob("*.png"))[:20]:
        crop = cv2.imread(str(path))
        assert crop.shape == (CLS_IMGSZ, CLS_IMGSZ, 3)
        assert (crop[..., 0] == crop[..., 2]).all()
    err = capsys.readouterr().err
    assert "WARNING: ayuraj: skipping unmapped folder '0'" in err
    assert "WARNING: danrasband: skipping unmapped folder 'asl-alphabet-test'" in err


def test_build_is_deterministic_and_rebuilds_from_scratch(sources, tmp_path, monkeypatch):
    first = cls_data.build(detector=FakeDetector())
    files = pngs(config.CLS_DIR)
    data = {f: (config.CLS_DIR / f).read_bytes() for f in files}
    stale = config.CLS_DIR / "train" / "00_A" / "stale.png"
    stale.write_bytes(b"old")

    walk = cls_data._walk_class_root
    monkeypatch.setattr(cls_data, "_walk_class_root", lambda *a: list(reversed(walk(*a))))  # other walk order
    second = cls_data.build(detector=FakeDetector())
    assert not stale.exists()
    assert pngs(config.CLS_DIR) == files
    assert all((config.CLS_DIR / f).read_bytes() == data[f] for f in files)
    assert second["split_counts"] == first["split_counts"]


def test_build_without_detector_or_hagrid(sources, capsys):
    entries = json.loads(config.SOURCES_FILE.read_text())
    del entries["hagrid30k"], entries["debashishsau"]
    config.SOURCES_FILE.write_text(json.dumps(entries))
    stats = cls_data.build(sources=["grassknoted", "debashishsau"], no_detector=True)
    assert stats["detector"] is None
    assert stats["nothing_bg"] == 0 and "hagrid30k" not in stats["counts"]
    assert set(stats["sources"]) == {"grassknoted"}
    assert stats["fallback"]["grassknoted"] == 26 * 3 + 2 + 2 * 3  # every letter/SPACE/DELETE image
    err = capsys.readouterr().err
    assert "debashishsau is not in sources.json" in err
    assert "built with --no-detector" in err


def test_build_errors(sources, tmp_path):
    with pytest.raises(cls_data.ClsDataError, match="train the detector first: python -m src.train_detector"):
        cls_data.build()  # no models/hand_detector.pt under the tmp DET_WEIGHTS
    with pytest.raises(cls_data.ClsDataError, match="must include grassknoted"):
        cls_data.build(sources=["ayuraj"], detector=FakeDetector())
    shutil.rmtree(sources["grassknoted"] / "asl_alphabet_train")
    with pytest.raises(cls_data.ClsDataError, match="grassknoted: class root"):
        cls_data.build(detector=FakeDetector())
    assert not config.CLS_DIR.exists()  # nothing written before the inputs validate


# --- CLI ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [["--sources", "hand_keypoints"], ["--sources", "danrasband"], ["--sources", "debashishsau"],
     ["--max-per-class", "0"], ["--det-batch", "-1"], ["--device", "gpu"]],
)
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        cls_data.main(argv)
    assert exc.value.code == 2


def test_cli_missing_grassknoted_exits_1(capsys):
    assert cls_data.main([]) == 1
    assert "src.download_data --sources grassknoted" in capsys.readouterr().err


def test_cli_missing_detector_exits_1(sources, capsys):
    assert cls_data.main([]) == 1
    assert "train the detector first: python -m src.train_detector" in capsys.readouterr().err
    assert not config.CLS_DIR.exists()
