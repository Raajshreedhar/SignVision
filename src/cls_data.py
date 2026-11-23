"""Build the letter-classifier datasets data/cls/ and data/cls_test/ (design section 4.5).

Every image goes through the trained hand detector and the shared
preprocess.classifier_input, so training crops look exactly like live crops:
  1. test manifest first: danrasband and the grassknoted *_test.jpg files,
     md5-hashed;
  2. training manifest: the class folders of grassknoted, debashishsau and
     ayuraj (class_map maps the folder names; digits and unknown folders are
     skipped with a WARNING);
  3. training images whose md5 matches a test image are dropped (test-leak
     removal);
  4. md5 duplicates are dropped, first occurrence wins in SOURCE_ORDER;
  5. at most --max-per-class images per (source, class), drawn
     deterministically with one random.Random(SEED) from the sorted manifest;
  6. crops: the best detector box (tracking.best_detection); a letter, SPACE
     or DELETE image without a box falls back to the centered short-side
     square (counted per source as "fallback"); a NOTHING image uses the
     detector crop when the detector fired (a false positive the classifier
     must reject), else a seeded random square of 50-100% of the short side;
  7. NOTHING_BG_CROPS HaGRID background squares (15-40% of the short side)
     are added to NOTHING; each overlaps every annotated hand by less than
     5% of the smaller box, which implies IoU < 0.05;
  8. a stratified CLS_VAL_SPLIT split by class (at least one image per split
     per class), crops written as PNG to
     data/cls/{train,val}/<class_dir_name>/<source>__<stem>.png;
  9. the test manifest goes the same way into data/cls_test/test/;
 10. data/cls/stats.json records per-source counts, fallback, test_fallback,
     n_test, dedupe and leak removals, nothing_bg, the detector md5,
     preprocess_version, crop_pad and cls_imgsz.
Both output trees are deleted and rebuilt on every run.

Run:
    .venv/bin/python -m src.cls_data [--sources grassknoted,debashishsau,ayuraj]
           [--max-per-class 1500] [--det-batch 32] [--device auto] [--no-detector]
"""

import argparse
import hashlib
import itertools
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src import config
from src.class_map import (
    DatasetLayoutError,
    canonical_label,
    class_dir_name,
    label_from_filename,
    resolve_class_root,
)
from src.cli import DEVICE_CHOICES, int_range
from src.config import (
    CLASSES,
    CLS_IMGSZ,
    CLS_MAX_PER_CLASS_PER_SOURCE,
    CLS_VAL_SPLIT,
    CROP_PAD,
    NOTHING_BG_CROPS,
    NUM_CLASSES,
    PREPROCESS_VERSION,
    SEED,
)
from src.download_data import REGISTRY, read_sources, sources_arg
from src.preprocess import classifier_input
from src.tracking import best_detection

SOURCE_ORDER = ("grassknoted", "debashishsau", "ayuraj")
REQUIRED = "grassknoted"
TEST_SOURCE = "danrasband"
HAGRID = "hagrid30k"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FALLBACK_WARN_RATE = 0.30
LEAK_WARN_RATE = 0.01
NOTHING_SCALE = (0.5, 1.0)  # random NOTHING square side as a fraction of the short side
BG_SCALE = (0.15, 0.40)  # HaGRID background square side as a fraction of the short side
BG_MAX_IOU = 0.05
BG_TRIES = 20  # random squares tried per HaGRID image
PROGRESS_EVERY = 5000
DOWNLOAD_CMD = ".venv/bin/python -m src.download_data"
DETECTOR_HINT = "train the detector first: python -m src.train_detector"


class ClsDataError(Exception):
    """A required input is missing or unusable; the build stops before touching the output."""


@dataclass(frozen=True)
class Item:
    path: Path
    label: str
    source: str
    rel: str  # path relative to the source's class root: the deterministic sort key
    md5: str = ""
    box: tuple | None = None  # preset square (HaGRID background crops), cropped without padding


def _md5(path):
    return hashlib.md5(Path(path).read_bytes(), usedforsecurity=False).hexdigest()


def _label_index(item):
    return CLASSES.index(item.label)


def _manifest_key(item):
    return (SOURCE_ORDER.index(item.source), CLASSES.index(item.label), item.rel)


def _warn(message):
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


# --- manifests --------------------------------------------------------------------


def _walk_class_root(source, root, unmapped):
    """Items for every image in the mapped class folders directly under ``root``."""
    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        label = canonical_label(folder.name)
        if label is None:
            if folder.name not in unmapped[source]:
                unmapped[source].add(folder.name)
                _warn(f"{source}: skipping unmapped folder '{folder.name}'")
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                items.append(Item(path, label, source, path.relative_to(root).as_posix()))
    return items


def _hash_items(items, read_failures):
    """The items with their md5 filled in; unreadable files are counted and left out."""
    hashed, failed = [], Counter()
    for item in items:
        try:
            hashed.append(Item(item.path, item.label, item.source, item.rel, _md5(item.path), item.box))
        except OSError:
            failed[item.source] += 1
    for source, n in sorted(failed.items()):
        read_failures[source] += n
        _warn(f"{source}: {n} unreadable files skipped")
    return hashed


def held_out_manifest(paths, unmapped):
    """danrasband (via its class root) plus every registry test_glob match, unhashed."""
    items = []
    if TEST_SOURCE in paths:
        try:
            root = resolve_class_root(paths[TEST_SOURCE], REGISTRY[TEST_SOURCE].class_root)
            items += _walk_class_root(TEST_SOURCE, root, unmapped)
        except DatasetLayoutError as exc:
            _warn(f"{TEST_SOURCE}: {exc}; held-out test set built without it")
    else:
        _warn(f"{TEST_SOURCE} is not in sources.json; held-out test set built without it")
    for name, src in REGISTRY.items():
        if src.test_glob is None or name not in paths:
            continue
        root = Path(paths[name])
        files = sorted(p for p in root.glob(src.test_glob) if p.is_file())
        if not files:
            _warn(f"{name}: test_glob matched no files")
        for path in files:
            label = label_from_filename(path.name)
            if label is None:
                if path.name not in unmapped[name]:
                    unmapped[name].add(path.name)
                    _warn(f"{name}: skipping unmapped test file '{path.name}'")
                continue
            items.append(Item(path, label, name, path.relative_to(root).as_posix()))
    return items


def remove_test_leaks(items, test_hashes):
    """(kept, removed per source): drop training items whose md5 is a test image's md5."""
    kept, removed, totals = [], Counter(), Counter()
    for item in items:
        totals[item.source] += 1
        if item.md5 in test_hashes:
            removed[item.source] += 1
        else:
            kept.append(item)
    for source, n in sorted(removed.items()):
        if n > LEAK_WARN_RATE * totals[source]:
            _warn(f"{source}: {n}/{totals[source]} training images are copies of test images (removed)")
    return kept, removed


def dedupe(items):
    """(unique, removed per source): first md5 occurrence wins in (SOURCE_ORDER, class, path) order."""
    seen, unique, removed = set(), [], Counter()
    for item in sorted(items, key=_manifest_key):
        if item.md5 in seen:
            removed[item.source] += 1
            continue
        seen.add(item.md5)
        unique.append(item)
    return unique, removed


def cap_manifest(items, max_per_class, seed=SEED):
    """At most max_per_class items per (source, class), independent of the input order.

    The manifest is sorted by (SOURCE_ORDER, class, relative path) and the
    groups are sampled in that order with one random.Random(seed); each
    sample is sorted by relative path.
    """
    rng = random.Random(seed)
    capped = []
    for _, group in itertools.groupby(sorted(items, key=_manifest_key), key=lambda it: (it.source, it.label)):
        group = list(group)
        chosen = rng.sample(group, min(max_per_class, len(group)))
        capped.extend(sorted(chosen, key=lambda it: it.rel))
    return capped


def stratified_split(items, val_split, seed, label_of):
    """(train, val) split per class: every class with >= 2 items lands in both splits.

    Ported from the legacy src.dataset.stratified_split: groups in sorted
    label order, each shuffled with one seeded random.Random, round(n *
    val_split) items (clamped to 1..n-1) go to val.
    """
    by_class = {}
    for item in items:
        by_class.setdefault(label_of(item), []).append(item)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(by_class):
        group = list(by_class[key])
        rng.shuffle(group)
        n_val = int(round(len(group) * val_split))
        if len(group) >= 2:
            n_val = max(1, min(n_val, len(group) - 1))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


# --- crops ------------------------------------------------------------------------


def random_square(rng, height, width, scale):
    """A random in-frame square whose side is uniform(*scale) of the short side."""
    short = min(height, width)
    side = max(1, min(short, int(round(rng.uniform(*scale) * short))))
    x1, y1 = rng.randint(0, width - side), rng.randint(0, height - side)
    return (x1, y1, x1 + side, y1 + side)


def is_hand_free(square, hands):
    """True when ``square`` overlaps every hand box by less than BG_MAX_IOU of the smaller box.

    That implies IoU < BG_MAX_IOU, and it also rejects what a low IoU alone
    lets through: a large square around a whole small hand, or a small square
    inside a large hand box.
    """
    x1, y1, x2, y2 = square
    square_area = (x2 - x1) * (y2 - y1)
    for hx1, hy1, hx2, hy2 in hands:
        inter = max(0.0, min(x2, hx2) - max(x1, hx1)) * max(0.0, min(y2, hy2) - max(y1, hy1))
        hand_area = max(0.0, hx2 - hx1) * max(0.0, hy2 - hy1)
        if inter > 0 and inter >= BG_MAX_IOU * min(square_area, hand_area):
            return False
    return True


def plan_background_crops(records, n, seed=SEED):
    """Up to n NOTHING items: one random hand-free square (is_hand_free) per HaGRID image."""
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    planned = []
    for rec in order:
        if len(planned) >= n:
            break
        image = cv2.imread(str(rec.image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        hands = [(x * width, y * height, (x + w) * width, (y + h) * height) for x, y, w, h in rec.bboxes]
        for _ in range(BG_TRIES):
            square = random_square(rng, height, width, BG_SCALE)
            if is_hand_free(square, hands):
                planned.append(Item(Path(rec.image_path), "NOTHING", HAGRID, str(rec.image_id), box=square))
                break
    return planned


def crop_item(item, image, dets, rng):
    """(crop, kind) for one image.

    kind: "detector" / "fallback" for a letter, SPACE or DELETE image with /
    without a detector box; "nothing_detector" / "random" for a NOTHING image
    with / without one; "background" for a planned HaGRID square.
    """
    height, width = image.shape[:2]
    if item.box is not None:
        return classifier_input(image, item.box, pad=0.0), "background"
    best = best_detection(dets, (height, width)) if dets else None
    if item.label == "NOTHING":
        if best is not None:
            return classifier_input(image, best.xyxy), "nothing_detector"
        return classifier_input(image, random_square(rng, height, width, NOTHING_SCALE), pad=0.0), "random"
    if best is not None:
        return classifier_input(image, best.xyxy), "detector"
    return classifier_input(image, (0, 0, width, height)), "fallback"


def _with_detections(items, detector, det_batch):
    """Yield (item, detections) in input order; detections is None when the image was unreadable."""
    if detector is None:
        for item in items:
            yield item, []
        return
    pending = iter(items)
    for path, dets, _hw in detector.detect_batch([item.path for item in items], batch=det_batch):
        for item in pending:
            if item.path == path:
                yield item, dets
                break
            yield item, None  # skipped by detect_batch: undecodable
    for item in pending:
        yield item, None


class _Writer:
    """Writes the crops as PNG and prints progress every PROGRESS_EVERY images."""

    def __init__(self, total):
        self.total = total
        self.done = 0
        self.fallback = 0
        self.unreadable = 0
        self.used = set()
        self.started = time.time()

    def write(self, folder, item, crop):
        stem = f"{item.source}__{Path(item.rel).stem}"
        target = folder / f"{stem}.png"
        if target in self.used:  # the same stem twice in one class folder (e.g. a.jpg and a.png)
            target = folder / f"{stem}__{item.md5[:8]}.png"
        self.used.add(target)
        if not cv2.imwrite(str(target), crop):
            raise ClsDataError(f"failed to write {target}")

    def tick(self):
        self.done += 1
        if self.done % PROGRESS_EVERY == 0 or self.done == self.total:
            print(
                f"[cls_data] {self.done}/{self.total} images cropped ({self.fallback} fallback, "
                f"{self.unreadable} unreadable) {time.time() - self.started:.0f} s",
                flush=True,
            )


def _write_crops(items, detector, det_batch, folder_of, writer, rng):
    """Crop and write items; return (kinds, counts, unreadable), each keyed by source."""
    kinds, counts, unreadable = defaultdict(Counter), defaultdict(Counter), Counter()
    for item, dets in _with_detections(items, detector, det_batch):
        image = None if dets is None else cv2.imread(str(item.path))
        if image is None:
            unreadable[item.source] += 1
            writer.unreadable += 1
        else:
            crop, kind = crop_item(item, image, dets, rng)
            writer.write(folder_of(item), item, crop)
            kinds[item.source][kind] += 1
            counts[item.source][item.label] += 1
            writer.fallback += kind == "fallback"
        writer.tick()
    return kinds, counts, unreadable


def _count_pngs(root):
    return {class_dir_name(i): sum(1 for _ in (root / class_dir_name(i)).glob("*.png")) for i in range(NUM_CLASSES)}


def _fallback_rate(kinds):
    hands = kinds["detector"] + kinds["fallback"]
    return round(kinds["fallback"] / hands, 4) if hands else 0.0


def _ordered(counter):
    """A label Counter as a dict in canonical class order."""
    return {label: counter[label] for label in CLASSES if counter[label]}


# --- build ------------------------------------------------------------------------


def build(
    sources=None,
    max_per_class=CLS_MAX_PER_CLASS_PER_SOURCE,
    det_batch=32,
    device="auto",
    no_detector=False,
    detector=None,
    sources_file=None,
    out_dir=None,
    test_out_dir=None,
):
    """Rebuild data/cls and data/cls_test; return the stats dict.

    ``detector`` (anything with weights_path, md5 and detect_batch) defaults
    to HandDetector(config.DET_WEIGHTS) unless no_detector is set.
    """
    started = time.time()
    out_dir = Path(out_dir) if out_dir is not None else config.CLS_DIR
    test_out_dir = Path(test_out_dir) if test_out_dir is not None else config.CLS_TEST_DIR
    sources_file = Path(sources_file) if sources_file is not None else config.SOURCES_FILE
    paths = read_sources(sources_file)
    if REQUIRED not in paths:
        raise ClsDataError(f"grassknoted is not in {sources_file}; run {DOWNLOAD_CMD} --sources grassknoted")
    requested = list(SOURCE_ORDER) if sources is None else list(sources)
    if REQUIRED not in requested:
        raise ClsDataError("the sources must include grassknoted (the required classifier source)")

    if no_detector:
        detector = None
    elif detector is None:
        if not config.DET_WEIGHTS.is_file():
            raise ClsDataError(f"no hand detector at {config.DET_WEIGHTS}; {DETECTOR_HINT}")
        from src.pipeline import HandDetector
        from src.yolo_env import resolve_device

        detector = HandDetector(config.DET_WEIGHTS, resolve_device(device))
    det_info = None if detector is None else {"path": str(detector.weights_path), "md5": detector.md5}

    unmapped = defaultdict(set)
    read_failures = Counter()
    roots = {}
    for name in SOURCE_ORDER:
        if name not in requested:
            continue
        if name not in paths:
            _warn(f"{name} is not in {sources_file.name} (download skipped or failed); skipping it")
            continue
        try:
            roots[name] = resolve_class_root(paths[name], REGISTRY[name].class_root)
        except DatasetLayoutError as exc:
            if name == REQUIRED:
                raise ClsDataError(f"grassknoted: {exc}") from exc
            _warn(f"{name}: {exc}; skipping it")

    # 1. test manifest, hashed first
    tests = _hash_items(held_out_manifest(paths, unmapped), read_failures)
    test_hashes = {item.md5 for item in tests}
    print(f"test manifest: {len(tests)} images from {sorted({i.source for i in tests})}", flush=True)

    # 2. training manifest
    manifest = []
    for name, root in roots.items():
        t0 = time.time()
        items = _hash_items(_walk_class_root(name, root, unmapped), read_failures)
        manifest += items
        print(f"training manifest: {name} {len(items)} images hashed in {time.time() - t0:.0f} s ({root})", flush=True)

    # 3-5. leak removal, dedupe, cap
    manifest, leak_removed = remove_test_leaks(manifest, test_hashes)
    manifest, dedupe_removed = dedupe(manifest)
    available = defaultdict(Counter)
    for item in manifest:
        available[item.source][item.label] += 1
    capped = cap_manifest(manifest, max_per_class)

    # 7 (planned before the split so the background crops are split with NOTHING)
    background, hagrid_status = [], "not in sources.json"
    if HAGRID in paths:
        from src.hand_data import HandDataError, load_hagrid

        try:
            background = plan_background_crops(load_hagrid(paths[HAGRID]), NOTHING_BG_CROPS)
            hagrid_status = "included"
        except HandDataError as exc:
            hagrid_status = f"schema error: {exc}"
            _warn(f"hagrid30k background crops skipped: {exc}")
    else:
        _warn("hagrid30k is not in sources.json; NOTHING gets no background crops (nothing_bg 0)")

    # 8. split
    _, val_items = stratified_split(capped + background, CLS_VAL_SPLIT, SEED, _label_index)
    val_paths = {item.path for item in val_items}

    for root in (out_dir, test_out_dir):
        if root.exists():
            shutil.rmtree(root)
    for split in ("train", "val"):
        for i in range(NUM_CLASSES):
            (out_dir / split / class_dir_name(i)).mkdir(parents=True)
    test_dir = test_out_dir / "test"
    for i in range(NUM_CLASSES):
        (test_dir / class_dir_name(i)).mkdir(parents=True)

    def split_folder(item):
        split = "val" if item.path in val_paths else "train"
        return out_dir / split / class_dir_name(CLASSES.index(item.label))

    def test_folder(item):
        return test_dir / class_dir_name(CLASSES.index(item.label))

    # 6 + 8 + 9: detector crops, HaGRID background crops, held-out test crops
    writer = _Writer(len(capped) + len(background) + len(tests))
    print(f"cropping {writer.total} images (detector: {det_info['path'] if det_info else 'none'})", flush=True)
    kinds, counts, unreadable = _write_crops(capped, detector, det_batch, split_folder, writer, random.Random(SEED))
    bg_kinds, bg_counts, bg_unreadable = _write_crops(
        background, None, det_batch, split_folder, writer, random.Random(SEED)
    )
    test_kinds, test_src_counts, test_unreadable = _write_crops(
        tests, detector, det_batch, test_folder, writer, random.Random(SEED)
    )
    for counter in (unreadable, bg_unreadable, test_unreadable):
        read_failures.update(counter)
    counts.update(bg_counts)

    test_hand = sum(k["detector"] + k["fallback"] for k in test_kinds.values())
    test_fallback = sum(k["fallback"] for k in test_kinds.values())
    split_counts = {split: _count_pngs(out_dir / split) for split in ("train", "val")}
    test_counts = _count_pngs(test_dir)
    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "preprocess_version": PREPROCESS_VERSION,
        "crop_pad": CROP_PAD,
        "cls_imgsz": CLS_IMGSZ,
        "detector": det_info,
        "seed": SEED,
        "max_per_class": max_per_class,
        "val_split": CLS_VAL_SPLIT,
        "det_batch": det_batch,
        "sources": {name: str(root) for name, root in roots.items()},
        "counts": {source: _ordered(counts[source]) for source in [*roots, HAGRID] if counts[source]},
        "available": {source: _ordered(available[source]) for source in roots},
        "crop_kinds": {source: dict(k) for source, k in {**kinds, **bg_kinds}.items()},
        "fallback": {source: kinds[source]["fallback"] for source in roots},
        "fallback_rate": {source: _fallback_rate(kinds[source]) for source in roots},
        "nothing_detector_crops": {source: kinds[source]["nothing_detector"] for source in roots},
        "test_fallback": test_fallback,
        "test_fallback_rate": round(test_fallback / test_hand, 4) if test_hand else 0.0,
        "n_test": sum(test_counts.values()),
        "test_sources": {source: sum(c.values()) for source, c in test_src_counts.items()},
        "test_kinds": {source: dict(k) for source, k in test_kinds.items()},
        "test_counts": test_counts,
        "dedupe_removed": {source: dedupe_removed[source] for source in roots},
        "test_leak_removed": {source: leak_removed[source] for source in roots},
        "nothing_bg": sum(bg_counts[HAGRID].values()),
        "hagrid_status": hagrid_status,
        "read_failures": dict(read_failures),
        "unmapped": {source: sorted(names) for source, names in unmapped.items()},
        "split_counts": split_counts,
        "totals": {
            "train": sum(split_counts["train"].values()),
            "val": sum(split_counts["val"].values()),
            "test": sum(test_counts.values()),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    _print_summary(stats, out_dir, test_dir, no_detector=detector is None)
    return stats


def _print_summary(stats, out_dir, test_dir, no_detector):
    t = stats["totals"]
    print(f"classifier dataset written to {out_dir} (train {t['train']}, val {t['val']}) and {test_dir} ({t['test']})")
    for source in stats["sources"]:
        k = stats["crop_kinds"].get(source, {})
        print(
            f"  {source:<13} {sum(stats['counts'].get(source, {}).values()):>6} crops: "
            f"{k.get('detector', 0)} detector, {k.get('fallback', 0)} fallback "
            f"({stats['fallback_rate'][source]:.1%}), NOTHING {k.get('nothing_detector', 0)} detector + "
            f"{k.get('random', 0)} random; removed {stats['test_leak_removed'][source]} test leaks, "
            f"{stats['dedupe_removed'][source]} duplicates"
        )
    print(f"  {HAGRID:<13} {stats['nothing_bg']:>6} NOTHING background crops ({stats['hagrid_status']})")
    print(
        f"  {'test':<13} {stats['n_test']:>6} crops {stats['test_sources']}: "
        f"{stats['test_fallback']} fallback ({stats['test_fallback_rate']:.1%})"
    )
    per_class = ", ".join(
        f"{CLASSES[i]} {stats['split_counts']['train'][d]}/{stats['split_counts']['val'][d]}"
        for i, d in enumerate(class_dir_name(j) for j in range(NUM_CLASSES))
    )
    print(f"  train/val per class: {per_class}")
    if stats["read_failures"]:
        print(f"  unreadable images skipped: {stats['read_failures']}")
    detector = stats["detector"]
    print(f"  detector: {detector['path']} (md5 {detector['md5']})" if detector else "  detector: none (--no-detector)")
    print(f"  duration: {stats['duration_s']:.0f} s")

    if no_detector:
        _warn("built with --no-detector: every hand image is an uncropped center square (detector: null)")
    else:
        for source, rate in stats["fallback_rate"].items():
            if rate > FALLBACK_WARN_RATE:
                _warn(
                    f"{source}: the detector found no hand in {rate:.0%} of its images "
                    f"(> {FALLBACK_WARN_RATE:.0%}); those use the uncropped center square"
                )
    for split in ("train", "val"):
        empty = [d for d, n in stats["split_counts"][split].items() if n == 0]
        if empty:
            _warn(f"{split} has no images for {', '.join(empty)}")
    print("next: .venv/bin/python -m src.train_classifier --model n --time 1.5")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.cls_data",
        description="Build the detector-cropped classifier datasets data/cls/ and data/cls_test/.",
    )
    parser.add_argument(
        "--sources",
        type=sources_arg("cls_train"),
        default=None,
        help="comma-separated cls_train sources (default: every one recorded in data/sources.json; "
        "must include grassknoted)",
    )
    parser.add_argument(
        "--max-per-class", type=int_range(1), default=CLS_MAX_PER_CLASS_PER_SOURCE,
        help=f"images kept per source per class (default {CLS_MAX_PER_CLASS_PER_SOURCE})",
    )
    parser.add_argument("--det-batch", type=int_range(1), default=32, help="detector batch size (default 32)")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="auto = mps when available")
    parser.add_argument(
        "--no-detector", action="store_true",
        help="degraded mode: uncropped center squares instead of detector crops (stats detector: null)",
    )
    args = parser.parse_args(argv)
    if args.sources is not None and REQUIRED not in args.sources:
        parser.error("--sources must include grassknoted (the required classifier source)")

    try:
        recorded = read_sources()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: cannot read {config.SOURCES_FILE} ({exc}); re-run {DOWNLOAD_CMD}", file=sys.stderr)
        return 1
    if REQUIRED not in recorded:
        print(
            f"ERROR: grassknoted is missing from {config.SOURCES_FILE}; run {DOWNLOAD_CMD} --sources grassknoted",
            file=sys.stderr,
        )
        return 1
    if not args.no_detector and not config.DET_WEIGHTS.is_file():
        print(f"ERROR: no hand detector at {config.DET_WEIGHTS}; {DETECTOR_HINT}", file=sys.stderr)
        return 1

    try:
        build(
            sources=args.sources, max_per_class=args.max_per_class, det_batch=args.det_batch,
            device=args.device, no_detector=args.no_detector,
        )
    except ClsDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
