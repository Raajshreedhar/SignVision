"""Build the YOLO hand-detection dataset data/hand_det/ (design section 4.4).

Detector sources, read from data/sources.json:
  - hand_keypoints (required): Ultralytics hand-keypoints, one hand per image,
    mostly close-ups. Each 68-column pose line (class, box, 21 keypoints x 3)
    becomes the 5-column detection line "0 cx cy w h". The published
    train/val splits are kept.
  - hagrid30k (optional): webcam-style HaGRID frames. Normalized top-left
    xywh boxes become cx cy w h (clipped to the image); train/val is split by
    a sha1 hash of the user id, so one person never lands in both splits.

Output, deleted and rebuilt on every run so stale files never mix in:
  data/hand_det/images/{train,val}/<source>__<stem>.jpg
  data/hand_det/labels/{train,val}/<source>__<stem>.txt
  data/hand_det/hand.yaml    absolute path, names {0: hand}
  data/hand_det/stats.json   per-source counts, webcam_view_source

Run:
    .venv/bin/python -m src.hand_data [--sources hand_keypoints,hagrid30k] [--hagrid-val-frac 0.1]
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.cli import float_range
from src.download_data import REGISTRY, read_sources, sources_arg

HK = "hand_keypoints"
HAGRID = "hagrid30k"
SPLITS = ("train", "val")
POSE_COLUMNS = 5 + 21 * 3  # class, cx, cy, w, h, then 21 keypoints x (x, y, visibility)
MAX_DROPPED_FRAC = 0.01
MIN_HAGRID_BOX_AREA = 1e-4
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DOWNLOAD_CMD = ".venv/bin/python -m src.download_data --sources hand_keypoints"


class HandDataError(Exception):
    """The detector data is not in the expected format; the build (or one source) stops."""


@dataclass(frozen=True)
class HagridRecord:
    image_path: Path
    bboxes: tuple  # normalized (x_tl, y_tl, w, h) tuples
    user_id: str | None
    image_id: str


# --- hand-keypoints ---------------------------------------------------------


def find_hk_root(download_root):
    """First directory (sorted BFS, depth <= 3) holding images/train and labels/train."""
    root = Path(download_root)
    level = [root] if root.is_dir() else []
    for _depth in range(4):
        for folder in level:
            if (folder / "images" / "train").is_dir() and (folder / "labels" / "train").is_dir():
                return folder
        level = sorted(child for folder in level for child in folder.iterdir() if child.is_dir())
    raise HandDataError(f"no images/train + labels/train found under {root} (searched 3 levels deep)")


def convert_pose_line(line):
    """Turn one hand-keypoints pose line into "0 cx cy w h", or None if malformed.

    The line must have exactly 68 numeric columns, class 0 and box values in
    [0, 1]. The keypoint columns are discarded, so only their count is checked:
    their visibility flags are 0/1/2 and a few landmarks sit just outside the
    image, which says nothing about the box.
    """
    parts = line.split()
    if len(parts) != POSE_COLUMNS:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    if values[0] != 0 or not all(0.0 <= v <= 1.0 for v in values[1:5]):
        return None
    return "0 " + " ".join(parts[1:5])


def _read_hk_split(hk_root, split):
    """Convert one split's labels in memory: ([(image, [box lines])], stats)."""
    img_dir = hk_root / "images" / split
    lbl_dir = hk_root / "labels" / split
    if not img_dir.is_dir():
        raise HandDataError(f"hand_keypoints: {img_dir} is missing")
    items = []
    lines = dropped = unlabeled = 0
    for image in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = lbl_dir / f"{image.stem}.txt"
        boxes = []
        if label.is_file():
            for raw in label.read_text().splitlines():
                if not raw.strip():
                    continue
                lines += 1
                converted = convert_pose_line(raw)
                if converted is None:
                    dropped += 1
                else:
                    boxes.append(converted)
        else:
            unlabeled += 1  # kept as a background image with an empty label
        items.append((image, boxes))
    if lines == 0:
        raise HandDataError(f"hand_keypoints {split}: no label lines found under {lbl_dir}")
    if dropped / lines > MAX_DROPPED_FRAC:
        raise HandDataError(
            f"hand_keypoints {split}: dropped {dropped}/{lines} label lines (> {MAX_DROPPED_FRAC:.0%}); "
            f"expected {POSE_COLUMNS}-column pose lines with class 0 and box values in [0, 1]"
        )
    stats = {
        "images": len(items),
        "boxes": sum(len(b) for _, b in items),
        "label_lines": lines,
        "dropped_lines": dropped,
        "unlabeled_images": unlabeled,
    }
    return items, stats


# --- HaGRID -----------------------------------------------------------------


def load_hagrid(root):
    """Read every ann_train_val/*.json record whose image exists under ``root``."""
    root = Path(root)
    ann_files = sorted(root.rglob("ann_train_val/*.json"))
    if not ann_files:
        raise HandDataError(f"no ann_train_val/*.json annotation files under {root}")
    images = {}
    for path in sorted(root.rglob("*.jpg")):
        images.setdefault(path.stem, path)

    records = []
    for ann in ann_files:
        try:
            data = json.loads(ann.read_text())
        except (OSError, ValueError) as exc:
            raise HandDataError(f"{ann}: unreadable annotation file ({exc})") from exc
        if not isinstance(data, dict):
            raise HandDataError(f"{ann}: expected a JSON object keyed by image id")
        for image_id in sorted(data):
            image = images.get(image_id)
            if image is None:
                continue  # annotation for an image that is not in this sample
            rec = data[image_id]
            if not isinstance(rec, dict) or "bboxes" not in rec:
                raise HandDataError(f"{ann.name}: record {image_id!r} has no 'bboxes'")
            if not isinstance(rec["bboxes"], list):
                raise HandDataError(f"{ann.name}: record {image_id!r} 'bboxes' is not a list")
            boxes = []
            for box in rec["bboxes"]:
                try:
                    values = tuple(float(v) for v in box)
                except (TypeError, ValueError):
                    values = ()
                if len(values) != 4 or not all(math.isfinite(v) for v in values):
                    raise HandDataError(f"{ann.name}: record {image_id!r} has a malformed box {box!r}")
                boxes.append(values)
            user_id = rec.get("user_id")
            records.append(HagridRecord(image, tuple(boxes), str(user_id) if user_id else None, image_id))
    if not records:
        raise HandDataError(f"no annotated images found under {root}")
    return records


def hagrid_box_to_yolo(box):
    """Normalized top-left (x, y, w, h) -> (cx, cy, w, h) clipped to the image; None if tiny."""
    x, y, w, h = (float(v) for v in box)
    x1, y1 = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    x2, y2 = min(max(x + w, 0.0), 1.0), min(max(y + h, 0.0), 1.0)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or bw * bh < MIN_HAGRID_BOX_AREA:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, bw, bh)


def hagrid_split(user_id, image_id, val_frac):
    """'val' iff sha1(user_id) % 1000 < 1000 * val_frac (image id when user_id is missing)."""
    key = user_id if user_id else image_id
    bucket = int(hashlib.sha1(str(key).encode("utf-8")).hexdigest(), 16) % 1000
    return "val" if bucket < 1000 * val_frac else "train"


def _write_hagrid(records, val_frac, out_dir):
    stats = {s: {"images": 0, "boxes": 0, "background_images": 0, "dropped_boxes": 0} for s in SPLITS}
    for rec in records:
        split = hagrid_split(rec.user_id, rec.image_id, val_frac)
        lines = []
        for box in rec.bboxes:
            yolo = hagrid_box_to_yolo(box)
            if yolo is None:
                stats[split]["dropped_boxes"] += 1
            else:
                lines.append(_box_line(yolo))
        _write_sample(rec.image_path, lines, out_dir, split, f"{HAGRID}__{rec.image_id}")
        stats[split]["images"] += 1
        stats[split]["boxes"] += len(lines)
        stats[split]["background_images"] += not lines
    return stats


# --- output -----------------------------------------------------------------


def _box_line(box):
    cx, cy, w, h = box
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _write_sample(image, lines, out_dir, split, stem):
    """Copy (not symlink) the image and write its label file, empty for backgrounds."""
    shutil.copy2(image, out_dir / "images" / split / f"{stem}{image.suffix.lower()}")
    (out_dir / "labels" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in lines))


def _write_yaml(out_dir):
    text = (
        "# YOLO hand detection dataset built by src.hand_data\n"
        f"path: {json.dumps(str(out_dir.resolve()))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: hand\n"
    )
    (out_dir / "hand.yaml").write_text(text)


def _totals(out_dir):
    totals = {}
    for split in SPLITS:
        labels = sorted((out_dir / "labels" / split).glob("*.txt"))
        boxes = sum(len([ln for ln in p.read_text().splitlines() if ln.strip()]) for p in labels)
        totals[split] = {"images": len(labels), "boxes": boxes}
    return totals


def build(sources, val_frac, out_dir=None, sources_file=None):
    """Rebuild the detector dataset from the given det sources; return the stats dict."""
    out_dir = Path(out_dir) if out_dir is not None else config.HAND_DET_DIR
    sources_file = Path(sources_file) if sources_file is not None else config.SOURCES_FILE
    paths = read_sources(sources_file)
    if HK not in paths:
        raise HandDataError(f"hand_keypoints is not in {sources_file}; run src.download_data first: {DOWNLOAD_CMD}")

    # Parse and validate everything before the old output is deleted.
    hk_root = find_hk_root(paths[HK])
    hk_items, hk_stats = {}, {}
    for split in SPLITS:
        hk_items[split], hk_stats[split] = _read_hk_split(hk_root, split)

    hagrid_records, hagrid_reason = None, None
    if HAGRID not in paths:
        hagrid_reason = f"not in {sources_file.name}: download skipped or failed"
    elif HAGRID not in sources:
        hagrid_reason = "not selected with --sources"
    else:
        try:
            hagrid_records = load_hagrid(paths[HAGRID])
        except HandDataError as exc:
            print(f"ERROR: hagrid30k aborted: {exc}", file=sys.stderr)
            hagrid_reason = f"schema error: {exc}"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for kind in ("images", "labels"):
        for split in SPLITS:
            (out_dir / kind / split).mkdir(parents=True)

    per_source = {HK: hk_stats}
    for split in SPLITS:
        for image, lines in hk_items[split]:
            _write_sample(image, lines, out_dir, split, f"{HK}__{image.stem}")
    if hagrid_records is not None:
        per_source[HAGRID] = _write_hagrid(hagrid_records, val_frac, out_dir)

    composites = {"train": 0, "val": 0, "bg_nothing": 0, "bg_gradient": 0}

    _write_yaml(out_dir)
    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": per_source,
        "composites": composites,
        "webcam_view_source": hagrid_records is not None,
        "hagrid_status": "included" if hagrid_records is not None else hagrid_reason,
        "hagrid_val_frac": val_frac,
        "seed": config.SEED,
        "totals": _totals(out_dir),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    _print_summary(out_dir, stats)
    return stats


def _print_summary(out_dir, stats):
    print(f"hand detector dataset written to {out_dir}")
    for name, splits in stats["sources"].items():
        for split, s in splits.items():
            extra = {k: v for k, v in s.items() if k not in ("images", "boxes") and v}
            detail = f" ({', '.join(f'{k} {v}' for k, v in extra.items())})" if extra else ""
            print(f"  {name:<15} {split:<5} {s['images']:>6} images {s['boxes']:>6} boxes{detail}")
    c = stats["composites"]
    if c["train"] or c["val"]:
        print(
            f"  {'composites':<15} train {c['train']}, val {c['val']} "
            f"(backgrounds: {c['bg_nothing']} grassknoted nothing, {c['bg_gradient']} gradient)"
        )
    for split, t in stats["totals"].items():
        print(f"  {'total':<15} {split:<5} {t['images']:>6} images {t['boxes']:>6} boxes")
    print(f"  webcam_view_source: {str(stats['webcam_view_source']).lower()} (hagrid30k: {stats['hagrid_status']})")
    print("next: .venv/bin/python -m src.train_detector --model n --time 2.0")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.hand_data",
        description="Build the YOLO hand detection dataset data/hand_det/ from the downloaded det sources.",
    )
    parser.add_argument(
        "--sources",
        type=sources_arg("det"),
        default=None,
        help="comma-separated det sources (default: every det source recorded in data/sources.json)",
    )
    parser.add_argument(
        "--hagrid-val-frac",
        type=float_range(0, 0.5, lo_open=True),
        default=0.1,
        help="fraction of HaGRID users held out for validation, in (0, 0.5] (default 0.1)",
    )
    args = parser.parse_args(argv)

    try:
        recorded = read_sources()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: cannot read {config.SOURCES_FILE} ({exc}); re-run {DOWNLOAD_CMD}", file=sys.stderr)
        return 1
    if HK not in recorded:
        print(
            f"ERROR: hand_keypoints is missing from {config.SOURCES_FILE}; run src.download_data first: "
            f"{DOWNLOAD_CMD}",
            file=sys.stderr,
        )
        return 1
    if args.sources is None:
        sources = [name for name, src in REGISTRY.items() if src.role == "det" and name in recorded]
    else:
        sources = args.sources
    if HK not in sources:
        parser.error("--sources must include hand_keypoints (the required detector source)")

    try:
        build(sources, args.hagrid_val_frac)
    except HandDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
