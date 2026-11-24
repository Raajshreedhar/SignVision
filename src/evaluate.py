"""Offline evaluation of the trained models (design section 12, "Offline evaluation").

    .venv/bin/python -m src.evaluate [--split val|test|all] [--detector] [--device auto|mps|cpu]

eval_classifier runs the letter classifier over a folder tree of crops that
src.cls_data already preprocessed, laid out as <split_dir>/<class_dir_name>/
(data/cls/val, data/cls_test/test), and reports:
  - top-1 and top-5 accuracy;
  - per-class accuracy and a 29x29 confusion matrix (rows = true class,
    columns = predicted class, canonical config.CLASSES order);
  - the 5 most-confused class pairs and the 5 weakest classes.
The report is written to <reports_dir>/classifier_<split>.json with a heatmap
in confusion_<split>.png (default reports_dir: models/reports/).

eval_detector validates the hand detector with Ultralytics on the val split of
data/hand_det/hand.yaml (mAP50, mAP50-95, precision, recall), and once per
source on that source's val images. src.hand_data names every image
<source>__<stem>.jpg (hand_keypoints, hagrid30k or composite), and each subset
goes through a temporary list-file yaml in the reports folder that is deleted
afterwards. <reports_dir>/detector.json also records the cls_test detection
rate (1 - test_fallback / n_test from data/cls/stats.json) and
webcam_view_source from data/hand_det/stats.json. The Ultralytics val outputs
go to <reports_dir>/detector_val*/.

The CLI evaluates the classifier on data/cls/val and data/cls_test/test
(--split, default all) and, with --detector, the hand detector. Missing
weights or data exit 1 with the command that creates them.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.class_map import class_dir_name
from src.cli import DEVICE_CHOICES
from src.config import CLASSES, NUM_CLASSES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TOP_K = 5
N_REPORTED = 5  # most-confused pairs and weakest classes listed in a report
DET_SOURCES = ("hand_keypoints", "hagrid30k", "composite")  # <source>__ prefixes written by src.hand_data
DET_METRICS = ("map50", "map50_95", "precision", "recall")
HAND_DATA_CMD = ".venv/bin/python -m src.hand_data"
CLS_DATA_CMD = ".venv/bin/python -m src.cls_data"
TRAIN_DETECTOR_CMD = ".venv/bin/python -m src.hand_data && .venv/bin/python -m src.train_detector"
TRAIN_CLASSIFIER_CMD = ".venv/bin/python -m src.cls_data && .venv/bin/python -m src.train_classifier"
RETRAIN_CLASSIFIER_CMD = ".venv/bin/python -m src.train_classifier"


def _class_images(split_dir, index):
    folder = Path(split_dir) / class_dir_name(index)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def summarize(confusion, top5_hits):
    """Accuracy figures from a 29x29 confusion matrix and the number of top-5 hits."""
    confusion = np.asarray(confusion, dtype=np.int64)
    n = int(confusion.sum())
    per_class_n = confusion.sum(axis=1)
    per_class = {
        CLASSES[i]: (float(confusion[i, i] / per_class_n[i]) if per_class_n[i] else None) for i in range(NUM_CLASSES)
    }
    weakest = sorted(
        (i for i in range(NUM_CLASSES) if per_class_n[i]), key=lambda i: (confusion[i, i] / per_class_n[i], i)
    )[:N_REPORTED]
    pairs = sorted(
        ((int(confusion[t, p]), t, p) for t in range(NUM_CLASSES) for p in range(NUM_CLASSES)
         if t != p and confusion[t, p] > 0),
        key=lambda item: (-item[0], item[1], item[2]),
    )[:N_REPORTED]
    return {
        "n_images": n,
        "top1": float(np.trace(confusion) / n) if n else None,
        "top5": float(top5_hits / n) if n else None,
        "per_class": per_class,
        "per_class_n": {CLASSES[i]: int(per_class_n[i]) for i in range(NUM_CLASSES)},
        "most_confused": [
            {"true": CLASSES[t], "pred": CLASSES[p], "count": count, "rate": count / int(per_class_n[t])}
            for count, t, p in pairs
        ],
        "weakest_classes": [
            {"class": CLASSES[i], "accuracy": float(confusion[i, i] / per_class_n[i]), "n": int(per_class_n[i])}
            for i in weakest
        ],
    }


def _plot_confusion(confusion, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confusion = np.asarray(confusion, dtype=np.float64)
    rows = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, rows, out=np.zeros_like(confusion), where=rows > 0)
    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="fraction of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def eval_classifier(weights, meta, split_dir, device, classifier=None, *, split_name=None, reports_dir=None):
    """Evaluate the letter classifier on <split_dir>/<class_dir_name>/ crops; return the report dict.

    ``classifier`` (anything with classify(img) -> (29,) probabilities)
    defaults to LetterClassifier(weights, meta, device), which also validates
    the saved meta. The crops are already preprocessed, so they are passed to
    classify unchanged.
    """
    split_dir = Path(split_dir)
    split = split_name or split_dir.name
    reports_dir = Path(reports_dir) if reports_dir is not None else config.REPORTS_DIR
    if classifier is None:
        from src.pipeline import LetterClassifier

        classifier = LetterClassifier(weights, meta, device)

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    top5_hits = 0
    read_failures = 0
    for true_idx in range(NUM_CLASSES):
        for path in _class_images(split_dir, true_idx):
            image = cv2.imread(str(path))
            if image is None:
                read_failures += 1
                continue
            probs = np.asarray(classifier.classify(image), dtype=np.float64)
            if probs.shape != (NUM_CLASSES,):
                raise ValueError(f"classifier returned shape {probs.shape} for {path}, expected ({NUM_CLASSES},)")
            ranked = np.argsort(-probs, kind="stable")
            confusion[true_idx, ranked[0]] += 1
            top5_hits += int(true_idx in ranked[:TOP_K])

    json_path = reports_dir / f"classifier_{split}.json"
    png_path = reports_dir / f"confusion_{split}.png"
    report = {
        "split": split,
        "split_dir": str(split_dir),
        "weights": None if weights is None else str(weights),
        "meta": None if meta is None else str(meta),
        "device": device,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": list(CLASSES),
        **summarize(confusion, top5_hits),
        "read_failures": read_failures,
        "confusion": confusion.tolist(),
        "confusion_png": str(png_path),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    top1 = "n/a" if report["top1"] is None else f"{report['top1']:.4f}"
    top5 = "n/a" if report["top5"] is None else f"{report['top5']:.4f}"
    _plot_confusion(confusion, f"letter classifier, {split}: top-1 {top1} ({report['n_images']} images)", png_path)

    print(f"classifier {split}: top-1 {top1}  top-5 {top5}  ({report['n_images']} images) -> {json_path}")
    if report["n_images"] == 0:
        print(f"WARNING: no images found under {split_dir}/<NN_LABEL>/")
    if read_failures:
        print(f"WARNING: {read_failures} unreadable images skipped in {split_dir}")
    weakest = ", ".join(f"{w['class']} {w['accuracy']:.2f} (n={w['n']})" for w in report["weakest_classes"])
    confused = ", ".join(f"{c['true']}->{c['pred']} {c['count']}" for c in report["most_confused"])
    print(f"  weakest classes: {weakest or 'none'}")
    print(f"  most confused (true->pred count): {confused or 'none'}")
    return report


# --- hand detector --------------------------------------------------------------


def _read_json(path):
    """The JSON object stored in ``path``, or None when it is missing or unreadable."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def cls_test_detection_rate(stats):
    """1 - test_fallback / n_test from data/cls/stats.json, or None when the counts are absent.

    test_fallback counts the held-out letter, SPACE and DELETE images in which
    the detector found no hand, so src.cls_data cropped a center square instead.
    """
    if not stats:
        return None
    n_test, fallback = stats.get("n_test"), stats.get("test_fallback")
    if not isinstance(n_test, int) or not isinstance(fallback, int) or n_test <= 0:
        return None
    return 1.0 - fallback / n_test


def det_source(path):
    """Dataset source of a detector image, from its '<source>__<stem>' file name ('unknown' without one)."""
    name = Path(path).name
    return name.split("__", 1)[0] if "__" in name else "unknown"


def _val_images(data_yaml):
    """(dataset root, names, sorted val image paths) of a detection yaml whose val entry is a folder."""
    import yaml

    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text()) or {}
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val_dir = root / data["val"]
    if not val_dir.is_dir():
        raise FileNotFoundError(f"{val_dir} (the val split of {data_yaml}) is not a folder")
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return root, data.get("names") or {0: "hand"}, images


def _by_source(images):
    """{source: [paths]} with the DET_SOURCES first, in that order, then any other prefix alphabetically."""
    groups = {}
    for path in images:
        groups.setdefault(det_source(path), []).append(path)
    order = [s for s in DET_SOURCES if s in groups] + sorted(s for s in groups if s not in DET_SOURCES)
    return {source: groups[source] for source in order}


def _write_subset(reports_dir, source, root, names, images):
    """Write the temporary list file and yaml for one source's val images; return (yaml_path, list_path)."""
    import yaml

    list_path = reports_dir / f".detector_val_{source}.txt"
    yaml_path = reports_dir / f".detector_val_{source}.yaml"
    list_path.write_text("".join(f"{p}\n" for p in images))
    subset = {"path": str(root), "train": str(list_path), "val": str(list_path), "names": names}
    yaml_path.write_text(yaml.safe_dump(subset, sort_keys=False))
    return yaml_path, list_path


def _ultralytics_val(weights, imgsz, device, reports_dir):
    """(val_fn, device): Ultralytics val of ``weights`` into <reports_dir>/<name>/ on the resolved device."""
    from src import yolo_env  # torch + ultralytics, only when a real evaluation runs

    model = yolo_env.load_yolo(weights)
    device = yolo_env.resolve_device(device)

    def val_fn(data_yaml, name):
        m = model.val(data=str(data_yaml), imgsz=imgsz, device=device, project=str(reports_dir), name=name,
                      exist_ok=True)
        return {"map50": m.box.map50, "map50_95": m.box.map, "precision": m.box.mp, "recall": m.box.mr}

    return val_fn, device


def _metrics(values):
    return {key: float(values[key]) for key in DET_METRICS}


def eval_detector(weights=None, data_yaml=None, imgsz=None, device="auto", reports_dir=None, val_fn=None):
    """Validate the hand detector overall and per source; write <reports_dir>/detector.json and return it.

    Defaults: weights models/hand_detector.pt, data_yaml data/hand_det/hand.yaml,
    imgsz config.DET_IMGSZ, reports_dir models/reports. ``val_fn(data_yaml,
    name) -> {map50, map50_95, precision, recall}`` defaults to Ultralytics val
    of ``weights`` with project=<reports_dir> and name=<name>.

    The per-source subsets run before the full val split: Ultralytics keys its
    label cache by the labels folder, so each subset run rewrites
    data/hand_det/labels/val.cache, and the full run last restores it.
    """
    weights = Path(weights) if weights is not None else config.DET_WEIGHTS
    data_yaml = Path(data_yaml) if data_yaml is not None else config.HAND_DET_DIR / "hand.yaml"
    imgsz = imgsz or config.DET_IMGSZ
    reports_dir = (Path(reports_dir) if reports_dir is not None else config.REPORTS_DIR).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    root, names, images = _val_images(data_yaml)
    if val_fn is None:
        val_fn, device = _ultralytics_val(weights, imgsz, device, reports_dir)

    per_source = {}
    for source, paths in _by_source(images).items():
        name = f"detector_val_{source}"
        yaml_path, list_path = _write_subset(reports_dir, source, root, names, paths)
        try:
            metrics = _metrics(val_fn(yaml_path, name))
        finally:
            yaml_path.unlink(missing_ok=True)
            list_path.unlink(missing_ok=True)
        per_source[source] = {**metrics, "n_images": len(paths), "val_dir": str(reports_dir / name)}
    overall = _metrics(val_fn(data_yaml, "detector_val"))

    det_stats = _read_json(data_yaml.parent / "stats.json") or {}
    cls_stats = _read_json(config.CLS_DIR / "stats.json") or {}
    report = {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "device": device,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_images": len(images),
        **overall,
        "val_dir": str(reports_dir / "detector_val"),
        "per_source": per_source,
        "cls_test_detection_rate": cls_test_detection_rate(cls_stats),
        "cls_test_fallback": cls_stats.get("test_fallback"),
        "cls_test_n": cls_stats.get("n_test"),
        "webcam_view_source": det_stats.get("webcam_view_source"),
    }
    json_path = reports_dir / "detector.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    def line(m):
        return (f"mAP50 {m['map50']:.4f}  mAP50-95 {m['map50_95']:.4f}  precision {m['precision']:.4f}  "
                f"recall {m['recall']:.4f}")

    print(f"detector val: {line(report)}  ({len(images)} images) -> {json_path}")
    for source, m in per_source.items():
        print(f"  {source:<15} {line(m)}  ({m['n_images']} images)")
    rate = report["cls_test_detection_rate"]
    rate = "n/a" if rate is None else f"{rate:.4f} ({report['cls_test_fallback']} fallback of {report['cls_test_n']})"
    print(f"  cls_test detection rate {rate}; webcam_view_source {report['webcam_view_source']}")
    return report


# --- CLI ------------------------------------------------------------------------


def _fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate the letter classifier on data/cls/val and data/cls_test/test and, with --detector, "
                    "the hand detector on the data/hand_det val split. Reports go to models/reports/.",
    )
    parser.add_argument(
        "--split", choices=("val", "test", "all"), default="all",
        help="classifier split: val = data/cls/val, test = data/cls_test/test, all = both (default all)",
    )
    parser.add_argument(
        "--detector", action="store_true",
        help="also evaluate the hand detector, overall and per source (models/reports/detector.json)",
    )
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="auto = mps when available")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    split_dirs = {"val": config.CLS_DIR / "val", "test": config.CLS_TEST_DIR / "test"}
    splits = ["val", "test"] if args.split == "all" else [args.split]
    data_yaml = config.HAND_DET_DIR / "hand.yaml"

    required = [
        (config.CLS_WEIGHTS, f"train it: {TRAIN_CLASSIFIER_CMD}"),
        (config.CLS_META, f"train it: {TRAIN_CLASSIFIER_CMD}"),
        *((split_dirs[s], f"build it: {CLS_DATA_CMD}") for s in splits),
    ]
    if args.detector:
        required += [(config.DET_WEIGHTS, f"train it: {TRAIN_DETECTOR_CMD}"), (data_yaml, f"build it: {HAND_DATA_CMD}")]
    missing = [f"{path} is missing; {hint}" for path, hint in required if not Path(path).exists()]
    if missing:
        _fail("\n".join(missing))

    from src import yolo_env  # torch + ultralytics
    from src.class_map import ClassOrderError
    from src.pipeline import LetterClassifier, MetaError

    device = yolo_env.resolve_device(args.device)
    try:
        classifier = LetterClassifier(config.CLS_WEIGHTS, config.CLS_META, device)
    except (ClassOrderError, MetaError) as exc:
        _fail(f"{exc}\nretrain the classifier: {RETRAIN_CLASSIFIER_CMD}")
    for split in splits:
        eval_classifier(config.CLS_WEIGHTS, config.CLS_META, split_dirs[split], device, classifier=classifier,
                        split_name=split)
    if args.detector:
        eval_detector(device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
