"""Offline evaluation of the trained models (design section 12, "Offline evaluation").

eval_classifier runs the letter classifier over a folder tree of crops that
src.cls_data already preprocessed, laid out as <split_dir>/<class_dir_name>/
(data/cls/val, data/cls_test/test), and reports:
  - top-1 and top-5 accuracy;
  - per-class accuracy and a 29x29 confusion matrix (rows = true class,
    columns = predicted class, canonical config.CLASSES order);
  - the 5 most-confused class pairs and the 5 weakest classes.
The report is written to <reports_dir>/classifier_<split>.json with a heatmap
in confusion_<split>.png (default reports_dir: models/reports/).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.class_map import class_dir_name
from src.config import CLASSES, NUM_CLASSES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TOP_K = 5
N_REPORTED = 5  # most-confused pairs and weakest classes listed in a report


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
