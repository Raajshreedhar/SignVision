"""Train the stage-2 YOLO26-cls letter classifier (design section 8.2).

    .venv/bin/python -m src.train_classifier [--model n|s] [--epochs 20] [--time 1.5] [--imgsz 224]
           [--batch 64] [--patience 5] [--device auto] [--base PATH] [--resume] [--smoke-test]

Trains the 29 classes on data/cls/{train,val} (built by src.cls_data). The
class folders are named 00_A ... 28_NOTHING, so Ultralytics' sorted folder
names are the canonical order; the saved model's names are decoded and
checked before anything is written. Afterwards:
  - the best weights go to models/letter_classifier.pt;
  - models/letter_classifier.json records the classes, the crop geometry
    copied from data/cls/stats.json (crop_pad, imgsz, preprocess_version,
    detector_md5) and the run details;
  - val top-1/top-5 come from Ultralytics val (models/reports/classifier_train_val/),
    test top-1/top-5 from src.evaluate on the held-out data/cls_test/test
    (models/reports/classifier_test.json, confusion_test.png).

Pre-checks refuse data/cls when it does not match this code (class folders,
an empty train class, preprocess_version, crop_pad, --imgsz) and print
"rebuild with python -m src.cls_data".

--time is a wall-clock budget in hours: Ultralytics recomputes the epoch count
to fill it, so --epochs only applies with --time 0. A training error, or a
non-finite loss or metric in the last epoch, exits 1 and prints the same
command with --device cpu; the trainer never switches device silently.

--smoke-test trains 1 epoch on 29 x 3 synthetic crops into models/smoke/
through the same post-train steps, and never touches models/letter_classifier.pt.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.class_map import CLASS_DIRS, ClassOrderError, decode_class_names
from src.cli import DEVICE_CHOICES, MODEL_CHOICES, batch_arg, existing_pt, float_range, imgsz_arg, int_range
from src.config import CLASSES, CROP_PAD, NUM_CLASSES, PREPROCESS_VERSION

MODULE = "src.train_classifier"
REBUILD_HINT = "rebuild with python -m src.cls_data (.venv/bin/python -m src.cls_data)"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SMOKE_IMGSZ = 32

# Ultralytics disables ColorJitter whenever auto_augment is set (classify_augmentations:
# disable_color_jitter = not force_color_jitter; force_color_jitter is not a train arg),
# so hsv_* would be silently ignored. RandAugment's Brightness/Contrast ops supply the
# photometric variation; hsv_* are passed as 0 so the args reflect what actually runs.
AUG_ARGS = dict(fliplr=0.5, scale=0.2, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                erasing=0.2, auto_augment="randaugment")


def _fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _yolo_env():
    from src import yolo_env  # torch + ultralytics: imported only when a run starts

    return yolo_env


def _base_label(path):
    """'yolo26n-cls.pt' for pretrained weights, the given path otherwise."""
    path = Path(path)
    if path.parent.resolve() == config.PRETRAINED_DIR.resolve():
        return path.name
    return str(path)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.train_classifier",
        description="Train the YOLO26-cls letter classifier on data/cls (built by src.cls_data).",
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="n", help="YOLO26-cls size: n or s (default n)")
    parser.add_argument("--epochs", type=int_range(1, 1000), default=20, help="epochs when --time is 0 (default 20)")
    parser.add_argument(
        "--time", type=float_range(0, 24), default=1.5, help="training budget in hours; 0 = use --epochs (default 1.5)"
    )
    parser.add_argument(
        "--imgsz", type=imgsz_arg, default=None,
        help=f"image size, must equal the crop size of data/cls (default {config.CLS_IMGSZ})",
    )
    parser.add_argument("--batch", type=batch_arg, default=64, help="batch size, or -1 for AutoBatch (default 64)")
    parser.add_argument("--patience", type=int_range(0), default=5, help="early-stopping patience (default 5)")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="auto = mps when available")
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--base", type=existing_pt, default=None,
        help="start from these weights instead of pretrained yolo26<model>-cls.pt (e.g. models/letter_classifier.pt)",
    )
    start.add_argument(
        "--resume", action="store_true", help="resume the interrupted run in models/runs/letter_classifier_<model>/"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="1-epoch synthetic run into models/smoke/ (never touches models/letter_classifier.pt)",
    )
    return parser


def precheck(cls_dir, imgsz):
    """Return data/cls/stats.json if the dataset matches this code; otherwise exit 1 with REBUILD_HINT."""
    cls_dir = Path(cls_dir)
    stats_path = cls_dir / "stats.json"
    if not (cls_dir / "train").is_dir() or not stats_path.is_file():
        _fail(f"no classifier dataset in {cls_dir} (train/ and stats.json are required); {REBUILD_HINT}")
    try:
        stats = json.loads(stats_path.read_text())
    except (OSError, ValueError) as exc:
        _fail(f"{stats_path} is not readable JSON ({exc}); {REBUILD_HINT}")
    if not isinstance(stats, dict):
        _fail(f"{stats_path} must hold a JSON object; {REBUILD_HINT}")

    version = stats.get("preprocess_version")
    if version != PREPROCESS_VERSION:
        _fail(f"{cls_dir} was built with preprocess_version {version!r}, this code uses {PREPROCESS_VERSION!r}; "
              f"{REBUILD_HINT}")
    if stats.get("crop_pad") != CROP_PAD:
        _fail(f"{cls_dir} was built with crop_pad {stats.get('crop_pad')!r}, config.CROP_PAD is {CROP_PAD}; "
              f"{REBUILD_HINT}")
    if stats.get("cls_imgsz") != imgsz:
        _fail(f"--imgsz {imgsz} does not match the crop size {stats.get('cls_imgsz')!r} of {cls_dir}; pass "
              f"--imgsz {stats.get('cls_imgsz')} or {REBUILD_HINT}")

    for split in ("train", "val"):
        split_dir = cls_dir / split
        found = sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else []
        missing = [d for d in CLASS_DIRS if d not in found]
        extra = [d for d in found if d not in CLASS_DIRS]
        if missing or extra:
            detail = ", ".join(
                part for part in (f"missing {missing}" if missing else "", f"unexpected {extra}" if extra else "")
                if part
            )
            _fail(f"{split_dir} must hold exactly the {NUM_CLASSES} class folders 00_A ... 28_NOTHING ({detail}); "
                  f"{REBUILD_HINT}")
    empty = [
        d for d in CLASS_DIRS
        if not any(p.suffix.lower() in IMAGE_SUFFIXES for p in (cls_dir / "train" / d).iterdir())
    ]
    if empty:
        _fail(f"{cls_dir / 'train'} has no images for {', '.join(empty)}; {REBUILD_HINT}")
    return stats


def _train(model, train_kwargs, device, argv):
    """Run model.train(); exit 1 with the --device cpu command if the run failed."""
    yolo_env = _yolo_env()
    retry = yolo_env.cpu_retry_command(MODULE, argv)
    start = time.time()
    try:
        model.train(**train_kwargs)
    except RuntimeError as exc:
        _fail(f"training failed on {device} ({type(exc).__name__}: {exc})\nre-run on the CPU with:\n  {retry}")
    elapsed = time.time() - start
    run_dir = Path(model.trainer.save_dir)
    reason = yolo_env.run_failure_reason(run_dir)
    if reason:
        _fail(f"training on {device} did not finish cleanly: {reason}\nre-run on the CPU with:\n  {retry}")
    return run_dir, elapsed


def _write_meta(path, meta):
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")


def finalize_run(run_dir, *, data_dir, test_dir, out_weights, out_meta, reports_dir, device, base, model_size,
                 stats, train_time_s):
    """Decode and copy best.pt, write the meta, validate on val, evaluate on test; return the meta dict."""
    import ultralytics

    from src.evaluate import eval_classifier
    from src.pipeline import MetaError

    yolo_env = _yolo_env()
    run_dir = Path(run_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        _fail(f"{best} not found; the run saved no best checkpoint")
    try:
        classes = decode_class_names(yolo_env.load_yolo(best).names)
    except ClassOrderError as exc:
        _fail(f"{best}: {exc}\nno weights or meta written; {REBUILD_HINT} and retrain")
    if classes != CLASSES:
        _fail(f"{best} decodes to {classes}, expected config.CLASSES; no weights or meta written")

    out_weights, out_meta = Path(out_weights), Path(out_meta)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_weights)
    model = yolo_env.load_yolo(out_weights)
    imgsz = int(stats["cls_imgsz"])
    meta = {
        "classes": classes,
        "class_dirs": list(CLASS_DIRS),
        "preprocess_version": stats["preprocess_version"],
        "crop_pad": stats["crop_pad"],
        "imgsz": imgsz,
        "detector_md5": (stats.get("detector") or {}).get("md5"),
        "base": base,
        "model_size": model_size,
        "ultralytics_version": ultralytics.__version__,
        "epochs_completed": len(yolo_env.read_results(run_dir)),
        "train_time_s": round(train_time_s, 1),
        "device_used": device,
        "metrics": {"val_top1": None, "val_top5": None, "test_top1": None, "test_top5": None},
        "weights_md5": yolo_env.md5_file(out_weights),
        "run_dir": str(run_dir),
        "val_report_dir": str(Path(reports_dir) / "classifier_train_val"),
        "test_report": None,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_stats": stats,
    }
    _write_meta(out_meta, meta)  # provisional: eval_classifier loads it through LetterClassifier

    val = model.val(
        data=str(data_dir), split="val", imgsz=imgsz, device=device,
        project=str(reports_dir), name="classifier_train_val", exist_ok=True,
    )
    meta["metrics"]["val_top1"] = float(val.top1)
    meta["metrics"]["val_top5"] = float(val.top5)
    if test_dir is not None and Path(test_dir).is_dir():
        try:
            report = eval_classifier(
                out_weights, out_meta, test_dir, device, split_name="test", reports_dir=reports_dir
            )
        except (ClassOrderError, MetaError) as exc:
            _fail(f"the saved classifier failed validation: {exc}")
        meta["metrics"]["test_top1"] = report["top1"]
        meta["metrics"]["test_top5"] = report["top5"]
        meta["test_report"] = str(Path(reports_dir) / "classifier_test.json")
    else:
        print(f"WARNING: no held-out test set at {test_dir}; test_top1/test_top5 stay null")
    _write_meta(out_meta, meta)

    m = meta["metrics"]
    test = "n/a" if m["test_top1"] is None else f"{m['test_top1']:.4f} / {m['test_top5']:.4f}"
    print(
        f"letter classifier saved to {out_weights} ({base}, {meta['epochs_completed']} epochs, "
        f"{meta['train_time_s']:.0f} s on {device})\n"
        f"  val top-1/top-5 {m['val_top1']:.4f} / {m['val_top5']:.4f}   test top-1/top-5 {test}\n"
        f"  meta: {out_meta}"
    )
    return meta


def smoke_test(smoke_dir=None, device="auto"):
    """1-epoch run on synthetic crops through the real post-train steps and a LetterClassifier round trip."""
    import cv2
    import numpy as np

    from src.class_map import class_dir_name
    from src.pipeline import LetterClassifier
    from src.preprocess import preprocess_hand_crop

    started = time.time()
    yolo_env = _yolo_env()
    smoke_dir = Path(smoke_dir) if smoke_dir is not None else config.SMOKE_DIR
    dev = yolo_env.resolve_device(device)
    data_dir = smoke_dir / "cls_data"
    test_dir = smoke_dir / "cls_test" / "test"
    for folder in (data_dir, test_dir.parent):
        if folder.exists():
            shutil.rmtree(folder)

    rng = np.random.default_rng(config.SEED)
    sample = None
    for root, n in ((data_dir / "train", 3), (data_dir / "val", 2), (test_dir, 1)):
        for i in range(NUM_CLASSES):
            folder = root / class_dir_name(i)
            folder.mkdir(parents=True)
            for j in range(n):
                crop = preprocess_hand_crop(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8), size=SMOKE_IMGSZ)
                cv2.imwrite(str(folder / f"smoke_{j}.png"), crop)
                sample = crop if sample is None else sample
    stats = {
        "smoke_test": True,
        "preprocess_version": PREPROCESS_VERSION,
        "crop_pad": CROP_PAD,
        "cls_imgsz": SMOKE_IMGSZ,
        "detector": None,
    }
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    stats = precheck(data_dir, SMOKE_IMGSZ)

    model = yolo_env.load_yolo(yolo_env.pretrained_path("yolo26n-cls.pt"))
    train_kwargs = yolo_env.common_train_kwargs(smoke_dir)
    train_kwargs.update(
        data=str(data_dir), epochs=1, time=None, imgsz=SMOKE_IMGSZ, batch=16, device=dev, name="cls_smoke",
        **AUG_ARGS,
    )
    train_kwargs["workers"] = 0
    run_dir, elapsed = _train(model, train_kwargs, dev, ["--smoke-test", "--device", device])
    out_weights, out_meta = smoke_dir / "letter_classifier.pt", smoke_dir / "letter_classifier.json"
    finalize_run(
        run_dir, data_dir=data_dir, test_dir=test_dir, out_weights=out_weights, out_meta=out_meta,
        reports_dir=smoke_dir / "reports", device=dev, base="yolo26n-cls.pt", model_size="n", stats=stats,
        train_time_s=elapsed,
    )

    classifier = LetterClassifier(out_weights, out_meta, dev)
    probs = classifier.classify(sample)
    total = float(probs.sum())
    print(f"smoke test round trip: LetterClassifier.classify -> shape {probs.shape}, sum {total:.4f}, "
          f"top {CLASSES[int(probs.argmax())]}")
    if probs.shape != (NUM_CLASSES,) or abs(total - 1.0) > 1e-3:
        _fail(f"smoke test: expected a ({NUM_CLASSES},) probability vector summing to 1, got {probs.shape} "
              f"summing to {total}")
    print(f"classifier smoke test OK in {time.time() - started:.1f} s ({smoke_dir}; "
          "models/letter_classifier.pt untouched)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        smoke_test(device=args.device)
        return 0

    imgsz = args.imgsz or config.CLS_IMGSZ
    stats = precheck(config.CLS_DIR, imgsz)
    if stats.get("detector") is None:
        print("WARNING: data/cls was built with --no-detector (uncropped center squares); live accuracy on "
              "detector crops will be lower", file=sys.stderr)
    if args.time > 0 and not args.resume:
        print(f"NOTE: --epochs ignored because --time > 0 (Ultralytics sizes the run to fill {args.time:g} h)")

    yolo_env = _yolo_env()
    device = yolo_env.resolve_device(args.device)
    name = f"letter_classifier_{args.model}"
    if args.resume:
        last = config.RUNS_DIR / name / "weights" / "last.pt"
        if not last.is_file():
            _fail(f"--resume: {last} not found; start a new run without --resume")
        model = yolo_env.load_yolo(last)
        base = _base_label((model.ckpt or {}).get("train_args", {}).get("model", last))
        train_kwargs = {"resume": True, "device": device}
    else:
        base_path = args.base or yolo_env.pretrained_path(f"yolo26{args.model}-cls.pt")
        model = yolo_env.load_yolo(base_path)
        base = _base_label(base_path)
        train_kwargs = yolo_env.common_train_kwargs(config.RUNS_DIR)
        train_kwargs.update(
            data=str(config.CLS_DIR), epochs=args.epochs, time=yolo_env.train_time_arg(args.time), imgsz=imgsz,
            batch=args.batch, patience=args.patience, device=device, name=name, **AUG_ARGS,
        )

    run_dir, elapsed = _train(model, train_kwargs, device, argv)
    finalize_run(
        run_dir, data_dir=config.CLS_DIR, test_dir=config.CLS_TEST_DIR / "test", out_weights=config.CLS_WEIGHTS,
        out_meta=config.CLS_META, reports_dir=config.REPORTS_DIR, device=device, base=base,
        model_size=args.model, stats=stats, train_time_s=elapsed,
    )
    print(
        f"continue an interrupted run: .venv/bin/python -m {MODULE} --model {args.model} --resume\n"
        f"train longer from the best weights: .venv/bin/python -m {MODULE} --model {args.model} "
        f"--base models/letter_classifier.pt --time 1.0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
