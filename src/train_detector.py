"""Train the stage-1 YOLO26 hand detector (design section 8.1).

    .venv/bin/python -m src.train_detector [--model n|s] [--epochs 40] [--time 2.0] [--imgsz 512]
           [--batch 16] [--patience 10] [--device auto] [--base PATH] [--resume] [--smoke-test]

Trains one class ("hand") on data/hand_det/hand.yaml (built by src.hand_data),
copies the best weights to models/hand_detector.pt, validates them into
models/reports/detector_train_val/ and writes models/hand_detector.json with
mAP50, mAP50-95, precision, recall and the run details.

--time is a wall-clock budget in hours: Ultralytics recomputes the epoch count
to fill it, so --epochs only applies with --time 0. A training error, or a
non-finite loss or metric in the last epoch, exits 1 and prints the same
command with --device cpu; the trainer never switches device silently.

--smoke-test trains 1 epoch on 12 synthetic images into models/smoke/ through
the same post-train steps, and never touches models/hand_detector.pt.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.cli import DEVICE_CHOICES, MODEL_CHOICES, batch_arg, existing_pt, float_range, imgsz_arg, int_range
from src.hand_data import composite_warning

MODULE = "src.train_detector"
EXPECTED_NAMES = {0: "hand"}
HAND_DATA_CMD = ".venv/bin/python -m src.hand_data"


def _fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _yolo_env():
    from src import yolo_env  # torch + ultralytics: imported only when a run starts

    return yolo_env


def _base_label(path):
    """'yolo26n.pt' for pretrained weights, the given path otherwise."""
    path = Path(path)
    if path.parent.resolve() == config.PRETRAINED_DIR.resolve():
        return path.name
    return str(path)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.train_detector",
        description="Train the YOLO26 hand detector on data/hand_det (built by src.hand_data).",
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="n", help="YOLO26 size: n or s (default n)")
    parser.add_argument("--epochs", type=int_range(1, 1000), default=40, help="epochs when --time is 0 (default 40)")
    parser.add_argument(
        "--time", type=float_range(0, 24), default=2.0, help="training budget in hours; 0 = use --epochs (default 2.0)"
    )
    parser.add_argument(
        "--imgsz", type=imgsz_arg, default=None, help=f"image size, a multiple of 32 (default {config.DET_IMGSZ})"
    )
    parser.add_argument("--batch", type=batch_arg, default=16, help="batch size, or -1 for AutoBatch (default 16)")
    parser.add_argument("--patience", type=int_range(0), default=10, help="early-stopping patience (default 10)")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="auto = mps when available")
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--base", type=existing_pt, default=None,
        help="start from these weights instead of pretrained yolo26<model>.pt (e.g. models/hand_detector.pt)",
    )
    start.add_argument(
        "--resume", action="store_true", help="resume the interrupted run in models/runs/hand_detector_<model>/"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="1-epoch synthetic run into models/smoke/ (never touches models/hand_detector.pt)",
    )
    return parser


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
    return run_dir, elapsed, model.trainer.args.imgsz


def finalize_run(run_dir, *, data_yaml, out_weights, out_meta, reports_dir, imgsz, device, base, model_size,
                 stats, train_time_s):
    """Copy best.pt to out_weights, validate it, write the JSON meta; return the meta dict."""
    import ultralytics

    yolo_env = _yolo_env()
    run_dir = Path(run_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        _fail(f"{best} not found; the run saved no best checkpoint")
    epochs_completed = len(yolo_env.read_results(run_dir))

    out_weights = Path(out_weights)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_weights)
    model = yolo_env.load_yolo(out_weights)
    metrics = model.val(
        data=str(data_yaml), imgsz=imgsz, device=device,
        project=str(reports_dir), name="detector_train_val", exist_ok=True,
    )
    meta = {
        "base": base,
        "model_size": model_size,
        "ultralytics_version": ultralytics.__version__,
        "imgsz": imgsz,
        "epochs_completed": epochs_completed,
        "train_time_s": round(train_time_s, 1),
        "device_used": device,
        "metrics": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "webcam_view_source": bool(stats.get("webcam_view_source", False)),
        "names": {str(k): v for k, v in dict(model.names).items()},
        "weights_md5": yolo_env.md5_file(out_weights),
        "run_dir": str(run_dir),
        "val_report_dir": str(Path(reports_dir) / "detector_train_val"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_stats": stats,
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2) + "\n")
    m = meta["metrics"]
    print(
        f"hand detector saved to {out_weights} ({base}, {epochs_completed} epochs, "
        f"{meta['train_time_s']:.0f} s on {device})\n"
        f"  mAP50 {m['map50']:.4f}  mAP50-95 {m['map50_95']:.4f}  "
        f"precision {m['precision']:.4f}  recall {m['recall']:.4f}\n"
        f"  meta: {out_meta}"
    )
    return meta


def smoke_test(smoke_dir=None, device="auto"):
    """1-epoch run on synthetic rectangles through the real post-train steps."""
    import cv2
    import numpy as np

    started = time.time()
    yolo_env = _yolo_env()
    smoke_dir = Path(smoke_dir) if smoke_dir is not None else config.SMOKE_DIR
    dev = yolo_env.resolve_device(device)
    data_dir = smoke_dir / "det_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    rng = np.random.default_rng(config.SEED)
    sample = None
    for split, n in (("train", 8), ("val", 4)):
        img_dir, lbl_dir = data_dir / "images" / split, data_dir / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n):
            img = np.full((128, 128, 3), int(rng.integers(0, 60)), np.uint8)
            w, h = (int(v) for v in rng.integers(32, 72, 2))
            x1, y1 = int(rng.integers(0, 128 - w)), int(rng.integers(0, 128 - h))
            color = tuple(int(c) for c in rng.integers(150, 256, 3))
            cv2.rectangle(img, (x1, y1), (x1 + w - 1, y1 + h - 1), color, thickness=-1)
            path = img_dir / f"rect_{i}.jpg"
            cv2.imwrite(str(path), img)
            cx, cy = (x1 + w / 2) / 128, (y1 + h / 2) / 128
            (lbl_dir / f"rect_{i}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {w / 128:.6f} {h / 128:.6f}\n")
            sample = sample or (path if split == "val" else None)
    data_yaml = data_dir / "hand.yaml"
    data_yaml.write_text(
        f"path: {json.dumps(str(data_dir.resolve()))}\ntrain: images/train\nval: images/val\nnames:\n  0: hand\n"
    )
    stats = {"smoke_test": True, "webcam_view_source": False, "totals": {"train": 8, "val": 4}}

    model = yolo_env.load_yolo(yolo_env.pretrained_path("yolo26n.pt"))
    train_kwargs = yolo_env.common_train_kwargs(smoke_dir)
    train_kwargs.update(
        data=str(data_yaml), epochs=1, time=None, imgsz=64, batch=4, device=dev, name="det_smoke", workers=0
    )
    run_dir, elapsed, imgsz = _train(model, train_kwargs, dev, ["--smoke-test", "--device", device])
    finalize_run(
        run_dir, data_yaml=data_yaml, out_weights=smoke_dir / "hand_detector.pt",
        out_meta=smoke_dir / "hand_detector.json", reports_dir=smoke_dir / "reports", imgsz=imgsz,
        device=dev, base="yolo26n.pt", model_size="n", stats=stats, train_time_s=elapsed,
    )

    trained = yolo_env.load_yolo(smoke_dir / "hand_detector.pt")
    trained.predict(str(sample), imgsz=64, device=dev, verbose=False)
    names = dict(trained.names)
    print(f"smoke test names check: {names}")
    if names != EXPECTED_NAMES:
        _fail(f"smoke test: expected model.names == {EXPECTED_NAMES}, got {names}")
    print(f"detector smoke test OK in {time.time() - started:.1f} s ({smoke_dir}; models/hand_detector.pt untouched)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        smoke_test(device=args.device)
        return 0

    data_yaml = config.HAND_DET_DIR / "hand.yaml"
    stats_path = config.HAND_DET_DIR / "stats.json"
    if not data_yaml.is_file() or not stats_path.is_file():
        _fail(f"detector dataset not found in {config.HAND_DET_DIR}; build it first: {HAND_DATA_CMD}")
    stats = json.loads(stats_path.read_text())
    if not stats.get("webcam_view_source", False):
        c = stats.get("composites", {})
        warning = composite_warning(stats.get("hagrid_status", "unknown"), c.get("train", 0) + c.get("val", 0))
        print(warning, file=sys.stderr)
    if args.time > 0 and not args.resume:
        print(f"NOTE: --epochs ignored because --time > 0 (Ultralytics sizes the run to fill {args.time:g} h)")

    yolo_env = _yolo_env()
    device = yolo_env.resolve_device(args.device)
    name = f"hand_detector_{args.model}"
    if args.resume:
        last = config.RUNS_DIR / name / "weights" / "last.pt"
        if not last.is_file():
            _fail(f"--resume: {last} not found; start a new run without --resume")
        model = yolo_env.load_yolo(last)
        base = _base_label((model.ckpt or {}).get("train_args", {}).get("model", last))
        train_kwargs = {"resume": True, "device": device}
    else:
        base_path = args.base or yolo_env.pretrained_path(f"yolo26{args.model}.pt")
        model = yolo_env.load_yolo(base_path)
        base = _base_label(base_path)
        train_kwargs = yolo_env.common_train_kwargs(config.RUNS_DIR)
        train_kwargs.update(
            data=str(data_yaml), epochs=args.epochs, time=yolo_env.train_time_arg(args.time),
            imgsz=args.imgsz or config.DET_IMGSZ, batch=args.batch, patience=args.patience,
            device=device, name=name, close_mosaic=5,
        )

    run_dir, elapsed, imgsz = _train(model, train_kwargs, device, argv)
    finalize_run(
        run_dir, data_yaml=data_yaml, out_weights=config.DET_WEIGHTS, out_meta=config.DET_META,
        reports_dir=config.REPORTS_DIR, imgsz=imgsz, device=device, base=base, model_size=args.model,
        stats=stats, train_time_s=elapsed,
    )
    print(
        f"continue an interrupted run: .venv/bin/python -m {MODULE} --model {args.model} --resume\n"
        f"train longer from the best weights: .venv/bin/python -m {MODULE} --model {args.model} "
        f"--base models/hand_detector.pt --time 1.0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
