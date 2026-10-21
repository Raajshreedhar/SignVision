"""The single Ultralytics import point for SignVision.

Importing this module, in this order:
  1. sets PYTORCH_ENABLE_MPS_FALLBACK=1 (setdefault) before torch is imported,
     so MPS ops without a kernel fall back to CPU instead of erroring;
  2. imports ultralytics and switches its anonymous usage analytics off for
     this process only. The global Ultralytics settings file is left untouched,
     and YOLO_OFFLINE is not used because it also blocks weight downloads.

Every other module gets YOLO, the device and the weight paths from here. The
shared trainer helpers (time budget, common train args, run-health check, CPU
retry command, md5) live here too, since both trainers use them.
"""

import csv
import hashlib
import math
import os
import shlex

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch / ultralytics

import torch  # noqa: E402
import ultralytics  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.events import events as _ul_events  # noqa: E402

_ul_events.enabled = False  # in-process analytics opt-out; global settings untouched

from pathlib import Path  # noqa: E402

from src import config  # noqa: E402


def resolve_device(pref="auto"):
    """Map a --device preference ('auto' | 'mps' | 'cpu') to a torch device string.

    'auto' picks MPS when available, else CPU. An explicit 'mps' on a machine
    without MPS exits with a message instead of silently using the CPU.
    """
    if pref == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit(
                "--device mps requested but MPS is not available on this machine; "
                "re-run with --device cpu (or --device auto)."
            )
        return "mps"
    if pref == "cpu":
        return "cpu"
    raise ValueError(f"unknown device preference {pref!r}; expected auto, mps or cpu")


def pretrained_path(name):
    """Return models/pretrained/<name>, creating the folder.

    Passing this full path to YOLO() makes Ultralytics download the asset into
    models/pretrained/, so no stray .pt lands in the repo root.
    """
    config.PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    return config.PRETRAINED_DIR / name


def load_yolo(path):
    """Load a YOLO model from ``path``.

    Pretrained assets (anything under config.PRETRAINED_DIR) auto-download when
    missing. Any other missing path raises FileNotFoundError, so a missing
    trained model is reported instead of being fetched from the internet.
    """
    path = Path(path)
    pretrained = path.parent.resolve() == config.PRETRAINED_DIR.resolve()
    if not pretrained and not path.is_file():
        raise FileNotFoundError(path)
    return YOLO(str(path))


# --- shared trainer helpers ---------------------------------------------------


def train_time_arg(hours):
    """--time in hours -> Ultralytics ``time``. 0 means epochs only, passed as None."""
    return None if hours == 0 else float(hours)


def common_train_kwargs(project):
    """Train arguments shared by both trainers; ``project`` must be absolute."""
    return {
        "workers": 4,
        "seed": config.SEED,
        "deterministic": False,  # faster on MPS
        "project": str(project),
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }


def read_results(run_dir):
    """Rows of <run_dir>/results.csv as dicts, with keys and values stripped."""
    with open(Path(run_dir) / "results.csv", newline="", encoding="utf-8") as fh:
        return [
            {key.strip(): (value.strip() if isinstance(value, str) else value) for key, value in row.items() if key}
            for row in csv.DictReader(fh)
        ]


def run_failure_reason(run_dir):
    """Why a finished train() run is unusable, or None if it is healthy.

    A missing results.csv is a failure (training died before logging an
    epoch). Otherwise every loss column (name ending in 'loss': train/box_loss,
    the classifier's train/loss, ...) and every metrics/ column of the last row
    must be a finite number.
    """
    path = Path(run_dir) / "results.csv"
    if not path.is_file():
        return f"{path} is missing: training stopped before logging an epoch"
    rows = read_results(run_dir)
    if not rows:
        return f"{path} has no epoch rows"
    bad = []
    for key, value in rows[-1].items():
        if key.endswith("loss") or key.startswith("metrics/"):
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                bad.append(f"{key}={value}")
    if bad:
        return f"non-finite values in the last epoch of {path}: {', '.join(bad)}"
    return None


def cpu_retry_command(module, argv):
    """The same command line with --device replaced by (or extended with) --device cpu."""
    args, out, i, replaced = list(argv), [], 0, False
    while i < len(args):
        arg = args[i]
        if arg == "--device":
            out += ["--device", "cpu"]
            replaced = True
            i += 2
            continue
        if arg.startswith("--device="):
            out += ["--device", "cpu"]
            replaced = True
        else:
            out.append(arg)
        i += 1
    if not replaced:
        out += ["--device", "cpu"]
    return shlex.join([".venv/bin/python", "-m", module, *out])


def md5_file(path, chunk_bytes=1 << 20):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()
