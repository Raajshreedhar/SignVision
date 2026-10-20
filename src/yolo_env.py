"""The single Ultralytics import point for SignVision.

Importing this module, in this order:
  1. sets PYTORCH_ENABLE_MPS_FALLBACK=1 (setdefault) before torch is imported,
     so MPS ops without a kernel fall back to CPU instead of erroring;
  2. imports ultralytics and switches its anonymous usage analytics off for
     this process only. The global Ultralytics settings file is left untouched,
     and YOLO_OFFLINE is not used because it also blocks weight downloads.

Every other module gets YOLO, the device and the weight paths from here.
"""

import os

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
