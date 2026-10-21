"""Shared argparse validators for the SignVision CLIs (design section 9).

Every factory returns an argparse ``type=`` callable that raises
argparse.ArgumentTypeError, so a bad value is reported as a normal argparse
error and the command exits with status 2.
"""

import argparse
import math
from pathlib import Path

MODEL_CHOICES = ("n", "s")
DEVICE_CHOICES = ("auto", "mps", "cpu")


def _parse_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None


def int_range(lo, hi=None):
    """Integer in [lo, hi] (hi=None means no upper bound)."""

    def parse(text):
        value = _parse_int(text)
        if value < lo or (hi is not None and value > hi):
            bound = f">= {lo}" if hi is None else f"in {lo}..{hi}"
            raise argparse.ArgumentTypeError(f"expected an integer {bound}, got {value}")
        return value

    return parse


def float_range(lo, hi, lo_open=False):
    """Finite float in [lo, hi], or (lo, hi] when lo_open is True."""

    def parse(text):
        try:
            value = float(text)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
        too_low = value <= lo if lo_open else value < lo
        if not math.isfinite(value) or too_low or value > hi:
            interval = f"({lo}, {hi}]" if lo_open else f"[{lo}, {hi}]"
            raise argparse.ArgumentTypeError(f"expected a number in {interval}, got {text!r}")
        return value

    return parse


def imgsz_arg(text):
    """Image size: an integer multiple of 32 in 32..1280."""
    value = _parse_int(text)
    if value < 32 or value > 1280 or value % 32:
        raise argparse.ArgumentTypeError(f"expected a multiple of 32 in 32..1280, got {value}")
    return value


def batch_arg(text):
    """Batch size: an integer >= 1, or -1 for Ultralytics AutoBatch."""
    value = _parse_int(text)
    if value != -1 and value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1 (or -1 for AutoBatch), got {value}")
    return value


def existing_pt(text):
    """Path to an existing .pt weights file, returned as a Path."""
    path = Path(text)
    if path.suffix != ".pt":
        raise argparse.ArgumentTypeError(f"expected a .pt weights file, got {text!r}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"weights file not found: {text}")
    return path
