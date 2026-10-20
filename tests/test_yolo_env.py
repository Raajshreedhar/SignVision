"""src.yolo_env: telemetry opt-out (M2), device resolution, weight paths."""

import os

import pytest

import src.yolo_env as yolo_env
from src import config


def test_events_disabled_after_import():
    from ultralytics.utils.events import events

    assert events.enabled is False


def test_mps_fallback_env_is_set():
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ


def test_resolve_device_cpu():
    assert yolo_env.resolve_device("cpu") == "cpu"


def test_resolve_device_auto():
    assert yolo_env.resolve_device("auto") in {"mps", "cpu"}
    assert yolo_env.resolve_device() == yolo_env.resolve_device("auto")


def test_resolve_device_without_mps(monkeypatch):
    monkeypatch.setattr(yolo_env.torch.backends.mps, "is_available", lambda: False)
    assert yolo_env.resolve_device("auto") == "cpu"
    with pytest.raises(SystemExit):
        yolo_env.resolve_device("mps")


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError):
        yolo_env.resolve_device("cuda")


def test_pretrained_path_creates_dir(tmp_path, monkeypatch):
    target = tmp_path / "pretrained"
    monkeypatch.setattr(config, "PRETRAINED_DIR", target)
    path = yolo_env.pretrained_path("yolo26n.pt")
    assert path == target / "yolo26n.pt"
    assert target.is_dir()


def test_load_yolo_missing_trained_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRETRAINED_DIR", tmp_path / "pretrained")
    with pytest.raises(FileNotFoundError):
        yolo_env.load_yolo(tmp_path / "hand_detector.pt")
