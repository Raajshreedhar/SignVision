"""Config invariants: canonical class order and artifact path roots."""

import string

from src import config


def test_class_order_is_canonical():
    assert config.CLASSES[:26] == list(string.ascii_uppercase)
    assert config.CLASSES[26] == "SPACE"
    assert config.CLASSES[27] == "DELETE"
    assert config.CLASSES[28] == "NOTHING"


def test_nothing_index_and_class_count():
    assert config.NOTHING_IDX == 28
    assert config.CLASSES[config.NOTHING_IDX] == "NOTHING"
    assert config.NUM_CLASSES == len(config.CLASSES) == 29


def test_image_sizes_are_stride_multiples():
    assert config.DET_IMGSZ % 32 == 0
    assert config.CLS_IMGSZ % 32 == 0


def test_path_roots():
    data_paths = [
        config.RAW_DIR,
        config.SOURCES_FILE,
        config.HAND_DET_DIR,
        config.CLS_DIR,
        config.CLS_TEST_DIR,
    ]
    model_paths = [
        config.PRETRAINED_DIR,
        config.RUNS_DIR,
        config.REPORTS_DIR,
        config.SMOKE_DIR,
        config.DET_WEIGHTS,
        config.DET_META,
        config.CLS_WEIGHTS,
        config.CLS_META,
    ]
    for p in data_paths:
        assert p.is_relative_to(config.DATA_DIR), p
    for p in model_paths:
        assert p.is_relative_to(config.MODELS_DIR), p
    assert config.DATA_DIR.is_relative_to(config.PROJECT_ROOT)
    assert config.MODELS_DIR.is_relative_to(config.PROJECT_ROOT)
