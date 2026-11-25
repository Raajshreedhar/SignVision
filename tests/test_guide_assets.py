"""The committed sign-guide photos in assets/guide/: one per class, unchanged, with their provenance.

SOURCES.md must list every file with the SHA-256 of the dataset image it was
copied from, and name the CC0 license and the dataset URL, so the folder
stays redistributable.
"""

import hashlib
import re

import cv2

from src import config

DATASET_URL = "https://www.kaggle.com/datasets/danrasband/asl-alphabet-test"
ROW = re.compile(r"^\|\s*(\S+\.jpg)\s*\|\s*(\S+)\s*\|\s*([0-9a-f]{64})\s*\|$")


def _rows():
    """(file, source, sha256) for every file row of the SOURCES.md table, in table order."""
    text = (config.GUIDE_DIR / "SOURCES.md").read_text(encoding="utf-8")
    return [m.groups() for m in map(ROW.match, text.splitlines()) if m]


def test_guide_dir_holds_one_photo_per_class_and_the_sources_file():
    names = {p.name for p in config.GUIDE_DIR.iterdir()}
    assert names == {f"{c}.jpg" for c in config.CLASSES} | {"SOURCES.md"}
    assert config.GUIDE_DIR.is_relative_to(config.ASSETS_DIR)
    assert config.ASSETS_DIR.is_relative_to(config.PROJECT_ROOT)


def test_every_photo_matches_its_recorded_sha256():
    rows = _rows()
    assert [file for file, _, _ in rows] == [f"{c}.jpg" for c in config.CLASSES]
    for file, source, sha in rows:
        digest = hashlib.sha256((config.GUIDE_DIR / file).read_bytes()).hexdigest()
        assert digest == sha, f"{file} (from {source}) changed: {digest} != {sha}"


def test_every_photo_decodes_to_200_by_200_bgr():
    for c in config.CLASSES:
        image = cv2.imread(str(config.GUIDE_DIR / f"{c}.jpg"))
        assert image is not None, c
        assert image.shape == (200, 200, 3), (c, image.shape)


def test_sources_name_the_cc0_license_and_the_dataset():
    text = (config.GUIDE_DIR / "SOURCES.md").read_text(encoding="utf-8")
    assert "CC0" in text and "CC0: Public Domain" in text
    assert DATASET_URL in text
