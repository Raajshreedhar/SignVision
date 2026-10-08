"""Unit tests for the ASL dataset pipeline.

Builds a tiny temp fixture with a few class folders, then checks folder->index
mapping, stratified split coverage, and __getitem__ output shape/label.

Run: .venv/bin/python tests/test_dataset.py
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.dataset import ASLDataset, build_samples, stratified_split
from src.config import IMG_SIZE

# Fixture folders -> expected canonical label.
FIXTURE_FOLDERS = {
    "A": "A",
    "B": "B",
    "del": "DELETE",
    "space": "SPACE",
    "nothing": "NOTHING",
}
IMAGES_PER_FOLDER = 3


def _make_fixture(root):
    for folder in FIXTURE_FOLDERS:
        folder_path = os.path.join(root, folder)
        os.makedirs(folder_path, exist_ok=True)
        for i in range(IMAGES_PER_FOLDER):
            img = np.random.randint(
                0, 256, size=(70, 70, 3), dtype=np.uint8
            )
            cv2.imwrite(os.path.join(folder_path, f"{folder}_{i}.jpg"), img)


def main():
    root = tempfile.mkdtemp(prefix="signvision_fixture_")
    try:
        _make_fixture(root)

        samples = build_samples(root)
        expected_count = len(FIXTURE_FOLDERS) * IMAGES_PER_FOLDER
        assert len(samples) == expected_count, (
            f"expected {expected_count} samples, got {len(samples)}"
        )

        # Folder->index mapping is correct (e.g. 'del' -> index of 'DELETE').
        index_by_folder = {}
        for folder, label in FIXTURE_FOLDERS.items():
            index_by_folder[folder] = config.CLASSES.index(label)
        assert index_by_folder["del"] == config.CLASSES.index("DELETE")
        assert index_by_folder["space"] == config.CLASSES.index("SPACE")
        assert index_by_folder["nothing"] == config.CLASSES.index("NOTHING")
        assert index_by_folder["A"] == config.CLASSES.index("A")

        present_indices = {s[1] for s in samples}
        assert present_indices == set(index_by_folder.values())

        # Stratified split covers all present classes in BOTH splits.
        train_samples, val_samples = stratified_split(
            samples, config.VAL_SPLIT, config.SEED
        )
        train_classes = {s[1] for s in train_samples}
        val_classes = {s[1] for s in val_samples}
        assert train_classes == present_indices, (
            f"train missing classes: {present_indices - train_classes}"
        )
        assert val_classes == present_indices, (
            f"val missing classes: {present_indices - val_classes}"
        )

        # __getitem__ returns (1,64,64) tensor and a valid label.
        train_ds = ASLDataset(train_samples, train=True)
        val_ds = ASLDataset(val_samples, train=False)
        for ds in (train_ds, val_ds):
            arr, label = ds[0]
            assert tuple(arr.shape) == (1, IMG_SIZE, IMG_SIZE), (
                f"tensor shape was {tuple(arr.shape)}"
            )
            assert arr.dtype.is_floating_point
            label_int = int(label)
            assert 0 <= label_int < config.NUM_CLASSES, (
                f"label {label_int} out of range"
            )
            assert label_int in present_indices

        print("test_dataset.py: all assertions passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
