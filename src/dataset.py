"""PyTorch dataset and loaders for the ASL alphabet.

Builds (filepath, class_index) samples from the Kaggle folder layout, performs
a seeded stratified 85/15 train/val split, and applies augmentation to the
TRAIN split only. All images flow through the shared preprocess_frame
(invariant a) -- this module does NOT reimplement any preprocessing.
"""

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config
from src.preprocess import preprocess_frame


def build_samples(data_root):
    """Walk <data_root>/<folder>/*.jpg into a list of (filepath, class_index).

    Folder names are mapped to canonical labels via config.FOLDER_TO_LABEL and
    then to indices via config.CLASSES.index. Index positions NEVER depend on
    os.listdir ordering.
    """
    samples = []
    for folder, label in config.FOLDER_TO_LABEL.items():
        class_index = config.CLASSES.index(label)
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for filepath in glob.glob(os.path.join(folder_path, "*.jpg")):
            samples.append((filepath, class_index))
    return samples


def stratified_split(samples, val_split, seed):
    """Split samples 85/15 (by default) so every class appears in both splits.

    Groups samples by class, shuffles each group with a seeded random.Random,
    then slices per class. Returns (train_samples, val_samples).
    """
    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    rng = random.Random(seed)
    train_samples = []
    val_samples = []
    for class_index in sorted(by_class):
        group = by_class[class_index]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_split))
        # Guarantee at least one sample in each split when the class has >= 2.
        if len(group) >= 2:
            n_val = max(1, min(n_val, len(group) - 1))
        val_samples.extend(group[:n_val])
        train_samples.extend(group[n_val:])
    return train_samples, val_samples


def _augment(gray):
    """Train-only augmentation on an 8-bit grayscale image.

    Random rotation +-15deg with small zoom, plus random brightness/contrast
    scaling. Returns an 8-bit grayscale image for the shared preprocess step.
    """
    h, w = gray.shape[:2]
    angle = random.uniform(-15.0, 15.0)
    scale = random.uniform(0.9, 1.1)
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    # Small random translation (up to ~8% of the dimension).
    mat[0, 2] += random.uniform(-0.08, 0.08) * w
    mat[1, 2] += random.uniform(-0.08, 0.08) * h
    gray = cv2.warpAffine(
        gray, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    # Brightness/contrast: out = alpha*in + beta.
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-20.0, 20.0)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return gray


class ASLDataset(Dataset):
    """Dataset over (filepath, class_index) samples.

    Reads each image as grayscale, applies augmentation when train=True, then
    runs the shared preprocess_frame. Returns (float32 tensor (1,64,64),
    long label tensor).
    """

    def __init__(self, samples, train):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")

        if self.train:
            gray = _augment(gray)

        arr = preprocess_frame(gray)
        return torch.from_numpy(arr).float(), torch.tensor(label, dtype=torch.long)


def make_loaders(data_root):
    """Return (train_loader, val_loader) over the dataset at data_root."""
    samples = build_samples(data_root)
    train_samples, val_samples = stratified_split(
        samples, config.VAL_SPLIT, config.SEED
    )

    train_ds = ASLDataset(train_samples, train=True)
    val_ds = ASLDataset(val_samples, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Smoke check against the resolved dataset path if available.
    import sys

    if not config.DATA_PATH_FILE.exists():
        print(
            "No dataset_path.txt found; run src/download_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    root = config.DATA_PATH_FILE.read_text().strip()
    tl, vl = make_loaders(root)
    print(f"train batches: {len(tl)}, val batches: {len(vl)}")
