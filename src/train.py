"""MPS training loop for SignVisionCNN.

Trains on the ASL alphabet and saves the best-validation-accuracy checkpoint.

INVARIANT (b): the saved checkpoint stores config.CLASSES as 'classes' next to
the weights. src.app decodes argmax through that saved list, so model logit i
maps to CLASSES[i] identically at train and inference time. Do NOT drop or
reorder the 'classes' entry.

Usage:
    .venv/bin/python -m src.train                 # full run (needs dataset)
    .venv/bin/python -m src.train --data-root PATH # override dataset root
    .venv/bin/python -m src.train --smoke-test     # quick end-to-end on
                                                   # synthetic data, no dataset
"""

import argparse
import os
import sys

# Let unsupported MPS ops fall back to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import config
from src.model import SignVisionCNN


def get_device():
    """Prefer Apple MPS when available, otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _SyntheticDataset(Dataset):
    """Tiny random-noise dataset covering every class for the smoke test.

    Produces a handful of samples per class so the full train+val+save path
    exercises all 29 labels without the real ~1GB dataset.
    """

    def __init__(self, per_class, seed):
        gen = torch.Generator().manual_seed(seed)
        images = []
        labels = []
        for class_index in range(config.NUM_CLASSES):
            for _ in range(per_class):
                images.append(
                    torch.rand(1, config.IMG_SIZE, config.IMG_SIZE, generator=gen)
                )
                labels.append(class_index)
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].float(), torch.tensor(self.labels[idx], dtype=torch.long)


def make_smoke_loaders():
    """Return (train_loader, val_loader) over small synthetic data."""
    train_ds = _SyntheticDataset(per_class=4, seed=config.SEED)
    val_ds = _SyntheticDataset(per_class=2, seed=config.SEED + 1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """Run one pass over loader. Trains when optimizer is given, else evaluates.

    Returns (average loss, accuracy) over the pass.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    torch.set_grad_enabled(is_train)
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)

    torch.set_grad_enabled(True)
    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def save_checkpoint(model, best_val_acc):
    """Persist weights plus the canonical class list (invariant b)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            # INVARIANT (b): canonical label ordering must round-trip to app.py.
            "classes": config.CLASSES,
            "img_size": config.IMG_SIZE,
            "arch": "SignVisionCNN",
            "val_acc": best_val_acc,
        },
        config.MODEL_PATH,
    )


def resolve_data_root(args):
    """Resolve the dataset root from --data-root or the saved path file.

    Exits with a clear message if neither is available (and not a smoke test).
    """
    if args.data_root:
        return args.data_root
    if config.DATA_PATH_FILE.exists():
        return config.DATA_PATH_FILE.read_text().strip()
    print(
        "No dataset found. Pass --data-root PATH or run "
        "`.venv/bin/python -m src.download_data` first.",
        file=sys.stderr,
    )
    sys.exit(1)


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    if args.smoke_test:
        print("Smoke test: training on synthetic data (no real dataset).")
        train_loader, val_loader = make_smoke_loaders()
        epochs = 1
    else:
        data_root = resolve_data_root(args)
        print(f"Dataset root: {data_root}")
        from src.dataset import make_loaders

        train_loader, val_loader = make_loaders(data_root)
        epochs = args.epochs

    model = SignVisionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer,
            desc=f"epoch {epoch}/{epochs} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            desc=f"epoch {epoch}/{epochs} val",
        )
        print(
            f"epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_val_acc)
            print(f"  saved best checkpoint (val_acc={best_val_acc:.4f}) -> {config.MODEL_PATH}")

    # Guarantee a checkpoint exists even in the degenerate case of no val data.
    if best_val_acc < 0:
        save_checkpoint(model, best_val_acc)
        print(f"  saved checkpoint -> {config.MODEL_PATH}")

    print("Training complete.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train SignVisionCNN on the ASL alphabet.")
    parser.add_argument(
        "--data-root", default=None,
        help="Override the dataset root directory (defaults to the saved path file).",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="Number of training epochs for a full run.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run the full train+val+save path quickly on synthetic data.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
