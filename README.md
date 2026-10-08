# SignVision

Native macOS (Apple Silicon) real-time American Sign Language alphabet detector.
SignVision uses your laptop webcam to recognize ASL hand signs live and builds up
text as you sign. It runs a custom PyTorch CNN on the Apple **MPS** backend and
shows a single OpenCV window with the current prediction, a region-of-interest
box to prop your hand into, and the accumulated text so far.

## Overview

- **What it does:** You hold a hand sign inside the on-screen ROI box; the model
  predicts a letter and, once the sign is held steadily, commits it to the
  accumulated text shown at the bottom of the window.
- **29 classes:** `A`–`Z` plus three control signs:
  - `SPACE` — appends a space
  - `DELETE` — removes the last character
  - `NOTHING` — a no-op used to confirm the camera sees an empty box (and to
    separate repeated letters so holding a sign writes exactly one character)
- **Grayscale + CLAHE preprocessing** gives the model lighting-robust contrast,
  so varying room light is less of a problem.
- **Backend:** PyTorch with the Apple Silicon **MPS** device (CPU fallback if
  MPS is unavailable).

## Architecture

The project is a single Python package under `src/`:

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Single source of truth: the canonical 29-element `CLASSES` list, `FOLDER_TO_LABEL` dataset-folder mapping, `IMG_SIZE=64`, model/data paths, training hyperparameters, and live-inference parameters (confidence threshold, debounce/cooldown frames, ROI box). |
| `src/preprocess.py` | The shared `preprocess_frame()` — grayscale → CLAHE → resize to 64×64 → normalize to `[0,1]`, returning a `(1, 64, 64)` float32 array. **Invariant (a):** this is the only place preprocessing is defined. |
| `src/dataset.py` | Builds `(filepath, class_index)` samples from the Kaggle folder layout, a seeded stratified 85/15 train/val split, and **train-only** augmentation. All images flow through the shared `preprocess_frame()`. |
| `src/download_data.py` | Downloads the Kaggle ASL alphabet dataset via `kagglehub`, resolves the training folder, and records its path. Fails gracefully with a credential message (no traceback). |
| `src/model.py` | `SignVisionCNN`: a compact 3-block convolutional classifier mapping `(N, 1, 64, 64)` → `(N, 29)` raw logits. |
| `src/train.py` | MPS training loop. Saves the best-validation checkpoint and supports `--smoke-test`. |
| `src/app.py` | The live OpenCV app: webcam capture, ROI crop → shared preprocess → model → softmax, plus the debounce/cooldown `CommitStateMachine` that turns per-frame predictions into committed text. |

Generated artifacts live in `models/` (git-ignored, not committed):

- `models/signvision_cnn.pt` — trained weights **plus** the canonical class list.
- `models/dataset_path.txt` — the resolved path to the downloaded dataset.

The real dataset is cached by `kagglehub` under `~/.cache/kagglehub`, outside the repo.

## Requirements

- **macOS on Apple Silicon** (M-series). The MPS backend is used for training and inference.
- **Python 3.11+.** This repo was built and verified on **Python 3.14** with
  **torch 2.14.1** and a working **MPS** device.
- A working **webcam** to run the live app.
- A **Kaggle account** (to download the dataset for training).

Pinned dependencies are in `requirements.txt` (`torch==2.14.1`,
`torchvision==0.29.1`, `opencv-python==5.0.0.93`, `numpy==2.5.3`, `pillow`,
`kagglehub`, `tqdm`).

## Setup

Create a project virtual environment and install the pinned dependencies into it:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**Why the `.venv` is required (PEP 668):** Homebrew's Python is an
"externally-managed" environment, so a system-wide `pip install` is blocked and
will fail. Creating and using the project `.venv` sidesteps this. All commands in
this README deliberately invoke the venv interpreter directly
(`.venv/bin/python` / `.venv/bin/pip`), so you do not need to "activate" the
environment — but you may `source .venv/bin/activate` if you prefer.

## Kaggle credential setup

The dataset download uses the Kaggle API, which needs an API token. Pick one option:

**Option A — token file:**

1. Sign in to Kaggle, go to your account settings, and choose **Create New API
   Token**. This downloads a `kaggle.json` file.
2. Place it at `~/.kaggle/kaggle.json` and lock down its permissions:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Option B — environment variables:**

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

If no credentials are configured, `src/download_data.py` prints a clear,
actionable message and exits with status 1 (see [Troubleshooting](#troubleshooting)).

## Download the dataset

```bash
.venv/bin/python src/download_data.py
```

This downloads [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
(~1 GB, 29 classes × ~3000 images), resolves the training folder, and writes its
path to `models/dataset_path.txt`. It is idempotent — `kagglehub` caches under
`~/.cache/kagglehub`, so re-running is safe.

## Train

Full training run (requires the dataset to be downloaded first):

```bash
.venv/bin/python -m src.train
```

Useful flags:

```bash
# Point at a specific dataset root (otherwise models/dataset_path.txt is used):
.venv/bin/python -m src.train --data-root /path/to/asl_alphabet_train

# Override the number of epochs (default is set in src/config.py):
.venv/bin/python -m src.train --epochs 20
```

Quick end-to-end check without the real dataset (runs the full
train → validate → save path on tiny synthetic data covering all 29 classes):

```bash
.venv/bin/python -m src.train --smoke-test
```

Training saves the **best-validation-accuracy** checkpoint to
`models/signvision_cnn.pt`. The checkpoint stores the trained weights **and** the
canonical class list, so inference decodes predictions through the exact ordering
used at train time.

## Run the app

```bash
.venv/bin/python -m src.app
```

A single window titled **SignVision** opens showing your (mirrored) webcam feed:

- A green **ROI box** marks where to prop your hand. Hold your sign fully inside it.
- The current prediction and confidence appear just above the box.
- A black panel at the bottom shows the accumulated **Text**.

**How to sign:** hold a sign steadily inside the ROI box until it commits (the
letter appears in the Text panel). Flash a `NOTHING` (empty box) between repeats
of the same letter so it registers each one separately. Use `SPACE` to add a
space and `DELETE` to erase the last character.

**Controls (keys in the window):**

- `q` — quit
- `c` — clear the accumulated text

The app needs a trained model at `models/signvision_cnn.pt`; if it is missing,
the app prints a message telling you to train first and exits.

## Correctness invariants

Two invariants keep training and live inference consistent:

- **(a) Shared preprocessing.** Both training (`src/dataset.py`) and live
  inference (`src/app.py`) call the identical `preprocess_frame()` in
  `src/preprocess.py`. Preprocessing (grayscale → CLAHE → resize → normalize) is
  never reimplemented anywhere else, so the pixels the model sees are the same at
  train and inference time.
- **(b) Class ordering round-trip.** `src/config.py` defines the canonical
  29-element `CLASSES` ordering (`A`–`Z`, then `SPACE`, `DELETE`, `NOTHING`).
  `src/train.py` saves this list inside the checkpoint, and `src/app.py` decodes
  `argmax` through that saved list — so model logit `i` always maps to
  `CLASSES[i]`, and the labels never drift between training and inference.

## Troubleshooting

**MPS / unsupported-op errors.** The training and app scripts set
`PYTORCH_ENABLE_MPS_FALLBACK=1` automatically so any op not yet implemented on
MPS falls back to CPU. If you still hit an MPS error, export it yourself before
running:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

On a non-Apple-Silicon machine (no MPS), both `train` and `app` fall back to the
CPU device automatically — training will just be slower.

**`pip install` fails with an "externally-managed-environment" error.** You are
installing into Homebrew's system Python. Use the project `.venv` instead (see
[Setup](#setup)) — install with `.venv/bin/pip`, not the system `pip`.

**Download fails / "Could not download the Kaggle dataset."** Your Kaggle
credentials are missing or wrong. Set them up as described in
[Kaggle credential setup](#kaggle-credential-setup), then re-run
`.venv/bin/python src/download_data.py`.

**"No dataset found" when training.** Run the download step first, or pass
`--data-root /path/to/asl_alphabet_train` to `src.train`.

**No webcam / "Could not open the webcam (device 0)."** The app could not open
camera device `0`. Make sure a webcam is connected, close other apps that may be
using it, and grant your terminal camera permission under
**System Settings → Privacy & Security → Camera**.

**"No trained model found."** Train the model first with
`.venv/bin/python -m src.train` (or `--smoke-test` for a quick synthetic run) so
`models/signvision_cnn.pt` exists before launching the app.
