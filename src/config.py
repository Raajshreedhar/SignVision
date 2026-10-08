"""Project configuration: the single source of truth for SignVision.

CLASSES is the CANONICAL 29-element class ordering. Its index positions ARE the
model's output class indices (model logit i corresponds to CLASSES[i]). This
ordering MUST round-trip into every saved checkpoint: train.py stores CLASSES
alongside the weights and app.py decodes argmax through that saved list, so the
label mapping stays identical at train and inference time. Do NOT reorder.
"""

from pathlib import Path

# --- Canonical class ordering (index == model class index) ---
# A..Z (26 letters) then the three control signs, in this exact order.
CLASSES = [chr(ord("A") + i) for i in range(26)] + ["SPACE", "DELETE", "NOTHING"]
NUM_CLASSES = 29

# Maps raw Kaggle dataset folder names to the canonical CLASSES labels.
# The 26 letter folders ('A'..'Z') map to themselves; the control folders use
# the dataset's lowercase names.
FOLDER_TO_LABEL = {chr(ord("A") + i): chr(ord("A") + i) for i in range(26)}
FOLDER_TO_LABEL.update(
    {
        "space": "SPACE",
        "del": "DELETE",
        "nothing": "NOTHING",
    }
)

# --- Image preprocessing ---
IMG_SIZE = 64  # frames/images are resized to IMG_SIZE x IMG_SIZE, 1-channel.

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "signvision_cnn.pt"
DATA_PATH_FILE = MODELS_DIR / "dataset_path.txt"

# --- Training hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 42

# --- Live inference parameters ---
CONFIDENCE_THRESHOLD = 0.90
DEBOUNCE_FRAMES = 12
COOLDOWN_FRAMES = 15
# Square region-of-interest for the hand, as (x, y, w, h) in frame pixels.
ROI_BOX = (300, 100, 300, 300)
