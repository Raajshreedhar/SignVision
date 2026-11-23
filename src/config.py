"""Project configuration: the single source of truth for SignVision.

CLASSES is the CANONICAL 29-class order (A-Z, then SPACE, DELETE, NOTHING);
index i is the letter classifier's output index i. Do NOT reorder.

The order round-trips through the class folder names: src.cls_data writes
class i to data/cls/{train,val}/<class_dir_name(i)> = f"{i:02d}_{CLASSES[i]}"
(00_A ... 28_NOTHING), so the sorted folder names Ultralytics turns into class
indices are already in canonical order. class_map.decode_class_names checks a
trained model's names against that scheme when the classifier is saved and
when it is loaded, and the classifier meta stores CLASSES as well.

Filesystem paths are read from this module at call time (config.CLS_DIR, ...)
so tests can monkeypatch them; numeric constants may be imported directly.
"""

from pathlib import Path

# --- Canonical class ordering (index == model class index) ---
# A..Z (26 letters) then the three control signs, in this exact order.
CLASSES = [chr(ord("A") + i) for i in range(26)] + ["SPACE", "DELETE", "NOTHING"]
NUM_CLASSES = 29
NOTHING_IDX = CLASSES.index("NOTHING")  # 28; use this instead of a literal index

# --- Project paths and seed ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SEED = 42

# =============================================================================
# YOLO26 two-stage pipeline (hand detector -> letter classifier).
# =============================================================================

# --- Data and model artifact paths (all gitignored) ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SOURCES_FILE = DATA_DIR / "sources.json"
HAND_DET_DIR = DATA_DIR / "hand_det"  # contains hand.yaml, stats.json
CLS_DIR = DATA_DIR / "cls"  # train/ val/ stats.json
CLS_TEST_DIR = DATA_DIR / "cls_test"  # test/ (held-out, never trained on)
PRETRAINED_DIR = MODELS_DIR / "pretrained"
RUNS_DIR = MODELS_DIR / "runs"
REPORTS_DIR = MODELS_DIR / "reports"
SMOKE_DIR = MODELS_DIR / "smoke"
DET_WEIGHTS = MODELS_DIR / "hand_detector.pt"
DET_META = MODELS_DIR / "hand_detector.json"
CLS_WEIGHTS = MODELS_DIR / "letter_classifier.pt"
CLS_META = MODELS_DIR / "letter_classifier.json"

# --- Stage 1: hand detector ---
DET_IMGSZ = 512
DET_CONF = 0.35
DET_MAX_DET = 5
MIN_BOX_AREA_FRAC = 0.002  # ignore boxes smaller than 0.2% of the frame

# --- Detector data fallback (used when HaGRID is unavailable) ---
ZOOMOUT_COMPOSITES_TRAIN = 8000
ZOOMOUT_COMPOSITES_VAL = 800
COMPOSITE_CANVAS = (960, 540)  # (W, H)
COMPOSITE_SCALE = (0.15, 0.45)  # pasted image width as a fraction of canvas width

# --- Shared crop + preprocessing (the invariant) ---
CROP_PAD = 0.20  # square side = max(w, h) * (1 + 2*CROP_PAD)
MIN_CROP_PX = 32
CLS_IMGSZ = 224
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
PREPROCESS_VERSION = "gray-clahe-v2"  # bump whenever preprocess_hand_crop changes

# --- Hand selection ---
STICKY_IOU = 0.30
STICKY_RATIO = 0.50
BOX_EMA_ALPHA = 0.60  # weight of the new box when the same track continues
LOST_GRACE_FRAMES = 3

# --- Smoothing + commit (applied to the smoothed probabilities) ---
SMOOTH_WINDOW = 5
CONFIDENCE_THRESHOLD = 0.75  # was 0.90 on raw per-frame softmax
DEBOUNCE_FRAMES = 8  # was 12
COOLDOWN_FRAMES = 12  # was 15

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
MAX_READ_FAILURES = 5

# --- Classifier data ---
CLS_MAX_PER_CLASS_PER_SOURCE = 1500
CLS_VAL_SPLIT = 0.10
NOTHING_BG_CROPS = 1500  # HaGRID background crops added to NOTHING
FPS_TARGET = 25.0
