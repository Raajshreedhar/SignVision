"""Download the Kaggle ASL alphabet dataset via kagglehub.

On success, locate the asl_alphabet_train subfolder under the downloaded path,
write its absolute path to config.DATA_PATH_FILE, and print it. On ANY failure
(most commonly missing Kaggle credentials on this machine) print a clear,
actionable message pointing at the README credential setup and exit 1 WITHOUT
an unhandled traceback.

Idempotent: kagglehub caches under ~/.cache/kagglehub, so re-running is safe.

Run: .venv/bin/python src/download_data.py
"""

import os
import sys

# Allow running directly (python src/download_data.py) by putting the project
# root on sys.path so the `src` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

_CRED_MESSAGE = (
    "Could not download the Kaggle dataset.\n"
    "This usually means Kaggle credentials are not configured on this machine.\n"
    "Set up credentials as described in the README (Kaggle credential setup):\n"
    "  1. Create a Kaggle account and generate an API token (kaggle.json).\n"
    "  2. Place it at ~/.kaggle/kaggle.json (chmod 600), or set the\n"
    "     KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
    "Then re-run: .venv/bin/python src/download_data.py"
)


def _find_train_dir(path):
    """Return the asl_alphabet_train directory under path, or path itself."""
    # Common layouts: <path>/asl_alphabet_train/asl_alphabet_train/<classes>
    # or <path>/asl_alphabet_train/<classes>.
    for root, dirs, _files in os.walk(path):
        if os.path.basename(root) == "asl_alphabet_train":
            # Prefer the deepest asl_alphabet_train that directly holds class
            # folders (e.g. an 'A' subdirectory).
            candidate = root
            nested = os.path.join(root, "asl_alphabet_train")
            if os.path.isdir(nested):
                candidate = nested
            return candidate
    return path


def main():
    try:
        import kagglehub

        path = kagglehub.dataset_download("grassknoted/asl-alphabet")
        print("Path to dataset files:", path)

        train_dir = os.path.abspath(_find_train_dir(path))

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_PATH_FILE.write_text(train_dir + "\n")
        print("Resolved training data path:", train_dir)
        print("Wrote dataset path to:", config.DATA_PATH_FILE)
    except Exception as exc:  # noqa: BLE001 - graceful fallback, no traceback.
        print(_CRED_MESSAGE, file=sys.stderr)
        print(f"(details: {type(exc).__name__}: {exc})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
