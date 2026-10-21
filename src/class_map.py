"""Per-source folder names -> canonical labels, and the canonical class-dir round trip.

Every classifier dataset names its class folders differently ("A", "a", "del",
"Delete", "space", "Blank", ...). canonical_label maps one folder name onto
config.CLASSES (or None for digits and unknown folders); resolve_class_root
finds the folder that holds the class subfolders inside a download.

Ultralytics names classification classes by the sorted folder names of
train/. Plain names would sort DELETE between D and E, so the classifier
dataset uses class_dir_name(i) = f"{i:02d}_{CLASSES[i]}" (00_A ... 28_NOTHING),
whose lexicographic order is the canonical order. decode_class_names checks a
trained model's names against that scheme and strips the prefix; it is the
class-order gate at train-save and at classifier load.

Pure Python: no cv2, numpy or ultralytics.
"""

from pathlib import Path

from src.config import CLASSES, NUM_CLASSES

MIN_CLASS_FOLDERS = 20

# Lowercased folder names (beyond single letters) that map onto control classes.
ALIASES = {
    "space": "SPACE",
    "del": "DELETE",
    "delete": "DELETE",
    "nothing": "NOTHING",
    "blank": "NOTHING",
    "none": "NOTHING",
}


class DatasetLayoutError(Exception):
    """A download does not contain the expected class folders."""


class ClassOrderError(Exception):
    """A model's class names do not follow the canonical NN_LABEL order."""


def canonical_label(folder_name):
    """Canonical label for a class folder name, or None (digits, unknown folders)."""
    key = str(folder_name).strip().lower()
    if len(key) == 1 and "a" <= key <= "z":
        return key.upper()
    return ALIASES.get(key)


def label_from_filename(name):
    """Label of a loose test file such as 'A_test.jpg' or 'nothing_test.jpg' (None if unmapped)."""
    stem = Path(str(name)).stem
    return canonical_label(stem.partition("_test")[0])


def _mapped_labels(folder):
    """Distinct canonical labels among the immediate subfolders of ``folder``."""
    try:
        children = [child for child in folder.iterdir() if child.is_dir()]
    except OSError:
        return set()
    return {label for label in (canonical_label(child.name) for child in children) if label is not None}


def resolve_class_root(download_root, class_root, min_classes=MIN_CLASS_FOLDERS):
    """Folder whose immediate subfolders are the class folders of one source.

    With an explicit ``class_root`` (relative to ``download_root``) that folder
    must exist and hold at least ``min_classes`` mappable subfolders. Without
    one, the shallowest qualifying folder wins, ties broken by lexicographic
    path; this collapses nested duplicate copies of a dataset. Raises
    DatasetLayoutError when nothing qualifies.
    """
    root = Path(download_root)
    if class_root is not None:
        path = root / class_root
        if not path.is_dir():
            raise DatasetLayoutError(f"class root {path} does not exist")
        found = len(_mapped_labels(path))
        if found < min_classes:
            raise DatasetLayoutError(
                f"class root {path} has {found} mappable class folders (need at least {min_classes})"
            )
        return path

    if not root.is_dir():
        raise DatasetLayoutError(f"download root {root} does not exist")
    level = [root]
    while level:
        for folder in level:
            if len(_mapped_labels(folder)) >= min_classes:
                return folder
        nxt = []
        for folder in level:
            try:
                nxt.extend(child for child in folder.iterdir() if child.is_dir())
            except OSError:
                continue
        level = sorted(nxt, key=lambda p: p.relative_to(root).parts)
    raise DatasetLayoutError(f"no folder under {root} has at least {min_classes} mappable class folders")


def class_dir_name(i):
    """Canonical class folder name for class index i, e.g. 0 -> '00_A', 28 -> '28_NOTHING'."""
    return f"{i:02d}_{CLASSES[i]}"


CLASS_DIRS = [class_dir_name(i) for i in range(NUM_CLASSES)]


def decode_class_names(names):
    """Check model names against the NN_LABEL scheme and return the canonical labels.

    ``names`` is a model's names dict ({0: '00_A', ...}, ordered by key) or a
    list. Raises ClassOrderError with a readable diff unless there are exactly
    29 entries and entry i == class_dir_name(i).
    """
    if isinstance(names, dict):
        try:
            keys = sorted(names, key=int)
        except (TypeError, ValueError):
            raise ClassOrderError(f"class name keys must be integers, got {sorted(map(str, names))}") from None
        if [int(k) for k in keys] != list(range(len(keys))):
            raise ClassOrderError(f"class name keys must be 0..{len(keys) - 1}, got {[int(k) for k in keys]}")
        got = [str(names[k]) for k in keys]
    else:
        got = [str(n) for n in names]

    problems = []
    if len(got) != NUM_CLASSES:
        problems.append(f"expected {NUM_CLASSES} classes, got {len(got)}")
    for i in range(max(len(got), NUM_CLASSES)):
        want = CLASS_DIRS[i] if i < NUM_CLASSES else "<none>"
        have = got[i] if i < len(got) else "<missing>"
        if have != want:
            problems.append(f"  index {i}: expected {want!r}, got {have!r}")
    if problems:
        raise ClassOrderError("class names do not follow the canonical NN_LABEL order:\n" + "\n".join(problems))
    return [name.split("_", 1)[1] for name in got]
