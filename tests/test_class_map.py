"""src.class_map: per-source folder mapping, class roots, and the NN_ class-dir round trip."""

import string

import pytest

from src.class_map import (
    ClassOrderError,
    DatasetLayoutError,
    canonical_label,
    class_dir_name,
    decode_class_names,
    label_from_filename,
    resolve_class_root,
)
from src.config import CLASSES, NUM_CLASSES

GRASSKNOTED_FOLDERS = list(string.ascii_uppercase) + ["space", "del", "nothing"]


def make_class_dirs(root, names):
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


# --- canonical_label ------------------------------------------------------------


def test_grassknoted_and_danrasband_folders_map_to_all_29_classes():
    labels = [canonical_label(name) for name in GRASSKNOTED_FOLDERS]
    assert labels == CLASSES[:26] + ["SPACE", "DELETE", "NOTHING"]
    assert sorted(labels) == sorted(CLASSES)


def test_ayuraj_lowercase_letters_map_and_digits_are_skipped():
    assert [canonical_label(c) for c in string.ascii_lowercase] == CLASSES[:26]
    assert all(canonical_label(str(d)) is None for d in range(10))
    assert canonical_label("asl_dataset") is None  # the nested duplicate folder


@pytest.mark.parametrize(
    "folder, label",
    [("Space", "SPACE"), ("del", "DELETE"), ("Delete", "DELETE"), ("nothing", "NOTHING"), ("Blank", "NOTHING"),
     ("none", "NOTHING"), (" SPACE ", "SPACE")],
)
def test_debashishsau_style_aliases(folder, label):
    assert canonical_label(folder) == label


@pytest.mark.parametrize("folder", ["", "ab", "unknown", "10", "_", "space2"])
def test_unknown_folders_map_to_none(folder):
    assert canonical_label(folder) is None


def test_label_from_filename():
    assert label_from_filename("A_test.jpg") == "A"
    assert label_from_filename("z_test.png") == "Z"
    assert label_from_filename("nothing_test.jpg") == "NOTHING"
    assert label_from_filename("space_test.jpg") == "SPACE"
    assert label_from_filename("del_test.jpg") == "DELETE"
    assert label_from_filename("7_test.jpg") is None
    assert label_from_filename("readme.txt") is None


# --- resolve_class_root -----------------------------------------------------------


def test_generic_root_collapses_ayuraj_nested_duplicate(tmp_path):
    names = list(string.ascii_lowercase) + [str(d) for d in range(10)]
    top = make_class_dirs(tmp_path / "asl_dataset", names)
    make_class_dirs(top / "asl_dataset", names)  # identical nested copy one level down
    assert resolve_class_root(tmp_path, None) == top


def test_generic_root_collapses_danrasband_root_level_duplicate(tmp_path):
    make_class_dirs(tmp_path, GRASSKNOTED_FOLDERS)
    make_class_dirs(tmp_path / "asl-alphabet-test", GRASSKNOTED_FOLDERS)
    assert resolve_class_root(tmp_path, None) == tmp_path


def test_generic_root_breaks_ties_by_lexicographic_path(tmp_path):
    make_class_dirs(tmp_path / "b_copy", GRASSKNOTED_FOLDERS)
    make_class_dirs(tmp_path / "a_copy", GRASSKNOTED_FOLDERS)
    assert resolve_class_root(tmp_path, None) == tmp_path / "a_copy"


def test_generic_root_without_enough_classes_raises(tmp_path):
    make_class_dirs(tmp_path / "data", list(string.ascii_uppercase[:19]) + ["0", "1"])  # 19 mappable
    with pytest.raises(DatasetLayoutError):
        resolve_class_root(tmp_path, None)
    with pytest.raises(DatasetLayoutError):
        resolve_class_root(tmp_path / "missing", None)


def test_explicit_class_root_picks_train_over_sibling_test(tmp_path):
    base = tmp_path / "ASL_Alphabet_Dataset"
    make_class_dirs(base / "asl_alphabet_test", GRASSKNOTED_FOLDERS)  # sorts first, must not win
    train = make_class_dirs(base / "asl_alphabet_train", GRASSKNOTED_FOLDERS)
    assert resolve_class_root(tmp_path, "ASL_Alphabet_Dataset/asl_alphabet_train") == train


def test_explicit_class_root_missing_or_too_small_raises(tmp_path):
    with pytest.raises(DatasetLayoutError, match="does not exist"):
        resolve_class_root(tmp_path, "asl_alphabet_train/asl_alphabet_train")
    make_class_dirs(tmp_path / "small", ["A", "B", "del"])
    with pytest.raises(DatasetLayoutError, match="3 mappable"):
        resolve_class_root(tmp_path, "small")


# --- class_dir_name / decode_class_names ------------------------------------------------


def test_class_dir_names_sort_in_canonical_order():
    dirs = [class_dir_name(i) for i in range(NUM_CLASSES)]
    assert dirs[0] == "00_A" and dirs[26] == "26_SPACE" and dirs[27] == "27_DELETE" and dirs[28] == "28_NOTHING"
    assert sorted(dirs) == dirs  # what Ultralytics does with the train/ folders
    assert sorted(CLASSES) != CLASSES  # plain names would not round-trip


def test_decode_round_trip_from_dict_and_list():
    names = {i: class_dir_name(i) for i in range(NUM_CLASSES)}
    assert decode_class_names(names) == CLASSES
    assert decode_class_names(dict(reversed(list(names.items())))) == CLASSES  # dict order is irrelevant
    assert decode_class_names({str(k): v for k, v in names.items()}) == CLASSES  # JSON-style keys
    assert decode_class_names([class_dir_name(i) for i in range(NUM_CLASSES)]) == CLASSES


def test_decode_rejects_reordered_entries():
    names = [class_dir_name(i) for i in range(NUM_CLASSES)]
    names[3], names[4] = names[4], names[3]
    with pytest.raises(ClassOrderError, match="index 3: expected '03_D', got '04_E'"):
        decode_class_names(names)


def test_decode_rejects_plain_sorted_names():
    with pytest.raises(ClassOrderError):
        decode_class_names(dict(enumerate(sorted(CLASSES))))


def test_decode_rejects_missing_entry():
    names = {i: class_dir_name(i) for i in range(NUM_CLASSES - 1)}
    with pytest.raises(ClassOrderError, match="expected 29 classes, got 28"):
        decode_class_names(names)


def test_decode_rejects_renamed_entry():
    names = [class_dir_name(i) for i in range(NUM_CLASSES)]
    names[27] = "27_DEL"
    with pytest.raises(ClassOrderError, match="index 27: expected '27_DELETE', got '27_DEL'"):
        decode_class_names(names)


def test_decode_diff_is_capped_for_a_foreign_model():
    with pytest.raises(ClassOrderError) as exc:
        decode_class_names({i: f"imagenet_{i}" for i in range(1000)})
    message = str(exc.value)
    assert "expected 29 classes, got 1000" in message
    assert "... and 990 more mismatched entries" in message
    assert len(message.splitlines()) <= 13


def test_decode_rejects_gapped_keys():
    names = {i: class_dir_name(i) for i in range(NUM_CLASSES)}
    names[29] = names.pop(0)
    with pytest.raises(ClassOrderError):
        decode_class_names(names)
