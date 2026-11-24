"""Headless unit tests for the CommitStateMachine (no cv2/torch needed).

Feeds scripted (label, confidence) frame sequences and asserts the accumulated
text, covering debounce, cooldown, the must-pass-through-NOTHING rule, and
low-confidence suppression. The first six tests use small explicit timings;
the rest use the live app's config values (CONFIDENCE_THRESHOLD,
DEBOUNCE_FRAMES, COOLDOWN_FRAMES). The last test checks in a fresh
interpreter that `import src.app` loads neither cv2 nor ultralytics.

Run: .venv/bin/python tests/test_app_state.py
"""

import os
import subprocess
import sys

# Allow running directly by putting the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASSES, CONFIDENCE_THRESHOLD, COOLDOWN_FRAMES, DEBOUNCE_FRAMES, PROJECT_ROOT
from src.state_machine import CommitStateMachine

# Small, explicit timing so the sequences stay readable and deterministic.
DEBOUNCE = 3
COOLDOWN = 2
THRESHOLD = 0.9


def make_machine():
    return CommitStateMachine(
        classes=CLASSES,
        confidence_threshold=THRESHOLD,
        debounce_frames=DEBOUNCE,
        cooldown_frames=COOLDOWN,
    )


def feed(machine, label, conf, frames):
    """Step the machine `frames` times with the same (label, conf)."""
    text = machine.text
    for _ in range(frames):
        text = machine.step(label, conf)
    return text


# Enough frames to clear any pending cooldown AND satisfy the debounce streak,
# so a held label reliably reaches its commit/latch-release decision.
SETTLE = COOLDOWN + DEBOUNCE


def test_hold_commits_after_debounce():
    """Holding 'A' confidently for debounce_frames commits exactly 'A'."""
    m = make_machine()
    # Two frames: below the debounce threshold, nothing commits yet.
    feed(m, "A", 0.99, DEBOUNCE - 1)
    assert m.text == "", f"committed too early: {m.text!r}"
    # The debounce-th frame commits.
    feed(m, "A", 0.99, 1)
    assert m.text == "A", f"expected 'A', got {m.text!r}"


def test_no_double_commit_until_nothing():
    """Continuing to hold 'A' must NOT double-commit until a NOTHING passes."""
    m = make_machine()
    # Commit the first 'A', then keep holding well past cooldown + debounce.
    feed(m, "A", 0.99, DEBOUNCE)
    assert m.text == "A", f"expected 'A', got {m.text!r}"
    feed(m, "A", 0.99, DEBOUNCE + COOLDOWN + 5)
    assert m.text == "A", f"double-committed without NOTHING: {m.text!r}"

    # A NOTHING releases the latch; holding 'A' again now commits a second 'A'.
    feed(m, "NOTHING", 0.99, SETTLE)
    feed(m, "A", 0.99, SETTLE)
    assert m.text == "AA", f"expected 'AA' after NOTHING, got {m.text!r}"


def test_space_appends_after_nothing():
    """After committing 'A' then a NOTHING, holding 'SPACE' yields 'A '."""
    m = make_machine()
    feed(m, "A", 0.99, DEBOUNCE)
    feed(m, "NOTHING", 0.99, SETTLE)
    feed(m, "SPACE", 0.99, SETTLE)
    assert m.text == "A ", f"expected 'A ', got {m.text!r}"


def test_delete_removes_last_char():
    """DELETE (with NOTHING between commits) removes the last char: 'A ' -> 'A'."""
    m = make_machine()
    feed(m, "A", 0.99, DEBOUNCE)
    feed(m, "NOTHING", 0.99, SETTLE)
    feed(m, "SPACE", 0.99, SETTLE)
    assert m.text == "A ", f"setup failed, got {m.text!r}"
    feed(m, "NOTHING", 0.99, SETTLE)
    feed(m, "DELETE", 0.99, SETTLE)
    assert m.text == "A", f"expected 'A' after DELETE, got {m.text!r}"


def test_low_confidence_never_commits():
    """Frames below the confidence threshold never commit, however long held."""
    m = make_machine()
    feed(m, "A", 0.5, DEBOUNCE * 5)
    assert m.text == "", f"low-confidence frames committed: {m.text!r}"


def test_clear_resets_text():
    """clear() empties the accumulated text."""
    m = make_machine()
    feed(m, "A", 0.99, DEBOUNCE)
    assert m.text == "A"
    m.clear()
    assert m.text == "", f"clear did not reset text: {m.text!r}"


# --- The live app's timing (src/config.py) --------------------------------------

# Frames that clear the post-commit cooldown and then complete a debounce streak.
CONFIG_SETTLE = COOLDOWN_FRAMES + DEBOUNCE_FRAMES + 1
CONFIDENT = 0.99


def make_config_machine():
    return CommitStateMachine(
        classes=CLASSES,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        debounce_frames=DEBOUNCE_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES,
    )


def test_cooldown_lasts_exactly_cooldown_frames():
    """The COOLDOWN_FRAMES frames after a commit are ignored, NOTHING included."""
    # Cooldown + one debounce streak of NOTHING releases the latch, so B commits.
    m = make_config_machine()
    feed(m, "A", CONFIDENT, DEBOUNCE_FRAMES)
    assert m.text == "A" and m.cooldown == COOLDOWN_FRAMES
    feed(m, "NOTHING", CONFIDENT, COOLDOWN_FRAMES)
    assert m.cooldown == 0
    feed(m, "NOTHING", CONFIDENT, DEBOUNCE_FRAMES)
    feed(m, "B", CONFIDENT, DEBOUNCE_FRAMES)
    assert m.text == "AB", f"expected 'AB', got {m.text!r}"

    # One NOTHING frame less: a frame inside the cooldown did not count, so the latch holds.
    m = make_config_machine()
    feed(m, "A", CONFIDENT, DEBOUNCE_FRAMES)
    feed(m, "NOTHING", CONFIDENT, COOLDOWN_FRAMES + DEBOUNCE_FRAMES - 1)
    feed(m, "B", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "A", f"a cooldown frame counted toward NOTHING: {m.text!r}"


def test_space_then_delete_sequence():
    """A, SPACE, DELETE, DELETE with NOTHING between: 'A' -> 'A ' -> 'A' -> ''."""
    m = make_config_machine()
    texts = []
    for label in ("A", "SPACE", "DELETE", "DELETE"):
        feed(m, label, CONFIDENT, CONFIG_SETTLE)
        texts.append(m.text)
        feed(m, "NOTHING", CONFIDENT, CONFIG_SETTLE)
    assert texts == ["A", "A ", "A", ""], f"unexpected texts {texts!r}"


def test_delete_on_empty_text_is_harmless():
    """DELETE on empty text leaves it empty; the machine keeps working afterwards."""
    m = make_config_machine()
    feed(m, "DELETE", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "", f"DELETE on empty text produced {m.text!r}"
    feed(m, "NOTHING", CONFIDENT, CONFIG_SETTLE)
    feed(m, "A", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "A", f"expected 'A' after DELETE on empty text, got {m.text!r}"


def test_nothing_releases_latch_for_same_letter():
    """Holding A writes one A however long it is held; after a NOTHING, A commits again."""
    m = make_config_machine()
    feed(m, "A", CONFIDENT, CONFIG_SETTLE * 3)
    assert m.text == "A", f"held A double-committed: {m.text!r}"
    feed(m, "NOTHING", CONFIDENT, CONFIG_SETTLE)
    feed(m, "A", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "AA", f"expected 'AA' after NOTHING, got {m.text!r}"


def test_different_letter_requires_nothing():
    """The latch follows every commit: A then B without a NOTHING gives 'A'; with one, 'AB'."""
    m = make_config_machine()
    feed(m, "A", CONFIDENT, CONFIG_SETTLE)
    feed(m, "B", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "A", f"B committed without a NOTHING in between: {m.text!r}"
    feed(m, "NOTHING", CONFIDENT, CONFIG_SETTLE)
    feed(m, "B", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "AB", f"expected 'AB', got {m.text!r}"


def test_low_confidence_nothing_does_not_release():
    """A NOTHING below CONFIDENCE_THRESHOLD never releases the latch, however long it lasts."""
    m = make_config_machine()
    feed(m, "A", CONFIDENT, CONFIG_SETTLE)
    feed(m, "NOTHING", CONFIDENCE_THRESHOLD - 0.05, CONFIG_SETTLE * 3)
    feed(m, "A", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "A", f"low-confidence NOTHING released the latch: {m.text!r}"
    feed(m, "NOTHING", CONFIDENT, CONFIG_SETTLE)
    feed(m, "A", CONFIDENT, CONFIG_SETTLE)
    assert m.text == "AA", f"confident NOTHING did not release the latch: {m.text!r}"


def test_app_import_is_headless():
    """`import src.app` in a fresh interpreter loads neither ultralytics nor cv2."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, src.app; assert 'ultralytics' not in sys.modules and 'cv2' not in sys.modules"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"headless import check failed:\n{proc.stdout}{proc.stderr}"


if __name__ == "__main__":
    test_hold_commits_after_debounce()
    test_no_double_commit_until_nothing()
    test_space_appends_after_nothing()
    test_delete_removes_last_char()
    test_low_confidence_never_commits()
    test_clear_resets_text()
    test_cooldown_lasts_exactly_cooldown_frames()
    test_space_then_delete_sequence()
    test_delete_on_empty_text_is_harmless()
    test_nothing_releases_latch_for_same_letter()
    test_different_letter_requires_nothing()
    test_low_confidence_nothing_does_not_release()
    test_app_import_is_headless()
    print("test_app_state.py: all assertions passed")
