"""Headless unit tests for the CommitStateMachine (no cv2/torch needed).

Feeds scripted (label, confidence) frame sequences and asserts the accumulated
text, covering debounce, cooldown, the must-pass-through-NOTHING rule, and
low-confidence suppression.

Run: .venv/bin/python tests/test_app_state.py
"""

import os
import sys

# Allow running directly by putting the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import CommitStateMachine
from src.config import CLASSES

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


if __name__ == "__main__":
    test_hold_commits_after_debounce()
    test_no_double_commit_until_nothing()
    test_space_appends_after_nothing()
    test_delete_removes_last_char()
    test_low_confidence_never_commits()
    test_clear_resets_text()
    print("test_app_state.py: all assertions passed")
