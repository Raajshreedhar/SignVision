"""Sliding-window probability smoother (design section 7.3). numpy only.

ProbSmoother averages the classifier's per-frame probability vectors over the
last SMOOTH_WINDOW frames, so a single odd frame cannot flip the label that the
commit state machine sees. SignPipeline resets it when a new hand (a new track)
appears, so predictions from different hands never blend. Going from a hand to
no hand does not reset it: the NOTHING one-hots blend in over the window and
then release the commit latch.
"""

from collections import deque

import numpy as np

from src.config import NUM_CLASSES, SMOOTH_WINDOW


class ProbSmoother:
    """Mean of the last ``window`` renormalized probability vectors."""

    def __init__(self, window=SMOOTH_WINDOW, num_classes=NUM_CLASSES):
        self.window = window
        self.num_classes = num_classes
        self._history = deque(maxlen=window)

    def push(self, p):
        """Add one (num_classes,) probability vector and return the window mean as float32.

        Raises ValueError unless every value is finite and >= 0 and the sum is
        > 0. The vector is renormalized to sum 1 before it is stored.
        """
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (self.num_classes,):
            raise ValueError(f"probabilities must have shape ({self.num_classes},), got {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError("probabilities must be finite")
        if np.any(p < 0):
            raise ValueError("probabilities must be >= 0")
        total = float(p.sum())
        if total <= 0:
            raise ValueError("probabilities must have a positive sum")
        self._history.append(p / total)
        return np.mean(self._history, axis=0).astype(np.float32)

    def reset(self):
        """Forget every stored vector."""
        self._history.clear()
