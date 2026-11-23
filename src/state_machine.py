"""Commit state machine: turns per-frame (label, confidence) predictions into text.

The live app feeds it the smoothed pipeline output once per frame. A label
commits after DEBOUNCE_FRAMES consecutive confident frames, a cooldown follows
every commit, and a NOTHING must pass between any two commits (the latch is set
after every commit, not only after a repeated letter). SPACE appends a space,
DELETE removes the last character, NOTHING never writes.

Pure Python (no cv2, numpy, torch or ultralytics): src.app re-exports the class
at import time, and the tests drive it headlessly with scripted sequences.
"""


class CommitStateMachine:
    """Turns per-frame (label, confidence) predictions into committed text.

    A candidate label must be observed with confidence >= confidence_threshold
    for debounce_frames consecutive frames before it commits. After a commit a
    cooldown of cooldown_frames blocks any further commits. The
    awaiting_nothing latch is set after EVERY commit and blocks every
    non-NOTHING commit until NOTHING has been held confidently for
    debounce_frames: a NOTHING must pass between any two commits. Holding a
    sign steadily therefore produces exactly one letter, and "AB" needs a
    NOTHING (or a dropped hand) between the A and the B.

    Control labels: 'SPACE' appends a space, 'DELETE' removes the last
    character, 'NOTHING' is a no-op that also releases the awaiting_nothing
    latch. Everything else is treated as a single-character label to append.

    Pure Python with no cv2/torch dependency so it is unit-testable headlessly.
    """

    def __init__(self, classes, confidence_threshold, debounce_frames, cooldown_frames):
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.debounce_frames = debounce_frames
        self.cooldown_frames = cooldown_frames

        self.candidate = None
        self.stable_count = 0
        self.cooldown = 0
        self.awaiting_nothing = False
        self.text = ""

    def clear(self):
        """Reset the accumulated text (bound to the 'c' key in the app)."""
        self.text = ""

    def step(self, label, conf):
        """Advance the machine by one frame and return the current text.

        Decrements the cooldown while it is active (no commits happen during
        cooldown). Otherwise tracks how long the confident candidate has been
        stable and commits once it reaches debounce_frames.
        """
        # During the post-commit cooldown nothing new commits.
        if self.cooldown > 0:
            self.cooldown -= 1
            return self.text

        # Track the stable candidate. A confident repeat of the same label
        # extends the streak; anything else restarts it at this label.
        if conf >= self.confidence_threshold and label == self.candidate:
            self.stable_count += 1
        else:
            self.candidate = label
            self.stable_count = 1

        # Only confident labels can ever reach the commit threshold.
        if conf < self.confidence_threshold:
            return self.text

        if self.stable_count >= self.debounce_frames:
            if label == "NOTHING":
                # NOTHING never writes; it releases the latch so the previously
                # committed label is free to commit again.
                self.candidate = None
                self.stable_count = 0
                self.awaiting_nothing = False
                return self.text

            if self.awaiting_nothing:
                # A real sign is still held from the last commit; require a
                # NOTHING to pass before the same (or next) label commits.
                return self.text

            self._commit(label)
            self.cooldown = self.cooldown_frames
            self.awaiting_nothing = True

        return self.text

    def _commit(self, label):
        """Apply a committed label to the accumulated text."""
        if label == "SPACE":
            self.text += " "
        elif label == "DELETE":
            self.text = self.text[:-1]
        else:
            self.text += label
