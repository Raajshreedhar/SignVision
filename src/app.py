"""Live OpenCV ASL detector for SignVision.

Opens the laptop webcam, crops a fixed region-of-interest (ROI) for the hand,
runs it through the SHARED preprocess_frame (invariant a -- grayscale/CLAHE is
defined once in src.preprocess, never reimplemented here) and the trained
SignVisionCNN, then decodes the prediction through the saved checkpoint's
'classes' list (invariant b -- the class ordering is NOT re-hardcoded here).

A debounce/cooldown CommitStateMachine turns the noisy per-frame predictions
into committed text: a label must be held confidently for DEBOUNCE_FRAMES
before it commits, a COOLDOWN_FRAMES window blocks immediate re-commits, and the
same label will not recommit until a NOTHING sign passes through. SPACE appends
a space, DELETE removes the last character, NOTHING is a no-op.

The CommitStateMachine is pure Python (no cv2/torch) so it can be unit tested
headlessly. The webcam/inference path lives in main() and is only reached when
the module is run directly, so importing src.app never opens a camera.

Keys in the window: 'q' quits, 'c' clears the accumulated text.

Usage:
    .venv/bin/python -m src.app
"""

import os
import sys

# Let unsupported MPS ops fall back to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src import config


class CommitStateMachine:
    """Turns per-frame (label, confidence) predictions into committed text.

    A candidate label must be observed with confidence >= confidence_threshold
    for debounce_frames consecutive frames before it commits. After a commit a
    cooldown of cooldown_frames blocks any further commits, and an
    awaiting_nothing latch requires a NOTHING sign to pass before the SAME label
    can commit again (so holding a sign steadily produces exactly one letter).

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


def _load_model(device):
    """Load the trained checkpoint and build the model on device.

    Returns (model, classes). Exits with a friendly message (status 1, no
    traceback) if the checkpoint file is missing.
    """
    import torch

    from src.model import SignVisionCNN

    if not config.MODEL_PATH.exists():
        print(
            "No trained model found at "
            f"{config.MODEL_PATH}.\n"
            "Train the model first: .venv/bin/python -m src.train",
            file=sys.stderr,
        )
        sys.exit(1)

    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    # INVARIANT (b): decode predictions through the saved class ordering, never
    # a re-hardcoded list.
    classes = checkpoint["classes"]

    model = SignVisionCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return model, classes


def _get_device():
    """Prefer Apple MPS when available, otherwise fall back to CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _draw_overlay(frame, label, conf, text):
    """Draw the ROI box, current prediction, and accumulated-text panel."""
    import cv2

    x, y, w, h = config.ROI_BOX

    # ROI rectangle the user props their hand into.
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Current label + confidence just above the ROI.
    cv2.putText(
        frame,
        f"{label} {conf * 100:.0f}%",
        (x, max(y - 12, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Bottom panel showing the accumulated text.
    frame_h, frame_w = frame.shape[:2]
    panel_top = frame_h - 70
    cv2.rectangle(frame, (0, panel_top), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "Text: " + text,
        (10, frame_h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "q: quit   c: clear",
        (10, panel_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return frame


def main():
    """Run the live webcam detection loop."""
    import cv2
    import numpy as np
    import torch

    from src.preprocess import preprocess_frame

    device = _get_device()
    print(f"Using device: {device}")

    model, classes = _load_model(device)

    machine = CommitStateMachine(
        classes=classes,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        debounce_frames=config.DEBOUNCE_FRAMES,
        cooldown_frames=config.COOLDOWN_FRAMES,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam (device 0).", file=sys.stderr)
        sys.exit(1)

    x, y, w, h = config.ROI_BOX

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam.", file=sys.stderr)
                break

            # Mirror so moving right on screen matches moving right in reality.
            frame = cv2.flip(frame, 1)

            # Crop the ROI and run it through the SHARED preprocess (invariant a).
            roi = frame[y : y + h, x : x + w]
            processed = preprocess_frame(roi)

            tensor = torch.from_numpy(processed).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)

            label = classes[int(idx.item())]
            confidence = float(conf.item())

            text = machine.step(label, confidence)

            _draw_overlay(frame, label, confidence, text)
            cv2.imshow("SignVision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                machine.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
