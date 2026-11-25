"""The camera canvas: the mirrored frame with the hand boxes and the current label.

CameraView fills its pane edge to edge in black and draws each Snapshot's
frame aspect-fit inside it, so the video owns the content plane with no card
around it. Hand states differ by shape as well as color:

  selected  the signing hand: solid 3 px outline in the system accent color,
            with the label capsule ("A 91%") at its top-left;
  held      the signing hand while briefly lost (grace frames): dashed 2 px
            accent outline, same capsule;
  other     any other hand: thin white outline at 55 % opacity.

With no hand in view a "No hand → NOTHING" capsule sits at the video's
top-left. Capsules are white text on 60 % black, at least 5.7:1 over any
frame. The HUD values go to the status bar through hud_text.

fit_rect, overlay_boxes, label_text and hud_text are pure helpers.
"""

import time
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.tracking import Detection
from src.ui.guide import system_font

WAITING_TEXT = "Waiting for the camera…"
NO_HAND_TEXT = "No hand → NOTHING"
BOX_RADIUS = 8
CAPSULE_HEIGHT = 28
CAPSULE_PADDING = 10  # left and right of the capsule text
CAPSULE_GAP = 6  # between a box's top edge and its capsule
NO_HAND_INSET = 16
CAPSULE_FILL = QColor(0, 0, 0, 153)  # 60 % black
OTHER_COLOR = QColor(255, 255, 255, 140)  # 55 % white
WAITING_COLOR = QColor(255, 255, 255, 153)  # 60 % white


def fit_rect(frame_w, frame_h, w, h):
    """The largest rect with the frame's aspect ratio that fits a ``w`` x ``h`` view, centered (QRectF)."""
    scale = min(w / frame_w, h / frame_h)
    fw, fh = frame_w * scale, frame_h * scale
    return QRectF((w - fw) / 2, (h - fh) / 2, fw, fh)


@dataclass(frozen=True)
class OverlayBox:
    """One box to draw: kind is 'selected', 'held' or 'other'; label is the capsule text or None."""

    kind: str
    box: Detection
    label: str | None


def label_text(result):
    """The capsule text for the signing hand, e.g. 'A 91%' (label and smoothed confidence)."""
    return f"{result.label} {result.conf:.0%}"


def overlay_boxes(result):
    """The boxes of a FrameResult, other hands first; [] when no hand is selected.

    The signing hand uses sel.box, the smoothed box, as the OpenCV window did.
    """
    sel = result.selection
    if sel is None:
        return []
    others = [OverlayBox("other", det, None) for det in sel.others]
    return others + [OverlayBox("held" if sel.held else "selected", sel.box, label_text(result))]


def hud_text(hud):
    """The status-bar line: 'FPS 27.5 · Detector 11.2 ms · Classifier 4.3 ms · MPS' ('Classifier –' with no hand)."""
    cls = "–" if hud["cls_ms"] is None else f"{hud['cls_ms']:.1f} ms"
    return f"FPS {hud['fps']:.1f} · Detector {hud['det_ms']:.1f} ms · Classifier {cls} · {hud['device'].upper()}"


class CameraView(QWidget):
    """Paints the newest Snapshot: frame, hand boxes, label capsule or the no-hand capsule.

    ``snapshot`` is the snapshot on screen (None until the first one).
    frame_painted(ms) fires the first time each snapshot is painted, with the
    milliseconds from its camera read to the end of that paint.
    """

    frame_painted = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot = None
        self._painted = None
        self.setAccessibleName("Camera")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)  # every pixel is painted
        self._waiting_font = system_font(self, 15)
        self._label_font = system_font(self, 15, QFont.Weight.DemiBold)
        self._label_font.setFeature(QFont.Tag("tnum"), 1)  # steady capsule width as the percentage changes
        self._status_font = system_font(self, 13)

    def sizeHint(self):
        return QSize(960, 540)

    def minimumSizeHint(self):
        return QSize(320, 180)

    def set_snapshot(self, snapshot):
        """Show ``snapshot`` on the next paint."""
        self.snapshot = snapshot
        self.update()

    def _pen(self, kind):
        if kind == "other":
            pen = QPen(OTHER_COLOR, 1.5)
        else:
            pen = QPen(self.palette().color(QPalette.ColorRole.Accent), 3 if kind == "selected" else 2)
            if kind == "held":
                pen.setStyle(Qt.PenStyle.DashLine)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    @staticmethod
    def _capsule(painter, x, y, text, font):
        """A fully rounded 60 % black capsule at (x, y) with ``text`` in white."""
        width = QFontMetricsF(font).horizontalAdvance(text) + 2 * CAPSULE_PADDING
        rect = QRectF(x, y, width, CAPSULE_HEIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(CAPSULE_FILL)
        painter.drawRoundedRect(rect, CAPSULE_HEIGHT / 2, CAPSULE_HEIGHT / 2)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _label(self, painter, box_rect, text, video):
        """The label capsule above the box's top-left, or just inside its top edge when there is no room above."""
        width = QFontMetricsF(self._label_font).horizontalAdvance(text) + 2 * CAPSULE_PADDING
        x = max(video.left(), min(box_rect.left(), video.right() - width))
        y = box_rect.top() - CAPSULE_GAP - CAPSULE_HEIGHT
        if y < video.top():
            y = box_rect.top() + CAPSULE_GAP
        self._capsule(painter, x, y, text, self._label_font)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        snapshot = self.snapshot
        if snapshot is None:
            painter.setFont(self._waiting_font)
            painter.setPen(WAITING_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, WAITING_TEXT)
            painter.end()
            return

        image = snapshot.image
        video = fit_rect(image.width(), image.height(), self.width(), self.height())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(video, image)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(video)
        sx, sy = video.width() / image.width(), video.height() / image.height()
        labels = []
        for item in overlay_boxes(snapshot.result):
            b = item.box
            rect = QRectF(video.left() + b.x1 * sx, video.top() + b.y1 * sy, (b.x2 - b.x1) * sx, (b.y2 - b.y1) * sy)
            painter.setPen(self._pen(item.kind))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, BOX_RADIUS, BOX_RADIUS)
            if item.label is not None:
                labels.append((rect, item.label))
        for rect, text in labels:
            self._label(painter, rect, text, video)
        if snapshot.result.selection is None:
            self._capsule(painter, video.left() + NO_HAND_INSET, video.top() + NO_HAND_INSET, NO_HAND_TEXT,
                          self._status_font)
        painter.end()

        if snapshot is not self._painted:
            self._painted = snapshot
            self.frame_painted.emit((time.perf_counter() - snapshot.read_at) * 1000)
