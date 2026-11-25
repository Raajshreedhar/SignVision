"""The sign guide: a trailing pane that shows every sign the app reads.

GuidePane lists the 29 classes of config.CLASSES (A-Z, then SPACE, DELETE and
NOTHING) as tiles: the hand-shape photo from config.GUIDE_DIR (CC0, see
assets/guide/SOURCES.md), the sign's name and, for five signs, a short note.
Two paragraphs above the grid describe how the app commits a sign: hold it
until it appears, then lower the hand (the app reads that as NOTHING) before
the next one; J and Z are still poses because every dataset shows them as
single frames.

Styling is native: the system font, palette roles and no stylesheets. No
tile or section has a background, border or shadow; headings and spacing do
the grouping. The grids reflow to the pane's width (columns_for).
"""

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from src import config

TILE = 88  # photo side, in points
GAP = 12  # space between tile columns
ROW_GAP = 16  # space between tile rows
PADDING = 16  # around the pane's contents
PHOTO_SCALE = 2  # photos are rendered once at 2x, for Retina screens
PHOTO_RADIUS = 6
MIN_WIDTH = 236  # two columns, with room for a scroll bar
SECONDARY_ALPHA = 0.62  # secondary text: WindowText at 62 % (6.2:1 on white, 7.2:1 on #1e1e1e)
CREDIT = "Photos: Dan Rasband, ASL Alphabet Test (CC0)"
CONTROL_SIGNS = ("SPACE", "DELETE", "NOTHING")

GUIDE_INTRO = (
    "Hold a sign steady until it appears in the text. Then lower your hand out of view for about a second "
    "(the app reads that as NOTHING) before the next sign, even when it is a different letter.",
    "J and Z are read as still poses: hold the handshape shown, without the motion.",
)

NOTES = {
    "J": "Still pose only",
    "Z": "Still pose only",
    "SPACE": "Adds a space",
    "DELETE": "Deletes the last letter",
    "NOTHING": "No hand in view",
}


@dataclass(frozen=True)
class GuideEntry:
    """One sign of the guide: its class name and an optional short note."""

    name: str
    note: str | None = None

    @property
    def image_path(self):
        """config.GUIDE_DIR / '<name>.jpg', read at call time so tests can point GUIDE_DIR elsewhere."""
        return config.GUIDE_DIR / f"{self.name}.jpg"


GUIDE_ENTRIES = tuple(GuideEntry(name, NOTES.get(name)) for name in config.CLASSES)


def columns_for(width):
    """Tile columns that fit a viewport ``width`` points wide: 3 at 320, 2 at 236, 1 at 200."""
    return max(1, (width - 2 * PADDING + GAP) // (TILE + GAP))


def secondary_color(palette):
    """The secondary text color: ``palette``'s WindowText at SECONDARY_ALPHA."""
    color = QColor(palette.color(QPalette.ColorRole.WindowText))
    color.setAlphaF(SECONDARY_ALPHA)
    return color


def system_font(widget, point_size, weight=QFont.Weight.Normal):
    """``widget``'s font (the system font) at ``point_size`` and ``weight``."""
    font = QFont(widget.font())
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    return font


class SecondaryLabel(QLabel):
    """A plain-text QLabel painted in secondary_color.

    The color is worked out at paint time from the inherited palette, so it
    follows a live switch between light and dark appearance; QLabel still
    does the sizing, word wrap and accessibility.
    """

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(secondary_color(self.palette()))
        flags = self.alignment().value | (Qt.TextFlag.TextWordWrap.value if self.wordWrap() else 0)
        painter.drawText(self.contentsRect(), flags, self.text())


def rounded_photo(path, size=TILE, radius=PHOTO_RADIUS, scale=PHOTO_SCALE):
    """The image at ``path`` as a ``size``-point square pixmap at ``scale``x with rounded corners.

    Scaled smoothly (center-cropped if it is not square) and rendered once.
    Returns a null QPixmap when the file is missing or unreadable.
    """
    image = QImage(str(path))
    if image.isNull():
        return QPixmap()
    side = size * scale
    image = image.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                         Qt.TransformationMode.SmoothTransformation)
    image = image.copy((image.width() - side) // 2, (image.height() - side) // 2, side, side)
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(image))
    painter.drawRoundedRect(QRectF(0, 0, side, side), radius * scale, radius * scale)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap


class GuideTile(QWidget):
    """Photo, name and optional note of one sign, stacked; no background, border or shadow."""

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self.entry = entry
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.image = QLabel()
        self.image.setFixedSize(TILE, TILE)
        pixmap = rounded_photo(entry.image_path)
        if not pixmap.isNull():
            self.image.setPixmap(pixmap)
        self.image.setAccessibleName(
            "NOTHING: empty view" if entry.name == "NOTHING" else f"{entry.name} hand shape"
        )
        layout.addWidget(self.image)

        self.name_label = QLabel(entry.name)
        self.name_label.setFont(system_font(self, 13, QFont.Weight.DemiBold))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.name_label)

        self.note_label = None
        if entry.note:
            self.note_label = SecondaryLabel(entry.note)
            self.note_label.setFont(system_font(self, 11))
            self.note_label.setWordWrap(True)
            self.note_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self.note_label)
        layout.addStretch(1)
        self.setFixedWidth(TILE)


class GuidePane(QScrollArea):
    """The guide pane: how to sign, the A-Z grid, the three control signs and the photo credit.

    ``tiles`` holds the 29 GuideTiles in GUIDE_ENTRIES order and ``columns``
    the current number of tile columns.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(MIN_WIDTH)
        self.setAccessibleName("Sign guide")

        contents = QWidget()
        layout = QVBoxLayout(contents)
        layout.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        layout.setSpacing(8)

        layout.addWidget(self._heading("How to sign"))
        for paragraph in GUIDE_INTRO:
            label = QLabel(paragraph)
            label.setFont(system_font(self, 13))
            label.setWordWrap(True)
            layout.addWidget(label)

        self.tiles = [GuideTile(entry) for entry in GUIDE_ENTRIES]
        letters = [t for t in self.tiles if t.entry.name not in CONTROL_SIGNS]
        controls = [t for t in self.tiles if t.entry.name in CONTROL_SIGNS]
        self._sections = []
        for title, tiles in (("Letters", letters), ("Control signs", controls)):
            layout.addSpacing(12)
            layout.addWidget(self._heading(title))
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(GAP)
            grid.setVerticalSpacing(ROW_GAP)
            grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            layout.addLayout(grid)
            self._sections.append((grid, tiles))

        layout.addSpacing(12)
        credit = SecondaryLabel(CREDIT)
        credit.setFont(system_font(self, 11))
        credit.setWordWrap(True)
        layout.addWidget(credit)
        self.credit_label = credit
        layout.addStretch(1)
        self.setWidget(contents)

        self.columns = 0
        self._place(columns_for(self.viewport().width()))

    def _heading(self, text):
        label = QLabel(text)
        label.setFont(system_font(self, 13, QFont.Weight.Bold))
        return label

    def _place(self, columns):
        self.columns = columns
        for grid, tiles in self._sections:
            for tile in tiles:
                grid.removeWidget(tile)
            for i, tile in enumerate(tiles):
                grid.addWidget(tile, i // columns, i % columns, Qt.AlignmentFlag.AlignTop)

    def resizeEvent(self, event):
        # QAbstractScrollArea also routes the viewport's resizes here, for
        # example when a non-overlay scroll bar appears.
        super().resizeEvent(event)
        columns = columns_for(self.viewport().width())
        if columns != self.columns:
            self._place(columns)

    def scroll_to_end(self):
        """Scroll to the bottom of the guide (the control signs and the credit line)."""
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
