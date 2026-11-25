"""SignVision's main window and the run loop that ties it to the camera worker.

Pane map: [ content: committed text over the camera canvas ] | [ optional sign guide ].
The native title bar sits on top, then one toolbar (Delete and Clear at the
leading edge; Guide and Quit at the trailing edge), and a status bar with the
performance line at the bottom. The committed text is at the top, away from
the window's bottom edge; the camera canvas fills the rest edge to edge; the
guide is an adjacent pane behind a splitter divider, hidden until the Guide
button (or View > Guide) shows it.

Native styling only: QMainWindow, QToolBar, QSplitter and QStatusBar with the
system font and palette, no stylesheets. Toolbar symbols are SF Symbols that
macOS draws at runtime (QIcon.fromTheme); where they do not resolve (offscreen,
other platforms) the buttons show their text alone. Delete and Clear are also
in the Edit menu and Guide in the View menu; q quits and c clears, as in the
OpenCV window.

The window talks to a controller: LiveWorker in the app, fakes in the tests.
It needs a frame_ready signal and take_snapshot(), text(), delete_last(),
clear_text() and stop().
"""

import signal
import sys

from PySide6.QtCore import QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QKeySequence, QPainter, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QSplitterHandle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.ui.camera_view import CameraView, hud_text
from src.ui.guide import GuidePane, SecondaryLabel, secondary_color, system_font
from src.ui.worker import LiveWorker

WINDOW_TITLE = "SignVision"
DEFAULT_SIZE = (1180, 780)
MIN_SIZE = (720, 480)
TRANSCRIPT_HEIGHT = 64
TRANSCRIPT_MARGIN = 16
PLACEHOLDER = "Hold a sign until it appears here"
GUIDE_WIDTH = 320  # three tile columns
SEPARATOR_ALPHA = 0.15
TOOLBAR_LEAD = 6  # with the style's margins, puts the first symbol on the text's 16-point edge
HUD_LEAD = 14  # the same edge for the status line
KEEP_ALIVE_MS = 200  # lets Python run its SIGINT handler while Qt's event loop runs


class TranscriptView(QWidget):
    """The committed text in large type, elided on the left so the newest letters stay visible."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self.setFixedHeight(TRANSCRIPT_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Text")
        self.setAccessibleDescription(PLACEHOLDER)
        self._font = system_font(self, 26, QFont.Weight.DemiBold)
        self._placeholder_font = system_font(self, 15)

    def set_text(self, text):
        if text != self._text:
            self._text = text
            self.setAccessibleDescription(text or PLACEHOLDER)
            self.update()

    def text(self):
        return self._text

    def displayed_text(self):
        """The string drawn for the current width; '' while the placeholder shows."""
        if not self._text:
            return ""
        width = max(0, self.width() - 2 * TRANSCRIPT_MARGIN)
        return QFontMetrics(self._font).elidedText(self._text, Qt.TextElideMode.ElideLeft, width)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect().adjusted(TRANSCRIPT_MARGIN, 0, -TRANSCRIPT_MARGIN, 0)
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        if self._text:
            painter.setFont(self._font)
            painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
            painter.drawText(rect, flags, self.displayed_text())
        else:
            painter.setFont(self._placeholder_font)
            painter.setPen(secondary_color(self.palette()))
            painter.drawText(rect, flags, PLACEHOLDER)
        painter.end()


class _Divider(QSplitterHandle):
    """A 1-point pane divider in the separator color (WindowText at 15 %).

    With a handle width under 5, QSplitterHandle keeps a 5-point grab area
    and paints only its contents rect, so the line stays easy to drag.
    """

    def paintEvent(self, event):
        painter = QPainter(self)
        color = QColor(self.palette().color(QPalette.ColorRole.WindowText))
        color.setAlphaF(SEPARATOR_ALPHA)
        painter.fillRect(self.contentsRect(), color)
        painter.end()


class PaneSplitter(QSplitter):
    """QSplitter with a thin straight divider between panes instead of the style's dimpled handle."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(1)

    def createHandle(self):
        return _Divider(self.orientation(), self)


def _action(parent, name, text, symbol, tooltip, checkable=False):
    action = QAction(QIcon.fromTheme(symbol), text, parent)
    action.setObjectName(name)
    action.setToolTip(tooltip)
    action.setCheckable(checkable)
    action.setMenuRole(QAction.MenuRole.NoRole)  # keep Qt from moving it into the application menu
    return action


class MainWindow(QMainWindow):
    """The SignVision window. ``closed`` is emitted once, when the window closes for any reason."""

    closed = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._closed = False
        self._guide_sized = False
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(*MIN_SIZE)
        self.resize(*DEFAULT_SIZE)

        self.delete_action = _action(self, "delete", "Delete", "delete.left", "Delete the last letter")
        self.clear_action = _action(self, "clear", "Clear", "eraser", "Clear all text (C)")
        self.guide_action = _action(self, "guide", "Guide", "hand.raised", "Show or hide the sign guide",
                                    checkable=True)
        self.quit_action = _action(self, "quit", "Quit", "power", "Quit SignVision (Q)")
        self.delete_action.triggered.connect(self._delete)
        self.clear_action.triggered.connect(self._clear)
        self.guide_action.toggled.connect(self._show_guide)
        self.quit_action.triggered.connect(self.close)

        self.toolbar = self._build_toolbar()
        edit = self.menuBar().addMenu("Edit")
        edit.addAction(self.delete_action)
        edit.addAction(self.clear_action)
        view = self.menuBar().addMenu("View")
        view.addAction(self.guide_action)
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.quit_action.trigger)
        QShortcut(QKeySequence(Qt.Key.Key_C), self).activated.connect(self.clear_action.trigger)

        self.transcript = TranscriptView()
        self.camera = CameraView()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.transcript)
        layout.addWidget(self.camera, 1)

        self.guide = GuidePane()
        self.guide.hide()
        self.splitter = PaneSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(content)
        self.splitter.addWidget(self.guide)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.setCentralWidget(self.splitter)

        self.hud = SecondaryLabel("")
        self.hud.setObjectName("hud")
        self.hud.setAccessibleName("Performance")
        self.hud.setContentsMargins(HUD_LEAD, 0, 0, 0)
        font = system_font(self, 11)
        font.setFeature(QFont.Tag("tnum"), 1)  # figures keep their width as the numbers change
        self.hud.setFont(font)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addWidget(self.hud, 1)

        self._update_actions("")

    def _build_toolbar(self):
        toolbar = QToolBar("Controls", self)
        toolbar.setObjectName("controls")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(18, 18))
        lead = QWidget()
        lead.setFixedWidth(TOOLBAR_LEAD)
        toolbar.addWidget(lead)
        toolbar.addAction(self.delete_action)
        toolbar.addAction(self.clear_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addAction(self.guide_action)
        gap = QWidget()
        gap.setFixedWidth(12)
        toolbar.addWidget(gap)
        toolbar.addAction(self.quit_action)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        return toolbar

    def _update_actions(self, text):
        self.delete_action.setEnabled(bool(text))
        self.clear_action.setEnabled(bool(text))

    def _show_text(self, text):
        self.transcript.set_text(text)
        self._update_actions(text)

    @Slot()
    def apply_snapshot(self):
        """Show the controller's newest snapshot, text and HUD line (nothing when there is no new snapshot)."""
        snapshot = self.controller.take_snapshot()
        if snapshot is None:
            return
        self.camera.set_snapshot(snapshot)
        self._show_text(self.controller.text())
        self.hud.setText(hud_text(snapshot.hud))

    def _delete(self):
        self._show_text(self.controller.delete_last())

    def _clear(self):
        self._show_text(self.controller.clear_text())

    def _show_guide(self, visible):
        self.guide.setVisible(visible)
        if visible and not self._guide_sized:
            self._guide_sized = True
            total = self.splitter.width() - self.splitter.handleWidth()
            self.splitter.setSizes([total - GUIDE_WIDTH, GUIDE_WIDTH])

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self.controller.stop()
            self.closed.emit()
        event.accept()


def wire(worker, window, app):
    """Connect the worker, the window and the application.

    frame_ready reaches apply_snapshot through a queued connection (the
    worker emits it on its own thread). When the worker ends (the camera gave
    up or the loop raised), the window closes and the application quits;
    when the window closes, the application quits even if other top-level
    windows exist.
    """
    worker.frame_ready.connect(window.apply_snapshot)
    worker.finished.connect(window.close)
    worker.finished.connect(app.quit)
    window.closed.connect(app.quit)


def run(capture, pipeline, machine, perf, device):
    """Run the window with a LiveWorker on ``capture`` until it closes. Returns the exit code.

    Re-raises an exception from the worker's loop. A camera that stopped
    returning frames prints the worker's error and returns 1. Ctrl+C in the
    terminal closes the window.
    """
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(WINDOW_TITLE)
    worker = LiveWorker(capture, pipeline, machine, perf, device)
    window = MainWindow(worker)
    wire(worker, window, app)

    previous_sigint = signal.signal(signal.SIGINT, lambda *_: window.close())
    keep_alive = QTimer()
    keep_alive.timeout.connect(lambda: None)
    keep_alive.start(KEEP_ALIVE_MS)
    try:
        window.show()
        window.raise_()
        window.activateWindow()
        worker.start()
        app.exec()
    finally:
        worker.stop()
        worker.wait()
        keep_alive.stop()
        signal.signal(signal.SIGINT, previous_sigint)

    if worker.exception is not None:
        raise worker.exception
    if worker.error:
        print(worker.error, file=sys.stderr)
        return 1
    return 0
