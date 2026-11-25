"""SignVision's Qt user interface (PySide6): the main window, camera view, sign guide and camera worker.

Every module in this package imports PySide6 at import time. src.app imports
them inside main() only, so `import src.app` stays free of PySide6, cv2 and
ultralytics.
"""
