"""Per-frame latency tracking for the live app HUD and the benchmark (design section 7.5). numpy only.

LatencyTracker records det_ms, cls_ms (only for frames where the classifier
ran) and total_ms for every frame. fps() is the rolling value shown in the HUD
(1000 / mean total_ms over the last ``window`` frames); summary() gives mean,
p50, p95 and n per metric over the whole session.
"""

from collections import deque

import numpy as np

METRICS = ("det_ms", "cls_ms", "total_ms")


class LatencyTracker:
    def __init__(self, window=30):
        self.window = window
        self._samples = {metric: [] for metric in METRICS}
        self._recent_total = deque(maxlen=window)

    def record(self, det_ms, cls_ms, total_ms):
        """Record one frame. cls_ms=None means the classifier did not run; it is left out of cls_ms."""
        self._samples["det_ms"].append(float(det_ms))
        if cls_ms is not None:
            self._samples["cls_ms"].append(float(cls_ms))
        self._samples["total_ms"].append(float(total_ms))
        self._recent_total.append(float(total_ms))

    def fps(self):
        """1000 / mean total_ms over the last ``window`` frames; 0.0 before any frame was recorded."""
        if not self._recent_total:
            return 0.0
        mean = sum(self._recent_total) / len(self._recent_total)
        return 1000.0 / mean if mean > 0 else 0.0

    def summary(self):
        """{metric: {mean, p50, p95, n}} over every recorded frame; the statistics are None when n is 0."""
        out = {}
        for metric, values in self._samples.items():
            if values:
                arr = np.asarray(values, dtype=np.float64)
                out[metric] = {
                    "mean": float(arr.mean()),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "n": len(values),
                }
            else:
                out[metric] = {"mean": None, "p50": None, "p95": None, "n": 0}
        return out
