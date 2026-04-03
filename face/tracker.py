"""
face/tracker.py — IoU-based face tracker with identity smoothing
"""

import uuid
from collections import deque, Counter


class IdentityTracker:
    def __init__(self, window=15):
        self.history = deque(maxlen=window)

    def update(self, person_id):
        self.history.append(person_id)
        return Counter(self.history).most_common(1)[0][0]


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


class Track:
    def __init__(self, box, window):
        self.id      = str(uuid.uuid4())
        self.box     = box
        self.tracker = IdentityTracker(window)
        self.age     = 0
        self.missed  = 0


class FaceTrackerPool:
    def __init__(self, window=15, iou_threshold=0.35, max_missing=20):
        self.window        = window
        self.iou_threshold = iou_threshold
        self.max_missing   = max_missing
        self.tracks        = []

    def update(self, box, person_id):
        track = self._match(box)
        if track is None:
            track = Track(box, self.window)
            self.tracks.append(track)
        track.box    = box
        track.age   += 1
        track.missed = 0
        return track.tracker.update(person_id)

    def expire(self):
        for t in self.tracks:
            t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missing]

    def _match(self, box):
        best, best_score = None, 0
        for t in self.tracks:
            s = _iou(box, t.box)
            if s > best_score:
                best_score, best = s, t
        return best if best_score >= self.iou_threshold else None
