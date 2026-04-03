"""
face/quality.py — Face Quality Filter (safe + optimized)
"""

import cv2
import numpy as np

BLUR_THRESHOLD   = 100
MIN_FACE_SIZE    = 60
MIN_DET_SCORE    = 0.65
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 1.6
CROP_MARGIN      = 0.10


def crop_face(frame, face):
    try:
        if frame is None or frame.size == 0:
            return None

        x1, y1, x2, y2 = map(int, face.bbox)

        h, w = frame.shape[:2]
        mx = int((x2 - x1) * CROP_MARGIN)
        my = int((y2 - y1) * CROP_MARGIN)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        return crop

    except Exception as e:
        print("[crop error]", e)
        return None


def is_blurry(img, threshold=BLUR_THRESHOLD):
    try:
        if img is None or img.size == 0:
            return True

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        return variance < threshold

    except Exception as e:
        print("[blur check error]", e)
        return True


def face_is_usable(frame, face):
    try:
        if frame is None or frame.size == 0:
            return False, "invalid frame"

        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1

        # Size check
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return False, "face too small"

        # Confidence check
        if float(face.det_score) < MIN_DET_SCORE:
            return False, "low confidence"

        # Aspect ratio check
        if h == 0 or not (MIN_ASPECT_RATIO <= w / h <= MAX_ASPECT_RATIO):
            return False, "bad aspect"

        crop = crop_face(frame, face)
        if crop is None:
            return False, "invalid bbox"

        # Blur check
        if is_blurry(crop):
            return False, "blurry"

        return True, ""

    except Exception as e:
        print("[quality error]", e)
        return False, "error"