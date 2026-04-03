"""
face/pose.py — Face Pose Estimation (safe + stable)
"""

import numpy as np
from collections import deque, Counter

ALL_POSES   = {"front", "left", "right", "up", "down"}
POSE_ORDER  = ["front", "left", "right", "up", "down"]

LATERAL_RATIO   = 0.18
VERTICAL_RATIO  = 0.15
EXTREME_RATIO   = 0.45
POSE_SMOOTH_WINDOW = 5

POSE_ICONS = {
    "front": "⬤",
    "left":  "◀",
    "right": "▶",
    "up":    "▲",
    "down":  "▼",
}


class PoseSmoother:
    def __init__(self, window=POSE_SMOOTH_WINDOW):
        self.history = deque(maxlen=window)

    def update(self, pose):
        self.history.append(pose)
        return Counter(self.history).most_common(1)[0][0]


_smoother = PoseSmoother()


def estimate_pose(face):
    try:
        if face is None or face.kps is None:
            return "front"

        kps = face.kps
        if len(kps) < 3:
            return "front"

        left_eye  = np.array(kps[0])
        right_eye = np.array(kps[1])
        nose      = np.array(kps[2])

        eye_center = (left_eye + right_eye) / 2.0

        dx = nose[0] - eye_center[0]
        dy = nose[1] - eye_center[1]

        face_width = np.linalg.norm(left_eye - right_eye)

        if face_width <= 1 or np.isnan(face_width):
            return "front"

        dx_r = dx / face_width
        dy_r = dy / face_width

        # Pose classification
        if dy_r > EXTREME_RATIO:
            pose = "down"
        elif dx_r > LATERAL_RATIO:
            pose = "right"
        elif dx_r < -LATERAL_RATIO:
            pose = "left"
        elif dy_r < -VERTICAL_RATIO:
            pose = "up"
        elif dy_r > VERTICAL_RATIO * 2:
            pose = "down"
        else:
            pose = "front"

        return _smoother.update(pose)

    except Exception as e:
        print("[pose error]", e)
        return "front"


def pose_is_usable(pose):
    return pose in ALL_POSES

