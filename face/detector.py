"""
face/detector.py — MTCNN face detector via facenet-pytorch (GPU optimized)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from facenet_pytorch import MTCNN


@dataclass
class FaceResult:
    """Wrapper for compatibility with rest of pipeline."""
    bbox:      np.ndarray
    det_score: float
    kps:       np.ndarray
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class FaceDetector:
    def __init__(self, ctx_id: int = 0):
        # 🔥 Proper GPU selection
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and ctx_id >= 0 else "cpu"
        )
        print(f"[detector] using device: {self.device}")

        # 🔥 Optimized MTCNN for Jetson
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=80,              # ↑ stability
            thresholds=[0.7, 0.8, 0.8],     # ↑ precision
            factor=0.7,                    # ↑ speed
            device=self.device,
            select_largest=False,
            post_process=False,
        )

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Detect faces in frame safely.
        Returns list of FaceResult.
        """

        # 🔴 SAFETY CHECK (prevents crashes)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        try:
            from PIL import Image

            # BGR → RGB
            img_rgb = frame[:, :, ::-1]
            pil_img = Image.fromarray(img_rgb)

            # 🔥 Safe detection
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

            if boxes is None or probs is None:
                return []

            results = []

            for box, prob, lm in zip(boxes, probs, landmarks):
                if prob is None or prob < 0.6:
                    continue

                results.append(FaceResult(
                    bbox=np.array(box, dtype=np.float32),
                    det_score=float(prob),
                    kps=np.array(lm, dtype=np.float32),
                ))

            return results

        except Exception as e:
            # 🔥 Prevents stack smashing propagation
            print("[detector error]", e)
            return []
