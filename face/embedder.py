"""
face/embedder.py — Face embedding via InceptionResnetV1 (GPU optimized)
"""

import numpy as np
import torch
import cv2
from facenet_pytorch import InceptionResnetV1


class FaceEmbedder:
    def __init__(self, app=None):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(f"[embedder] loading InceptionResnetV1 on {self.device} …")

        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(self.device)

    @torch.no_grad()
    def get_embedding(self, frame: np.ndarray, face) -> np.ndarray | None:
        try:
            # 🔴 Safety checks
            if frame is None or frame.size == 0:
                return None

            x1, y1, x2, y2 = map(int, face.bbox)

            # Add margin
            h, w = frame.shape[:2]
            dx = int((x2 - x1) * 0.10)
            dy = int((y2 - y1) * 0.10)

            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(w, x2 + dx)
            y2 = min(h, y2 + dy)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                return None

            # 🔥 Faster preprocessing
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_rs  = cv2.resize(crop_rgb, (160, 160))

            # ⚡ Efficient tensor conversion
            tensor = torch.from_numpy(crop_rs).float()
            tensor = tensor.permute(2, 0, 1)  # HWC → CHW
            tensor = (tensor - 127.5) / 128.0
            tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)

            # 🔥 Forward pass
            emb = self.model(tensor)

            # Move to CPU only once
            emb = emb.squeeze(0).cpu().numpy()

            # 🔴 Safe normalization
            norm = np.linalg.norm(emb)
            if norm == 0 or np.isnan(norm):
                return None

            return emb / norm

        except Exception as e:
            print("[embedder error]", e)
            return None

