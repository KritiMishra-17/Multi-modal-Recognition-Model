# class Metrics:
#     def __init__(self):
#         self.face_correct = 0
#         self.voice_correct = 0
#         self.fusion_correct = 0

#         self.false_accept = 0
#         self.false_reject = 0

#         self.total = 0

#     def update(self, gt, face_pred, voice_pred, fusion_pred):

#         # ❌ skip invalid rows
#         # if not gt or gt.strip() == "":
#         #     return

#         self.total += 1

#         # Face accuracy
#         if face_pred == gt:
#             self.face_correct += 1

#         # Voice accuracy
#         if voice_pred == gt:
#             self.voice_correct += 1

#         # Fusion accuracy
#         if fusion_pred == gt:
#             self.fusion_correct += 1

#         # FAR
#         if fusion_pred != "wrong" and fusion_pred != gt:
#             self.false_accept += 1

#         # FRR
#         if fusion_pred == "wrong" and face_pred == gt:
#             self.false_reject += 1

#     def report(self):
#         if self.total == 0:
#             return {}

#         return {
#             "Face Accuracy": round(self.face_correct / self.total, 3),
#             "Voice Accuracy": round(self.voice_correct / self.total, 3),
#             "Fusion Accuracy": round(self.fusion_correct / self.total, 3),
#             "FAR": round(self.false_accept / self.total, 3),
#             "FRR": round(self.false_reject / self.total, 3),
#         }

import numpy as np

class Metrics:
    def __init__(self):
        self.face_correct = 0
        self.voice_correct = 0
        self.fusion_correct = 0
        self.total = 0

    def update(self, gt, face_pred, voice_pred, fusion_pred):
        self.total += 1

        if face_pred == gt:
            self.face_correct += 1

        if voice_pred == gt:
            self.voice_correct += 1

        if fusion_pred == gt:
            self.fusion_correct += 1

    def report(self):
        return {
            "Face Accuracy": self.face_correct / self.total if self.total else 0,
            "Voice Accuracy": self.voice_correct / self.total if self.total else 0,
            "Fusion Accuracy": self.fusion_correct / self.total if self.total else 0,
        }