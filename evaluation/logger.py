# import csv
# import os

# FILE = "evaluation/results.csv"

# def init_log():
#     if not os.path.exists("evaluation"):
#         os.makedirs("evaluation")

#     if not os.path.exists(FILE):
#         with open(FILE, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "GT", "Face", "Voice", "Fusion",
#                 "FaceScore", "VoiceScore", "FinalScore"
#             ])


# def log_result(gt, face, voice, fusion, fscore, vscore, final):
#     """
#     Log ONE event per verification
#     """
#     with open(FILE, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             gt,
#             face,
#             voice,
#             fusion,
#             round(fscore, 3),
#             round(vscore, 3),
#             round(final, 3)
#         ])

import csv
import os

FILE = "evaluation/results.csv"

def init_log():
    if not os.path.exists("evaluation"):
        os.makedirs("evaluation")

    if not os.path.exists(FILE):
        with open(FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "GT", "Face", "Voice", "Fusion",
                "FaceScore", "VoiceScore", "FinalScore"
            ])


def log_result(gt, face, voice, fusion, fscore, vscore, final):
    with open(FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gt, face, voice, fusion, fscore, vscore, final])