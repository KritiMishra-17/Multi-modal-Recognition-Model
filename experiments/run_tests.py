# import sys
# import os
# import csv

# # 🔥 Fix import path (so it works from anywhere)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from evaluation.metrics import Metrics


# def run_evaluation(file="evaluation/results.csv"):

#     if not os.path.exists(file):
#         print("❌ No results.csv found")
#         return

#     metrics = Metrics()
#     total_rows = 0
#     skipped_rows = 0

#     with open(file, "r") as f:
#         reader = csv.DictReader(f)

#         for row in reader:
#             total_rows += 1

#             gt = row.get("GT", "").strip()
#             face = row.get("Face", "").strip()
#             voice = row.get("Voice", "").strip()
#             fusion = row.get("Fusion", "").strip()

#             # 🚨 SKIP invalid rows (CRITICAL FIX)
#             if not gt:
#                 skipped_rows += 1
#                 continue

#             metrics.update(gt, face, voice, fusion)

#     # ───────────── OUTPUT ─────────────

#     print("\n📊 FINAL RESULTS")
#     results = metrics.report()

#     for k, v in results.items():
#         print(f"{k}: {v}")

#     print("\n📌 DATA SUMMARY")
#     print(f"Total Rows     : {total_rows}")
#     print(f"Used Rows      : {metrics.total}")
#     print(f"Skipped Rows   : {skipped_rows}")


# if __name__ == "__main__":
#     run_evaluation()

from evaluation.metrics import Metrics
import csv

def run_evaluation(file="evaluation/results.csv"):
    metrics = Metrics()

    with open(file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt = row["GT"]
            face = row["Face"]
            voice = row["Voice"]
            fusion = row["Fusion"]

            metrics.update(gt, face, voice, fusion)

    print("\n📊 FINAL RESULTS")
    print(metrics.report())


if __name__ == "__main__":
    run_evaluation()