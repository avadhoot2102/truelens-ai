# check_image.py
import os, sys, csv
from PIL import Image
from model import analyze_image_bytes

folder = sys.argv[1]   # e.g. dataset/ai
outcsv = sys.argv[2]   # e.g. results_ai.csv

rows = []
for root, _, files in os.walk(folder):
    for f in files:
        if f.lower().endswith((".jpg",".jpeg",".png",".webp")):
            path = os.path.join(root, f)
            with open(path, "rb") as fh:
                info = analyze_image_bytes(fh.read(), filename=f)
            rows.append([path, info.get("result"), info.get("confidence"), info.get("contrast"), info.get("sharpness"), info.get("has_exif"), info.get("error","")])

with open(outcsv, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["path","result","confidence","contrast","sharpness","has_exif","error"])
    writer.writerows(rows)
print("Done ->", outcsv)
