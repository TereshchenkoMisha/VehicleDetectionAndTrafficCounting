import random

import cv2
from pathlib import Path

root = Path("data/processed/detrac_yolo")
names = {0: "car", 1: "bus", 2: "truck"}
colors = {0: (0, 255, 0), 1: (0, 128, 255), 2: (255, 0, 0)}

imgs = sorted((root / "images/dev").glob("*.jpg")) + sorted((root / "images/test").glob("*.jpg"))
out = Path("reports/sanity")
out.mkdir(parents=True, exist_ok=True)

for p in random.sample(imgs, min(6, len(imgs))):
    im = cv2.imread(str(p))
    H, W = im.shape[:2]
    lbl = root / "labels" / p.parent.name / (p.stem + ".txt")
    n_boxes = 0
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            c, cx, cy, w, h = int(c), *[float(v) for v in (cx, cy, w, h)]
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            cv2.rectangle(im, (x1, y1), (x2, y2), colors[c], 2)
            cv2.putText(im, names[c], (x1, max(0, y1 - 5)), 0, 0.6, colors[c], 2)
            n_boxes += 1
    cv2.imwrite(str(out / f"{p.stem}_gt.jpg"), im)
    print(f"{p.stem}: {n_boxes} boxes")
print("saved to reports/sanity/")