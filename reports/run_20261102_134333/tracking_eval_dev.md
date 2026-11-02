# Tracking evaluation - dev split

Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, number of sequences: 5.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 65.91 | 59.58 | 73.36 | 66.23 | 83.61 | 79.50 | 68 |
| yolov8n_bytetrack_conf0.25 | 62.22 | 56.94 | 68.38 | 63.08 | 81.44 | 76.22 | 104 |
| yolov8s_botsort_conf0.25 | 69.17 | 63.07 | 76.51 | 69.09 | 83.99 | 82.60 | 56 |
| yolov8s_bytetrack_conf0.25 | 65.61 | 60.20 | 72.07 | 65.17 | 81.68 | 80.43 | 71 |

## Combined dataset metrics (TrackEval combination of all sequences)

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 65.02 | 64.73 | 78.43 |
| yolov8n_bytetrack_conf0.25 | 61.09 | 60.63 | 74.79 |
| yolov8s_botsort_conf0.25 | 68.20 | 67.81 | 81.38 |
| yolov8s_bytetrack_conf0.25 | 64.36 | 62.74 | 78.78 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20033 | 76.52 | 88.23 | 94.33 | 0 |
| yolov8n_botsort_conf0.25 | MVI_40141 | 59.19 | 46.22 | 74.50 | 9 |
| yolov8n_botsort_conf0.25 | MVI_40171 | 54.89 | 47.59 | 60.14 | 25 |
| yolov8n_botsort_conf0.25 | MVI_40773 | 75.50 | 80.82 | 88.66 | 8 |
| yolov8n_botsort_conf0.25 | MVI_63553 | 63.45 | 68.31 | 79.90 | 26 |
| yolov8n_bytetrack_conf0.25 | MVI_20033 | 75.30 | 88.23 | 94.09 | 4 |
| yolov8n_bytetrack_conf0.25 | MVI_40141 | 58.03 | 45.31 | 72.24 | 12 |
| yolov8n_bytetrack_conf0.25 | MVI_40171 | 49.56 | 43.42 | 55.09 | 33 |
| yolov8n_bytetrack_conf0.25 | MVI_40773 | 72.35 | 80.29 | 85.54 | 16 |
| yolov8n_bytetrack_conf0.25 | MVI_63553 | 55.85 | 58.15 | 74.14 | 39 |
| yolov8s_botsort_conf0.25 | MVI_20033 | 77.92 | 91.00 | 95.63 | 0 |
| yolov8s_botsort_conf0.25 | MVI_40141 | 62.54 | 47.30 | 78.03 | 19 |
| yolov8s_botsort_conf0.25 | MVI_40171 | 58.36 | 46.07 | 61.66 | 16 |
| yolov8s_botsort_conf0.25 | MVI_40773 | 79.90 | 86.32 | 92.58 | 3 |
| yolov8s_botsort_conf0.25 | MVI_63553 | 67.10 | 74.77 | 85.09 | 18 |
| yolov8s_bytetrack_conf0.25 | MVI_20033 | 76.26 | 90.42 | 95.28 | 1 |
| yolov8s_bytetrack_conf0.25 | MVI_40141 | 60.53 | 46.26 | 76.78 | 23 |
| yolov8s_bytetrack_conf0.25 | MVI_40171 | 54.19 | 41.34 | 59.49 | 16 |
| yolov8s_bytetrack_conf0.25 | MVI_40773 | 77.95 | 86.14 | 92.24 | 7 |
| yolov8s_bytetrack_conf0.25 | MVI_63553 | 59.11 | 61.70 | 78.36 | 24 |
