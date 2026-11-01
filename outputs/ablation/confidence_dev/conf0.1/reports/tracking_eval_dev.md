# Tracking evaluation — dev split

Metrics: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, Number of sequences: 5.
GT and predicted observations with centres inside ignored regions are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | 65.92 | 59.59 | 73.36 | 66.26 | 83.61 | 79.51 | 68 |
| yolov8n_bytetrack_conf0.1 | 62.23 | 56.95 | 68.39 | 63.10 | 81.44 | 76.23 | 104 |
| yolov8s_botsort_conf0.1 | 69.19 | 63.09 | 76.53 | 69.16 | 84.00 | 82.62 | 56 |
| yolov8s_bytetrack_conf0.1 | 65.62 | 60.21 | 72.08 | 65.20 | 81.69 | 80.44 | 71 |

## Combined dataset metrics

Combine sequence statistics with TrackEval before calculating dataset metrics

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.1 | 65.02 | 64.75 | 78.43 |
| yolov8n_bytetrack_conf0.1 | 61.10 | 60.65 | 74.80 |
| yolov8s_botsort_conf0.1 | 68.23 | 67.90 | 81.41 |
| yolov8s_bytetrack_conf0.1 | 64.38 | 62.78 | 78.79 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | MVI_20033 | 76.53 | 88.27 | 94.35 | 0 |
| yolov8n_botsort_conf0.1 | MVI_40141 | 59.21 | 46.29 | 74.51 | 9 |
| yolov8n_botsort_conf0.1 | MVI_40171 | 54.89 | 47.60 | 60.14 | 25 |
| yolov8n_botsort_conf0.1 | MVI_40773 | 75.49 | 80.82 | 88.65 | 8 |
| yolov8n_botsort_conf0.1 | MVI_63553 | 63.46 | 68.31 | 79.87 | 26 |
| yolov8n_bytetrack_conf0.1 | MVI_20033 | 75.30 | 88.23 | 94.09 | 4 |
| yolov8n_bytetrack_conf0.1 | MVI_40141 | 58.04 | 45.32 | 72.25 | 12 |
| yolov8n_bytetrack_conf0.1 | MVI_40171 | 49.56 | 43.42 | 55.09 | 33 |
| yolov8n_bytetrack_conf0.1 | MVI_40773 | 72.36 | 80.30 | 85.55 | 16 |
| yolov8n_bytetrack_conf0.1 | MVI_63553 | 55.88 | 58.21 | 74.16 | 39 |
| yolov8s_botsort_conf0.1 | MVI_20033 | 77.91 | 90.98 | 95.62 | 0 |
| yolov8s_botsort_conf0.1 | MVI_40141 | 62.56 | 47.36 | 78.05 | 19 |
| yolov8s_botsort_conf0.1 | MVI_40171 | 58.39 | 46.27 | 61.72 | 16 |
| yolov8s_botsort_conf0.1 | MVI_40773 | 79.90 | 86.33 | 92.59 | 3 |
| yolov8s_botsort_conf0.1 | MVI_63553 | 67.17 | 74.88 | 85.13 | 18 |
| yolov8s_bytetrack_conf0.1 | MVI_20033 | 76.24 | 90.40 | 95.26 | 1 |
| yolov8s_bytetrack_conf0.1 | MVI_40141 | 60.55 | 46.34 | 76.81 | 23 |
| yolov8s_bytetrack_conf0.1 | MVI_40171 | 54.19 | 41.34 | 59.49 | 16 |
| yolov8s_bytetrack_conf0.1 | MVI_40773 | 77.95 | 86.14 | 92.24 | 7 |
| yolov8s_bytetrack_conf0.1 | MVI_63553 | 59.17 | 61.81 | 78.40 | 24 |
