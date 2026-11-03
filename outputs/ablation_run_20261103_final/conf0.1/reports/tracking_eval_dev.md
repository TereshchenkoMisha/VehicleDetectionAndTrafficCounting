# Tracking evaluation - dev split

Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, number of sequences: 5.
GT and tracker boxes with the centre inside an ignored region are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | 65.32 | 59.35 | 72.37 | 66.18 | 83.72 | 78.51 | 119 |
| yolov8n_bytetrack_conf0.1 | 62.24 | 56.65 | 68.80 | 62.94 | 81.52 | 76.78 | 136 |
| yolov8s_botsort_conf0.1 | 69.48 | 63.55 | 76.60 | 70.16 | 84.05 | 83.18 | 74 |
| yolov8s_bytetrack_conf0.1 | 65.21 | 60.35 | 70.99 | 65.51 | 81.80 | 80.04 | 89 |

## Combined dataset metrics (TrackEval combination of all sequences)

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.1 | 64.35 | 64.46 | 77.32 |
| yolov8n_bytetrack_conf0.1 | 61.05 | 60.30 | 75.30 |
| yolov8s_botsort_conf0.1 | 68.50 | 68.80 | 81.96 |
| yolov8s_bytetrack_conf0.1 | 64.00 | 63.09 | 78.43 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | MVI_20033 | 76.60 | 89.00 | 94.65 | 0 |
| yolov8n_botsort_conf0.1 | MVI_40141 | 59.03 | 48.26 | 73.66 | 18 |
| yolov8n_botsort_conf0.1 | MVI_40171 | 55.22 | 47.97 | 59.46 | 34 |
| yolov8n_botsort_conf0.1 | MVI_40773 | 73.31 | 78.52 | 86.20 | 14 |
| yolov8n_botsort_conf0.1 | MVI_63553 | 62.44 | 67.14 | 78.57 | 53 |
| yolov8n_bytetrack_conf0.1 | MVI_20033 | 75.26 | 89.05 | 94.37 | 4 |
| yolov8n_bytetrack_conf0.1 | MVI_40141 | 58.50 | 46.88 | 73.44 | 19 |
| yolov8n_bytetrack_conf0.1 | MVI_40171 | 50.86 | 43.92 | 57.02 | 40 |
| yolov8n_bytetrack_conf0.1 | MVI_40773 | 71.33 | 77.77 | 85.70 | 26 |
| yolov8n_bytetrack_conf0.1 | MVI_63553 | 55.22 | 57.09 | 73.35 | 47 |
| yolov8s_botsort_conf0.1 | MVI_20033 | 78.36 | 92.46 | 96.31 | 0 |
| yolov8s_botsort_conf0.1 | MVI_40141 | 62.79 | 48.44 | 78.38 | 24 |
| yolov8s_botsort_conf0.1 | MVI_40171 | 58.82 | 48.10 | 63.26 | 14 |
| yolov8s_botsort_conf0.1 | MVI_40773 | 80.52 | 86.90 | 93.28 | 7 |
| yolov8s_botsort_conf0.1 | MVI_63553 | 66.88 | 74.90 | 84.68 | 29 |
| yolov8s_bytetrack_conf0.1 | MVI_20033 | 76.06 | 90.29 | 95.18 | 1 |
| yolov8s_bytetrack_conf0.1 | MVI_40141 | 58.71 | 46.51 | 73.32 | 32 |
| yolov8s_bytetrack_conf0.1 | MVI_40171 | 54.27 | 42.73 | 61.32 | 15 |
| yolov8s_bytetrack_conf0.1 | MVI_40773 | 78.12 | 86.78 | 92.40 | 8 |
| yolov8s_bytetrack_conf0.1 | MVI_63553 | 58.91 | 61.25 | 77.99 | 33 |
