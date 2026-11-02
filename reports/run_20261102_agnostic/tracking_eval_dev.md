# Tracking evaluation - dev split

Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, number of sequences: 5.
GT and tracker boxes with the centre inside an ignored region are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 66.37 | 60.27 | 73.58 | 67.32 | 83.62 | 80.22 | 66 |
| yolov8n_bytetrack_conf0.25 | 62.68 | 57.58 | 68.67 | 64.18 | 81.42 | 76.97 | 91 |
| yolov8s_botsort_conf0.25 | 69.70 | 63.87 | 76.70 | 70.90 | 84.00 | 83.32 | 48 |
| yolov8s_bytetrack_conf0.25 | 66.17 | 60.95 | 72.40 | 66.96 | 81.65 | 81.36 | 56 |

## Combined dataset metrics (TrackEval combination of all sequences)

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 65.41 | 65.72 | 79.08 |
| yolov8n_bytetrack_conf0.25 | 61.48 | 61.65 | 75.46 |
| yolov8s_botsort_conf0.25 | 68.74 | 69.32 | 82.09 |
| yolov8s_bytetrack_conf0.25 | 64.91 | 64.25 | 79.68 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20033 | 77.06 | 89.65 | 94.99 | 0 |
| yolov8n_botsort_conf0.25 | MVI_40141 | 59.42 | 46.62 | 75.54 | 8 |
| yolov8n_botsort_conf0.25 | MVI_40171 | 54.90 | 47.63 | 60.16 | 25 |
| yolov8n_botsort_conf0.25 | MVI_40773 | 76.90 | 84.05 | 90.35 | 7 |
| yolov8n_botsort_conf0.25 | MVI_63553 | 63.58 | 68.67 | 80.05 | 26 |
| yolov8n_bytetrack_conf0.25 | MVI_20033 | 75.91 | 89.60 | 94.94 | 1 |
| yolov8n_bytetrack_conf0.25 | MVI_40141 | 58.28 | 45.61 | 73.28 | 10 |
| yolov8n_bytetrack_conf0.25 | MVI_40171 | 49.57 | 43.47 | 55.11 | 33 |
| yolov8n_bytetrack_conf0.25 | MVI_40773 | 73.78 | 83.61 | 87.29 | 11 |
| yolov8n_bytetrack_conf0.25 | MVI_63553 | 55.88 | 58.59 | 74.25 | 36 |
| yolov8s_botsort_conf0.25 | MVI_20033 | 78.54 | 92.55 | 96.37 | 0 |
| yolov8s_botsort_conf0.25 | MVI_40141 | 64.10 | 53.89 | 80.22 | 16 |
| yolov8s_botsort_conf0.25 | MVI_40171 | 58.27 | 46.06 | 61.54 | 15 |
| yolov8s_botsort_conf0.25 | MVI_40773 | 79.95 | 86.44 | 92.63 | 3 |
| yolov8s_botsort_conf0.25 | MVI_63553 | 67.63 | 75.54 | 85.83 | 14 |
| yolov8s_bytetrack_conf0.25 | MVI_20033 | 76.88 | 92.04 | 96.09 | 0 |
| yolov8s_bytetrack_conf0.25 | MVI_40141 | 62.49 | 52.59 | 79.98 | 16 |
| yolov8s_bytetrack_conf0.25 | MVI_40171 | 54.16 | 41.29 | 59.46 | 18 |
| yolov8s_bytetrack_conf0.25 | MVI_40773 | 77.95 | 86.27 | 92.30 | 5 |
| yolov8s_bytetrack_conf0.25 | MVI_63553 | 59.37 | 62.62 | 78.98 | 17 |
