# Tracking evaluation — dev split

Metrics: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, Number of sequences: 5.
GT and predicted observations with centres inside ignored regions are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | 64.95 | 58.77 | 72.36 | 67.62 | 84.19 | 79.07 | 59 |
| yolov8n_bytetrack_conf0.4 | 62.08 | 56.36 | 68.92 | 65.11 | 81.95 | 77.66 | 68 |
| yolov8s_botsort_conf0.4 | 69.96 | 64.29 | 76.72 | 73.03 | 84.52 | 83.92 | 47 |
| yolov8s_bytetrack_conf0.4 | 66.10 | 61.22 | 71.88 | 69.22 | 82.21 | 81.19 | 58 |

## Combined dataset metrics

Combine sequence statistics with TrackEval before calculating dataset metrics

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.4 | 63.86 | 65.14 | 77.71 |
| yolov8n_bytetrack_conf0.4 | 60.75 | 61.93 | 76.03 |
| yolov8s_botsort_conf0.4 | 69.11 | 71.52 | 82.95 |
| yolov8s_bytetrack_conf0.4 | 64.99 | 66.75 | 79.83 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | MVI_20033 | 76.40 | 91.46 | 95.65 | 0 |
| yolov8n_botsort_conf0.4 | MVI_40141 | 61.99 | 58.65 | 79.50 | 4 |
| yolov8n_botsort_conf0.4 | MVI_40171 | 57.22 | 50.68 | 62.94 | 18 |
| yolov8n_botsort_conf0.4 | MVI_40773 | 68.83 | 73.65 | 80.79 | 5 |
| yolov8n_botsort_conf0.4 | MVI_63553 | 60.30 | 63.68 | 76.47 | 32 |
| yolov8n_bytetrack_conf0.4 | MVI_20033 | 75.07 | 90.89 | 95.28 | 3 |
| yolov8n_bytetrack_conf0.4 | MVI_40141 | 60.74 | 57.97 | 78.81 | 7 |
| yolov8n_bytetrack_conf0.4 | MVI_40171 | 53.71 | 47.85 | 61.49 | 18 |
| yolov8n_bytetrack_conf0.4 | MVI_40773 | 67.33 | 73.32 | 80.55 | 7 |
| yolov8n_bytetrack_conf0.4 | MVI_63553 | 53.53 | 55.53 | 72.17 | 33 |
| yolov8s_botsort_conf0.4 | MVI_20033 | 78.64 | 94.08 | 97.04 | 0 |
| yolov8s_botsort_conf0.4 | MVI_40141 | 63.14 | 54.11 | 78.04 | 14 |
| yolov8s_botsort_conf0.4 | MVI_40171 | 61.34 | 55.48 | 66.74 | 9 |
| yolov8s_botsort_conf0.4 | MVI_40773 | 80.07 | 87.31 | 93.02 | 7 |
| yolov8s_botsort_conf0.4 | MVI_63553 | 66.61 | 74.16 | 84.76 | 17 |
| yolov8s_bytetrack_conf0.4 | MVI_20033 | 76.68 | 92.82 | 96.29 | 1 |
| yolov8s_bytetrack_conf0.4 | MVI_40141 | 59.86 | 52.94 | 74.74 | 18 |
| yolov8s_bytetrack_conf0.4 | MVI_40171 | 57.00 | 50.66 | 63.92 | 9 |
| yolov8s_bytetrack_conf0.4 | MVI_40773 | 78.15 | 86.97 | 92.32 | 5 |
| yolov8s_bytetrack_conf0.4 | MVI_63553 | 58.82 | 62.72 | 78.67 | 25 |
