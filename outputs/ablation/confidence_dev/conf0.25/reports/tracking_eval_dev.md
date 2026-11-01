# Tracking evaluation — dev split

Metrics: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, Number of sequences: 5.
GT and predicted observations with centres inside ignored regions are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 65.25 | 59.65 | 71.89 | 67.10 | 83.78 | 78.71 | 100 |
| yolov8n_bytetrack_conf0.25 | 62.38 | 57.01 | 68.68 | 63.99 | 81.60 | 77.20 | 115 |
| yolov8s_botsort_conf0.25 | 69.76 | 63.95 | 76.71 | 71.17 | 84.13 | 83.64 | 64 |
| yolov8s_bytetrack_conf0.25 | 65.82 | 60.76 | 71.84 | 66.72 | 81.87 | 80.90 | 90 |

## Combined dataset metrics

Combine sequence statistics with TrackEval before calculating dataset metrics

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 64.24 | 65.19 | 77.48 |
| yolov8n_bytetrack_conf0.25 | 61.13 | 61.19 | 75.65 |
| yolov8s_botsort_conf0.25 | 68.81 | 69.83 | 82.50 |
| yolov8s_bytetrack_conf0.25 | 64.62 | 64.33 | 79.38 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20033 | 77.09 | 90.71 | 95.43 | 0 |
| yolov8n_botsort_conf0.25 | MVI_40141 | 60.04 | 50.14 | 76.37 | 15 |
| yolov8n_botsort_conf0.25 | MVI_40171 | 55.16 | 49.34 | 60.43 | 27 |
| yolov8n_botsort_conf0.25 | MVI_40773 | 71.74 | 78.52 | 83.01 | 14 |
| yolov8n_botsort_conf0.25 | MVI_63553 | 62.22 | 66.79 | 78.31 | 44 |
| yolov8n_bytetrack_conf0.25 | MVI_20033 | 75.77 | 90.47 | 95.07 | 4 |
| yolov8n_bytetrack_conf0.25 | MVI_40141 | 58.68 | 49.02 | 73.97 | 20 |
| yolov8n_bytetrack_conf0.25 | MVI_40171 | 51.16 | 45.36 | 58.12 | 31 |
| yolov8n_bytetrack_conf0.25 | MVI_40773 | 71.33 | 78.15 | 85.85 | 17 |
| yolov8n_bytetrack_conf0.25 | MVI_63553 | 54.93 | 56.94 | 73.01 | 43 |
| yolov8s_botsort_conf0.25 | MVI_20033 | 78.66 | 93.24 | 96.68 | 0 |
| yolov8s_botsort_conf0.25 | MVI_40141 | 62.83 | 49.05 | 78.45 | 23 |
| yolov8s_botsort_conf0.25 | MVI_40171 | 59.69 | 51.23 | 64.57 | 15 |
| yolov8s_botsort_conf0.25 | MVI_40773 | 80.62 | 87.29 | 93.47 | 2 |
| yolov8s_botsort_conf0.25 | MVI_63553 | 67.00 | 75.03 | 85.02 | 24 |
| yolov8s_bytetrack_conf0.25 | MVI_20033 | 76.50 | 91.33 | 95.69 | 1 |
| yolov8s_bytetrack_conf0.25 | MVI_40141 | 60.04 | 47.73 | 75.58 | 30 |
| yolov8s_bytetrack_conf0.25 | MVI_40171 | 55.42 | 45.66 | 62.58 | 17 |
| yolov8s_bytetrack_conf0.25 | MVI_40773 | 78.09 | 87.00 | 92.31 | 15 |
| yolov8s_bytetrack_conf0.25 | MVI_63553 | 59.06 | 61.89 | 78.35 | 27 |
