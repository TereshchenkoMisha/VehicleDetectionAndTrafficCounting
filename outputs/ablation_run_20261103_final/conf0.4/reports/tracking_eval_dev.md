# Tracking evaluation - dev split

Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, number of sequences: 5.
GT and tracker boxes with the centre inside an ignored region are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | 67.64 | 61.20 | 75.37 | 69.54 | 83.71 | 81.87 | 34 |
| yolov8n_bytetrack_conf0.4 | 64.33 | 58.67 | 71.13 | 66.91 | 81.56 | 79.73 | 40 |
| yolov8s_botsort_conf0.4 | 70.32 | 64.68 | 77.05 | 72.74 | 84.12 | 84.42 | 28 |
| yolov8s_bytetrack_conf0.4 | 66.71 | 61.72 | 72.63 | 69.10 | 81.80 | 81.96 | 45 |

## Combined dataset metrics (TrackEval combination of all sequences)

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.4 | 66.56 | 67.45 | 80.66 |
| yolov8n_bytetrack_conf0.4 | 63.05 | 64.10 | 78.30 |
| yolov8s_botsort_conf0.4 | 69.37 | 71.27 | 83.32 |
| yolov8s_bytetrack_conf0.4 | 65.49 | 66.64 | 80.46 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | MVI_20033 | 78.82 | 94.50 | 97.27 | 0 |
| yolov8n_botsort_conf0.4 | MVI_40141 | 62.42 | 51.24 | 79.45 | 1 |
| yolov8n_botsort_conf0.4 | MVI_40171 | 57.34 | 50.60 | 63.27 | 10 |
| yolov8n_botsort_conf0.4 | MVI_40773 | 75.97 | 83.04 | 89.09 | 4 |
| yolov8n_botsort_conf0.4 | MVI_63553 | 63.64 | 68.30 | 80.28 | 19 |
| yolov8n_bytetrack_conf0.4 | MVI_20033 | 77.50 | 94.04 | 96.94 | 1 |
| yolov8n_bytetrack_conf0.4 | MVI_40141 | 61.25 | 50.43 | 78.57 | 2 |
| yolov8n_bytetrack_conf0.4 | MVI_40171 | 53.39 | 47.56 | 61.02 | 11 |
| yolov8n_bytetrack_conf0.4 | MVI_40773 | 72.92 | 82.50 | 86.08 | 7 |
| yolov8n_bytetrack_conf0.4 | MVI_63553 | 56.61 | 59.99 | 76.06 | 19 |
| yolov8s_botsort_conf0.4 | MVI_20033 | 79.47 | 95.17 | 97.61 | 0 |
| yolov8s_botsort_conf0.4 | MVI_40141 | 63.80 | 52.01 | 79.90 | 5 |
| yolov8s_botsort_conf0.4 | MVI_40171 | 60.57 | 53.91 | 65.71 | 9 |
| yolov8s_botsort_conf0.4 | MVI_40773 | 80.26 | 87.22 | 93.13 | 2 |
| yolov8s_botsort_conf0.4 | MVI_63553 | 67.48 | 75.41 | 85.76 | 12 |
| yolov8s_bytetrack_conf0.4 | MVI_20033 | 77.58 | 94.01 | 96.91 | 1 |
| yolov8s_bytetrack_conf0.4 | MVI_40141 | 61.61 | 51.19 | 77.75 | 12 |
| yolov8s_bytetrack_conf0.4 | MVI_40171 | 56.40 | 49.29 | 63.04 | 9 |
| yolov8s_bytetrack_conf0.4 | MVI_40773 | 78.49 | 87.18 | 92.74 | 3 |
| yolov8s_bytetrack_conf0.4 | MVI_63553 | 59.48 | 63.82 | 79.38 | 20 |
