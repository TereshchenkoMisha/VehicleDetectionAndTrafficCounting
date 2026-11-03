# Tracking evaluation - test split

Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, number of sequences: 7.
GT and tracker boxes with the centre inside an ignored region are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 67.60 | 63.81 | 71.96 | 67.38 | 84.58 | 81.08 | 122 |
| yolov8n_bytetrack_conf0.25 | 65.40 | 61.76 | 69.65 | 66.28 | 83.15 | 80.01 | 165 |
| yolov8s_botsort_conf0.25 | 70.60 | 65.77 | 76.39 | 67.51 | 85.32 | 83.58 | 76 |
| yolov8s_bytetrack_conf0.25 | 68.10 | 63.69 | 73.41 | 66.86 | 83.66 | 82.39 | 104 |

## Combined dataset metrics (TrackEval combination of all sequences)

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 66.18 | 69.93 | 79.48 |
| yolov8n_bytetrack_conf0.25 | 64.51 | 69.05 | 78.66 |
| yolov8s_botsort_conf0.25 | 69.19 | 71.79 | 82.26 |
| yolov8s_bytetrack_conf0.25 | 67.02 | 70.98 | 80.91 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20064 | 65.09 | 74.69 | 80.19 | 17 |
| yolov8n_botsort_conf0.25 | MVI_39031 | 77.35 | 90.21 | 94.51 | 3 |
| yolov8n_botsort_conf0.25 | MVI_39051 | 70.14 | 71.49 | 86.33 | 3 |
| yolov8n_botsort_conf0.25 | MVI_39311 | 58.85 | 67.23 | 73.42 | 53 |
| yolov8n_botsort_conf0.25 | MVI_39821 | 52.88 | 19.98 | 62.54 | 19 |
| yolov8n_botsort_conf0.25 | MVI_40792 | 79.26 | 79.83 | 89.36 | 11 |
| yolov8n_botsort_conf0.25 | MVI_40902 | 69.61 | 68.21 | 81.23 | 16 |
| yolov8n_bytetrack_conf0.25 | MVI_20064 | 61.59 | 73.02 | 76.21 | 37 |
| yolov8n_bytetrack_conf0.25 | MVI_39031 | 74.34 | 88.64 | 93.96 | 10 |
| yolov8n_bytetrack_conf0.25 | MVI_39051 | 66.45 | 69.57 | 84.62 | 5 |
| yolov8n_bytetrack_conf0.25 | MVI_39311 | 59.20 | 67.34 | 74.06 | 53 |
| yolov8n_bytetrack_conf0.25 | MVI_39821 | 52.24 | 20.36 | 62.56 | 21 |
| yolov8n_bytetrack_conf0.25 | MVI_40792 | 77.06 | 78.35 | 88.81 | 8 |
| yolov8n_bytetrack_conf0.25 | MVI_40902 | 66.93 | 66.70 | 79.87 | 31 |
| yolov8s_botsort_conf0.25 | MVI_20064 | 69.98 | 79.32 | 84.78 | 18 |
| yolov8s_botsort_conf0.25 | MVI_39031 | 79.00 | 92.19 | 95.80 | 1 |
| yolov8s_botsort_conf0.25 | MVI_39051 | 72.78 | 68.94 | 86.08 | 1 |
| yolov8s_botsort_conf0.25 | MVI_39311 | 61.48 | 68.86 | 75.90 | 30 |
| yolov8s_botsort_conf0.25 | MVI_39821 | 55.15 | 6.65 | 65.33 | 18 |
| yolov8s_botsort_conf0.25 | MVI_40792 | 81.53 | 82.17 | 91.25 | 5 |
| yolov8s_botsort_conf0.25 | MVI_40902 | 74.27 | 74.42 | 85.92 | 3 |
| yolov8s_bytetrack_conf0.25 | MVI_20064 | 65.99 | 77.98 | 80.16 | 28 |
| yolov8s_bytetrack_conf0.25 | MVI_39031 | 75.48 | 89.57 | 94.43 | 5 |
| yolov8s_bytetrack_conf0.25 | MVI_39051 | 69.61 | 69.86 | 86.07 | 2 |
| yolov8s_bytetrack_conf0.25 | MVI_39311 | 60.69 | 68.90 | 75.31 | 34 |
| yolov8s_bytetrack_conf0.25 | MVI_39821 | 54.21 | 8.58 | 65.03 | 17 |
| yolov8s_bytetrack_conf0.25 | MVI_40792 | 79.15 | 80.60 | 90.57 | 5 |
| yolov8s_bytetrack_conf0.25 | MVI_40902 | 71.57 | 72.56 | 85.17 | 13 |
