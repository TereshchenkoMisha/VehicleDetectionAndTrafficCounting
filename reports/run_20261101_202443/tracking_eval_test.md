# Tracking evaluation - test split

Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, number of sequences: 7.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 67.58 | 63.78 | 71.95 | 67.30 | 84.58 | 81.06 | 122 |
| yolov8n_bytetrack_conf0.25 | 65.39 | 61.75 | 69.65 | 66.26 | 83.15 | 80.01 | 165 |
| yolov8s_botsort_conf0.25 | 70.56 | 65.73 | 76.37 | 67.43 | 85.32 | 83.55 | 76 |
| yolov8s_bytetrack_conf0.25 | 68.09 | 63.67 | 73.40 | 66.81 | 83.66 | 82.38 | 104 |

## Combined dataset metrics (TrackEval combination of all sequences)

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 66.17 | 69.88 | 79.47 |
| yolov8n_bytetrack_conf0.25 | 64.50 | 69.03 | 78.65 |
| yolov8s_botsort_conf0.25 | 69.16 | 71.72 | 82.23 |
| yolov8s_bytetrack_conf0.25 | 67.01 | 70.95 | 80.90 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20064 | 65.10 | 74.69 | 80.19 | 17 |
| yolov8n_botsort_conf0.25 | MVI_39031 | 77.33 | 90.18 | 94.49 | 3 |
| yolov8n_botsort_conf0.25 | MVI_39051 | 70.12 | 71.45 | 86.32 | 3 |
| yolov8n_botsort_conf0.25 | MVI_39311 | 58.85 | 67.21 | 73.41 | 53 |
| yolov8n_botsort_conf0.25 | MVI_39821 | 52.80 | 19.58 | 62.44 | 19 |
| yolov8n_botsort_conf0.25 | MVI_40792 | 79.25 | 79.82 | 89.35 | 11 |
| yolov8n_botsort_conf0.25 | MVI_40902 | 69.58 | 68.16 | 81.21 | 16 |
| yolov8n_bytetrack_conf0.25 | MVI_20064 | 61.58 | 73.02 | 76.20 | 37 |
| yolov8n_bytetrack_conf0.25 | MVI_39031 | 74.32 | 88.61 | 93.95 | 10 |
| yolov8n_bytetrack_conf0.25 | MVI_39051 | 66.45 | 69.57 | 84.62 | 5 |
| yolov8n_bytetrack_conf0.25 | MVI_39311 | 59.21 | 67.35 | 74.06 | 53 |
| yolov8n_bytetrack_conf0.25 | MVI_39821 | 52.22 | 20.23 | 62.53 | 21 |
| yolov8n_bytetrack_conf0.25 | MVI_40792 | 77.07 | 78.37 | 88.82 | 8 |
| yolov8n_bytetrack_conf0.25 | MVI_40902 | 66.91 | 66.65 | 79.85 | 31 |
| yolov8s_botsort_conf0.25 | MVI_20064 | 69.95 | 79.25 | 84.75 | 18 |
| yolov8s_botsort_conf0.25 | MVI_39031 | 78.96 | 92.13 | 95.77 | 1 |
| yolov8s_botsort_conf0.25 | MVI_39051 | 72.76 | 68.90 | 86.06 | 1 |
| yolov8s_botsort_conf0.25 | MVI_39311 | 61.46 | 68.82 | 75.89 | 30 |
| yolov8s_botsort_conf0.25 | MVI_39821 | 55.10 | 6.45 | 65.28 | 18 |
| yolov8s_botsort_conf0.25 | MVI_40792 | 81.50 | 82.11 | 91.23 | 5 |
| yolov8s_botsort_conf0.25 | MVI_40902 | 74.22 | 74.32 | 85.88 | 3 |
| yolov8s_bytetrack_conf0.25 | MVI_20064 | 65.99 | 77.96 | 80.15 | 28 |
| yolov8s_bytetrack_conf0.25 | MVI_39031 | 75.46 | 89.54 | 94.42 | 5 |
| yolov8s_bytetrack_conf0.25 | MVI_39051 | 69.61 | 69.86 | 86.07 | 2 |
| yolov8s_bytetrack_conf0.25 | MVI_39311 | 60.68 | 68.89 | 75.31 | 34 |
| yolov8s_bytetrack_conf0.25 | MVI_39821 | 54.17 | 8.35 | 64.98 | 17 |
| yolov8s_bytetrack_conf0.25 | MVI_40792 | 79.13 | 80.56 | 90.56 | 5 |
| yolov8s_bytetrack_conf0.25 | MVI_40902 | 71.55 | 72.53 | 85.16 | 13 |
