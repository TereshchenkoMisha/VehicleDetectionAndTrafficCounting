# Tracking evaluation — test split

Метрики: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05–0.95), **MOTA** (1 − (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), **IDF1** (F1 of identity matching).

Конфигов: 4, последовательностей: 7.

## Aggregate (overall, class-agnostic)

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 67.45 | 63.97 | 71.42 | 68.67 | 84.87 | 81.57 | 157 |
| yolov8n_bytetrack_conf0.25 | 65.20 | 61.98 | 68.90 | 67.70 | 83.41 | 80.40 | 199 |
| yolov8s_botsort_conf0.25 | 70.77 | 66.38 | 75.95 | 70.19 | 85.46 | 84.00 | 93 |
| yolov8s_bytetrack_conf0.25 | 68.33 | 64.16 | 73.27 | 69.19 | 83.78 | 83.24 | 120 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20064 | 63.62 | 70.82 | 80.44 | 20 |
| yolov8n_botsort_conf0.25 | MVI_39031 | 76.73 | 88.94 | 93.99 | 2 |
| yolov8n_botsort_conf0.25 | MVI_39051 | 69.56 | 72.33 | 86.28 | 4 |
| yolov8n_botsort_conf0.25 | MVI_39311 | 58.56 | 68.84 | 73.98 | 71 |
| yolov8n_botsort_conf0.25 | MVI_39821 | 53.50 | 27.61 | 63.68 | 26 |
| yolov8n_botsort_conf0.25 | MVI_40792 | 80.11 | 82.69 | 90.55 | 14 |
| yolov8n_botsort_conf0.25 | MVI_40902 | 70.06 | 69.44 | 82.10 | 20 |
| yolov8n_bytetrack_conf0.25 | MVI_20064 | 59.81 | 69.54 | 75.82 | 41 |
| yolov8n_bytetrack_conf0.25 | MVI_39031 | 73.55 | 87.11 | 92.92 | 9 |
| yolov8n_bytetrack_conf0.25 | MVI_39051 | 66.08 | 70.82 | 84.95 | 5 |
| yolov8n_bytetrack_conf0.25 | MVI_39311 | 59.06 | 68.89 | 74.85 | 74 |
| yolov8n_bytetrack_conf0.25 | MVI_39821 | 52.84 | 28.46 | 64.07 | 25 |
| yolov8n_bytetrack_conf0.25 | MVI_40792 | 77.96 | 81.17 | 89.97 | 13 |
| yolov8n_bytetrack_conf0.25 | MVI_40902 | 67.07 | 67.92 | 80.22 | 32 |
| yolov8s_botsort_conf0.25 | MVI_20064 | 66.95 | 75.18 | 81.78 | 22 |
| yolov8s_botsort_conf0.25 | MVI_39031 | 78.92 | 91.92 | 95.63 | 1 |
| yolov8s_botsort_conf0.25 | MVI_39051 | 74.14 | 74.09 | 88.04 | 1 |
| yolov8s_botsort_conf0.25 | MVI_39311 | 61.22 | 70.30 | 75.92 | 38 |
| yolov8s_botsort_conf0.25 | MVI_39821 | 56.62 | 20.96 | 66.61 | 22 |
| yolov8s_botsort_conf0.25 | MVI_40792 | 82.01 | 83.16 | 91.66 | 7 |
| yolov8s_botsort_conf0.25 | MVI_40902 | 75.57 | 75.72 | 88.34 | 2 |
| yolov8s_bytetrack_conf0.25 | MVI_20064 | 64.03 | 74.67 | 79.20 | 35 |
| yolov8s_bytetrack_conf0.25 | MVI_39031 | 75.17 | 89.18 | 94.13 | 6 |
| yolov8s_bytetrack_conf0.25 | MVI_39051 | 70.32 | 73.71 | 87.53 | 2 |
| yolov8s_bytetrack_conf0.25 | MVI_39311 | 60.80 | 70.19 | 75.85 | 41 |
| yolov8s_bytetrack_conf0.25 | MVI_39821 | 56.08 | 21.73 | 68.12 | 17 |
| yolov8s_bytetrack_conf0.25 | MVI_40792 | 79.46 | 81.46 | 90.93 | 7 |
| yolov8s_bytetrack_conf0.25 | MVI_40902 | 72.49 | 73.40 | 86.96 | 12 |
