# Tracking evaluation — dev split

Метрики: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05–0.95), **MOTA** (1 − (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), **IDF1** (F1 of identity matching).

Конфигов: 4, последовательностей: 5.

## Aggregate (overall, class-agnostic)

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 54.59 | 46.85 | 66.68 | 51.18 | 84.05 | 64.08 | 99 |
| yolov8n_bytetrack_conf0.25 | 51.82 | 44.36 | 63.46 | 48.10 | 81.72 | 62.49 | 112 |
| yolov8s_botsort_conf0.25 | 58.84 | 50.76 | 71.47 | 54.72 | 84.42 | 68.71 | 62 |
| yolov8s_bytetrack_conf0.25 | 55.23 | 47.95 | 66.86 | 50.64 | 82.04 | 66.11 | 85 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20033 | 23.61 | 11.18 | 21.73 | 0 |
| yolov8n_botsort_conf0.25 | MVI_40141 | 60.03 | 50.16 | 76.37 | 15 |
| yolov8n_botsort_conf0.25 | MVI_40171 | 55.17 | 49.36 | 60.44 | 27 |
| yolov8n_botsort_conf0.25 | MVI_40773 | 71.74 | 78.44 | 83.01 | 14 |
| yolov8n_botsort_conf0.25 | MVI_63553 | 62.40 | 66.76 | 78.83 | 43 |
| yolov8n_bytetrack_conf0.25 | MVI_20033 | 23.07 | 11.18 | 21.54 | 1 |
| yolov8n_bytetrack_conf0.25 | MVI_40141 | 58.69 | 49.08 | 73.98 | 20 |
| yolov8n_bytetrack_conf0.25 | MVI_40171 | 51.18 | 45.32 | 58.10 | 31 |
| yolov8n_bytetrack_conf0.25 | MVI_40773 | 71.30 | 78.07 | 85.83 | 17 |
| yolov8n_bytetrack_conf0.25 | MVI_63553 | 54.89 | 56.87 | 72.97 | 43 |
| yolov8s_botsort_conf0.25 | MVI_20033 | 24.11 | 11.20 | 22.01 | 0 |
| yolov8s_botsort_conf0.25 | MVI_40141 | 62.81 | 48.97 | 78.43 | 23 |
| yolov8s_botsort_conf0.25 | MVI_40171 | 59.68 | 51.23 | 64.56 | 15 |
| yolov8s_botsort_conf0.25 | MVI_40773 | 80.65 | 87.31 | 93.49 | 2 |
| yolov8s_botsort_conf0.25 | MVI_63553 | 66.96 | 74.87 | 85.07 | 22 |
| yolov8s_bytetrack_conf0.25 | MVI_20033 | 23.63 | 11.02 | 21.85 | 0 |
| yolov8s_bytetrack_conf0.25 | MVI_40141 | 60.02 | 47.67 | 75.57 | 30 |
| yolov8s_bytetrack_conf0.25 | MVI_40171 | 55.42 | 45.68 | 62.58 | 17 |
| yolov8s_bytetrack_conf0.25 | MVI_40773 | 78.07 | 87.05 | 92.25 | 11 |
| yolov8s_bytetrack_conf0.25 | MVI_63553 | 59.00 | 61.79 | 78.30 | 27 |
