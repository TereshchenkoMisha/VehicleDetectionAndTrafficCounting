# Tracking evaluation — dev split

Metrics: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, Number of sequences: 5.
GT and predicted observations with centres inside ignored regions are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | 57.04 | 49.65 | 66.05 | 57.15 | 85.16 | 70.94 | 50 |
| yolov8n_bytetrack_conf0.6 | 54.76 | 47.83 | 63.18 | 55.79 | 82.96 | 70.07 | 49 |
| yolov8s_botsort_conf0.6 | 67.06 | 60.98 | 74.32 | 69.96 | 85.27 | 81.51 | 27 |
| yolov8s_bytetrack_conf0.6 | 64.00 | 58.37 | 70.70 | 67.66 | 82.92 | 80.17 | 30 |

## Combined dataset metrics

Combine sequence statistics with TrackEval before calculating dataset metrics

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.6 | 56.48 | 55.28 | 69.65 |
| yolov8n_bytetrack_conf0.6 | 53.90 | 53.52 | 68.54 |
| yolov8s_botsort_conf0.6 | 66.56 | 68.49 | 80.69 |
| yolov8s_bytetrack_conf0.6 | 63.21 | 65.54 | 79.05 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | MVI_20033 | 67.63 | 77.83 | 87.78 | 1 |
| yolov8n_botsort_conf0.6 | MVI_40141 | 50.42 | 49.29 | 67.35 | 7 |
| yolov8n_botsort_conf0.6 | MVI_40171 | 55.86 | 48.55 | 60.88 | 6 |
| yolov8n_botsort_conf0.6 | MVI_40773 | 56.19 | 55.39 | 69.45 | 7 |
| yolov8n_botsort_conf0.6 | MVI_63553 | 55.10 | 54.70 | 69.28 | 29 |
| yolov8n_bytetrack_conf0.6 | MVI_20033 | 66.69 | 77.29 | 87.48 | 1 |
| yolov8n_bytetrack_conf0.6 | MVI_40141 | 49.36 | 49.31 | 67.19 | 6 |
| yolov8n_bytetrack_conf0.6 | MVI_40171 | 52.97 | 46.94 | 59.93 | 5 |
| yolov8n_bytetrack_conf0.6 | MVI_40773 | 55.36 | 55.35 | 69.43 | 7 |
| yolov8n_bytetrack_conf0.6 | MVI_63553 | 49.44 | 50.05 | 66.30 | 30 |
| yolov8s_botsort_conf0.6 | MVI_20033 | 74.48 | 88.18 | 93.85 | 0 |
| yolov8s_botsort_conf0.6 | MVI_40141 | 59.24 | 54.89 | 76.11 | 4 |
| yolov8s_botsort_conf0.6 | MVI_40171 | 61.82 | 55.74 | 67.67 | 4 |
| yolov8s_botsort_conf0.6 | MVI_40773 | 75.76 | 81.77 | 88.44 | 1 |
| yolov8s_botsort_conf0.6 | MVI_63553 | 64.01 | 69.20 | 81.48 | 18 |
| yolov8s_bytetrack_conf0.6 | MVI_20033 | 72.93 | 87.41 | 93.44 | 0 |
| yolov8s_bytetrack_conf0.6 | MVI_40141 | 57.75 | 54.79 | 76.05 | 4 |
| yolov8s_bytetrack_conf0.6 | MVI_40171 | 57.74 | 52.65 | 65.80 | 4 |
| yolov8s_bytetrack_conf0.6 | MVI_40773 | 74.45 | 81.65 | 88.07 | 2 |
| yolov8s_bytetrack_conf0.6 | MVI_63553 | 57.12 | 61.78 | 77.48 | 20 |
