# Tracking evaluation — dev split

Metrics: **HOTA** (geometry mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (count of identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, Number of sequences: 5.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | 65.25 | 59.65 | 71.89 | 67.09 | 83.79 | 78.71 | 100 |
| yolov8n_bytetrack_conf0.25 | 62.36 | 57.00 | 68.67 | 63.96 | 81.61 | 77.19 | 115 |
| yolov8s_botsort_conf0.25 | 69.74 | 63.92 | 76.69 | 71.14 | 84.13 | 83.60 | 65 |
| yolov8s_bytetrack_conf0.25 | 65.80 | 60.73 | 71.82 | 66.70 | 81.87 | 80.88 | 86 |

## Combined dataset metrics

Combine sequence statistics with TrackEval before calculating dataset metrics

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.25 | 64.24 | 65.18 | 77.48 |
| yolov8n_bytetrack_conf0.25 | 61.12 | 61.16 | 75.63 |
| yolov8s_botsort_conf0.25 | 68.78 | 69.80 | 82.46 |
| yolov8s_bytetrack_conf0.25 | 64.60 | 64.30 | 79.35 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | MVI_20033 | 77.10 | 90.69 | 95.42 | 0 |
| yolov8n_botsort_conf0.25 | MVI_40141 | 60.03 | 50.16 | 76.37 | 15 |
| yolov8n_botsort_conf0.25 | MVI_40171 | 55.17 | 49.36 | 60.44 | 27 |
| yolov8n_botsort_conf0.25 | MVI_40773 | 71.74 | 78.44 | 83.01 | 14 |
| yolov8n_botsort_conf0.25 | MVI_63553 | 62.22 | 66.78 | 78.31 | 44 |
| yolov8n_bytetrack_conf0.25 | MVI_20033 | 75.76 | 90.44 | 95.06 | 4 |
| yolov8n_bytetrack_conf0.25 | MVI_40141 | 58.69 | 49.08 | 73.98 | 20 |
| yolov8n_bytetrack_conf0.25 | MVI_40171 | 51.18 | 45.32 | 58.10 | 31 |
| yolov8n_bytetrack_conf0.25 | MVI_40773 | 71.30 | 78.07 | 85.83 | 17 |
| yolov8n_bytetrack_conf0.25 | MVI_63553 | 54.89 | 56.87 | 72.97 | 43 |
| yolov8s_botsort_conf0.25 | MVI_20033 | 78.68 | 93.26 | 96.69 | 0 |
| yolov8s_botsort_conf0.25 | MVI_40141 | 62.81 | 48.97 | 78.43 | 23 |
| yolov8s_botsort_conf0.25 | MVI_40171 | 59.68 | 51.23 | 64.56 | 15 |
| yolov8s_botsort_conf0.25 | MVI_40773 | 80.64 | 87.31 | 93.49 | 2 |
| yolov8s_botsort_conf0.25 | MVI_63553 | 66.87 | 74.93 | 84.84 | 25 |
| yolov8s_bytetrack_conf0.25 | MVI_20033 | 76.49 | 91.33 | 95.69 | 1 |
| yolov8s_bytetrack_conf0.25 | MVI_40141 | 60.02 | 47.67 | 75.57 | 30 |
| yolov8s_bytetrack_conf0.25 | MVI_40171 | 55.42 | 45.68 | 62.58 | 17 |
| yolov8s_bytetrack_conf0.25 | MVI_40773 | 78.07 | 87.05 | 92.25 | 11 |
| yolov8s_bytetrack_conf0.25 | MVI_63553 | 59.00 | 61.79 | 78.30 | 27 |
