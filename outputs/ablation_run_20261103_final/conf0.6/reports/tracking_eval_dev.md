# Tracking evaluation - dev split

Metrics: **HOTA** (geometric mean of DetA & AssA, averaged over IoU 0.05-0.95), **MOTA** (1 - (FN+FP+IDSW)/GT), **IDSW** (identity switches), **IDF1** (F1 of identity matching).

Number of configs: 4, number of sequences: 5.
GT and tracker boxes with the centre inside an ignored region are excluded.

## Macro average across sequences

| Config | HOTA (%) | DetA (%) | AssA (%) | MOTA (%) | MOTP (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | 66.20 | 59.02 | 74.88 | 67.66 | 84.07 | 80.56 | 20 |
| yolov8n_bytetrack_conf0.6 | 63.26 | 56.77 | 71.09 | 65.77 | 82.13 | 78.94 | 23 |
| yolov8s_botsort_conf0.6 | 70.80 | 64.96 | 77.75 | 74.62 | 84.56 | 85.37 | 14 |
| yolov8s_bytetrack_conf0.6 | 67.45 | 62.27 | 73.57 | 72.23 | 82.34 | 83.43 | 17 |

## Combined dataset metrics (TrackEval combination of all sequences)

| Config | HOTA (%) | MOTA (%) | IDF1 (%) |
|---|---|---|---|
| yolov8n_botsort_conf0.6 | 65.20 | 65.49 | 79.35 |
| yolov8n_bytetrack_conf0.6 | 62.03 | 63.11 | 77.54 |
| yolov8s_botsort_conf0.6 | 69.93 | 72.82 | 84.40 |
| yolov8s_bytetrack_conf0.6 | 66.32 | 69.78 | 82.20 |

## Per-sequence (HOTA / MOTA / IDF1 / IDSW)

| Config | Sequence | HOTA (%) | MOTA (%) | IDF1 (%) | IDSW |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | MVI_20033 | 76.94 | 92.11 | 95.99 | 0 |
| yolov8n_botsort_conf0.6 | MVI_40141 | 61.61 | 53.63 | 79.14 | 3 |
| yolov8n_botsort_conf0.6 | MVI_40171 | 58.05 | 51.12 | 63.74 | 3 |
| yolov8n_botsort_conf0.6 | MVI_40773 | 71.68 | 75.66 | 85.50 | 1 |
| yolov8n_botsort_conf0.6 | MVI_63553 | 62.70 | 65.78 | 78.43 | 13 |
| yolov8n_bytetrack_conf0.6 | MVI_20033 | 75.95 | 91.64 | 95.75 | 0 |
| yolov8n_bytetrack_conf0.6 | MVI_40141 | 60.55 | 53.33 | 78.79 | 3 |
| yolov8n_bytetrack_conf0.6 | MVI_40171 | 54.67 | 49.19 | 62.18 | 4 |
| yolov8n_bytetrack_conf0.6 | MVI_40773 | 69.10 | 74.65 | 82.99 | 2 |
| yolov8n_bytetrack_conf0.6 | MVI_63553 | 56.01 | 60.05 | 74.96 | 14 |
| yolov8s_botsort_conf0.6 | MVI_20033 | 79.65 | 95.83 | 97.92 | 0 |
| yolov8s_botsort_conf0.6 | MVI_40141 | 65.39 | 59.55 | 82.52 | 0 |
| yolov8s_botsort_conf0.6 | MVI_40171 | 62.11 | 56.88 | 68.29 | 4 |
| yolov8s_botsort_conf0.6 | MVI_40773 | 79.79 | 86.77 | 93.09 | 0 |
| yolov8s_botsort_conf0.6 | MVI_63553 | 67.05 | 74.07 | 85.02 | 10 |
| yolov8s_bytetrack_conf0.6 | MVI_20033 | 78.17 | 95.12 | 97.56 | 0 |
| yolov8s_bytetrack_conf0.6 | MVI_40141 | 62.98 | 59.03 | 80.07 | 1 |
| yolov8s_bytetrack_conf0.6 | MVI_40171 | 58.09 | 53.95 | 66.38 | 4 |
| yolov8s_bytetrack_conf0.6 | MVI_40773 | 78.18 | 86.63 | 92.36 | 1 |
| yolov8s_bytetrack_conf0.6 | MVI_63553 | 59.84 | 66.43 | 80.79 | 11 |
