# Counting evaluation - test split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (7).

| class | GT total | yolov8n_botsort_conf0.25 | yolov8n_bytetrack_conf0.25 | yolov8s_botsort_conf0.25 | yolov8s_bytetrack_conf0.25 |
|---|---|---|---|---|---|
| car | 226 | 1.14 / 3.54% | 1.71 / 5.31% | 2.00 / 6.19% | 1.86 / 5.75% |
| bus | 22 | 0.71 / 22.73% | 1.00 / 31.82% | 0.86 / 27.27% | 1.00 / 31.82% |
| truck | 2 | 0.86 / 300.00% | 0.86 / 300.00% | 0.71 / 250.00% | 0.71 / 250.00% |
| all | 250 | 1.57 / 4.40% | 1.57 / 4.40% | 1.86 / 5.20% | 1.57 / 4.40% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | -1 | -1 | +0 | +2 | +0 | -1 | -3 |
| yolov8n_bytetrack_conf0.25 | -1 | -1 | +0 | +3 | +0 | -1 | -6 |
| yolov8s_botsort_conf0.25 | -1 | +0 | -1 | +4 | +2 | -1 | -5 |
| yolov8s_bytetrack_conf0.25 | -1 | -1 | -1 | +3 | +2 | -1 | -4 |

**bus**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | +0 | +1 | +0 | +0 | +4 |
| yolov8n_bytetrack_conf0.25 | +0 | +0 | +0 | +1 | +0 | +0 | +6 |
| yolov8s_botsort_conf0.25 | +0 | +0 | +1 | +1 | +0 | +0 | +4 |
| yolov8s_bytetrack_conf0.25 | +0 | +0 | +2 | +1 | +0 | +0 | +4 |

**truck**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +1 | +0 | +0 | +3 | +1 | +0 | +1 |
| yolov8n_bytetrack_conf0.25 | +1 | +0 | +0 | +3 | +1 | +0 | +1 |
| yolov8s_botsort_conf0.25 | +1 | +0 | +1 | +1 | +1 | +0 | -1 |
| yolov8s_bytetrack_conf0.25 | +1 | +1 | +0 | +1 | +1 | +0 | -1 |

**all**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | -1 | +0 | +6 | +1 | -1 | +2 |
| yolov8n_bytetrack_conf0.25 | +0 | -1 | +0 | +7 | +1 | -1 | +1 |
| yolov8s_botsort_conf0.25 | +0 | +0 | +1 | +6 | +3 | -1 | -2 |
| yolov8s_bytetrack_conf0.25 | +0 | +0 | +1 | +5 | +3 | -1 | -1 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.25: +1.00 / 43% / 29%, yolov8n_bytetrack_conf0.25: +1.00 / 43% / 29%, yolov8s_botsort_conf0.25: +1.00 / 43% / 29%, yolov8s_bytetrack_conf0.25: +1.00 / 43% / 29%

