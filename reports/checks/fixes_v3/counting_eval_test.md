# Counting evaluation - test split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (7).

| class | GT total | yolov8n_botsort_conf0.25 | yolov8n_bytetrack_conf0.25 | yolov8s_botsort_conf0.25 | yolov8s_bytetrack_conf0.25 |
|---|---|---|---|---|---|
| car | 226 | 1.86 / 5.75% | 2.29 / 7.08% | 2.29 / 7.08% | 2.14 / 6.64% |
| bus | 22 | 0.43 / 13.64% | 0.43 / 13.64% | 0.71 / 22.73% | 0.86 / 27.27% |
| truck | 2 | 0.71 / 250.00% | 0.86 / 300.00% | 0.71 / 250.00% | 0.71 / 250.00% |
| all | 250 | 1.86 / 5.20% | 1.86 / 5.20% | 2.29 / 6.40% | 2.00 / 5.60% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | -2 | -3 | +0 | +1 | +0 | -1 | -6 |
| yolov8n_bytetrack_conf0.25 | -2 | -3 | -1 | +1 | -1 | -1 | -7 |
| yolov8s_botsort_conf0.25 | -1 | -1 | -1 | +4 | +2 | -1 | -6 |
| yolov8s_bytetrack_conf0.25 | -1 | -2 | -1 | +2 | +2 | -1 | -6 |

**bus**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | +0 | +1 | +0 | +0 | +2 |
| yolov8n_bytetrack_conf0.25 | +0 | +0 | +0 | +0 | +0 | +0 | +3 |
| yolov8s_botsort_conf0.25 | +0 | +0 | +1 | +1 | +0 | +0 | +3 |
| yolov8s_bytetrack_conf0.25 | +0 | +0 | +2 | +1 | +0 | +0 | +3 |

**truck**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +1 | +0 | +0 | +2 | +1 | +0 | +1 |
| yolov8n_bytetrack_conf0.25 | +1 | +0 | +0 | +3 | +1 | +0 | +1 |
| yolov8s_botsort_conf0.25 | +1 | +0 | +1 | +1 | +1 | +0 | -1 |
| yolov8s_bytetrack_conf0.25 | +1 | +1 | +0 | +1 | +1 | +0 | -1 |

**all**

| config | MVI_20064 | MVI_39031 | MVI_39051 | MVI_39311 | MVI_39821 | MVI_40792 | MVI_40902 |
|---|---|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | -1 | -3 | +0 | +4 | +1 | -1 | -3 |
| yolov8n_bytetrack_conf0.25 | -1 | -3 | -1 | +4 | +0 | -1 | -3 |
| yolov8s_botsort_conf0.25 | +0 | -1 | +1 | +6 | +3 | -1 | -4 |
| yolov8s_bytetrack_conf0.25 | +0 | -1 | +1 | +4 | +3 | -1 | -4 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.25: -0.43 / 29% / 57%, yolov8n_bytetrack_conf0.25: -0.71 / 14% / 71%, yolov8s_botsort_conf0.25: +0.57 / 43% / 43%, yolov8s_bytetrack_conf0.25: +0.29 / 43% / 43%

