# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.4 | yolov8n_bytetrack_conf0.4 | yolov8s_botsort_conf0.4 | yolov8s_bytetrack_conf0.4 |
|---|---|---|---|---|---|
| car | 245 | 11.60 / 23.67% | 11.40 / 23.27% | 6.00 / 12.24% | 6.00 / 12.24% |
| bus | 5 | 0.20 / 20.00% | 0.20 / 20.00% | 0.40 / 40.00% | 0.60 / 60.00% |
| truck | 1 | 0.40 / 200.00% | 0.40 / 200.00% | 0.80 / 400.00% | 0.60 / 300.00% |
| all | 251 | 11.40 / 22.71% | 11.20 / 22.31% | 5.60 / 11.16% | 5.60 / 11.16% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | -1 | +0 | -9 | -3 | -45 |
| yolov8n_bytetrack_conf0.4 | -1 | +0 | -9 | -3 | -44 |
| yolov8s_botsort_conf0.4 | -1 | -1 | -2 | +0 | -26 |
| yolov8s_bytetrack_conf0.4 | -1 | +1 | -2 | +0 | -26 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | +0 | +0 | +0 | +0 | +1 |
| yolov8n_bytetrack_conf0.4 | +0 | +0 | +0 | +0 | +1 |
| yolov8s_botsort_conf0.4 | +0 | +0 | +0 | +0 | +2 |
| yolov8s_bytetrack_conf0.4 | +0 | +0 | +0 | +0 | +3 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | +1 | +0 | +0 | +0 | -1 |
| yolov8n_bytetrack_conf0.4 | +1 | +0 | +0 | +0 | -1 |
| yolov8s_botsort_conf0.4 | +1 | +2 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.4 | +1 | +1 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | +0 | +0 | -9 | -3 | -45 |
| yolov8n_bytetrack_conf0.4 | +0 | +0 | -9 | -3 | -44 |
| yolov8s_botsort_conf0.4 | +0 | +1 | -2 | +0 | -25 |
| yolov8s_bytetrack_conf0.4 | +0 | +2 | -2 | +0 | -24 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.4: -11.40 / 0% / 60%, yolov8n_bytetrack_conf0.4: -11.20 / 0% / 60%, yolov8s_botsort_conf0.4: -5.20 / 20% / 40%, yolov8s_bytetrack_conf0.4: -4.80 / 20% / 40%

