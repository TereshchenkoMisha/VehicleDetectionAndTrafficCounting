# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.4 | yolov8n_bytetrack_conf0.4 | yolov8s_botsort_conf0.4 | yolov8s_bytetrack_conf0.4 |
|---|---|---|---|---|---|
| car | 245 | 7.40 / 15.10% | 7.60 / 15.51% | 5.20 / 10.61% | 5.40 / 11.02% |
| bus | 5 | 0.20 / 20.00% | 0.40 / 40.00% | 0.40 / 40.00% | 0.60 / 60.00% |
| truck | 1 | 0.40 / 200.00% | 0.40 / 200.00% | 0.60 / 300.00% | 0.60 / 300.00% |
| all | 251 | 7.20 / 14.34% | 7.20 / 14.34% | 5.00 / 9.96% | 5.00 / 9.96% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | -1 | +0 | -4 | -2 | -30 |
| yolov8n_bytetrack_conf0.4 | -1 | +0 | -4 | -2 | -31 |
| yolov8s_botsort_conf0.4 | -1 | +0 | -1 | +1 | -23 |
| yolov8s_bytetrack_conf0.4 | -1 | +1 | -1 | +1 | -23 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | +0 | +0 | +0 | +0 | +1 |
| yolov8n_bytetrack_conf0.4 | +0 | +0 | +0 | +0 | +2 |
| yolov8s_botsort_conf0.4 | +0 | +0 | +0 | +0 | +2 |
| yolov8s_bytetrack_conf0.4 | +0 | +0 | +0 | +0 | +3 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | +1 | +0 | +0 | +0 | -1 |
| yolov8n_bytetrack_conf0.4 | +1 | +0 | +0 | +0 | -1 |
| yolov8s_botsort_conf0.4 | +1 | +1 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.4 | +1 | +1 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.4 | +0 | +0 | -4 | -2 | -30 |
| yolov8n_bytetrack_conf0.4 | +0 | +0 | -4 | -2 | -30 |
| yolov8s_botsort_conf0.4 | +0 | +1 | -1 | +1 | -22 |
| yolov8s_bytetrack_conf0.4 | +0 | +2 | -1 | +1 | -21 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.4: -7.20 / 0% / 60%, yolov8n_bytetrack_conf0.4: -7.20 / 0% / 60%, yolov8s_botsort_conf0.4: -4.20 / 40% / 40%, yolov8s_bytetrack_conf0.4: -3.80 / 40% / 40%

