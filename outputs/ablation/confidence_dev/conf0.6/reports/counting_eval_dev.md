# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.6 | yolov8n_bytetrack_conf0.6 | yolov8s_botsort_conf0.6 | yolov8s_bytetrack_conf0.6 |
|---|---|---|---|---|---|
| car | 245 | 17.20 / 35.10% | 17.20 / 35.10% | 8.00 / 16.33% | 8.00 / 16.33% |
| bus | 5 | 0.00 / 0.00% | 0.00 / 0.00% | 0.40 / 40.00% | 0.40 / 40.00% |
| truck | 1 | 0.40 / 200.00% | 0.40 / 200.00% | 0.40 / 200.00% | 0.40 / 200.00% |
| all | 251 | 17.20 / 34.26% | 17.20 / 34.26% | 7.60 / 15.14% | 7.60 / 15.14% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | -3 | -3 | -13 | -8 | -59 |
| yolov8n_bytetrack_conf0.6 | -3 | -3 | -13 | -8 | -59 |
| yolov8s_botsort_conf0.6 | -1 | -4 | -2 | -2 | -31 |
| yolov8s_bytetrack_conf0.6 | -1 | -4 | -2 | -2 | -31 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | +0 | +0 | +0 | +0 | +0 |
| yolov8n_bytetrack_conf0.6 | +0 | +0 | +0 | +0 | +0 |
| yolov8s_botsort_conf0.6 | +0 | +0 | +0 | +0 | +2 |
| yolov8s_bytetrack_conf0.6 | +0 | +0 | +0 | +0 | +2 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | +1 | +0 | +0 | +0 | -1 |
| yolov8n_bytetrack_conf0.6 | +1 | +0 | +0 | +0 | -1 |
| yolov8s_botsort_conf0.6 | +1 | +0 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.6 | +1 | +0 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | -2 | -3 | -13 | -8 | -60 |
| yolov8n_bytetrack_conf0.6 | -2 | -3 | -13 | -8 | -60 |
| yolov8s_botsort_conf0.6 | +0 | -4 | -2 | -2 | -30 |
| yolov8s_bytetrack_conf0.6 | +0 | -4 | -2 | -2 | -30 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.6: -17.20 / 0% / 100%, yolov8n_bytetrack_conf0.6: -17.20 / 0% / 100%, yolov8s_botsort_conf0.6: -7.60 / 0% / 80%, yolov8s_bytetrack_conf0.6: -7.60 / 0% / 80%

