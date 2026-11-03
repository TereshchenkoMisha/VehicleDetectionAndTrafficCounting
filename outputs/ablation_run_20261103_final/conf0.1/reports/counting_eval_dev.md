# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.1 | yolov8n_bytetrack_conf0.1 | yolov8s_botsort_conf0.1 | yolov8s_bytetrack_conf0.1 |
|---|---|---|---|---|---|
| car | 245 | 8.40 / 17.14% | 8.60 / 17.55% | 5.00 / 10.20% | 4.80 / 9.80% |
| bus | 5 | 0.60 / 60.00% | 0.60 / 60.00% | 0.80 / 80.00% | 0.60 / 60.00% |
| truck | 1 | 0.80 / 400.00% | 1.00 / 500.00% | 0.60 / 300.00% | 0.60 / 300.00% |
| all | 251 | 7.80 / 15.54% | 7.80 / 15.54% | 4.40 / 8.76% | 4.40 / 8.76% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | -1 | +0 | -5 | -1 | -35 |
| yolov8n_bytetrack_conf0.1 | -2 | +0 | -6 | -1 | -34 |
| yolov8s_botsort_conf0.1 | -1 | +1 | -1 | +1 | -21 |
| yolov8s_bytetrack_conf0.1 | -1 | +1 | -1 | +1 | -20 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | +0 | +0 | +1 | +0 | +2 |
| yolov8n_bytetrack_conf0.1 | +0 | +0 | +1 | +0 | +2 |
| yolov8s_botsort_conf0.1 | +0 | +0 | +0 | +0 | +4 |
| yolov8s_bytetrack_conf0.1 | +0 | +0 | +0 | +0 | +3 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | +1 | +0 | +0 | +2 | -1 |
| yolov8n_bytetrack_conf0.1 | +2 | +0 | +0 | +2 | -1 |
| yolov8s_botsort_conf0.1 | +1 | +1 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.1 | +1 | +1 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.1 | +0 | +0 | -4 | +1 | -34 |
| yolov8n_bytetrack_conf0.1 | +0 | +0 | -5 | +1 | -33 |
| yolov8s_botsort_conf0.1 | +0 | +2 | -1 | +1 | -18 |
| yolov8s_bytetrack_conf0.1 | +0 | +2 | -1 | +1 | -18 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.1: -7.40 / 20% / 40%, yolov8n_bytetrack_conf0.1: -7.40 / 20% / 40%, yolov8s_botsort_conf0.1: -3.20 / 40% / 40%, yolov8s_bytetrack_conf0.1: -3.20 / 40% / 40%

