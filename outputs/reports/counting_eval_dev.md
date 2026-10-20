# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.25 | yolov8n_bytetrack_conf0.25 | yolov8s_botsort_conf0.25 | yolov8s_bytetrack_conf0.25 |
|---|---|---|---|---|---|
| car | 245 | 6.20 / 12.65% | 6.20 / 12.65% | 4.60 / 9.39% | 4.40 / 8.98% |
| bus | 5 | 0.60 / 60.00% | 0.60 / 60.00% | 0.80 / 80.00% | 0.80 / 80.00% |
| truck | 1 | 0.80 / 400.00% | 1.00 / 500.00% | 0.60 / 300.00% | 0.60 / 300.00% |
| all | 251 | 5.60 / 11.16% | 5.40 / 10.76% | 4.00 / 7.97% | 3.80 / 7.57% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | -1 | +0 | -2 | -1 | -27 |
| yolov8n_bytetrack_conf0.25 | -1 | +0 | -2 | -2 | -26 |
| yolov8s_botsort_conf0.25 | -1 | +1 | -1 | +1 | -19 |
| yolov8s_bytetrack_conf0.25 | -1 | +2 | -1 | +1 | -17 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | +1 | +0 | +2 |
| yolov8n_bytetrack_conf0.25 | +0 | +0 | +1 | +0 | +2 |
| yolov8s_botsort_conf0.25 | +0 | +0 | +0 | +0 | +4 |
| yolov8s_bytetrack_conf0.25 | +0 | +0 | +0 | +0 | +4 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +1 | +0 | +0 | +2 | -1 |
| yolov8n_bytetrack_conf0.25 | +2 | +0 | +0 | +2 | -1 |
| yolov8s_botsort_conf0.25 | +1 | +1 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.25 | +1 | +1 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | -1 | +1 | -26 |
| yolov8n_bytetrack_conf0.25 | +1 | +0 | -1 | +0 | -25 |
| yolov8s_botsort_conf0.25 | +0 | +2 | -1 | +1 | -16 |
| yolov8s_bytetrack_conf0.25 | +0 | +3 | -1 | +1 | -14 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.25: -5.20 / 20% / 40%, yolov8n_bytetrack_conf0.25: -5.00 / 20% / 40%, yolov8s_botsort_conf0.25: -2.80 / 40% / 40%, yolov8s_bytetrack_conf0.25: -2.20 / 40% / 40%

