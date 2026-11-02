# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.25 | yolov8n_bytetrack_conf0.25 | yolov8s_botsort_conf0.25 | yolov8s_bytetrack_conf0.25 |
|---|---|---|---|---|---|
| car | 245 | 7.60 / 15.51% | 7.60 / 15.51% | 5.00 / 10.20% | 5.00 / 10.20% |
| bus | 5 | 0.40 / 40.00% | 0.60 / 60.00% | 0.60 / 60.00% | 0.60 / 60.00% |
| truck | 1 | 0.60 / 300.00% | 0.60 / 300.00% | 0.60 / 300.00% | 0.60 / 300.00% |
| all | 251 | 7.00 / 13.94% | 6.80 / 13.55% | 4.20 / 8.37% | 4.20 / 8.37% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | -1 | +0 | -3 | -2 | -32 |
| yolov8n_bytetrack_conf0.25 | -1 | +0 | -3 | -2 | -32 |
| yolov8s_botsort_conf0.25 | -1 | -1 | -1 | +1 | -21 |
| yolov8s_bytetrack_conf0.25 | -1 | -1 | -1 | +1 | -21 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | +1 | +0 | +1 |
| yolov8n_bytetrack_conf0.25 | +0 | +0 | +1 | +0 | +2 |
| yolov8s_botsort_conf0.25 | +0 | +0 | +0 | +0 | +3 |
| yolov8s_bytetrack_conf0.25 | +0 | +0 | +0 | +0 | +3 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +1 | +0 | +0 | +1 | -1 |
| yolov8n_bytetrack_conf0.25 | +1 | +0 | +0 | +1 | -1 |
| yolov8s_botsort_conf0.25 | +1 | +1 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.25 | +1 | +1 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | -2 | -1 | -32 |
| yolov8n_bytetrack_conf0.25 | +0 | +0 | -2 | -1 | -31 |
| yolov8s_botsort_conf0.25 | +0 | +0 | -1 | +1 | -19 |
| yolov8s_bytetrack_conf0.25 | +0 | +0 | -1 | +1 | -19 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.25: -7.00 / 0% / 60%, yolov8n_bytetrack_conf0.25: -6.80 / 0% / 60%, yolov8s_botsort_conf0.25: -3.80 / 20% / 40%, yolov8s_bytetrack_conf0.25: -3.80 / 20% / 40%

