# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.25 | yolov8n_bytetrack_conf0.25 | yolov8s_botsort_conf0.25 | yolov8s_bytetrack_conf0.25 |
|---|---|---|---|---|---|
| car | 245 | 8.80 / 17.96% | 9.00 / 18.37% | 4.80 / 9.80% | 4.60 / 9.39% |
| bus | 5 | 0.60 / 60.00% | 0.60 / 60.00% | 0.80 / 80.00% | 0.60 / 60.00% |
| truck | 1 | 0.60 / 300.00% | 0.80 / 400.00% | 0.60 / 300.00% | 0.60 / 300.00% |
| all | 251 | 8.00 / 15.94% | 8.00 / 15.94% | 4.20 / 8.37% | 4.20 / 8.37% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | -1 | +0 | -5 | -2 | -36 |
| yolov8n_bytetrack_conf0.25 | -2 | +0 | -5 | -2 | -36 |
| yolov8s_botsort_conf0.25 | -1 | +1 | -1 | +0 | -21 |
| yolov8s_bytetrack_conf0.25 | -1 | +1 | -1 | +0 | -20 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | +1 | +0 | +2 |
| yolov8n_bytetrack_conf0.25 | +0 | +0 | +1 | +0 | +2 |
| yolov8s_botsort_conf0.25 | +0 | +0 | +0 | +0 | +4 |
| yolov8s_bytetrack_conf0.25 | +0 | +0 | +0 | +0 | +3 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +1 | +0 | +0 | +1 | -1 |
| yolov8n_bytetrack_conf0.25 | +2 | +0 | +0 | +1 | -1 |
| yolov8s_botsort_conf0.25 | +1 | +1 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.25 | +1 | +1 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.25 | +0 | +0 | -4 | -1 | -35 |
| yolov8n_bytetrack_conf0.25 | +0 | +0 | -4 | -1 | -35 |
| yolov8s_botsort_conf0.25 | +0 | +2 | -1 | +0 | -18 |
| yolov8s_bytetrack_conf0.25 | +0 | +2 | -1 | +0 | -18 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.25: -8.00 / 0% / 60%, yolov8n_bytetrack_conf0.25: -8.00 / 0% / 60%, yolov8s_botsort_conf0.25: -3.40 / 20% / 40%, yolov8s_bytetrack_conf0.25: -3.40 / 20% / 40%

