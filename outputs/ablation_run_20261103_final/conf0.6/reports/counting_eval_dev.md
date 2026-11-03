# Counting evaluation - dev split

Line-crossing counts vs GT (`exclude_ignored`): MAE / WAPE per class, sequences in splits.json order (5).

| class | GT total | yolov8n_botsort_conf0.6 | yolov8n_bytetrack_conf0.6 | yolov8s_botsort_conf0.6 | yolov8s_bytetrack_conf0.6 |
|---|---|---|---|---|---|
| car | 245 | 9.40 / 19.18% | 10.00 / 20.41% | 5.60 / 11.43% | 5.60 / 11.43% |
| bus | 5 | 0.00 / 0.00% | 0.00 / 0.00% | 0.60 / 60.00% | 0.60 / 60.00% |
| truck | 1 | 0.40 / 200.00% | 0.40 / 200.00% | 0.60 / 300.00% | 0.60 / 300.00% |
| all | 251 | 9.40 / 18.73% | 10.00 / 19.92% | 4.80 / 9.56% | 4.80 / 9.56% |

## Signed per-sequence error (SCE, pred - gt)

**car**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | -2 | +0 | -8 | -4 | -33 |
| yolov8n_bytetrack_conf0.6 | -2 | +0 | -9 | -4 | -35 |
| yolov8s_botsort_conf0.6 | -1 | -1 | -1 | -1 | -24 |
| yolov8s_bytetrack_conf0.6 | -1 | -1 | -1 | -1 | -24 |

**bus**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | +0 | +0 | +0 | +0 | +0 |
| yolov8n_bytetrack_conf0.6 | +0 | +0 | +0 | +0 | +0 |
| yolov8s_botsort_conf0.6 | +0 | +0 | +0 | +0 | +3 |
| yolov8s_bytetrack_conf0.6 | +0 | +0 | +0 | +0 | +3 |

**truck**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | +1 | +0 | +0 | +0 | -1 |
| yolov8n_bytetrack_conf0.6 | +1 | +0 | +0 | +0 | -1 |
| yolov8s_botsort_conf0.6 | +1 | +1 | +0 | +0 | -1 |
| yolov8s_bytetrack_conf0.6 | +1 | +1 | +0 | +0 | -1 |

**all**

| config | MVI_20033 | MVI_40141 | MVI_40171 | MVI_40773 | MVI_63553 |
|---|---|---|---|---|---|
| yolov8n_botsort_conf0.6 | -1 | +0 | -8 | -4 | -34 |
| yolov8n_bytetrack_conf0.6 | -1 | +0 | -9 | -4 | -36 |
| yolov8s_botsort_conf0.6 | +0 | +0 | -1 | -1 | -22 |
| yolov8s_bytetrack_conf0.6 | +0 | +0 | -1 | -1 | -22 |

mean SCE / over-rate / under-rate (all classes): yolov8n_botsort_conf0.6: -9.40 / 0% / 80%, yolov8n_bytetrack_conf0.6: -10.00 / 0% / 80%, yolov8s_botsort_conf0.6: -4.80 / 0% / 60%, yolov8s_bytetrack_conf0.6: -4.80 / 0% / 60%

