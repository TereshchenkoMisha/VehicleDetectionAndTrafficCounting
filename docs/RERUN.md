# Запуск pipeline: refactor с контролем эксперимента и детерминизмом

Текущая версия использует подачу детекций из refactor: детектор сохраняет боксы от min(track_low_thresh, track_high_thresh), обычно 0.1. --conf одновременно задаёт track_high_thresh и new_track_thresh (по умолчанию 0.25). Сохранены проверки и снимки входов ablation, детерминизм по умолчанию и отключение TF32. Для трекинга зафиксирован NMS по классам (средняя dev MAE 5.40 против 5.55 у общего NMS); результаты и обоснование — в NMS_SELECTION.md. Структура src/ + scripts/ сохранена, новых Python-модулей нет.

## Сохранённые результаты

Основной запуск run_20261102_164923 и sweep run_20261102_164923_metadata_fixed сделаны до формирования текущей версии. Они корректно рассчитаны для своих записанных настроек, но не являются новым запуском текущей версии. Их файлы и manifest не изменены. Подробности аудита — в RESULTS_REVIEW.md.

Исходный ablation_run_20261102_164923 остался прерванным. Итоговые результаты находятся в каталоге с суффиксом metadata_fixed.

## Окружение и данные

Из корня VehicleDetectionAndTrafficCounting, в PowerShell:

~~~powershell
# Создание .venv нужно только при его отсутствии
& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,notebook]'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
~~~

На этом компьютере проверена пара torch 2.14.1+cu126 / torchvision 0.29.1+cu126 и RTX 4060. Если в окружение попала CPU-сборка:

~~~powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
~~~

Для --device 0 torch.cuda.is_available() должен вернуть True. --device cpu работает без GPU, но будет медленнее.

Исходный UA-DETRAC уже находится в data/raw/UA-DETRAC/. Для новой машины нужны каталоги DETRAC-Images/DETRAC-Images/MVI_*/img*.jpg и обе папки DETRAC-*-Annotations-XML. Фиксированные reports/splits.json, ignored_regions.json и counting_lines.json нужно сохранить. Подготовленные данные находятся в data/processed/detrac_yolo/. Веса — в models/, при отсутствии скачиваются автоматически.

## Полный новый запуск

~~~powershell
$runTag = 'run_hybrid_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe scripts/run_all.py --tag $runTag --splits dev test --device 0 --to ablation
~~~

Этапы 1–10: тесты, подготовка YOLO-данных, оценка двух детекторов, четыре конфигурации трекинга, TrackEval, GT-подсчёт, подсчёт по трекам, counting-метрики, таблицы/графики и проверка. Этап 11 — 16 условий confidence sweep на dev и его анализ. Без --to ablation выполняются только этапы 1–10.

Существующие подготовленные YOLO-данные переиспользуются. Трекинг использует исходные кадры с полной частотой, детекторная оценка — каждый пятый кадр. Новые tag обязательны при смене протокола: старые результаты нельзя дополнять условиями с другими правилами.

Только новый ablation:

~~~powershell
$ablationTag = 'run_hybrid_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe scripts/run_all.py --tag $ablationTag --from ablation --splits dev --device 0
~~~

## Где смотреть

| Путь | Содержимое |
|---|---|
| reports/<tag>/ | Метрики детекции, трекинга и подсчёта, JSON/Markdown |
| outputs/tracks_<tag>/<split>/<config>/ | MOT-файлы и timing.json |
| outputs/reports_<tag>/<split>/tables/ и charts/ | Таблицы и PNG графики |
| outputs/reports_<tag>/notebooks/ | Выполненные results_dev/test.ipynb |
| outputs/ablation_<tag>/ | Manifest, shared/data, GT, треки и отчёты каждого confidence |
| outputs/ablation_<tag>/analysis/ | Таблицы, графики, выполненный notebook и analysis_manifest.json |

Успех основного pipeline: нулевые Exit, 0 check(s) failed и FINISHED stages. Успех sweep: complete=true, status=complete в manifest и выполненный анализ. Без nbconvert runner может пропустить построение графиков, поэтому наличие каталога charts нужно проверить отдельно.

## Проверка и восстановленные опции

~~~powershell
# Проверить существующий основной запуск по записанным настройкам
.\.venv\Scripts\python.exe scripts/verify_run.py --tag run_20261102_164923 --splits dev test

# Проверить новый запуск именно на соответствие текущему протоколу
.\.venv\Scripts\python.exe scripts/verify_run.py --tag $runTag --splits dev test --protocol current

# Проверить завершённый sweep (поддерживаются оба старых формата manifest)
.\.venv\Scripts\python.exe scripts/verify_run.py --ablation-dir outputs/ablation_run_20261102_164923_metadata_fixed

# Проверить подготовленные YOLO-боксы без пересборки датасета
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --preview-only --preview 6

# Отдельно экспортировать GT MOT
.\.venv\Scripts\python.exe scripts/evaluate_tracking.py --prepare-gt-only --splits dev test

# Дополнительно получить tracking-метрики по car/bus/truck
.\.venv\Scripts\python.exe scripts/evaluate_tracking.py --splits dev --classes-aware --tracks-dir "outputs/tracks_$runTag" --out-prefix "reports/$runTag/tracking_per_class"
~~~

Sanity-картинки сохраняются в reports/sanity/. Метрики по классам находятся в JSON per_seq; all и агрегаты остаются совместимыми с основным анализом.

## Продолжение

Повторить ablation с тем же tag: runner передаст --resume. Снимки входов, XML, инвентарь кадров, веса и готовые выходы проверяются; план должен совпадать с текущими кодом, настройками и окружением. При смене кода или протокола нужен новый tag. Проверять и анализировать старые завершённые результаты можно без миграции manifest и без пересчёта.

Файл .running защищает sweep от двух процессов. После системного сбоя удалять оставшийся lock можно только после проверки, что записанный PID больше не работает.

Если основной трекинг закончен, продолжить с --from trackeval и тем же tag. Для пересчёта диапазона этапов использовать --from/--to и --clean: runner сначала покажет удаляемые пути. --clean действует на продукты выбранных этапов для tag целиком; при включённом prepare удаляется общий подготовленный датасет. Для обычного нового прогона --clean не нужен.

Скорость сравнивать в одном протоколе и одинаковых условиях нагрузки, лучше по нескольким повторам. Сохранённые замеры показали заметный разброс FPS при одинаковых треках.
