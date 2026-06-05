# Role And Track Audit Acceptance

Дата фиксации: 2026-06-02.

## Что Закрыто

- Пункт 6 закрыт как безопасный audit foundation: создана заготовка для ручной role-разметки и ranking `move` папок, похожих на сотрудников в форме.
- Пункт 7 закрыт как безопасный audit foundation: построены кандидаты virtual source groups по filename time tokens.
- Роль `customer/employee` не используется как default-признак модели.
- Weak labels вида `move -> customer`, `work -> employee` запрещены для supervised role-classifier.
- Filesystem `mtime` не используется для автоматического объединения папок.

## Артефакты

- `notebooks/06_role_track_audit.ipynb`
- `notebooks/07_source_track_linkage_audit.ipynb`
- `outputs/manifests/manual_role_seed_manifest.csv`
- `outputs/manifests/train_val_split_with_virtual_track_groups.csv`
- `outputs/reports/role_green_proxy_frames.csv`
- `outputs/reports/role_green_proxy_folder_summary.csv`
- `outputs/reports/move_role_review_candidates.csv`
- `outputs/reports/folder_time_token_summary.csv`
- `outputs/reports/virtual_track_link_candidates.csv`
- `outputs/reports/virtual_track_group_split_summary.csv`
- `outputs/reports/source_track_linkage_acceptance.md`

## Актуальные Числа

- `manual_role_seed_manifest.csv`: 364 sample, все `role_label=unknown`, все `role_label_source=manual_review_needed`.
- `move_role_review_candidates.csv`: 106 папок `move`, отсортированных для ручной проверки.
- `virtual_track_link_candidates.csv`: 4 строки, 2 virtual groups:
  - `move::filename_time::00-28-44`
  - `work::filename_time::00-23-56`
- Все virtual links имеют `link_reason=same_label_filename_time`.
- Все virtual links имеют `uses_mtime_for_link=False`.
- Hash audit: 369 duplicate hash clusters, 51 cross-label duplicate clusters.
- Старый `train_val_split.csv`: 22 hash попадают и в train, и в val.
- Новый `train_val_split_hash_guarded.csv`: 0 hash leakage и 0 filename-time virtual group leakage.

## Важное Предупреждение

Текущий frozen split по `folder_rel` имеет 0 leakage по `source_track_id`, но virtual-group audit нашёл 1 потенциальную stricter leakage group:

- `move::filename_time::00-28-44`

Это не означает автоматическую ошибку разметки. Это означает, что перед финальным production baseline стоит вручную подтвердить или отклонить virtual group и при подтверждении пересобрать split по `source_track_group_id`.

Отдельно: hash/source audit уже построил более строгий `train_val_split_hash_guarded.csv`. Для baseline-метрик его стоит предпочесть старому split.

## Правила Для Baseline

- В модель можно подавать только изображения или pixel-derived embeddings.
- Нельзя подавать `sample_id`, `folder_rel`, `source_track_id`, `source_track_group_id`, `file_name`, `rel_path`, `path`, `mtime`, original size, aspect ratio или role labels как features.
- Role-classifier можно обучать только при наличии отдельной ручной role-разметки с `role_label_source=manual`.
- CatBoost/любой табличный классификатор должен валидироваться на frozen grouped split и не получать служебные категориальные признаки.

## Проверка

- Notebook `notebooks/06_role_track_audit.ipynb` выполнен без ошибок.
- Актуальная общая команда `PYTHONPATH=src python3 -m unittest discover -s tests -v` проходит 75 тестов.
