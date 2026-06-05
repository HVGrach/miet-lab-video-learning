# Source Track Linkage Acceptance

Дата фиксации: 2026-06-02.

## Что Закрыто

- Построен file metadata manifest с MD5 для всех 24 170 кадров.
- Найдены exact duplicate frame clusters.
- Проверен текущий `train_val_split.csv` на exact hash leakage.
- Построен hash-guarded split без exact MD5 leakage и без filename-time virtual group leakage.

## Артефакты

- `notebooks/07_source_track_linkage_audit.ipynb`
- `outputs/manifests/file_metadata_manifest.csv`
- `outputs/manifests/duplicate_frame_audit.csv`
- `outputs/manifests/split_frame_hashes.csv`
- `outputs/manifests/sample_overlap_audit.csv`
- `outputs/manifests/hash_source_linkage_candidates.csv`
- `outputs/manifests/train_val_split_hash_guarded.csv`
- `outputs/reports/duplicate_frame_cluster_summary.csv`
- `outputs/reports/hash_split_leakage_summary.csv`
- `outputs/reports/hash_guarded_split_summary.csv`
- `outputs/reports/hash_guarded_split_leakage_summary.csv`
- `outputs/reports/hash_virtual_guarded_split_leakage_summary.csv`
- `outputs/reports/source_group_split_leakage_summary.csv`

## Актуальные Числа

- `file_metadata_manifest.csv`: 24 170 кадров, абсолютной колонки `path` нет.
- Duplicate hash clusters: 369.
- Cross-label duplicate clusters: 51.
- В исходном `train_val_split.csv`: 22 hash попадают и в train, и в val.
- В исходном `train_val_split.csv`: 28 sample-pair overlaps пересекают train/val.
- `hash_source_linkage_candidates.csv`: 126 строк, 63 exact hash groups.
- `train_val_split_hash_guarded.csv`: 1 546 sample, 272 `hash_guard_group_id`.
- В `train_val_split_hash_guarded.csv`: 0 hash leakage и 0 filename-time virtual group leakage.

Hash-guarded split:

| Класс | Split | Sample |
|---|---|---:|
| `inaction` | train | 455 |
| `inaction` | val | 117 |
| `move` | train | 323 |
| `move` | val | 105 |
| `work` | train | 444 |
| `work` | val | 102 |

## Правила Для Baseline

- Для baseline/smoke-метрик использовать `train_val_split_hash_guarded.csv`, где учтены exact hash overlaps и filename-time virtual groups. Старый `train_val_split.csv` оставлен только как исторический артефакт и для сравнения.
- MD5, `mtime`, filenames, paths and source ids are split/audit metadata only, not model features.
- Cross-label duplicate clusters are audit debt: they can indicate duplicated/cropped identical frames across action labels and should be checked before interpreting errors.

## Проверка

- Notebook `notebooks/07_source_track_linkage_audit.ipynb` выполнен без ошибок.
- Актуальная общая команда `PYTHONPATH=src python3 -m unittest discover -s tests -v` проходит 75 тестов.
