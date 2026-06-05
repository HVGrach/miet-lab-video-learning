# Sampling And Preprocessing Acceptance

Дата фиксации: 2026-06-02.

## Что закрыто

- Пункт 1: `inaction` может брать 8 кадров с произвольными пропусками. В default manifest порядок сохраняется, shuffle вынесен в отдельный ablation manifest.
- Пункт 2: `move`/`work` поддерживают dilation/lowering FPS.
- Пункт 3: `move`/`work` поддерживают stride windows поверх длинных no-prefixed треков.
- Пункт 5: sequence-level preprocessing применяет одинаковые color/JPEG/downscale параметры ко всем 8 кадрам sample.

## Default Sampling

- Всегда включаются 364 явных 8-кадровых sample из `sample_manifest.csv`.
- `inaction`: 24 ordered random-skip sample на каждую длинную no-prefixed папку.
- `move/work`: стратегии `(dilation, stride) = (1,8), (2,8), (4,16), (8,32)`.
- Cap на папку и стратегию: `move=2`, `work=1`.
- Папки с `N_im1.jpg` ... `N_im8.jpg` не sliding-ятся повторно, чтобы не склеивать независимые объекты.
- API-дефолт `build_training_sample_candidates(frame_manifest)` совпадает с этим проектным default: `inaction_shuffle=False`, `inaction_samples_per_folder=24`, стратегии `(1,8),(2,8),(4,16),(8,32)`, caps `move=2`, `work=1`.

## Актуальные Числа

| Класс | Default sample |
|---|---:|
| `inaction` | 572 |
| `move` | 428 |
| `work` | 546 |
| Итого | 1 546 |

Первичный grouped split `train_val_split.csv` был создан по `source_track_id`/`folder_rel` и полезен как исторический артефакт пункта 1/2/3/5, но после hash-аудита он superseded для baseline-метрик.

| Класс | Split | Sample |
|---|---|---:|
| `inaction` | train | 449 |
| `inaction` | val | 123 |
| `move` | train | 337 |
| `move` | val | 91 |
| `work` | train | 422 |
| `work` | val | 124 |

Leakage check: 0 `source_track_id` одновременно в train и val.

Финальный baseline/smoke использует более строгий `train_val_split_hash_guarded.csv`:

| Класс | Split | Sample |
|---|---|---:|
| `inaction` | train | 455 |
| `inaction` | val | 117 |
| `move` | train | 323 |
| `move` | val | 105 |
| `work` | train | 444 |
| `work` | val | 102 |

Hash-guarded leakage check: 0 exact hash leakage и 0 filename-time virtual group leakage.

## Артефакты

- `outputs/manifests/training_sample_candidates.csv`
- `outputs/manifests/train_val_split.csv`
- `outputs/manifests/inaction_shuffle_ablation_candidates.csv`
- `outputs/reports/training_candidate_summary.csv`
- `outputs/reports/train_val_split_summary.csv`
- `outputs/eda/preprocessing_sequence_contact_sheet.png`
- `notebooks/05_sampling_preprocessing.ipynb`

## Правила Для Обучения

- Для baseline-метрик использовать `outputs/manifests/train_val_split_hash_guarded.csv`; `train_val_split.csv` оставлен только как исторический grouped split для сравнения/smoke.
- Не делать random split по строкам или кадрам.
- Не подавать в модель `sample_id`, `folder_rel`, `source_track_id`, `file_name`, original size или path-like признаки.
- Для inference preprocessing должен совпадать с train/val: fixed output shape, letterbox resize, ровно 8 изображений.
- Shuffle manifest для `inaction` считать ablation, не default.

## Проверка

- Notebook `notebooks/05_sampling_preprocessing.ipynb` выполнен без ошибок.
- Contract assertions внутри notebook прошли.
- Актуальная общая команда `PYTHONPATH=src python3 -m unittest discover -s tests -v` проходит 75 тестов.
