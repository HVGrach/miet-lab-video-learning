# Lab6: классификация действий сотрудников

Готовое Docker-решение для классификации папки с ровно 8 изображениями в один из трех классов:

- `inaction`
- `move`
- `work`

Основной сценарий: восстановить веса, собрать Docker-образ, передать папку с 8 изображениями и получить один класс в stdout.

## Быстрая проверка

```bash
git clone https://github.com/HVGrach/miet-lab-video-learning.git
cd miet-lab-video-learning

scripts/download_google_drive_weights.sh
docker build -t timesformer-infer .

docker run --rm \
  -v /absolute/path/to/8_images:/app/input:ro \
  timesformer-infer
```

При успешном запуске выводится ровно одна строка:

```text
inaction
```

или:

```text
move
```

или:

```text
work
```

Docker-образ по умолчанию запускает:

```bash
python -m src.predict_frames --input /app/input
```

Входная папка должна содержать ровно 8 файлов изображений. Неизображения отклоняются; скрытый мусор вроде `.DS_Store` игнорируется.

## Веса на Google Drive

Крупные веса не хранятся в Git. Их восстанавливает скрипт `scripts/download_google_drive_weights.sh`.

- Архив весов: [lab6_weights_bundle.tar.gz](https://drive.google.com/open?id=1q4rjJALqwzE-Hb3QRtvxand6QDoSSZJh)
- Файл контрольной суммы: [lab6_weights_bundle.tar.gz.sha256](https://drive.google.com/open?id=1yCFZfxLh2ap_nS-yrnwUlb6TjSxRrbIE)
- Ожидаемый SHA256: `967d087056729d2828d6aa48e65a17d1ec3f38cc62d694ad3f0d4a23537199d3`
- Размер архива на Google Drive: `907906865` байт

Архив восстанавливает:

```text
weights/
  timesformer_checkpoint.pt
  timesformer_hf/
  bagging_best.joblib
  labels.json

outputs/models/lab6_action_classifier.pt
```

`Dockerfile` проверяет наличие этих файлов во время `docker build`, поэтому клон без восстановленных весов не соберет сломанный образ.

Запасной вариант через GitHub Release тоже оставлен:

```bash
scripts/download_release_weights.sh
```

Для него нужен GitHub CLI (`gh`). Скрипт скачивает тег релиза `lab6-timesformer-artifacts`.

## Метрики

Финальный запуск TimeSFormer:

```text
outputs/timesformer_runs/timesformer_head_b4_e3_20260605/
```

Валидационная разбивка:

```text
outputs/manifests/train_val_split_hash_guarded.csv
```

Сводка:

| Метрика | Значение |
|---|---:|
| Accuracy | `0.5741` |
| Macro F1 | `0.5678` |
| F1 `inaction` | `0.4432` |
| F1 `move` | `0.6730` |
| F1 `work` | `0.5873` |

Матрица ошибок: строки - истинный класс, столбцы - предсказанный класс.

| истинный \ предсказанный | inaction | move | work |
|---|---:|---:|---:|
| inaction | 41 | 29 | 47 |
| move | 5 | 71 | 29 |
| work | 22 | 6 | 74 |

Подробные файлы:

- `outputs/timesformer_runs/timesformer_head_b4_e3_20260605/metrics.json`
- `outputs/timesformer_runs/timesformer_head_b4_e3_20260605/per_class_metrics.csv`
- `outputs/timesformer_runs/timesformer_head_b4_e3_20260605/confusion_matrix.csv`
- `outputs/reports/timesformer_classical_deployment_status.md`

Важно: оригинальный checkpoint из присланного notebook не был передан вместе с файлами, поэтому `timesformer_checkpoint.pt` обучен локально на hash-guarded split. В этом запуске обучалась только классификационная голова, а базовая модель HuggingFace TimeSFormer хранится локально для Docker-инференса без доступа к сети.

## Инференс по кадрам

Локальный Python:

```bash
python -m src.predict_frames --input /path/to/8_images
```

Docker:

```bash
docker run --rm \
  -v /absolute/path/to/8_images:/app/input:ro \
  timesformer-infer \
  --input /app/input
```

Docker Compose:

```bash
LAB6_INPUT_DIR=/absolute/path/to/8_images docker compose run --rm infer
```

## Инференс по эмбеддингам

Ветка с предоставленным `bagging_best.joblib` сохранена для `.npy`-файлов эмбеддингов с 768 признаками `emb`.

Локальный Python:

```bash
python -m src.predict_embeddings --input /path/to/object_embedding.npy
```

Docker:

```bash
docker run --rm \
  --entrypoint python \
  -v /absolute/path/to/object_embedding.npy:/app/input/object_embedding.npy:ro \
  timesformer-infer \
  -m src.predict_embeddings --input /app/input/object_embedding.npy
```

## Проверено локально

Проверенные команды:

```text
scripts/download_google_drive_weights.sh /tmp/lab6_drive_script_download.tar.gz
restored weights from /tmp/lab6_drive_script_download.tar.gz
```

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
81 tests OK
```

```text
docker build -t timesformer-infer .
OK
```

```text
docker run --rm --network none -v ...:/app/input:ro timesformer-infer
stdout: inaction
real: 7.69 sec
```

Проверка на свежем клоне из GitHub:

```text
git clone https://github.com/HVGrach/miet-lab-video-learning.git
scripts/download_google_drive_weights.sh /tmp/lab6_drive_script_download.tar.gz
docker build -t timesformer-infer-fresh .
docker run --rm --network none -v ...:/app/input:ro timesformer-infer-fresh
stdout: inaction
real: 5.34 sec
```

Docker-инференс по кадрам проверен с `--network none`: образ содержит нужные веса и не скачивает модели во время запуска.

## Старая точка входа

Предыдущая точка входа лабораторной оставлена в образе для совместимости:

```bash
docker run --rm \
  --entrypoint python \
  -v /absolute/path/to/8_images:/app/input:ro \
  timesformer-infer \
  infer.py /app/input
```
