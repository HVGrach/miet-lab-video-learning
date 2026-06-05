# Inference And Docker Acceptance

Дата фиксации: 2026-06-02.

## Что закрыто

- CLI: `python infer.py /path/to/folder_with_8_images`.
- Successful stdout: ровно один класс из `inaction`, `move`, `work`.
- Default checkpoint: `outputs/models/lab6_action_classifier.pt`.
- Checkpoint source run: `outputs/baseline_runs/mobilenet_v3_small_160_frozen_e30_balanced_minclass/model_best.pt`.
- Checkpoint architecture: `mobilenet_v3_small` frame encoder + temporal statistics head.
- Checkpoint preprocessing: `image_size=[160, 160]`, letterbox resize.
- Docker image: `lab6-video-infer:latest`.
- Dockerfile copies only `src`, `infer.py`, `requirements-inference.txt`, and `outputs/models/lab6_action_classifier.pt`.
- `.dockerignore` excludes `lab6/`, notebooks, tests, old reports, and backup datasets from the build context.

## Проверенные команды

Local CPU smoke:

```bash
python3 infer.py /tmp/lab6_infer_default.mrRyl1 --device cpu
```

Result:

```text
move
real 2.30
```

Docker build:

```bash
docker build -t lab6-video-infer .
```

Result: build OK. Image size: `8.03GB`.

Docker run:

```bash
docker run --rm -v /tmp/lab6_docker_mobilenet.ezbmuB:/input:ro lab6-video-infer /input
```

Result:

```text
move
real 3.11
```

## Ограничения

- Docker runtime укладывается в лимит 10 секунд на 8 изображений.
- Docker image слишком большой из-за `torch==2.11.0` Linux wheel, который подтянул CUDA-зависимости. Это не ломает запуск, но стоит оптимизировать перед финальной упаковкой.
- Текущий best baseline существенно лучше tiny CNN, но validation small/frozen, поэтому результат нельзя считать гарантией на закрытом тесте.

## Актуализация 2026-06-05

Текущий `Dockerfile` расширен под внешний TimeSFormer/classical контракт:

- копирует `src/`, `assets/`, `scripts/`, `weights/`, `infer.py` и `outputs/models/lab6_action_classifier.pt`;
- оставляет legacy entrypoint `infer.py /input`;
- поддерживает pass-through команды вида `python -m src.predict_frames ...` и `python -m src.predict_embeddings ...`;
- включает offline-настройки HuggingFace runtime.

Актуальный TimeSFormer image:

```text
docker build -t timesformer-infer .
image size: 3 986 980 038 bytes
```

Проверенный offline raw-frame inference:

```text
docker run --rm --network none ... timesformer-infer python -m src.predict_frames --input /app/input
stdout: inaction
real: 5.52 sec
```

Проверенный offline embedding inference:

```text
docker run --rm --network none ... timesformer-infer python -m src.predict_embeddings --input /app/input/object_embedding.npy
stdout: inaction
```

## Финальная актуализация Google Drive/Docker 2026-06-05

Этот блок заменяет предыдущий legacy-oriented способ проверки для публичной сдачи.

Весовой архив опубликован на Google Drive:

```text
lab6_weights_bundle.tar.gz
https://drive.google.com/open?id=1q4rjJALqwzE-Hb3QRtvxand6QDoSSZJh
sha256: 967d087056729d2828d6aa48e65a17d1ec3f38cc62d694ad3f0d4a23537199d3
size: 907906865 bytes
```

Публичный clone восстанавливает веса командой:

```bash
scripts/download_google_drive_weights.sh
```

Актуальный Docker-контракт:

```bash
docker build -t timesformer-infer .
docker run --rm \
  -v /absolute/path/to/8_images:/app/input:ro \
  timesformer-infer
```

Dockerfile теперь проверяет наличие `weights/timesformer_checkpoint.pt`, `weights/timesformer_hf/`, `weights/bagging_best.joblib`, `weights/labels.json` и `outputs/models/lab6_action_classifier.pt` во время build.

Проверка на 2026-06-05:

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
81 tests OK
```

```text
scripts/download_google_drive_weights.sh /tmp/lab6_drive_script_download.tar.gz
restored weights from /tmp/lab6_drive_script_download.tar.gz
```

```text
docker build -t timesformer-infer .
OK
image size: 3 986 983 172 bytes
```

```text
docker run --rm --network none -v /tmp/lab6_teacher_8_images:/app/input:ro timesformer-infer
stdout: inaction
real: 7.69 sec
```

```text
scripts/run_frames.sh /tmp/lab6_teacher_8_images
stdout: inaction
```

```text
LAB6_INPUT_DIR=/tmp/lab6_teacher_8_images docker compose run --rm infer
stdout: inaction
```

Вывод: текущий публичный Docker путь принимает папку ровно с 8 изображениями и печатает ровно один класс.
