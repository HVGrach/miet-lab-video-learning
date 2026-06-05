# Lab6 Employee Action Inference

Docker-ready solution for classifying a folder with exactly 8 images into one class:

- `inaction`
- `move`
- `work`

The teacher-facing contract is: restore weights, build the Docker image, mount a folder with exactly 8 images, and read one class from stdout.

## Quick Check

```bash
git clone https://github.com/HVGrach/miet-lab-video-learning.git
cd miet-lab-video-learning

scripts/download_google_drive_weights.sh
docker build -t timesformer-infer .

docker run --rm \
  -v /absolute/path/to/8_images:/app/input:ro \
  timesformer-infer
```

Successful output is exactly one line:

```text
inaction
```

or:

```text
move
```

or:

```text
work
```

The Docker image defaults to:

```bash
python -m src.predict_frames --input /app/input
```

The input folder must contain exactly 8 image files. Non-image files are rejected; hidden junk such as `.DS_Store` is ignored.

## Google Drive Weights

Large weights are stored outside Git and restored by `scripts/download_google_drive_weights.sh`.

- Weights bundle: [lab6_weights_bundle.tar.gz](https://drive.google.com/open?id=1q4rjJALqwzE-Hb3QRtvxand6QDoSSZJh)
- Checksum file: [lab6_weights_bundle.tar.gz.sha256](https://drive.google.com/open?id=1yCFZfxLh2ap_nS-yrnwUlb6TjSxRrbIE)
- Expected SHA256: `967d087056729d2828d6aa48e65a17d1ec3f38cc62d694ad3f0d4a23537199d3`
- Bundle size on Google Drive: `907906865` bytes

The bundle restores:

```text
weights/
  timesformer_checkpoint.pt
  timesformer_hf/
  bagging_best.joblib
  labels.json

outputs/models/lab6_action_classifier.pt
```

The Dockerfile checks for these files during `docker build`, so a clone without restored weights fails early instead of producing a broken image.

GitHub Release fallback is also available:

```bash
scripts/download_release_weights.sh
```

That fallback requires GitHub CLI (`gh`) and downloads release tag `lab6-timesformer-artifacts`.

## Metrics

Final TimeSFormer run:

```text
outputs/timesformer_runs/timesformer_head_b4_e3_20260605/
```

Validation split:

```text
outputs/manifests/train_val_split_hash_guarded.csv
```

Summary:

| Metric | Value |
|---|---:|
| Accuracy | `0.5741` |
| Macro F1 | `0.5678` |
| F1 `inaction` | `0.4432` |
| F1 `move` | `0.6730` |
| F1 `work` | `0.5873` |

Confusion matrix, rows=true and columns=predicted:

| true \ pred | inaction | move | work |
|---|---:|---:|---:|
| inaction | 41 | 29 | 47 |
| move | 5 | 71 | 29 |
| work | 22 | 6 | 74 |

Detailed files:

- `outputs/timesformer_runs/timesformer_head_b4_e3_20260605/metrics.json`
- `outputs/timesformer_runs/timesformer_head_b4_e3_20260605/per_class_metrics.csv`
- `outputs/timesformer_runs/timesformer_head_b4_e3_20260605/confusion_matrix.csv`
- `outputs/reports/timesformer_classical_deployment_status.md`

Note: the original notebook checkpoint path from the received spec was not included in the transferred files, so `timesformer_checkpoint.pt` was trained locally on the hash-guarded split. The run trains the classifier head only and keeps the HuggingFace TimeSFormer base files local for offline Docker inference.

## Raw-Frame Inference

Local Python:

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

## Embedding Inference

The provided `bagging_best.joblib` path remains available for `.npy` embeddings with 768-dimensional `emb` features.

Local Python:

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

## Verified Locally

Current checked commands:

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

The Docker raw-frame path was checked with `--network none`; the image contains the required weights and does not download models at runtime.

## Legacy Entry Point

The earlier lab entrypoint remains in the image for compatibility:

```bash
docker run --rm \
  --entrypoint python \
  -v /absolute/path/to/8_images:/app/input:ro \
  timesformer-infer \
  infer.py /app/input
```
