# Lab6 Employee Action Inference

Docker-ready inference for the three classes:

- `inaction`
- `move`
- `work`

The external deployment spec defines two independent runtime paths.

## Raw Frames

Input: a folder with exactly 8 image files.

```bash
python -m src.predict_frames --input /path/to/8_frame_folder
```

The script sorts image files lexicographically, applies `224_no_pad` validation preprocessing, loads `weights/timesformer_checkpoint.pt`, and prints exactly one class to stdout.

## Embeddings

Input: a `.npy` embedding file with the same 768-dimensional `emb` features used to train `bagging_best.joblib`.

```bash
python -m src.predict_embeddings --input /path/to/object_embedding.npy
```

For a folder of `.npy` embeddings, request a batch format explicitly:

```bash
python -m src.predict_embeddings --input /path/to/embeddings --format csv
```

## Required Weights

```text
weights/
├── timesformer_checkpoint.pt
├── timesformer_hf/
├── bagging_best.joblib
└── labels.json
```

`bagging_best.joblib` and `labels.json` are included from the provided artifacts. The original TimeSFormer notebook checkpoint was not included, so `weights/timesformer_checkpoint.pt` was trained locally on the hash-guarded split. `weights/timesformer_hf/` keeps the HuggingFace base model local so Docker inference works without network access.

The large TimeSFormer files are published as GitHub release assets instead of regular Git files:

```bash
scripts/download_release_weights.sh
```

This requires the GitHub CLI (`gh`). By default the script downloads release tag `lab6-timesformer-artifacts` from `HVGrach/miet-lab-video-learning`, verifies:

- `timesformer_checkpoint.pt` with `timesformer_checkpoint.pt.sha256`
- `timesformer_hf.tar.gz` with `timesformer_hf.tar.gz.sha256`

## Docker

Restore release weights first if you cloned the repository:

```bash
scripts/download_release_weights.sh
```

Build:

```bash
docker build -t timesformer-infer .
```

Run raw-frame inference:

```bash
docker run --rm \
  -v /host/input_frames:/app/input:ro \
  timesformer-infer \
  python -m src.predict_frames --input /app/input
```

Run embedding inference:

```bash
docker run --rm \
  -v /host/object_embedding.npy:/app/input/object_embedding.npy:ro \
  timesformer-infer \
  python -m src.predict_embeddings --input /app/input/object_embedding.npy
```

Both runtime paths have been checked with `--network none`; the image already contains the required weights.

The legacy laboratory entrypoint remains available:

```bash
python infer.py /path/to/8_frame_folder
```
