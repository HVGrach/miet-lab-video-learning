# TimeSFormer/Classical Deployment Status

Date: 2026-06-05.

## Implemented

- Added `src.predict_frames` for the external TimeSFormer raw-frame CLI.
- Added `src.predict_embeddings` for the external `bagging_best.joblib` embedding CLI.
- Added shared runtime modules for labels, image collection, `224_no_pad` preprocessing, TimeSFormer checkpoint loading, and bagging prediction.
- Copied provided `bagging_best.joblib` to `weights/bagging_best.joblib`.
- Added `weights/labels.json` with `{0: inaction, 1: move, 2: work}`.
- Added a locally trained TimeSFormer checkpoint at `weights/timesformer_checkpoint.pt`.
- Added local HuggingFace TimeSFormer base files at `weights/timesformer_hf/` for offline Docker inference.
- Updated Docker image to include the new runtime modules, `weights/`, scripts, offline HF settings, and pinned `scikit-learn==1.6.1`.
- Kept the legacy `infer.py /input` Docker entrypoint working, while adding pass-through support for the spec command style: `docker run ... timesformer-infer python -m src.predict_* ...`.

## TimeSFormer Training

The original notebook checkpoint `/content/ckpts/fold-1_fold0_partial_best.pt` was not included in the received files, so a compatible checkpoint was trained locally.

Run:

```text
outputs/timesformer_runs/timesformer_head_b4_e3_20260605/
```

Training contract:

- Split: `outputs/manifests/train_val_split_hash_guarded.csv`.
- Preprocessing: `224_no_pad`.
- Model source: local `weights/timesformer_hf/`.
- Train scope: classifier head only; this is a conservative substitute for the external spec's `last_block_only` setting.
- Epochs: 3.
- Batch size: 4.
- Optimizer: AdamW, lr `1e-4`, weight decay `0.03`.

Best validation metrics:

```text
accuracy: 0.5740740740740741
macro_f1: 0.567843537511784
per-class f1:
  inaction: 0.4432432432432432
  move:     0.6729857819905213
  work:     0.5873015873015873
confusion matrix rows=true, cols=pred:
  inaction: [41, 29, 47]
  move:     [5, 71, 29]
  work:     [22, 6, 74]
```

## Verified

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
81 tests OK
```

```text
docker build -t timesformer-infer .
OK
image size: 3,986,980,038 bytes
```

Offline TimeSFormer raw-frame inference:

```text
docker run --rm --network none ... timesformer-infer python -m src.predict_frames --input /app/input
stdout: inaction
real: 5.52 sec
```

Offline embedding inference:

```text
docker run --rm --network none ... timesformer-infer python -m src.predict_embeddings --input /app/input/object_embedding.npy
stdout: inaction
```

Offline legacy frame inference:

```text
docker run --rm --network none -v /tmp/...:/input:ro timesformer-infer /input
stdout: inaction
real: 1.88 sec
```

Strict raw-frame input guard:

```text
When the frame folder also contained object_embedding.npy, src.predict_frames failed with:
error: input folder contains non-image files: object_embedding.npy
```

## Notes

- The Docker raw-frame and embedding paths both run with `--network none`.
- `src.predict_frames` prints only the predicted class to stdout on success.
- The trained TimeSFormer checkpoint is large (`463M`); publishing through ordinary Git requires Git LFS, release assets, or an external artifact link.
