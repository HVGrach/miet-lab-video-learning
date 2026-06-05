# Baseline Training Smoke Acceptance

Дата фиксации: 2026-06-02.

## Что закрыто

- Run directory: `outputs/baseline_runs/tiny_sequence_smoke`.
- Notebook: `notebooks/08_baseline_training.ipynb`.
- Split: `outputs/manifests/train_val_split_hash_guarded.csv`.
- Split SHA256: `f0ffb2e86e90c49f57b215cc9ea7d9a807a6ae223cba371054fe125115412d6c`.
- Hash leakage rows: 0.
- Virtual leakage rows: 0.
- Model: `TinySequenceClassifier`.
- Device: `mps`.
- Train samples: 1 222.
- Val samples: 324.
- Smoke train batches: 20.
- Val accuracy: 0.3549.
- Val macro F1: 0.2784.

## Артефакты

- `config.json`
- `environment.json`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `val_predictions.csv`
- `model_best.pt`
- compatibility copies: `outputs/reports/baseline_smoke_*`, `outputs/models/tiny_sequence_smoke.pt`

## Ограничения

This is a smoke baseline, not a final leaderboard model. It proves that the training loop, split, preprocessing, metrics, and checkpoint path are operational.

## Longer Baseline

- Run directory: `outputs/baseline_runs/tiny_sequence_mps_96_e8`.
- Default checkpoint copy: `outputs/models/lab6_action_classifier.pt`.
- Split: `outputs/manifests/train_val_split_hash_guarded.csv`.
- Device: `mps`.
- Image size: `96x96`.
- Epochs: 8.
- Best epoch: 6.
- Train samples: 1 222.
- Val samples: 324.
- Val accuracy: 0.4599.
- Val macro F1: 0.3832.
- Per-class F1: `inaction=0.5686`, `move=0.4876`, `work=0.0935`.

This longer baseline is now the default inference checkpoint, but it still has weak `work` recall.

## MobileNet Congress Runs

First pretrained run:

- Run directory: `outputs/baseline_runs/mobilenet_v3_small_160_frozen_e30_workselect`.
- Model: `mobilenet_v3_small`, ImageNet pretrained, frozen encoder.
- Image size: `160x160`.
- Epochs: 30.
- Selection: `work_f1_macro_f1`.
- Val accuracy: 0.5648.
- Val macro F1: 0.5186.
- Work F1 / recall: 0.6927 / 0.6961.
- Problem: `inaction` F1 collapsed to 0.2254.

Balanced selected run:

- Run directory: `outputs/baseline_runs/mobilenet_v3_small_160_frozen_e30_balanced_minclass`.
- Default checkpoint copy: `outputs/models/lab6_action_classifier.pt`.
- Model: `mobilenet_v3_small`, ImageNet pretrained, frozen encoder.
- Image size: `160x160`.
- Epochs: 30.
- Selection: `min_class_f1_macro_f1`.
- Val accuracy: 0.5556.
- Val macro F1: 0.5326.
- Per-class F1: `inaction=0.3926`, `move=0.6479`, `work=0.5572`.
- Work recall: 0.5490.

This balanced MobileNet checkpoint is the current default for inference. Docker/inference CLI status is tracked separately in `outputs/reports/inference_docker_acceptance.md`.
