# Model Congress Acceptance

Дата фиксации: 2026-06-02.

## Консенсус агентов

- Не продолжать основной путь на `TinySequenceClassifier`: он почти не предсказывал `work`.
- Использовать только `outputs/manifests/train_val_split_hash_guarded.csv`.
- Проверять split SHA256: `f0ffb2e86e90c49f57b215cc9ea7d9a807a6ae223cba371054fe125115412d6c`.
- Считать успех не только по accuracy/macro F1, но и по `work_f1`, `work_recall`, `min_class_f1`.
- Не использовать paths, names, source ids, hash ids, dimensions, role labels, `sample_kind`, `stride`, `dilation` as features.

## Запуски

Work-focused MobileNet:

- Run: `outputs/baseline_runs/mobilenet_v3_small_160_frozen_e30_workselect`.
- Model: `mobilenet_v3_small`, ImageNet pretrained, frozen encoder.
- Image size: `160x160`.
- Epochs: 30.
- Selection: `work_f1_macro_f1`.
- Val accuracy: 0.5648.
- Val macro F1: 0.5186.
- Work F1 / recall: 0.6927 / 0.6961.
- Problem: `inaction_f1=0.2254`.

Balanced MobileNet:

- Run: `outputs/baseline_runs/mobilenet_v3_small_160_frozen_e30_balanced_minclass`.
- Model: `mobilenet_v3_small`, ImageNet pretrained, frozen encoder.
- Image size: `160x160`.
- Epochs: 30.
- Selection: `min_class_f1_macro_f1`.
- Val accuracy: 0.5556.
- Val macro F1: 0.5326.
- Min class F1: 0.3926.
- Per-class F1: `inaction=0.3926`, `move=0.6479`, `work=0.5572`.
- Work recall: 0.5490.

## Решение

Current default checkpoint: `outputs/models/lab6_action_classifier.pt`, copied from balanced MobileNet.

Docker status:

- Image: `lab6-video-infer:latest`.
- Image size: `8.03GB`.
- Docker run on 8 images: stdout one class, `real 3.11 sec`.

## Следующие Улучшения

- Add epoch progress logging and early stopping.
- Cache frame embeddings for fast head/loss/sampler sweeps.
- Try `mobilenet_v3_large` or partial unfreeze only after preserving Docker runtime.
- Tune sampling/loss to improve `inaction` without losing `work`.
