from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from lab6_analysis.preprocessing import SequenceAugmentationConfig
from lab6_analysis.training import (
    LABEL_TO_INDEX,
    Lab6SequenceDataset,
    TinySequenceClassifier,
    build_sequence_model,
    compute_class_weights,
    evaluate_model,
    metrics_to_frames,
    set_seed,
    train_one_epoch,
)


@dataclass(frozen=True)
class BaselineExperimentConfig:
    run_id: str
    dataset_root: str | Path = Path("lab6")
    split_manifest: str | Path = Path("outputs/manifests/train_val_split_hash_guarded.csv")
    output_root: str | Path = Path("outputs/baseline_runs")
    image_size: tuple[int, int] = (96, 96)
    batch_size: int = 16
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 20260602
    device: str = "cpu"
    model_name: str = "tiny_sequence"
    pretrained: bool = False
    freeze_backbone: bool = True
    label_smoothing: float = 0.0
    class_weight_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0)
    selection_metric: str = "macro_f1"
    expected_split_sha256: str | None = None
    augment: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    num_workers: int = 0


def run_baseline_experiment(config: BaselineExperimentConfig) -> dict[str, object]:
    """Train/evaluate the tiny sequence baseline and persist run artifacts."""
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_seed(config.seed)
    start_time = time.time()
    run_dir = Path(config.output_root) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(config.split_manifest)
    split_sha256 = _sha256_file(manifest_path)
    if (
        config.expected_split_sha256 is not None
        and split_sha256 != config.expected_split_sha256
    ):
        raise ValueError(
            "split manifest SHA256 mismatch: "
            f"expected {config.expected_split_sha256}, got {split_sha256}"
        )
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest.columns:
        raise ValueError("split manifest must contain a split column")
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if train_manifest.empty or val_manifest.empty:
        raise ValueError("split manifest must contain non-empty train and val splits")

    augmentation = SequenceAugmentationConfig(
        brightness=(0.85, 1.15),
        contrast=(0.85, 1.15),
        color=(0.9, 1.1),
        jpeg_quality=(70, 95),
        downscale=(0.75, 1.0),
    )
    train_dataset = Lab6SequenceDataset(
        config.dataset_root,
        train_manifest,
        output_size=config.image_size,
        augmentation=augmentation,
        augment=config.augment,
        seed=config.seed,
    )
    val_dataset = Lab6SequenceDataset(
        config.dataset_root,
        val_manifest,
        output_size=config.image_size,
        augment=False,
        seed=config.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device(config.device)
    model = build_sequence_model(
        config.model_name,
        num_classes=len(LABEL_TO_INDEX),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    )
    train_labels = [LABEL_TO_INDEX[label] for label in train_manifest["label"].tolist()]
    class_weights = compute_class_weights(train_labels, num_classes=len(LABEL_TO_INDEX))
    class_weights = class_weights * torch.tensor(
        config.class_weight_multipliers,
        dtype=class_weights.dtype,
    )
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_score: tuple[float, ...] = (-1.0,)
    best_epoch = 0
    best_payload: dict[str, object] | None = None
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion=criterion,
            max_batches=config.max_train_batches,
        )
        val_result = evaluate_model(
            model,
            val_loader,
            device,
            criterion=criterion,
            max_batches=config.max_val_batches,
        )
        metrics = val_result["metrics"]
        val_macro_f1 = float(metrics["macro_f1"])
        work_f1 = float(metrics["per_class"]["work"]["f1"])
        work_recall = float(metrics["per_class"]["work"]["recall"])
        min_class_f1 = min(
            float(class_metrics["f1"])
            for class_metrics in metrics["per_class"].values()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_result["loss"]),
                "val_loss": float(metrics["loss"]),
                "val_accuracy": float(metrics["accuracy"]),
                "val_macro_f1": val_macro_f1,
                "val_work_f1": work_f1,
                "val_work_recall": work_recall,
                "val_min_class_f1": min_class_f1,
            }
        )
        score = _selection_score(metrics, config.selection_metric)
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            best_payload = {
                "metrics": metrics,
                "predictions": val_result["predictions"],
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            }

    if best_payload is None:
        raise RuntimeError("training did not produce a validation result")

    best_metrics = best_payload["metrics"]
    per_class_df, confusion_df = metrics_to_frames(best_metrics)
    predictions_df = pd.DataFrame(best_payload["predictions"])
    elapsed = time.time() - start_time

    summary = {
        **_jsonable_config(config),
        "split_manifest_sha256": split_sha256,
        "model": model.__class__.__name__,
        "model_name": config.model_name,
        "pretrained": bool(config.pretrained),
        "freeze_backbone": bool(config.freeze_backbone),
        "optimizer": "AdamW",
        "selection_metric": config.selection_metric,
        "best_epoch": int(best_epoch),
        "best_score": list(best_score),
        "device": config.device,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "val_loss": float(best_metrics["loss"]),
        "val_accuracy": float(best_metrics["accuracy"]),
        "val_macro_f1": float(best_metrics["macro_f1"]),
        "val_work_f1": float(best_metrics["per_class"]["work"]["f1"]),
        "val_work_recall": float(best_metrics["per_class"]["work"]["recall"]),
        "val_min_class_f1": min(
            float(class_metrics["f1"])
            for class_metrics in best_metrics["per_class"].values()
        ),
        "elapsed_sec": float(elapsed),
    }

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }

    _write_json(run_dir / "config.json", _jsonable_config(config))
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "metrics.json", summary)
    pd.DataFrame(history).to_csv(run_dir / "epoch_metrics.csv", index=False)
    per_class_df.to_csv(run_dir / "per_class_metrics.csv", index=False)
    confusion_df.to_csv(run_dir / "confusion_matrix.csv", index=False)
    predictions_df.to_csv(run_dir / "val_predictions.csv", index=False)
    torch.save(
        {
            "model_state_dict": best_payload["model_state_dict"],
            "summary": summary,
            "label_to_index": LABEL_TO_INDEX,
            "preprocessing": {"image_size": list(config.image_size), "letterbox": True},
            "model_config": _model_config(config),
        },
        run_dir / "model_best.pt",
    )
    return summary


def _jsonable_config(config: BaselineExperimentConfig) -> dict[str, object]:
    raw = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in raw.items()
    }


def _write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_score(metrics: dict[str, object], selection_metric: str) -> tuple[float, ...]:
    per_class = metrics["per_class"]
    macro_f1 = float(metrics["macro_f1"])
    work_f1 = float(per_class["work"]["f1"])
    min_class_f1 = min(float(class_metrics["f1"]) for class_metrics in per_class.values())
    if selection_metric == "macro_f1":
        return (macro_f1,)
    if selection_metric == "macro_f1_work_f1":
        return (macro_f1, work_f1)
    if selection_metric == "min_class_f1_macro_f1":
        return (min_class_f1, macro_f1)
    if selection_metric == "work_f1_macro_f1":
        return (work_f1, macro_f1)
    raise ValueError(f"unknown selection_metric: {selection_metric}")


def _model_config(config: BaselineExperimentConfig) -> dict[str, object]:
    return {
        "model_name": config.model_name,
        "pretrained": False,
        "freeze_backbone": config.freeze_backbone,
    }
