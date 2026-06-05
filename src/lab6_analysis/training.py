from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from lab6_analysis.preprocessing import (
    SequenceAugmentationConfig,
    apply_sequence_preprocessing,
)


LABELS = ("inaction", "move", "work")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


class Lab6SequenceDataset(Dataset):
    """PyTorch dataset for lab6 8-frame sequence manifests."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: pd.DataFrame,
        *,
        output_size: tuple[int, int] = (96, 96),
        augmentation: SequenceAugmentationConfig | None = None,
        augment: bool = False,
        seed: int = 20260602,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest = manifest.reset_index(drop=True).copy()
        self.output_size = output_size
        self.augmentation = augmentation
        self.augment = augment
        self.seed = seed
        _require_columns(self.manifest, {"sample_id", "label", "frame_paths"})
        self._validate_manifest()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        frame_paths = json.loads(row["frame_paths"])
        images = []
        for rel_path in frame_paths:
            with Image.open(self.dataset_root / rel_path) as image:
                images.append(image.convert("RGB"))
        config = self.augmentation if self.augment else SequenceAugmentationConfig()
        processed, _ = apply_sequence_preprocessing(
            images,
            output_size=self.output_size,
            config=config,
            seed=self.seed + index if self.augment else self.seed,
        )
        frames = torch.stack([_image_to_tensor(image) for image in processed])
        label = torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long)
        return {
            "frames": frames,
            "label": label,
            "sample_id": row["sample_id"],
        }

    def _validate_manifest(self) -> None:
        bad_labels = sorted(set(self.manifest["label"]) - set(LABELS))
        if bad_labels:
            raise ValueError(f"unknown labels: {bad_labels}")
        for _, row in self.manifest.iterrows():
            frame_paths = json.loads(row["frame_paths"])
            if len(frame_paths) != 8:
                raise ValueError(
                    f"sample {row['sample_id']} must contain exactly 8 frames"
                )


class TinySequenceClassifier(nn.Module):
    """Small frame CNN with mean pooling over the 8 frames."""

    def __init__(self, num_classes: int = 3, feature_dim: int = 96) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, frames, channels, height, width]")
        batch, frame_count, channels, height, width = frames.shape
        encoded = self.frame_encoder(frames.reshape(batch * frame_count, channels, height, width))
        encoded = encoded.flatten(1).reshape(batch, frame_count, -1)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


class PretrainedSequenceClassifier(nn.Module):
    """Torchvision frame encoder with temporal statistics pooling."""

    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.frame_encoder, feature_dim = _build_torchvision_encoder(
            model_name,
            pretrained=pretrained,
        )
        if freeze_backbone:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad = False

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        temporal_dim = feature_dim * 4
        hidden_dim = max(128, min(512, feature_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, frames, channels, height, width]")
        batch, frame_count, channels, height, width = frames.shape
        flat = frames.reshape(batch * frame_count, channels, height, width)
        flat = (flat - self.imagenet_mean) / self.imagenet_std
        encoded = self.frame_encoder(flat).flatten(1).reshape(batch, frame_count, -1)
        mean = encoded.mean(dim=1)
        max_values = encoded.max(dim=1).values
        std = encoded.std(dim=1, unbiased=False)
        delta = encoded[:, -1, :] - encoded[:, 0, :]
        pooled = torch.cat([mean, max_values, std, delta], dim=1)
        return self.classifier(pooled)


def build_sequence_model(
    model_name: str = "tiny_sequence",
    *,
    num_classes: int = 3,
    pretrained: bool = False,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a sequence classifier from a checkpoint-friendly model name."""
    if model_name == "tiny_sequence":
        return TinySequenceClassifier(num_classes=num_classes)
    if model_name in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        return PretrainedSequenceClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"unsupported model_name: {model_name}")


def select_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 20260602) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: Iterable[int], num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(list(labels), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    total_loss = 0.0
    total_examples = 0

    for batch_index, batch in enumerate(loader):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    return {"loss": total_loss / max(total_examples, 1)}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.to(device)
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict[str, object]] = []
    total_loss = 0.0
    total_examples = 0

    for batch_index, batch in enumerate(loader):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        logits = model(frames)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1).detach().cpu()
        preds = probs.argmax(dim=1)

        for sample_id, true_index, pred_index, prob in zip(
            batch["sample_id"],
            labels.detach().cpu().tolist(),
            preds.tolist(),
            probs.tolist(),
        ):
            y_true.append(int(true_index))
            y_pred.append(int(pred_index))
            predictions.append(
                {
                    "sample_id": sample_id,
                    "true_label": INDEX_TO_LABEL[int(true_index)],
                    "pred_label": INDEX_TO_LABEL[int(pred_index)],
                    "prob_inaction": float(prob[0]),
                    "prob_move": float(prob[1]),
                    "prob_work": float(prob[2]),
                }
            )

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return {"metrics": metrics, "predictions": predictions}


def compute_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], *, num_classes: int = 3
) -> dict[str, object]:
    true = np.asarray(list(y_true), dtype=np.int64)
    pred = np.asarray(list(y_pred), dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_index, pred_index in zip(true, pred):
        confusion[true_index, pred_index] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for index, label in INDEX_TO_LABEL.items():
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        support = int(confusion[index, :].sum())
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)

    accuracy = float((true == pred).mean()) if len(true) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": pd.DataFrame(confusion, index=LABELS, columns=LABELS),
    }


def metrics_to_frames(metrics: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_class = pd.DataFrame.from_dict(metrics["per_class"], orient="index").reset_index()
    per_class = per_class.rename(columns={"index": "label"})
    confusion = metrics["confusion_matrix"].copy()
    confusion.index.name = "true_label"
    return per_class, confusion.reset_index()


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")


def _build_torchvision_encoder(
    model_name: str,
    *,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        feature_dim = int(backbone.classifier[0].in_features)
        backbone.classifier = nn.Identity()
        return backbone, feature_dim
    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        feature_dim = int(backbone.classifier[0].in_features)
        backbone.classifier = nn.Identity()
        return backbone, feature_dim
    raise ValueError(f"unsupported torchvision encoder: {model_name}")
